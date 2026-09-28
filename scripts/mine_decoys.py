"""Mined hard negatives: the candidates that actually outrank the answer.

train_bit_reranker.py fitted against 32 decoys *sampled* from a structure's
mass window. It improved that 33-way contest by 35% and changed nothing
end-to-end, because a window holds thousands of candidates and 32 random
ones almost never include the handful that beat the true structure. Fitting
to separate the answer from a typical competitor is not the same problem as
separating it from its closest one.

So score the whole window with the current model and keep the top-K wrong
candidates. Those are, by construction, exactly what the ranker has to
overcome.

Output: data/processed/decoys_mined.npz, same schema as build_decoys.py
  (keys, true_fp, decoy_fp, n_window, n_valid), plus
  true_rank  (n,)  where the answer sat before any reranking -- so the
                   difficulty of the mined set is visible, not assumed

Run: PYTHONPATH=. python3 scripts/mine_decoys.py
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from src.candidates import FP_BITS
from src.large_pool import load_large_pool, window
from src.pipeline_v3 import load_fingerprint_model
from src.propagation import MASS_WINDOW_DA, load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
EPS = 1e-4
WIDEN_FACTOR, WIDEN_CAP = 10.0, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoys", type=int, default=32)
    ap.add_argument("--spectra-per-structure", type=int, default=2)
    ap.add_argument("--max-window", type=int, default=20000,
                    help="skip absurdly dense windows; they are rare and slow")
    ap.add_argument("--out", default=str(P / "decoys_mined.npz"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read each array out of the npz ONCE. Indexing an NpzFile re-reads and
    # re-allocates the whole array every time, and a view kept from it pins
    # that copy alive, so building a key -> view map inside a comprehension
    # allocates (n x array) and gets the process OOM-killed.
    src = np.load(P / "decoys.npz", allow_pickle=True)
    keys = np.asarray(src["keys"])
    true_fp_all = src["true_fp"]
    row_of_key = {k: i for i, k in enumerate(keys)}
    print(f"mining for the same {len(keys):,} structures as the sampled set", flush=True)

    d = np.load(P / "fp_train_data.npz", allow_pickle=True)
    Xcsr = sp.csr_matrix((d["X_data"], d["X_indices"], d["X_indptr"]), shape=tuple(d["X_shape"]))
    pk = np.load(P / "peak_train_data.npz", allow_pickle=True)
    peaks = {k: pk[k] for k in ("mz", "inten", "mask", "precursor", "mode")}
    spec_key = np.asarray(d["keys"])[d["target_idx"]]

    by_key = {}
    for i, k in enumerate(spec_key):
        lst = by_key.setdefault(k, [])
        if len(lst) < args.spectra_per_structure:
            lst.append(i)

    spec = importlib.util.spec_from_file_location("trbr", ROOT / "scripts/train_bit_reranker.py")
    trbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trbr)

    usable = [k for k in keys if k in by_key]
    rows = np.array([i for k in usable for i in by_key[k]])
    owner = np.array([ki for ki, k in enumerate(usable) for _ in by_key[k]])
    models = (load_fingerprint_model(str(P / "fp_model.pt"), device),
              load_fingerprint_model(str(P / "peak_model.pt"), device))
    t0 = time.time()
    Z = trbr.ensemble_logits(models, Xcsr, peaks, rows, device)
    print(f"logits for {len(rows):,} spectra in {time.time()-t0:.0f}s", flush=True)
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # one logit vector per structure: the mean over its spectra, which is what
    # the reranker is fitted against
    logit_of = np.zeros((len(usable), FP_BITS), dtype=np.float32)
    counts = np.zeros(len(usable), dtype=np.int32)
    np.add.at(logit_of, owner, Z)
    np.add.at(counts, owner, 1)
    logit_of /= np.maximum(counts, 1)[:, None]

    small = load_candidate_pool(P / "candidate_fingerprints.parquet")
    mass_of = dict(zip(small.inchikey14, small.exact_mass))
    large = load_large_pool(ROOT / "data/pubchem/kaggle_pool")

    K = args.decoys
    out_keys, out_true, out_decoy, out_n, out_valid, out_rank = [], [], [], [], [], []
    t0 = time.time()
    for ki, key in enumerate(usable):
        if key not in mass_of:
            continue
        true_packed = true_fp_all[row_of_key[key]]
        true_words = true_packed.view(np.uint64)

        lo, hi = window(large, float(mass_of[key]), MASS_WINDOW_DA)
        widen = MASS_WINDOW_DA
        while hi <= lo and widen < WIDEN_CAP:
            widen *= WIDEN_FACTOR
            lo, hi = window(large, float(mass_of[key]), widen)
        n_win = hi - lo
        if n_win < 2 or n_win > args.max_window:
            continue

        cand = np.asarray(large.fp_words[lo:hi])
        same = (cand == true_words[None, :]).all(axis=1)
        v = logit_of[ki]
        scores = np.empty(n_win, dtype=np.float32)
        for c0 in range(0, n_win, 8192):
            c1 = min(n_win, c0 + 8192)
            blk = np.unpackbits(cand[c0:c1].view(np.uint8), axis=1)[:, :FP_BITS].astype(np.float32)
            scores[c0:c1] = blk @ v

        true_bits = np.unpackbits(true_packed[None, :], axis=1)[0, :FP_BITS].astype(np.float32)
        s_true = float(true_bits @ v)
        wrong = ~same
        out_rank.append(int(1 + (scores[wrong] > s_true).sum()))

        # the top-K wrong candidates: what the ranker actually has to beat
        ws = np.flatnonzero(wrong)
        if len(ws) == 0:
            continue
        order = ws[np.argsort(scores[ws])[::-1][:K]]
        n_keep = len(order)
        slot = np.zeros((K, 32), dtype=np.uint64)
        slot[:n_keep] = cand[order]

        out_keys.append(key)
        out_true.append(true_packed)
        out_decoy.append(slot.view(np.uint8).reshape(K, -1))
        out_n.append(n_win)
        out_valid.append(n_keep)

        if len(out_keys) % 2500 == 0:
            print(f"  {len(out_keys):,} mined ({time.time()-t0:.0f}s)", flush=True)

    np.savez(args.out, keys=np.array(out_keys, dtype=object), true_fp=np.stack(out_true),
             decoy_fp=np.stack(out_decoy), n_window=np.array(out_n, dtype=np.int32),
             n_valid=np.array(out_valid, dtype=np.int32),
             true_rank=np.array(out_rank[:len(out_keys)], dtype=np.int32))
    r = np.array(out_rank[:len(out_keys)])
    print(f"\nwrote {args.out}: {len(out_keys):,} structures x {K} mined decoys "
          f"({(time.time()-t0)/60:.1f} min)")
    print(f"answer's rank in its own window before reranking: median {np.median(r):.0f}, "
          f"mean {r.mean():.0f}; already rank 1 for {(r == 1).mean():.1%}")
    print(f"window size: median {np.median(out_n):.0f}")


if __name__ == "__main__":
    main()
