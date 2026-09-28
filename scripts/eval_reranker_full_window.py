"""Does the learned reranker still help against a real mass window?

train_bit_reranker.py fits (w, b) on a 33-way task: the true structure
against 32 sampled decoys. That is a proxy. Inference faces the entire
window -- a median of ~4,650 PubChem candidates at the test molecules'
masses -- and a reranker can easily improve a 33-way contest while doing
nothing for the real one, since the 32 sampled decoys are unlikely to
include the handful of candidates that actually outrank the answer.

So this scores the FULL window for the hard-validation molecules and
reports the true structure's rank with and without the reranker, split by
whether COCONUT already contains it. The 91 that it does not are the
molecules PubChem exists to reach.

Run: PYTHONPATH=. python3 scripts/eval_reranker_full_window.py
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baseline import N_GUESSES
from src.candidates import FP_BITS, _fingerprint_one
from src.data import load_train
from src.large_pool import load_large_pool, window
from src.pipeline_v2 import MASS_WINDOW_WIDEN_CAP, MASS_WINDOW_WIDEN_FACTOR, merge_spectra
from src.pipeline_v3 import load_fingerprint_model, predict_bit_probs
from src.propagation import MASS_WINDOW_DA, neutral_mass

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
EPS = 1e-4


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rr = np.load(P / "bit_reranker.npz")
    w, b = rr["w"].astype(np.float64), rr["b"].astype(np.float64)
    print(f"reranker: 33-way val MRR {float(rr['val_mrr']):.4f} vs baseline "
          f"{float(rr['baseline_mrr']):.4f}", flush=True)

    model = load_fingerprint_model([str(P / "fp_model.pt"), str(P / "peak_model.pt")], device)
    large = load_large_pool(ROOT / "data/pubchem/kaggle_pool")

    spec = importlib.util.spec_from_file_location("evalpc", ROOT / "scripts/eval_pubchem_pool.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    train = load_train()
    hard = helpers.hard_set(train)
    coconut_keys = set(pd.read_parquet(ROOT / "data/coconut/coconut_structures.parquet",
                                       columns=["inchikey"])["inchikey"].str.split("-").str[0])

    # Fingerprint the answers from SMILES rather than looking them up in the
    # model's training arrays: none of the 150 hard-set structures is in them
    # (they are held out of model training too, not just of the pipeline), and
    # this uses the same packing the pool was built with.
    answers = hard.groupby("inchikey14")["normalized_smiles"].first().to_dict()

    rows = []
    t0 = time.time()
    for key, g in hard.groupby("inchikey14"):
        packed, _mass = _fingerprint_one(answers[key])
        if packed is None:
            continue
        true_words = np.frombuffer(packed, dtype=np.uint64)

        recs = g.to_dict("records")
        by_adduct = {}
        for r in recs:
            by_adduct.setdefault(r["adduct"], []).append(r)
        group = max(by_adduct.values(), key=lambda gr: max(len(x["ms2_mzs"]) for x in gr))
        merged = merge_spectra(group) if len(group) > 1 else group[0]
        qmass = neutral_mass(float(merged["precursor_mz"]), merged["adduct"])
        if qmass is None:
            continue

        lo, hi = window(large, qmass, MASS_WINDOW_DA)
        widen = MASS_WINDOW_DA
        while hi <= lo and widen < MASS_WINDOW_WIDEN_CAP:
            widen *= MASS_WINDOW_WIDEN_FACTOR
            lo, hi = window(large, qmass, widen)
        if hi <= lo:
            continue

        cand = np.asarray(large.fp_words[lo:hi])
        hit = np.flatnonzero((cand == true_words[None, :]).all(axis=1))
        if len(hit) == 0:
            rows.append({"key": key, "in_coconut": key in coconut_keys, "n": hi - lo,
                         "rank_base": None, "rank_rr": None})
            continue

        probs = np.clip(predict_bit_probs(model, group, device).mean(axis=0).astype(np.float64), EPS, 1 - EPS)
        logit = np.log(probs) - np.log1p(-probs)
        v_base = logit.astype(np.float32)
        v_rr = (w * logit + b).astype(np.float32)

        # chunked: the widest windows hold ~44k candidates, and materializing
        # that as one float bit matrix is needlessly large
        s_base = np.empty(hi - lo, dtype=np.float32)
        s_rr = np.empty(hi - lo, dtype=np.float32)
        for c0 in range(0, hi - lo, 8192):
            c1 = min(hi - lo, c0 + 8192)
            blk = np.unpackbits(cand[c0:c1].view(np.uint8), axis=1)[:, :FP_BITS].astype(np.float32)
            s_base[c0:c1] = blk @ v_base
            s_rr[c0:c1] = blk @ v_rr
        r_base = int(1 + (s_base > s_base[hit[0]]).sum())
        r_rr = int(1 + (s_rr > s_rr[hit[0]]).sum())
        rows.append({"key": key, "in_coconut": key in coconut_keys, "n": hi - lo,
                     "rank_base": r_base, "rank_rr": r_rr})

    df = pd.DataFrame(rows)
    print(f"{len(df)} molecules in {time.time()-t0:.0f}s; "
          f"true structure present in the PubChem window for "
          f"{df['rank_base'].notna().sum()}", flush=True)
    print(f"window size: median {df['n'].median():.0f}, mean {df['n'].mean():.0f}\n")

    def rr_at(ranks):
        return np.mean([1.0 / r if r is not None and r <= N_GUESSES else 0.0 for r in ranks])

    for label, sub in (("all", df), ("in COCONUT", df[df.in_coconut]),
                       ("NOT in COCONUT (what PubChem is for)", df[~df.in_coconut])):
        present = sub[sub.rank_base.notna()]
        if not len(present):
            continue
        print(f"{label} ({len(sub)} molecules, true structure in window for {len(present)})")
        print(f"  MRR@25  base {rr_at(sub.rank_base):.4f}  ->  reranked {rr_at(sub.rank_rr):.4f}")
        print(f"  median rank  base {present.rank_base.median():.0f}  ->  reranked "
              f"{present.rank_rr.median():.0f}")
        print(f"  in top 25    base {(present.rank_base <= 25).sum()}  ->  reranked "
              f"{(present.rank_rr <= 25).sum()}   of {len(present)}")
    df.to_csv(ROOT / "logs/reranker_full_window.csv", index=False)


if __name__ == "__main__":
    main()
