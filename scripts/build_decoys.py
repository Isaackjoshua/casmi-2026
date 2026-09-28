"""Hard negatives for learning to rank: for each training structure, other
structures from the PubChem pool inside the same ~1 mDa mass window.

The model is trained with per-bit BCE, which optimizes fingerprint
prediction and not the thing inference actually does -- pick the true
structure out of a few hundred isobaric candidates. The gating experiment
showed the consequence directly: the predicted fingerprint cannot make the
true structure stand out from its isomers. These decoys are exactly the
competitors it fails against, so they are what a ranking objective needs.

A decoy whose packed fingerprint equals the true one is dropped: PubChem
contains most known compounds, so the true structure is usually *in* its
own window, and keeping it would train the model against the right answer.

Output: data/processed/decoys.npz
  keys        (n,)          InChIKey14 of each structure
  true_fp     (n, 256)      its packed Morgan fingerprint
  decoy_fp    (n, K, 256)   packed fingerprints of K hard negatives
  n_window    (n,)          how many candidates its window held
  n_valid     (n,)          how many of the K decoy slots are real (rest are
                            padding, to be masked out of the loss)

Run: PYTHONPATH=. python3 scripts/build_decoys.py --n-structures 30000
"""

import argparse
import time
from pathlib import Path

import numpy as np

from src.large_pool import load_large_pool, window
from src.propagation import MASS_WINDOW_DA, load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
WIDEN_FACTOR = 10.0
WIDEN_CAP = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-structures", type=int, default=30000)
    ap.add_argument("--decoys", type=int, default=32)
    ap.add_argument("--exclude-hard-set", action="store_true", default=True,
                    help="never train on a structure the hard validation set scores")
    ap.add_argument("--out", default=str(ROOT / "data/processed/decoys.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(ROOT / "data/processed/fp_train_data.npz", allow_pickle=True)
    keys = np.asarray(d["keys"])
    fp_table = d["fp_table"]

    banned = set()
    if args.exclude_hard_set:
        import importlib.util
        spec = importlib.util.spec_from_file_location("evalpc", ROOT / "scripts/eval_pubchem_pool.py")
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        from src.data import load_train
        banned = set(helpers.hard_set(load_train())["inchikey14"])
        print(f"excluding {len(banned)} hard-validation structures", flush=True)

    small = load_candidate_pool(ROOT / "data/processed/candidate_fingerprints.parquet")
    mass_of = dict(zip(small.inchikey14, small.exact_mass))
    large = load_large_pool(ROOT / "data/pubchem/kaggle_pool")

    eligible = [i for i, k in enumerate(keys) if k not in banned and k in mass_of]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(eligible, size=min(args.n_structures, len(eligible)), replace=False)
    print(f"{len(eligible):,} eligible structures; sampling {len(chosen):,}", flush=True)

    K = args.decoys
    out_keys, out_true, out_decoy, out_n, out_valid = [], [], [], [], []
    t0 = time.time()
    for count, si in enumerate(chosen):
        key = keys[si]
        true_packed = fp_table[si]
        true_words = true_packed.view(np.uint64)

        # exactly the window inference uses -- widened only when empty, the way
        # the pipeline widens. Widening to reach K instead would draw decoys
        # from a 10x wider mass range, where competitors have different
        # molecular formulas and are therefore much easier than the real ones.
        lo, hi = window(large, float(mass_of[key]), MASS_WINDOW_DA)
        widen = MASS_WINDOW_DA
        while hi <= lo and widen < WIDEN_CAP:
            widen *= WIDEN_FACTOR
            lo, hi = window(large, float(mass_of[key]), widen)
        n_win = hi - lo
        if n_win < 2:
            continue

        cand = np.asarray(large.fp_words[lo:hi])
        # drop the true structure itself: PubChem usually contains it
        same = (cand == true_words[None, :]).all(axis=1)
        cand = cand[~same]
        if len(cand) == 0:
            continue
        n_keep = min(K, len(cand))
        pick = rng.choice(len(cand), size=n_keep, replace=False)
        slot = np.zeros((K, 32), dtype=np.uint64)
        slot[:n_keep] = cand[pick]

        out_keys.append(key)
        out_true.append(true_packed)
        out_decoy.append(slot.view(np.uint8).reshape(K, -1))
        out_n.append(n_win)
        out_valid.append(n_keep)

        if (count + 1) % 5000 == 0:
            print(f"  {count+1:,}/{len(chosen):,} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(
        args.out,
        keys=np.array(out_keys, dtype=object),
        true_fp=np.stack(out_true),
        decoy_fp=np.stack(out_decoy),
        n_window=np.array(out_n, dtype=np.int32),
        n_valid=np.array(out_valid, dtype=np.int32),
    )
    nw = np.array(out_n)
    print(f"\nwrote {args.out}: {len(out_keys):,} structures x {K} decoys "
          f"({(time.time()-t0)/60:.1f} min)")
    nv = np.array(out_valid)
    print(f"window sizes: median {np.median(nw):.0f}, mean {nw.mean():.0f}, "
          f"min {nw.min()}, max {nw.max()}")
    print(f"usable decoys per structure: median {np.median(nv):.0f}, "
          f"{100*(nv == K).mean():.0f}% have the full {K}")


if __name__ == "__main__":
    main()
