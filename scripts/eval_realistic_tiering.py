"""A Class 2 simulation that does not distort tier 1.

The previous simulation made the answer unreachable in tier 1 by shrinking
the curated pool to COCONUT-only (479,717 of 729,387). That is honest about
coverage but dishonest about density: tier 1 then filled a median of 10 of
the 25 slots, where on the real test set it fills a median of 25. Tier 2
only ever takes slots tier 1 leaves, so the simulation handed it early
ranks it never gets in production -- which is why a +15% proxy gain
produced +0.002 on the leaderboard (0.230 -> 0.232).

The fix is to remove from the pool and from the anchor library exactly the
structures being scored, and nothing else. Tier 1 then has production
density and still cannot contain the answer, which is the real Class 2
position.

That also re-opens a question the broken proxy closed. If tier 1 fills all
25 slots for half of all molecules -- with candidates that are necessarily
wrong whenever the answer is not in its pool -- then capping tier 1 should
matter far more than the earlier sweep suggested, because there the cap
barely bound.

Run: PYTHONPATH=. python3 scripts/eval_realistic_tiering.py
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baseline import N_GUESSES, build_library
from src.data import load_train
from src.large_pool import load_large_pool
from src.metric import mrr_at_25, to_inchikey14
from src.pipeline_v3 import load_fingerprint_model, predict_molecule as predict_curated
from src.pipeline_v4 import LARGE_ALPHA, rank_large_pool
from src.propagation import CandidatePool, load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
ALL_LIBS = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2",
            "massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"]


def pool_without(pool, drop_keys):
    keep = ~pd.Index(pool.inchikey14).isin(drop_keys)
    keys = pool.inchikey14[keep]
    return CandidatePool(
        inchikey14=keys,
        normalized_smiles=pool.normalized_smiles[keep],
        exact_mass=pool.exact_mass[keep],
        fp_words=pool.fp_words[keep],
        popcount=pool.popcount[keep],
        index_by_key={k: i for i, k in enumerate(keys)},
    )


def tiered(rows, lib, curated, large, model, device, cap):
    out, seen = [], set()
    for smi in predict_curated(rows, lib, curated, model, device):
        key = to_inchikey14(smi) or smi
        if key in seen:
            continue
        seen.add(key)
        out.append(smi)
        if len(out) >= cap:
            break
    if len(out) < N_GUESSES and large is not None:
        for smi in rank_large_pool(rows, lib, curated, large, model, device, LARGE_ALPHA):
            key = to_inchikey14(smi) or smi
            if key in seen:
                continue
            seen.add(key)
            out.append(smi)
            if len(out) >= N_GUESSES:
                break
    return out[:N_GUESSES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="3,5,8,12,18,25")
    ap.add_argument("--reachable-weight", type=float, default=0.49,
                    help="share of the real test set whose answer IS in the curated pool; "
                         "0.232 = f*0.40 + (1-f)*0.07 implies f ~ 0.49")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_train()
    spec = importlib.util.spec_from_file_location("evalpc", ROOT / "scripts/eval_pubchem_pool.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    hard = helpers.hard_set(train)
    all_keys = sorted(hard["inchikey14"].unique())
    # Remove only HALF the scored structures from the pool and library. The
    # removed half is the Class 2 case (answer unreachable in tier 1); the kept
    # half is the ordinary case (answer reachable). Both are then measured at
    # production density, which the previous simulation could not do -- it
    # removed enough of the pool to halve tier 1's slot filling.
    rng = np.random.default_rng(7)
    unreachable = set(rng.choice(all_keys, size=len(all_keys) // 2, replace=False).tolist())
    reachable = [k for k in all_keys if k not in unreachable]
    scored = unreachable

    # production density, minus exactly the structures under test
    lib_src = train[train["ingest_lib"].isin(ALL_LIBS) & ~train["inchikey14"].isin(scored)]
    lib = build_library(lib_src)
    small = load_candidate_pool(P / "candidate_fingerprints.parquet")
    curated = pool_without(small, scored)
    large = load_large_pool(ROOT / "data/pubchem/kaggle_pool")
    print(f"anchor library {len(lib):,} spectra (all {len(ALL_LIBS)} sources)")
    print(f"curated pool {len(curated):,} of {len(small):,}")
    print(f"{len(unreachable)} molecules made unreachable in tier 1, {len(reachable)} left reachable")

    answers = hard.groupby("inchikey14")["normalized_smiles"].first().to_dict()
    groups = list(hard.groupby("inchikey14"))
    model = load_fingerprint_model([str(P / "fp_model.pt"), str(P / "peak_model.pt")], device)

    fills = []
    for k, g in groups[:40]:
        out, seen = [], set()
        for smi in predict_curated(g.to_dict("records"), lib, curated, model, device):
            key = to_inchikey14(smi) or smi
            if key not in seen:
                seen.add(key)
                out.append(smi)
        fills.append(min(len(out), N_GUESSES))
    print(f"tier 1 fills a median of {np.median(fills):.0f} slots here "
          f"(real test set: 25; old simulation: 10)\n", flush=True)

    w = args.reachable_weight

    def report(label, preds, t0):
        r = {k: preds[k] for k in reachable if k in preds}
        u = {k: preds[k] for k in unreachable if k in preds}
        mr = mrr_at_25(r, {k: answers[k] for k in r})
        mu = mrr_at_25(u, {k: answers[k] for k in u})
        blend = w * mr + (1 - w) * mu
        print(f"  {label:26s} reachable {mr:.4f}  unreachable {mu:.4f}  "
              f"weighted {blend:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        return blend

    print(f"weighted column uses reachable share {w:.2f}\n", flush=True)
    t0 = time.time()
    best = ("curated only", report("curated only (no tier 2)",
            {k: predict_curated(g.to_dict("records"), lib, curated, model, device)
             for k, g in groups}, t0))
    for cap in [int(c) for c in args.caps.split(",")]:
        t0 = time.time()
        b = report(f"tiered, tier-1 cap {cap}",
                   {k: tiered(g.to_dict("records"), lib, curated, large, model, device, cap)
                    for k, g in groups}, t0)
        if b > best[1]:
            best = (f"tier-1 cap {cap}", b)
    print(f"\nbest: {best[0]} at {best[1]:.4f}")


if __name__ == "__main__":
    main()
