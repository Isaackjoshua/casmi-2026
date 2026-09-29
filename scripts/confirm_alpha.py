"""Confirm (or drop) the alpha change on enough molecules to see past noise.

The 150-molecule sweep puts tier-1 alpha at 0.45 rather than the deployed
0.3, worth +1.2%. Two of the last two proxy-driven changes lost on the
leaderboard, and +1.2% at n=150 is inside the noise that produced them, so
this re-runs the three candidate values on a much larger sample before
anything is shipped.

Same Class 2 condition as the sweep: the answer is in the curated pool, and
every scored structure's own spectra are removed from the anchor library.

Run: PYTHONPATH=. python3 scripts/confirm_alpha.py --n 600
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.baseline import build_library
from src.data import load_train
from src.metric import mrr_at_25
from src.pipeline_v3 import load_fingerprint_model, predict_molecule
from src.propagation import load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]
ALL_LIBS = FIVE + ["massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--alphas", default="0.3,0.45")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_train()

    # a larger draw than hard_set's 150, same recipe: novel structures, held
    # out of the five core libraries
    lib_keys = set(train[train["ingest_lib"].isin(FIVE)]["inchikey14"])
    novel = train[train["ingest_lib"].isin(["massbank", "mona"])]
    novel = novel[~novel["inchikey14"].isin(lib_keys)]
    keys = set(novel["inchikey14"].drop_duplicates().sample(
        n=min(args.n, novel["inchikey14"].nunique()), random_state=args.seed))
    sample = novel[novel["inchikey14"].isin(keys)]

    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")
    scored = set(sample["inchikey14"])
    lib = build_library(train[train["ingest_lib"].isin(ALL_LIBS)
                              & ~train["inchikey14"].isin(scored)])
    model = load_fingerprint_model([str(P / "fp_model.pt"), str(P / "peak_model.pt")], device)

    answers = sample.groupby("inchikey14")["normalized_smiles"].first().to_dict()
    groups = list(sample.groupby("inchikey14"))
    present = sum(1 for k in answers if k in pool.index_by_key)
    print(f"{len(groups)} molecules (answer in pool for {present}); "
          f"library {len(lib):,} spectra\n", flush=True)

    per_mol = {}
    for a in [float(x) for x in args.alphas.split(",")]:
        t0 = time.time()
        preds = {k: predict_molecule(g.to_dict("records"), lib, pool, model, device, alpha=a)
                 for k, g in groups}
        per_mol[a] = preds
        print(f"  alpha {a:4.2f}: MRR@25 {mrr_at_25(preds, answers):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    alphas = sorted(per_mol)
    if len(alphas) == 2:
        lo, hi = alphas
        # paired comparison: same molecules, so look at the per-molecule
        # difference rather than two independent means
        def rr(preds, k):
            from src.metric import to_inchikey14
            guesses = preds[k]
            target = k
            for i, s in enumerate(guesses[:25], 1):
                if (to_inchikey14(s) or s) == target:
                    return 1.0 / i
            return 0.0
        d = np.array([rr(per_mol[hi], k) - rr(per_mol[lo], k) for k, _ in groups])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\npaired difference (alpha {hi} minus {lo}): {d.mean():+.4f} "
              f"+/- {1.96*se:.4f} (95% CI)")
        print(f"  molecules better {int((d > 0).sum())}, worse {int((d < 0).sum())}, "
              f"unchanged {int((d == 0).sum())}")
        verdict = "significant" if abs(d.mean()) > 1.96 * se else "NOT significant"
        print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
