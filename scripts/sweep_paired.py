"""Sweep the pipeline's constants with an instrument that can see the effects.

Most of this project's constants were set at n=150, and some were never
tuned at all -- the README calls PEAK_TOP_K, BIN_WIDTH, COARSE_TOP_K and
FINE_TOP_N "initial guesses". n=150 is roughly the noise floor for a
0.01 MRR effect here, which is how a +1.2% reading for alpha 0.45 survived
long enough to be shipped and lose 0.003 on the leaderboard.

So: every candidate is compared against the shipped configuration on the
same molecules, scored per molecule, and reported as a paired mean
difference with a 95% CI. Paired differences cancel the molecule-to-molecule
variance that swamps two independent means, which is what makes a few
hundred molecules enough to resolve effects this small.

Condition matches the real test set as closely as it can be made to:
  pool    = full curated pool, so the answer is present (two submissions
            imply ~every test molecule's answer is)
  library = all sources except the scored structures' own spectra, so the
            query is novel the way a Class 2 molecule is

Run: PYTHONPATH=. python3 scripts/sweep_paired.py --n 600
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import src.baseline as baseline
from src.baseline import build_library
from src.data import load_train
from src.metric import to_inchikey14
from src.pipeline_v3 import ALPHA, load_fingerprint_model, predict_molecule
from src.propagation import MASS_WINDOW_DA, PROPAGATION_EXPONENT, load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]
ALL_LIBS = FIVE + ["massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"]

BASE_MODEL = ["fp_model.pt", "peak_model.pt"]


def reciprocal_rank(guesses, target):
    for i, s in enumerate(guesses[:25], 1):
        if (to_inchikey14(s) or s) == target:
            return 1.0 / i
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--only", default=None,
                    help="substring filter on candidate labels, for confirming one result")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_train()

    lib_keys = set(train[train["ingest_lib"].isin(FIVE)]["inchikey14"])
    novel = train[train["ingest_lib"].isin(["massbank", "mona"])]
    novel = novel[~novel["inchikey14"].isin(lib_keys)]
    keys = set(novel["inchikey14"].drop_duplicates().sample(
        n=min(args.n, novel["inchikey14"].nunique()), random_state=args.seed))
    sample = novel[novel["inchikey14"].isin(keys)]
    scored = set(sample["inchikey14"])

    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")
    lib = build_library(train[train["ingest_lib"].isin(ALL_LIBS)
                              & ~train["inchikey14"].isin(scored)])
    groups = list(sample.groupby("inchikey14"))
    print(f"{len(groups)} molecules; library {len(lib):,} spectra; "
          f"pool {len(pool):,}\n", flush=True)

    models = {}

    def get_model(names):
        k = tuple(names)
        if k not in models:
            models[k] = load_fingerprint_model([str(P / n) for n in names], device)
        return models[k]

    def run(model_names=BASE_MODEL, exponent=PROPAGATION_EXPONENT,
            window=MASS_WINDOW_DA, alpha=ALPHA, coarse=None, fine=None):
        """Per-molecule reciprocal rank under one configuration."""
        old_c, old_f = baseline.COARSE_TOP_K, baseline.FINE_TOP_N
        if coarse is not None:
            baseline.COARSE_TOP_K = coarse
        if fine is not None:
            baseline.FINE_TOP_N = fine
        try:
            model = get_model(model_names)
            out = {}
            for key, g in groups:
                guesses = predict_molecule(g.to_dict("records"), lib, pool, model, device,
                                           mass_window_da=window, exponent=exponent, alpha=alpha)
                out[key] = reciprocal_rank(guesses, key)
            return out
        finally:
            baseline.COARSE_TOP_K, baseline.FINE_TOP_N = old_c, old_f

    t0 = time.time()
    base = run()
    base_mrr = float(np.mean(list(base.values())))
    print(f"shipped configuration: MRR@25 {base_mrr:.4f}  ({time.time()-t0:.0f}s)\n", flush=True)

    candidates = [
        ("exponent 1.5", dict(exponent=1.5)),
        ("exponent 2.0", dict(exponent=2.0)),
        ("exponent 3.0", dict(exponent=3.0)),
        ("exponent 3.5", dict(exponent=3.5)),
        ("COARSE_TOP_K 150", dict(coarse=150)),
        ("COARSE_TOP_K 600", dict(coarse=600)),
        ("anchors FINE_TOP_N 25", dict(fine=25)),
        ("anchors FINE_TOP_N 100", dict(fine=100)),
        ("model peak_model_l alone", dict(model_names=["peak_model_l.pt"])),
        ("model fp+peak_l", dict(model_names=["fp_model.pt", "peak_model_l.pt"])),
    ]

    if args.only:
        candidates = [c for c in candidates if args.only.lower() in c[0].lower()]
    print(f"{'candidate':26s} {'MRR':>7} {'paired diff':>12} {'95% CI':>9}  verdict")
    wins = []
    for label, kw in candidates:
        t0 = time.time()
        res = run(**kw)
        d = np.array([res[k] - base[k] for k, _ in groups])
        mrr = float(np.mean(list(res.values())))
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        sig = abs(d.mean()) > ci
        verdict = ("BETTER" if d.mean() > 0 else "worse") if sig else "ns"
        if sig and d.mean() > 0:
            wins.append((label, d.mean(), ci))
        print(f"{label:26s} {mrr:7.4f} {d.mean():+12.4f} {ci:9.4f}  {verdict}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    print()
    if wins:
        for label, dm, ci in sorted(wins, key=lambda w: -w[1]):
            print(f"significant improvement: {label}  {dm:+.4f} +/- {ci:.4f}")
    else:
        print("nothing beat the shipped configuration at 95% confidence.")


if __name__ == "__main__":
    main()
