"""Is tier 1's propagation weight biased against exactly the molecules we score?

Propagation ranks a candidate by max_a sim(anchor)^2.5 * Tanimoto(candidate,
anchor). A candidate that has its own library spectrum can be its own anchor
and score highly; a candidate with no library spectrum can only be reached
through someone else's. 80% of the candidates in a real test window already
have library spectra -- and a Class 2 molecule, by definition, does not. So
propagation may be systematically favouring the competitors over the answer.

There is already one observation of exactly this: tier 2, where the answer
is always absent from the library, scored best with propagation turned off
entirely (alpha 0.1907 vs 0.1768 at alpha 0.3). Tier 1 still runs at
ALPHA = 0.3, tuned on a validation set whose answers *do* have library
spectra -- the same regime mismatch, unexamined.

The right condition is structure known, spectrum unknown:
  pool    = full curated pool, so the answer is present (the leaderboard
            implies ~every test molecule's answer is)
  library = every source except the scored structures, so the answer has
            no spectrum of its own, as a Class 2 molecule would not

Run: PYTHONPATH=. python3 scripts/sweep_tier1_alpha_class2.py
"""

import argparse
import importlib.util
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
ALL_LIBS = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2",
            "massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.45")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_train()
    spec = importlib.util.spec_from_file_location("evalpc", ROOT / "scripts/eval_pubchem_pool.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    hard = helpers.hard_set(train)
    scored = set(hard["inchikey14"])

    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")
    present = sum(1 for k in scored if k in pool.index_by_key)
    model = load_fingerprint_model([str(P / "fp_model.pt"), str(P / "peak_model.pt")], device)
    answers = hard.groupby("inchikey14")["normalized_smiles"].first().to_dict()
    groups = list(hard.groupby("inchikey14"))

    # two library conditions, differing only in whether the answer has a
    # spectrum of its own
    lib_with = build_library(train[train["ingest_lib"].isin(ALL_LIBS)])
    lib_without = build_library(train[train["ingest_lib"].isin(ALL_LIBS)
                                      & ~train["inchikey14"].isin(scored)])
    print(f"{len(groups)} molecules; answer in the curated pool for {present}")
    print(f"library with answers' own spectra: {len(lib_with):,}; without: {len(lib_without):,}\n",
          flush=True)

    print(f"{'alpha':>6}  {'answer HAS a spectrum':>22}  {'answer has NONE (Class 2)':>26}")
    best = None
    for a in [float(x) for x in args.alphas.split(",")]:
        t0 = time.time()
        with_ = mrr_at_25({k: predict_molecule(g.to_dict("records"), lib_with, pool, model,
                                               device, alpha=a) for k, g in groups}, answers)
        without = mrr_at_25({k: predict_molecule(g.to_dict("records"), lib_without, pool, model,
                                                 device, alpha=a) for k, g in groups}, answers)
        star = ""
        if best is None or without > best[1]:
            best, star = (a, without), "  *"
        print(f"{a:6.2f}  {with_:22.4f}  {without:26.4f}   ({time.time()-t0:.0f}s){star}", flush=True)

    print(f"\nbest alpha when the answer has no library spectrum: {best[0]:.2f} -> {best[1]:.4f}")
    print("the left column is the regime ALPHA was originally tuned in; the right column\n"
          "is the regime the real test set is in.")


if __name__ == "__main__":
    main()
