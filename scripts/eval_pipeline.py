"""End-to-end MRR@25 for a fingerprint-model checkpoint on two validation
sets, at one or more alpha values:

  hard      -- 150 massbank/mona structures absent from the 5-source anchor
               library (random_state=42), the benchmark every Phase 2/3
               number was measured on. Public-library chemistry and
               instruments, so it under-represents the real test set's
               timsTOF domain shift.
  timstof   -- 200 enveda-180 structures from the model's 2% structure
               holdout (never trained on), measured on the same Bruker
               timsTOF as the real test set. Drug-like chemistry rather
               than natural products, but instrument-matched -- the
               signal the hard set can't give.

The anchor library for each set excludes that set's structures; the
candidate pool (COCONUT + all training structures) is unchanged.

Run: PYTHONPATH=. python3 scripts/eval_pipeline.py --model data/processed/fp_model.pt --alphas 0.3 0.15 0.0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baseline import build_library
from src.data import load_train
from src.metric import mrr_at_25
from src.pipeline_v3 import load_fingerprint_model, predict_molecule
from src.propagation import load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
FIVE_SOURCES = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]
ALL_SOURCES = FIVE_SOURCES + ["massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"]


def hard_set(train: pd.DataFrame):
    lib_keys = set(train[train["ingest_lib"].isin(FIVE_SOURCES)]["inchikey14"])
    excluded = train[train["ingest_lib"].isin(["massbank", "mona"])]
    novel = excluded[~excluded["inchikey14"].isin(lib_keys)]
    keys = set(novel["inchikey14"].drop_duplicates().sample(n=150, random_state=42))
    sample = novel[novel["inchikey14"].isin(keys)]
    lib = build_library(train[train["ingest_lib"].isin(FIVE_SOURCES)])
    return sample, lib


def timstof_set(train: pd.DataFrame, n: int = 200):
    z = np.load(ROOT / "data/processed/fp_train_data.npz", allow_pickle=True)
    held_out = set(z["keys"][np.unique(z["target_idx"][z["is_val"]])])
    tims = train[(train["ingest_lib"] == "enveda-180") & train["inchikey14"].isin(held_out)]
    keys = set(tims["inchikey14"].drop_duplicates().sample(n=n, random_state=1))
    sample = tims[tims["inchikey14"].isin(keys)]
    lib = build_library(train[train["ingest_lib"].isin(ALL_SOURCES) & ~train["inchikey14"].isin(held_out)])
    return sample, lib


def evaluate(sample, lib, pool, model, device, alpha):
    answers = sample.groupby("inchikey14")["normalized_smiles"].first().to_dict()
    t0 = time.time()
    preds = {k: predict_molecule(g.to_dict("records"), lib, pool, model, device, alpha=alpha)
             for k, g in sample.groupby("inchikey14")}
    n_empty = sum(1 for p in preds.values() if not p)
    return mrr_at_25(preds, answers), time.time() - t0, n_empty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, nargs="+", help="one checkpoint, or several to ensemble")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3])
    ap.add_argument("--sets", nargs="+", default=["hard", "timstof"])
    args = ap.parse_args()

    train = load_train()
    pool = load_candidate_pool(ROOT / "data/processed/candidate_fingerprints.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fingerprint_model(args.model if len(args.model) > 1 else args.model[0], device)
    print(f"model: {args.model}")

    sets = {}
    if "hard" in args.sets:
        sets["hard"] = hard_set(train)
    if "timstof" in args.sets:
        sets["timstof"] = timstof_set(train)

    for name, (sample, lib) in sets.items():
        print(f"[{name}] {sample['inchikey14'].nunique()} molecules, {len(sample)} spectra, anchor library {len(lib)} rows")
        for alpha in args.alphas:
            score, secs, n_empty = evaluate(sample, lib, pool, model, device, alpha)
            print(f"[{name}] alpha={alpha:.2f}  MRR@25={score:.4f}  ({secs:.0f}s, {n_empty} empty)", flush=True)


if __name__ == "__main__":
    main()
