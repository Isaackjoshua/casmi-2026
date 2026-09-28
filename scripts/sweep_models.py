"""Which model config should the tiered pipeline ship?

Gating turned out to be limited by the spectrum->fingerprint model, not by
pool logic, so this compares checkpoints in the setting that actually
predicts the leaderboard: the Class 2 simulation, where the curated pool is
COCONUT-only and 91 of the 150 hard-validation answers are therefore
unreachable without PubChem -- exactly the position a real test molecule in
no public library is in.

Earlier model comparisons used the full 729k pool, which contains every
answer by construction and so measures only ranking among candidates that
are all present. That flattered every model equally and could not see the
coverage term.

Run: PYTHONPATH=. python3 scripts/sweep_models.py
"""

import importlib.util
import time
from pathlib import Path

import pandas as pd
import torch

from src.baseline import build_library
from src.data import load_train
from src.large_pool import load_large_pool
from src.metric import mrr_at_25
from src.pipeline_v3 import load_fingerprint_model, predict_molecule as predict_curated
from src.pipeline_v4 import predict_molecule as predict_tiered
from src.propagation import load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]
P = ROOT / "data/processed"

CONFIGS = {
    "deployed (MLP + transformer)": [P / "fp_model.pt", P / "peak_model.pt"],
    "MLP + large transformer":      [P / "fp_model.pt", P / "peak_model_l_tan.pt"],
    "large transformer alone":      [P / "peak_model_l_tan.pt"],
    "MLP alone":                    [P / "fp_model.pt"],
    "MLP + both transformers":      [P / "fp_model.pt", P / "peak_model.pt", P / "peak_model_l_tan.pt"],
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_train()
    lib = build_library(train[train["ingest_lib"].isin(FIVE)])

    spec = importlib.util.spec_from_file_location("evalpc", ROOT / "scripts/eval_pubchem_pool.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    small = load_candidate_pool(P / "candidate_fingerprints.parquet")
    large = load_large_pool(ROOT / "data/pubchem/kaggle_pool")
    coconut_keys = set(pd.read_parquet(ROOT / "data/coconut/coconut_structures.parquet",
                                       columns=["inchikey"])["inchikey"].str.split("-").str[0])
    curated = helpers.coconut_only_pool(small, coconut_keys)
    hard = helpers.hard_set(train)
    answers = hard.groupby("inchikey14")["normalized_smiles"].first().to_dict()
    groups = list(hard.groupby("inchikey14"))
    print(f"{len(groups)} molecules; curated pool {len(curated):,}, PubChem {len(large):,}\n", flush=True)

    print(f"{'config':30s} {'curated only':>14s} {'tiered':>10s}")
    for name, paths in CONFIGS.items():
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"{name:30s} SKIPPED (missing {[m.name for m in missing]})")
            continue
        model = load_fingerprint_model([str(p) for p in paths], device)
        t0 = time.time()
        cur = {k: predict_curated(g.to_dict("records"), lib, curated, model, device)
               for k, g in groups}
        tie = {k: predict_tiered(g.to_dict("records"), lib, curated, large, model, device)
               for k, g in groups}
        print(f"{name:30s} {mrr_at_25(cur, answers):14.4f} {mrr_at_25(tie, answers):10.4f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
