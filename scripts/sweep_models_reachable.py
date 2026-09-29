"""Model selection in the regime that actually matters.

Every previous model comparison (scripts/sweep_models.py) ran against the
COCONUT-only pool, i.e. the case where the answer is *absent* and only
PubChem can reach it. Two leaderboard submissions since then imply the
reachable share of the real test set is ~0.94-1.05: essentially every test
molecule already has its answer in the curated pool. So the comparison was
run in the regime that barely occurs, and the one that dominates -- rank
the true structure among the ~49 mass-matched candidates of the full
curated pool -- was never used to choose a checkpoint.

Checkpoints have been selected on validation loss or Tanimoto@0.5
throughout, and that proxy has pointed the wrong way repeatedly. This
scores them on the retrieval task itself.

Run: PYTHONPATH=. python3 scripts/sweep_models_reachable.py
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from src.baseline import build_library
from src.data import load_train
from src.metric import mrr_at_25
from src.pipeline_v3 import load_fingerprint_model, predict_molecule
from src.propagation import load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]

# every checkpoint on disk, plus the ensembles worth asking about
SINGLES = [
    "fp_model.pt", "fp_model_v2.pt", "fp_model_v2_tan.pt",
    "fp_model_ft.pt", "fp_model_ft_tan.pt", "fp_model_ft2.pt", "fp_model_ft2_tan.pt",
    "peak_model.pt", "peak_model_tan.pt", "peak_model_l.pt", "peak_model_l_tan.pt",
]
ENSEMBLES = {
    "deployed (fp_model + peak_model)": ["fp_model.pt", "peak_model.pt"],
    "ft2_tan + peak_l_tan": ["fp_model_ft2_tan.pt", "peak_model_l_tan.pt"],
    "ft2_tan + peak_model": ["fp_model_ft2_tan.pt", "peak_model.pt"],
    "fp_model + peak_l_tan": ["fp_model.pt", "peak_model_l_tan.pt"],
    "three-way (ft2_tan + peak + peak_l_tan)":
        ["fp_model_ft2_tan.pt", "peak_model.pt", "peak_model_l_tan.pt"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--singles-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_train()
    lib = build_library(train[train["ingest_lib"].isin(FIVE)])
    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")

    import importlib.util
    spec = importlib.util.spec_from_file_location("evalpc", ROOT / "scripts/eval_pubchem_pool.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    hard = helpers.hard_set(train)
    answers = hard.groupby("inchikey14")["normalized_smiles"].first().to_dict()
    groups = list(hard.groupby("inchikey14"))
    present = sum(1 for k in answers if k in pool.index_by_key)
    print(f"{len(groups)} molecules; answer present in the curated pool for {present} "
          f"({present/len(groups):.0%}) -- the reachable regime\n", flush=True)

    def score(paths):
        model = load_fingerprint_model([str(P / p) for p in paths], device)
        preds = {k: predict_molecule(g.to_dict("records"), lib, pool, model, device, alpha=args.alpha)
                 for k, g in groups}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return mrr_at_25(preds, answers)

    results = []
    print("single checkpoints")
    for name in SINGLES:
        if not (P / name).exists():
            print(f"  {name:28s} missing")
            continue
        t0 = time.time()
        m = score([name])
        results.append((name, m))
        print(f"  {name:28s} MRR@25 {m:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    if not args.singles_only:
        print("\nensembles")
        for name, paths in ENSEMBLES.items():
            if any(not (P / p).exists() for p in paths):
                print(f"  {name:40s} missing a checkpoint")
                continue
            t0 = time.time()
            m = score(paths)
            results.append((name, m))
            print(f"  {name:40s} MRR@25 {m:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    results.sort(key=lambda r: -r[1])
    print("\nranked:")
    for name, m in results[:6]:
        print(f"  {m:.4f}  {name}")


if __name__ == "__main__":
    main()
