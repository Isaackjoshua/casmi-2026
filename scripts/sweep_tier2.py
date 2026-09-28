"""Tune tier 2 on its own terms.

pipeline_v4 inherits tier 1's blend weight (ALPHA = 0.3) and mass window
(1 mDa), both tuned when a window held ~49 candidates. PubChem's holds
~4,650 at the test molecules' masses, which changes both trade-offs:

  alpha  -- with 95x more competitors, whether the propagation signal
            (Tanimoto to a spectral analog) or the model discriminates
            better is an open question, and it was never asked at this
            pool size.
  window -- distractors scale linearly with it, and 1 mDa is roughly a
            timsTOF-grade tolerance (3 ppm at m/z 350), so it may be
            wider than the instrument needs. Tightening the window for
            tier 1 was the single largest gain in the project (20x), and
            that knob has never been revisited for a pool this dense.

Reported on the hard set split by whether COCONUT holds the answer,
because tier 2 can only affect the coverage half.

Run: PYTHONPATH=. python3 scripts/sweep_tier2.py
"""

import argparse
import importlib.util
import time
from pathlib import Path

import pandas as pd
import torch

from src.baseline import build_library
from src.data import load_train
from src.large_pool import load_large_pool
from src.metric import mrr_at_25
from src.pipeline_v3 import load_fingerprint_model
from src.pipeline_v4 import predict_molecule as predict_tiered
from src.propagation import load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.0,0.15,0.3,0.5,0.7")
    ap.add_argument("--windows", default="0.0003,0.0005,0.001,0.002")
    args = ap.parse_args()

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
    in_coconut = {k for k in answers if k in coconut_keys}
    cov_keys = [k for k, _ in groups if k not in in_coconut]
    model = load_fingerprint_model([str(P / "fp_model.pt"), str(P / "peak_model.pt")], device)
    print(f"{len(groups)} molecules; coverage subset {len(cov_keys)}\n", flush=True)

    def run(la, win):
        preds = {k: predict_tiered(g.to_dict("records"), lib, curated, large, model, device,
                                   large_alpha=la, large_window=win) for k, g in groups}
        cov = {k: preds[k] for k in cov_keys}
        return mrr_at_25(preds, answers), mrr_at_25(cov, {k: answers[k] for k in cov_keys})

    print("alpha sweep (window 1 mDa)")
    best = None
    for la in [float(x) for x in args.alphas.split(",")]:
        t0 = time.time()
        w, c = run(la, 0.001)
        star = ""
        if best is None or w > best[0]:
            best, star = (w, la, 0.001), "  *"
        print(f"  tier2 alpha {la:4.2f}: weighted {w:.4f}  coverage {c:.4f}  ({time.time()-t0:.0f}s){star}",
              flush=True)

    print(f"\nwindow sweep (tier2 alpha {best[1]:.2f})")
    for win in [float(x) for x in args.windows.split(",")]:
        t0 = time.time()
        w, c = run(best[1], win)
        star = ""
        if w > best[0]:
            best, star = (w, best[1], win), "  *"
        print(f"  window {win*1000:5.2f} mDa: weighted {w:.4f}  coverage {c:.4f}  "
              f"({time.time()-t0:.0f}s){star}", flush=True)

    print(f"\nbest: tier2 alpha {best[1]:.2f}, window {best[2]*1000:.2f} mDa -> {best[0]:.4f}")


if __name__ == "__main__":
    main()
