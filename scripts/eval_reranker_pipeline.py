"""The decisive test of the learned reranker: end-to-end pipeline MRR@25.

The full-window measurement (eval_reranker_full_window.py) locates the true
structure by exact fingerprint match, which finds it in only 89 of 138
windows; the pipeline matches on tautomer-canonical InChIKey14, which is
more permissive, and dedups its 25 guesses the same way. So that measurement
cannot be compared with a leaderboard score. This one runs the real
pipeline, with and without the reranker installed, in the Class 2 simulation
that has tracked real outcomes.

Run: PYTHONPATH=. python3 scripts/eval_reranker_pipeline.py
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baseline import build_library
from src.data import load_train
from src.fingerprint_model import set_bit_logit_transform
from src.large_pool import load_large_pool
from src.metric import mrr_at_25
from src.pipeline_v3 import load_fingerprint_model, predict_molecule as predict_curated
from src.pipeline_v4 import predict_molecule as predict_tiered
from src.propagation import load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]


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
    in_coconut = {k for k in answers if k in coconut_keys}
    print(f"{len(groups)} molecules ({len(in_coconut)} in COCONUT, "
          f"{len(groups)-len(in_coconut)} not)\n", flush=True)

    model = load_fingerprint_model([str(P / "fp_model.pt"), str(P / "peak_model.pt")], device)
    rr = np.load(P / "bit_reranker.npz")

    def run(label):
        out = {}
        for name, fn in (("curated only", lambda g: predict_curated(g, lib, curated, model, device)),
                         ("tiered", lambda g: predict_tiered(g, lib, curated, large, model, device))):
            t0 = time.time()
            preds = {k: fn(g.to_dict("records")) for k, g in groups}
            sub_p = {k: v for k, v in preds.items() if k in in_coconut}
            sub_c = {k: v for k, v in preds.items() if k not in in_coconut}
            out[name] = (mrr_at_25(preds, answers),
                         mrr_at_25(sub_p, {k: answers[k] for k in sub_p}),
                         mrr_at_25(sub_c, {k: answers[k] for k in sub_c}))
            print(f"  {label:12s} {name:14s} weighted {out[name][0]:.4f}   "
                  f"precision {out[name][1]:.4f}   coverage {out[name][2]:.4f}   "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return out

    set_bit_logit_transform(None, None)
    base = run("baseline")
    set_bit_logit_transform(rr["w"], rr["b"])
    rer = run("reranked")

    print("\n=== change from the reranker ===")
    for name in base:
        d = rer[name][0] - base[name][0]
        print(f"  {name:14s} {base[name][0]:.4f} -> {rer[name][0]:.4f}  ({d:+.4f}, "
              f"{100*(rer[name][0]/base[name][0]-1):+.1f}%)")


if __name__ == "__main__":
    main()
