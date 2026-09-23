"""Decide whether the PubChem pool is worth shipping, by measuring the two
effects it has -- separately, because they pull in opposite directions.

  coverage  -- structures absent from BOTH the anchor library AND the
               current COCONUT/training pool. Today these score exactly
               0.0: the right answer isn't in the candidate pool at all,
               so no amount of ranking can find it. This is the gain.

  precision -- the existing 150-molecule hard validation set, which
               already has 100% pool coverage. PubChem adds ~500
               same-mass distractors per query where there used to be a
               handful, so the true answer has to out-rank far more
               competition. This is the cost.

Ships only if the coverage gain outweighs the precision loss. Run both
pools through the identical pipeline so the comparison is clean.

Run: PYTHONPATH=. python3 scripts/eval_pubchem_pool.py --model data/processed/fp_model_ft2_tan.pt
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
from src.propagation import CandidatePool, load_candidate_pool

ROOT = Path(__file__).resolve().parents[1]
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]


def load_pubchem_pool(path: Path, merge_with: CandidatePool | None = None) -> CandidatePool:
    """Memory-map the PubChem arrays (tens of GB) rather than loading them.
    Optionally merge the existing COCONUT/training pool in, keeping the
    combined arrays mass-sorted -- the pipeline requires that ordering for
    its binary-search mass window.
    """
    masses = np.load(path / "masses.npy", mmap_mode="r")
    fps = np.load(path / "fps.npy", mmap_mode="r")
    meta = pd.read_parquet(path / "meta.parquet")
    keys = meta["inchikey14"].to_numpy()
    smiles = meta["normalized_smiles"].to_numpy()

    if merge_with is not None:
        # concatenate, drop structures already in PubChem, re-sort by mass
        extra = ~np.isin(merge_with.inchikey14, keys)
        masses = np.concatenate([np.asarray(masses), merge_with.exact_mass[extra]])
        fps = np.concatenate([np.asarray(fps), merge_with.fp_words[extra]])
        keys = np.concatenate([keys, merge_with.inchikey14[extra]])
        smiles = np.concatenate([smiles, merge_with.normalized_smiles[extra]])
        order = np.argsort(masses, kind="stable")
        masses, fps, keys, smiles = masses[order], fps[order], keys[order], smiles[order]

    return CandidatePool(
        inchikey14=keys,
        normalized_smiles=smiles,
        exact_mass=np.asarray(masses, dtype=np.float64),
        fp_words=fps,
        popcount=np.bitwise_count(np.asarray(fps)).sum(axis=1).astype(np.int64),
        index_by_key={k: i for i, k in enumerate(keys)},
    )


def hard_set(train: pd.DataFrame):
    lib_keys = set(train[train["ingest_lib"].isin(FIVE)]["inchikey14"])
    novel = train[train["ingest_lib"].isin(["massbank", "mona"])]
    novel = novel[~novel["inchikey14"].isin(lib_keys)]
    keys = set(novel["inchikey14"].drop_duplicates().sample(n=150, random_state=42))
    return novel[novel["inchikey14"].isin(keys)]


def coverage_set(train: pd.DataFrame, current_pool_keys: set, n: int = 150):
    """Structures the current pool cannot reach at all: absent from the
    anchor library and absent from COCONUT+training. Scores 0.0 today.
    """
    lib_keys = set(train[train["ingest_lib"].isin(FIVE)]["inchikey14"])
    novel = train[train["ingest_lib"].isin(["massbank", "mona", "spectraverse", "msdial"])]
    novel = novel[~novel["inchikey14"].isin(lib_keys) & ~novel["inchikey14"].isin(current_pool_keys)]
    uniq = novel["inchikey14"].drop_duplicates()
    if len(uniq) == 0:
        return novel.head(0)
    keys = set(uniq.sample(n=min(n, len(uniq)), random_state=11))
    return novel[novel["inchikey14"].isin(keys)]


def evaluate(sample, lib, pool, model, device, alpha=0.3):
    answers = sample.groupby("inchikey14")["normalized_smiles"].first().to_dict()
    t0 = time.time()
    preds = {k: predict_molecule(g.to_dict("records"), lib, pool, model, device, alpha=alpha)
             for k, g in sample.groupby("inchikey14")}
    return mrr_at_25(preds, answers), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data/processed/fp_model_ft2_tan.pt"))
    ap.add_argument("--pubchem", default=str(ROOT / "data/pubchem/pubchem_pool"))
    ap.add_argument("--alpha", type=float, default=0.3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fingerprint_model(args.model, device)
    train = load_train()
    lib = build_library(train[train["ingest_lib"].isin(FIVE)])

    current = load_candidate_pool(ROOT / "data/processed/candidate_fingerprints.parquet")
    print(f"current pool: {len(current)} structures")
    merged = load_pubchem_pool(Path(args.pubchem), merge_with=current)
    print(f"merged pool:  {len(merged)} structures")

    hard = hard_set(train)
    cover = coverage_set(train, set(current.inchikey14))
    print(f"hard set:     {hard['inchikey14'].nunique()} molecules (100% covered today)")
    print(f"coverage set: {cover['inchikey14'].nunique()} molecules (0% covered today)")
    if len(cover):
        in_new = np.isin(cover["inchikey14"].unique(), merged.inchikey14).mean()
        print(f"              of which PubChem now covers: {in_new:.0%}")

    for name, sample in [("hard", hard), ("coverage", cover)]:
        if not len(sample):
            continue
        for label, pool in [("current", current), ("merged", merged)]:
            score, secs = evaluate(sample, lib, pool, model, device, args.alpha)
            print(f"[{name:8s}] {label:7s} pool: MRR@25={score:.4f} ({secs:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
