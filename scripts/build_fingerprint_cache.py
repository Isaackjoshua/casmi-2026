"""One-time (expensive) step: fingerprint the full candidate pool (COCONUT
+ unique training-library structures) and cache it to parquet, so neither
local iteration nor a Kaggle submission run has to redo ~1M RDKit
fingerprint calls every time.

Run: python3 scripts/build_fingerprint_cache.py
Output: data/processed/candidate_fingerprints.parquet
"""

import time
from pathlib import Path

import pandas as pd

from src.candidates import build_candidate_pool, fingerprint_many, load_coconut
from src.data import load_train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    t0 = time.time()
    coconut = load_coconut(ROOT / "data/coconut/coconut_structures.parquet")
    print(f"COCONUT: {len(coconut)} structures ({time.time()-t0:.1f}s)")

    train = load_train()
    pool = build_candidate_pool(coconut, train)
    print(f"combined candidate pool: {len(pool)} unique structures ({time.time()-t0:.1f}s)")

    t1 = time.time()
    fps, masses = fingerprint_many(pool["normalized_smiles"].tolist())
    print(f"fingerprinted {len(fps)} structures in {time.time()-t1:.1f}s")

    # RDKit-computed mass replaces the source's own (or fills the training
    # library's missing exact_mass), so every row uses the same convention
    # for the propagation step's mass-window filter.
    pool = pool.assign(fingerprint=fps, exact_mass=masses)
    n_failed = pool["fingerprint"].isna().sum()
    print(f"{n_failed} structures failed to parse/fingerprint (dropped)")
    pool = pool.dropna(subset=["fingerprint", "exact_mass"]).reset_index(drop=True)

    out_path = ROOT / "data/processed/candidate_fingerprints.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out_path)
    print(f"wrote {len(pool)} rows to {out_path} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
