"""Featurize every labelled training spectrum for the spectrum->fingerprint
model and cache the result, so training runs don't redo ~2.5M
featurizations each time.

Output: data/processed/fp_train_data.npz with
  X          -- (n_spectra x N_FEATURES) CSR feature matrix (saved via
                its data/indices/indptr arrays)
  target_idx -- (n_spectra,) index into fp_table for each spectrum
  fp_table   -- (n_unique_structures x 256) uint8 packed fingerprints
  keys       -- (n_unique_structures,) inchikey14 per fp_table row
  is_val     -- (n_spectra,) bool: held-out structures for val loss
  ingest_lib -- (n_spectra,) source library per spectrum

Structures in the 150-molecule hard-validation set (massbank/mona
structures absent from the 5-source anchor library, random_state=42 --
the benchmark every Phase 2 number was measured on) are excluded from
training entirely, so end-to-end MRR on that set stays an honest
out-of-sample number.

Run: PYTHONPATH=. python3 scripts/build_fp_training_data.py
"""

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data import load_train
from src.fingerprint_model import N_FEATURES, featurize

ROOT = Path(__file__).resolve().parents[1]
VAL_FRACTION = 0.02
SEED = 0


def _featurize_row(args):
    mzs, ints, precursor, mode = args
    return featurize(mzs, ints, precursor, mode)


def hard_validation_keys(train: pd.DataFrame) -> set:
    lib_sources = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]
    lib_keys = set(train[train["ingest_lib"].isin(lib_sources)]["inchikey14"])
    excluded = train[train["ingest_lib"].isin(["massbank", "mona"])]
    novel = excluded[~excluded["inchikey14"].isin(lib_keys)]
    return set(novel["inchikey14"].drop_duplicates().sample(n=150, random_state=42))


def main() -> None:
    t0 = time.time()
    train = load_train()
    fp_cache = pd.read_parquet(ROOT / "data/processed/candidate_fingerprints.parquet", columns=["inchikey14", "fingerprint"])
    fp_by_key = dict(zip(fp_cache["inchikey14"], fp_cache["fingerprint"]))
    print(f"loaded train ({len(train)}) + fingerprint cache ({len(fp_by_key)}) in {time.time()-t0:.1f}s")

    hard_val = hard_validation_keys(train)
    df = train.dropna(subset=["inchikey14", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    df = df[df["inchikey14"].isin(fp_by_key.keys()) & ~df["inchikey14"].isin(hard_val)]
    df = df[df["ms2_mzs"].map(len) > 0].reset_index(drop=True)
    print(f"usable training spectra: {len(df)} ({df['inchikey14'].nunique()} structures); "
          f"excluded {len(hard_val)} hard-validation structures")

    keys, target_idx = np.unique(df["inchikey14"].to_numpy(), return_inverse=True)
    fp_table = np.stack([np.frombuffer(fp_by_key[k], dtype=np.uint8) for k in keys])
    print(f"fingerprint table: {fp_table.shape}")

    rng = np.random.default_rng(SEED)
    val_structures = rng.random(len(keys)) < VAL_FRACTION
    is_val = val_structures[target_idx]
    print(f"val split: {val_structures.sum()} structures, {is_val.sum()} spectra")

    t1 = time.time()
    args = list(zip(df["ms2_mzs"], df["ms2_normalized_intensities"], df["precursor_mz"].astype(float), df["ionization_mode"]))
    with Pool(24) as pool:
        feats = pool.map(_featurize_row, args, chunksize=2000)
    print(f"featurized {len(feats)} spectra in {time.time()-t1:.1f}s")

    indptr = np.zeros(len(feats) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(c) for c, _ in feats])
    X = sp.csr_matrix(
        (np.concatenate([v for _, v in feats]), np.concatenate([c for c, _ in feats]), indptr),
        shape=(len(feats), N_FEATURES),
    )
    X.sum_duplicates()
    print(f"feature matrix: {X.shape}, nnz={X.nnz}, {X.data.nbytes/1e9:.2f} GB data")

    out = ROOT / "data/processed/fp_train_data.npz"
    np.savez(
        out,
        X_data=X.data, X_indices=X.indices, X_indptr=X.indptr, X_shape=np.array(X.shape),
        target_idx=target_idx, fp_table=fp_table, keys=keys, is_val=is_val,
        ingest_lib=df["ingest_lib"].to_numpy(),
    )
    print(f"wrote {out} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
