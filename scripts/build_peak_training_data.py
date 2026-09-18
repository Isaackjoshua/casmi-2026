"""Padded per-peak arrays for the peak transformer (src/peak_transformer.py),
row-aligned with fp_train_data.npz so its target_idx / fp_table / is_val
split can be reused as-is (same filtering, same order -- asserted).

Output: data/processed/peak_train_data.npz with mz, inten (n x MAX_PEAKS
float32), mask (n x MAX_PEAKS bool), precursor (n,), mode (n,) int8.

Run: PYTHONPATH=. python3 scripts/build_peak_training_data.py
"""

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_fp_training_data import hard_validation_keys
from src.data import load_train
from src.peak_transformer import MAX_MZ, MAX_PEAKS, mode_id, prepare_peaks

ROOT = Path(__file__).resolve().parents[1]


def _prep(args):
    mzs, ints, precursor = args
    return prepare_peaks(mzs, ints, precursor, MAX_PEAKS)


def main() -> None:
    t0 = time.time()
    train = load_train()
    z = np.load(ROOT / "data/processed/fp_train_data.npz", allow_pickle=True)
    fp_keys = set(z["keys"])
    hard_val = hard_validation_keys(train)

    df = train.dropna(subset=["inchikey14", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    df = df[df["inchikey14"].isin(fp_keys | hard_val) & ~df["inchikey14"].isin(hard_val)]
    df = df[df["ms2_mzs"].map(len) > 0].reset_index(drop=True)
    assert len(df) == len(z["target_idx"]), f"row mismatch: {len(df)} vs {len(z['target_idx'])}"
    assert (np.unique(df["inchikey14"].to_numpy(), return_inverse=True)[1] == z["target_idx"]).all(), "order mismatch"
    print(f"{len(df)} spectra, aligned with fp_train_data.npz ({time.time()-t0:.1f}s)")

    t1 = time.time()
    args = list(zip(df["ms2_mzs"], df["ms2_normalized_intensities"], df["precursor_mz"].astype(float)))
    with Pool(24) as pool:
        arrs = pool.map(_prep, args, chunksize=2000)
    print(f"prepared peaks in {time.time()-t1:.1f}s")

    out = ROOT / "data/processed/peak_train_data.npz"
    np.savez(
        out,
        mz=np.stack([a[0] for a in arrs]),
        inten=np.stack([a[1] for a in arrs]),
        mask=np.stack([a[2] for a in arrs]),
        precursor=np.minimum(df["precursor_mz"].to_numpy(dtype=np.float32), MAX_MZ),
        mode=np.array([mode_id(m) for m in df["ionization_mode"]], dtype=np.int8),
    )
    print(f"wrote {out} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
