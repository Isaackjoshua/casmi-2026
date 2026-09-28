"""Lay the PubChem pool out for upload: the 24 GB fingerprint array as
~4 GB .npy shards, everything else copied as-is.

Kaggle's uploader is unreliable on single files of tens of GB, and a
failure arrives at the end of an hour-long upload. src.large_pool reads
either layout (fps.npy, or fps_000.npy, fps_001.npy, ...).

Run: PYTHONPATH=. python3 scripts/shard_pool_for_kaggle.py
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARD_BYTES = 4_000_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/pubchem/pubchem_pool"))
    ap.add_argument("--out", default=str(ROOT / "data/pubchem/kaggle_pool"))
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fps = np.load(src / "fps.npy", mmap_mode="r")
    row_bytes = fps.shape[1] * 8
    rows_per_shard = SHARD_BYTES // row_bytes
    n_shards = (len(fps) + rows_per_shard - 1) // rows_per_shard
    print(f"{len(fps):,} rows -> {n_shards} shards of <= {rows_per_shard:,} rows", flush=True)

    t0 = time.time()
    for i in range(n_shards):
        lo = i * rows_per_shard
        hi = min(len(fps), lo + rows_per_shard)
        dst = out / f"fps_{i:03d}.npy"
        if dst.exists() and len(np.load(dst, mmap_mode="r")) == hi - lo:
            print(f"  shard {i}: already written", flush=True)
            continue
        arr = np.lib.format.open_memmap(dst, mode="w+", dtype=fps.dtype, shape=(hi - lo, fps.shape[1]))
        step = 1_000_000
        for j in range(lo, hi, step):
            k = min(hi, j + step)
            arr[j - lo : k - lo] = fps[j:k]
        arr.flush()
        del arr
        print(f"  shard {i}: rows {lo:,}-{hi:,} ({(time.time()-t0)/60:.1f} min)", flush=True)

    for name in ("masses.npy", "popcount.npy", "meta.parquet"):
        dst = out / name
        if not dst.exists():
            shutil.copy2(src / name, dst)
            print(f"  copied {name}", flush=True)

    size = sum(f.stat().st_size for f in out.iterdir()) / 1e9
    print(f"{out}: {size:.1f} GB in {len(list(out.iterdir()))} files ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
