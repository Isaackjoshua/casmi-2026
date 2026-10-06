"""Compute the six-family fingerprint stack for the pool and the training
structures (src/fp_stack2.py).

Same shape as scripts/build_multi_fingerprints.py, and the same two
safeguards, both of which earned their place:

  - built in the POOL's row order, taken from the loaded pool, because
    candidate_pool_from_df sorts by exact_mass and indexing one order with
    the other's indices scores every candidate against an unrelated
    molecule (that bug read -0.18 MRR).
  - aborts unless the morgan2 block reproduces the pool's own fingerprint.
    The first 3239 bits of this stack are also bit-for-bit the v1 stack, so
    that is checked against the existing v1 array as well.

Run: PYTHONPATH=. python3 scripts/build_fp_stack2.py
"""

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from src.fp_stack2 import OFFSETS, PACKED_BYTES, TOTAL_BITS, V1_BITS, packed_from_smiles

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
WORKERS = 28
CHUNK = 2000


def _one(smiles):
    try:
        return packed_from_smiles(smiles)
    except Exception:
        return None


def compute(smiles_list, label):
    t0 = time.time()
    out = np.zeros((len(smiles_list), PACKED_BYTES), dtype=np.uint8)
    ok = np.zeros(len(smiles_list), dtype=bool)
    with Pool(WORKERS) as pool:
        for i, packed in enumerate(pool.imap(_one, smiles_list, chunksize=CHUNK)):
            if packed is not None:
                out[i] = np.frombuffer(packed, dtype=np.uint8)
                ok[i] = True
            if (i + 1) % 200_000 == 0:
                print(f"  {label}: {i+1:,}/{len(smiles_list):,} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  {label}: {len(smiles_list):,} done, {ok.sum():,} ok "
          f"({ok.mean():.2%}) in {time.time()-t0:.0f}s", flush=True)
    return out, ok


def main():
    from src.multi_fingerprint import TOTAL_BITS as V1_TOTAL
    from src.pipeline_v3 import candidate_bits
    from src.propagation import load_candidate_pool

    print(f"stack = {TOTAL_BITS} bits in {len(OFFSETS)} families, "
          f"packed into {PACKED_BYTES} bytes", flush=True)
    for name, (lo, hi) in OFFSETS.items():
        print(f"  {name:9s} bits {lo:5d}-{hi:5d}")
    print(flush=True)

    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")
    packed, ok = compute(list(pool.normalized_smiles), "pool")

    rng = np.random.default_rng(0)
    probe = np.sort(rng.choice(len(pool), size=2000, replace=False))
    bits = np.unpackbits(packed[probe], axis=1)
    a, b = OFFSETS["morgan2"]
    if not np.array_equal(bits[:, a:b], np.stack([candidate_bits(pool, int(i), int(i) + 1)[0]
                                                  for i in probe])):
        raise SystemExit("ABORT: morgan2 block disagrees with the pool's own fingerprint "
                         "-- the rows are misaligned")
    print(f"  verified: morgan2 block matches the pool's fingerprint ({len(probe)} rows)",
          flush=True)

    v1_path = P / "pool_multi_fp.npy"
    if v1_path.exists() and V1_TOTAL == V1_BITS:
        v1 = np.load(v1_path, mmap_mode="r")
        v1_bits = np.unpackbits(np.asarray(v1[probe]), axis=1)[:, :V1_BITS]
        if not np.array_equal(bits[:, :V1_BITS], v1_bits):
            raise SystemExit("ABORT: the first three blocks do not reproduce the v1 stack")
        print(f"  verified: first {V1_BITS} bits reproduce the v1 stack exactly", flush=True)

    np.save(P / "pool_fp2.npy", packed)
    print(f"wrote pool_fp2.npy {packed.shape} ({packed.nbytes/1e9:.2f} GB)\n", flush=True)

    d = np.load(P / "fp_train_data.npz", allow_pickle=True)
    keys = np.asarray(d["keys"])
    train = pd.read_parquet(ROOT / "data/raw/train.parquet",
                            columns=["inchikey14", "normalized_smiles"])
    smiles_of = dict(zip(train["inchikey14"], train["normalized_smiles"]))
    t_packed, t_ok = compute([smiles_of.get(k, "") for k in keys], "train")
    np.savez(P / "train_fp2.npz", packed=t_packed, ok=t_ok, keys=keys)
    print(f"wrote train_fp2.npz {t_packed.shape}")


if __name__ == "__main__":
    main()
