"""Compute the multi-fingerprint stack for the candidate pool and for the
training structures.

Two aligned outputs, both packed to src.multi_fingerprint.PACKED_BYTES:

  data/processed/pool_multi_fp.npy    (729387, 405)  row-aligned with
                                      candidate_fingerprints.parquet, so a
                                      mass window is still a contiguous slice
  data/processed/train_multi_fp.npz   targets for the 275,659 structures in
                                      fp_train_data.npz, in its `keys` order,
                                      plus an `ok` mask for the few that fail

The training targets are keyed by InChIKey14 and fp_train_data carries no
SMILES, so they are resolved through the train parquet's normalized_smiles.

Run: PYTHONPATH=. python3 scripts/build_multi_fingerprints.py
"""

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from src.multi_fingerprint import PACKED_BYTES, TOTAL_BITS, packed_from_smiles

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
    print(f"stack = {TOTAL_BITS} bits packed into {PACKED_BYTES} bytes\n", flush=True)

    # Build in the POOL's row order, not the parquet's. candidate_pool_from_df
    # sorts by exact_mass, so the two differ -- and indexing one with the
    # other's indices scores every candidate against an unrelated molecule's
    # fingerprint, which cost -0.18 MRR before it was caught. Taking the
    # SMILES from the loaded pool makes the alignment true by construction,
    # rather than re-deriving a sort (pandas' default sort is not stable, so
    # mass ties need not order the same way twice).
    from src.pipeline_v3 import candidate_bits
    from src.propagation import load_candidate_pool
    from src.multi_fingerprint import OFFSETS

    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")
    packed, ok = compute(list(pool.normalized_smiles), "pool")

    # and check it: the morgan2 block must equal the pool's own fingerprint
    a, b = OFFSETS["morgan2"]
    rng = np.random.default_rng(0)
    probe = np.sort(rng.choice(len(pool), size=2000, replace=False))
    mine = np.unpackbits(packed[probe], axis=1)[:, a:b]
    theirs = np.stack([candidate_bits(pool, int(i), int(i) + 1)[0] for i in probe])
    if not np.array_equal(mine, theirs):
        bad = int((mine != theirs).any(axis=1).sum())
        raise SystemExit(f"ABORT: morgan2 block disagrees with the pool's own "
                         f"fingerprint for {bad}/{len(probe)} probed rows -- "
                         f"the rows are misaligned")
    print(f"  verified: morgan2 block matches the pool's fingerprint on "
          f"{len(probe)} probed rows", flush=True)

    np.save(P / "pool_multi_fp.npy", packed)
    np.save(P / "pool_multi_fp_ok.npy", ok)
    print(f"wrote pool_multi_fp.npy {packed.shape} "
          f"({packed.nbytes/1e9:.2f} GB)\n", flush=True)

    d = np.load(P / "fp_train_data.npz", allow_pickle=True)
    keys = np.asarray(d["keys"])
    train = pd.read_parquet(ROOT / "data/raw/train.parquet",
                            columns=["inchikey14", "normalized_smiles"])
    smiles_of = dict(zip(train["inchikey14"], train["normalized_smiles"]))
    missing = sum(1 for k in keys if k not in smiles_of)
    print(f"{len(keys):,} training structures, {missing} without a SMILES in train.parquet",
          flush=True)
    t_packed, t_ok = compute([smiles_of.get(k, "") for k in keys], "train")
    np.savez(P / "train_multi_fp.npz", packed=t_packed, ok=t_ok, keys=keys)
    print(f"wrote train_multi_fp.npz {t_packed.shape}")


if __name__ == "__main__":
    main()
