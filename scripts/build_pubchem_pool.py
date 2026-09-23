"""Turn PubChem's bulk CID-SMILES dump into a mass-sorted, fingerprinted
candidate pool.

Why: the candidate pool (COCONUT + training structures) covers only ~39%
of real natural-product chemistry -- measured against the GNPS library and
our own hard-validation set -- and 72% of what COCONUT misses is in
PubChem. Since the test set is explicitly "natural products, hypothesised
natural products, natural product analogs, and synthetic molecules that
might plausibly occur in nature", the analogs and synthetics live in
PubChem, not COCONUT. Coverage, not ranking, is what caps the score.

Filters (a candidate must at least be a plausible small organic molecule
in the test's mass range):
  - largest fragment only (PubChem mixtures are mostly salts)
  - elements restricted to ELEMENTS (CHNOPS + halogens)
  - exact mass within [MIN_MASS, MAX_MASS]
  - deduplicated on InChIKey14, which is what the competition scores on --
    this collapses PubChem's many stereoisomers and salt forms of the same
    skeleton, and is a large fraction of the 124M input

Two passes, because 124M rows will not fit in memory (an earlier
single-pass version reached 2.7 GB RSS at 4M rows, heading for ~84 GB):
  1. stream the dump, and flush each SHARD_ROWS-row batch to its own
     mass-sorted shard on disk
  2. k-way merge the shards by mass, deduplicating on the way (identical
     InChIKey14 implies identical mass, so duplicates arrive adjacent),
     writing the final arrays sequentially

Output (data/pubchem/pubchem_pool/):
  masses.npy     float64 (n,)    exact masses, ascending -- the sort order
  fps.npy        uint64  (n,32)  Morgan fingerprints, same order
  meta.parquet                   inchikey14 + normalized_smiles, same order
Mass order means a query's mass window is a contiguous slice, and keeping
fingerprints in their own .npy lets the Kaggle kernel memory-map them
rather than loading tens of GB into RAM.

Run: PYTHONPATH=. python3 scripts/build_pubchem_pool.py
"""

import argparse
import gzip
import heapq
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from src.candidates import FP_BITS, FP_RADIUS

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MIN_MASS = 120.0  # test monoisotopic masses run 157-1159; pad for adduct/loss slack
MAX_MASS = 1300.0
ELEMENTS = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Se", "B", "Si"}
CHUNK = 200_000
SHARD_ROWS = 4_000_000

_GEN = None


def _gen():
    global _GEN
    if _GEN is None:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    return _GEN


def _process(smiles: str):
    """SMILES -> (inchikey14, canonical smiles, exact mass, packed fp) or None."""
    if "." in smiles:
        smiles = max(smiles.split("."), key=len)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ELEMENTS:
            return None
    mass = Descriptors.ExactMolWt(mol)
    if not (MIN_MASS <= mass <= MAX_MASS):
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not key:
        return None
    fp = _gen().GetFingerprint(mol)
    packed = np.packbits(np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")).tobytes()
    return key.split("-")[0], Chem.MolToSmiles(mol), mass, packed


def _flush_shard(rows: list, shard_dir: Path, idx: int) -> Path:
    """Write one mass-sorted shard. rows: list of (key, smiles, mass, fp)."""
    rows.sort(key=lambda r: r[2])
    path = shard_dir / f"shard_{idx:04d}.npz"
    np.savez(
        path,
        masses=np.array([r[2] for r in rows], dtype=np.float64),
        fps=np.stack([np.frombuffer(r[3], dtype=np.uint64) for r in rows]),
        keys=np.array([r[0] for r in rows], dtype=object),
        smiles=np.array([r[1] for r in rows], dtype=object),
    )
    return path


def _merge_shards(shard_paths: list, out: Path) -> int:
    """K-way merge by mass, dropping duplicate InChIKey14, writing the
    final arrays sequentially so peak memory stays at one shard.
    """
    shards = [np.load(p, allow_pickle=True) for p in shard_paths]
    streams = []
    for si, z in enumerate(shards):
        m, f, k, s = z["masses"], z["fps"], z["keys"], z["smiles"]
        streams.append((m, f, k, s))

    total = sum(len(st[0]) for st in streams)
    masses_out = np.lib.format.open_memmap(out / "masses.npy", mode="w+", dtype=np.float64, shape=(total,))
    fps_out = np.lib.format.open_memmap(out / "fps.npy", mode="w+", dtype=np.uint64, shape=(total, 32))

    writer = None
    buf_keys, buf_smiles = [], []
    seen = set()
    n = 0

    heap = [(streams[i][0][0], i, 0) for i in range(len(streams)) if len(streams[i][0])]
    heapq.heapify(heap)
    while heap:
        mass, si, pos = heapq.heappop(heap)
        m, f, k, s = streams[si]
        key = k[pos]
        if key not in seen:
            seen.add(key)
            masses_out[n] = mass
            fps_out[n] = f[pos]
            buf_keys.append(key)
            buf_smiles.append(s[pos])
            n += 1
            if len(buf_keys) >= 1_000_000:
                tbl = pa.table({"inchikey14": buf_keys, "normalized_smiles": buf_smiles})
                writer = writer or pq.ParquetWriter(out / "meta.parquet", tbl.schema)
                writer.write_table(tbl)
                buf_keys, buf_smiles = [], []
        if pos + 1 < len(m):
            heapq.heappush(heap, (m[pos + 1], si, pos + 1))

    if buf_keys:
        tbl = pa.table({"inchikey14": buf_keys, "normalized_smiles": buf_smiles})
        writer = writer or pq.ParquetWriter(out / "meta.parquet", tbl.schema)
        writer.write_table(tbl)
    if writer:
        writer.close()

    # trim the memmaps to the deduplicated length
    masses_out.flush(); fps_out.flush()
    del masses_out, fps_out
    m = np.load(out / "masses.npy", mmap_mode="r")[:n]
    np.save(out / "masses_trim.npy", m)
    del m
    f = np.load(out / "fps.npy", mmap_mode="r")[:n]
    np.save(out / "fps_trim.npy", f)
    del f
    (out / "masses.npy").unlink(); (out / "fps.npy").unlink()
    (out / "masses_trim.npy").rename(out / "masses.npy")
    (out / "fps_trim.npy").rename(out / "fps.npy")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/pubchem/CID-SMILES.gz"))
    ap.add_argument("--out", default=str(ROOT / "data/pubchem/pubchem_pool"))
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--limit", type=int, default=0, help="stop after N input lines (smoke tests)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / "shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir()

    t0 = time.time()
    rows, shard_paths, n_in, n_kept = [], [], 0, 0

    with gzip.open(args.input, "rt") as fh, Pool(args.workers) as pool:
        def batches():
            batch = []
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    batch.append(parts[1])
                if len(batch) >= CHUNK:
                    yield batch
                    batch = []
            if batch:
                yield batch

        for batch in batches():
            n_in += len(batch)
            rows.extend(r for r in pool.map(_process, batch, chunksize=1000) if r is not None)
            if len(rows) >= SHARD_ROWS:
                n_kept += len(rows)
                shard_paths.append(_flush_shard(rows, shard_dir, len(shard_paths)))
                rows = []
                el = time.time() - t0
                print(f"  read {n_in/1e6:.1f}M, kept {n_kept/1e6:.1f}M, {len(shard_paths)} shards, "
                      f"{n_in/el/1000:.0f}k/s, {el/60:.1f} min", flush=True)
            if args.limit and n_in >= args.limit:
                break

    if rows:
        n_kept += len(rows)
        shard_paths.append(_flush_shard(rows, shard_dir, len(shard_paths)))
    print(f"pass 1 done: read {n_in}, kept {n_kept} in {len(shard_paths)} shards ({(time.time()-t0)/60:.1f} min)")

    n = _merge_shards(shard_paths, out)
    print(f"pass 2 done: {n} unique structures after InChIKey14 dedup "
          f"({n/max(n_kept,1):.0%} of kept)")
    shutil.rmtree(shard_dir)
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e9
    print(f"wrote {out}: {n} structures, {size:.1f} GB ({(time.time()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()
