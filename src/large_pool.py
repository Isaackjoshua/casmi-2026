"""A candidate pool too large to hold in memory.

The COCONUT+training pool (729k structures) fits comfortably in RAM as a
CandidatePool. The PubChem pool does not: 94M structures is 24 GB of
fingerprints, ~10 GB of Python strings for the SMILES and InChIKey14
columns, and an index_by_key dict over 94M keys on top of that -- an
eager version of this reached 36 GB RSS in under a minute before being
killed.

Three things make it fit instead:

  - fingerprints stay memory-mapped. Every query only touches the handful
    of rows inside its ~1 mDa mass window, so the OS pages in kilobytes,
    not gigabytes.
  - popcounts are precomputed once into an int16 array (188 MB), since
    recomputing them per query would defeat the memmap.
  - SMILES and InChIKey14 are never loaded. They're read from the parquet
    by row index, and only for the ~25 candidates that actually get
    returned.

There is deliberately no index_by_key: that dict exists to look up an
*anchor's* fingerprint during propagation, and anchors are always
training-library structures, which live in the small in-memory pool. So
propagation resolves anchors against the small pool and scores candidates
against this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class SplitMemmap:
    """A (n, 32) uint64 array stored as several .npy shards, sliceable as one.

    Kaggle's uploader is unreliable on single files of tens of GB, so the
    24 GB fingerprint array ships as ~4 GB pieces. Only contiguous [lo:hi]
    slicing and single-row indexing are needed, because every query reads
    one contiguous mass window.
    """

    def __init__(self, paths):
        self.parts = [np.load(p, mmap_mode="r") for p in paths]
        self.offsets = np.cumsum([0] + [len(x) for x in self.parts])
        self.shape = (int(self.offsets[-1]), int(self.parts[0].shape[1]))
        self.dtype = self.parts[0].dtype

    def __len__(self):
        return self.shape[0]

    def _part(self, row):
        return int(np.searchsorted(self.offsets, row, side="right") - 1)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            i = self._part(int(key))
            return self.parts[i][int(key) - self.offsets[i]]
        lo, hi, step = key.indices(self.shape[0])
        if step != 1:
            raise ValueError("SplitMemmap supports contiguous slices only")
        if hi <= lo:
            return np.empty((0, self.shape[1]), dtype=self.dtype)
        first, last = self._part(lo), self._part(hi - 1)
        if first == last:
            base = self.offsets[first]
            return self.parts[first][lo - base : hi - base]
        chunks = []
        for i in range(first, last + 1):
            base, end = self.offsets[i], self.offsets[i + 1]
            chunks.append(self.parts[i][max(lo, base) - base : min(hi, end) - base])
        return np.concatenate(chunks)


@dataclass
class LargePool:
    exact_mass: np.ndarray  # (n,) float64, ascending -- mass windows are slices of this
    fp_words: np.ndarray  # (n, 32) uint64, memory-mapped
    popcount: np.ndarray  # (n,) int16
    meta_path: Path  # parquet with inchikey14 + normalized_smiles, row-aligned
    _rg_offsets: np.ndarray | None = None  # row-group start offsets, built on first use
    _pf: object = None

    def __len__(self) -> int:
        return len(self.exact_mass)

    def rows(self, indices) -> list[tuple[str, str]]:
        """(inchikey14, smiles) for the given row indices, read from parquet
        rather than held in memory. Indices from one mass window are
        contiguous, so this touches one or two row groups.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return []
        if self._rg_offsets is None:
            pf = pq.ParquetFile(self.meta_path)
            sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
            self._rg_offsets = np.cumsum([0] + sizes)
            self._pf = pf

        out = {}
        # group the requested indices by the row group holding them
        rgs = np.searchsorted(self._rg_offsets, indices, side="right") - 1
        for rg in np.unique(rgs):
            tbl = self._pf.read_row_group(int(rg), columns=["inchikey14", "normalized_smiles"])
            k = tbl.column("inchikey14").to_numpy(zero_copy_only=False)
            s = tbl.column("normalized_smiles").to_numpy(zero_copy_only=False)
            base = self._rg_offsets[rg]
            for gi in indices[rgs == rg]:
                out[int(gi)] = (k[gi - base], s[gi - base])
        return [out[int(i)] for i in indices]


def load_large_pool(path: Path | str) -> LargePool:
    path = Path(path)
    masses = np.load(path / "masses.npy", mmap_mode="r")
    if (path / "fps.npy").exists():
        fps = np.load(path / "fps.npy", mmap_mode="r")
    else:
        shards = sorted(path.glob("fps_*.npy"))
        if not shards:
            raise FileNotFoundError(f"no fps.npy or fps_*.npy in {path}")
        fps = SplitMemmap(shards)
        if len(fps) != len(masses):
            raise ValueError(f"fps shards hold {len(fps)} rows, masses.npy has {len(masses)}")

    pc_path = path / "popcount.npy"
    if pc_path.exists():
        popcount = np.load(pc_path)
    else:
        # one pass, chunked, so the 24 GB of fingerprints never lands in RAM
        popcount = np.empty(len(masses), dtype=np.int16)
        step = 2_000_000
        for i in range(0, len(masses), step):
            popcount[i : i + step] = np.bitwise_count(np.asarray(fps[i : i + step])).sum(axis=1)
        np.save(pc_path, popcount)

    return LargePool(
        exact_mass=np.asarray(masses, dtype=np.float64),
        fp_words=fps,
        popcount=popcount,
        meta_path=path / "meta.parquet",
    )


def window(pool: LargePool, neutral_mass: float, half_width: float) -> tuple[int, int]:
    lo = int(np.searchsorted(pool.exact_mass, neutral_mass - half_width, side="left"))
    hi = int(np.searchsorted(pool.exact_mass, neutral_mass + half_width, side="right"))
    return lo, hi


def tanimoto_against(pool: LargePool, lo: int, hi: int, anchor_words: np.ndarray, anchor_pop: int) -> np.ndarray:
    """Tanimoto of every candidate in pool[lo:hi] against one anchor
    fingerprint. Materializes only the window.
    """
    cand = np.asarray(pool.fp_words[lo:hi])
    inter = np.bitwise_count(cand & anchor_words[None, :]).sum(axis=1)
    union = pool.popcount[lo:hi].astype(np.int64) + anchor_pop - inter
    return np.divide(inter, union, out=np.zeros(hi - lo, dtype=np.float64), where=union > 0)

def tanimoto_matrix(cand_words, cand_pop, anchor_words, anchor_pop):
    """Tanimoto of every candidate against every anchor -> (n_cand, n_anchor).

    Takes the window's fingerprints already materialized, because the caller
    reads it once: computing this one anchor at a time re-sliced the memmap
    per anchor, which on Kaggle's network-mounted input is the dominant cost
    of a query.
    """
    cand_words = np.asarray(cand_words)
    anchor_words = np.asarray(anchor_words)
    if cand_words.shape[0] == 0 or anchor_words.shape[0] == 0:
        return np.zeros((cand_words.shape[0], anchor_words.shape[0]), dtype=np.float64)
    inter = np.bitwise_count(cand_words[:, None, :] & anchor_words[None, :, :]).sum(axis=2)
    union = np.asarray(cand_pop, dtype=np.int64)[:, None] + np.asarray(anchor_pop, dtype=np.int64)[None, :] - inter
    return np.divide(inter, union, out=np.zeros(inter.shape, dtype=np.float64), where=union > 0)
