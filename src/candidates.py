"""Phase 2: candidate-pool expansion beyond the training library.

The Phase 1 baseline can only ever guess structures already present in its
training library -- confirmed by a held-out validation using structures
absent from the library, which scored 0.0000 MRR@25 (it never abstained,
it just confidently guessed wrong every time). Public-notebook research
into this competition found the actual score-driver is expanding the
candidate pool to an external natural-product database (COCONUT, ~738k
structures) and ranking those by chemical (Tanimoto fingerprint)
similarity to training-library structures that scored well on spectral
similarity -- i.e. "this spectrum looks like compound X; what does X's
close chemical relatives look like, even if we've never seen their
spectra."

This module builds and caches Morgan fingerprints for the combined
candidate pool (COCONUT structures + unique training-library structures)
so the expensive fingerprinting step (RDKit, ~1M molecules) happens once
and is reused across pipeline runs -- and, packaged as a Kaggle Dataset,
across Kaggle kernel submissions too, so a 9h-budget run doesn't have to
redo it.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import ExplicitBitVect

FP_RADIUS = 2
FP_BITS = 2048

_GENERATOR = None  # lazily created per worker process; not picklable to share across mp.Pool


def _get_generator() -> rdFingerprintGenerator.FingerprintGenerator64:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    return _GENERATOR


def _fingerprint_one(smiles: str) -> tuple[bytes, float] | tuple[None, None]:
    """Returns (packed fingerprint bytes, RDKit-computed exact mass), both
    from the same parsed mol -- computing mass here (rather than trusting
    whatever mass a source table shipped) keeps COCONUT and training-library
    candidates on the same convention for the mass-window filter.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = _get_generator().GetFingerprint(mol)
    return bytes_from_bitvect(fp), Descriptors.ExactMolWt(mol)


def bytes_from_bitvect(fp: ExplicitBitVect) -> bytes:
    return np.packbits(np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")).tobytes()


def bitvect_from_bytes(packed: bytes) -> ExplicitBitVect:
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:FP_BITS]
    fp = ExplicitBitVect(FP_BITS)
    fp.SetBitsFromList(np.nonzero(bits)[0].tolist())
    return fp


def fingerprint_many(
    smiles_list: list[str], n_workers: int | None = None
) -> tuple[list[bytes | None], list[float | None]]:
    """Parallel Morgan fingerprinting -- the dominant cost of building the
    candidate pool (roughly 1M molecules between COCONUT and the training
    library's unique structures). Returns (fingerprints, exact_masses),
    parallel lists aligned to smiles_list.
    """
    n_workers = n_workers or max(1, mp.cpu_count() - 1)
    with mp.Pool(n_workers) as pool:
        results = pool.map(_fingerprint_one, smiles_list, chunksize=500)
    fps, masses = zip(*results) if results else ((), ())
    return list(fps), list(masses)


def load_coconut(path: Path | str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["inchikey14"] = df["inchikey"].str.split("-").str[0]
    return df.rename(columns={"canonical_smiles": "normalized_smiles"})[
        ["inchikey14", "normalized_smiles", "exact_mass", "formula"]
    ]


def library_structures(train_df: pd.DataFrame) -> pd.DataFrame:
    """One representative SMILES per unique training-library structure --
    the other half of the candidate pool (COCONUT covers structures with
    no training spectra at all; this half keeps the structures we do have
    direct spectral evidence for in the same pool, so Class 1 candidates
    aren't lost when this pool replaces plain library lookup).
    """
    df = train_df.dropna(subset=["inchikey14", "normalized_smiles"])
    df = df.drop_duplicates(subset=["inchikey14"], keep="first")
    return df[["inchikey14", "normalized_smiles"]].assign(exact_mass=np.nan, formula=None)


def build_candidate_pool(coconut_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Combined, deduplicated candidate pool: COCONUT ∪ unique training
    structures, keyed by inchikey14 (COCONUT wins on conflicts since it
    carries a real exact_mass).
    """
    lib = library_structures(train_df)
    combined = pd.concat([coconut_df, lib], ignore_index=True)
    combined = combined.drop_duplicates(subset=["inchikey14"], keep="first")
    return combined.reset_index(drop=True)
