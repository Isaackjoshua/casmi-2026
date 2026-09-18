"""Phase 2: rank an external candidate pool (COCONUT + training-library
structures, see src/candidates.py) by chemical similarity to whichever
training spectra scored well against a query -- "this spectrum looks like
compound X; what does X's close chemical relatives look like" -- rather
than only ever proposing structures we have direct spectral evidence for.

This is what actually closes the gap the Phase 1 baseline can't: a hard
validation using structures absent from the training library scored
0.0000 MRR@25 for library-only search (it confidently guessed wrong every
time, never abstaining), which lines up with the real public score of
0.063 -- badly under even the ~0.15-0.23 "library-only ceiling" public
notebooks reported, most likely because our restricted 5-library subset
covers less of the real test set's structures than their full-library
baselines do. Candidate-pool expansion is the fix public notebooks
converged on: `score(candidate) = max_a sim(a)^4 * Tanimoto(fp_candidate, fp_a)`
over a large external candidate pool, not just the training library.

Fingerprint arithmetic here is pure numpy (bitwise AND + popcount over
2048-bit Morgan fingerprints packed as 32 uint64 words), which is far
faster at this scale (100k+ candidates per query) than converting every
candidate to an RDKit ExplicitBitVect and calling BulkTanimotoSimilarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FP_WORDS = 32  # 2048 bits / 64 bits per word
# Tuned by local sweep on the structure-absent-from-library hard validation
# set (see README). Counterintuitively, this window isn't an "analog hop"
# tolerance at all: candidates are always matched against the QUERY's own
# measured neutral mass, so it's really just accounting for instrument
# mass-measurement error. The original 150 Da ("generous analog window",
# following the public-notebook ablation's own description) diluted every
# ranking with tens of thousands of same-mass-region-but-chemically-
# irrelevant candidates. Score improved *monotonically* as the window
# shrank, bottoming out around 1 mDa -- roughly the real accuracy limit of
# the instrument, since going tighter starts excluding true answers to
# real measurement noise (see pipeline_v2's adaptive widening for spectra
# that come up empty at this width). 0.001 Da / exponent 2.5 scored 0.4392
# MRR@25 on the 150-molecule hard validation set, vs. 0.0211 at the
# original (150 Da, exponent 4) setting -- a >20x improvement.
PROPAGATION_EXPONENT = 2.5
MASS_WINDOW_DA = 0.001
TOP_N_PER_SPECTRUM = 50

# Monoisotopic masses for the ten adducts the test set uses (see the
# competition's Data Landscape notes). Precise to ~1 mDa; any residual
# instrument calibration error (e.g. the +1.55 ppm timsTOF bias a data-audit
# notebook found) is well inside MASS_WINDOW_DA's margin.
_PROTON = 1.007276
_ELECTRON = 0.000549
_H2O = 18.010565
_NH4 = 18.033823
_NA = 22.989770 - _ELECTRON
_K = 38.963707 - _ELECTRON
_CH2O2 = 46.005480
_CL = 34.968853 + _ELECTRON

ADDUCT_MASS_SHIFT = {
    "[M+H]+": _PROTON,
    "[M+NH4]+": _NH4 - _ELECTRON,
    "[M-H2O+H]+": _PROTON - _H2O,
    "[M-2H2O+H]+": _PROTON - 2 * _H2O,
    "[M+Na]+": _NA,
    "[M+K]+": _K,
    "[M-H]-": -_PROTON,
    "[M-H2O-H]-": -_PROTON - _H2O,
    "[M+CH2O2-H]-": _CH2O2 - _PROTON,
    "[M+Cl]-": -_CL,
}


def neutral_mass(precursor_mz: float, adduct: str) -> float | None:
    shift = ADDUCT_MASS_SHIFT.get(adduct)
    return None if shift is None else precursor_mz - shift


@dataclass
class CandidatePool:
    inchikey14: np.ndarray
    normalized_smiles: np.ndarray
    exact_mass: np.ndarray  # sorted ascending
    fp_words: np.ndarray  # (n, FP_WORDS) uint64
    popcount: np.ndarray  # (n,) int32
    index_by_key: dict

    def __len__(self) -> int:
        return len(self.exact_mass)


def candidate_pool_from_df(df: pd.DataFrame) -> CandidatePool:
    """df must have inchikey14, normalized_smiles, exact_mass, fingerprint
    (packed bytes) columns -- shared by both the cached-parquet path
    (load_candidate_pool) and building the pool inline inside a Kaggle
    kernel run, where fingerprinting happens at runtime instead of being
    precomputed.
    """
    df = df.sort_values("exact_mass").reset_index(drop=True)
    fp_words = np.stack([np.frombuffer(b, dtype=np.uint64) for b in df["fingerprint"]])
    popcount = np.bitwise_count(fp_words).sum(axis=1).astype(np.int64)
    return CandidatePool(
        inchikey14=df["inchikey14"].to_numpy(),
        normalized_smiles=df["normalized_smiles"].to_numpy(),
        exact_mass=df["exact_mass"].to_numpy(dtype=np.float64),
        fp_words=fp_words,
        popcount=popcount,
        index_by_key={k: i for i, k in enumerate(df["inchikey14"])},
    )


def load_candidate_pool(path: str) -> CandidatePool:
    return candidate_pool_from_df(pd.read_parquet(path))


def mass_window(pool: CandidatePool, query_neutral_mass: float, mass_window_da: float) -> tuple[int, int]:
    """[lo, hi) index range of pool candidates within +/- mass_window_da of
    the query's neutral mass (pool.exact_mass is sorted ascending).
    """
    lo = int(np.searchsorted(pool.exact_mass, query_neutral_mass - mass_window_da, side="left"))
    hi = int(np.searchsorted(pool.exact_mass, query_neutral_mass + mass_window_da, side="right"))
    return lo, hi


def propagation_scores(
    anchors: list[tuple[str, str, float]],
    lo: int,
    hi: int,
    pool: CandidatePool,
    exponent: float = PROPAGATION_EXPONENT,
) -> np.ndarray:
    """Propagated score for every candidate in pool[lo:hi]:
    max over anchors of sim(anchor)^exponent * Tanimoto(candidate, anchor).
    """
    cand_words = pool.fp_words[lo:hi]  # (M, FP_WORDS)
    cand_pop = pool.popcount[lo:hi]  # (M,)
    best = np.zeros(hi - lo, dtype=np.float64)

    for key, _smiles, sim in anchors:
        anchor_idx = pool.index_by_key.get(key)
        if anchor_idx is None:
            continue
        anchor_words = pool.fp_words[anchor_idx]
        anchor_pop = pool.popcount[anchor_idx]

        intersection = np.bitwise_count(cand_words & anchor_words[None, :]).sum(axis=1)
        union = cand_pop + anchor_pop - intersection
        tanimoto = np.divide(intersection, union, out=np.zeros_like(union, dtype=np.float64), where=union > 0)

        weighted = (sim**exponent) * tanimoto
        np.maximum(best, weighted, out=best)
    return best


def propagate(
    anchors: list[tuple[str, str, float]],
    query_neutral_mass: float,
    pool: CandidatePool,
    mass_window_da: float = MASS_WINDOW_DA,
    exponent: float = PROPAGATION_EXPONENT,
    top_n: int = TOP_N_PER_SPECTRUM,
) -> list[tuple[str, str, float]]:
    """anchors: (inchikey14, smiles, spectral_similarity) tuples from
    score_spectrum -- library spectra that scored well against the query.
    Returns (inchikey14, smiles, propagated_score) for the best candidates
    in the mass-windowed pool, best first.
    """
    if not anchors or query_neutral_mass is None:
        return []

    lo, hi = mass_window(pool, query_neutral_mass, mass_window_da)
    if hi <= lo:
        return []

    best = propagation_scores(anchors, lo, hi, pool, exponent)

    nonzero = np.nonzero(best)[0]
    if len(nonzero) == 0:
        return []
    order = nonzero[np.argsort(best[nonzero])[::-1]][:top_n]
    return [(pool.inchikey14[lo + i], pool.normalized_smiles[lo + i], float(best[i])) for i in order]

