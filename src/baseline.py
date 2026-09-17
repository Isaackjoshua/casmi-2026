"""Phase 1 baseline: precursor-mass-windowed spectral similarity search
against the training library, aggregated per test molecule.

This is deliberately close to the public "analog propagation" / "spectral
cosine" notebooks that currently top the leaderboard (~0.335-0.339
MRR@25) -- the goal is a working, submittable pipeline first, then layer
Phase 2+ improvements (formula filtering, learned embeddings, domain
adaptation) on top of this scaffold.

Usage (once data/raw/{train,test}.parquet exist):

    import pandas as pd
    from src.data import load_train, load_test
    from src.baseline import build_library, predict_test_set
    from src.submission import write_submission

    train = load_train()
    test = load_test()

    # Instrument-matched + natural-product libraries first -- see the
    # roadmap doc's "Data landscape" section for why this subset matters.
    lib_sources = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]
    library = build_library(train[train["ingest_lib"].isin(lib_sources)])

    predictions = predict_test_set(test, library)
    write_submission(predictions, "submissions/baseline_v1.csv")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spectral_similarity import modified_cosine_similarity

PPM_WINDOW = 30  # precursor mass tolerance for candidate pre-filtering
ANALOG_DA_WINDOW = 100  # wider net for structural analogs (Class 2-ish)
TOP_N_PER_SPECTRUM = 50  # candidates kept per test spectrum before aggregation
N_GUESSES = 25


@dataclass
class Library:
    """Preprocessed training spectra, sorted by precursor_mz for fast
    windowed lookup via binary search.
    """

    precursor_mz: np.ndarray
    ionization_mode: np.ndarray
    ms2_mzs: list
    ms2_intensities: list
    inchikey14: np.ndarray
    normalized_smiles: np.ndarray


def build_library(train_df: pd.DataFrame) -> Library:
    df = train_df.dropna(subset=["precursor_mz", "inchikey14", "normalized_smiles"])
    df = df.sort_values("precursor_mz").reset_index(drop=True)
    return Library(
        precursor_mz=df["precursor_mz"].to_numpy(),
        ionization_mode=df["ionization_mode"].to_numpy(),
        ms2_mzs=[np.asarray(x, dtype=float) for x in df["ms2_mzs"]],
        ms2_intensities=[np.asarray(x, dtype=float) for x in df["ms2_normalized_intensities"]],
        inchikey14=df["inchikey14"].to_numpy(),
        normalized_smiles=df["normalized_smiles"].to_numpy(),
    )


def _candidate_window(lib: Library, precursor_mz: float, da_window: float) -> np.ndarray:
    lo = np.searchsorted(lib.precursor_mz, precursor_mz - da_window, side="left")
    hi = np.searchsorted(lib.precursor_mz, precursor_mz + da_window, side="right")
    return np.arange(lo, hi)


def score_spectrum(
    query_mzs: np.ndarray,
    query_intensities: np.ndarray,
    query_precursor: float,
    query_ionization_mode: str,
    lib: Library,
) -> list[tuple[str, str, float]]:
    """Return (inchikey14, smiles, similarity) tuples for the best-matching
    library spectra to one query spectrum, best first.
    """
    ppm_da = query_precursor * PPM_WINDOW / 1e6
    tight = _candidate_window(lib, query_precursor, max(ppm_da, 0.01))
    wide = _candidate_window(lib, query_precursor, ANALOG_DA_WINDOW)
    idx = np.union1d(tight, wide)
    idx = idx[lib.ionization_mode[idx] == query_ionization_mode]

    scored = []
    for i in idx:
        sim = modified_cosine_similarity(
            query_mzs, query_intensities, query_precursor,
            lib.ms2_mzs[i], lib.ms2_intensities[i], lib.precursor_mz[i],
        )
        if sim > 0:
            scored.append((lib.inchikey14[i], lib.normalized_smiles[i], sim))

    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:TOP_N_PER_SPECTRUM]


def predict_molecule(spectra_rows: list[dict], lib: Library) -> list[str]:
    """Aggregate evidence across all of a molecule's spectra (different
    collision energies / adducts) into one ranked SMILES list. Aggregation
    rule: best similarity score seen for a given structure across any of
    the molecule's spectra (max-pooling), which rewards a structure that
    matches well even from just one supporting spectrum.
    """
    best_score: dict[str, float] = {}
    best_smiles: dict[str, str] = {}

    for row in spectra_rows:
        hits = score_spectrum(
            np.asarray(row["ms2_mzs"], dtype=float),
            np.asarray(row["ms2_normalized_intensities"], dtype=float),
            float(row["precursor_mz"]),
            row["ionization_mode"],
            lib,
        )
        for key, smiles, sim in hits:
            if sim > best_score.get(key, 0.0):
                best_score[key] = sim
                best_smiles[key] = smiles

    ranked_keys = sorted(best_score, key=best_score.get, reverse=True)[:N_GUESSES]
    return [best_smiles[k] for k in ranked_keys]


def predict_test_set(test_df: pd.DataFrame, lib: Library) -> dict[str, list[str]]:
    predictions = {}
    for mid, group in test_df.groupby("molecule_id"):
        guesses = predict_molecule(group.to_dict("records"), lib)
        if not guesses:
            # No similarity hit at all -- fall back to the single most
            # common structure in-window would be better than nothing;
            # placeholder for now so the submission stays valid.
            guesses = [lib.normalized_smiles[0]]
        predictions[mid] = guesses
    return predictions
