"""Phase 2 pipeline: Phase 1's library search finds spectrally-similar
training analogs; this module propagates from those analogs into the
external candidate pool (COCONUT + training-library structures) by
chemical similarity, so the pipeline can guess structures it has never
seen a spectrum for. See src/propagation.py for why this is necessary --
the Phase 1 baseline scored 0.0000 MRR@25 on a held-out set of structures
absent from its library, and 0.063 on the real hidden test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import FINE_TOP_N, N_GUESSES, Library, _global_fallback_candidates, score_spectrum
from .metric import to_inchikey14
from .propagation import MASS_WINDOW_DA, PROPAGATION_EXPONENT, CandidatePool, neutral_mass, propagate

MERGE_TOL_DA = 0.02


def merge_spectra(rows: list[dict], tol_da: float = MERGE_TOL_DA) -> dict:
    """Combine spectra measured at different collision energies (same
    molecule, same adduct) into one consensus spectrum, so anchor search
    sees the union of fragment evidence in one pass instead of searching
    once per collision energy independently. Peaks within tol_da of each
    other are merged into one (intensity-weighted-average m/z, summed
    intensity), then renormalized so the base peak is 1.0.
    """
    all_mzs = np.concatenate([np.asarray(r["ms2_mzs"], dtype=np.float64) for r in rows])
    all_ints = np.concatenate([np.asarray(r["ms2_normalized_intensities"], dtype=np.float64) for r in rows])
    order = np.argsort(all_mzs)
    all_mzs, all_ints = all_mzs[order], all_ints[order]

    merged_mzs, merged_ints = [], []
    i, n = 0, len(all_mzs)
    while i < n:
        j = i
        while j + 1 < n and all_mzs[j + 1] - all_mzs[i] <= tol_da:
            j += 1
        cluster_ints = all_ints[i : j + 1]
        cluster_mzs = all_mzs[i : j + 1]
        total = cluster_ints.sum()
        merged_mzs.append(float(np.average(cluster_mzs, weights=cluster_ints)) if total > 0 else float(cluster_mzs.mean()))
        merged_ints.append(float(total))
        i = j + 1

    merged_mzs = np.asarray(merged_mzs)
    merged_ints = np.asarray(merged_ints)
    peak = merged_ints.max() if len(merged_ints) else 0.0
    if peak > 0:
        merged_ints = merged_ints / peak

    return {
        "ms2_mzs": merged_mzs,
        "ms2_normalized_intensities": merged_ints,
        "precursor_mz": rows[0]["precursor_mz"],
        "adduct": rows[0]["adduct"],
        "ionization_mode": rows[0]["ionization_mode"],
    }


def predict_molecule(
    spectra_rows: list[dict],
    lib: Library,
    pool: CandidatePool,
    mass_window_da: float = MASS_WINDOW_DA,
    exponent: float = PROPAGATION_EXPONENT,
) -> list[str]:
    """Group a molecule's spectra by adduct and merge each group's collision
    energies into one consensus spectrum (merge_spectra) before anchor
    search -- richer fragment evidence per search than treating every raw
    spectrum independently. Find library analogs (Phase 1's score_spectrum)
    for each merged spectrum, propagate to the external candidate pool by
    chemical similarity, max-pool the propagated scores across adducts,
    then dedupe on the tautomer-canonical InChIKey14 the competition
    actually scores against, same as Phase 1.
    """
    best_score: dict[str, float] = {}
    best_smiles: dict[str, str] = {}

    by_adduct: dict[str, list[dict]] = {}
    for row in spectra_rows:
        by_adduct.setdefault(row["adduct"], []).append(row)

    for group in by_adduct.values():
        merged = merge_spectra(group) if len(group) > 1 else group[0]
        anchors = score_spectrum(
            merged["ms2_mzs"], merged["ms2_normalized_intensities"], float(merged["precursor_mz"]),
            str(merged["ionization_mode"]).strip().lower(), lib,
        )
        qmass = neutral_mass(float(merged["precursor_mz"]), merged["adduct"])
        if qmass is None:
            continue

        for key, smiles, score in propagate(anchors, qmass, pool, mass_window_da, exponent):
            if score > best_score.get(key, 0.0):
                best_score[key] = score
                best_smiles[key] = smiles

    ranked_raw_keys = sorted(best_score, key=best_score.get, reverse=True)[:FINE_TOP_N]

    final: dict[str, str] = {}
    for raw_key in ranked_raw_keys:
        smiles = best_smiles[raw_key]
        canonical_key = to_inchikey14(smiles) or raw_key
        if canonical_key not in final:
            final[canonical_key] = smiles
        if len(final) >= N_GUESSES:
            break

    return list(final.values())


def predict_test_set(
    test_df: pd.DataFrame,
    lib: Library,
    pool: CandidatePool,
    mass_window_da: float = MASS_WINDOW_DA,
    exponent: float = PROPAGATION_EXPONENT,
) -> dict[str, list[str]]:
    predictions = {}
    n_fallback = 0
    for mid, group in test_df.groupby("molecule_id"):
        guesses = predict_molecule(group.to_dict("records"), lib, pool, mass_window_da, exponent)
        if not guesses:
            n_fallback += 1
            mode = str(group.iloc[0]["ionization_mode"]).strip().lower()
            guesses = _global_fallback_candidates(lib, mode, N_GUESSES)
        predictions[mid] = guesses

    if n_fallback:
        print(f"WARNING: {n_fallback}/{len(predictions)} molecules had zero propagated hits "
              f"and used the frequency fallback -- inspect these before trusting the score.")
    return predictions
