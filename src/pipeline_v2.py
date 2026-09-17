"""Phase 2 pipeline: Phase 1's library search finds spectrally-similar
training analogs; this module propagates from those analogs into the
external candidate pool (COCONUT + training-library structures) by
chemical similarity, so the pipeline can guess structures it has never
seen a spectrum for. See src/propagation.py for why this is necessary --
the Phase 1 baseline scored 0.0000 MRR@25 on a held-out set of structures
absent from its library, and 0.063 on the real hidden test set.
"""

from __future__ import annotations

import pandas as pd

from .baseline import FINE_TOP_N, N_GUESSES, Library, _global_fallback_candidates, score_spectrum
from .metric import to_inchikey14
from .propagation import CandidatePool, neutral_mass, propagate


def predict_molecule(spectra_rows: list[dict], lib: Library, pool: CandidatePool) -> list[str]:
    """Per-spectrum: find library analogs (Phase 1's score_spectrum), then
    propagate to the external candidate pool by chemical similarity to
    those analogs. Max-pool the propagated scores across a molecule's
    spectra, then dedupe on the tautomer-canonical InChIKey14 the
    competition actually scores against, same as Phase 1.
    """
    best_score: dict[str, float] = {}
    best_smiles: dict[str, str] = {}

    for row in spectra_rows:
        anchors = score_spectrum(
            row["ms2_mzs"], row["ms2_normalized_intensities"], float(row["precursor_mz"]),
            str(row["ionization_mode"]).strip().lower(), lib,
        )
        qmass = neutral_mass(float(row["precursor_mz"]), row["adduct"])
        if qmass is None:
            continue

        for key, smiles, score in propagate(anchors, qmass, pool):
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


def predict_test_set(test_df: pd.DataFrame, lib: Library, pool: CandidatePool) -> dict[str, list[str]]:
    predictions = {}
    n_fallback = 0
    for mid, group in test_df.groupby("molecule_id"):
        guesses = predict_molecule(group.to_dict("records"), lib, pool)
        if not guesses:
            n_fallback += 1
            mode = str(group.iloc[0]["ionization_mode"]).strip().lower()
            guesses = _global_fallback_candidates(lib, mode, N_GUESSES)
        predictions[mid] = guesses

    if n_fallback:
        print(f"WARNING: {n_fallback}/{len(predictions)} molecules had zero propagated hits "
              f"and used the frequency fallback -- inspect these before trusting the score.")
    return predictions
