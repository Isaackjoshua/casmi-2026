"""Phase 3 pipeline: Phase 2's analog propagation fused with the learned
spectrum -> fingerprint model (src/fingerprint_model.py).

Both signals score every candidate in the same tight mass window around
the query's measured neutral mass; propagation is at its best when a
close spectral analog exists in the library, the model is what carries
the cases where one doesn't (the domain-shift gap that capped Phase 2 at
~0.19 on the real leaderboard). Scores are min-max normalized within the
window and blended with weight ALPHA on propagation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .baseline import FINE_TOP_N, N_GUESSES, Library, _global_fallback_candidates, score_spectrum
from .candidates import FP_BITS
from .fingerprint_model import FingerprintEnsemble, FingerprintMLP, build_feature_matrix, fingerprint_loglik_scores, predict_probs
from .metric import to_inchikey14
from .peak_transformer import PeakTransformer, predict_probs_peaks
from .pipeline_v2 import MASS_WINDOW_WIDEN_CAP, MASS_WINDOW_WIDEN_FACTOR, merge_spectra
from .propagation import MASS_WINDOW_DA, PROPAGATION_EXPONENT, CandidatePool, mass_window, neutral_mass, propagation_scores

# Weight on propagation vs. the model, tuned on the 150-molecule hard
# validation set: propagation-only 0.4456, model-only 0.4836, fused at
# 0.3 -> 0.4932 (0.5 -> 0.4795, 0.15 -> 0.4873).
ALPHA = 0.3
MODEL_FLOOR = 0.05  # every mass-consistent candidate keeps some model credit, so none is dropped as "no evidence"


def _load_one(path: str, device):
    """Dispatch on checkpoint contents: the peak transformer saves its
    constructor config under "config", the binned MLP saves n_features/hidden.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" in ckpt:
        model = PeakTransformer(**ckpt["config"])
    else:
        model = FingerprintMLP(n_features=ckpt["n_features"], hidden=ckpt["hidden"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def load_fingerprint_model(path, device):
    """One checkpoint path -> that model; a list of paths -> an ensemble
    (models of either type may be mixed; predict_bit_probs averages their
    bit probabilities).
    """
    if isinstance(path, (list, tuple)):
        return FingerprintEnsemble([_load_one(p, device) for p in path]).to(device).eval()
    return _load_one(path, device)


def predict_bit_probs(model, rows: list[dict], device) -> np.ndarray:
    """(n_rows, FP_BITS) bit probabilities from any supported model: the
    binned MLP (sparse feature matrix), the peak transformer (raw peaks),
    or an ensemble of either, averaged at the probability level.
    """
    if isinstance(model, FingerprintEnsemble):
        return np.mean([predict_bit_probs(m, rows, device) for m in model.models], axis=0)
    if isinstance(model, PeakTransformer):
        return predict_probs_peaks(model, rows, device)
    return predict_probs(model, build_feature_matrix(rows), device)


def candidate_bits(pool: CandidatePool, lo: int, hi: int) -> np.ndarray:
    """(hi-lo, FP_BITS) 0/1 matrix for pool[lo:hi], unpacked from the
    uint64 word representation used for fast Tanimoto.
    """
    words = pool.fp_words[lo:hi]
    return np.unpackbits(words.view(np.uint8), axis=1)[:, :FP_BITS]


def _normalize(x: np.ndarray, degenerate_value: float) -> np.ndarray:
    """Min-max to [0, 1] within a window. At a ~1 mDa mass window many
    windows hold a single candidate (or several with identical scores);
    naive min-max maps those to 0 and silently drops the only
    mass-consistent structure -- degenerate_value says what an
    all-equal window should score instead.
    """
    lo, hi = x.min(), x.max()
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.full(len(x), degenerate_value, dtype=np.float64)


def predict_molecule(
    spectra_rows: list[dict],
    lib: Library,
    pool: CandidatePool,
    model: FingerprintMLP,
    device,
    mass_window_da: float = MASS_WINDOW_DA,
    exponent: float = PROPAGATION_EXPONENT,
    alpha: float = ALPHA,
) -> list[str]:
    best_score: dict[str, float] = {}
    best_smiles: dict[str, str] = {}

    by_adduct: dict[str, list[dict]] = {}
    for row in spectra_rows:
        by_adduct.setdefault(row["adduct"], []).append(row)

    for group in by_adduct.values():
        merged = merge_spectra(group) if len(group) > 1 else group[0]
        qmass = neutral_mass(float(merged["precursor_mz"]), merged["adduct"])
        if qmass is None:
            continue

        lo, hi = mass_window(pool, qmass, mass_window_da)
        widen = mass_window_da
        while hi <= lo and widen < MASS_WINDOW_WIDEN_CAP:
            widen *= MASS_WINDOW_WIDEN_FACTOR
            lo, hi = mass_window(pool, qmass, widen)
        if hi <= lo:
            continue

        fused = np.zeros(hi - lo, dtype=np.float64)

        if alpha > 0:
            anchors = score_spectrum(
                merged["ms2_mzs"], merged["ms2_normalized_intensities"], float(merged["precursor_mz"]),
                str(merged["ionization_mode"]).strip().lower(), lib,
            )
            if anchors:
                prop = propagation_scores(anchors, lo, hi, pool, exponent)
                # All-equal-and-nonzero (e.g. a lone candidate that IS the
                # anchor's neighbour) keeps full credit; all-zero means no
                # analog evidence at all and contributes nothing.
                fused += alpha * _normalize(prop, 1.0 if prop.max() > 0 else 0.0)

        if alpha < 1:
            # Predict per raw spectrum (the models trained on single spectra,
            # not merged ones) and average the bit probabilities.
            probs = predict_bit_probs(model, group, device).mean(axis=0)
            loglik = fingerprint_loglik_scores(probs, candidate_bits(pool, lo, hi))
            # Every candidate in the window is mass-consistent, so each has
            # real (if weak) evidence -- keep them all above zero.
            fused += (1 - alpha) * (MODEL_FLOOR + (1 - MODEL_FLOOR) * _normalize(loglik, 1.0))

        order = np.argsort(fused)[::-1][:FINE_TOP_N]
        for i in order:
            if fused[i] <= 0:
                break
            key = pool.inchikey14[lo + i]
            if fused[i] > best_score.get(key, 0.0):
                best_score[key] = float(fused[i])
                best_smiles[key] = pool.normalized_smiles[lo + i]

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
    model: FingerprintMLP,
    device,
    mass_window_da: float = MASS_WINDOW_DA,
    exponent: float = PROPAGATION_EXPONENT,
    alpha: float = ALPHA,
) -> dict[str, list[str]]:
    predictions = {}
    n_fallback = 0
    n_error = 0
    for mid, group in test_df.groupby("molecule_id"):
        try:
            guesses = predict_molecule(group.to_dict("records"), lib, pool, model, device, mass_window_da, exponent, alpha)
        except Exception as e:
            n_error += 1
            print(f"WARNING: prediction failed for {mid} ({type(e).__name__}: {e}); using frequency fallback")
            guesses = []

        if not guesses:
            n_fallback += 1
            try:
                mode = str(group.iloc[0]["ionization_mode"]).strip().lower()
                guesses = _global_fallback_candidates(lib, mode, N_GUESSES)
            except Exception:
                guesses = _global_fallback_candidates(lib, None, N_GUESSES)
        predictions[mid] = guesses

    if n_fallback:
        print(f"WARNING: {n_fallback}/{len(predictions)} molecules had zero candidates "
              f"and used the frequency fallback ({n_error} from errors).")
    return predictions
