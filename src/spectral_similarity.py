"""Spectrum-to-spectrum similarity: plain cosine and "modified" cosine
(which also matches peaks shifted by the precursor mass difference, so it
can find structural analogs whose fragments shift together with an added
or removed substituent -- this is what the public "analog propagation"
baselines are built on).

Pure numpy, no external spectral library dependency, so it works before
matchms finishes installing.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def entropy_weight(intensities: np.ndarray) -> np.ndarray:
    """Reweight intensities per Li et al. 2021: plain cosine similarity is
    dominated by whichever single peak has the highest intensity product,
    so two chemically unrelated spectra that happen to share one strong
    generic fragment (a water loss, a common small ion) can score
    deceptively high regardless of how the rest of the spectrum -- and the
    underlying structure -- differs. Diagnosed directly in this pipeline:
    a held-out validation found "0.99 similar" anchors whose true Tanimoto
    chemical similarity to the query was ~0.1, and every top public
    notebook for this competition uses this exact reweighting instead of
    raw cosine.

    Low-entropy (peaky, one-or-few-peaks-dominant) spectra get compressed
    by intensity ** w with w = 0.25 + 0.25*S (S = Shannon entropy of the
    normalized intensities), which pulls the dominant peak's influence
    down and lets the rest of the spectrum's shape matter; spectra that
    are already reasonably flat (S >= 3) are left alone.
    """
    intensities = np.asarray(intensities, dtype=np.float64)
    total = intensities.sum()
    if total <= 0:
        return intensities
    p = intensities / total
    nonzero = p > 0
    entropy = -np.sum(p[nonzero] * np.log(p[nonzero]))
    if entropy >= 3.0:
        return intensities
    w = 0.25 + 0.25 * entropy
    return intensities**w


def _match_peaks(mzs_a: np.ndarray, mzs_b: np.ndarray, tol_da: float) -> list[tuple[int, int]]:
    """Greedy one-to-one peak matching within +/- tol_da, highest combined
    intensity pairs matched first. O(n log n + n*k) for k = avg matches per peak.
    """
    pairs = []
    for i, mz_a in enumerate(mzs_a):
        lo = np.searchsorted(mzs_b, mz_a - tol_da, side="left")
        hi = np.searchsorted(mzs_b, mz_a + tol_da, side="right")
        for j in range(lo, hi):
            pairs.append((i, j))
    return pairs


def cosine_similarity(
    mzs_a: np.ndarray,
    ints_a: np.ndarray,
    mzs_b: np.ndarray,
    ints_b: np.ndarray,
    tol_da: float = 0.02,
) -> float:
    """Direct peak-to-peak cosine similarity, greedy-matched by m/z tolerance.
    Assumes mzs_a / mzs_b are sorted ascending (true for the competition arrays).
    """
    if len(mzs_a) == 0 or len(mzs_b) == 0:
        return 0.0

    ints_a, ints_b = entropy_weight(ints_a), entropy_weight(ints_b)
    candidates = _match_peaks(mzs_a, mzs_b, tol_da)
    if not candidates:
        return 0.0

    candidates.sort(key=lambda p: ints_a[p[0]] * ints_b[p[1]], reverse=True)
    used_a, used_b = set(), set()
    numerator = 0.0
    for i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        numerator += ints_a[i] * ints_b[j]

    denom = np.sqrt(np.sum(ints_a**2)) * np.sqrt(np.sum(ints_b**2))
    return float(numerator / denom) if denom > 0 else 0.0


# --- Vectorized coarse pre-filter -------------------------------------------
#
# Scoring every library spectrum with the exact peak-matching functions above
# is too slow at real scale (2.5M library spectra x 1000s of test candidates).
# These helpers build a cheap, approximate, binned-cosine index with scipy
# sparse matrices so a whole library can be screened with one sparse
# matrix-vector product per query, and only the top few hundred survivors go
# through the expensive exact modified_cosine_similarity above.


def bin_query_vector(mzs: np.ndarray, intensities: np.ndarray, bin_width: float, n_bins: int) -> np.ndarray:
    """Bin one spectrum into a dense, L2-normalized vector for the coarse
    prefilter. Entropy-weighted (see entropy_weight) so the coarse screen
    uses the same damping of dominant-peak dominance as the exact rescoring
    stage -- otherwise a candidate the fine scorer would reject for exactly
    that reason could still win the coarse screen and crowd out a real one.
    """
    cols = np.clip((np.asarray(mzs) / bin_width).astype(np.int64), 0, n_bins - 1)
    vals = entropy_weight(np.clip(np.asarray(intensities, dtype=np.float64), 0, None))
    vec = np.zeros(n_bins, dtype=np.float32)
    np.add.at(vec, cols, vals)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _segment_entropy_weight(offsets: np.ndarray, ints_flat: np.ndarray) -> np.ndarray:
    """entropy_weight applied per-spectrum to a CSR-style flattened peak
    array -- each spectrum's entropy is computed over its own peaks only,
    not the whole flattened array.
    """
    row_idx = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    seg_sum = np.add.reduceat(ints_flat, offsets[:-1])
    seg_sum = np.where(seg_sum > 0, seg_sum, 1.0)

    p = ints_flat / seg_sum[row_idx]
    plogp = np.where(p > 0, -p * np.log(p), 0.0)
    seg_entropy = np.add.reduceat(plogp, offsets[:-1])

    w = np.where(seg_entropy >= 3.0, 1.0, 0.25 + 0.25 * seg_entropy)
    return ints_flat ** w[row_idx]


def build_binned_index(
    offsets: np.ndarray,
    mzs_flat: np.ndarray,
    ints_flat: np.ndarray,
    bin_width: float,
    n_bins: int,
) -> sp.csr_matrix:
    """Build a (n_spectra x n_bins) sparse, row-normalized matrix from a
    CSR-style flattened spectra collection (see baseline.Library), so a
    query's coarse similarity to every library spectrum is one sparse dot
    product: `index.dot(query_vector)`.
    """
    n_spectra = len(offsets) - 1
    row = np.repeat(np.arange(n_spectra), np.diff(offsets))
    col = np.clip((mzs_flat / bin_width).astype(np.int64), 0, n_bins - 1)
    val = _segment_entropy_weight(offsets, np.clip(ints_flat, 0, None)).astype(np.float32)

    mat = sp.coo_matrix((val, (row, col)), shape=(n_spectra, n_bins)).tocsr()
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ mat


def modified_cosine_similarity(
    mzs_a: np.ndarray,
    ints_a: np.ndarray,
    precursor_a: float,
    mzs_b: np.ndarray,
    ints_b: np.ndarray,
    precursor_b: float,
    tol_da: float = 0.02,
) -> float:
    """Cosine similarity that also matches peaks shifted by the precursor
    mass difference (precursor_a - precursor_b), so a fragment shared between
    two analogs -- one carrying an extra substituent -- still counts as a
    match even though its raw m/z differs. Falls back to plain cosine when
    the precursors are equal.
    """
    if len(mzs_a) == 0 or len(mzs_b) == 0:
        return 0.0

    ints_a, ints_b = entropy_weight(ints_a), entropy_weight(ints_b)
    shift = precursor_a - precursor_b
    direct = _match_peaks(mzs_a, mzs_b, tol_da)
    shifted = _match_peaks(mzs_a, mzs_b + shift, tol_da) if abs(shift) > tol_da else []

    candidates = list({(i, j) for i, j in direct + shifted})
    if not candidates:
        return 0.0

    candidates.sort(key=lambda p: ints_a[p[0]] * ints_b[p[1]], reverse=True)
    used_a, used_b = set(), set()
    numerator = 0.0
    for i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        numerator += ints_a[i] * ints_b[j]

    denom = np.sqrt(np.sum(ints_a**2)) * np.sqrt(np.sum(ints_b**2))
    return float(numerator / denom) if denom > 0 else 0.0
