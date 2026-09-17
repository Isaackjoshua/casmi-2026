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
