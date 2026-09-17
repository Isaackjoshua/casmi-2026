"""Phase 1 baseline: coarse vectorized spectral-similarity screening
against the training library, exact modified-cosine rescoring of the
survivors, aggregated per test molecule.

Rewritten after a correctness/scale review of the first version (see git
history) found: the original tight-vs-wide precursor window was dead code
(tight was always a subset of wide, so it never affected ranking), the
per-candidate Python-loop scoring would take ~20+ hours against the real
2.5M-row library (the Kaggle runtime cap is 9h), the no-hit fallback
returned a near-arbitrary structure, and dedup used the raw dataset
inchikey14 instead of the tautomer-canonical key the competition actually
scores against. This version:

- Deduplicates the library to one representative spectrum per
  (inchikey14, adduct) before doing anything else -- cuts ~2.5M rows to
  the low hundreds of thousands, since compounds are measured repeatedly
  across the 11 source libraries.
- Truncates every spectrum (library and query) to its top-N most intense
  peaks, bounding the cost of exact peak matching.
- Screens the whole (deduplicated, truncated) library with one sparse
  matrix-vector product per query spectrum (see spectral_similarity.
  build_binned_index), then only rescoring the top COARSE_TOP_K survivors
  with the exact modified-cosine peak matcher. This replaces the old
  precursor-mass windowing entirely.
- Adds a small score bonus for candidates within a tight ppm window of the
  query's precursor mass, so exact/near-exact mass matches (Class 1) are
  still preferred over same-score analogs when the coarse+fine scores tie.
- Deduplicates the final ranked guesses on the tautomer-canonical
  InChIKey14 (src.metric.to_inchikey14), matching the competition's own
  matching rule, instead of the raw dataset key.
- Falls back to the most common library structures (globally, or within
  the query's ionization mode) rather than an arbitrary single row when a
  spectrum gets no similarity hits at all.

Usage (once data/raw/{train,test}.parquet exist):

    from src.data import load_train, load_test
    from src.baseline import build_library, predict_test_set
    from src.submission import write_submission

    train = load_train()
    test = load_test()

    lib_sources = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]
    library = build_library(train[train["ingest_lib"].isin(lib_sources)])

    predictions = predict_test_set(test, library)
    write_submission(predictions, "submissions/baseline_v1.csv", sample_submission_path="data/raw/sample_submission.csv")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metric import to_inchikey14
from .spectral_similarity import build_binned_index, bin_query_vector, modified_cosine_similarity

PEAK_TOP_K = 40  # peaks kept per spectrum (library at build time, query at score time)
BIN_WIDTH = 0.5  # Da, coarse pre-filter bin size
MAX_MZ = 2100.0  # covers precursor + 2 Da headroom for the heaviest test molecules (~1159 Da x[M+K]+)
COARSE_TOP_K = 300  # candidates kept after the cheap sparse screen, per query spectrum
FINE_TOP_N = 50  # top candidates (raw dedup key) kept per spectrum after exact rescoring
N_GUESSES = 25
PRECURSOR_PPM_BONUS_WINDOW = 30  # ppm; candidates this close to the query's precursor get a small bump
PRECURSOR_BONUS = 0.05
N_BINS = int(MAX_MZ / BIN_WIDTH) + 1


def _top_k_peaks(mzs: np.ndarray, intensities: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(mzs) <= k:
        return mzs, intensities
    keep = np.argpartition(intensities, -k)[-k:]
    keep = keep[np.argsort(mzs[keep])]
    return mzs[keep], intensities[keep]


@dataclass
class Library:
    """Deduplicated, peak-truncated training spectra in CSR-style flat
    arrays (one big mzs/intensity array + an offsets index), which is far
    cheaper in memory and faster to build a sparse index from than a
    Python list of small per-spectrum arrays.
    """

    offsets: np.ndarray  # length n+1; spectrum i's peaks are [offsets[i]:offsets[i+1]]
    mzs_flat: np.ndarray
    ints_flat: np.ndarray
    precursor_mz: np.ndarray
    ionization_mode: np.ndarray
    inchikey14: np.ndarray
    normalized_smiles: np.ndarray
    coarse_index: object  # scipy.sparse.csr_matrix, (n_spectra x N_BINS)

    def __len__(self) -> int:
        return len(self.precursor_mz)

    def peaks(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.offsets[i], self.offsets[i + 1]
        return self.mzs_flat[s:e], self.ints_flat[s:e]


def build_library(train_df: pd.DataFrame) -> Library:
    required = ["precursor_mz", "inchikey14", "normalized_smiles", "adduct", "ionization_mode",
                "ms2_mzs", "ms2_normalized_intensities"]
    df = train_df.dropna(subset=required).copy()
    df["ionization_mode"] = df["ionization_mode"].str.strip().str.lower()
    df = df[df["ms2_mzs"].map(len) > 0]

    # One representative spectrum per (structure, adduct): keep the one with
    # the most peaks, a cheap proxy for spectral quality/information content.
    df["_num_peaks"] = df["ms2_mzs"].map(len)
    df = df.sort_values("_num_peaks", ascending=False)
    df = df.drop_duplicates(subset=["inchikey14", "adduct"], keep="first")
    df = df.sort_values("precursor_mz").reset_index(drop=True)

    mzs_list, ints_list, lengths = [], [], []
    for mzs, ints in zip(df["ms2_mzs"], df["ms2_normalized_intensities"]):
        mzs_arr = np.asarray(mzs, dtype=np.float64)
        ints_arr = np.asarray(ints, dtype=np.float64)
        mzs_arr, ints_arr = _top_k_peaks(mzs_arr, ints_arr, PEAK_TOP_K)
        mzs_list.append(mzs_arr)
        ints_list.append(ints_arr)
        lengths.append(len(mzs_arr))

    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    mzs_flat = np.concatenate(mzs_list) if mzs_list else np.array([], dtype=np.float64)
    ints_flat = np.concatenate(ints_list) if ints_list else np.array([], dtype=np.float64)

    coarse_index = build_binned_index(offsets, mzs_flat, ints_flat, BIN_WIDTH, N_BINS)

    return Library(
        offsets=offsets,
        mzs_flat=mzs_flat,
        ints_flat=ints_flat,
        precursor_mz=df["precursor_mz"].to_numpy(),
        ionization_mode=df["ionization_mode"].to_numpy(),
        inchikey14=df["inchikey14"].to_numpy(),
        normalized_smiles=df["normalized_smiles"].to_numpy(),
        coarse_index=coarse_index,
    )


def _global_fallback_candidates(lib: Library, ionization_mode: str | None, n: int) -> list[str]:
    """Most frequent structures in the library, optionally restricted to
    one ionization mode -- used only when a spectrum gets zero similarity
    hits, so the guess is at least a plausible common natural product
    rather than an arbitrary row.
    """
    mask = lib.ionization_mode == ionization_mode if ionization_mode is not None else np.ones(len(lib), dtype=bool)
    keys, smiles = lib.inchikey14[mask], lib.normalized_smiles[mask]
    if len(keys) == 0:
        keys, smiles = lib.inchikey14, lib.normalized_smiles
    _, first_idx, counts = np.unique(keys, return_index=True, return_counts=True)
    order = np.argsort(counts)[::-1][:n]
    return [smiles[first_idx[i]] for i in order]


def score_spectrum(
    query_mzs: np.ndarray,
    query_intensities: np.ndarray,
    query_precursor: float,
    query_ionization_mode: str,
    lib: Library,
) -> list[tuple[str, str, float]]:
    """Return (inchikey14, smiles, score) for the best-matching library
    spectra to one query spectrum, best first: a cheap sparse coarse
    screen over the whole library, exact modified-cosine rescoring of the
    survivors, with a small bonus for a close precursor-mass match.
    """
    query_mzs, query_intensities = _top_k_peaks(
        np.asarray(query_mzs, dtype=np.float64), np.asarray(query_intensities, dtype=np.float64), PEAK_TOP_K
    )
    if len(query_mzs) == 0:
        return []

    query_vec = bin_query_vector(query_mzs, query_intensities, BIN_WIDTH, N_BINS)
    coarse_scores = np.asarray(lib.coarse_index.dot(query_vec)).ravel()

    mode_mask = lib.ionization_mode == query_ionization_mode
    coarse_scores = np.where(mode_mask, coarse_scores, -1.0)

    k = min(COARSE_TOP_K, np.count_nonzero(coarse_scores > 0))
    if k == 0:
        return []
    shortlist = np.argpartition(coarse_scores, -k)[-k:]

    ppm_bonus_da = query_precursor * PRECURSOR_PPM_BONUS_WINDOW / 1e6
    scored = []
    for i in shortlist:
        lib_mzs, lib_ints = lib.peaks(i)
        sim = modified_cosine_similarity(
            query_mzs, query_intensities, query_precursor, lib_mzs, lib_ints, lib.precursor_mz[i]
        )
        if abs(query_precursor - lib.precursor_mz[i]) <= ppm_bonus_da:
            sim += PRECURSOR_BONUS
        if sim > 0:
            scored.append((lib.inchikey14[i], lib.normalized_smiles[i], sim))

    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:FINE_TOP_N]


def predict_molecule(spectra_rows: list[dict], lib: Library) -> list[str]:
    """Aggregate evidence across all of a molecule's spectra (different
    collision energies / adducts) into one ranked SMILES list: max-pool
    the raw-key similarity score across spectra, then collapse to the
    competition's actual matching key (tautomer-canonical InChIKey14) so
    two raw candidates that are really the same scored structure don't
    both consume a guess slot.
    """
    best_score: dict[str, float] = {}
    best_smiles: dict[str, str] = {}

    for row in spectra_rows:
        hits = score_spectrum(
            row["ms2_mzs"], row["ms2_normalized_intensities"], float(row["precursor_mz"]),
            str(row["ionization_mode"]).strip().lower(), lib,
        )
        for key, smiles, sim in hits:
            if sim > best_score.get(key, 0.0):
                best_score[key] = sim
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


def predict_test_set(test_df: pd.DataFrame, lib: Library) -> dict[str, list[str]]:
    predictions = {}
    n_fallback = 0
    for mid, group in test_df.groupby("molecule_id"):
        guesses = predict_molecule(group.to_dict("records"), lib)
        if not guesses:
            n_fallback += 1
            mode = str(group.iloc[0]["ionization_mode"]).strip().lower()
            guesses = _global_fallback_candidates(lib, mode, N_GUESSES)
        predictions[mid] = guesses

    if n_fallback:
        print(f"WARNING: {n_fallback}/{len(predictions)} molecules had zero similarity hits "
              f"and used the frequency fallback -- inspect these before trusting the score.")
    return predictions
