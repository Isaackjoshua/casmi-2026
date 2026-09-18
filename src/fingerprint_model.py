"""Phase 3: a learned spectrum -> Morgan-fingerprint model.

Why: the propagation approach (src/propagation.py) plateaued at ~0.19 on
the real leaderboard despite ~0.49 on local validation. It depends on
finding a spectrally-similar *library analog* first, then hopping to its
chemical neighbours -- and the real test set is Enveda's own timsTOF
measurements of natural products, where a good library analog often just
isn't there (the instrument/chemistry domain shift the roadmap flagged
on day one). Predicting the fingerprint *directly* from the spectrum
removes that dependency: the model learns fragment -> substructure
associations from 2.5M labelled spectra and applies them to any
spectrum, analog or not. This is the "4th channel" every ~0.33+ public
notebook for this competition uses.

Input representation: entropy-weighted fragment intensities binned at
0.2 Da (up to MAX_MZ), concatenated with the same for *neutral losses*
(precursor m/z minus each fragment m/z) -- losses identify what
substructure came off, which is exactly what a fingerprint bit encodes.
Sparse (~80 non-zeros per spectrum), so the full training set fits in
memory as one CSR matrix.

Output: 2048 sigmoid logits, one per Morgan bit (radius 2, matching
src/candidates.py), trained with BCE. Candidates are then ranked by the
log-likelihood of their fingerprint under the predicted bit
probabilities, which reduces to a single matrix multiply.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .candidates import FP_BITS
from .spectral_similarity import entropy_weight

BIN_WIDTH = 0.2
MAX_MZ = 1200.0
N_FRAG_BINS = int(MAX_MZ / BIN_WIDTH) + 1
N_LOSS_BINS = N_FRAG_BINS
N_META = 3  # precursor_mz / MAX_MZ, is_positive, is_negative
N_FEATURES = N_FRAG_BINS + N_LOSS_BINS + N_META


def featurize(mzs, intensities, precursor_mz: float, ionization_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """One spectrum -> (column indices, values) for a sparse feature row."""
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = entropy_weight(np.clip(np.asarray(intensities, dtype=np.float64), 0, None))
    if len(mzs) == 0 or ints.sum() <= 0:
        cols = np.array([N_FRAG_BINS + N_LOSS_BINS], dtype=np.int64)
        vals = np.array([min(precursor_mz / MAX_MZ, 1.0)], dtype=np.float32)
        return cols, vals
    ints = ints / np.linalg.norm(ints)

    frag_cols = np.clip((mzs / BIN_WIDTH).astype(np.int64), 0, N_FRAG_BINS - 1)
    losses = precursor_mz - mzs
    keep = losses > 0
    loss_cols = N_FRAG_BINS + np.clip((losses[keep] / BIN_WIDTH).astype(np.int64), 0, N_LOSS_BINS - 1)

    meta_base = N_FRAG_BINS + N_LOSS_BINS
    mode = str(ionization_mode).strip().lower()
    meta_cols = np.array([meta_base, meta_base + 1, meta_base + 2], dtype=np.int64)
    meta_vals = np.array(
        [min(precursor_mz / MAX_MZ, 1.0), 1.0 if mode == "positive" else 0.0, 1.0 if mode == "negative" else 0.0],
        dtype=np.float32,
    )

    cols = np.concatenate([frag_cols, loss_cols, meta_cols])
    vals = np.concatenate([ints.astype(np.float32), ints[keep].astype(np.float32), meta_vals])
    return cols, vals


def build_feature_matrix(rows) -> sp.csr_matrix:
    """rows: iterable of dicts with ms2_mzs, ms2_normalized_intensities,
    precursor_mz, ionization_mode. Returns an (n x N_FEATURES) CSR matrix.
    Duplicate (row, col) entries -- two fragments in one bin -- are summed.
    """
    indptr = [0]
    all_cols, all_vals = [], []
    for r in rows:
        cols, vals = featurize(r["ms2_mzs"], r["ms2_normalized_intensities"], float(r["precursor_mz"]), r["ionization_mode"])
        all_cols.append(cols)
        all_vals.append(vals)
        indptr.append(indptr[-1] + len(cols))
    mat = sp.csr_matrix(
        (np.concatenate(all_vals), np.concatenate(all_cols), np.asarray(indptr, dtype=np.int64)),
        shape=(len(indptr) - 1, N_FEATURES),
    )
    mat.sum_duplicates()
    return mat


def unpack_fingerprint(packed: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:FP_BITS].astype(np.float32)


class FingerprintMLP(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 4096, n_bits: int = FP_BITS, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sparse_batch_to_tensor(mat: sp.csr_matrix, idx: np.ndarray, device) -> torch.Tensor:
    sub = mat[idx].tocoo()
    t = torch.sparse_coo_tensor(
        np.vstack([sub.row, sub.col]), sub.data, size=sub.shape, dtype=torch.float32
    )
    return t.to_dense().to(device)


@torch.no_grad()
def predict_probs(model: nn.Module, mat: sp.csr_matrix, device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, mat.shape[0], batch_size):
        idx = np.arange(start, min(start + batch_size, mat.shape[0]))
        x = sparse_batch_to_tensor(mat, idx, device)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, FP_BITS), dtype=np.float32)


def fingerprint_loglik_scores(probs: np.ndarray, candidate_bits: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """probs: (n_bits,) predicted P(bit=1). candidate_bits: (M, n_bits) in
    {0,1}. Returns per-candidate log-likelihood of each candidate's
    fingerprint under the predicted probabilities, up to a constant:
        sum_i [c_i log p_i + (1 - c_i) log(1 - p_i)]
      = C @ (log p - log(1-p)) + const
    so ranking only needs one matrix multiply against the logit vector.
    """
    p = np.clip(probs.astype(np.float64), eps, 1 - eps)
    logit = np.log(p) - np.log1p(-p)
    return candidate_bits.astype(np.float64) @ logit
