"""A peak-level transformer for spectrum -> Morgan fingerprint prediction.

The binned MLP (src/fingerprint_model.py) tops out around model-only
MRR@25 ~0.49 (natural products) / ~0.43 (timsTOF) and every training
variant lands in the same narrow band, which points at the 0.2 Da binned
input as the ceiling: it discards mass-defect information that separates
formulas, and can't relate pairs of fragments except through whatever the
first dense layer learns. This model instead treats a spectrum as a set of
peaks -- each embedded from its exact m/z (sinusoidal, as in the host's
tutorial notebook and MIST/DreaMS), its intensity, and its neutral loss
from the precursor -- and lets self-attention model fragment-pair
structure directly. Same 2048-bit target, same BCE loss, so it slots into
pipeline_v3 through the same predict_probs / log-likelihood ranking.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .candidates import FP_BITS
from .spectral_similarity import entropy_weight

MAX_PEAKS = 64
MAX_MZ = 1500.0


def prepare_peaks(mzs, intensities, precursor_mz: float, max_peaks: int = MAX_PEAKS):
    """One spectrum -> (mz[max_peaks], intensity[max_peaks], mask[max_peaks])
    keeping the max_peaks most intense peaks, entropy-weighted and L2
    normalized like every other stage of the pipeline.
    """
    mzs = np.asarray(mzs, dtype=np.float32)
    ints = entropy_weight(np.clip(np.asarray(intensities, dtype=np.float64), 0, None)).astype(np.float32)
    if len(mzs) > max_peaks:
        keep = np.argpartition(ints, -max_peaks)[-max_peaks:]
        mzs, ints = mzs[keep], ints[keep]
    norm = np.linalg.norm(ints)
    if norm > 0:
        ints = ints / norm
    n = len(mzs)
    out_mz = np.zeros(max_peaks, dtype=np.float32)
    out_int = np.zeros(max_peaks, dtype=np.float32)
    mask = np.zeros(max_peaks, dtype=bool)
    out_mz[:n], out_int[:n], mask[:n] = mzs, ints, True
    return out_mz, out_int, mask


class SinusoidalMass(nn.Module):
    """Sinusoidal embedding of a mass value over a wide range of
    wavelengths, so both the integer part and the mass defect are
    resolvable to the model.
    """

    def __init__(self, dim: int, min_wavelength: float = 0.001, max_wavelength: float = 2000.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(math.log(2 * math.pi / max_wavelength), math.log(2 * math.pi / min_wavelength), half))
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (..., )
        ang = x.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class PeakTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 8, dropout: float = 0.1, n_bits: int = FP_BITS):
        super().__init__()
        self.mass_embed = SinusoidalMass(d_model)
        self.loss_embed = SinusoidalMass(d_model)
        self.inten_proj = nn.Linear(1, d_model)
        self.peak_proj = nn.Linear(3 * d_model, d_model)
        self.precursor_proj = nn.Linear(d_model, d_model)
        self.mode_embed = nn.Embedding(3, d_model)  # 0 unknown, 1 positive, 2 negative
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, n_bits))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, mz: torch.Tensor, inten: torch.Tensor, mask: torch.Tensor, precursor: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        # mz/inten/mask: (B, K); precursor: (B,); mode: (B,) long
        loss = (precursor.unsqueeze(1) - mz).clamp(min=0)
        peaks = self.peak_proj(torch.cat([self.mass_embed(mz), self.loss_embed(loss), self.inten_proj(inten.unsqueeze(-1))], dim=-1))
        prec = (self.precursor_proj(self.mass_embed(precursor)) + self.mode_embed(mode)).unsqueeze(1)
        cls = self.cls.expand(mz.shape[0], -1, -1)
        x = torch.cat([cls, prec, peaks], dim=1)
        pad = torch.cat([torch.zeros(mz.shape[0], 2, dtype=torch.bool, device=mz.device), ~mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=pad)
        return self.head(self.norm(x[:, 0]))


def mode_id(ionization_mode) -> int:
    m = str(ionization_mode).strip().lower()
    return 1 if m == "positive" else 2 if m == "negative" else 0


@torch.no_grad()
def predict_probs_peaks(model: nn.Module, rows, device, batch_size: int = 512) -> np.ndarray:
    """rows: dicts with ms2_mzs, ms2_normalized_intensities, precursor_mz,
    ionization_mode. Returns (n_rows, FP_BITS) bit probabilities.
    """
    model.eval()
    rows = list(rows)
    out = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        arrs = [prepare_peaks(r["ms2_mzs"], r["ms2_normalized_intensities"], float(r["precursor_mz"])) for r in chunk]
        mz = torch.from_numpy(np.stack([a[0] for a in arrs])).to(device)
        it = torch.from_numpy(np.stack([a[1] for a in arrs])).to(device)
        mk = torch.from_numpy(np.stack([a[2] for a in arrs])).to(device)
        pr = torch.tensor([min(float(r["precursor_mz"]), MAX_MZ) for r in chunk], dtype=torch.float32, device=device)
        md = torch.tensor([mode_id(r["ionization_mode"]) for r in chunk], dtype=torch.long, device=device)
        out.append(torch.sigmoid(model(mz, it, mk, pr, md)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, FP_BITS), dtype=np.float32)
