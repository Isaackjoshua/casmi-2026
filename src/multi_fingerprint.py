"""Several orthogonal fingerprints per structure, not one.

Every model in this project predicts the same target -- 2048-bit Morgan
radius 2 -- under the same per-bit BCE loss, and the family is measurably
saturated: eleven checkpoints from a 62M-param MLP to a 4.6M-param
transformer span 5% of MRR, ensembling them does not help, and across
those real checkpoints +0.01 fingerprint Tanimoto buys only +0.0031 MRR
(scripts/eval_tanimoto_vs_mrr.py). Training a better Morgan predictor is
therefore not worth the GPU time: the relationship is convex, and every
model sits in its flat part.

What that measurement does *not* rule out is giving the ranker more views
of each candidate. Ranking scores a candidate as `bits @ logit`, which is
a sum over bits, so independent fingerprints simply extend the sum. Two
predictions at Tanimoto 0.3 that make *different* mistakes constrain a
candidate more than one at 0.35 -- and the whole difficulty is telling
near-isomers apart, where different descriptor families disagree most.

The stack, chosen for independence rather than size:
  morgan2  2048 bits  circular, radius 2 -- the existing target
  morgan3  1024 bits  circular, radius 3 -- larger environments, so it
                      separates structures that agree out to radius 2
  maccs     167 bits  curated substructure keys -- hand-designed
                      functional-group questions, not hashed environments
                      (bit 0 is unused by definition, kept for alignment)

Concatenated in that order into one 3239-bit vector, packed big-endian to
405 bytes, so scoring stays a single matrix multiply.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

MORGAN2_BITS = 2048
MORGAN3_BITS = 1024
MACCS_BITS = 167
TOTAL_BITS = MORGAN2_BITS + MORGAN3_BITS + MACCS_BITS  # 3239
PACKED_BYTES = (TOTAL_BITS + 7) // 8  # 405

# offsets, so a caller can score or inspect one block on its own
OFFSETS = {
    "morgan2": (0, MORGAN2_BITS),
    "morgan3": (MORGAN2_BITS, MORGAN2_BITS + MORGAN3_BITS),
    "maccs": (MORGAN2_BITS + MORGAN3_BITS, TOTAL_BITS),
}

_GEN2 = None
_GEN3 = None


def _generators():
    global _GEN2, _GEN3
    if _GEN2 is None:
        _GEN2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN2_BITS)
        _GEN3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=MORGAN3_BITS)
    return _GEN2, _GEN3


def bits_from_mol(mol) -> np.ndarray:
    """(TOTAL_BITS,) uint8 of 0/1 for one parsed molecule."""
    gen2, gen3 = _generators()
    out = np.zeros(TOTAL_BITS, dtype=np.uint8)
    for name, fp in (("morgan2", gen2.GetFingerprint(mol)),
                     ("morgan3", gen3.GetFingerprint(mol)),
                     ("maccs", MACCSkeys.GenMACCSKeys(mol))):
        lo, hi = OFFSETS[name]
        idx = np.fromiter(fp.GetOnBits(), dtype=np.int64)
        if len(idx):
            # MACCS returns 167 bits indexed 0..166; all three fit their block
            out[lo + idx[idx < (hi - lo)]] = 1
    return out


def packed_from_smiles(smiles: str):
    """SMILES -> PACKED_BYTES of packed bits, or None if RDKit cannot parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.packbits(bits_from_mol(mol)).tobytes()


def unpack(packed: np.ndarray) -> np.ndarray:
    """(n, PACKED_BYTES) uint8 -> (n, TOTAL_BITS) uint8 of 0/1."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.ndim == 1:
        packed = packed[None, :]
    return np.unpackbits(packed, axis=1)[:, :TOTAL_BITS]
