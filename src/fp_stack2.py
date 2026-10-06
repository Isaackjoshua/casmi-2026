"""A wider fingerprint stack: six descriptor families instead of three.

src/multi_fingerprint.py (Morgan r2 + Morgan r3 + MACCS, 3239 bits) is
deployed and scored 0.245, up from 0.240. Ablating its two orthogonal
blocks on 2500 held-out spectraverse structures says what to do next:

  MACCS alone      +0.0002 +/- 0.0042   nothing
  Morgan r3 alone  +0.0042 +/- 0.0069
  both             +0.0086 +/- 0.0070   significant

The pair beats the sum of its parts, and the block that contributes
nothing on its own is half of why the other one pays. That is complementary
information -- one axis only earns its keep once another is also pinned
down -- so adding more families should keep paying, and a family being
useless alone is not a reason to leave it out.

Note also which block carried it: MACCS is the *best* predicted block
(Tanimoto 0.72 against Morgan r2's 0.37) and contributes nothing by
itself, while Morgan r3 is the worst predicted (0.31) and carries the
signal. Predictability is not usefulness. Coarse functional-group keys are
easy to read off a spectrum precisely because they do not separate isomers
sharing a mass window. So the three additions here are all fine-grained
topology, and deliberately sparse:

  atompair   2048  atom pairs with topological distance -- distance-based,
                   not circular, so the most orthogonal family available
  torsion    2048  four-atom torsion paths
  fcfp       2048  Morgan r2 over pharmacophoric features rather than atom
                   identities, so it generalizes across substitutions

RDKit's path fingerprint was tried and left out: at 291 on-bits of 2048 it
is by far the densest option and the most MACCS-like of the candidates,
and it is the slowest to compute, which matters because the Kaggle kernel
fingerprints the whole pool at runtime.

This is a separate module rather than a change to multi_fingerprint.py so
the deployed 0.245 path keeps working untouched.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from .fingerprint_model import fingerprint_loglik_scores

# (name, width); order fixes the layout and must never be reordered, since
# the pool arrays and the model's output head are both indexed by it
BLOCKS = [
    ("morgan2", 2048),
    ("morgan3", 1024),
    ("maccs", 167),
    ("atompair", 2048),
    ("torsion", 2048),
    ("fcfp", 2048),
]
TOTAL_BITS = sum(w for _n, w in BLOCKS)  # 9383
PACKED_BYTES = (TOTAL_BITS + 7) // 8  # 1173

OFFSETS = {}
_o = 0
for _name, _w in BLOCKS:
    OFFSETS[_name] = (_o, _o + _w)
    _o += _w

# the first three blocks are bit-for-bit the v1 stack, so a v1 pool array or
# model output can be read straight out of a v2 one
V1_BITS = OFFSETS["maccs"][1]  # 3239

_GENS = None


def _generators():
    global _GENS
    if _GENS is None:
        _GENS = {
            "morgan2": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
            "morgan3": rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024),
            "atompair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048),
            "torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048),
            "fcfp": rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=2048,
                atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()),
        }
    return _GENS


def bits_from_mol(mol) -> np.ndarray:
    """(TOTAL_BITS,) uint8 of 0/1 for one parsed molecule."""
    gens = _generators()
    out = np.zeros(TOTAL_BITS, dtype=np.uint8)
    for name, width in BLOCKS:
        fp = MACCSkeys.GenMACCSKeys(mol) if name == "maccs" else gens[name].GetFingerprint(mol)
        lo, _hi = OFFSETS[name]
        idx = np.fromiter(fp.GetOnBits(), dtype=np.int64)
        if len(idx):
            out[lo + idx[idx < width]] = 1
    return out


def packed_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.packbits(bits_from_mol(mol)).tobytes()


def unpack(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.ndim == 1:
        packed = packed[None, :]
    return np.unpackbits(packed, axis=1)[:, :TOTAL_BITS]


def _packed_one(smiles):
    try:
        return packed_from_smiles(smiles)
    except Exception:
        return None


def packed_many(smiles_list, workers: int = 4, chunk: int = 2000):
    """(n, PACKED_BYTES) uint8 for a list of SMILES, in the order given.

    A row of zeros for anything unparseable: such a candidate contributes
    nothing to its own score rather than failing the run.
    """
    from multiprocessing import Pool

    smiles_list = list(smiles_list)
    out = np.zeros((len(smiles_list), PACKED_BYTES), dtype=np.uint8)
    ok = np.zeros(len(smiles_list), dtype=bool)
    with Pool(workers) as pool:
        for i, packed in enumerate(pool.imap(_packed_one, smiles_list, chunksize=chunk)):
            if packed is not None:
                out[i] = np.frombuffer(packed, dtype=np.uint8)
                ok[i] = True
    return out, ok


def _z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def block_weighted_loglik(morgan2_probs, multi_probs, cand_packed, weight: float,
                          blocks=None):
    """Morgan r2 from one model, the orthogonal families from another.

    Each block is standardized within the mass window before combining, so
    `weight` sets the balance rather than whichever block happens to have the
    widest spread -- without that, MACCS alone swamps Morgan r2 and the
    combination scores worse than Morgan r2 by itself. weight = 0 reproduces
    Morgan-r2-only scoring.

    `blocks` restricts which orthogonal families are used, for ablation;
    None means all of them.
    """
    bits = unpack(np.asarray(cand_packed))
    lo2, hi2 = OFFSETS["morgan2"]
    score = _z(fingerprint_loglik_scores(morgan2_probs, bits[:, lo2:hi2]))
    if weight <= 0:
        return score
    names = [n for n, _w in BLOCKS if n != "morgan2"] if blocks is None else list(blocks)
    multi_probs = np.asarray(multi_probs)
    # one standardized term per family, so a family's contribution does not
    # depend on how many bits it happens to have
    for name in names:
        lo, hi = OFFSETS[name]
        score = score + (weight / len(names)) * _z(
            fingerprint_loglik_scores(multi_probs[lo:hi], bits[:, lo:hi]))
    return score
