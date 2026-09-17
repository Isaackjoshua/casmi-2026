"""Local implementation of the competition metric: MRR@25 via InChIKey14
matching after RDKit tautomer canonicalization.

This mirrors the official evaluation described on the competition's
Overview > Evaluation page, so it can be used for offline validation before
submitting. It is a best-effort reimplementation, not the host's exact code
--- always sanity-check against real leaderboard feedback once you have any.
"""

from functools import lru_cache
from typing import Sequence

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_ENUMERATOR = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=200_000)
def to_inchikey14(smiles: str) -> str | None:
    """Canonicalize a tautomer and reduce to the first InChIKey block
    (the part before the first hyphen), which encodes 2D connectivity only.
    Returns None if the SMILES can't be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = _ENUMERATOR.Canonicalize(mol)
    inchikey = Chem.MolToInchiKey(canon)
    if not inchikey:
        return None
    return inchikey.split("-")[0]


def reciprocal_rank(predicted_smiles: Sequence[str], true_smiles: str) -> float:
    """Score one molecule: 1/rank of the first predicted SMILES whose
    InChIKey14 matches the answer's, 0 if none match within the first 25.
    """
    target_key = to_inchikey14(true_smiles)
    if target_key is None:
        raise ValueError(f"Could not parse ground-truth SMILES: {true_smiles!r}")

    for rank, smiles in enumerate(predicted_smiles[:25], start=1):
        if to_inchikey14(smiles) == target_key:
            return 1.0 / rank
    return 0.0


def mrr_at_25(predictions: dict[str, Sequence[str]], answers: dict[str, str]) -> float:
    """predictions: molecule_id -> ranked list of up to 25 SMILES guesses.
    answers: molecule_id -> ground-truth SMILES.
    Returns the mean reciprocal rank across all molecules in `answers`.
    """
    scores = [
        reciprocal_rank(predictions.get(mid, []), true_smiles)
        for mid, true_smiles in answers.items()
    ]
    return sum(scores) / len(scores)
