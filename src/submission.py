"""Build a valid submission.csv from per-molecule ranked SMILES lists."""

from pathlib import Path
from typing import Sequence

import pandas as pd


def build_submission(predictions: dict[str, Sequence[str]]) -> pd.DataFrame:
    """predictions: molecule_id -> ranked list of up to 25 SMILES guesses,
    best guess first. Validates the hard requirements the host enforces:
    every molecule_id present exactly once, no nulls, <= 25 guesses each.
    """
    rows = []
    for mid, smiles_list in predictions.items():
        if len(smiles_list) == 0:
            raise ValueError(f"{mid} has zero guesses; must have >= 1")
        if len(smiles_list) > 25:
            raise ValueError(f"{mid} has {len(smiles_list)} guesses; max is 25")
        rows.append({"molecule_id": mid, "smiles": ";".join(smiles_list)})

    df = pd.DataFrame(rows)
    if df["molecule_id"].duplicated().any():
        raise ValueError("duplicate molecule_id in predictions")
    return df


def write_submission(predictions: dict[str, Sequence[str]], path: Path) -> None:
    df = build_submission(predictions)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
