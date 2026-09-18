"""Build a valid submission.csv from per-molecule ranked SMILES lists."""

from pathlib import Path
from typing import Sequence

import pandas as pd


def build_submission(
    predictions: dict[str, Sequence[str]],
    sample_submission_path: Path | str | None = None,
) -> pd.DataFrame:
    """predictions: molecule_id -> ranked list of up to 25 SMILES guesses,
    best guess first. Validates the hard requirements the host enforces:
    every molecule_id present exactly once, no nulls, <= 25 guesses each.

    If sample_submission_path is given, also checks that the predicted
    molecule_id set matches it and prints a warning (never raises) on any
    mismatch. This is a coverage sanity check, not a hard requirement --
    sample_submission.csv may not track the actual test set exactly (e.g.
    during a competition's hidden-test rerun, where a stale or
    differently-sized reference file previously turned a coverage mismatch
    into an unhandled exception that failed the whole submission). The
    predictions dict, built directly from the real test set, is the
    authoritative source; this check only ever informs, never blocks.
    """
    rows = []
    for mid, smiles_list in predictions.items():
        if len(smiles_list) == 0:
            raise ValueError(f"{mid} has zero guesses; must have >= 1")
        if len(smiles_list) > 25:
            raise ValueError(f"{mid} has {len(smiles_list)} guesses; max is 25")
        if any(not s for s in smiles_list):
            raise ValueError(f"{mid} has an empty/null SMILES guess")
        rows.append({"molecule_id": mid, "smiles": ";".join(smiles_list)})

    df = pd.DataFrame(rows)
    if df["molecule_id"].duplicated().any():
        raise ValueError("duplicate molecule_id in predictions")

    if sample_submission_path is not None:
        try:
            expected = set(pd.read_csv(sample_submission_path)["molecule_id"])
            actual = set(df["molecule_id"])
            missing = expected - actual
            extra = actual - expected
            if missing:
                print(f"WARNING: {len(missing)} molecule_id(s) in sample_submission.csv missing from "
                      f"predictions, e.g. {sorted(missing)[:5]}")
            if extra:
                print(f"WARNING: {len(extra)} predicted molecule_id(s) not in sample_submission.csv, "
                      f"e.g. {sorted(extra)[:5]}")
        except Exception as e:
            print(f"WARNING: sample_submission.csv coverage check failed ({e}); skipping it, "
                  f"submission proceeds from predictions as-is")

    return df


def write_submission(
    predictions: dict[str, Sequence[str]],
    path: Path,
    sample_submission_path: Path | str | None = None,
) -> None:
    df = build_submission(predictions, sample_submission_path=sample_submission_path)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
