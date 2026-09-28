"""Build a valid submission.csv from per-molecule ranked SMILES lists."""

from pathlib import Path
from typing import Sequence

import pandas as pd


def build_submission(
    predictions: dict[str, Sequence[str]],
    sample_submission_path: Path | str | None = None,
    fallback: Sequence[str] | None = None,
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
    rows, patched = [], 0
    for mid, smiles_list in predictions.items():
        clean = [s for s in (smiles_list or []) if s]
        if not clean:
            # A molecule with no guesses used to raise here, which fails the
            # entire notebook over one row. That is the wrong trade in a code
            # competition: the run is scored on all the other molecules, and a
            # wrong guess costs only this molecule's reciprocal rank. It
            # happens for real -- an adduct outside ADDUCT_SPEC, or a
            # precursor mass with no candidate in any window.
            if not fallback:
                raise ValueError(f"{mid} has zero guesses and no fallback was provided")
            clean = list(fallback)
            patched += 1
        rows.append({"molecule_id": mid, "smiles": ";".join(clean[:25])})
    if patched:
        print(f"WARNING: {patched} molecule(s) had no guesses and were filled with the "
              f"library-frequency fallback")

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
                # Add them rather than only reporting them: a molecule the host
                # expects and we omit is scored as wrong anyway, but an
                # incomplete file can be rejected outright.
                if fallback:
                    df = pd.concat([df, pd.DataFrame(
                        [{"molecule_id": m, "smiles": ";".join(list(fallback)[:25])}
                         for m in sorted(missing)])], ignore_index=True)
                    print(f"  filled them with the library-frequency fallback")
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
    fallback: Sequence[str] | None = None,
) -> None:
    df = build_submission(predictions, sample_submission_path=sample_submission_path,
                          fallback=fallback)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
