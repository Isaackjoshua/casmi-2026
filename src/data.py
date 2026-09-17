"""Loading helpers for the CASMI 2026 train/test parquet files."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_train(path: Path = RAW_DIR / "train.parquet") -> pd.DataFrame:
    """One row per training spectrum. Columns include normalized_smiles,
    inchikey / inchikey14, ms2_mzs, ms2_normalized_intensities, adduct,
    collision_energy_ev, ingest_lib, precursor_error_ppm, etc.
    """
    return pd.read_parquet(path)


def load_test(path: Path = RAW_DIR / "test.parquet") -> pd.DataFrame:
    """One row per test spectrum, no structure label. Group by molecule_id
    to get all spectra for a given molecule before predicting.
    """
    return pd.read_parquet(path)


def load_sample_submission(path: Path = RAW_DIR / "sample_submission.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def spectra_by_molecule(df: pd.DataFrame) -> dict:
    """Group a spectra dataframe by molecule_id -> list of row dicts.

    Test molecules can have 1-16 spectra (median 3) at different collision
    energies / adducts; predictions are made per molecule, so callers need
    all spectra for a molecule together.
    """
    return {mid: g.to_dict("records") for mid, g in df.groupby("molecule_id")}
