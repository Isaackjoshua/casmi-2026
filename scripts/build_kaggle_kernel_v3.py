"""Assembles the Phase 3 pipeline (analog propagation fused with the
learned spectrum->fingerprint model) as a self-contained Kaggle Notebook.
See build_kaggle_kernel.py for why it has to be a Notebook, and
build_kaggle_kernel_v2.py for the dataset-mount and rdkit-wheel details
this inherits.

The trained model (data/processed/fp_model.pt, ~236 MB) is shipped as a
private Kaggle dataset (kaggle_model_dataset/, `kaggle datasets create`)
and attached like the COCONUT and rdkit-wheel datasets; torch itself is
preinstalled on Kaggle's image. Inference is a ~62M-param MLP over ~1.2k
spectra, so the CPU kernel is fine -- no GPU needed.

Run: python3 scripts/build_kaggle_kernel_v3.py
Then: cd kaggle_kernel_v3 && kaggle kernels push -p .
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle_kernel_v3"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
KAGGLE_USERNAME = "isaackjoshua"
KERNEL_SLUG = "casmi26-phase-3-fingerprint-model"
KERNEL_TITLE = "CASMI26 Phase 3 - Fingerprint Model"
RDKIT_WHEEL_DATASET = "kami1976/rdkit-cp312"
COCONUT_DATASET = "aidensong123/casmi26-coconut-202609"
MODEL_DATASET = "isaackjoshua/casmi26-fp-model"

# Order matters: later files' functions may depend on earlier ones, and
# pipeline_v3 deliberately redefines predict_molecule/predict_test_set
# after pipeline_v2 so the final cell picks up the fused versions.
SOURCE_FILES = [
    "src/metric.py",
    "src/spectral_similarity.py",
    "src/baseline.py",
    "src/candidates.py",
    "src/propagation.py",
    "src/pipeline_v2.py",
    "src/fingerprint_model.py",
    "src/pipeline_v3.py",
    "src/submission.py",
]

_STRIP_PATTERNS = [
    re.compile(r"^from __future__ import annotations\s*$", re.MULTILINE),
    re.compile(r"^from \.\w+ import .+$", re.MULTILINE),
]


def _clean(source: str) -> str:
    for pattern in _STRIP_PATTERNS:
        source = pattern.sub("", source)
    return source.strip()


def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def _markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build_notebook() -> dict:
    cells = [
        _markdown_cell(
            "# CASMI 2026 Phase 3: propagation + learned fingerprint model\n\n"
            "Phase 2's analog propagation (0.19 public) fused with an MLP that predicts a "
            "molecule's 2048-bit Morgan fingerprint directly from its spectrum (trained on "
            "the 2.5M labelled training spectra, split by structure). Both signals score "
            "every COCONUT candidate in a ~1 mDa mass window; the model carries the cases "
            "where no close library analog exists. See the project repo README for the "
            "validation numbers behind this design."
        ),
        _code_cell(
            f'!pip install --no-index --find-links=/kaggle/input/datasets/{RDKIT_WHEEL_DATASET} rdkit -q\n'
        ),
        _code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import scipy.sparse as sp\n"
            "import multiprocessing as mp\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "from dataclasses import dataclass\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem import Descriptors, rdFingerprintGenerator\n"
            "from rdkit.Chem.MolStandardize import rdMolStandardize\n"
            "from rdkit.DataStructs import ExplicitBitVect\n"
            "from functools import lru_cache\n"
        ),
    ]

    for rel_path in SOURCE_FILES:
        raw = (ROOT / rel_path).read_text()
        cells.append(_markdown_cell(f"## {rel_path}"))
        cells.append(_code_cell(_clean(raw)))

    cells.append(_markdown_cell("## Run on the real competition data"))
    cells.append(
        _code_cell(
            f'DATA_DIR = "/kaggle/input/competitions/{COMPETITION}"\n'
            f'COCONUT_DIR = "/kaggle/input/datasets/{COCONUT_DATASET}"\n'
            f'MODEL_PATH = "/kaggle/input/datasets/{MODEL_DATASET}/fp_model.pt"\n\n'
            'train = pd.read_parquet(f"{DATA_DIR}/train.parquet")\n'
            'test = pd.read_parquet(f"{DATA_DIR}/test.parquet")\n\n'
            'lib_sources = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2", '
            '"massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"]\n'
            'library = build_library(train[train["ingest_lib"].isin(lib_sources)])\n'
            'print(f"anchor library: {len(library)} unique (structure, adduct) spectra")\n\n'
            'coconut = load_coconut(f"{COCONUT_DIR}/coconut_structures.parquet")\n'
            "candidate_df = build_candidate_pool(coconut, train)\n"
            'fps, masses = fingerprint_many(candidate_df["normalized_smiles"].tolist())\n'
            'candidate_df = candidate_df.assign(fingerprint=fps, exact_mass=masses)\n'
            'candidate_df = candidate_df.dropna(subset=["fingerprint", "exact_mass"]).reset_index(drop=True)\n'
            'print(f"candidate pool: {len(candidate_df)} structures")\n'
            "pool = candidate_pool_from_df(candidate_df)\n\n"
            'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
            "model = load_fingerprint_model(MODEL_PATH, device)\n"
            'print(f"fingerprint model loaded on {device}")\n\n'
            "try:\n"
            "    predictions = predict_test_set(test, library, pool, model, device)\n"
            "except Exception as e:\n"
            "    print(f'ERROR: predict_test_set failed entirely ({type(e).__name__}: {e}); '\n"
            "          f'falling back to library-frequency guesses for every molecule')\n"
            "    predictions = {mid: _global_fallback_candidates(library, None, N_GUESSES) "
            "for mid in test['molecule_id'].unique()}\n\n"
            'write_submission(predictions, "submission.csv", '
            f'sample_submission_path=f"{{DATA_DIR}}/sample_submission.csv")\n'
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata() -> dict:
    return {
        "id": f"{KAGGLE_USERNAME}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "casmi26_phase3.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
        "dataset_sources": [RDKIT_WHEEL_DATASET, COCONUT_DATASET, MODEL_DATASET],
        "kernel_sources": [],
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "casmi26_phase3.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2))
    print(f"Wrote {OUT_DIR}/casmi26_phase3.ipynb and kernel-metadata.json")


if __name__ == "__main__":
    main()
