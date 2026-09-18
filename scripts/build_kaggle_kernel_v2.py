"""Assembles the Phase 2 (candidate-pool expansion) pipeline as a
self-contained Kaggle Notebook -- see build_kaggle_kernel.py for why this
has to be a Notebook pushed to Kaggle rather than a direct CSV submission.

Fingerprinting the ~729k-structure candidate pool happens inline in the
notebook (took ~13s locally with 32 cores; still comfortably fast on
Kaggle's smaller instance) rather than shipping a precomputed cache, to
keep the submission self-contained and avoid managing a second dataset
upload every time src/candidates.py changes.

Run: python3 scripts/build_kaggle_kernel_v2.py
Then: cd kaggle_kernel_v2 && kaggle kernels push -p .
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle_kernel_v2"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
KAGGLE_USERNAME = "isaackjoshua"
# Must match what Kaggle actually derives from the title (a clean-URL slug),
# or pushes conflict/land at a different URL than kernel-metadata.json claims.
KERNEL_SLUG = "casmi26-phase-2-candidate-expansion"
RDKIT_WHEEL_DATASET = "kami1976/rdkit-cp312"
COCONUT_DATASET = "aidensong123/casmi26-coconut-202609"

# Order matters: later files' functions may depend on earlier ones.
SOURCE_FILES = [
    "src/metric.py",
    "src/spectral_similarity.py",
    "src/baseline.py",
    "src/candidates.py",
    "src/propagation.py",
    "src/pipeline_v2.py",
    "src/submission.py",
]

_STRIP_PATTERNS = [
    re.compile(r"^from __future__ import annotations\s*$", re.MULTILINE),
    re.compile(r"^from \.\w+ import .+$", re.MULTILINE),  # relative package imports -- flattened into one file
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
            "# CASMI 2026 Phase 2: candidate-pool expansion\n\n"
            "Phase 1's library search (0.063 public score) can only ever guess structures "
            "already in the training library. This expands the candidate pool to COCONUT "
            "(~739k natural products) plus the training library's own structures, ranked by "
            "`max_a sim(anchor)^4 * Tanimoto(candidate, anchor)` against spectrally-similar "
            "library analogs found via entropy-weighted spectral similarity. See the project "
            "repo's README for the local validation numbers behind this design."
        ),
        _code_cell(
            # With >1 dataset_sources attached, Kaggle mounts each one under
            # input/datasets/<owner>/<slug>/ instead of input/<slug>/ directly
            # (confirmed by an os.walk diagnostic run against this exact
            # kernel config) -- the single-dataset v1 kernel used the flat
            # path and that's easy to assume carries over; it doesn't.
            f'!pip install --no-index --find-links=/kaggle/input/datasets/{RDKIT_WHEEL_DATASET} rdkit -q\n'
        ),
        _code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import scipy.sparse as sp\n"
            "import multiprocessing as mp\n"
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
            # Competition data mounts under input/competitions/<slug>/, confirmed
            # by an earlier os.walk diagnostic run against this exact setup.
            f'DATA_DIR = "/kaggle/input/competitions/{COMPETITION}"\n'
            f'COCONUT_DIR = "/kaggle/input/datasets/{COCONUT_DATASET}"\n\n'
            'train = pd.read_parquet(f"{DATA_DIR}/train.parquet")\n'
            'test = pd.read_parquet(f"{DATA_DIR}/test.parquet")\n\n'
            # All 11 training libraries. Validated locally with two independent
            # held-out pairs (massbank+mona held out, then spectraverse+msdial
            # held out instead): adding the smaller/remaining sources lifted the
            # hard-validation MRR@25 from 0.4586 (5 sources) to 0.4878-0.4914
            # (9 sources) -- more anchors to search against, directly.
            'lib_sources = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2", '
            '"massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"]\n'
            'library = build_library(train[train["ingest_lib"].isin(lib_sources)])\n'
            'print(f"anchor library: {len(library)} unique (structure, adduct) spectra")\n\n'
            'coconut = load_coconut(f"{COCONUT_DIR}/coconut_structures.parquet")\n'
            "candidate_df = build_candidate_pool(coconut, train)\n"
            'print(f"candidate pool (pre-fingerprint): {len(candidate_df)} unique structures")\n\n'
            'fps, masses = fingerprint_many(candidate_df["normalized_smiles"].tolist())\n'
            'candidate_df = candidate_df.assign(fingerprint=fps, exact_mass=masses)\n'
            'candidate_df = candidate_df.dropna(subset=["fingerprint", "exact_mass"]).reset_index(drop=True)\n'
            'print(f"candidate pool (fingerprinted): {len(candidate_df)} structures")\n\n'
            "pool = candidate_pool_from_df(candidate_df)\n\n"
            "# predict_test_set already guards each molecule individually (see\n"
            "# pipeline_v2.py), but this outer guard is the last resort: the hidden\n"
            "# test set can differ from the public one in ways nothing here was\n"
            "# tested against, and a hard crash forfeits the whole submission --\n"
            "# better to hand back frequency-based guesses for every molecule than\n"
            "# nothing at all.\n"
            "try:\n"
            "    predictions = predict_test_set(test, library, pool)\n"
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
        "title": "CASMI26 Phase 2 - Candidate Expansion",
        "code_file": "casmi26_phase2_v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
        "dataset_sources": [RDKIT_WHEEL_DATASET, COCONUT_DATASET],
        "kernel_sources": [],
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "casmi26_phase2_v1.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2))
    print(f"Wrote {OUT_DIR}/casmi26_phase2_v1.ipynb and kernel-metadata.json")


if __name__ == "__main__":
    main()
