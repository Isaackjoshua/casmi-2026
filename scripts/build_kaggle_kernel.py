"""Assembles a self-contained Kaggle Notebook (.ipynb) from src/*.py so it
can be pushed and run on Kaggle's own infrastructure -- required because
this is a Code Competition: submissions must come from a Notebook run
there (internet disabled, competition dataset attached), not a direct CSV
upload (the plain `kaggle competitions submit` API call is rejected with a
400 for this competition).

Run: python3 scripts/build_kaggle_kernel.py
Then: cd kaggle_kernel && kaggle kernels push -p .
The push both creates/updates the kernel on kaggle.com and runs it there.
After it finishes, open the kernel's page on kaggle.com and click "Submit
to Competition" -- that click is a Kaggle website action with no public
API equivalent, so it has to be done by the account owner in a browser.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle_kernel"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
KAGGLE_USERNAME = "isaackjoshua"
KERNEL_SLUG = "casmi26-baseline-v1"
# Kaggle's default Python 3.12 notebook image doesn't ship rdkit, and internet
# is disabled for a submittable run, so it has to come from a pre-uploaded
# wheel dataset instead of pip's index. This one carries a cp312 wheel.
RDKIT_WHEEL_DATASET = "kami1976/rdkit-cp312"

# Order matters: later files' functions may depend on earlier ones.
SOURCE_FILES = ["src/metric.py", "src/spectral_similarity.py", "src/baseline.py", "src/submission.py"]

_STRIP_PATTERNS = [
    re.compile(r"^from __future__ import annotations\s*$", re.MULTILINE),
    re.compile(r"^from \.\w+ import .+$", re.MULTILINE),  # relative package imports -- flattened into one file
]


def _clean(source: str) -> str:
    for pattern in _STRIP_PATTERNS:
        source = pattern.sub("", source)
    return source.strip()


def _code_cell(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def _markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build_notebook() -> dict:
    cells = [
        _markdown_cell(
            "# CASMI 2026 baseline v1\n\n"
            "Deduplicated-library + sparse coarse pre-filter + exact modified-cosine "
            "rescoring, max-pooled across a molecule's spectra, deduplicated on the "
            "tautomer-canonical InChIKey14. See the project repo for the annotated "
            "source and design notes."
        ),
        _code_cell(
            f'!pip install --no-index --find-links=/kaggle/input/{RDKIT_WHEEL_DATASET.split("/")[-1]} rdkit -q\n'
        ),
        _code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import scipy.sparse as sp\n"
            "from dataclasses import dataclass\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem.MolStandardize import rdMolStandardize\n"
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
            # Competition data is mounted under input/competitions/<slug>/, not
            # directly under input/<slug>/ (confirmed by an os.walk diagnostic
            # run against this exact kernel -- the two are easy to conflate).
            f'DATA_DIR = "/kaggle/input/competitions/{COMPETITION}"\n'
            'train = pd.read_parquet(f"{DATA_DIR}/train.parquet")\n'
            'test = pd.read_parquet(f"{DATA_DIR}/test.parquet")\n\n'
            'lib_sources = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]\n'
            'library = build_library(train[train["ingest_lib"].isin(lib_sources)])\n'
            'print(f"library: {len(library)} unique (structure, adduct) spectra")\n\n'
            "predictions = predict_test_set(test, library)\n\n"
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
        "title": "CASMI26 Baseline v1",
        "code_file": "casmi26_baseline_v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
        "dataset_sources": [RDKIT_WHEEL_DATASET],
        "kernel_sources": [],
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "casmi26_baseline_v1.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2))
    print(f"Wrote {OUT_DIR}/casmi26_baseline_v1.ipynb and kernel-metadata.json")


if __name__ == "__main__":
    main()
