"""Assembles the Phase 4 TIERED-POOL pipeline as a self-contained Kaggle
Notebook. See build_kaggle_kernel.py for why it must be a Notebook and
build_kaggle_kernel_v2.py for the dataset-mount and rdkit-wheel details.

What is new vs. v4: the 94M-structure PubChem pool
(isaackjoshua/casmi26-pubchem-pool, 28 GB) is attached and consulted as a
second tier. It is memory-mapped, never loaded -- 24 GB of fingerprints
would not fit in Kaggle's 30 GB of RAM, and each query only touches the
rows inside its own ~1 mDa mass window. See src/pipeline_v4.py for why
the tiers are consulted in order rather than merged.

Run: python3 scripts/build_kaggle_kernel_v5.py
Then: cd kaggle_kernel_v5 && kaggle kernels push -p .
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle_kernel_v5"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
KAGGLE_USERNAME = "isaackjoshua"
KERNEL_SLUG = "casmi26-phase-4-tiered-pool"
KERNEL_TITLE = "CASMI26 Phase 4 Tiered Pool"
RDKIT_WHEEL_DATASET = "kami1976/rdkit-cp312"
COCONUT_DATASET = "aidensong123/casmi26-coconut-202609"
MODEL_DATASET = "isaackjoshua/casmi26-fp-model"
PUBCHEM_DATASET = "isaackjoshua/casmi26-pubchem-pool"

# Order matters: later files' functions may depend on earlier ones, and each
# pipeline_vN deliberately redefines predict_molecule/predict_test_set after
# the previous one so the final cell picks up the newest version.
SOURCE_FILES = [
    "src/metric.py",
    "src/spectral_similarity.py",
    "src/baseline.py",
    "src/candidates.py",
    "src/propagation.py",
    "src/pipeline_v2.py",
    "src/fingerprint_model.py",
    "src/peak_transformer.py",
    "src/pipeline_v3.py",
    "src/large_pool.py",
    "src/pipeline_v4.py",
    "src/submission.py",
]

_STRIP_PATTERNS = [
    re.compile(r"^from __future__ import annotations\s*$", re.MULTILINE),
    re.compile(r"^from \.\w+ import .+$", re.MULTILINE),
]

# Flattening strips the relative imports, so any name a later file imports
# under a new alias has to be bound explicitly before that file's cell.
ALIASES = {
    "src/pipeline_v4.py": "predict_curated = predict_molecule  # pipeline_v3's, before v4 rebinds the name\n",
}


def _clean(source):
    for pattern in _STRIP_PATTERNS:
        source = pattern.sub("", source)
    return source.strip()


def _code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def _markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build_notebook():
    cells = [
        _markdown_cell(
            "# CASMI 2026 Phase 4: a tiered candidate pool\n\n"
            "Phase 3 (propagation + learned fingerprint model, 0.230 public) with the "
            "candidate pool extended by 94M PubChem structures.\n\n"
            "Coverage, not ranking, was the bottleneck: locally the pipeline scores ~0.50 "
            "where the pool contains the answer, against 0.230 on the leaderboard, implying "
            "roughly half the test molecules are not in the COCONUT-based pool at all. "
            "PubChem contains 86 of the 91 hard-validation structures COCONUT misses.\n\n"
            "But merging the pools *loses* (0.1605 -> 0.1479): 200-470 mass-matched "
            "distractors per window cost more than the coverage gains. Measured separately, "
            "the curated pool wins where it has the answer (0.3854 vs 0.2052) and PubChem is "
            "the only route where it does not (0.1107 vs 0.0147). So the tiers are consulted "
            "in order -- curated candidates keep the ranks they earn, PubChem fills only the "
            "slots of the 25 they leave empty -- which spends none of the distractor cost.\n\n"
            "The 24 GB of PubChem fingerprints are memory-mapped, not loaded; each query "
            "touches only its own ~1 mDa mass window."
        ),
        _code_cell(
            f'!pip install --no-index --find-links=/kaggle/input/datasets/{RDKIT_WHEEL_DATASET} rdkit -q\n'
        ),
        _code_cell(
            "import math\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import pyarrow.parquet as pq\n"
            "import scipy.sparse as sp\n"
            "import multiprocessing as mp\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "from dataclasses import dataclass\n"
            "from pathlib import Path\n"
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
        if rel_path in ALIASES:
            cells.append(_code_cell(ALIASES[rel_path]))
        cells.append(_code_cell(_clean(raw)))

    cells.append(_markdown_cell("## Run on the real competition data"))
    cells.append(
        _code_cell(
            f'DATA_DIR = "/kaggle/input/competitions/{COMPETITION}"\n'
            f'COCONUT_DIR = "/kaggle/input/datasets/{COCONUT_DATASET}"\n'
            f'PUBCHEM_DIR = "/kaggle/input/datasets/{PUBCHEM_DATASET}"\n'
            f'MODEL_PATHS = ["/kaggle/input/datasets/{MODEL_DATASET}/fp_model.pt", '
            f'"/kaggle/input/datasets/{MODEL_DATASET}/peak_model.pt"]\n\n'
            'train = pd.read_parquet(f"{DATA_DIR}/train.parquet")\n'
            'test = pd.read_parquet(f"{DATA_DIR}/test.parquet")\n\n'
            'lib_sources = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2", '
            '"massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"]\n'
            'library = build_library(train[train["ingest_lib"].isin(lib_sources)])\n'
            'print(f"anchor library: {len(library)} unique (structure, adduct) spectra", flush=True)\n\n'
            'coconut = load_coconut(f"{COCONUT_DIR}/coconut_structures.parquet")\n'
            "candidate_df = build_candidate_pool(coconut, train)\n"
            'fps, masses = fingerprint_many(candidate_df["normalized_smiles"].tolist())\n'
            'candidate_df = candidate_df.assign(fingerprint=fps, exact_mass=masses)\n'
            'candidate_df = candidate_df.dropna(subset=["fingerprint", "exact_mass"]).reset_index(drop=True)\n'
            "pool = candidate_pool_from_df(candidate_df)\n"
            'print(f"curated pool: {len(candidate_df)} structures", flush=True)\n\n'
            "# tier 2. A missing or unreadable pool must not lose the submission: the\n"
            "# pipeline then degrades to exactly the Phase 3 behaviour that scored 0.230.\n"
            "try:\n"
            "    large = load_large_pool(PUBCHEM_DIR)\n"
            '    print(f"PubChem pool: {len(large):,} structures (memory-mapped)", flush=True)\n'
            "except Exception as e:\n"
            "    large = None\n"
            "    print(f'WARNING: PubChem pool unavailable ({type(e).__name__}: {e}); "
            "curated pool only', flush=True)\n\n"
            'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
            "model = load_fingerprint_model(MODEL_PATHS, device)\n"
            'print(f"ensemble of {len(MODEL_PATHS)} models loaded on {device}", flush=True)\n\n'
            "# A valid submission exists before any of the expensive work starts, so\n"
            "# nothing later can leave the run with no file at all. Version 1 of this\n"
            "# notebook threw on the hidden-test rerun, and a single molecule with an\n"
            "# adduct outside ADDUCT_SPEC is enough to do that.\n"
            "FALLBACK = _global_fallback_candidates(library, None, N_GUESSES)\n"
            'write_submission({mid: FALLBACK for mid in test["molecule_id"].unique()}, "submission.csv",\n'
            "                 fallback=FALLBACK)\n"
            'print(f"placeholder submission written; now computing the real one", flush=True)\n\n'
            "try:\n"
            "    predictions = predict_test_set(test, library, pool, large, model, device,\n"
            "                                   fallback=FALLBACK)\n"
            "except Exception as e:\n"
            "    print(f'ERROR: predict_test_set failed entirely ({type(e).__name__}: {e}); '\n"
            "          f'keeping library-frequency guesses for every molecule')\n"
            "    predictions = {mid: FALLBACK for mid in test['molecule_id'].unique()}\n\n"
            'write_submission(predictions, "submission.csv",\n'
            f'                 sample_submission_path=f"{{DATA_DIR}}/sample_submission.csv",\n'
            "                 fallback=FALLBACK)\n"
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


def build_metadata():
    return {
        "id": f"{KAGGLE_USERNAME}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "casmi26_phase4_tiered_pool.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
        "dataset_sources": [RDKIT_WHEEL_DATASET, COCONUT_DATASET, MODEL_DATASET, PUBCHEM_DATASET],
        "kernel_sources": [],
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "casmi26_phase4_tiered_pool.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2))
    print(f"Wrote {OUT_DIR}/casmi26_phase4_tiered_pool.ipynb and kernel-metadata.json")


if __name__ == "__main__":
    main()
