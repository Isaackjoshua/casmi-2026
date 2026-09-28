"""Run a built Kaggle notebook locally, against local paths and a handful of
test molecules.

The notebook is a flattened copy of src/, so the ordinary test suite cannot
catch what actually breaks it: the flattener strips `from __future__` and
every relative import, so a name that was imported rather than defined is
simply undefined at runtime -- and the failure surfaces an hour into a
Kaggle run. This executes the real cells in one namespace and would raise
on exactly that.

Run: PYTHONPATH=. python3 scripts/dryrun_kernel.py kaggle_kernel_v5/<nb>.ipynb
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOCAL_PATHS = {
    '"/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra"': f'"{ROOT}/data/raw"',
    '"/kaggle/input/datasets/aidensong123/casmi26-coconut-202609"': f'"{ROOT}/data/coconut"',
    '"/kaggle/input/datasets/isaackjoshua/casmi26-pubchem-pool"': f'"{ROOT}/data/pubchem/kaggle_pool"',
    '"/kaggle/input/datasets/isaackjoshua/casmi26-fp-model/fp_model.pt"': f'"{ROOT}/data/processed/fp_model.pt"',
    '"/kaggle/input/datasets/isaackjoshua/casmi26-fp-model/peak_model.pt"': f'"{ROOT}/data/processed/peak_model.pt"',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--molecules", type=int, default=12)
    args = ap.parse_args()

    nb = json.loads(Path(args.notebook).read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    cells = [c for c in cells if not "".join(c["source"]).lstrip().startswith("!pip")]

    # A real notebook's cells live in __main__, which is importable, so
    # multiprocessing can pickle the worker functions they define by
    # reference. An exec into a bare dict cannot, so give the namespace a
    # registered module of its own.
    mod = types.ModuleType("kernel_ns")
    sys.modules["kernel_ns"] = mod
    ns = mod.__dict__
    t0 = time.time()
    for i, cell in enumerate(cells):
        src = "".join(cell["source"])
        for kaggle, local in LOCAL_PATHS.items():
            src = src.replace(kaggle, local)
        if "read_parquet(f\"{DATA_DIR}/test.parquet\")" in src:
            # only a handful of molecules, and write the submission here
            src = src.replace(
                'test = pd.read_parquet(f"{DATA_DIR}/test.parquet")',
                'test = pd.read_parquet(f"{DATA_DIR}/test.parquet")\n'
                f'_keep = test["molecule_id"].unique()[:{args.molecules}]\n'
                'test = test[test["molecule_id"].isin(_keep)].copy()\n'
                'print(f"DRY RUN: {len(_keep)} molecules, {len(test)} spectra", flush=True)',
            )
            src = src.replace('"submission.csv"', f'"{ROOT}/submissions/dryrun_v5.csv"')
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception:
            print(f"\n!!! cell {i} failed after {time.time()-t0:.0f}s", flush=True)
            raise
    print(f"\ndry run OK in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
