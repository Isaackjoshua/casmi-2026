"""Turn PubChem's bulk CID-SMILES dump into a mass-sorted, fingerprinted
candidate pool.

Why: the candidate pool (COCONUT + training structures) covers only ~39%
of real natural-product chemistry -- measured against the GNPS library and
our own hard-validation set -- and 72% of what COCONUT misses is in
PubChem. Since the test set is explicitly "natural products, hypothesised
natural products, natural product analogs, and synthetic molecules that
might plausibly occur in nature", the analogs and synthetics live in
PubChem, not COCONUT. Coverage, not ranking, is what caps the score.

Filters applied (a mass spectrometry candidate must at minimum be a
plausible small organic molecule in the test's mass range):
  - exact mass within [MIN_MASS, MAX_MASS]
  - elements restricted to ELEMENTS (CHNOPS + halogens)
  - single fragment (no salts/mixtures -- the largest component is what a
    spectrum would represent, and PubChem's mixtures are mostly salts)

Output (data/pubchem/pubchem_pool/):
  masses.npy     float64 (n,)   exact masses, ascending -- the sort order
  fps.npy        uint64  (n,32) Morgan fingerprints, same order
  meta.parquet                  inchikey14 + normalized_smiles, same order
Written in mass order so a query's mass window is a contiguous slice, and
split across .npy files so the kernel can memory-map rather than load
~20 GB into RAM.

Run: PYTHONPATH=. python3 scripts/build_pubchem_pool.py
"""

import argparse
import gzip
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from src.candidates import FP_BITS, FP_RADIUS

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MIN_MASS = 120.0  # test monoisotopic masses run 157-1159; pad for adduct/loss slack
MAX_MASS = 1300.0
ELEMENTS = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Se", "B", "Si"}
CHUNK = 200_000

_GEN = None


def _gen():
    global _GEN
    if _GEN is None:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    return _GEN


def _process(smiles: str):
    """SMILES -> (inchikey14, canonical smiles, exact mass, packed fp) or None."""
    if "." in smiles:  # mixture/salt: keep the largest fragment only
        smiles = max(smiles.split("."), key=len)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ELEMENTS:
            return None
    mass = Descriptors.ExactMolWt(mol)
    if not (MIN_MASS <= mass <= MAX_MASS):
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not key:
        return None
    fp = _gen().GetFingerprint(mol)
    packed = np.packbits(np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")).tobytes()
    return key.split("-")[0], Chem.MolToSmiles(mol), mass, packed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/pubchem/CID-SMILES.gz"))
    ap.add_argument("--out", default=str(ROOT / "data/pubchem/pubchem_pool"))
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--limit", type=int, default=0, help="stop after N input lines (smoke tests)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    keys, smis, masses, fps = [], [], [], []
    n_in = 0
    with gzip.open(args.input, "rt") as fh, Pool(args.workers) as pool:
        def batches():
            batch = []
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    batch.append(parts[1])
                if len(batch) >= CHUNK:
                    yield batch
                    batch = []
            if batch:
                yield batch

        for batch in batches():
            n_in += len(batch)
            for r in pool.map(_process, batch, chunksize=1000):
                if r is not None:
                    keys.append(r[0]); smis.append(r[1]); masses.append(r[2]); fps.append(r[3])
            if n_in % (CHUNK * 5) == 0:
                el = time.time() - t0
                print(f"  read {n_in/1e6:.1f}M, kept {len(keys)/1e6:.2f}M ({len(keys)/max(n_in,1):.0%}), "
                      f"{n_in/el/1000:.0f}k/s, {el/60:.1f} min", flush=True)
            if args.limit and n_in >= args.limit:
                break

    print(f"parsed {n_in} lines, kept {len(keys)} ({time.time()-t0:.0f}s)")

    # Deduplicate on structure, then sort by mass so a mass window is a slice.
    df = pd.DataFrame({"inchikey14": keys, "normalized_smiles": smis, "exact_mass": masses})
    df["_fp"] = fps
    df = df.drop_duplicates(subset=["inchikey14"], keep="first").sort_values("exact_mass").reset_index(drop=True)
    print(f"unique structures: {len(df)}")

    fp_arr = np.stack([np.frombuffer(b, dtype=np.uint64) for b in df["_fp"]])
    np.save(out / "masses.npy", df["exact_mass"].to_numpy(dtype=np.float64))
    np.save(out / "fps.npy", fp_arr)
    df[["inchikey14", "normalized_smiles"]].to_parquet(out / "meta.parquet", index=False)
    print(f"wrote {out}: masses {df.shape[0]}, fps {fp_arr.shape}, "
          f"{fp_arr.nbytes/1e9:.1f} GB ({(time.time()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()
