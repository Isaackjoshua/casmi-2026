"""Can a cheap descriptor filter strip PubChem's distractors without
dropping real natural products?

The merged-pool experiment showed PubChem's cost is its 200-470
mass-matched distractors per window. If most of those are chemistry that
never occurs in nature, a filter removes cost without removing coverage.

Calibrated two ways, because a filter is only useful if both hold:
  keep-rate on COCONUT     -- true natural products; must stay high
  keep-rate on PubChem     -- the distractor population; should fall a lot
  keep-rate on the 86      -- the hard-set answers PubChem uniquely
                              recovers; these are what coverage is *for*

Run: PYTHONPATH=. python3 scripts/calibrate_np_filter.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.data import load_train

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

# groups that essentially do not occur in natural products
UNNATURAL = [Chem.MolFromSmarts(s) for s in [
    "[N+](=O)[O-]",          # nitro
    "S(=O)(=O)[NX3]",        # sulfonamide
    "[F][CX4][F]",           # geminal di/trifluoro
    "c1ccccc1[F,Cl,Br,I]",   # halogenated arene
    "[NX2]=[NX2]",           # azo
    "[SX2][SX2][SX2]",       # tri-sulfide chains
    "[CX4]([F])([F])([F])",  # CF3
]]


def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "o": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O"),
        "n": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N"),
        "halo": sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I")),
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "arom_frac": (sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(mol.GetNumHeavyAtoms(), 1)),
        "stereo": rdMolDescriptors.CalcNumAtomStereoCenters(mol) if mol.GetNumAtoms() < 200 else 0,
        "unnatural": any(mol.HasSubstructMatch(p) for p in UNNATURAL if p is not None),
    }


def rates(name, smiles_list, filters):
    d = [descriptors(s) for s in smiles_list]
    d = [x for x in d if x is not None]
    print(f"\n{name} (n={len(d)})")
    for fname, fn in filters.items():
        k = sum(1 for x in d if fn(x))
        print(f"  {fname:28s} keeps {k/len(d):6.1%}")
    return d


FILTERS = {
    "no unnatural group": lambda x: not x["unnatural"],
    "halo <= 1": lambda x: x["halo"] <= 1,
    "has O or N": lambda x: x["o"] + x["n"] >= 1,
    "O+N >= 2": lambda x: x["o"] + x["n"] >= 2,
    "fsp3 >= 0.2": lambda x: x["fsp3"] >= 0.2,
    "arom_frac <= 0.7": lambda x: x["arom_frac"] <= 0.7,
    "has a stereocentre": lambda x: x["stereo"] >= 1,
    "COMBINED (loose)": lambda x: (not x["unnatural"]) and x["halo"] <= 1 and x["o"] + x["n"] >= 1,
    "COMBINED (medium)": lambda x: (not x["unnatural"]) and x["halo"] <= 1 and x["o"] + x["n"] >= 2
                                   and x["fsp3"] >= 0.2,
    "COMBINED (tight)": lambda x: (not x["unnatural"]) and x["halo"] <= 1 and x["o"] + x["n"] >= 2
                                   and x["fsp3"] >= 0.2 and x["arom_frac"] <= 0.7,
}

SAMPLE = 8000


def main():
    t0 = time.time()
    coconut = pd.read_parquet(ROOT / "data/coconut/coconut_structures.parquet",
                              columns=["inchikey", "canonical_smiles"])
    coconut["k14"] = coconut["inchikey"].str.split("-").str[0]
    rates("COCONUT (true natural products)",
          coconut["canonical_smiles"].dropna().sample(SAMPLE, random_state=0).tolist(), FILTERS)

    # the PubChem distractor population, as a mass-window query actually sees
    # it: one row group from the middle of the mass-sorted pool
    pf = pq.ParquetFile(ROOT / "data/pubchem/pubchem_pool/meta.parquet")
    mid = pf.num_row_groups // 2
    tbl = pf.read_row_group(mid, columns=["inchikey14", "normalized_smiles"])
    pub = tbl.column("normalized_smiles").to_pylist()
    rng = np.random.default_rng(0)
    rates("PubChem (the distractors)",
          [pub[i] for i in rng.choice(len(pub), SAMPLE, replace=False)], FILTERS)

    # the 86 answers PubChem uniquely recovers -- the whole point of coverage
    train = load_train()
    lib_keys = set(train[train["ingest_lib"].isin(
        ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"])]["inchikey14"])
    novel = train[train["ingest_lib"].isin(["massbank", "mona"])]
    novel = novel[~novel["inchikey14"].isin(lib_keys)]
    keys = set(novel["inchikey14"].drop_duplicates().sample(n=150, random_state=42))
    hard = novel[novel["inchikey14"].isin(keys)]
    coconut_keys = set(coconut["k14"])
    uniq = hard[~hard["inchikey14"].isin(coconut_keys)].groupby("inchikey14")["normalized_smiles"].first()
    rates("hard-set answers only PubChem has", uniq.tolist(), FILTERS)
    print(f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
