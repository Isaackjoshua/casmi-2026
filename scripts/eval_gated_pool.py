"""Can we tell, per molecule, which pool holds the answer?

Tiering leaves headroom: it scores 0.1785 on the hard set where a perfect
per-molecule choice between the curated pool and PubChem would score
0.2187. Choosing requires comparing evidence across pools, and the naive
comparison is exactly what makes merging lose -- the best of PubChem's
~400 mass-matched candidates beats the best of the curated pool's ~2 by
luck alone, because the maximum of more draws is larger.

So compare each pool's best candidate to its *own* window, and correct
for how many draws it took: for n candidates the maximum of n standard
normals sits near sqrt(2 ln n), so

    adjusted = (best_loglik - mean) / std  -  sqrt(2 * ln n)

is roughly how surprising that pool's best candidate is, net of pool
size. The pool with the larger adjusted score should be the one holding
the answer.

This measures the gate directly, as classification: the hard set is split
59/91 by whether COCONUT contains the answer, so the gate has to beat
61% (always guess PubChem) to be worth building a ranker on.

Run: PYTHONPATH=. python3 scripts/eval_gated_pool.py
"""

import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baseline import build_library
from src.candidates import FP_BITS
from src.data import load_train
from src.fingerprint_model import fingerprint_loglik_scores
from src.large_pool import load_large_pool, window
from src.pipeline_v2 import MASS_WINDOW_WIDEN_CAP, MASS_WINDOW_WIDEN_FACTOR, merge_spectra
from src.pipeline_v3 import load_fingerprint_model, predict_bit_probs
from src.propagation import MASS_WINDOW_DA, load_candidate_pool, mass_window, neutral_mass

ROOT = Path(__file__).resolve().parents[1]
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]


def _load_eval_helpers():
    spec = importlib.util.spec_from_file_location("evalpc", ROOT / "scripts/eval_pubchem_pool.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def adjusted_score(loglik):
    """How surprising this window's best candidate is, net of pool size."""
    n = len(loglik)
    if n == 0:
        return -np.inf, 0
    if n < 3:
        # too few draws for a z-score; the correction is meaningless here
        return 0.0, n
    sd = float(loglik.std())
    if sd <= 0:
        return 0.0, n
    z = (float(loglik.max()) - float(loglik.mean())) / sd
    return z - math.sqrt(2.0 * math.log(n)), n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fingerprint_model([str(ROOT / "data/processed/fp_model.pt"),
                                    str(ROOT / "data/processed/peak_model.pt")], device)
    train = load_train()
    lib = build_library(train[train["ingest_lib"].isin(FIVE)])
    helpers = _load_eval_helpers()

    small = load_candidate_pool(ROOT / "data/processed/candidate_fingerprints.parquet")
    large = load_large_pool(ROOT / "data/pubchem/kaggle_pool")
    coconut_keys = set(pd.read_parquet(ROOT / "data/coconut/coconut_structures.parquet",
                                       columns=["inchikey"])["inchikey"].str.split("-").str[0])
    baseline_pool = helpers.coconut_only_pool(small, coconut_keys)
    hard = helpers.hard_set(train)
    print(f"curated (COCONUT-only) pool: {len(baseline_pool):,}   PubChem: {len(large):,}", flush=True)

    rows_out = []
    t0 = time.time()
    for key, g in hard.groupby("inchikey14"):
        recs = g.to_dict("records")
        by_adduct = {}
        for r in recs:
            by_adduct.setdefault(r["adduct"], []).append(r)
        # judge on the richest spectrum for this molecule
        group = max(by_adduct.values(), key=lambda gr: max(len(x["ms2_mzs"]) for x in gr))
        merged = merge_spectra(group) if len(group) > 1 else group[0]
        qmass = neutral_mass(float(merged["precursor_mz"]), merged["adduct"])
        if qmass is None:
            continue
        probs = predict_bit_probs(model, group, device).mean(axis=0)

        # curated window (mass_window returns a [lo, hi) range, like window)
        clo, chi = mass_window(baseline_pool, qmass, MASS_WINDOW_DA)
        widen = MASS_WINDOW_DA
        while chi <= clo and widen < MASS_WINDOW_WIDEN_CAP:
            widen *= MASS_WINDOW_WIDEN_FACTOR
            clo, chi = mass_window(baseline_pool, qmass, widen)
        if chi > clo:
            bits_c = np.unpackbits(np.asarray(baseline_pool.fp_words[clo:chi]).view(np.uint8), axis=1)[:, :FP_BITS]
            adj_c, n_c = adjusted_score(fingerprint_loglik_scores(probs, bits_c))
        else:
            adj_c, n_c = -np.inf, 0

        # PubChem window, widened the same way so an empty one is not scored -inf
        lo, hi = window(large, qmass, MASS_WINDOW_DA)
        widen = MASS_WINDOW_DA
        while hi <= lo and widen < MASS_WINDOW_WIDEN_CAP:
            widen *= MASS_WINDOW_WIDEN_FACTOR
            lo, hi = window(large, qmass, widen)
        if hi > lo:
            bits_p = np.unpackbits(np.asarray(large.fp_words[lo:hi]).view(np.uint8), axis=1)[:, :FP_BITS]
            adj_p, n_p = adjusted_score(fingerprint_loglik_scores(probs, bits_p))
        else:
            adj_p, n_p = -np.inf, 0

        rows_out.append({
            "key": key, "in_coconut": key in coconut_keys,
            "n_curated": n_c, "n_pubchem": n_p, "adj_curated": adj_c, "adj_pubchem": adj_p,
        })

    df = pd.DataFrame(rows_out)
    print(f"{len(df)} molecules in {time.time()-t0:.0f}s\n", flush=True)
    print(f"window sizes: curated median {df['n_curated'].median():.0f}, "
          f"PubChem median {df['n_pubchem'].median():.0f}")

    truth = df["in_coconut"]              # True  -> curated holds the answer
    pick_curated = df["adj_curated"] > df["adj_pubchem"]
    base = max(truth.mean(), 1 - truth.mean())
    acc = (pick_curated == truth).mean()
    print(f"\nbase rate (always guess the larger class): {base:.1%}")
    print(f"gate accuracy                            : {acc:.1%}")
    print(f"  of {truth.sum()} in-COCONUT molecules, gate picks curated for {(pick_curated & truth).sum()}")
    print(f"  of {(~truth).sum()} not-in-COCONUT,     gate picks PubChem for {((~pick_curated) & (~truth)).sum()}")
    finite = np.isfinite(df["adj_curated"]) & np.isfinite(df["adj_pubchem"])
    print(f"\nadjusted-score separation (mean, over {int(finite.sum())} molecules with both windows non-empty):")
    for label, sub in (("answer in COCONUT", df[truth & finite]), ("answer not in COCONUT", df[~truth & finite])):
        print(f"  {label:24s}: curated {sub['adj_curated'].mean():+.2f}, "
              f"PubChem {sub['adj_pubchem'].mean():+.2f}, margin {(sub['adj_curated']-sub['adj_pubchem']).mean():+.2f}")
    accf = (pick_curated[finite] == truth[finite]).mean()
    print(f"gate accuracy on those {int(finite.sum())}: {accf:.1%} "
          f"(base rate {max(truth[finite].mean(), 1-truth[finite].mean()):.1%})")
    df.to_csv(ROOT / "logs/gate_scores.csv", index=False)


if __name__ == "__main__":
    main()
