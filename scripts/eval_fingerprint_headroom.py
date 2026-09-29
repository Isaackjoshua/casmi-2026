"""How much ranking does a better fingerprint prediction actually buy?

The premise of a retrain is that improving the predicted fingerprint
improves the ranking. That is an assumption, and the last phase was built
on an assumption of the same shape ("coverage is the bottleneck") that the
leaderboard then refuted. This tests it before any GPU time is spent.

The model predicts P(bit = 1) for 2048 Morgan bits, and a candidate is
scored by `bits @ logit`. Interpolating the prediction toward the truth,

    p(lam) = (1 - lam) * p_model + lam * true_bits

sweeps the whole quality range in one experiment: lam = 0 is today's model,
lam = 1 is a perfect fingerprint, and the curve between them says what a
retrain is worth. Three shapes, three different decisions:

  steep early   -- small gains in prediction quality pay off; retrain.
  flat then     -- only a near-perfect fingerprint helps, which no
  steep late       realistic training run will deliver; the Morgan target
                   itself is the limit, and richer targets (MACCS, larger
                   radius, atom pairs) are the move instead.
  flat overall  -- the mass window contains structures Morgan cannot tell
                   apart at all, and no fingerprint work helps.

Reported against the curated pool, because two submissions now imply
essentially every test molecule's answer is already in it.

Run: PYTHONPATH=. python3 scripts/eval_fingerprint_headroom.py
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.candidates import FP_BITS, _fingerprint_one
from src.data import load_train
from src.pipeline_v2 import MASS_WINDOW_WIDEN_CAP, MASS_WINDOW_WIDEN_FACTOR, merge_spectra
from src.pipeline_v3 import load_fingerprint_model, predict_bit_probs
from src.propagation import MASS_WINDOW_DA, load_candidate_pool, mass_window, neutral_mass

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
EPS = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", default="0,0.1,0.2,0.35,0.5,0.7,0.85,1.0")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fingerprint_model([str(P / "fp_model.pt"), str(P / "peak_model.pt")], device)
    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")
    train = load_train()

    spec = importlib.util.spec_from_file_location("evalpc", ROOT / "scripts/eval_pubchem_pool.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    hard = helpers.hard_set(train)
    answers = hard.groupby("inchikey14")["normalized_smiles"].first().to_dict()

    lams = [float(x) for x in args.lambdas.split(",")]
    ranks = {lam: [] for lam in lams}
    windows, tanimotos = [], {lam: [] for lam in lams}

    t0 = time.time()
    for key, g in hard.groupby("inchikey14"):
        packed, _m = _fingerprint_one(answers[key])
        if packed is None:
            continue
        true_words = np.frombuffer(packed, dtype=np.uint64)
        true_bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8)[None, :], axis=1)[0, :FP_BITS]

        recs = g.to_dict("records")
        by_adduct = {}
        for r in recs:
            by_adduct.setdefault(r["adduct"], []).append(r)
        group = max(by_adduct.values(), key=lambda gr: max(len(x["ms2_mzs"]) for x in gr))
        merged = merge_spectra(group) if len(group) > 1 else group[0]
        qmass = neutral_mass(float(merged["precursor_mz"]), merged["adduct"])
        if qmass is None:
            continue

        lo, hi = mass_window(pool, qmass, MASS_WINDOW_DA)
        widen = MASS_WINDOW_DA
        while hi <= lo and widen < MASS_WINDOW_WIDEN_CAP:
            widen *= MASS_WINDOW_WIDEN_FACTOR
            lo, hi = mass_window(pool, qmass, widen)
        if hi <= lo:
            continue

        cand = np.asarray(pool.fp_words[lo:hi])
        hit = np.flatnonzero((cand == true_words[None, :]).all(axis=1))
        if len(hit) == 0:
            continue  # answer not in this window; the reachable regime is the point
        windows.append(hi - lo)

        bits = np.unpackbits(cand.view(np.uint8), axis=1)[:, :FP_BITS].astype(np.float64)
        p_model = predict_bit_probs(model, group, device).mean(axis=0).astype(np.float64)

        for lam in lams:
            p = np.clip((1 - lam) * p_model + lam * true_bits, EPS, 1 - EPS)
            logit = np.log(p) - np.log1p(-p)
            s = bits @ logit
            ranks[lam].append(int(1 + (s > s[hit[0]]).sum()))
            # Tanimoto@0.5 of this interpolated "prediction", the metric
            # checkpoints have been selected on
            pred_on = p >= 0.5
            true_on = true_bits.astype(bool)
            inter = (pred_on & true_on).sum()
            union = (pred_on | true_on).sum()
            tanimotos[lam].append(inter / union if union else 0.0)

    n = len(ranks[lams[0]])
    print(f"{n} molecules with the answer in their window; median window "
          f"{np.median(windows):.0f} candidates ({time.time()-t0:.0f}s)\n")
    print(f"{'lambda':>7} {'fp Tanimoto':>12} {'MRR@25':>8} {'top-1':>7} {'median rank':>12}")
    base = None
    for lam in lams:
        r = np.array(ranks[lam])
        mrr = np.mean([1.0 / x if x <= 25 else 0.0 for x in r])
        if base is None:
            base = mrr
        print(f"{lam:7.2f} {np.mean(tanimotos[lam]):12.3f} {mrr:8.4f} "
              f"{(r == 1).mean():7.3f} {np.median(r):12.0f}")
    print(f"\nlambda 0 is the deployed model; lambda 1 is a perfect fingerprint.")


if __name__ == "__main__":
    main()
