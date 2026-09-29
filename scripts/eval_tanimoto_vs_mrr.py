"""Does a better predicted fingerprint actually rank better -- for real models?

scripts/eval_fingerprint_headroom.py interpolates the prediction toward the
true bits and shows a violently steep curve: +10% fingerprint Tanimoto for
+66% MRR. That curve cannot be trusted, because interpolation nudges every
bit toward the answer at once. A real improvement is noisy -- better on some
bits, worse on others -- and carries no such privileged signal.

So measure the thing itself. Eleven checkpoints exist, spanning a 62M-param
binned MLP to a 4.6M-param peak transformer, several fine-tunes and two
sizes. Score each one's predicted-fingerprint quality and its retrieval MRR
on the same molecules, with the same protocol, and look at the relationship
between them.

If real Tanimoto gains convert into ranking gains, a retrain is the right
next move. If they do not, then prediction quality is not what limits
ranking, and more training against the same target is wasted GPU time --
the same shape of question as "is coverage the bottleneck?", asked before
building this time rather than after.

Run: PYTHONPATH=. python3 scripts/eval_tanimoto_vs_mrr.py
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

from src.candidates import FP_BITS, _fingerprint_one
from src.data import load_train
from src.pipeline_v2 import MASS_WINDOW_WIDEN_CAP, MASS_WINDOW_WIDEN_FACTOR, merge_spectra
from src.pipeline_v3 import load_fingerprint_model, predict_bit_probs
from src.propagation import MASS_WINDOW_DA, load_candidate_pool, mass_window, neutral_mass

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
EPS = 1e-4

CHECKPOINTS = [
    "fp_model.pt", "fp_model_v2.pt", "fp_model_v2_tan.pt",
    "fp_model_ft.pt", "fp_model_ft_tan.pt", "fp_model_ft2.pt", "fp_model_ft2_tan.pt",
    "peak_model.pt", "peak_model_tan.pt", "peak_model_l.pt", "peak_model_l_tan.pt",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")
    train = load_train()
    spec = importlib.util.spec_from_file_location("evalpc", ROOT / "scripts/eval_pubchem_pool.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    hard = helpers.hard_set(train)
    answers = hard.groupby("inchikey14")["normalized_smiles"].first().to_dict()

    # resolve every molecule's window and true fingerprint once
    cases = []
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
            continue
        bits = np.unpackbits(cand.view(np.uint8), axis=1)[:, :FP_BITS].astype(np.float64)
        cases.append((group, bits, int(hit[0]), true_bits))
    print(f"{len(cases)} molecules with the answer in their window "
          f"(median window {np.median([len(c[1]) for c in cases]):.0f})\n", flush=True)

    rows = []
    for name in CHECKPOINTS:
        if not (P / name).exists():
            continue
        t0 = time.time()
        model = load_fingerprint_model(str(P / name), device)
        tans, cos, ranks = [], [], []
        for group, bits, hit, true_bits in cases:
            p = np.clip(predict_bit_probs(model, group, device).mean(axis=0).astype(np.float64),
                        EPS, 1 - EPS)
            logit = np.log(p) - np.log1p(-p)
            s = bits @ logit
            ranks.append(int(1 + (s > s[hit]).sum()))
            pred_on, true_on = p >= 0.5, true_bits.astype(bool)
            union = (pred_on | true_on).sum()
            tans.append((pred_on & true_on).sum() / union if union else 0.0)
            # a threshold-free quality measure too, since Tanimoto@0.5 has
            # misled repeatedly on this project
            cos.append(float(p @ true_bits / (np.linalg.norm(p) * np.linalg.norm(true_bits) + 1e-12)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        r = np.array(ranks)
        mrr = np.mean([1.0 / x if x <= 25 else 0.0 for x in r])
        rows.append((name, float(np.mean(tans)), float(np.mean(cos)), mrr, (r == 1).mean()))
        print(f"  {name:24s} Tanimoto {rows[-1][1]:.4f}  cosine {rows[-1][2]:.4f}  "
              f"MRR {mrr:.4f}  top-1 {rows[-1][4]:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    t = np.array([r[1] for r in rows])
    c = np.array([r[2] for r in rows])
    m = np.array([r[3] for r in rows])
    print(f"\nacross {len(rows)} real checkpoints:")
    print(f"  Tanimoto spans {t.min():.4f}-{t.max():.4f}  ({100*(t.max()/t.min()-1):+.0f}%)")
    print(f"  cosine   spans {c.min():.4f}-{c.max():.4f}  ({100*(c.max()/c.min()-1):+.0f}%)")
    print(f"  MRR      spans {m.min():.4f}-{m.max():.4f}  ({100*(m.max()/m.min()-1):+.0f}%)")
    if len(rows) > 2:
        print(f"  corr(Tanimoto, MRR) = {np.corrcoef(t, m)[0,1]:+.3f}")
        print(f"  corr(cosine,   MRR) = {np.corrcoef(c, m)[0,1]:+.3f}")
        # the conversion rate that matters: how much MRR per unit Tanimoto
        slope = np.polyfit(t, m, 1)[0]
        print(f"  fitted slope: +0.01 Tanimoto -> {slope*0.01:+.4f} MRR")
        print(f"  for comparison, the interpolation oracle gave "
              f"+0.027 Tanimoto -> +0.363 MRR (i.e. +0.01 -> +0.134)")


if __name__ == "__main__":
    main()
