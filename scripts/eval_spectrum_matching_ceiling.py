"""Before building a spectrum predictor, measure what a perfect one is worth.

Forward simulation (CFM-ID / ICEBERG style) would predict a spectrum for
every candidate and compare it with the query, instead of compressing
structure into a fingerprint. The premise is that spectra discriminate
between mass-window competitors better than fingerprints do. Two phases
have now been spent on premises of that shape, so this one gets tested
first.

The ceiling is measurable without training anything: substitute a *real*
library spectrum for the predicted one. A perfect predictor would output
something like a real measurement, so ranking candidates by similarity to
their own real spectra is the best any predictor could do, and more.

Design, to keep it honest:
  - a molecule qualifies only if it has spectra from two different
    ingest_lib sources. One is the query; the other stands in for the
    prediction. Using the same spectrum for both would score a trivial 1.0.
  - competitors are the candidates in the same ~1 mDa mass window that also
    have a library spectrum, scored the same way.
  - the fingerprint ranker is scored on exactly the same molecules and the
    same competitor sets, so the two are directly comparable.

A low ceiling kills the approach outright. A high one only establishes the
ceiling -- the conversion from predictor quality to ranking may be as
convex as it turned out to be for fingerprints (scripts/eval_tanimoto_vs_mrr.py).

Run: PYTHONPATH=. python3 scripts/eval_spectrum_matching_ceiling.py
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.candidates import FP_BITS
from src.data import load_train
from src.pipeline_v3 import load_fingerprint_model, predict_bit_probs
from src.propagation import MASS_WINDOW_DA, load_candidate_pool, mass_window, neutral_mass
from src.spectral_similarity import modified_cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
EPS = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")
    train = load_train()

    # index every library spectrum by structure, remembering its source
    by_key = defaultdict(list)
    for r in train.itertuples(index=False):
        by_key[r.inchikey14].append(r)

    multi = [k for k, v in by_key.items() if len({x.ingest_lib for x in v}) >= 2]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(multi, size=min(args.molecules, len(multi)), replace=False)
    print(f"{len(multi):,} structures have spectra from >=2 sources; sampling {len(chosen)}",
          flush=True)

    model = load_fingerprint_model([str(P / "fp_model.pt"), str(P / "peak_model.pt")], device)

    spec_ranks, fp_ranks, n_comp = [], [], []
    t0 = time.time()
    for key in chosen:
        rows = by_key[key]
        sources = sorted({x.ingest_lib for x in rows})
        q_src = sources[0]
        queries = [x for x in rows if x.ingest_lib == q_src]
        refs = [x for x in rows if x.ingest_lib != q_src]
        query = max(queries, key=lambda x: len(x.ms2_mzs))

        qmass = neutral_mass(float(query.precursor_mz), query.adduct)
        if qmass is None:
            continue
        lo, hi = mass_window(pool, qmass, MASS_WINDOW_DA)
        if hi - lo < 2:
            continue

        # competitors: window candidates that have a library spectrum of their
        # own, excluding any spectrum from the query's own source so the true
        # structure gets no unfair advantage
        cand_keys = list(pool.inchikey14[lo:hi])
        scored, fp_rows = [], []
        for ck in cand_keys:
            pool_rows = [x for x in by_key.get(ck, []) if x.ingest_lib != q_src]
            if not pool_rows:
                continue
            best = 0.0
            for cr in pool_rows:
                best = max(best, modified_cosine_similarity(
                    np.asarray(query.ms2_mzs), np.asarray(query.ms2_normalized_intensities),
                    float(query.precursor_mz),
                    np.asarray(cr.ms2_mzs), np.asarray(cr.ms2_normalized_intensities),
                    float(cr.precursor_mz)))
            scored.append((ck, best))
            fp_rows.append(ck)
        if len(scored) < 2 or key not in [c for c, _ in scored]:
            continue
        if key not in [x for x in fp_rows]:
            continue

        s_true = dict(scored)[key]
        spec_ranks.append(1 + sum(1 for _c, v in scored if v > s_true))

        # the fingerprint ranker on exactly the same competitor set
        idx = [pool.index_by_key[c] for c in fp_rows]
        bits = np.unpackbits(np.asarray(pool.fp_words[idx]).view(np.uint8), axis=1)[:, :FP_BITS].astype(np.float64)
        p = np.clip(predict_bit_probs(model, [query._asdict()], device).mean(axis=0).astype(np.float64),
                    EPS, 1 - EPS)
        f = bits @ (np.log(p) - np.log1p(-p))
        ti = fp_rows.index(key)
        fp_ranks.append(1 + int((f > f[ti]).sum()))
        n_comp.append(len(scored))

        if len(spec_ranks) % 50 == 0:
            print(f"  {len(spec_ranks)} scored ({time.time()-t0:.0f}s)", flush=True)

    def mrr(rs):
        return float(np.mean([1.0 / r if r <= 25 else 0.0 for r in rs]))

    sr, fr = np.array(spec_ranks), np.array(fp_ranks)
    print(f"\n{len(sr)} molecules; median {np.median(n_comp):.0f} competitors with spectra "
          f"({time.time()-t0:.0f}s)\n")
    print(f"{'ranker':32s} {'MRR@25':>8} {'top-1':>7} {'median rank':>12}")
    print(f"{'spectrum match (perfect oracle)':32s} {mrr(sr):8.4f} {(sr==1).mean():7.3f} "
          f"{np.median(sr):12.0f}")
    print(f"{'fingerprint (current model)':32s} {mrr(fr):8.4f} {(fr==1).mean():7.3f} "
          f"{np.median(fr):12.0f}")
    print(f"\nthe oracle is what a *perfect* predictor would achieve, and more: it uses a "
          f"real measurement of each candidate.")


if __name__ == "__main__":
    main()
