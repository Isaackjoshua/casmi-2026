"""Score every multi-fingerprint checkpoint on the retrieval task itself.

Checkpoints on this project have been selected by validation loss and by
Tanimoto@0.5, and both have pointed the wrong way more than once; the
currently deployed one was chosen by retrieval MRR after the fact. So the
multi-fingerprint run saves every epoch and this picks between them on
MRR, paired per molecule against the shipped configuration.

Two ways to use a multi-fingerprint model are measured, because they are
not the same bet:

  multi alone   score candidates on all 3239 bits from the new model.
                Clean, but it throws away the Morgan-r2 predictor that two
                confirmed submissions were built on.

  hybrid        take the Morgan-r2 block from the shipped ensemble
                (fp_model + peak_model_l_tan) and the Morgan-r3 and MACCS
                blocks from the new model, and sum the log-likelihoods.
                The score is a sum over bits, so blocks from different
                models compose without any rescaling -- this keeps the
                known-good ranking and adds only the orthogonal views,
                which is the whole premise of the retrain.

Run: PYTHONPATH=. python3 scripts/select_multi_checkpoint.py --n 600
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.baseline import FINE_TOP_N, N_GUESSES, build_library, score_spectrum
from src.candidates import FP_BITS
from src.data import load_train
from src.fingerprint_model import fingerprint_loglik_scores
from src.metric import to_inchikey14
from src.multi_fingerprint import OFFSETS, TOTAL_BITS
from src.pipeline_v2 import MASS_WINDOW_WIDEN_CAP, MASS_WINDOW_WIDEN_FACTOR, merge_spectra
from src.pipeline_v3 import (ALPHA, MODEL_FLOOR, _normalize, candidate_bits,
                            load_fingerprint_model, predict_bit_probs)
from src.propagation import (MASS_WINDOW_DA, PROPAGATION_EXPONENT, load_candidate_pool,
                             mass_window, neutral_mass, propagation_scores)

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]
ALL_LIBS = FIVE + ["massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"]
SHIPPED = ["fp_model.pt", "peak_model_l_tan.pt"]
M2_LO, M2_HI = OFFSETS["morgan2"]


def _z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def predict_molecule_scored(rows, lib, pool, mpacked, device, alpha,
                            base_model=None, multi_model=None, mode="shipped",
                            block_weight=None):
    """pipeline_v3.predict_molecule, with the model term swapped per `mode`.

    mode "shipped" reproduces the deployed scorer exactly (Morgan r2 only,
    from base_model). "multi" scores all 3239 bits from multi_model.
    "hybrid" takes the Morgan-r2 block from base_model and the remaining
    blocks from multi_model.
    """
    best_score, best_smiles = {}, {}
    by_adduct = {}
    for r in rows:
        by_adduct.setdefault(r["adduct"], []).append(r)

    for group in by_adduct.values():
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

        fused = np.zeros(hi - lo, dtype=np.float64)
        if alpha > 0:
            anchors = score_spectrum(
                merged["ms2_mzs"], merged["ms2_normalized_intensities"],
                float(merged["precursor_mz"]),
                str(merged["ionization_mode"]).strip().lower(), lib,
            )
            if anchors:
                prop = propagation_scores(anchors, lo, hi, pool, PROPAGATION_EXPONENT)
                fused += alpha * _normalize(prop, 1.0 if prop.max() > 0 else 0.0)

        if alpha < 1:
            if mode == "shipped":
                probs = predict_bit_probs(base_model, group, device).mean(axis=0)
                loglik = fingerprint_loglik_scores(probs, candidate_bits(pool, lo, hi))
            else:
                mbits = np.unpackbits(mpacked[lo:hi], axis=1)[:, :TOTAL_BITS]
                pm = predict_bit_probs(multi_model, group, device).mean(axis=0)
                if mode == "multi":
                    loglik = fingerprint_loglik_scores(pm, mbits)
                else:  # hybrid, optionally with per-block standardization
                    pb = predict_bit_probs(base_model, group, device).mean(axis=0)
                    l_m2 = fingerprint_loglik_scores(pb, mbits[:, M2_LO:M2_HI])
                    l_rest = fingerprint_loglik_scores(pm[M2_HI:], mbits[:, M2_HI:])
                    if block_weight is None:
                        loglik = l_m2 + l_rest
                    else:
                        # Summing raw log-likelihoods lets whichever block has
                        # the larger spread dominate, and MACCS is both dense
                        # (28% of bits on) and the best-predicted block, so it
                        # can swamp the Morgan-r2 signal two submissions were
                        # built on. Standardizing each block within the window
                        # makes the weight, not the scale, decide.
                        loglik = _z(l_m2) + block_weight * _z(l_rest)
            fused += (1 - alpha) * (MODEL_FLOOR + (1 - MODEL_FLOOR) * _normalize(loglik, 1.0))

        top = np.argsort(fused)[::-1][:FINE_TOP_N]
        top = top[fused[top] > 0]
        for j in top:
            key = pool.inchikey14[lo + j]
            if fused[j] > best_score.get(key, 0.0):
                best_score[key] = float(fused[j])
                best_smiles[key] = pool.normalized_smiles[lo + j]

    final = {}
    for key in sorted(best_score, key=best_score.get, reverse=True)[:FINE_TOP_N]:
        smi = best_smiles[key]
        ck = to_inchikey14(smi) or key
        if ck not in final:
            final[ck] = smi
        if len(final) >= N_GUESSES:
            break
    return list(final.values())


def reciprocal_rank(guesses, target):
    for i, s in enumerate(guesses[:25], 1):
        if (to_inchikey14(s) or s) == target:
            return 1.0 / i
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--epochs", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--weights", default="0.0,0.25,0.5,1.0",
                    help="weight on the standardized morgan3+maccs blocks; 0 recovers shipped")
    ap.add_argument("--sources", default="massbank,mona",
                    help="which libraries to draw the held-out molecules from. The "
                         "massbank/mona pool is only 2095 structures, so asking for more "
                         "than that silently returns all of them and --seed stops doing "
                         "anything -- which turned one measurement into a fake pair. "
                         "spectraverse gives 9,537 structures disjoint from massbank/mona, "
                         "so it is a genuinely independent confirmation set.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_train()
    lib_keys = set(train[train["ingest_lib"].isin(FIVE)]["inchikey14"])
    srcs = [x.strip() for x in args.sources.split(",")]
    novel = train[train["ingest_lib"].isin(srcs)]
    novel = novel[~novel["inchikey14"].isin(lib_keys)]
    if args.sources != "massbank,mona":
        # keep confirmation sets disjoint from the massbank/mona one
        used = set(train[train["ingest_lib"].isin(["massbank", "mona"])]["inchikey14"]) - lib_keys
        novel = novel[~novel["inchikey14"].isin(used)]
    avail = novel["inchikey14"].nunique()
    if args.n > avail:
        print(f"NOTE: asked for {args.n} but only {avail} are available from "
              f"{args.sources}; --seed has no effect when the whole set is taken", flush=True)
    keys = set(novel["inchikey14"].drop_duplicates().sample(
        n=min(args.n, avail), random_state=args.seed))
    sample = novel[novel["inchikey14"].isin(keys)]
    scored = set(sample["inchikey14"])
    lib = build_library(train[train["ingest_lib"].isin(ALL_LIBS)
                              & ~train["inchikey14"].isin(scored)])
    pool = load_candidate_pool(P / "candidate_fingerprints.parquet")
    mpacked = np.load(P / "pool_multi_fp.npy")
    groups = list(sample.groupby("inchikey14"))
    print(f"{len(groups)} molecules; pool multi-fp {mpacked.shape}\n", flush=True)

    base_model = load_fingerprint_model([str(P / m) for m in SHIPPED], device)
    t0 = time.time()
    base = {k: reciprocal_rank(
        predict_molecule_scored(g.to_dict("records"), lib, pool, mpacked, device, ALPHA,
                                base_model=base_model, mode="shipped"), k)
        for k, g in groups}
    print(f"shipped (Morgan r2 only): MRR@25 {np.mean(list(base.values())):.4f} "
          f"({time.time()-t0:.0f}s)\n", flush=True)

    print(f"{'checkpoint':22s} {'mode':8s} {'MRR':>7} {'paired diff':>12} {'95% CI':>9}  verdict")
    for e in [int(x) for x in args.epochs.split(",")]:
        path = P / f"multi_model_e{e}.pt"
        if not path.exists():
            continue
        mm = load_fingerprint_model(str(path), device)
        modes = [("multi", None), ("hybrid", None)]
        modes += [(f"w={w}", w) for w in [float(x) for x in args.weights.split(",")]]
        for label, bw in modes:
            mode = "multi" if label == "multi" else "hybrid"
            t0 = time.time()
            res = {k: reciprocal_rank(
                predict_molecule_scored(g.to_dict("records"), lib, pool, mpacked, device, ALPHA,
                                        base_model=base_model, multi_model=mm, mode=mode,
                                        block_weight=bw), k)
                for k, g in groups}
            d = np.array([res[k] - base[k] for k, _ in groups])
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            sig = abs(d.mean()) > ci
            verdict = ("BETTER" if d.mean() > 0 else "worse") if sig else "ns"
            print(f"{'multi_model_e'+str(e):22s} {label:8s} "
                  f"{np.mean(list(res.values())):7.4f} {d.mean():+12.4f} {ci:9.4f}  {verdict}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        del mm
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
