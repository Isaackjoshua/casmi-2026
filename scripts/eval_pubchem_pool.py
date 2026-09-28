"""Decide whether the 94M-structure PubChem pool is worth shipping, by
measuring the two effects it has -- separately, because they pull in
opposite directions.

  coverage  -- a real Class 2 molecule has no public spectrum anywhere, so
               it is NOT in any training library: its only route into the
               pool is COCONUT. Sampling "uncovered" molecules from the
               training data is therefore impossible (the pool contains
               every training structure by construction). Instead, the
               Class 2 situation is simulated by dropping training
               structures from the pool, leaving COCONUT-only -- under
               which the 61% of our hard-validation structures absent from
               COCONUT are exactly as unreachable as a real Class 2
               molecule. That subset is the coverage test.

  precision -- the existing 150-molecule hard validation set, which
               already has 100% pool coverage. PubChem takes a 1 mDa mass
               window from ~2 candidates to ~200-470 near the test's
               median mass, so the true answer must out-rank far more
               competition. This is the cost.

Ships only if the gain outweighs the cost. Both pools run through the
same scoring logic (a port of pipeline_v3.predict_molecule onto
LargePool, which memory-maps rather than loading 24 GB).

Run: PYTHONPATH=. python3 scripts/eval_pubchem_pool.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from src.baseline import FINE_TOP_N, N_GUESSES, build_library, score_spectrum
from src.candidates import FP_BITS
from src.data import load_train
from src.fingerprint_model import fingerprint_loglik_scores
from src.large_pool import load_large_pool, tanimoto_against, window
from src.metric import mrr_at_25, to_inchikey14
from src.pipeline_v2 import MASS_WINDOW_WIDEN_CAP, MASS_WINDOW_WIDEN_FACTOR, merge_spectra
from src.pipeline_v3 import MODEL_FLOOR, _normalize, load_fingerprint_model, predict_bit_probs
from src.pipeline_v3 import predict_molecule as predict_small
from src.propagation import MASS_WINDOW_DA, PROPAGATION_EXPONENT, neutral_mass

ROOT = Path(__file__).resolve().parents[1]
FIVE = ["enveda-180", "enveda-np-examples", "gnps", "riken", "pluskal_ms2"]


def predict_molecule_large(rows, lib, anchor_pool, pool, model, device, alpha=0.3,
                           mass_window_da=MASS_WINDOW_DA, exponent=PROPAGATION_EXPONENT):
    """pipeline_v3.predict_molecule against a LargePool. Anchors resolve
    against anchor_pool (the small in-memory pool: anchors are always
    training structures); candidates come from pool.
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

        lo, hi = window(pool, qmass, mass_window_da)
        widen = mass_window_da
        while hi <= lo and widen < MASS_WINDOW_WIDEN_CAP:
            widen *= MASS_WINDOW_WIDEN_FACTOR
            lo, hi = window(pool, qmass, widen)
        if hi <= lo:
            continue

        fused = np.zeros(hi - lo, dtype=np.float64)

        if alpha > 0:
            anchors = score_spectrum(
                merged["ms2_mzs"], merged["ms2_normalized_intensities"], float(merged["precursor_mz"]),
                str(merged["ionization_mode"]).strip().lower(), lib,
            )
            prop = np.zeros(hi - lo, dtype=np.float64)
            for key, _smi, sim in anchors:
                ai = anchor_pool.index_by_key.get(key)
                if ai is None:
                    continue
                t = tanimoto_against(pool, lo, hi, anchor_pool.fp_words[ai], int(anchor_pool.popcount[ai]))
                np.maximum(prop, (sim ** exponent) * t, out=prop)
            fused += alpha * _normalize(prop, 1.0 if prop.max() > 0 else 0.0)

        if alpha < 1:
            probs = predict_bit_probs(model, group, device).mean(axis=0)
            bits = np.unpackbits(np.asarray(pool.fp_words[lo:hi]).view(np.uint8), axis=1)[:, :FP_BITS]
            loglik = fingerprint_loglik_scores(probs, bits)
            fused += (1 - alpha) * (MODEL_FLOOR + (1 - MODEL_FLOOR) * _normalize(loglik, 1.0))

        top = np.argsort(fused)[::-1][:FINE_TOP_N]
        top = top[fused[top] > 0]
        for (key, smi), sc in zip(pool.rows(lo + top), fused[top]):
            if sc > best_score.get(key, 0.0):
                best_score[key] = float(sc)
                best_smiles[key] = smi

    final = {}
    for key in sorted(best_score, key=best_score.get, reverse=True)[:FINE_TOP_N]:
        smi = best_smiles[key]
        ck = to_inchikey14(smi) or key
        if ck not in final:
            final[ck] = smi
        if len(final) >= N_GUESSES:
            break
    return list(final.values())


def predict_hybrid(rows, lib, curated, large, model, device, alpha, tier1_cap=N_GUESSES):
    """The pool PubChem should actually ship as: curated candidates keep the
    ranks they already earn, and PubChem only fills the slots of the 25 they
    leave empty.

    The two pools are good at opposite things -- curated wins on molecules it
    contains (0.3854 vs 0.2052), PubChem is the only route to the ones it
    doesn't (0.1107 vs 0.0147) -- and the cost of the big pool is entirely
    that its distractors outrank true answers. Appending rather than merging
    spends none of that cost: a curated hit cannot be displaced by a PubChem
    candidate, because every PubChem candidate sits below all of them.
    """
    out, seen = [], set()
    for smi in predict_small(rows, lib, curated, model, device, alpha=alpha):
        k = to_inchikey14(smi) or smi
        if k not in seen:
            seen.add(k)
            out.append(smi)
        if len(out) >= tier1_cap:
            break
    if len(out) < N_GUESSES:
        for smi in predict_molecule_large(rows, lib, curated, large, model, device, alpha):
            k = to_inchikey14(smi) or smi
            if k not in seen:
                seen.add(k)
                out.append(smi)
            if len(out) >= N_GUESSES:
                break
    return out[:N_GUESSES]


def keys_present(meta_path, wanted):
    """Which of `wanted` the pool contains.

    Not `set(read_parquet(...)["inchikey14"])`: materializing 94M Python
    strings into a set cost ~25 minutes of CPU and 10 GB of RAM to answer a
    question about 150 keys. pyarrow's is_in scans the column in C against a
    150-element value set instead, and the answer is cached because it does
    not change between runs.
    """
    import json
    import pyarrow as pa
    import pyarrow.compute as pc

    cache = Path(meta_path).parent / "key_presence_cache.json"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    unknown = [k for k in wanted if k not in store]
    if unknown:
        value_set = pa.array(unknown, type=pa.string())
        found = set()
        pf = pq.ParquetFile(meta_path)
        for batch in pf.iter_batches(columns=["inchikey14"], batch_size=2_000_000):
            col = batch.column("inchikey14")
            found.update(col.filter(pc.is_in(col, value_set=value_set)).to_pylist())
            if len(found) == len(unknown):
                break
        store.update({k: (k in found) for k in unknown})
        cache.write_text(json.dumps(store))
    return {k for k in wanted if store.get(k)}


def hard_set(train):
    lib_keys = set(train[train["ingest_lib"].isin(FIVE)]["inchikey14"])
    novel = train[train["ingest_lib"].isin(["massbank", "mona"])]
    novel = novel[~novel["inchikey14"].isin(lib_keys)]
    keys = set(novel["inchikey14"].drop_duplicates().sample(n=150, random_state=42))
    return novel[novel["inchikey14"].isin(keys)]


def coconut_only_pool(full_pool, coconut_keys):
    """The pool a real Class 2 molecule actually faces: COCONUT, without
    the training structures it could never have been in.
    """
    from src.propagation import CandidatePool

    # Not np.isin: these keys are an object-dtype array of Python strings, for
    # which numpy falls back to an O(n*m) loop -- 34 minutes here, measured.
    # pandas hashes instead and returns the identical 479,717 rows in ~1s.
    # (np.isin against the *set* is worse than slow: it wraps the set in a 0-d
    # object array and silently matches nothing.)
    keep = pd.Index(full_pool.inchikey14).isin(coconut_keys)
    keys = full_pool.inchikey14[keep]
    return CandidatePool(
        inchikey14=keys,
        normalized_smiles=full_pool.normalized_smiles[keep],
        exact_mass=full_pool.exact_mass[keep],
        fp_words=full_pool.fp_words[keep],
        popcount=full_pool.popcount[keep],
        index_by_key={k: i for i, k in enumerate(keys)},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data/processed/peak_model_l_tan.pt"))
    ap.add_argument("--pubchem", default=str(ROOT / "data/pubchem/pubchem_pool"))
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--caps", default="3,5,8,12,25",
                    help="how many of the 25 slots tier 1 may keep")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fingerprint_model(args.model, device)
    train = load_train()
    lib = build_library(train[train["ingest_lib"].isin(FIVE)])

    from src.propagation import load_candidate_pool
    small = load_candidate_pool(ROOT / "data/processed/candidate_fingerprints.parquet")
    large = load_large_pool(args.pubchem)
    print(f"current pool: {len(small):,}   pubchem pool: {len(large):,}", flush=True)
    _t = time.time()

    coconut_keys = set(
        pd.read_parquet(ROOT / "data/coconut/coconut_structures.parquet", columns=["inchikey"])["inchikey"]
        .str.split("-").str[0]
    )
    baseline_pool = coconut_only_pool(small, coconut_keys)
    print(f"COCONUT-only pool (what a real Class 2 molecule faces): {len(baseline_pool):,}"
          f"  [{time.time()-_t:.0f}s]", flush=True)
    _t = time.time()

    hard = hard_set(train)
    keys = hard["inchikey14"].unique()
    in_coconut = set(k for k in keys if k in coconut_keys)
    missing = [k for k in keys if k not in coconut_keys]
    pubchem_keys = keys_present(Path(args.pubchem) / "meta.parquet", missing)
    recovered = [k for k in missing if k in pubchem_keys]
    print(f"hard set: {len(keys)} molecules -- {len(in_coconut)} in COCONUT (reachable today), "
          f"{len(keys)-len(in_coconut)} not; PubChem recovers {len(recovered)} of those"
          f"  [{time.time()-_t:.0f}s]", flush=True)

    subsets = {
        "precision": hard[hard["inchikey14"].isin(in_coconut)],           # reachable either way
        "coverage": hard[~hard["inchikey14"].isin(in_coconut)],           # unreachable without PubChem
    }
    caps = [int(c) for c in args.caps.split(",")]
    scores, sizes = {}, {}
    for name, sample in subsets.items():
        if not len(sample):
            continue
        answers = sample.groupby("inchikey14")["normalized_smiles"].first().to_dict()
        groups = list(sample.groupby("inchikey14"))
        sizes[name] = len(groups)
        print(f"\n[{name}] {len(groups)} molecules", flush=True)

        def report(label, preds, t0):
            m = mrr_at_25(preds, answers)
            scores[(name, label)] = m
            print(f"  {label:22s}: MRR@25={m:.4f} ({time.time()-t0:.0f}s)", flush=True)

        t0 = time.time()
        report("COCONUT only", {k: predict_small(g.to_dict("records"), lib, baseline_pool, model,
                                                 device, alpha=args.alpha) for k, g in groups}, t0)
        t0 = time.time()
        report("COCONUT+PubChem", {k: predict_molecule_large(g.to_dict("records"), lib, small, large,
                                                             model, device, args.alpha) for k, g in groups}, t0)
        for cap in caps:
            t0 = time.time()
            report(f"tiered (cap {cap})",
                   {k: predict_hybrid(g.to_dict("records"), lib, baseline_pool, large, model, device,
                                      args.alpha, tier1_cap=cap) for k, g in groups}, t0)

    # what matters is the whole hard set: the subsets are a partition of it, so
    # weight each by its size. A strategy only ships if this column improves.
    total = sum(sizes.values())
    labels = ["COCONUT only", "COCONUT+PubChem"] + [f"tiered (cap {c})" for c in caps]
    print(f"\n=== weighted over all {total} molecules ===", flush=True)
    for label in labels:
        if all((n, label) in scores for n in sizes):
            w = sum(scores[(n, label)] * sizes[n] for n in sizes) / total
            print(f"  {label:22s}: MRR@25={w:.4f}", flush=True)


if __name__ == "__main__":
    main()
