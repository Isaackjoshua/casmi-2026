"""Phase 4: a tiered candidate pool.

The 94M-structure PubChem pool was measured two ways on the 150-molecule
hard validation set, splitting it by whether COCONUT contains the answer
(see scripts/eval_pubchem_pool.py):

                      curated pool    +PubChem merged
  reachable (59)         0.3854           0.2052
  unreachable (91)       0.0147           0.1107

Merging the pools is a net loss: PubChem's coverage gain is worth less
than what its 200-470 mass-matched distractors cost the molecules we
could already find. But the two columns win in disjoint places, and the
cost is *only* displacement -- a true answer outranked by distractors.

So PubChem is consulted, never merged: curated candidates keep the ranks
they already earn, and PubChem fills only the slots of the 25 they leave
empty. A curated hit cannot be displaced, because every PubChem candidate
sits strictly below all of them; a molecule absent from the curated pool
used to be unrankable and now lands somewhere in the tail.
"""

from __future__ import annotations

import numpy as np
import torch

from .baseline import FINE_TOP_N, N_GUESSES, Library, score_spectrum
from .candidates import FP_BITS
from .fingerprint_model import fingerprint_loglik_scores
from .large_pool import LargePool, tanimoto_matrix, window
from .metric import to_inchikey14
from .pipeline_v2 import MASS_WINDOW_WIDEN_CAP, MASS_WINDOW_WIDEN_FACTOR, merge_spectra
from .pipeline_v3 import ALPHA, MODEL_FLOOR, _normalize, predict_bit_probs
from .pipeline_v3 import predict_molecule as predict_curated
from .propagation import MASS_WINDOW_DA, PROPAGATION_EXPONENT, CandidatePool, neutral_mass


def rank_large_pool(rows, lib, anchor_pool, pool, model, device, alpha=ALPHA,
                    mass_window_da=MASS_WINDOW_DA, exponent=PROPAGATION_EXPONENT):
    """pipeline_v3's scoring against a LargePool instead of an in-memory one.

    Anchors resolve against anchor_pool: propagation needs an anchor's own
    fingerprint, anchors are always training-library structures, and the
    LargePool deliberately has no key index (see src/large_pool.py).
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

        # one read of the window, reused by both signals
        cand_words = np.asarray(pool.fp_words[lo:hi])
        cand_pop = np.asarray(pool.popcount[lo:hi], dtype=np.int64)
        fused = np.zeros(hi - lo, dtype=np.float64)

        if alpha > 0:
            anchors = score_spectrum(
                merged["ms2_mzs"], merged["ms2_normalized_intensities"], float(merged["precursor_mz"]),
                str(merged["ionization_mode"]).strip().lower(), lib,
            )
            a_words, a_pop, sims = [], [], []
            for key, _smi, sim in anchors:
                ai = anchor_pool.index_by_key.get(key)
                if ai is None:
                    continue
                a_words.append(anchor_pool.fp_words[ai])
                a_pop.append(int(anchor_pool.popcount[ai]))
                sims.append(sim)
            if a_words:
                tan = tanimoto_matrix(cand_words, cand_pop, np.stack(a_words), np.array(a_pop))
                prop = (tan * (np.asarray(sims, dtype=np.float64) ** exponent)[None, :]).max(axis=1)
            else:
                prop = np.zeros(hi - lo, dtype=np.float64)
            fused += alpha * _normalize(prop, 1.0 if prop.max() > 0 else 0.0)

        if alpha < 1:
            probs = predict_bit_probs(model, group, device).mean(axis=0)
            bits = np.unpackbits(cand_words.view(np.uint8), axis=1)[:, :FP_BITS]
            loglik = fingerprint_loglik_scores(probs, bits)
            fused += (1 - alpha) * (MODEL_FLOOR + (1 - MODEL_FLOOR) * _normalize(loglik, 1.0))

        top = np.argsort(fused)[::-1][:FINE_TOP_N]
        top = top[fused[top] > 0]
        for (key, smi), sc in zip(pool.rows(lo + top), fused[top]):
            if sc > best_score.get(key, 0.0):
                best_score[key] = float(sc)
                best_smiles[key] = smi

    out = []
    for key in sorted(best_score, key=best_score.get, reverse=True)[:FINE_TOP_N]:
        out.append(best_smiles[key])
    return out


def predict_molecule(rows, lib: Library, pool: CandidatePool, large: LargePool | None,
                     model, device, alpha=ALPHA) -> list:
    """Curated guesses first, then PubChem to fill the rest of the 25."""
    guesses, seen = [], set()

    def add(smiles_iter):
        for smi in smiles_iter:
            key = to_inchikey14(smi) or smi
            if key in seen:
                continue
            seen.add(key)
            guesses.append(smi)
            if len(guesses) >= N_GUESSES:
                return True
        return False

    if add(predict_curated(rows, lib, pool, model, device, alpha=alpha)):
        return guesses
    if large is not None:
        add(rank_large_pool(rows, lib, pool, large, model, device, alpha))
    return guesses[:N_GUESSES]


def predict_test_set(test_df, lib, pool, large, model, device, alpha=ALPHA, verbose=True):
    """molecule_id -> guesses, one prediction per molecule, never raising."""
    out = {}
    groups = list(test_df.groupby("molecule_id"))
    for i, (mid, rows) in enumerate(groups):
        try:
            out[mid] = predict_molecule(rows.to_dict("records"), lib, pool, large, model, device, alpha)
        except Exception as exc:  # a single bad molecule must not lose the submission
            print(f"  molecule {mid} failed ({type(exc).__name__}: {exc}); falling back to curated-only")
            try:
                out[mid] = predict_curated(rows.to_dict("records"), lib, pool, model, device, alpha=alpha)
            except Exception:
                out[mid] = []
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(groups)} molecules", flush=True)
    return out
