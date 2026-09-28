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


# Tier 2 takes no propagation signal at all. Swept on the hard set, the
# weighted score falls monotonically as tier-2 alpha rises -- 0.1907, 0.1894,
# 0.1865, 0.1785, 0.1768 at alpha 0, 0.15, 0.3, 0.5, 0.7 -- and the coverage
# half falls with it (0.0562 -> 0.0334). Propagation ranks a candidate by
# Tanimoto to a library spectral analog, and a molecule that needs tier 2 is
# precisely one with no analog in any library, so the signal is noise there;
# among ~4,650 mass-matched candidates that noise costs real ranks. Tier 1
# keeps ALPHA = 0.3, where a genuine analog usually does exist.
LARGE_ALPHA = 0.0

# How many of the 25 slots tier 1 may keep. Tier 1 fills all 25 for half the
# real test molecules, and whenever its pool lacks the answer those are 25
# guesses that cannot be right. Capping is nearly free because tier 1's deep
# ranks carry almost no value: measured at production density, going from 25
# to 5 costs the reachable half 0.3% (0.7121 -> 0.7098) -- when the curated
# pool holds the answer the ranker has it in the top 5 -- while the
# unreachable half gains 19% (0.0580 -> 0.0693).
#
#   cap          3       5       8      12      18      25
#   reachable  0.6978  0.7098  0.7098  0.7113  0.7121  0.7121
#   unreach.   0.0743  0.0693  0.0652  0.0623  0.0599  0.0580
#   weighted   0.3798  0.3832  0.3811  0.3803  0.3795  0.3785
#
# An earlier sweep found this knob flat, but it used a simulation that shrank
# the curated pool to make answers unreachable, which also halved tier 1's
# slot filling, so the cap barely bound. See scripts/eval_realistic_tiering.py.
TIER1_CAP = 5

# The 1 mDa window is instrument-limited, not distractor-limited: tightening it
# to 0.5 or 0.3 mDa loses more answers than it removes competitors (0.1851 and
# 0.1859 against 0.1907), and widening to 2 mDa is also worse (0.1869).
LARGE_MASS_WINDOW_DA = MASS_WINDOW_DA


def predict_molecule(rows, lib: Library, pool: CandidatePool, large: LargePool | None,
                     model, device, alpha=ALPHA, large_alpha=LARGE_ALPHA,
                     large_window=LARGE_MASS_WINDOW_DA, tier1_cap=TIER1_CAP) -> list:
    """Curated guesses first, then PubChem to fill the rest of the 25.

    The tier-2 blend weight and mass window are settable independently of
    tier 1's: both were tuned against a window holding ~49 candidates, and
    PubChem's holds ~4,650, so the balance between the propagation and model
    signals and the tolerance worth allowing need not be the same.
    """
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

    # Capping tier 1 only pays for itself if tier 2 can use the freed slots;
    # with no large pool, giving up 20 of 25 guesses would be a pure loss.
    cap = min(tier1_cap, N_GUESSES) if large is not None else N_GUESSES
    for smi in predict_curated(rows, lib, pool, model, device, alpha=alpha):
        key = to_inchikey14(smi) or smi
        if key in seen:
            continue
        seen.add(key)
        guesses.append(smi)
        if len(guesses) >= cap:
            break
    if len(guesses) >= N_GUESSES:
        return guesses
    if large is not None:
        add(rank_large_pool(rows, lib, pool, large, model, device,
                            alpha if large_alpha is None else large_alpha,
                            mass_window_da=large_window))
    return guesses[:N_GUESSES]


def predict_test_set(test_df, lib, pool, large, model, device, alpha=ALPHA, verbose=True,
                     fallback=None):
    """molecule_id -> guesses, one prediction per molecule, never raising.

    Every molecule ends up with at least one guess: an empty result would
    otherwise propagate to the submission writer, and one unguessable molecule
    should cost its own reciprocal rank, not the whole run.
    """
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
        if not out[mid] and fallback:
            out[mid] = list(fallback)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(groups)} molecules", flush=True)
    return out
