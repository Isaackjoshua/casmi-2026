# Enveda CASMI 2026 — Molecule ID from Mass Spectra

Kaggle competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra

Full roadmap (phases, technical approach comparison, risks): see the project doc
shared in chat (Claude Artifact "Enveda CASMI 2026 Roadmap").

## Task

Predict up to 25 ranked SMILES candidates per `molecule_id` from LC-MS/MS spectra.
Scored with Mean Reciprocal Rank @ 25 (MRR@25), matched via RDKit tautomer
canonicalization + InChIKey14 (connectivity only, stereochemistry ignored).

- Start: 2026-09-14
- Entry / team-merge deadline: 2026-12-07
- Final submission deadline: 2026-12-14
- Max 5 submissions/day, 2 final submissions selected

## Project layout

```
casmi-2026/
├── data/
│   ├── raw/          # untouched train.parquet, test.parquet, sample_submission.csv
│   └── processed/    # cleaned/curated spectra, cached features
├── notebooks/         # exploration notebooks
├── src/               # reusable pipeline code (loading, similarity, scoring, submission)
├── submissions/        # generated submission.csv files, timestamped
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtualenv / conda env and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up the Kaggle API (one-time, manual — see below), then download the data:
   ```bash
   bash scripts/download_data.sh
   ```

### Kaggle API setup (manual step — do this yourself)

1. Go to https://www.kaggle.com/settings → **API** section → **Create New Token**.
   This downloads a `kaggle.json` file containing your username and API key.
2. Move it into place and lock down permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. On the competition page, click **Join Competition** and accept the rules
   (required before the API will let you download data or submit —
   this has to be done by you, logged in, in a real browser).
4. Verify it works:
   ```bash
   kaggle competitions list -s casmi
   ```

## Current status

- [x] Kaggle API configured, rules accepted
- [x] Data downloaded (train: 2,539,608 spectra; test: 1,213 spectra / 400 molecules)
- [x] Baseline pipeline reproduced — see `src/baseline.py`. Coarse sparse-cosine
      pre-filter over a deduplicated library (~500k unique structure/adduct
      spectra from `enveda-180`, `enveda-np-examples`, `gnps`, `riken`,
      `pluskal_ms2`) + exact modified-cosine rescoring of the top candidates,
      max-pooled across a molecule's spectra, deduplicated on the
      tautomer-canonical InChIKey14 the competition actually scores against.
      Full test set runs in ~1-2 minutes (well under the 9h Kaggle limit).
      **Local validation MRR@25: 0.4454** (held out `enveda-np-examples`, the
      library closest to the test distribution, as a pseudo-test set with
      known ground truth — optimistic vs. the real hidden test set, since
      these are common compounds that also appear elsewhere in training, but
      confirms the pipeline mechanics are correct). First real submission
      written to `submissions/baseline_v1.csv`, validated against
      `sample_submission.csv`. **Real public leaderboard score: 0.063** —
      badly under even the ~0.15-0.23 "library-only ceiling" public notebooks
      reported. A harder local validation (structures entirely absent from
      the library, so the real Class 2/3 failure mode) confirmed why:
      **0.0000 MRR@25** — the pipeline never abstains, it confidently guesses
      wrong every time a structure isn't literally already in the library.
- [x] Candidate-pool expansion (Phase 2, in progress) — see `src/candidates.py`,
      `src/propagation.py`, `src/pipeline_v2.py`. Expands guesses beyond the
      training library to COCONUT (~729k combined candidates, natural
      products + training structures, fingerprinted and cached), ranking
      candidates via `max_a sim(anchor)^4 * Tanimoto(candidate, anchor)`
      against spectrally-similar library analogs (the recipe public
      notebooks converged on). Two validated fixes on the same hard,
      structure-absent-from-library validation set:
      - Candidate expansion alone: 0.0000 -> 0.0145 MRR@25.
      - + entropy-weighted similarity (Li et al. 2021, replacing plain
        cosine — diagnosed directly here: plain cosine gave "0.99 similar"
        anchors that were chemically unrelated, Tanimoto ~0.1, because a
        single shared dominant fragment peak dominates raw cosine
        regardless of the rest of the spectrum): 0.0145 -> **0.0211**.
      First submission (window=150 Da, exponent=4, no spectrum merging):
      real public leaderboard score **0.075** (up from Phase 1's 0.063) — a
      real, positive move in the direction the local hard validation
      predicted, though smaller in absolute terms (local: +0.0211,
      real: +0.012). Encouraging as a sanity check on the validation
      methodology, but still well under research-suggested ceilings.

      **Follow-up tuning found the mass window was badly miscalibrated.**
      A local parameter sweep on the same hard-validation set found scores
      *monotonically improving* as the window shrank — the "generous
      analog window" was diluting every ranking with tens of thousands of
      same-mass-region-but-chemically-irrelevant candidates. Combined with
      merging a molecule's multi-collision-energy spectra into one
      consensus spectrum before anchor search (`pipeline_v2.merge_spectra`,
      grouped by adduct), the tuned defaults
      (`MASS_WINDOW_DA=0.3`, `PROPAGATION_EXPONENT=3.0`, both in
      `src/propagation.py`) score **0.1725 MRR@25** on the full 150-molecule
      hard validation set — up from 0.0211, an 8x improvement — and run in
      56s instead of 1120s (20x faster, since far fewer candidates get
      scored per query). Resubmitted (kernel v5) 2026-09-18: **real public
      leaderboard score 0.138** — more than double the original 0.063
      baseline, and ~1.8x the previous Phase 2 submission (0.075). Real-world
      gain (0.075 -> 0.138, +0.063) is smaller in relative terms than the
      local validation jump (0.0211 -> 0.1725, ~8x) but large and
      unambiguous either way — strong confirmation the tuned mass window
      and spectrum merging generalize to the real hidden test set.

      **Kept sweeping past 0.3 Da — the window isn't an analog-hop
      tolerance.** Candidates are always matched against the query's own
      measured neutral mass, so tightening the window just means trusting
      the instrument's mass accuracy more. Score kept improving
      monotonically all the way to ~1 mDa (`MASS_WINDOW_DA=0.001`,
      `PROPAGATION_EXPONENT=2.5`: **0.4392 MRR@25**), the real accuracy
      floor — tighter than that and real measurement noise starts
      excluding true answers. Added adaptive widening for spectra that
      land with zero candidates at that width (retry at 10x wider, up to
      a 1 Da cap, before the frequency fallback): cut zero-hit spectra
      from 16/150 to 6/150 and pushed the validation score to **0.4586**
      — within reach of the 0.4454 measured on the much easier "common
      compounds" holdout from Phase 1, but on a validation set built
      specifically to be hard.

      **Kernel v6 (these settings) crashed on submit**: ran cleanly in
      manual testing (public test set) but threw an unhandled exception
      when Kaggle re-ran it against the actual hidden test set. Most
      likely cause: `build_submission`'s `sample_submission.csv` coverage
      check used to hard-`raise` on any molecule_id mismatch, which only
      passed locally because the public test set's ids happen to match
      the public sample file. Fixed in three layers (see commit
      `3a25a55`): the coverage check now warns instead of raising,
      `predict_test_set` wraps each molecule's prediction in try/except
      with a frequency-fallback on any failure, and the notebook's
      outermost cell has one more try/except around the whole prediction
      step as a last resort. Resubmitted (kernel v7): ran without error
      and scored **real public leaderboard score 0.194** — our best score
      yet, confirming both that the tight-window tuning generalizes to
      the real hidden test set and that the defensive fixes resolved
      whatever crashed v6.

      **Score trajectory this session: 0.063 → 0.075 → 0.138 → (crash) →
      0.194** — a 3x improvement from the original baseline.

      **Kernel v8 (all 11 training libraries as anchors): 0.193 — flat.**
      Expanding the anchor library from 5 to all 11 sources lifted the
      local hard validation from 0.4586 to 0.4878–0.4914 (validated with
      two independent held-out pairs), but that did *not* transfer to the
      real hidden test set (0.194 → 0.193, noise). Instructive: the local
      validation tracked the big structural changes well (mass-window
      tuning: huge local gain → big real gain) but not this subtler one.
      Likely reason: the held-out compounds come from *other public
      libraries*, so adding more public libraries naturally helps find
      their analogs — but the real test set is Enveda's own timsTOF
      measurements of specific natural products, where extra public-
      library coverage adds little. This is the instrument/chemistry
      domain shift flagged in the original roadmap, showing up as a
      local-vs-real gap (~0.49 local vs ~0.19 real) that library-count
      tuning can't close. Two real submissions now plateaued at ~0.19,
      suggesting this is near the ceiling of the pure
      "propagate-from-spectral-analogs" approach.
- [x] Learned spectrum -> fingerprint model (Phase 3) — see
      `src/fingerprint_model.py`, `src/pipeline_v3.py`,
      `scripts/build_fp_training_data.py`, `scripts/train_fingerprint_model.py`.
      An MLP (61.8M params) over 0.2 Da fragment bins + neutral-loss bins +
      precursor/ion-mode features, predicting the 2048-bit Morgan fingerprint
      with BCE; trained on 2.49M spectra split by structure (hard-validation
      structures excluded), 11 min on an RTX A4000, held-out Tanimoto@0.5
      ~0.31. Candidates ranked by fingerprint log-likelihood under the
      predicted bit probabilities (one matrix multiply), fused with
      propagation at alpha=0.3. Hard validation: propagation-only 0.4456,
      model-only 0.4836, fused **0.4932**. **Real public leaderboard score:
      0.214** (up from 0.194) — broke the ~0.19 plateau two Phase 2
      configurations hit, confirming the model transfers to the timsTOF
      test data where analog propagation stalled. Model checkpoint shipped
      as a private Kaggle dataset (`isaackjoshua/casmi26-fp-model`).

      **Score trajectory: 0.063 → 0.075 → 0.138 → 0.194 → 0.193 → 0.214.**
- [x] Fingerprint model improvement round 1 — see `scripts/eval_pipeline.py`
      (dual validation: the massbank/mona hard set + a new 200-structure
      enveda-180 timsTOF holdout from the model's own structure split).
      The two sets disagree on which signal wins (propagation dominates
      on held-out drug-like timsTOF structures, the model on natural
      products); the real test's ordering matches the natural-product set,
      so alpha stays 0.3. Tried: heavier regularization (a wash), timsTOF-
      only fine-tuning (+11% timsTOF model-only, −3% natural products, and
      it *lowered* val Tanimoto@0.5 — that threshold proxy is misleading
      for a log-likelihood ranker), ensembles of similar models (no gain).
      **Winner: fine-tune v1 on timsTOF + natural-product libraries
      (`enveda-180`, `enveda-np-examples`, `gnps`, `riken`), 3 epochs at
      lr 2e-4, best-Tanimoto checkpoint (`fp_model_ft2_tan.pt`)** — a
      strict improvement over the deployed model on every metric:
      hard α=0.3 0.4932→0.4998, timsTOF α=0.3 0.5661→0.5838, timsTOF
      model-only 0.3966→0.4339. Uploaded as v2 of the model dataset;
      kernel rebuilt, awaiting the daily submission quota reset.
- [ ] Next structural step: a peak-level transformer (the host tutorial's
      approach) to replace 0.2 Da binning — all MLP variants cluster in a
      narrow band, suggesting the binned input is the ceiling.
- [ ] Class 3 de novo exploration
- [ ] Final ensemble + validation

## Design notes / known limitations

- Library dedup keeps one representative spectrum per (structure, adduct) —
  picking the one with the most peaks. (Phase 2's `pipeline_v2.merge_spectra`
  does merge a *query* molecule's multiple collision-energy spectra before
  anchor search; the *library* side is still one spectrum per structure/adduct.)
- The frequency-based fallback (`baseline._global_fallback_candidates`) only
  fires when a spectrum gets zero similarity hits; `predict_test_set` prints
  a warning with the count when it happens, worth checking each run.
- `PEAK_TOP_K`, `BIN_WIDTH`, `COARSE_TOP_K`, `ANALOG`-related bonus constants
  in `src/baseline.py` are initial guesses, not tuned — a natural next step
  once a real local CV split (not just the np-examples holdout) is built.
