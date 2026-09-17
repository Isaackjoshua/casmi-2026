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
      `sample_submission.csv`.
- [ ] Formula prediction filter added
- [ ] Domain-adapted spectral embeddings added
- [ ] Class 2 retrieval/rerank pipeline
- [ ] Class 3 de novo exploration
- [ ] Final ensemble + validation

## Design notes / known limitations

- Library dedup keeps one representative spectrum per (structure, adduct) —
  picking the one with the most peaks. Spectra at different collision
  energies for the same compound are not merged; that's a plausible Phase 2
  improvement (more fragment coverage per candidate).
- The frequency-based fallback (`baseline._global_fallback_candidates`) only
  fires when a spectrum gets zero similarity hits; `predict_test_set` prints
  a warning with the count when it happens, worth checking each run.
- `PEAK_TOP_K`, `BIN_WIDTH`, `COARSE_TOP_K`, `ANALOG`-related bonus constants
  in `src/baseline.py` are initial guesses, not tuned — a natural next step
  once a real local CV split (not just the np-examples holdout) is built.
