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

- [ ] Kaggle API configured, rules accepted
- [ ] Data downloaded
- [ ] Baseline pipeline reproduced (target: ~0.30+ MRR@25)
- [ ] Formula prediction filter added
- [ ] Domain-adapted spectral embeddings added
- [ ] Class 2 retrieval/rerank pipeline
- [ ] Class 3 de novo exploration
- [ ] Final ensemble + validation
