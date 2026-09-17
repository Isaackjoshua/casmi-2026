#!/usr/bin/env bash
# Downloads and unzips the CASMI 2026 competition data into data/raw/.
# Requires: ~/.kaggle/kaggle.json in place, and the competition rules accepted
# on kaggle.com (Join Competition button) under that same account.
set -euo pipefail

COMPETITION="enveda-CASMI26-molecule-id-mass-spectra"
DEST="$(dirname "$0")/../data/raw"

mkdir -p "$DEST"
kaggle competitions download -c "$COMPETITION" -p "$DEST"
unzip -o "$DEST/${COMPETITION}.zip" -d "$DEST"
rm -f "$DEST/${COMPETITION}.zip"

echo "Downloaded to $DEST:"
ls -la "$DEST"
