#!/usr/bin/env bash
# Restart the fingerprint training if it is not running, continuing from the
# last epoch checkpoint. Safe to run at any time: if training is already
# alive it reports that and does nothing.
#
# Backgrounded training has NOT reliably survived this project's sessions
# ending -- the v1 run lived through several, the v2 run died mid-epoch -- so
# per-epoch checkpoints plus --resume are the mechanism that actually works,
# not process survival. A restart loses at most the epoch in progress.
#
#   bash scripts/resume_training.sh          # v2 stack (default)
#   bash scripts/resume_training.sh v1       # v1 stack
set -euo pipefail

cd "$(dirname "$0")/.."
STACK="${1:-v2}"
EPOCHS="${EPOCHS:-8}"
case "$STACK" in
  v1) PREFIX=multi_model ;;
  v2) PREFIX=fp2_model ;;
  *) echo "unknown stack '$STACK' (expected v1 or v2)" >&2; exit 2 ;;
esac

# the bracket keeps the pattern from matching this script's own command line
if pgrep -f "train_multi[_]fingerprint.*--stack $STACK" > /dev/null; then
  echo "training (stack $STACK) is already running:"
  ps -eo pid,etime,cmd | grep "[t]rain_multi_fingerprint" | sed 's/^/  /'
  exit 0
fi

LAST=$(ls -1 data/processed/${PREFIX}_e*.pt 2>/dev/null \
       | sed 's/.*_e\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
if [ -n "${LAST:-}" ]; then
  echo "last checkpoint: ${PREFIX}_e${LAST}.pt -> continuing at epoch $((LAST + 1)) of $EPOCHS"
else
  echo "no checkpoint found -> starting from epoch 1 of $EPOCHS"
fi

LOG="logs/train_${PREFIX}_resume_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
PYTHONPATH=. setsid nohup python3 -u scripts/train_multi_fingerprint.py \
  --stack "$STACK" --epochs "$EPOCHS" --resume > "$LOG" 2>&1 < /dev/null &
disown || true

sleep 20
if pgrep -f "train_multi[_]fingerprint.*--stack $STACK" > /dev/null; then
  echo "started; logging to $LOG"
  grep -E "resuming|continuing|device" "$LOG" | sed 's/^/  /' || true
else
  echo "FAILED to start -- see $LOG" >&2
  tail -20 "$LOG" >&2
  exit 1
fi
