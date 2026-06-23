#!/usr/bin/env bash
# Overnight SLM coordinator training + val compare when training completes.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${MAT_SLM_CHECKPOINT:-$HOME/.config/mat/coordinator/latest_slm.json}"
LOG_DIR="traces"
mkdir -p "$LOG_DIR"

echo "=== mat overnight SLM train $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$LOG_DIR/overnight_slm.log"

python3.11 -m pip install -e ".[slm]" -q

mat-train-live-slm \
  --tasks "${MAT_SLM_TASKS:-20}" \
  --generations "${MAT_SLM_GENERATIONS:-10}" \
  --population "${MAT_SLM_POPULATION:-8}" \
  --parallel "${MAT_SLM_PARALLEL:-4}" \
  --task-workers "${MAT_SLM_TASK_WORKERS:-2}" \
  --checkpoint "$CKPT" \
  2>&1 | tee -a "$LOG_DIR/train_live_slm.log"

echo "=== training done; val compare $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG_DIR/overnight_slm.log"

MAT_CHECKPOINT="$CKPT" MAT_COORDINATOR=slm mat-benchmark \
  --split val \
  --mode compare \
  --connector deepseek-v4-flash@venice \
  --checkpoint "$CKPT" \
  --out "$LOG_DIR/val_compare_slm.json" \
  2>&1 | tee -a "$LOG_DIR/val_compare_slm.log"

echo "=== complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG_DIR/overnight_slm.log"
