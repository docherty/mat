#!/usr/bin/env bash
set -euo pipefail

# One-command dev loop for mat on a local machine.
# Requires:
# - Python 3.11+
# - LM Studio server running and models loaded
# - .env configured (VENICE_API_KEY optional)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3.11 -m pip install -e ".[dev]" >/dev/null

echo "== Pool =="
mat-pool rehash >/dev/null || true
mat-pool sync-lmstudio >/dev/null || true
mat-pool apply --curated connectors/curated/local-dev-pool.yaml --keep deepseek-v4-flash@venice >/dev/null || true
mat-pool list
mat-pool verify || true

echo
echo "== Calibrate (dry run) =="
mat-calibrate --connector mlx-community-qwen3-6-35b-a3b-8bit@lmstudio --limit 3 --dry-run || true
mat-calibrate --connector deepseek-v4-flash@venice --limit 2 --dry-run || true

echo
echo "== Bench (smoke) =="
mkdir -p traces
mat-benchmark --split val --limit 3 --mode compare --connector mlx-community-qwen3-6-35b-a3b-8bit@lmstudio --out traces/compare_val3.json || true

echo
echo "Done."

