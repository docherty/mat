# Installed connectors (optional repo copy)

**Production pool path:** `~/.config/mat/connectors/` (default for `mat-benchmark`, `mat-serve`, `mat-train-live`).

Full setup: [docs/getting-started.md](../docs/getting-started.md).

YAML files here are gitignored. Use this directory only if you set `MAT_POOL_DIR=connectors/installed`.

## Install from your LM Studio cache

```bash
export ARTIFICIAL_ANALYSIS_API_KEY=...   # free tier at artificialanalysis.ai/data-api
mat-sync-aa
mat-discover-lmstudio
```

This scans `~/.cache/lm-studio/models`, matches each download to Artificial Analysis benchmarks, and writes **benchmark_import** connectors (not hand guesses).

## Single model

```bash
mat-import-aa qwen3-6-35b-a3b --local --model-name qwen3.6-35b-a3b
```

## Calibrate after install (optional)

Blends live HumanEval **train** pass rate into `coding` score:

```bash
mat-calibrate --connector mlx-community-qwen3-6-35b-a3b-8bit@lmstudio --limit 10
```
