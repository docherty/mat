# Getting started

This guide walks through a **local LM Studio** setup on macOS. The same commands work with mlx-lm or remote APIs once connectors are installed.

## Prerequisites

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) with at least one model downloaded
- LM Studio local server on `http://127.0.0.1:1234/v1`

```bash
git clone <repo>
cd mat
pip install -e ".[dev]"
```

## 1. Install your model pool

mat never guesses capability scores. Connectors are built from **Artificial Analysis** benchmarks.

```bash
# Optional but recommended: full AA catalog (free API key)
export ARTIFICIAL_ANALYSIS_API_KEY=...
mat-sync-aa

# Scan ~/.cache/lm-studio/models and write connectors
mat-discover-lmstudio
mat-pool list
mat-pool verify   # optional: ping each endpoint
```

Installed YAML files land in **`~/.config/mat/connectors/`**. The repo's `connectors/examples/` directory is boilerplate only.

Override the pool path:

```bash
export MAT_POOL_DIR=/path/to/connectors
```

## 2. Smoke benchmark

Start LM Studio and load one model (e.g. `qwen3.6-35b-a3b`).

```bash
mat-benchmark --split val --limit 2 --mode single \
  --connector <id-from-mat-pool-list>
```

Modes:

| Mode | What it measures |
|------|------------------|
| `single` | One connector, one pass |
| `single_reflect` | Single connector + verifier loop |
| `orchestrated` | Think → work → verify routing |
| `compare` | Baselines + orchestrated; pass `--connector` to compare one model only |

See [eval-protocol.md](eval-protocol.md) for train/val splits and the +8pp success gate.

## 3. Train coordinator (optional)

Training uses the **train** split only (41 HumanEval tasks). Val is held out.

```bash
# Dry run with mock worker (no API calls)
mat-train-live --mock --tasks 3 --generations 2

# Live training on local LLMs
mat-train-live --tasks 5 --generations 8 \
  --checkpoint ~/.config/mat/coordinator/latest.json
```

Evaluate on val:

```bash
mat-benchmark --split val --mode compare \
  --checkpoint ~/.config/mat/coordinator/latest.json
```

## 4. Run the gateway

`mat-serve` exposes an OpenAI-compatible API. By default **`MAT_LIVE=1`** — real LLM calls when a pool is installed.

```bash
export MAT_GATEWAY_KEY=local-dev-key
export MAT_CHECKPOINT=~/.config/mat/coordinator/latest.json   # optional
mat-serve
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "balanced",
    "messages": [{"role": "user", "content": "Write def add(a,b): return a+b"}]
  }'
```

Quality tiers: `fast`, `balanced`, `max`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `MAT_LIVE` | `1` | `0` = proxy responses (tests / offline) |
| `MAT_POOL_DIR` | `~/.config/mat/connectors` | Installed connector directory |
| `MAT_CHECKPOINT` | `~/.config/mat/coordinator/latest.json` if present | Trained routing weights |
| `MAT_COORDINATOR` | auto | `prompted` or `trained` |
| `MAT_GATEWAY_KEY` | `local-dev-key` | Bearer token for `/v1/*` |
| `MAT_LMSTUDIO_MODEL` | auto from `/v1/models` | Force model id for all local connectors |
| `MAT_LLM_TIMEOUT` | `300` | Seconds per LLM request |
| `MAT_HOST` / `MAT_PORT` | `127.0.0.1` / `8080` | Bind address for `mat-serve` |

## 5. Calibrate a connector (optional)

Blend live HumanEval **train** pass rate into the `coding` capability score:

```bash
mat-calibrate --connector <id> --limit 10
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `no connectors` | Run `mat-discover-lmstudio` or set `MAT_POOL_DIR` |
| Connection refused on :1234 | Start LM Studio server; load a model |
| All models show same AA scores | Re-run `mat-discover-lmstudio` after upgrading mat |
| `mlx_lm.server: command not found` | Use `python -m mlx_lm server` |
| Slow compare mode | One LM Studio instance = one loaded model; run modes sequentially |

## Next steps

- [Connector spec](connector-spec.md) — YAML schema and capability vectors
- [Eval protocol](eval-protocol.md) — honest benchmarking rules
- [README](../README.md) — command reference and architecture
