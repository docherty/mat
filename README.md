# mat

**Small models. Mighty results.**

mat is an open-source orchestration layer that looks like a normal OpenAI API endpoint.
Point your existing client at it; behind the scenes it decomposes requests, routes work
to a swappable model pool, verifies where ground truth exists, and returns one answer.

run it locally on your machine and route to a pool of local or api-accessed models.

## What it is

- An OpenAI-compatible front (`POST /v1/chat/completions`, `GET /v1/models`)
- A **trained coordinator** that routes by capability profile (not model names)
- **Installed connectors** with Artificial Analysis benchmark attestations
- Multi-turn **think → work → verify** on verifiable coding tasks
- LM Studio / mlx-lm / OpenRouter / Venice as pool members

## Quick start (local, LM Studio)

See **[docs/getting-started.md](docs/getting-started.md)** for the full walkthrough.

```bash
pip install -e ".[dev]"   # use: python3.11 -m pip install -e ".[dev]" if pip is Python 3.9

# Optional: keep env vars tidy
cp .env.example .env
# edit .env (gitignored)

# Optional: full AA catalog (free API key)
export ARTIFICIAL_ANALYSIS_API_KEY=...
mat-sync-aa

# Install connectors from ~/.cache/lm-studio/models (uses AA scrape per model if needed)
mat-discover-lmstudio   # requires LM Studio up; maps exact /v1/models ids
mat-pool verify         # fails if a connector model is not served

# Smoke benchmark (needs LM Studio server on :1234)
mat-benchmark --split val --limit 2 --mode single --connector <id-from-mat-pool-list>

# OpenAI-compatible gateway (MAT_LIVE=1 by default)
export MAT_GATEWAY_KEY=local-dev-key
mat-serve
```

Installed connectors live in **`~/.config/mat/connectors/`** — not `connectors/examples/`.
See [docs/eval-protocol.md](docs/eval-protocol.md) for honest benchmarking rules.

## Commands

| Command | Purpose |
|---------|---------|
| `mat-sync-aa` | Cache Artificial Analysis model catalog |
| `mat-discover-lmstudio` | Build connectors from LM Studio downloads + AA |
| `mat-import-aa <slug>` | Import one model (`--local` for LM Studio) |
| `mat-calibrate --connector <id>` | Blend live coding pass rate into scores |
| `mat-pool list` / `verify` / `sync-lmstudio` / `lmstudio-models` | Inspect installed pool |
| `mat-benchmark` | Live HumanEval eval vs single-model baselines |
| `mat-train-live` | CMA-ES coordinator on train split (live LLMs) |
| `mat-phase-a` | Simulation generalization spike |
| `mat-serve` | OpenAI-compatible gateway |

## Training workflow

```bash
# 1. Install pool with real AA benchmarks
mat-discover-lmstudio

# 2. Train coordinator on train split (local LLMs, $0 API)
mat-train-live --mock --tasks 3 --generations 2   # dry run, no API
mat-train-live --tasks 5 --generations 8 --checkpoint ~/.config/mat/coordinator/latest.json

# 3. Evaluate on held-out val (never used in training)
mat-benchmark --split val --mode compare --checkpoint ~/.config/mat/coordinator/latest.json
```

Success gate: orchestrated **+8pp** pass@1 vs best single on val. See [docs/eval-protocol.md](docs/eval-protocol.md).

## Path 2 SLM coordinator (optional)

```bash
pip install -e ".[slm]"   # torch + transformers; best on M4 / 4090
```

`coordinator/slm_coordinator.py` — Trinity-style hidden-state head (CMA-ES training wired next).

## Architecture

```
┌─────────────────────────────────────────┐
│  OpenAI-compatible API (mat-serve)      │
├─────────────────────────────────────────┤
│  Coordinator + installed connector pool │
│  THINK → WORK → VERIFY → SYNTHESIZE     │
├─────────────────────────────────────────┤
│  Tool-calling backend                   │
└─────────────────────────────────────────┘
```

## Layout

| Path | Purpose |
|------|---------|
| `connectors/examples/` | Boilerplate only — not the runtime pool |
| `~/.config/mat/connectors/` | **Your installed pool** |
| `eval/` | Live loop, benchmark, HumanEval splits |
| `workers/` | OpenAI-compatible LLM client |
| `coordinator/` | Routing policy, checkpoint, SLM (optional) |
| `docs/` | [Getting started](docs/getting-started.md), [connector spec](docs/connector-spec.md), [eval protocol](docs/eval-protocol.md) |

## Status

Alpha. Phase A simulation passed. Live eval pipeline and **`mat-serve` live path** implemented;
**live lift gate pending** on your hardware with installed pool.

## License

Apache-2.0. See [LICENSE](LICENSE).
