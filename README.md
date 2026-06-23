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
mat-migrate-pool        # copy onboarded connectors into connectors/library + write active.yaml
mat-pool sync-lmstudio  # align connector model_name to LM Studio /v1/models (active pool by default)
mat-pool verify         # fails if a connector model is not served (active pool by default)

# Optional: add Venice models (API; put VENICE_API_KEY in .env)
mat-import-aa deepseek-v4-flash --base-url https://api.venice.ai/api/v1 --model-name deepseek-v4-flash --auth-env VENICE_API_KEY --connector-id deepseek-v4-flash@venice
mat-import-aa mimo-v2-5-0424 --base-url https://api.venice.ai/api/v1 --model-name xiaomi-mimo-v2-5 --auth-env VENICE_API_KEY --connector-id xiaomi-mimo-v2-5@venice

# Smoke benchmark (needs LM Studio server on :1234)
mat-benchmark --split val --limit 2 --mode single --connector <id-from-mat-pool-list>

# Default: gateway + live dashboard
mat
# mat --no-tui   # gateway only
```

**Trinity alignment (orchestration lift):** see [docs/trinity-progress.md](docs/trinity-progress.md) for the live plan, status, and long-run commands (`tail -f traces/*.log` for progress).

For a one-command local dev loop (pool sync + smoke benchmarks), see `bash scripts/dev-loop.sh`.

Installed connectors live in **`connectors/library/`** with the active set in **`active.yaml`**. See [docs/runbook.md](docs/runbook.md).

```bash
mat                 # gateway + dashboard (default)
mat --no-tui        # gateway only
```

## Commands

| Command | Purpose |
|---------|---------|
| **`mat`** | **Default:** gateway + live dashboard (`--no-tui` = gateway only) |
| `mat-sync-aa` | Cache Artificial Analysis model catalog |
| `mat-discover-lmstudio` | Build connectors from LM Studio downloads + AA |
| `mat-import-aa <slug>` | Import one model (`--local` for LM Studio) |
| `mat-calibrate --connector <id>` | Blend live coding pass rate into scores |
| `mat-pool list` / `verify` / `sync-lmstudio` / `lmstudio-models` / `rehash` / `apply` / `sync-pricing` | Manage library + active pool |
| `mat-migrate-pool` | Copy legacy ~/.config/mat/connectors into library |
| `mat-benchmark` | Live HumanEval eval vs single-model baselines |
| `mat-train-live` | CMA-ES coordinator on train split (live LLMs) |
| `mat-phase-a` | Simulation generalization spike |

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
| `connectors/library/` | **Shareable connector library** |
| `active.yaml` | **Active pool selection** (connector IDs) |
| `connectors/examples/` | Boilerplate only — not the runtime pool |
| `eval/` | Live loop, benchmark, HumanEval splits |
| `workers/` | OpenAI-compatible LLM client |
| `coordinator/` | Routing policy, checkpoint, SLM (optional) |
| `docs/` | [Getting started](docs/getting-started.md), [connector spec](docs/connector-spec.md), [eval protocol](docs/eval-protocol.md) |

## Status

Alpha. Phase A simulation passed. Live eval pipeline and **`mat-serve` live path** implemented;
**live lift gate pending** on your hardware with installed pool.

## License

Apache-2.0. See [LICENSE](LICENSE).
