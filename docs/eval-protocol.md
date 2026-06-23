# Live eval protocol

mat benchmarks exist to answer one question: **does orchestration beat using a single
pool member, on tasks that matter, at acceptable cost?**

This document defines rules so we do not benchmaxx ourselves.

## Datasets

| File | Split | Use |
|------|-------|-----|
| `eval/tasks/humaneval_train.json` | train (41) | Coordinator CMA-ES fitness only |
| `eval/tasks/humaneval_val.json` | val (23) | **Reporting only** — never training fitness |
| `eval/tasks/phase_a_tasks.json` | — | Simulation / unit tests only |

HumanEval split: seed **42**, fixed IDs, built via `python -m eval.datasets.build_humaneval_split`.

**Do not** tune connectors, prompts, or coordinator weights on val. If you iterate on val
more than once, refresh with a new held-out split.

## Modes (`mat-benchmark`)

| Mode | What it measures |
|------|------------------|
| `single` | One model, one worker shot (no think/verify) |
| `single_reflect` | Same model, full think→work→verify — **budget-matched** baseline |
| `orchestrated` | mat multi-model T/W/V with capability routing |
| `compare` | All of the above; add `--connector <id>` to baseline one model (recommended for local LM Studio) |

## Success criteria (val split)

Report all of these — not just pass@1:

1. **Lift:** `orchestrated.pass@1 − best(single, single_reflect).pass@1` ≥ **0.08**
2. **Cost:** `orchestrated.mean_cost_usd` ≤ **2×** best single (local pool = $0)
3. **Token efficiency:** report `mean_output_tokens` per mode; calibrate with `mat-calibrate` for `speed.token_efficiency`
4. **Stability:** per-task trace logs; no manual cherry-picking

If lift is under 5pp after pool/prompt iteration, change the pool before more coordinator tuning.

## What we do not report as achievements

- pass@1 on train split
- `best_connector_for_task` oracle routing (cheating — knows answer key via tag weights)
- phase_a simulated harness scores as live performance

## Hardware notes

- **M4 Max 128GB:** LM Studio + `mlx_lm.server` workers; mlx-lm for Path 2 coordinator SLM
- **4090 / Verda:** LM Studio or vLLM-style OpenAI server on the GPU box
- **Venice $16/day:** reserve for val `compare` runs on API models, not training rollouts

Local workers use `endpoint.type: openai` against LM Studio (`:1234`) or mlx-lm (`:8080`).

**LM Studio:** each connector's `endpoint.model_name` must be an exact id from `GET /v1/models`.
mat does not substitute models. Run `mat-pool verify` before benchmarking.
See `connectors/examples/local/README.md`.

## Commands

```bash
# Smoke (installed pool, 2 val tasks)
mat-benchmark --split val --limit 2

# Full honest comparison
mat-benchmark --split val --mode compare --out traces/benchmark_val.json
```

Pool defaults to `~/.config/mat/connectors/`. See `connectors/examples/local/README.md`.
