# Trinity alignment — why mat underperforms and what to do

Paper: [Trinity: An Evolved LLM Coordinator](https://arxiv.org/abs/2512.04695) (Sakana AI, arXiv:2512.04695).

mat implements the *shape* of Trinity (think / work / verify, CMA-ES, capability pool) but not the *mechanism* that makes it work. The gap is architectural and operational, not a small tuning bug.

## What Trinity actually does

| Mechanism | Trinity | mat today |
|-----------|---------|-----------|
| Coordinator input | **Full multi-turn transcript** → 0.6B SLM hidden state | Static task metadata + capability scores |
| Decision each turn | **Joint (model, role)** from ~10K head | Fixed role pipeline; pick model per role |
| Trainable params | ~10K head + SLM singular-value scales | 9-dim linear residual on catalog scores |
| Optimizer | **sep-CMA-ES**, pop≈32, **16 replicates** per candidate | Full CMA-ES on 9 dims, pop 4–8, **1 replicate** |
| Training budget | **1.5k–40k** trajectory evaluations per task family | ~16–64 evals (3 tasks × 4 gens × 4 pop) |
| Turn budget | **5 turns**, enforced | `revision_cap=2`; `max_turns` unused |
| Pool | **7 diverse specialists** (GPT-5, Gemini, Claude, R1-reasoner, Qwen-direct, …) | 3 similar local MLX + 2 Venice APIs |
| Baseline | **5× self-reflection** (same turn budget) | `single_reflect` with 2 revision caps |
| Success ceiling | Near **per-question-best** union over pool | Not reported |

Trinity wins because **models disagree by task** and a **transcript-conditioned** coordinator learns who does what when. mat routes from **benchmark cards** (AA scores) without reading the conversation.

## Why we look “way off”

### 1. Pool has no complementarity (biggest issue)

Three local generalists (Qwen, Gemma, LFM2) on one LM Studio box are substitutes, not specialists. Trinity’s LiveCodeBench gain comes from routing **GPT-5 vs Gemini** — different failure modes.

**Per-question-best** on our pool is likely ≈ best single model, not much higher. Orchestration cannot beat a single model if the pool does not.

### 2. Coordinator cannot see the transcript

Routing after a failed worker attempt should change; mat picks from the same static scores. `SLMCoordinator` exists but is optional, untrained, and not wired into the live loop with transcript input.

### 3. Training is ~100× too small

CMA-ES with 3 tasks and unchanged weights means the optimizer never left the heuristic warm-start. Trinity uses thousands of Bernoulli trajectory evaluations with replication.

### 4. Protocol mismatch (not wrong metric, wrong budget)

- `max_turns` was never enforced (fixed in code).
- `revision_cap=2` vs Trinity’s 5-turn budget makes `single_reflect` an unfair or weak baseline depending on model.
- For strong local coders, **forced thinker hurts**: LFM2 100% single vs 60% reflect on val — orchestration copies that failure mode.

### 5. Wrong order of operations

We tuned a 9-weight coordinator before fixing pool and SLM path. Correct order:

1. **Pool** — complementary specialists (reasoner + coder + 1–2 APIs).
2. **Measure union ceiling** — `per_question_best` on val; if ≈ best single, stop expecting routing magic.
3. **SLM coordinator** — transcript → hidden state → head; sep-CMA-ES at scale on train split.
4. **Full val compare** — orchestrated vs max(single, 5× reflect); target +8pp lift.

## Metrics (keep and add)

**Keep:** val `pass@1` lift vs `max(single, single_reflect)` — honest product bar.

**Add:**

- **`per_question_best`** — fraction of val tasks where *any* pool member passes (union ceiling). Orchestration should approach this.
- **`gap_to_union`** — `per_question_best − orchestrated` (how much routing leaves on the table).
- Report **turn/token budget** matched to 5 turns for baselines.

## Target pool (M4 Max + Venice)

| Role | Example | Runtime |
|------|---------|---------|
| Fast coder | Qwen2.5-Coder / warm Qwen3.6 | LM Studio |
| Reasoner | Qwen3-8B thinking / R1-distill | mlx-lm server |
| Frontier API | DeepSeek V4 / GPT-class | Venice |

Drop duplicate local generalists from **active** routing; keep them in **library** for A/B.

## Implementation roadmap

| Priority | Work | Expected impact |
|----------|------|-----------------|
| P0 | Complementary active pool + `local_primary` | Enables complementarity |
| P0 | Enforce `max_turns`; benchmark `revision_cap=5` | Fair Trinity comparison |
| P0 | `per_question_best` in `mat-benchmark compare` | Know the ceiling |
| P1 | Wire transcript into coordinator `pick()` | Core Trinity mechanism |
| P1 | Train `SLMCoordinator` with sep-CMA-ES, ≥41 train tasks, replicates | Learned routing |
| P2 | Joint (model, role) action space | Full paper fidelity |
| P2 | SLM singular-value fine-tuning | Paper ablation +6pp class gains |

## What is not the problem

- OpenAI gateway, health, dashboard — shipping infrastructure is fine.
- HumanEval val as holdout — correct discipline.
- CMA-ES family choice — right algorithm; wrong scale and parameterization.

## Bottom line

We are not failing because “orchestration is fake.” We are running **catalog routing with a non-complementary pool and no transcript coordinator**, then training 9 numbers on 3 tasks. That is not Trinity.

Success path: **fix the pool → measure the union ceiling → build transcript SLM coordinator → train at paper-scale → then judge +8pp on val.**

**Live progress:** [trinity-progress.md](./trinity-progress.md)
