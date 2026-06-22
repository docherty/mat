# Connector Profile v1.1

A mat connector is a shareable YAML file describing **one model in an orchestration pool**.
The coordinator reads capability **scores** (and coarse tiers derived from them), never the
model name.

This spec deliberately avoids specification theatre: every field is either used by mat
today or reserved for a named near-term need documented below.

## What a connector is for

| Need | Field(s) |
|------|----------|
| Call the model | `endpoint`, `context_window`, `max_output_tokens`, `supports`, `tool_format` |
| Route work by skill | `capabilities.*.score` (primary), `capabilities.*.tier` (coordinator training) |
| Break ties between similar models | `capabilities.*.score`, then `benchmarks`, then `speed`, then `pricing` |
| Cost-aware routing | `pricing` |
| Fast-path / bulk routing | `speed.tier` |
| Share without re-measuring | file + `integrity_sha256` |
| Trace where data came from | `profile` |

## What a connector is not

- Not a model card for ethics/compliance (see [Model Cards](https://modelcards.withgoogle.com/about))
- Not a full Hugging Face hub record
- Not a guarantee of live performance — attestations go stale; re-profile or refresh

## File layout

```yaml
connector_version: "1.1"
id: deepseek-v4-flash@openrouter
display_name: DeepSeek V4 Flash

endpoint:
  type: openai              # openai | ollama | anthropic | custom
  base_url: https://openrouter.ai/api/v1
  model_name: deepseek/deepseek-v4-flash
  auth_env: OPENROUTER_API_KEY   # env var NAME only — never the key value

context_window: 1048576
max_output_tokens: 65536
modalities: [text]
locality: api               # api | local

pricing:                    # used by cost-aware fitness / routing
  input_per_1k: 0.00009     # USD per 1k input tokens
  output_per_1k: 0.00018
  currency: USD

supports:                   # hard constraints — do not route incompatible work here
  tools: true
  json_mode: true
  streaming: true
  system_prompt: true
  reasoning: true             # model accepts a reasoning/thinking effort parameter

tool_format: openai         # openai | anthropic | none

capabilities:               # score ∈ [0,1] is what the router optimises over
  reasoning:    { score: 0.78, tier: strong }
  coding:       { score: 0.92, tier: frontier }
  long_context: { score: 0.63, tier: mid }
  instruction_following: { score: 0.74, tier: strong }
  verification: { score: 0.81, tier: strong }
  tool_use:     { score: 0.95, tier: frontier }

speed:
  tier: fast                  # fast | medium | slow — API throughput class
  tokens_per_sec: null        # optional; set when measured locally

benchmarks:                 # optional cited measurements for tie-break / audit
  - source: artificial_analysis
    metric: intelligence_index_v4.1
    value: 47
    unit: index
    as_of: "2026-04-24"
    url: https://artificialanalysis.ai/models/deepseek-v4-flash
  - source: artificial_analysis
    metric: livecodebench
    value: 91.6
    unit: percent
    as_of: "2026-04-24"

profile:
  profile_method: benchmark_import   # mat_probe | benchmark_import | hand
  catalog: openrouter
  catalog_id: deepseek/deepseek-v4-flash
  probe_suite_version: null          # set when profile_method=mat_probe
  profiled_at: "2026-06-22T12:00:00Z"
  contributor: mat:examples
  notes: Capability scores derived from AA sub-benchmarks; see connectors/examples/.
  integrity_sha256: "<computed>"
```

## Capability taxonomy

Stable routing vocabulary — changing it implies retraining the coordinator.

| Tag | Orchestration stage | Score means |
|-----|-------------------|-------------|
| `reasoning` | THINK, hard WORK | multi-step problem solving |
| `coding` | WORK | code generation quality |
| `long_context` | any | reliable use of large context |
| `instruction_following` | SYNTHESIZE | format/constraint adherence |
| `verification` | VERIFY (model-judge) | judging correctness of outputs |
| `tool_use` | agentic WORK | reliable tool-call formation |

**Tier** is derived from score and cached in the file for human readability and
coordinator generalisation training:

| Tier | Score range |
|------|-------------|
| `weak` | 0.00 – 0.45 |
| `mid` | 0.45 – 0.65 |
| `strong` | 0.65 – 0.85 |
| `frontier` | 0.85 – 1.00 |

The validator rejects a tier that does not match its score.

## Benchmark attestations

Optional. Use when two pool members have similar capability scores and you need a
documented tie-break.

| Field | Purpose |
|-------|---------|
| `source` | Who measured it (`artificial_analysis`, `mat_probe`, `swe_bench`, …) |
| `metric` | What was measured (`intelligence_index_v4.1`, `gpqa_diamond`, …) |
| `value` | Raw number as published |
| `unit` | `index` (composite), `ratio` (0–1), or `percent` (0–100) |
| `as_of` | Publication or measurement date |
| `url` | Link for human verification |

**Routing precedence today:** mat probe scores in `capabilities` (when
`profile_method: mat_probe`) beat imported benchmarks. Among imports, prefer the most
specific metric for the task tag (e.g. `livecodebench` for `coding`) over composite
indices.

## Profile methods

| Method | When to use |
|--------|-------------|
| `mat_probe` | You ran `mat-profile` against the model (canonical, community-comparable) |
| `benchmark_import` | Operational fields + capability scores built from cited public benchmarks |
| `hand` | Manual connector; document rationale in `profile.notes` |

`catalog` + `catalog_id` trace back to a provider catalog (e.g. OpenRouter slug). Used
by import tooling — not read by the coordinator.

## Integrity hash

`integrity_sha256` covers all fields except itself. Recompute with:

```bash
python -c "from connectors.loader import load_connector, dump_connector; ..."
```

Or save via `dump_connector()` which sets the hash automatically.

## Freshness

A connector is **stale** when:

- `profiled_at` is older than 90 days, or
- `profile_method: mat_probe` and `probe_suite_version` ≠ current suite (`2026.1`)

Stale connectors are flagged in logs. Re-run `mat-profile` or refresh attestations.

## Trust

Third-party connectors are untrusted input. Validate schema, check hash, review
`endpoint.base_url` before loading.

## v1.0 migration

v1.0 files with tier-only capabilities (`coding: strong`) load as v1.1 with scores
inferred from tier midpoints. Re-save with `dump_connector()` to upgrade on disk.

## Examples

Real connectors (OpenRouter, benchmark-imported):

- [`connectors/examples/openrouter-deepseek-v4-flash.yaml`](../connectors/examples/openrouter-deepseek-v4-flash.yaml)
- [`connectors/examples/openrouter-deepseek-v4-pro.yaml`](../connectors/examples/openrouter-deepseek-v4-pro.yaml)
- [`connectors/examples/openrouter-tencent-hy3-preview.yaml`](../connectors/examples/openrouter-tencent-hy3-preview.yaml)
- [`connectors/examples/openrouter-mimo-v2.5.yaml`](../connectors/examples/openrouter-mimo-v2.5.yaml)
- [`connectors/examples/openrouter-mimo-v2.5-pro.yaml`](../connectors/examples/openrouter-mimo-v2.5-pro.yaml)

Phase A simulation fixtures remain in `connectors/examples/alpha-*.yaml`.

## Examples vs installed

| Directory | Role |
|-----------|------|
| `connectors/examples/` | Boilerplate and reference imports — **not** the runtime pool |
| `~/.config/mat/connectors/` | Default **installed** pool (`MAT_POOL_DIR` to override) |

Installed connectors must use `profile_method: benchmark_import` or `mat_probe`, with cited
`benchmarks[]` from Artificial Analysis (or `mat_calibration` after `mat-calibrate`).

```bash
export ARTIFICIAL_ANALYSIS_API_KEY=...
mat-sync-aa
mat-discover-lmstudio    # from ~/.cache/lm-studio/models
mat-import-aa <slug> --local --model-name <lm-studio-id>
mat-calibrate --connector <id> --limit 10
```

## Related tooling

```bash
# Import operational skeleton from OpenRouter (capabilities still need scores)
python -m connectors.import_openrouter deepseek/deepseek-v4-flash

# Full probe run (when live probes are wired)
mat-profile --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4-flash --auth-env OPENROUTER_API_KEY \
  --output my-connector.yaml
```
