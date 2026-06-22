# Connector specification (v1.0)

Connectors are YAML files that describe one model pool member. The coordinator reads
**capability tiers only** — never the model ID — so routing generalises to unseen models.

## Schema

```yaml
connector_version: "1.0"
id: "deepseek-v4-flash@openrouter"
display_name: "DeepSeek V4 Flash"

endpoint:
  type: "openai"           # openai | ollama | anthropic | custom
  base_url: "https://openrouter.ai/api/v1"
  model_name: "deepseek/deepseek-v4-flash"
  auth_env: "OPENROUTER_API_KEY"

context_window: 1000000
max_output_tokens: 32768
modalities: ["text"]
pricing:
  input_per_1k: 0.09
  output_per_1k: 0.18
  currency: "USD"
locality: "api"            # api | local
supports:
  tools: true
  json_mode: true
  streaming: true
  system_prompt: true
tool_format: "openai"      # openai | anthropic | none

capabilities:              # tier ∈ {weak, mid, strong, frontier}
  reasoning: "strong"
  coding: "strong"
  long_context: "strong"
  instruction_following: "strong"
  verification: "mid"
  tool_use: "strong"

speed:
  tokens_per_sec: 60
  tier: "fast"             # fast | medium | slow

profile:
  probe_suite_version: "2026.1"
  profiled_at: "2026-06-22T00:00:00Z"
  probe_scores:
    reasoning: 0.78
    coding: 0.77
  contributor: "github:username"
  integrity_sha256: "<hash>"
```

## Capability taxonomy

| Tag | Used in stage |
|-----|---------------|
| `reasoning` | THINK, hard WORK |
| `coding` | WORK |
| `long_context` | any |
| `instruction_following` | SYNTHESIZE |
| `verification` | VERIFY (model-judge) |
| `tool_use` | agentic WORK |
| `speed` | fast-path, bulk |

Tiers: `weak`, `mid`, `strong`, `frontier`. Keep this vocabulary stable — the
coordinator trains against it.

## Integrity hash

`integrity_sha256` covers the canonical fields (everything except `profile.integrity_sha256`
itself). The validator recomputes and rejects mismatches.

## Freshness

Connectors older than 90 days or profiled under a stale `probe_suite_version` are flagged
`stale` in logs. Re-profile with `mat-profile` to refresh.

## Trust

Third-party connectors are untrusted input. Validate schema, check the hash, review the
endpoint URL before use.
