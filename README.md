# mat

**Small models. Mighty results.**

mat is an open-source orchestration layer that looks like a normal OpenAI API endpoint.
Point your existing client at it; behind the scenes it decomposes requests, routes work
to a swappable model pool, verifies where ground truth exists, and returns one answer.

You run it. You hold the downstream API keys. We host nothing.

## What it is

- An OpenAI-compatible front (`POST /v1/chat/completions`, `GET /v1/models`) so
  chat UIs, agent frameworks, and IDEs work without integration work.
- A coordinator that routes by **capability profile**, not hard-coded model names.
  New models join via [connector files](docs/connector-spec.md).
- A tool-calling backend that normalises format and executes internal tools in a
  sandbox (client tools are relayed per OpenAI semantics).
- Oracle-first verification on coding tasks — compile, run hidden tests, then optional
  model-judge.

## What it is not

The success bar is **verifiable coding tasks**, not general open-ended chat. mat will
not replace Claude or ChatGPT for everything. Where answers cannot be checked, the
verifier cannot help.

## Architecture

```
┌─────────────────────────────────────────┐
│  OpenAI-compatible API (fixed front)    │
├─────────────────────────────────────────┤
│  Coordinator + swappable model pool     │
│  THINK → WORK → VERIFY → SYNTHESIZE     │
├─────────────────────────────────────────┤
│  Tool-calling backend (fixed back)      │
└─────────────────────────────────────────┘
```

Connectors describe each pool member's coarse capability tiers. The coordinator reads
profiles only — never model IDs — so a connector for a model it has never seen can
still be routed sensibly after training.

## Status

Alpha. Phase A (generalization micro-spike) is the current gate. See
[docs/phaseA-result.md](docs/phaseA-result.md) for outcomes.

## Install

```bash
git clone https://github.com/docherty/mat.git
cd mat
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Phase A spike

```bash
mat-phase-a
```

Runs the 10-task generalization experiment: train a coordinator on three connector
profiles, test against a hand-prompted baseline, then swap in a held-out fourth
connector without retraining.

## Start the API server

```bash
export MAT_GATEWAY_KEY=local-dev-key
export OPENROUTER_API_KEY=sk-...   # if using API pool members
mat-serve
```

Clients use `Authorization: Bearer $MAT_GATEWAY_KEY`. Downstream provider keys live in
config/env only — see [docs/secrets-policy.md](docs/secrets-policy.md).

## Profile a model

```bash
mat-profile --endpoint openai --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-chat --auth-env OPENROUTER_API_KEY \
  --output connectors/my-connector.yaml
```

## Layout

| Path | Purpose |
|------|---------|
| `api/` | OpenAI-compatible HTTP server |
| `coordinator/` | Routing policy and CMA-ES training |
| `connectors/` | Connector schema, loader, example profiles |
| `loop/` | Think/Work/Verify/Revise/Synthesize runtime |
| `tool_backend/` | Tool format adapters and sandbox execution |
| `probe_set/` | Canonical probe suite and profiling CLI |
| `eval/` | Oracle, harness, Phase A experiment |
| `traces/` | Run logging with secret redaction |
| `configs/` | Default pool and server config |
| `docs/` | Specs, policies, gate decisions |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Connector submissions are welcome once the
profiling tooling and schema validator are in place.

## License

Apache-2.0. See [LICENSE](LICENSE).
