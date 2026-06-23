# Production runbook

## Daily startup

```bash
cd mat
cp .env.example .env   # once — edit MAT_GATEWAY_KEY and API keys
mat                    # gateway + dashboard (default)
# mat --no-tui         # headless gateway only
```

## Active pool

- Library: `connectors/library/*.yaml`
- Selection: `active.yaml` at repo root
- Change pool without deleting connectors: `mat-pool apply --curated <file>`

## Health checks

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /health` | none | Load balancers — pool status + issues |
| `GET /v1/mat/status` | Bearer | Full pool detail + metrics |
| `GET /v1/mat/metrics` | Bearer | Request counters |
| `GET /v1/mat/recent` | Bearer | Last routing decisions |

```bash
curl -s http://127.0.0.1:8080/health | jq
curl -s -H "Authorization: Bearer $MAT_GATEWAY_KEY" http://127.0.0.1:8080/v1/mat/status | jq
```

## Before sharing the repo

1. `mat-pool verify` — all LM Studio models loaded
2. `mat-pool sync-pricing` — API connectors have current prices
3. Commit `connectors/library/` + `active.yaml`
4. Never commit `.env`

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `pool_error` / missing connector | ID in `active.yaml` must exist in `connectors/library/` |
| LM Studio 400 | `mat-pool sync-lmstudio`; load exact model id |
| Gateway 503 | Empty pool — check `active.yaml` |
| Venice errors | `VENICE_API_KEY` in `.env`; `mat-pool sync-pricing` |
| Orchestration picks wrong model | `mat-calibrate`; check `speed.token_efficiency` |

## Honest eval (before trusting routing)

```bash
mat-benchmark --split val --mode compare --out traces/val_compare.json
```

Success gate: orchestrated pass@1 ≥ best single + **8pp** on val. See [eval-protocol.md](eval-protocol.md).
