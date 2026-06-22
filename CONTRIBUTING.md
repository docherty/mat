# Contributing to mat

Thanks for taking a look. mat is early-stage; the most useful contributions right now
are bug reports, connector files, and probe-suite items.

## Development setup

```bash
git clone https://github.com/docherty/mat.git
cd mat
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

CI runs the same lint and test steps on every push.

## Submitting a connector

A connector is a YAML file that describes one model's endpoint, static metadata, and
coarse capability tiers. Full schema: [docs/connector-spec.md](docs/connector-spec.md).

**Preferred path:** generate yours with the profiling CLI so scores are comparable:

```bash
mat-profile --endpoint openai --base-url <url> --model <name> \
  --auth-env <ENV_VAR_NAME> --output my-connector.yaml
```

**Manual submission:** allowed, but your PR should include:

1. The connector YAML in `connectors/community/<your-id>.yaml`
2. How you produced the capability tiers (probe run log, or justification if hand-authored)
3. Confirmation you did not embed API keys in the file — only `auth_env` names

We validate schema and integrity hash on every connector. Malformed files are rejected.

### Trust model

Third-party connectors are **untrusted input**. Review endpoint URLs and `auth_env`
names before loading a community connector. A future "verified" tier with signing is
planned; until then, treat unknown connectors like any other config from the internet.

### Freshness

Connectors profiled under an old `probe_suite_version` or older than 90 days are flagged
`stale`. Re-run `mat-profile` and open a PR to refresh yours.

## Code changes

- Open an issue first for large features (new loop stages, API surface changes).
- Keep PRs focused. Phase A/B gates in the project plan are sequential — check
  [docs/decisions/](docs/decisions/) for what's already settled.
- Never commit secrets. See [docs/secrets-policy.md](docs/secrets-policy.md).
- Run `ruff check .` and `pytest` before pushing.

## Capability taxonomy

The routing vocabulary (`reasoning`, `coding`, `verification`, etc.) is frozen once
coordinator training starts. Proposals to add or rename tags need an issue with
migration notes — changing the taxonomy implies retraining.

## Questions

Open a GitHub issue with the `question` label.
