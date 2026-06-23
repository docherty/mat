# Connector library

YAML files here are **shareable model profiles** — benchmarks, pricing, endpoints, modalities.

The **active pool** is selected separately in repo-root `active.yaml` (list of connector IDs).

```bash
mat-pool list              # active connectors only
mat-pool list --all        # entire library
mat-pool apply --curated connectors/curated/local-dev-pool.yaml
mat-pool sync-pricing
mat-pool verify
```

Onboard new models with `mat-discover-lmstudio` or `mat-import-aa`, then `mat-migrate-pool` or copy YAMLs here manually.
