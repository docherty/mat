# Examples vs installed connectors

| Location | Purpose | `profile_method` | Used by mat runtime? |
|----------|---------|------------------|----------------------|
| `connectors/examples/` | Boilerplate, Phase A fixtures, OpenRouter **reference** imports | mixed | **No** (unless you opt in) |
| `~/.config/mat/connectors/` | Your pool — LM Studio, Venice, OpenRouter | `benchmark_import` or `mat_probe` | **Yes** (default) |
| `connectors/installed/` | Optional gitignored copy of same (`MAT_POOL_DIR`) | same | If `MAT_POOL_DIR` set |

**Rule:** Never route production traffic using `connectors/examples/`. Examples may ship with stale or hand scores for documentation.

## Install your LM Studio downloads

Models already on disk under `~/.cache/lm-studio/models` — no re-download:

```bash
export ARTIFICIAL_ANALYSIS_API_KEY=...
mat-sync-aa                  # cache AA catalog → ~/.config/mat/cache/aa_models.json
mat-discover-lmstudio        # write one connector per cached model
```

Edit `endpoint.model_name` only if `curl localhost:1234/v1/models` differs from the guess.

## Compare new API models

```bash
mat-import-aa deepseek-v4-flash --model-name deepseek/deepseek-v4-flash
# or OpenRouter operational + AA scores combined via import tooling
```

Swap pool members by adding/removing YAML files in `~/.config/mat/connectors/`.
