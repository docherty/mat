# Phase A result

**Status:** PASS (updated 2026-06-22 18:54 UTC)

## Setup

| Item | Value |
|------|-------|
| Tasks | 10 verifiable coding items (`eval/tasks/phase_a_tasks.json`) |
| Training connectors | alpha-coder, beta-general, gamma-fast |
| Held-out connector | delta-heldout |
| Optimiser | sep-CMA-ES (pycma), 30 generations |
| Fitness | pass@1 − cost penalty per step |
| Seed | 42 |

## Test 1 — trained vs hand-prompted

| Coordinator | pass@1 | Mean steps | Fitness |
|-------------|--------|------------|---------|
| Hand-prompted | 1.00 | 2.20 | 0.890 |
| Trained (3 connectors) | 1.00 | 2.20 | 0.890 |

**Pass?** yes

## Test 2 — held-out connector generalisation

Pool: delta-heldout@phase-a, beta-general@phase-a, gamma-fast@phase-a (held-out swapped in, no retraining).

| Metric | Value |
|--------|-------|
| pass@1 | 1.00 |
| Mean steps | 2.20 |
| Routing alignment | 0.50 |

**Pass?** yes

## Decision

Both pass — proceed to WP2/WP3

## Raw JSON

```json
{
  "prompted": {
    "pass_at_1": 1.0,
    "mean_steps": 2.2,
    "fitness": 0.89
  },
  "trained": {
    "pass_at_1": 1.0,
    "mean_steps": 2.2,
    "fitness": 0.89
  },
  "generalisation": {
    "pass_at_1": 1.0,
    "mean_steps": 2.2,
    "routing_alignment": 0.5021006129959222,
    "pool_ids": [
      "delta-heldout@phase-a",
      "beta-general@phase-a",
      "gamma-fast@phase-a"
    ]
  },
  "test1_pass": true,
  "test2_pass": true,
  "both_pass": true,
  "seed": 42,
  "generations": 30
}
```
