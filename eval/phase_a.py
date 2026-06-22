"""Phase A generalization micro-spike."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from connectors.loader import dump_connector, load_connector
from coordinator.policy import PromptedCoordinator
from coordinator.train import train_coordinator
from eval.harness import EvalHarness
from eval.oracle import load_tasks

EXAMPLES = Path(__file__).resolve().parents[1] / "connectors" / "examples"
RESULT_DOC = Path(__file__).resolve().parents[1] / "docs" / "phaseA-result.md"

TRAIN_IDS = ("alpha-coder", "beta-general", "gamma-fast")
HELDOUT_ID = "delta-heldout"


def _fix_connector_hashes() -> None:
    import yaml

    from connectors.schema import Connector

    for path in EXAMPLES.glob("*.yaml"):
        connector = Connector.model_validate(yaml.safe_load(path.read_text()))
        if not connector.profile.integrity_sha256:
            dump_connector(connector, path)


def run_phase_a(*, generations: int = 30, seed: int = 42) -> dict:
    _fix_connector_hashes()

    train_pool = [load_connector(EXAMPLES / f"{name}.yaml") for name in TRAIN_IDS]
    heldout = load_connector(EXAMPLES / f"{HELDOUT_ID}.yaml")
    harness = EvalHarness()
    prompted = PromptedCoordinator()

    prompted_metrics = harness.evaluate_routing(prompted.pick, train_pool, seed=seed)
    trained = train_coordinator(train_pool, generations=generations, seed=seed)
    trained_metrics = harness.evaluate_routing(trained.pick, train_pool, seed=seed)

    # Generalisation: swap held-out into pool (replace weakest coding on hard tasks)
    generalisation_pool = [heldout, train_pool[1], train_pool[2]]
    gen_metrics = harness.evaluate_routing(trained.pick, generalisation_pool, seed=seed)

    tasks = load_tasks()
    routing_align = sum(
        trained.score_connector_alignment(t, generalisation_pool) for t in tasks
    ) / len(tasks)

    train_acc = trained.routing_accuracy(harness.tasks, train_pool)
    test1_pass = trained_metrics.pass_at_1 >= prompted_metrics.pass_at_1 and (
        trained_metrics.pass_at_1 > prompted_metrics.pass_at_1
        or train_acc >= 0.95
    )
    test2_pass = (
        gen_metrics.pass_at_1 >= trained_metrics.pass_at_1 * 0.9
        and routing_align >= 0.35
    )

    return {
        "prompted": {
            "pass_at_1": prompted_metrics.pass_at_1,
            "mean_steps": prompted_metrics.mean_steps,
            "fitness": harness.fitness(prompted_metrics.pass_at_1, prompted_metrics.mean_steps),
        },
        "trained": {
            "pass_at_1": trained_metrics.pass_at_1,
            "mean_steps": trained_metrics.mean_steps,
            "fitness": harness.fitness(trained_metrics.pass_at_1, trained_metrics.mean_steps),
        },
        "generalisation": {
            "pass_at_1": gen_metrics.pass_at_1,
            "mean_steps": gen_metrics.mean_steps,
            "routing_alignment": routing_align,
            "pool_ids": [c.id for c in generalisation_pool],
        },
        "test1_pass": test1_pass,
        "test2_pass": test2_pass,
        "both_pass": test1_pass and test2_pass,
        "seed": seed,
        "generations": generations,
    }


def _write_result_doc(results: dict) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    p = results["prompted"]
    t = results["trained"]
    g = results["generalisation"]
    decision = (
        "Both pass — proceed to WP2/WP3"
        if results["both_pass"]
        else "Review pivot options (prompted-only framework or stop)"
    )
    body = f"""# Phase A result

**Status:** {"PASS" if results["both_pass"] else "FAIL"} (updated {ts})

## Setup

| Item | Value |
|------|-------|
| Tasks | 10 verifiable coding items (`eval/tasks/phase_a_tasks.json`) |
| Training connectors | {", ".join(TRAIN_IDS)} |
| Held-out connector | {HELDOUT_ID} |
| Optimiser | sep-CMA-ES (pycma), {results["generations"]} generations |
| Fitness | pass@1 − cost penalty per step |
| Seed | {results["seed"]} |

## Test 1 — trained vs hand-prompted

| Coordinator | pass@1 | Mean steps | Fitness |
|-------------|--------|------------|---------|
| Hand-prompted | {p["pass_at_1"]:.2f} | {p["mean_steps"]:.2f} | {p["fitness"]:.3f} |
| Trained (3 connectors) | {t["pass_at_1"]:.2f} | {t["mean_steps"]:.2f} | {t["fitness"]:.3f} |

**Pass?** {"yes" if results["test1_pass"] else "no"}

## Test 2 — held-out connector generalisation

Pool: {", ".join(g["pool_ids"])} (held-out swapped in, no retraining).

| Metric | Value |
|--------|-------|
| pass@1 | {g["pass_at_1"]:.2f} |
| Mean steps | {g["mean_steps"]:.2f} |
| Routing alignment | {g["routing_alignment"]:.2f} |

**Pass?** {"yes" if results["test2_pass"] else "no"}

## Decision

{decision}

## Raw JSON

```json
{json.dumps(results, indent=2)}
```
"""
    RESULT_DOC.write_text(body)


def main() -> None:
    results = run_phase_a()
    _write_result_doc(results)
    print(json.dumps(results, indent=2))
    if results["both_pass"]:
        print("\nPhase A: PASS")
    else:
        print("\nPhase A: FAIL — see docs/phaseA-result.md")


if __name__ == "__main__":
    main()
