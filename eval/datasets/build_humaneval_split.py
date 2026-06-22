"""Build fixed HumanEval train/val JSON splits for mat live eval.

HumanEval is Apache-2.0 (OpenAI). We shuffle all 164 tasks with seed 42,
assign 41 to train and 23 to val. Val is never used for coordinator fitness.
"""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
OUT_DIR = Path(__file__).resolve().parents[1] / "tasks"
TRAIN_COUNT = 41
VAL_COUNT = 23
SEED = 42


def _difficulty(task_id: str, index: int) -> float:
    # Spread difficulties for routing experiments; not used for pass/fail.
    return round(0.2 + (index % 17) / 20.0, 2)


def _to_mat_task(row: dict, *, split: str, index: int) -> dict:
    entry_point = row["entry_point"]
    tests = row["test"].rstrip() + f"\n\ncheck({entry_point})\n"
    return {
        "id": row["task_id"],
        "prompt": row["prompt"],
        "entry_point": entry_point,
        "tests": tests,
        "difficulty": _difficulty(row["task_id"], index),
        "required_tags": {"coding": 1.0},
        "split": split,
    }


def download_rows() -> list[dict]:
    with urllib.request.urlopen(HUMANEVAL_URL, timeout=60) as resp:
        raw = gzip.decompress(resp.read())
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def build_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    import random

    rng = random.Random(SEED)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    total = TRAIN_COUNT + VAL_COUNT
    if len(shuffled) < total:
        raise ValueError(f"need at least {total} HumanEval rows, got {len(shuffled)}")
    chosen = shuffled[:total]
    train = [
        _to_mat_task(row, split="train", index=i) for i, row in enumerate(chosen[:TRAIN_COUNT])
    ]
    val = [
        _to_mat_task(row, split="val", index=i)
        for i, row in enumerate(chosen[TRAIN_COUNT:total])
    ]
    return train, val


def write_tasks(train: list[dict], val: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "humaneval_train.json").write_text(json.dumps(train, indent=2) + "\n")
    (OUT_DIR / "humaneval_val.json").write_text(json.dumps(val, indent=2) + "\n")
    manifest = {
        "dataset": "HumanEval",
        "seed": SEED,
        "train_count": len(train),
        "val_count": len(val),
        "rule": "val never used for coordinator fitness; report val pass@1 only",
    }
    (OUT_DIR / "humaneval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HumanEval train/val split")
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional local HumanEval.jsonl.gz (skip download)",
    )
    args = parser.parse_args()
    if args.input:
        raw = gzip.decompress(args.input.read_bytes())
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    else:
        rows = download_rows()
    train, val = build_split(rows)
    write_tasks(train, val)
    print(f"wrote {len(train)} train + {len(val)} val tasks to {OUT_DIR}")


if __name__ == "__main__":
    main()
