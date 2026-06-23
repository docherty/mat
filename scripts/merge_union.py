#!/usr/bin/env python3
"""Merge per-connector single JSON reports into union ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.benchmark import BenchmarkRow, BenchmarkSummary, _per_question_best


def load_summary(path: Path) -> BenchmarkSummary:
    data = json.loads(path.read_text())
    if "summaries" in data:
        block = data["summaries"][0]
    else:
        block = data
    rows = [BenchmarkRow(**r) for r in block["rows"]]
    return BenchmarkSummary(
        mode=block["mode"],
        split=block["split"],
        tasks=block["tasks"],
        pass_at_1=block["pass_at_1"],
        mean_steps=block["mean_steps"],
        mean_input_tokens=block["mean_input_tokens"],
        mean_output_tokens=block["mean_output_tokens"],
        mean_cost_usd=block["mean_cost_usd"],
        timestamp=block.get("timestamp", ""),
        rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summaries = [load_summary(p) for p in args.reports]
    ceiling = _per_question_best(summaries)
    per_conn = {s.rows[0].connector_ids[0] if s.rows else "?": s.pass_at_1 for s in summaries}
    report = {
        "mode": "union",
        "per_question_best_pass_at_1": ceiling,
        "per_connector_pass_at_1": per_conn,
        "source_reports": [str(p) for p in args.reports],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
