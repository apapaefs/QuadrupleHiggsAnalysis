#!/usr/bin/env python3
"""Summarize completion of the extended-scalar Herwig mass-scan campaign."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


COMPLETE_MARKER = "Number of events that pass basic cuts"


def classify(repo: Path, row: dict[str, str]) -> str:
    card = repo / row["card"]
    output = repo / row["output_root"]
    log = repo / row["run_log"]
    if output.is_file() and output.stat().st_size > 0 and log.is_file():
        text = log.read_text(errors="replace")
        if COMPLETE_MARKER in text:
            if output.stat().st_mtime >= card.stat().st_mtime and log.stat().st_mtime >= card.stat().st_mtime:
                return "complete"
            return "stale"
    if log.is_file():
        text = log.read_text(errors="replace")
        if "exit_code:" in text or "Error" in text or "Exception" in text:
            return "failed_or_incomplete"
        return "running_or_interrupted"
    return "pending"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="List every non-complete point")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = Path(__file__).resolve().parent
    repo = campaign.parents[1]
    with (campaign / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    overall: Counter[str] = Counter()
    per_scenario: dict[str, Counter[str]] = {}
    incomplete: list[tuple[str, str, str]] = []
    for row in rows:
        status = classify(repo, row)
        overall[status] += 1
        per_scenario.setdefault(row["scenario"], Counter())[status] += 1
        if status != "complete":
            incomplete.append((row["scenario"], row["run_name"], status))

    order = ("complete", "pending", "running_or_interrupted", "failed_or_incomplete", "stale")
    print(f"Total points: {len(rows)}")
    for scenario in ("direct", "cascade"):
        counts = per_scenario.get(scenario, Counter())
        summary = ", ".join(f"{key}={counts[key]}" for key in order if counts[key])
        print(f"{scenario:8s}: {summary or 'none'}")
    print("overall : " + ", ".join(f"{key}={overall[key]}" for key in order if overall[key]))

    if args.verbose:
        for scenario, run_name, status in incomplete:
            print(f"{scenario:8s} {status:24s} {run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
