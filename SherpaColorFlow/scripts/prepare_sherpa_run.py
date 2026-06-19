#!/usr/bin/env python3
"""Prepare a Sherpa example run directory with MPI-aware total event counts."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "gg4b2c2j": ROOT / "Examples" / "GluonFusion_GG_2bbbar_ccbar_2j_LHE",
    "gg4b4c": ROOT / "Examples" / "GluonFusion_GG_2bbbar_2ccbar_LHE",
    "gg4b4j": ROOT / "Examples" / "GluonFusion_GG_2bbbar_4j_LHE",
    "gg6b2j": ROOT / "Examples" / "GluonFusion_GG_3bbbar_2j_LHE",
    "gg6bcc": ROOT / "Examples" / "GluonFusion_GG_3bbbar_ccbar_LHE",
    "gg8b": ROOT / "Examples" / "GluonFusion_GG_4bbbar_LHE",
    "z6b": ROOT / "Examples" / "PP_Z_6bbbar_Zbb_DecayOS_LHE",
}


def replace_card_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    replacement = f"{key}: {value}"
    if not pattern.search(text):
        raise SystemExit(f"Could not find '{key}:' in Sherpa.yaml")
    return pattern.sub(replacement, text, count=1)


def upsert_card_value(text: str, key: str, value: str, after_key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    replacement = f"{key}: {value}"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)

    anchor = re.compile(rf"^({re.escape(after_key)}:\s*.*)$", re.MULTILINE)
    if not anchor.search(text):
        raise SystemExit(f"Could not find '{after_key}:' in Sherpa.yaml")
    return anchor.sub(rf"\1\n{replacement}", text, count=1)


def enforce_mpi_run_defaults(text: str) -> str:
    text = upsert_card_value(text, "MPI_EVENT_MODE", "1", "EVENTS")
    text = upsert_card_value(text, "BATCH_MODE", "5", "MPI_EVENT_MODE")
    return upsert_card_value(text, "EVENT_DISPLAY_INTERVAL", "100", "BATCH_MODE")


def replace_event_output(text: str, prefix: str) -> str:
    pattern = re.compile(r"^EVENT_OUTPUT:\s*LHEF\[[^\]]+\]\s*$", re.MULTILINE)
    replacement = f"EVENT_OUTPUT: LHEF[{prefix}]"
    if not pattern.search(text):
        raise SystemExit("Could not find 'EVENT_OUTPUT: LHEF[...]' in Sherpa.yaml")
    return pattern.sub(replacement, text, count=1)


def copy_example(example_dir: Path, run_dir: Path, force: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        if not force:
            raise SystemExit(f"{run_dir} exists and is not empty; pass --force to overwrite Sherpa.yaml")
        shutil.copy2(example_dir / "Sherpa.yaml", run_dir / "Sherpa.yaml")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in example_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, run_dir / path.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", choices=sorted(EXAMPLES))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--total-events", type=int, help="desired total events over all MPI ranks")
    parser.add_argument("--np", type=int, default=1, help="MPI rank count")
    parser.add_argument("--output-prefix", help="LHEF output prefix")
    parser.add_argument("--round-up", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help="overwrite Sherpa.yaml in an existing run directory")
    args = parser.parse_args()

    if args.np < 1:
        raise SystemExit("--np must be positive")
    if args.total_events is not None and args.total_events < 1:
        raise SystemExit("--total-events must be positive")

    example_dir = EXAMPLES[args.example]
    copy_example(example_dir, args.run_dir, args.force)

    card = args.run_dir / "Sherpa.yaml"
    text = card.read_text()
    text = enforce_mpi_run_defaults(text)

    if args.total_events is not None:
        text = replace_card_value(text, "EVENTS", str(args.total_events))

    if args.output_prefix:
        text = replace_event_output(text, args.output_prefix)

    card.write_text(text)

    print(f"Prepared {args.example} run in {args.run_dir}")
    if args.total_events is not None:
        print(f"EVENTS total: {args.total_events}")
        print(f"MPI_EVENT_MODE: 1, so Sherpa distributes this total over -np {args.np}")
    print("BATCH_MODE: 5")
    print("EVENT_DISPLAY_INTERVAL: 100")
    print("Run command:")
    print(f"  mpirun --use-hwthread-cpus -np {args.np} --bind-to hwthread --map-by hwthread Sherpa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
