#!/usr/bin/env python3
"""Validate the 153-point hhhg forced-splitting signal contribution."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


EXPECTED_POINTS = 153
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ROOT_PATTERN = re.compile(
    rf"^(run_gg_hhhg_[^_]+_({FLOAT_PATTERN})_({FLOAT_PATTERN}))"
    r"_hhhbb_stage2\.root$"
)
HERWIG_TOTAL = re.compile(
    r"^Total:\s+(\d+)\s+\d+\s+([0-9.+\-eE()]+)"
)
DEFAULT_WORKDIRS = (
    "gg_hhhg_c3d4_10k_hhhbb_153",
)


def parse_args() -> argparse.Namespace:
    campaign_dir = Path(__file__).resolve().parent
    repo_dir = campaign_dir.parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Require one complete hhhg forced-splitting ROOT sample for every "
            "authoritative c3/d4 point and verify its exact merged cross section."
        )
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=campaign_dir,
        help="Signals/c3d4_40k campaign directory.",
    )
    parser.add_argument(
        "--workdir",
        action="append",
        type=Path,
        help=(
            "Forced-splitting work directory; may be repeated. The consolidated "
            "153-point campaign directory is used when omitted."
        ),
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=5.0e-3,
        help="Maximum relative Stage-2 Herwig-vs-merged-LHE cross-section difference.",
    )
    parser.add_argument("--write-csv", type=Path)
    parser.add_argument("--write-json", type=Path)
    parser.set_defaults(repo_dir=repo_dir)
    return parser.parse_args()


def read_herwig_total(out_path: Path) -> tuple[float, int]:
    for line in out_path.read_text(errors="replace").splitlines():
        match = HERWIG_TOTAL.search(line.strip())
        if match:
            xsec_nb_text = re.sub(r"\([^)]*\)", "", match.group(2))
            return float(xsec_nb_text), int(match.group(1))
    raise ValueError(f"no Herwig Total line in {out_path}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    manifest_path = campaign_dir / "metadata" / "points_153.csv"
    base_dir = repo_dir / "HerwigForcedSplitting"
    workdirs = (
        [path.expanduser().resolve() for path in args.workdir]
        if args.workdir
        else [(base_dir / name).resolve() for name in DEFAULT_WORKDIRS]
    )
    issues: list[str] = []

    with manifest_path.open(newline="") as stream:
        expected_rows = list(csv.DictReader(stream))
    expected_coordinates = {
        (float(row["c3"]), float(row["d4"])) for row in expected_rows
    }
    if len(expected_rows) != EXPECTED_POINTS:
        issues.append(
            f"authoritative manifest has {len(expected_rows)} rows; "
            f"expected {EXPECTED_POINTS}"
        )
    if len(expected_coordinates) != len(expected_rows):
        issues.append("authoritative manifest contains duplicate coordinates")

    roots_by_coordinate: dict[tuple[float, float], list[tuple[Path, Path, str]]] = {}
    for workdir in workdirs:
        if not workdir.is_dir():
            issues.append(f"missing forced-splitting workdir: {workdir}")
            continue
        events_dir = workdir / "events"
        for root_path in sorted(events_dir.glob("*.root")):
            if "_var.smear" in root_path.name:
                continue
            match = ROOT_PATTERN.match(root_path.name)
            if not match:
                issues.append(f"unexpected raw ROOT filename: {root_path}")
                continue
            run_name = match.group(1)
            coordinate = (float(match.group(2)), float(match.group(3)))
            roots_by_coordinate.setdefault(coordinate, []).append(
                (root_path, workdir, run_name)
            )

    observed_coordinates = set(roots_by_coordinate)
    missing = sorted(expected_coordinates - observed_coordinates)
    extra = sorted(observed_coordinates - expected_coordinates)
    duplicates = {
        coordinate: entries
        for coordinate, entries in roots_by_coordinate.items()
        if len(entries) != 1
    }
    if missing:
        issues.append(f"{len(missing)} expected hhhbb points are missing")
    if extra:
        issues.append(f"{len(extra)} unexpected hhhbb points were found")
    if duplicates:
        issues.append(f"{len(duplicates)} hhhbb coordinates have duplicate ROOT files")

    audit_rows: list[dict[str, object]] = []
    maximum_relative_difference = 0.0
    total_zero_weight_events = 0
    points_with_zero_weight_events = 0
    for expected in expected_rows:
        coordinate = (float(expected["c3"]), float(expected["d4"]))
        entries = roots_by_coordinate.get(coordinate, [])
        if len(entries) != 1:
            continue
        root_path, workdir, run_name = entries[0]
        merge_summary_path = workdir / run_name / "merge_summary.json"
        out_path = workdir / run_name / f"{run_name}_hhhbb_stage2.out"
        point_issues: list[str] = []
        if root_path.stat().st_size <= 0:
            point_issues.append("empty Stage-2 ROOT")

        merged_xsec_pb = math.nan
        merged_events = -1
        zero_weight_events = -1
        if not merge_summary_path.is_file():
            point_issues.append("missing merge_summary.json")
        else:
            try:
                merge_summary = json.loads(merge_summary_path.read_text())
                merged_xsec_pb = float(merge_summary["merged_xsec_pb"])
                merged_events = int(merge_summary["total_events"])
                zero_weight_events = int(
                    merge_summary.get("zero_weight_events", 0)
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                point_issues.append(f"malformed merge summary: {error}")

        herwig_xsec_nb = math.nan
        herwig_events = -1
        if not out_path.is_file():
            point_issues.append("missing Stage-2 Herwig .out")
        else:
            try:
                herwig_xsec_nb, herwig_events = read_herwig_total(out_path)
            except (OSError, ValueError) as error:
                point_issues.append(str(error))

        exact_xsec_fb = merged_xsec_pb * 1.0e3
        herwig_xsec_fb = herwig_xsec_nb * 1.0e6
        relative_difference = math.nan
        if (
            math.isfinite(exact_xsec_fb)
            and math.isfinite(herwig_xsec_fb)
            and exact_xsec_fb > 0.0
        ):
            relative_difference = abs(herwig_xsec_fb - exact_xsec_fb) / exact_xsec_fb
            maximum_relative_difference = max(
                maximum_relative_difference, relative_difference
            )
            if relative_difference > args.relative_tolerance:
                point_issues.append(
                    "Stage-2/merged xsec relative difference "
                    f"{relative_difference:.6g} exceeds "
                    f"{args.relative_tolerance:.6g}"
                )
        else:
            point_issues.append("nonpositive or nonfinite forced-splitting cross section")
        if merged_events != 10000:
            point_issues.append(
                f"merged LHE has {merged_events} events instead of 10000"
            )
        if herwig_events != 10000:
            point_issues.append(
                f"Stage-2 Herwig Total records {herwig_events} events instead of 10000"
            )
        if zero_weight_events > 0:
            points_with_zero_weight_events += 1
            total_zero_weight_events += zero_weight_events
        if point_issues:
            issues.append(f"{run_name}: " + "; ".join(point_issues))

        audit_rows.append(
            {
                "index": int(expected["index"]),
                "c3": expected["c3"],
                "d4": expected["d4"],
                "run_name": run_name,
                "workdir": str(workdir),
                "root_file": str(root_path),
                "root_bytes": root_path.stat().st_size,
                "merge_summary": str(merge_summary_path),
                "merged_events": merged_events,
                "zero_weight_events": zero_weight_events,
                "merged_xsec_pb": f"{merged_xsec_pb:.12g}",
                "analysis_xsec_fb": f"{exact_xsec_fb:.12g}",
                "herwig_total_xsec_nb": f"{herwig_xsec_nb:.12g}",
                "herwig_total_xsec_fb": f"{herwig_xsec_fb:.12g}",
                "relative_difference": f"{relative_difference:.12g}",
                "status": "ok" if not point_issues else "failed",
            }
        )

    status = "ok" if not issues else "failed"
    summary = {
        "status": status,
        "expected_points": EXPECTED_POINTS,
        "manifest_points": len(expected_rows),
        "selected_hhhbb_roots": len(roots_by_coordinate),
        "missing_coordinates": [list(item) for item in missing],
        "extra_coordinates": [list(item) for item in extra],
        "duplicate_coordinates": {
            f"{coordinate[0]:.12g},{coordinate[1]:.12g}": [
                str(item[0]) for item in entries
            ]
            for coordinate, entries in duplicates.items()
        },
        "events_per_point": 10000,
        "analysis_cross_section_source": "merge_summary.json merged_xsec_pb * 1e3",
        "relative_tolerance": args.relative_tolerance,
        "maximum_relative_difference": maximum_relative_difference,
        "points_with_zero_weight_events": points_with_zero_weight_events,
        "total_zero_weight_events": total_zero_weight_events,
        "workdirs": [str(path) for path in workdirs],
        "issues": issues,
    }
    if args.write_csv and audit_rows:
        write_csv(args.write_csv.expanduser().resolve(), audit_rows)
    if args.write_json:
        write_json(args.write_json.expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
