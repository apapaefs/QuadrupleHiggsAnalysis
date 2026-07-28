#!/usr/bin/env python3
"""Validate the exact c3/d4 40k campaign and its cross-section handoff."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from pathlib import Path


EXPECTED_POINTS = 153
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
INTEGRATED_WEIGHT = re.compile(
    rf"Integrated weight\s*\(pb\)\s*:\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)
HERWIG_TOTAL = re.compile(
    r"^Total:\s+(\d+)\s+\d+\s+([0-9.+\-eE()]+)"
)


def parse_args() -> argparse.Namespace:
    campaign_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Require the authoritative 153-point manifest, exact Herwig raw-ROOT "
            "selection, 40k completion counts, and consistent MG5/Herwig cross sections."
        )
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=campaign_dir,
        help="Signals/c3d4_40k campaign directory.",
    )
    parser.add_argument(
        "--herwig-dir",
        type=Path,
        default=campaign_dir.parents[1] / "HerwigSignalPoints" / "c3d4_40k",
        help="HerwigSignalPoints/c3d4_40k directory.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=5.0e-4,
        help="Maximum relative Herwig-vs-MG5 cross-section difference.",
    )
    parser.add_argument("--write-csv", type=Path)
    parser.add_argument("--write-json", type=Path)
    return parser.parse_args()


def read_mg5_xsec_pb(lhe_path: Path) -> float:
    with gzip.open(lhe_path, mode="rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = INTEGRATED_WEIGHT.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"no Integrated weight (pb) line in {lhe_path}")


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
    herwig_dir = args.herwig_dir.expanduser().resolve()
    manifest_path = campaign_dir / "metadata" / "points_153.csv"
    events_dir = campaign_dir / "Events"
    herwig_events_dir = herwig_dir / "events"
    issues: list[str] = []

    if not manifest_path.is_file():
        print(f"ERROR: missing manifest: {manifest_path}", file=sys.stderr)
        return 1
    with manifest_path.open(newline="") as stream:
        points = list(csv.DictReader(stream))

    if len(points) != EXPECTED_POINTS:
        issues.append(
            f"manifest has {len(points)} rows; expected {EXPECTED_POINTS}"
        )
    run_names = [row["run_name"] for row in points]
    coordinates = [(float(row["c3"]), float(row["d4"])) for row in points]
    if len(set(run_names)) != len(run_names):
        issues.append("manifest contains duplicate run names")
    if len(set(coordinates)) != len(coordinates):
        issues.append("manifest contains duplicate (c3,d4) coordinates")

    expected_roots = {f"HW-{run_name}.root" for run_name in run_names}
    selected_roots = {
        path.name
        for path in herwig_events_dir.rglob("*.root")
        if "_var.smear" not in path.name
        and "debug" not in path.name.lower()
        and "smoke" not in path.name.lower()
    }
    missing_selected = sorted(expected_roots - selected_roots)
    unexpected_selected = sorted(selected_roots - expected_roots)
    if missing_selected:
        issues.append(
            f"{len(missing_selected)} expected production ROOT files are missing"
        )
    if unexpected_selected:
        issues.append(
            f"{len(unexpected_selected)} unexpected production ROOT files would be selected"
        )

    audit_rows: list[dict[str, object]] = []
    max_relative_difference = 0.0
    for row in points:
        run_name = row["run_name"]
        target_events = int(row["target_events"])
        lhe_path = events_dir / run_name / "unweighted_events.lhe.gz"
        root_path = herwig_events_dir / f"HW-{run_name}.root"
        out_path = herwig_dir / f"HW-{run_name}.out"
        point_issues: list[str] = []

        if not lhe_path.is_file():
            point_issues.append("missing LHE")
        if not root_path.is_file() or root_path.stat().st_size <= 0:
            point_issues.append("missing or empty Herwig ROOT")
        if not out_path.is_file():
            point_issues.append("missing Herwig .out")

        mg5_pb = math.nan
        herwig_nb = math.nan
        generated = -1
        relative_difference = math.nan
        if lhe_path.is_file():
            try:
                mg5_pb = read_mg5_xsec_pb(lhe_path)
            except (OSError, ValueError) as error:
                point_issues.append(str(error))
        if out_path.is_file():
            try:
                herwig_nb, generated = read_herwig_total(out_path)
            except (OSError, ValueError) as error:
                point_issues.append(str(error))

        mg5_fb = mg5_pb * 1.0e3
        herwig_fb = herwig_nb * 1.0e6
        if math.isfinite(mg5_fb) and math.isfinite(herwig_fb) and mg5_fb > 0.0:
            relative_difference = abs(herwig_fb - mg5_fb) / mg5_fb
            max_relative_difference = max(
                max_relative_difference, relative_difference
            )
            if relative_difference > args.relative_tolerance:
                point_issues.append(
                    "Herwig/MG5 xsec relative difference "
                    f"{relative_difference:.6g} exceeds {args.relative_tolerance:.6g}"
                )
        else:
            point_issues.append("nonpositive or nonfinite cross section")
        if generated != target_events:
            point_issues.append(
                f"Herwig Total records {generated} generated events, expected {target_events}"
            )

        if point_issues:
            issues.append(f"{run_name}: " + "; ".join(point_issues))
        audit_rows.append(
            {
                "index": int(row["index"]),
                "c3": row["c3"],
                "d4": row["d4"],
                "run_name": run_name,
                "target_events": target_events,
                "herwig_generated_events": generated,
                "mg5_xsec_pb": f"{mg5_pb:.12g}",
                "mg5_xsec_fb": f"{mg5_fb:.12g}",
                "herwig_total_xsec_nb": f"{herwig_nb:.12g}",
                "analysis_xsec_fb": f"{herwig_fb:.12g}",
                "relative_difference": f"{relative_difference:.12g}",
                "root_bytes": root_path.stat().st_size if root_path.is_file() else 0,
                "status": "ok" if not point_issues else "failed",
            }
        )

    status = "ok" if not issues else "failed"
    summary = {
        "status": status,
        "expected_points": EXPECTED_POINTS,
        "manifest_points": len(points),
        "unique_coordinates": len(set(coordinates)),
        "selected_production_roots": len(selected_roots),
        "missing_selected_roots": missing_selected,
        "unexpected_selected_roots": unexpected_selected,
        "target_events_per_point": sorted(
            {int(row["target_events"]) for row in points}
        ),
        "cross_section_handoff": "MG5 pb -> Herwig Total nb -> analysis fb",
        "analysis_conversion": "Herwig Total [nb] * 1e6",
        "relative_tolerance": args.relative_tolerance,
        "maximum_relative_difference": max_relative_difference,
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
