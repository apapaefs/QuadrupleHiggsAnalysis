#!/usr/bin/env python3
"""Fit the Sherpa ``gg -> hh + 4b`` cross section as a function of c3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_DIR / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from hh4b_c3_xsec import (  # noqa: E402
    evaluate_hh4b_c3_fit,
    fit_hh4b_c3_cross_section,
)


EXPECTED_POINTS = {
    "c3_m20": -20.0,
    "c3_m2": -2.0,
    "c3_m1": -1.0,
    "c3_0": 0.0,
    "c3_p20": 20.0,
}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
PROCESS_RESULT = re.compile(
    r":\s*"
    r"(?P<xsec>[0-9.+\-eE]+)\s+pb\s+\+-\s+\(\s*"
    r"(?P<error>[0-9.+\-eE]+)\s+pb\s*=\s*"
    r"(?P<percent>[0-9.+\-eE]+)\s*%"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_final_log_result(path: Path) -> dict[str, float]:
    matches = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        match = PROCESS_RESULT.search(line)
        if match:
            matches.append(
                {
                    "cross_section_pb": float(match.group("xsec")),
                    "integration_error_pb": float(match.group("error")),
                    "relative_error_percent": float(match.group("percent")),
                }
            )
    if not matches:
        raise ValueError(f"No final Sherpa process cross section found in {path}")
    return matches[-1]


def _validate_result_archive(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            broken_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid Sherpa result archive {path}: {exc}") from exc
    if broken_member is not None:
        raise ValueError(
            f"Invalid Sherpa result archive {path}: corrupt member "
            f"{broken_member}"
        )


def _accepted_overrides(campaign_dir: Path) -> dict[str, dict[str, str]]:
    path = campaign_dir / "accepted_results.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["label"]): row for row in rows}


def collect_campaign_points(campaign_dir: Path) -> list[dict[str, object]]:
    """Collect the exact five audited integration points."""

    overrides = _accepted_overrides(campaign_dir)
    points = []
    missing = []
    for label, c3 in EXPECTED_POINTS.items():
        point_dir = campaign_dir / label
        override = overrides.get(label)
        if override is not None:
            log_path = campaign_dir / override["log"]
            archive_path = campaign_dir / override["result_archive"]
            result = {
                "cross_section_pb": float(override["cross_section_pb"]),
                "integration_error_pb": float(
                    override["integration_error_pb"]
                ),
                "relative_error_percent": float(
                    override["relative_error_percent"]
                ),
            }
            status = str(override["status"])
        else:
            logs = sorted(point_dir.glob("integrate*.log"))
            archives = sorted(point_dir.glob("Results*.zip"))
            if not logs or not archives:
                missing.append(label)
                continue
            log_path = logs[-1]
            archive_path = archives[-1]
            result = _parse_final_log_result(log_path)
            status = "completed_integration"
        if not log_path.is_file() or not archive_path.is_file():
            missing.append(label)
            continue
        _validate_result_archive(archive_path)
        points.append(
            {
                "label": label,
                "c3": c3,
                "kappa_lambda": 1.0 + c3,
                "status": status,
                **result,
                "log": str(log_path.relative_to(campaign_dir)),
                "log_sha256": _sha256(log_path),
                "result_archive": str(
                    archive_path.relative_to(campaign_dir)
                ),
                "result_archive_sha256": _sha256(archive_path),
            }
        )
    if missing:
        raise ValueError(
            "Incomplete hh+4b c3 integration campaign; missing completed "
            "results for: " + ", ".join(missing)
        )
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-relative-error-percent",
        type=float,
        default=1.5,
        help="Reject any point less precise than this percentage (default: 1.5).",
    )
    args = parser.parse_args()

    campaign_dir = args.campaign_dir.resolve()
    try:
        points = collect_campaign_points(campaign_dir)
    except ValueError as exc:
        parser.error(str(exc))
    imprecise = [
        point
        for point in points
        if float(point["relative_error_percent"])
        > float(args.max_relative_error_percent)
    ]
    if imprecise:
        detail = ", ".join(
            f"{point['label']}={float(point['relative_error_percent']):g}%"
            for point in imprecise
        )
        raise SystemExit(
            "Refusing to fit imprecise hh+4b integration points "
            f"(limit {args.max_relative_error_percent:g}%): {detail}"
        )

    try:
        source_campaign = str(campaign_dir.relative_to(REPO_DIR))
    except ValueError:
        source_campaign = str(campaign_dir)
    fit = fit_hh4b_c3_cross_section(
        points,
        source_campaign=source_campaign,
    )
    fit["max_accepted_relative_error_percent"] = float(
        args.max_relative_error_percent
    )
    fit["evaluation_range_c3"] = [-20.0, 20.0]
    for index in range(401):
        c3 = -20.0 + 40.0 * index / 400.0
        evaluate_hh4b_c3_fit(fit, c3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print(
        "sigma_pb(c3) = "
        f"{fit['coefficients_pb'][0]:.12g} + "
        f"{fit['coefficients_pb'][1]:.12g}*c3 + "
        f"{fit['coefficients_pb'][2]:.12g}*c3^2"
    )
    print(f"chi2 / ndof = {fit['chi2']:.6g} / {fit['ndof']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
