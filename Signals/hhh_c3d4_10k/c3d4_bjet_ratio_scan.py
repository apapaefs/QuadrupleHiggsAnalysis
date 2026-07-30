#!/usr/bin/env python3
"""Analyze and combine HHH, HHHbb, and HHHH >=6 b-tag cross sections."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO_DIR = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

from campaign import (  # noqa: E402
    ANALYSIS_ID,
    EXPECTED_POINTS,
    INTEGRATED_WEIGHT,
    PRODUCTION_CPUS,
    Point,
    atomic_write_csv,
    atomic_write_json,
    default_source_repo,
    find_lhe,
    hhhbb_inventory,
    load_points,
    paths_from_args,
    require_cpu_budget,
)


HBB_BRANCHING_RATIO = 0.5824
BTAG_EFFICIENCY = 0.85
PT_CUT_GEV = 20.0
ABS_ETA_CUT = 2.5
SMEARING_SEED = 14101983
XSEC_RELATIVE_TOLERANCE = 5.0e-4
HHHBB_XSEC_RELATIVE_TOLERANCE = 5.0e-3
# consolidated_sources.json records analysis_xsec_fb to 12 significant
# decimal digits, whereas merge_summary.json keeps the full binary value.
HHHBB_CONSOLIDATED_RELATIVE_TOLERANCE = 5.0e-12
RATIO_LEVELS = (0.01, 0.05, 0.1, 1.0, 10.0)
RATIO_LEVEL_STYLES = {
    0.01: {"color": "black", "linestyle": "dotted"},
    0.05: {"color": "blue", "linestyle": "dashed"},
    0.1: {"color": "red", "linestyle": "solid"},
}
RATIO_FALLBACK_STYLE = {"color": "black", "linestyle": "solid"}
PLOT_C3_RANGE = (-20.0, 20.0)
PLOT_D4_RANGE = (-300.0, 300.0)
CONTOUR_GRID_POINTS_PER_AXIS = 601
HERWIG_TOTAL = re.compile(r"^Total:\s+(\d+)\s+\d+\s+(\S+)")
PARENTHETICAL_VALUE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:\((\d+)\))?([eE][+-]?\d+)?$"
)
CATEGORIES = ("exact6", "exact7", "ge8", "ge6")
PRIMARY_RATIO_STEM = (
    "c3d4_hhhh_ge6btag_over_hhh_plus_hhhbb_ge6btag_ratio_points"
)
DIAGNOSTIC_RATIO_STEM = (
    "c3d4_hhhh_ge6btag_over_hhh_ge6btag_ratio_points"
)
PRIMARY_PLOT_STEM = (
    "c3d4_hhhh_ge6btag_over_hhh_plus_hhhbb_ge6btag_ratio_contours"
)
DIAGNOSTIC_PLOT_STEM = (
    "c3d4_hhhh_ge6btag_over_hhh_ge6btag_ratio_contours"
)


@dataclass(frozen=True)
class AnalysisPaths:
    source_repo: Path
    mg5_process: Path
    hhh_herwig_dir: Path
    hhhh_herwig_dir: Path
    hhhbb_workdir: Path
    results_dir: Path
    analyzer: Path
    points_file: Path


@dataclass(frozen=True)
class SampleInput:
    process: str
    point: Point
    run_name: str
    root_file: Path
    hbb_power: int
    expected_events: int
    herwig_out: Path | None = None
    lhe_file: Path | None = None
    merge_summary: Path | None = None
    consolidated_xsec_fb: float | None = None


def analysis_paths(args: argparse.Namespace) -> AnalysisPaths:
    campaign_paths = paths_from_args(args)
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir
        else CAMPAIGN_DIR / "results"
    )
    analyzer = (
        args.analyzer.expanduser().resolve()
        if args.analyzer
        else REPO_DIR / "Code" / "BJetMultiplicityAnalysis"
    )
    return AnalysisPaths(
        source_repo=campaign_paths.source_repo,
        mg5_process=campaign_paths.mg5_process,
        hhh_herwig_dir=campaign_paths.herwig_dir,
        hhhh_herwig_dir=campaign_paths.source_repo
        / "HerwigSignalPoints"
        / "c3d4_40k",
        hhhbb_workdir=campaign_paths.source_repo
        / "HerwigForcedSplitting"
        / "gg_hhhg_c3d4_10k_hhhbb_153",
        results_dir=results_dir,
        analyzer=analyzer,
        points_file=campaign_paths.points_file,
    )


def binomial_probability(trials: int, successes: int, efficiency: float) -> float:
    if successes < 0 or successes > trials:
        return 0.0
    if efficiency == 0.0:
        return 1.0 if successes == 0 else 0.0
    if efficiency == 1.0:
        return 1.0 if successes == trials else 0.0
    logarithm = (
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
        + successes * math.log(efficiency)
        + (trials - successes) * math.log1p(-efficiency)
    )
    return math.exp(logarithm)


def binomial_tag_probabilities(
    truth_bjets: int, efficiency: float = BTAG_EFFICIENCY
) -> dict[str, float]:
    exact6 = binomial_probability(truth_bjets, 6, efficiency)
    exact7 = binomial_probability(truth_bjets, 7, efficiency)
    below8 = sum(
        binomial_probability(truth_bjets, successes, efficiency)
        for successes in range(8)
    )
    ge8 = max(0.0, min(1.0, 1.0 - below8)) if truth_bjets >= 8 else 0.0
    return {
        "exact6": exact6,
        "exact7": exact7,
        "ge8": ge8,
        "ge6": exact6 + exact7 + ge8,
    }


def parse_parenthetical_number(text: str) -> tuple[float, float]:
    match = PARENTHETICAL_VALUE.match(text.strip())
    if not match:
        raise ValueError(f"cannot parse Herwig value {text!r}")
    mantissa_text, uncertainty_digits, exponent_text = match.groups()
    exponent = int(exponent_text[1:]) if exponent_text else 0
    central = float(mantissa_text) * 10.0**exponent
    uncertainty = 0.0
    if uncertainty_digits:
        decimal_places = (
            len(mantissa_text.split(".", 1)[1])
            if "." in mantissa_text
            else 0
        )
        uncertainty = (
            int(uncertainty_digits)
            * 10.0 ** (-decimal_places)
            * 10.0**exponent
        )
    return central, uncertainty


def read_herwig_total(path: Path) -> tuple[float, float, int]:
    for line in path.read_text(errors="replace").splitlines():
        match = HERWIG_TOTAL.match(line.strip())
        if match:
            central_nb, uncertainty_nb = parse_parenthetical_number(
                match.group(2)
            )
            return central_nb, uncertainty_nb, int(match.group(1))
    raise ValueError(f"no Herwig Total line in {path}")


def read_lhe_xsec_pb(path: Path) -> float:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = INTEGRATED_WEIGHT.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"no Integrated weight (pb) line in {path}")


def hhhh_run_name(point: Point) -> str:
    return f"run_gg_4h_5_{point.c3}_{point.d4}"


def hhh_inputs(paths: AnalysisPaths, points: list[Point]) -> list[SampleInput]:
    samples = []
    for point in points:
        run_name = point.run_name
        samples.append(
            SampleInput(
                process="hhh",
                point=point,
                run_name=run_name,
                root_file=paths.hhh_herwig_dir
                / "events"
                / f"HW-{run_name}.root",
                herwig_out=paths.hhh_herwig_dir / f"HW-{run_name}.out",
                lhe_file=find_lhe(
                    paths.mg5_process / "Events" / run_name
                ),
                hbb_power=3,
                expected_events=10_000,
            )
        )
    return samples


def hhhh_inputs(paths: AnalysisPaths, points: list[Point]) -> list[SampleInput]:
    samples = []
    for point in points:
        run_name = hhhh_run_name(point)
        samples.append(
            SampleInput(
                process="hhhh",
                point=point,
                run_name=run_name,
                root_file=paths.hhhh_herwig_dir
                / "events"
                / f"HW-{run_name}.root",
                herwig_out=paths.hhhh_herwig_dir / f"HW-{run_name}.out",
                lhe_file=paths.source_repo
                / "Signals"
                / "c3d4_40k"
                / "Events"
                / run_name
                / "unweighted_events.lhe.gz",
                hbb_power=4,
                expected_events=40_000,
            )
        )
    return samples


def hhhbb_inputs(paths: AnalysisPaths, points: list[Point]) -> list[SampleInput]:
    consolidated_path = paths.hhhbb_workdir / "consolidated_sources.json"
    payload = json.loads(consolidated_path.read_text())
    consolidated_points = {
        (float(row["c3"]), float(row["d4"])): row
        for row in payload["points"]
    }
    expected_coordinates = {point.coordinate for point in points}
    missing = expected_coordinates - set(consolidated_points)
    extra = (
        set(consolidated_points) - expected_coordinates
        if len(points) == EXPECTED_POINTS
        else set()
    )
    if missing or extra:
        raise ValueError(
            "HHHbb consolidated point mismatch: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    samples = []
    for point in points:
        metadata = consolidated_points[point.coordinate]
        run_name = str(metadata["run_name"])
        samples.append(
            SampleInput(
                process="hhhbb",
                point=point,
                run_name=run_name,
                root_file=paths.hhhbb_workdir
                / "events"
                / f"{run_name}_hhhbb_stage2.root",
                herwig_out=paths.hhhbb_workdir
                / run_name
                / f"{run_name}_hhhbb_stage2.out",
                merge_summary=paths.hhhbb_workdir
                / run_name
                / "merge_summary.json",
                consolidated_xsec_fb=float(metadata["analysis_xsec_fb"]),
                hbb_power=3,
                expected_events=10_000,
            )
        )
    return samples


def all_inputs(
    paths: AnalysisPaths, points: list[Point]
) -> dict[str, list[SampleInput]]:
    return {
        "hhh": hhh_inputs(paths, points),
        "hhhbb": hhhbb_inputs(paths, points),
        "hhhh": hhhh_inputs(paths, points),
    }


def validate_sample_inventory(samples: Iterable[SampleInput]) -> list[str]:
    issues: list[str] = []
    for sample in samples:
        if not sample.root_file.is_file() or sample.root_file.stat().st_size <= 0:
            issues.append(f"{sample.process} {sample.run_name}: missing ROOT")
        if sample.herwig_out is not None and not sample.herwig_out.is_file():
            issues.append(
                f"{sample.process} {sample.run_name}: missing Herwig .out"
            )
        if sample.lhe_file is None or (
            sample.lhe_file is not None and not sample.lhe_file.is_file()
        ):
            if sample.process != "hhhbb":
                issues.append(f"{sample.process} {sample.run_name}: missing LHE")
        if (
            sample.merge_summary is not None
            and not sample.merge_summary.is_file()
        ):
            issues.append(
                f"{sample.process} {sample.run_name}: missing merge summary"
            )
    return issues


def cache_path(paths: AnalysisPaths, sample: SampleInput) -> Path:
    coordinate = (
        f"{sample.point.index:03d}_{sample.point.c3}_{sample.point.d4}"
        .replace("-", "m")
        .replace(".", "p")
    )
    return (
        paths.results_dir
        / "cache"
        / sample.process
        / f"{coordinate}.json"
    )


def analysis_cache_is_current(path: Path, sample: SampleInput) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    stat = sample.root_file.stat()
    return (
        payload.get("format_version") == 1
        and payload.get("analysis_id") == ANALYSIS_ID
        and payload.get("process") == sample.process
        and Path(payload.get("input_file", "")).resolve()
        == sample.root_file.resolve()
        and int(payload.get("input_size_bytes", -1)) == stat.st_size
        and int(payload.get("input_mtime_unix", -1)) == int(stat.st_mtime)
        and int(payload.get("processed_events", -1))
        == sample.expected_events
        and math.isclose(
            float(payload.get("btag_efficiency", math.nan)),
            BTAG_EFFICIENCY,
        )
        and math.isclose(
            float(payload.get("jet_pt_cut_gev", math.nan)), PT_CUT_GEV
        )
        and math.isclose(
            float(payload.get("jet_abs_eta_cut", math.nan)), ABS_ETA_CUT
        )
        and int(payload.get("smearing_seed", -1)) == SMEARING_SEED
    )


def run_analyzer(
    paths: AnalysisPaths,
    sample: SampleInput,
    force: bool = False,
    max_events: int | None = None,
    output: Path | None = None,
) -> Path:
    if not paths.analyzer.is_file():
        raise FileNotFoundError(f"missing analyzer executable: {paths.analyzer}")
    target = output or cache_path(paths, sample)
    if (
        output is None
        and not force
        and analysis_cache_is_current(target, sample)
    ):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    command = [
        str(paths.analyzer),
        str(sample.root_file),
        "--output",
        str(temporary),
        "--process",
        sample.process,
        "--analysis-id",
        ANALYSIS_ID,
        "--seed",
        str(SMEARING_SEED),
        "--pt-cut",
        str(PT_CUT_GEV),
        "--eta-cut",
        str(ABS_ETA_CUT),
        "--btag-efficiency",
        str(BTAG_EFFICIENCY),
    ]
    if max_events is not None:
        command.extend(["--max-events", str(max_events)])
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command,
        cwd=REPO_DIR,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = paths.results_dir / "logs" / sample.process / f"{sample.run_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "$ " + " ".join(command) + "\n" + completed.stdout
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"analyzer failed for {sample.run_name}; see {log}"
        )
    temporary.replace(target)
    return target


def normalization(sample: SampleInput) -> dict[str, object]:
    if sample.process in {"hhh", "hhhh"}:
        if sample.herwig_out is None or sample.lhe_file is None:
            raise ValueError(f"missing normalization input for {sample.run_name}")
        central_nb, uncertainty_nb, generated_events = read_herwig_total(
            sample.herwig_out
        )
        inclusive_pb = central_nb * 1.0e3
        uncertainty_pb = uncertainty_nb * 1.0e3
        mg5_pb = read_lhe_xsec_pb(sample.lhe_file)
        relative_difference = abs(inclusive_pb - mg5_pb) / mg5_pb
        issues = []
        if generated_events != sample.expected_events:
            issues.append(
                f"Herwig records {generated_events} events, "
                f"expected {sample.expected_events}"
            )
        if relative_difference > XSEC_RELATIVE_TOLERANCE:
            issues.append(
                f"Herwig/MG5 relative difference {relative_difference:.6g} "
                f"exceeds {XSEC_RELATIVE_TOLERANCE:.6g}"
            )
        return {
            "inclusive_xsec_pb": inclusive_pb,
            "inclusive_xsec_error_pb": uncertainty_pb,
            "generated_events": generated_events,
            "reference_xsec_pb": mg5_pb,
            "relative_difference": relative_difference,
            "herwig_reference_xsec_pb": inclusive_pb,
            "herwig_reference_xsec_error_pb": uncertainty_pb,
            "herwig_relative_difference": 0.0,
            "zero_weight_events": 0,
            "probe_trial_weight_correction_applied": False,
            "normalization_source": "Herwig Total [nb] * 1e3",
            "status": "ok" if not issues else "failed",
            "issues": issues,
        }

    if sample.merge_summary is None or sample.herwig_out is None:
        raise ValueError(
            f"missing HHHbb normalization metadata for {sample.run_name}"
        )
    payload = json.loads(sample.merge_summary.read_text())
    inclusive_pb = float(payload["merged_xsec_pb"])
    uncertainty_pb = float(payload.get("merged_xsec_error_pb", 0.0))
    generated_events = int(payload["total_events"])
    stage2_nb, stage2_uncertainty_nb, stage2_events = read_herwig_total(
        sample.herwig_out
    )
    stage2_pb = stage2_nb * 1.0e3
    stage2_uncertainty_pb = stage2_uncertainty_nb * 1.0e3
    consolidated_pb = (
        float(sample.consolidated_xsec_fb) / 1.0e3
        if sample.consolidated_xsec_fb is not None
        else math.nan
    )
    relative_difference = (
        abs(inclusive_pb - consolidated_pb) / inclusive_pb
        if inclusive_pb > 0.0
        else math.inf
    )
    herwig_relative_difference = (
        abs(stage2_pb - inclusive_pb) / inclusive_pb
        if inclusive_pb > 0.0
        else math.inf
    )
    issues = []
    if inclusive_pb <= 0.0 or uncertainty_pb < 0.0:
        issues.append("nonpositive cross section or negative uncertainty")
    if generated_events != sample.expected_events:
        issues.append(
            f"merged LHE records {generated_events} events, "
            f"expected {sample.expected_events}"
        )
    if stage2_events != sample.expected_events:
        issues.append(
            f"Stage-2 Herwig records {stage2_events} events, "
            f"expected {sample.expected_events}"
        )
    if relative_difference > HHHBB_CONSOLIDATED_RELATIVE_TOLERANCE:
        issues.append(
            "consolidated HHHbb cross section differs from merge summary "
            f"by {relative_difference:.6g}, exceeding rounded-metadata "
            f"tolerance {HHHBB_CONSOLIDATED_RELATIVE_TOLERANCE:.6g}"
        )
    if herwig_relative_difference > HHHBB_XSEC_RELATIVE_TOLERANCE:
        issues.append(
            "Stage-2 Herwig/merge relative difference "
            f"{herwig_relative_difference:.6g} exceeds "
            f"{HHHBB_XSEC_RELATIVE_TOLERANCE:.6g}"
        )
    return {
        "inclusive_xsec_pb": inclusive_pb,
        "inclusive_xsec_error_pb": uncertainty_pb,
        "generated_events": generated_events,
        "reference_xsec_pb": consolidated_pb,
        "relative_difference": relative_difference,
        "herwig_reference_xsec_pb": stage2_pb,
        "herwig_reference_xsec_error_pb": stage2_uncertainty_pb,
        "herwig_relative_difference": herwig_relative_difference,
        "zero_weight_events": int(payload.get("zero_weight_events", 0)),
        "probe_trial_weight_correction_applied": True,
        "normalization_source": (
            "probe-trial-corrected forced-splitting "
            "merge_summary.json merged_xsec_pb"
        ),
        "status": "ok" if not issues else "failed",
        "issues": issues,
    }


def cross_section_with_error(
    inclusive_pb: float,
    inclusive_error_pb: float,
    branching_factor: float,
    acceptance: float,
    acceptance_error: float,
) -> tuple[float, float]:
    value = inclusive_pb * branching_factor * acceptance
    error = math.hypot(
        branching_factor * acceptance * inclusive_error_pb,
        branching_factor * inclusive_pb * acceptance_error,
    )
    return value, error


def make_cross_section_row(
    sample: SampleInput, analysis_json: Path
) -> dict[str, object]:
    analysis = json.loads(analysis_json.read_text())
    norm = normalization(sample)
    branching_factor = HBB_BRANCHING_RATIO**sample.hbb_power
    audit_issues = list(norm["issues"])
    processed_events = int(analysis["processed_events"])
    available_events = int(analysis["available_events"])
    if processed_events != sample.expected_events:
        audit_issues.append(
            f"analyzer processed {processed_events} events, "
            f"expected {sample.expected_events}"
        )
    if available_events != sample.expected_events:
        audit_issues.append(
            f"ROOT Data tree has {available_events} events, "
            f"expected {sample.expected_events}"
        )
    if analysis.get("process") != sample.process:
        audit_issues.append("analyzer process label does not match sample")
    if analysis.get("analysis_id") != ANALYSIS_ID:
        audit_issues.append("analyzer configuration ID does not match")
    if (
        not math.isclose(float(analysis["btag_efficiency"]), BTAG_EFFICIENCY)
        or not math.isclose(float(analysis["jet_pt_cut_gev"]), PT_CUT_GEV)
        or not math.isclose(
            float(analysis["jet_abs_eta_cut"]), ABS_ETA_CUT
        )
        or int(analysis["smearing_seed"]) != SMEARING_SEED
    ):
        audit_issues.append("analyzer selection metadata does not match")
    if (
        not math.isfinite(float(analysis["sum_weights"]))
        or float(analysis["sum_weights"]) == 0.0
        or float(analysis["sum_weights_squared"]) <= 0.0
    ):
        audit_issues.append("invalid ROOT event-weight sums")
    if float(analysis["maximum_probability_closure_residual"]) > 1.0e-12:
        audit_issues.append("per-event tag-probability closure failed")

    sample_definitions = {
        "hhh": "inclusive full-loop gg->hhh, Herwig shower",
        "hhhbb": (
            "full-loop gg->hhhg plus weighted forced g->bb; "
            "generation pT_b>15 GeV, |eta_b|<3, DeltaR_bb>0.3"
        ),
        "hhhh": "inclusive full-loop gg->hhhh, Herwig shower",
    }
    row: dict[str, object] = {
        "index": sample.point.index,
        "c3": float(sample.point.c3),
        "d4": float(sample.point.d4),
        "process": sample.process,
        "sample_definition": sample_definitions[sample.process],
        "run_name": sample.run_name,
        "root_file": str(sample.root_file),
        "analysis_file": str(analysis_json),
        "generated_events": norm["generated_events"],
        "root_available_events": available_events,
        "root_processed_events": processed_events,
        "sum_weights": analysis["sum_weights"],
        "sum_weights_squared": analysis["sum_weights_squared"],
        "effective_events": analysis["effective_events"],
        "inclusive_xsec_pb": norm["inclusive_xsec_pb"],
        "inclusive_xsec_error_pb": norm["inclusive_xsec_error_pb"],
        "inclusive_xsec_fb": float(norm["inclusive_xsec_pb"]) * 1.0e3,
        "inclusive_xsec_error_fb": float(norm["inclusive_xsec_error_pb"])
        * 1.0e3,
        "hbb_branching_ratio": HBB_BRANCHING_RATIO,
        "hbb_power": sample.hbb_power,
        "branching_factor": branching_factor,
        "normalization_source": norm["normalization_source"],
        "normalization_reference_xsec_pb": norm["reference_xsec_pb"],
        "normalization_relative_difference": norm["relative_difference"],
        "herwig_reference_xsec_pb": norm["herwig_reference_xsec_pb"],
        "herwig_reference_xsec_error_pb": norm[
            "herwig_reference_xsec_error_pb"
        ],
        "herwig_relative_difference": norm["herwig_relative_difference"],
        "zero_weight_events": norm["zero_weight_events"],
        "probe_trial_weight_correction_applied": norm[
            "probe_trial_weight_correction_applied"
        ],
        "hhhbb_generation_pt_b_min_gev": (
            15.0 if sample.process == "hhhbb" else ""
        ),
        "hhhbb_generation_abs_eta_b_max": (
            3.0 if sample.process == "hhhbb" else ""
        ),
        "hhhbb_generation_delta_r_bb_min": (
            0.3 if sample.process == "hhhbb" else ""
        ),
        "analysis_id": analysis["analysis_id"],
        "audit_status": "ok" if not audit_issues else "failed",
        "audit_issues": "; ".join(audit_issues),
    }
    for category in CATEGORIES:
        category_data = analysis["tag_categories"][category]
        acceptance = float(category_data["acceptance"])
        acceptance_error = float(
            category_data["acceptance_stat_error"]
        )
        sigma_pb, sigma_error_pb = cross_section_with_error(
            inclusive_pb=float(norm["inclusive_xsec_pb"]),
            inclusive_error_pb=float(norm["inclusive_xsec_error_pb"]),
            branching_factor=branching_factor,
            acceptance=acceptance,
            acceptance_error=acceptance_error,
        )
        row[f"acceptance_{category}"] = acceptance
        row[f"acceptance_error_{category}"] = acceptance_error
        row[f"probability_sum_{category}"] = category_data[
            "probability_sum"
        ]
        row[f"weighted_probability_sum_{category}"] = category_data[
            "weighted_sum"
        ]
        row[f"sigma_{category}_pb"] = sigma_pb
        row[f"sigma_{category}_error_pb"] = sigma_error_pb
        row[f"sigma_{category}_fb"] = sigma_pb * 1.0e3
        row[f"sigma_{category}_error_fb"] = sigma_error_pb * 1.0e3
    closure = abs(
        float(row["sigma_ge6_pb"])
        - float(row["sigma_exact6_pb"])
        - float(row["sigma_exact7_pb"])
        - float(row["sigma_ge8_pb"])
    )
    row["sigma_tag_component_closure_pb"] = closure
    if closure > max(1.0e-18, 1.0e-10 * abs(float(row["sigma_ge6_pb"]))):
        row["audit_status"] = "failed"
        row["audit_issues"] = (
            str(row["audit_issues"]) + "; tag component closure failed"
        ).strip("; ")
    return row


def combine_component_rows(
    hhh_rows: list[dict[str, object]],
    hhhbb_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    hhh_by_coordinate = {
        (float(row["c3"]), float(row["d4"])): row for row in hhh_rows
    }
    hhhbb_by_coordinate = {
        (float(row["c3"]), float(row["d4"])): row for row in hhhbb_rows
    }
    if set(hhh_by_coordinate) != set(hhhbb_by_coordinate):
        raise ValueError("HHH and HHHbb coordinates do not match exactly")
    combined = []
    for coordinate in sorted(
        hhh_by_coordinate, key=lambda item: hhh_by_coordinate[item]["index"]
    ):
        hhh = hhh_by_coordinate[coordinate]
        hhhbb = hhhbb_by_coordinate[coordinate]
        component_audit_issues = "; ".join(
            part
            for part in (
                str(hhh.get("audit_issues", "")).strip(),
                str(hhhbb.get("audit_issues", "")).strip(),
            )
            if part
        )
        row: dict[str, object] = {
            "index": hhh["index"],
            "c3": coordinate[0],
            "d4": coordinate[1],
            "process": "hhh_plus_hhhbb",
            "combination_scheme": "additive_unmatched",
            "overlap_caveat": (
                "inclusive showered hhh can contain g->bb; "
                "components are not matched"
            ),
            "hhhbb_probe_trial_weight_correction_applied": hhhbb[
                "probe_trial_weight_correction_applied"
            ],
            "hhh_audit_status": hhh["audit_status"],
            "hhhbb_audit_status": hhhbb["audit_status"],
            "audit_status": (
                "ok"
                if hhh["audit_status"] == "ok"
                and hhhbb["audit_status"] == "ok"
                else "failed"
            ),
            "audit_issues": component_audit_issues,
        }
        for category in CATEGORIES:
            hhh_value = float(hhh[f"sigma_{category}_pb"])
            hhhbb_value = float(hhhbb[f"sigma_{category}_pb"])
            hhh_error = float(hhh[f"sigma_{category}_error_pb"])
            hhhbb_error = float(
                hhhbb[f"sigma_{category}_error_pb"]
            )
            error = math.hypot(
                hhh_error,
                hhhbb_error,
            )
            total = hhh_value + hhhbb_value
            row[f"hhh_acceptance_{category}"] = hhh[
                f"acceptance_{category}"
            ]
            row[f"hhh_acceptance_error_{category}"] = hhh[
                f"acceptance_error_{category}"
            ]
            row[f"hhhbb_acceptance_{category}"] = hhhbb[
                f"acceptance_{category}"
            ]
            row[f"hhhbb_acceptance_error_{category}"] = hhhbb[
                f"acceptance_error_{category}"
            ]
            row[f"hhh_sigma_{category}_pb"] = hhh_value
            row[f"hhh_sigma_{category}_error_pb"] = hhh_error
            row[f"hhh_sigma_{category}_fb"] = hhh_value * 1.0e3
            row[f"hhh_sigma_{category}_error_fb"] = hhh_error * 1.0e3
            row[f"hhhbb_sigma_{category}_pb"] = hhhbb_value
            row[f"hhhbb_sigma_{category}_error_pb"] = hhhbb_error
            row[f"hhhbb_sigma_{category}_fb"] = hhhbb_value * 1.0e3
            row[f"hhhbb_sigma_{category}_error_fb"] = (
                hhhbb_error * 1.0e3
            )
            row[f"sigma_{category}_pb"] = total
            row[f"sigma_{category}_error_pb"] = error
            row[f"sigma_{category}_fb"] = total * 1.0e3
            row[f"sigma_{category}_error_fb"] = error * 1.0e3
        denominator = float(row["sigma_ge6_pb"])
        hhh_ge6 = float(hhh["sigma_ge6_pb"])
        hhhbb_ge6 = float(hhhbb["sigma_ge6_pb"])
        row["hhhbb_fraction_ge6"] = (
            hhhbb_ge6 / denominator
            if denominator > 0.0
            else math.nan
        )
        row["hhhbb_fraction_ge6_error"] = (
            math.hypot(
                hhh_ge6 * float(hhhbb["sigma_ge6_error_pb"]),
                hhhbb_ge6 * float(hhh["sigma_ge6_error_pb"]),
            )
            / denominator**2
            if denominator > 0.0
            else math.nan
        )
        closure = abs(
            float(row["sigma_ge6_pb"])
            - float(row["sigma_exact6_pb"])
            - float(row["sigma_exact7_pb"])
            - float(row["sigma_ge8_pb"])
        )
        row["sigma_tag_component_closure_pb"] = closure
        if closure > max(
            1.0e-18, 1.0e-10 * abs(float(row["sigma_ge6_pb"]))
        ):
            row["audit_status"] = "failed"
            row["audit_issues"] = (
                str(row["audit_issues"])
                + "; combined tag component closure failed"
            ).strip("; ")
        combined.append(row)
    return combined


def ratio_with_error(
    numerator: float,
    numerator_error: float,
    denominator: float,
    denominator_error: float,
) -> tuple[float, float]:
    if denominator <= 0.0 or numerator < 0.0:
        return math.nan, math.nan
    ratio = numerator / denominator
    if numerator == 0.0:
        return ratio, abs(numerator_error / denominator)
    relative_error = math.hypot(
        numerator_error / numerator, denominator_error / denominator
    )
    return ratio, abs(ratio) * relative_error


def make_ratio_rows(
    hhhh_rows: list[dict[str, object]],
    hhh_rows: list[dict[str, object]],
    hhhbb_rows: list[dict[str, object]],
    combined_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    components = {}
    for label, rows in (
        ("hhhh", hhhh_rows),
        ("hhh", hhh_rows),
        ("hhhbb", hhhbb_rows),
        ("combined", combined_rows),
    ):
        components[label] = {
            (float(row["c3"]), float(row["d4"])): row for row in rows
        }
    coordinate_sets = [set(rows) for rows in components.values()]
    if any(coordinates != coordinate_sets[0] for coordinates in coordinate_sets):
        raise ValueError("ratio components do not have an exact coordinate join")
    output = []
    for coordinate in sorted(
        components["hhhh"],
        key=lambda item: components["hhhh"][item]["index"],
    ):
        hhhh = components["hhhh"][coordinate]
        hhh = components["hhh"][coordinate]
        hhhbb = components["hhhbb"][coordinate]
        combined = components["combined"][coordinate]
        primary, primary_error = ratio_with_error(
            float(hhhh["sigma_ge6_pb"]),
            float(hhhh["sigma_ge6_error_pb"]),
            float(combined["sigma_ge6_pb"]),
            float(combined["sigma_ge6_error_pb"]),
        )
        diagnostic, diagnostic_error = ratio_with_error(
            float(hhhh["sigma_ge6_pb"]),
            float(hhhh["sigma_ge6_error_pb"]),
            float(hhh["sigma_ge6_pb"]),
            float(hhh["sigma_ge6_error_pb"]),
        )
        output.append(
            {
                "index": hhhh["index"],
                "c3": coordinate[0],
                "d4": coordinate[1],
                "combination_scheme": "additive_unmatched",
                "hhhh_sigma_ge6_pb": hhhh["sigma_ge6_pb"],
                "hhhh_sigma_ge6_error_pb": hhhh[
                    "sigma_ge6_error_pb"
                ],
                "hhh_sigma_ge6_pb": hhh["sigma_ge6_pb"],
                "hhh_sigma_ge6_error_pb": hhh["sigma_ge6_error_pb"],
                "hhhbb_sigma_ge6_pb": hhhbb["sigma_ge6_pb"],
                "hhhbb_sigma_ge6_error_pb": hhhbb[
                    "sigma_ge6_error_pb"
                ],
                "denominator_sigma_ge6_pb": combined["sigma_ge6_pb"],
                "denominator_sigma_ge6_error_pb": combined[
                    "sigma_ge6_error_pb"
                ],
                "hhhbb_fraction_ge6": combined["hhhbb_fraction_ge6"],
                "hhhbb_fraction_ge6_error": combined[
                    "hhhbb_fraction_ge6_error"
                ],
                "ratio_hhhh_over_hhh_plus_hhhbb": primary,
                "ratio_hhhh_over_hhh_plus_hhhbb_error": primary_error,
                "ratio_hhhh_over_hhh": diagnostic,
                "ratio_hhhh_over_hhh_error": diagnostic_error,
                "audit_status": (
                    "ok"
                    if all(
                        row["audit_status"] == "ok"
                        for row in (hhhh, hhh, hhhbb, combined)
                    )
                    else "failed"
                ),
                "audit_issues": "; ".join(
                    str(row.get("audit_issues", "")).strip()
                    for row in (hhhh, hhh, hhhbb, combined)
                    if str(row.get("audit_issues", "")).strip()
                ),
            }
        )
    return output


def write_rows(
    path_stem: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path_stem}")
    atomic_write_csv(path_stem.with_suffix(".csv"), rows, list(rows[0]))
    atomic_write_json(
        path_stem.with_suffix(".json"),
        {"metadata": metadata, "rows": rows},
    )


def write_pointwise_ratio_tables(
    results_dir: Path, rows: list[dict[str, object]]
) -> None:
    primary_fields = (
        "index",
        "c3",
        "d4",
        "combination_scheme",
        "hhhh_sigma_ge6_pb",
        "hhhh_sigma_ge6_error_pb",
        "hhh_sigma_ge6_pb",
        "hhh_sigma_ge6_error_pb",
        "hhhbb_sigma_ge6_pb",
        "hhhbb_sigma_ge6_error_pb",
        "denominator_sigma_ge6_pb",
        "denominator_sigma_ge6_error_pb",
        "hhhbb_fraction_ge6",
        "hhhbb_fraction_ge6_error",
        "ratio_hhhh_over_hhh_plus_hhhbb",
        "ratio_hhhh_over_hhh_plus_hhhbb_error",
        "audit_status",
        "audit_issues",
    )
    diagnostic_fields = (
        "index",
        "c3",
        "d4",
        "hhhh_sigma_ge6_pb",
        "hhhh_sigma_ge6_error_pb",
        "hhh_sigma_ge6_pb",
        "hhh_sigma_ge6_error_pb",
        "ratio_hhhh_over_hhh",
        "ratio_hhhh_over_hhh_error",
        "audit_status",
        "audit_issues",
    )
    primary_rows = [
        {field: row[field] for field in primary_fields} for row in rows
    ]
    diagnostic_rows = [
        {field: row[field] for field in diagnostic_fields} for row in rows
    ]
    write_rows(
        results_dir / PRIMARY_RATIO_STEM,
        primary_rows,
        {
            "ratio": "hhhh_ge6/(hhh_ge6+hhhbb_ge6)",
            "combination_scheme": "additive_unmatched",
            "points": EXPECTED_POINTS,
        },
    )
    write_rows(
        results_dir / DIAGNOSTIC_RATIO_STEM,
        diagnostic_rows,
        {
            "ratio": "hhhh_ge6/hhh_ge6",
            "role": "HHH-only diagnostic",
            "points": EXPECTED_POINTS,
        },
    )


def aggregate_results(
    paths: AnalysisPaths,
    inputs: dict[str, list[SampleInput]],
) -> dict[str, list[dict[str, object]]]:
    process_rows: dict[str, list[dict[str, object]]] = {}
    for process in ("hhh", "hhhbb", "hhhh"):
        process_rows[process] = [
            make_cross_section_row(sample, cache_path(paths, sample))
            for sample in inputs[process]
        ]
        write_rows(
            paths.results_dir / f"{process}_ge6b_cross_sections",
            process_rows[process],
            dict(
                {
                "process": process,
                "analysis_id": ANALYSIS_ID,
                "hbb_branching_ratio": HBB_BRANCHING_RATIO,
                "btag_efficiency": BTAG_EFFICIENCY,
                "jet_pt_cut_gev": PT_CUT_GEV,
                "jet_abs_eta_cut": ABS_ETA_CUT,
                "smearing_seed": SMEARING_SEED,
                "points": EXPECTED_POINTS,
                },
                **(
                    {
                        "sample_definition": (
                            "full-loop gg->hhhg followed by weighted forced "
                            "g->bb"
                        ),
                        "generation_level_requirements": {
                            "pt_b_min_gev": 15.0,
                            "abs_eta_b_max": 3.0,
                            "delta_r_bb_min": 0.3,
                        },
                        "effective_cross_section_source": (
                            "probe-trial-corrected merge_summary.json"
                        ),
                    }
                    if process == "hhhbb"
                    else {}
                ),
            ),
        )
    combined = combine_component_rows(
        process_rows["hhh"], process_rows["hhhbb"]
    )
    write_rows(
        paths.results_dir / "hhh_plus_hhhbb_ge6b_cross_sections",
        combined,
        {
            "process": "hhh_plus_hhhbb",
            "combination_scheme": "additive_unmatched",
            "overlap_caveat": (
                "inclusive showered hhh can contain g->bb; "
                "this is an additive estimate, not a matched prediction"
            ),
            "points": EXPECTED_POINTS,
        },
    )
    ratios = make_ratio_rows(
        process_rows["hhhh"],
        process_rows["hhh"],
        process_rows["hhhbb"],
        combined,
    )
    write_rows(
        paths.results_dir / "ratio_points",
        ratios,
        {
            "primary_ratio": "hhhh_ge6/(hhh_ge6+hhhbb_ge6)",
            "diagnostic_ratio": "hhhh_ge6/hhh_ge6",
            "combination_scheme": "additive_unmatched",
            "points": EXPECTED_POINTS,
        },
    )
    write_pointwise_ratio_tables(paths.results_dir, ratios)
    process_rows["combined"] = combined
    process_rows["ratios"] = ratios
    return process_rows


def analyze_all(
    paths: AnalysisPaths,
    points: list[Point],
    cpus: int,
    force: bool,
) -> None:
    require_cpu_budget(cpus)
    inputs = all_inputs(paths, points)
    issues = validate_sample_inventory(
        sample for samples in inputs.values() for sample in samples
    )
    if issues:
        raise RuntimeError(
            "analysis input preflight failed:\n"
            + "\n".join(f"- {issue}" for issue in issues[:30])
            + (
                f"\n- ... {len(issues) - 30} more"
                if len(issues) > 30
                else ""
            )
        )
    jobs = [
        sample for process in ("hhh", "hhhbb", "hhhh") for sample in inputs[process]
    ]
    normalization_failures = []
    for sample in jobs:
        audit = normalization(sample)
        if audit["status"] != "ok":
            normalization_failures.append(
                f"{sample.process} {sample.run_name}: "
                + "; ".join(audit["issues"])
            )
    if normalization_failures:
        raise RuntimeError(
            f"{len(normalization_failures)} normalization preflight "
            "failure(s):\n"
            + "\n".join(normalization_failures[:30])
        )

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpus) as executor:
        future_to_sample = {
            executor.submit(run_analyzer, paths, sample, force): sample
            for sample in jobs
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(future_to_sample), start=1
        ):
            sample = future_to_sample[future]
            try:
                output = future.result()
                print(
                    f"[{index}/{len(jobs)}] {sample.process} "
                    f"{sample.run_name}: {output}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001
                failures.append(f"{sample.process} {sample.run_name}: {error}")
                print(f"ERROR: {failures[-1]}", file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} analyzer job(s) failed:\n"
            + "\n".join(failures[:20])
        )
    result_rows = aggregate_results(paths, inputs)
    failed_rows = [
        f"{process} index={row['index']}"
        for process, rows in result_rows.items()
        for row in rows
        if row.get("audit_status") != "ok"
    ]
    if failed_rows:
        raise RuntimeError(
            f"{len(failed_rows)} output audit row(s) failed: "
            + ", ".join(failed_rows[:20])
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_ratio_contours(
    rows: list[dict[str, object]],
    value_field: str,
    output_pdf: Path,
    title: str,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np

    values = [
        (
            float(row["c3"]),
            float(row["d4"]),
            float(row[value_field]),
        )
        for row in rows
        if math.isfinite(float(row[value_field]))
        and float(row[value_field]) > 0.0
    ]
    if len(values) < 3:
        raise ValueError("at least three finite positive ratio points are required")
    x = np.asarray([item[0] for item in values], dtype=float)
    y = np.asarray([item[1] for item in values], dtype=float)
    z = np.asarray([item[2] for item in values], dtype=float)
    c3_center = 0.5 * (PLOT_C3_RANGE[0] + PLOT_C3_RANGE[1])
    d4_center = 0.5 * (PLOT_D4_RANGE[0] + PLOT_D4_RANGE[1])
    c3_scale = 0.5 * (PLOT_C3_RANGE[1] - PLOT_C3_RANGE[0])
    d4_scale = 0.5 * (PLOT_D4_RANGE[1] - PLOT_D4_RANGE[0])
    scaled_x = (x - c3_center) / c3_scale
    scaled_y = (y - d4_center) / d4_scale
    point_log_values = np.log10(z)
    triangulation = mtri.Triangulation(scaled_x, scaled_y)
    interpolator = mtri.CubicTriInterpolator(
        triangulation,
        point_log_values,
        kind="geom",
    )
    interpolated_points = np.ma.asarray(
        interpolator(scaled_x, scaled_y)
    )
    if np.any(np.ma.getmaskarray(interpolated_points)):
        raise ValueError("cubic interpolation masked one or more scan points")
    interpolation_residual = float(
        np.max(
            np.abs(
                np.asarray(interpolated_points, dtype=float)
                - point_log_values
            )
        )
    )
    if interpolation_residual > 1.0e-8:
        raise ValueError(
            "cubic interpolation does not reproduce the pointwise ratios: "
            f"maximum log10 residual {interpolation_residual:.6g}"
        )

    grid_scaled_x = np.linspace(
        (PLOT_C3_RANGE[0] - c3_center) / c3_scale,
        (PLOT_C3_RANGE[1] - c3_center) / c3_scale,
        CONTOUR_GRID_POINTS_PER_AXIS,
    )
    grid_scaled_y = np.linspace(
        (PLOT_D4_RANGE[0] - d4_center) / d4_scale,
        (PLOT_D4_RANGE[1] - d4_center) / d4_scale,
        CONTOUR_GRID_POINTS_PER_AXIS,
    )
    mesh_scaled_x, mesh_scaled_y = np.meshgrid(
        grid_scaled_x, grid_scaled_y
    )
    grid_log_values = interpolator(mesh_scaled_x, mesh_scaled_y)
    mesh_c3 = c3_center + c3_scale * mesh_scaled_x
    mesh_d4 = d4_center + d4_scale * mesh_scaled_y
    visible_levels = [
        level
        for level in RATIO_LEVELS
        if float(np.min(z)) <= level <= float(np.max(z))
    ]

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")
    fig, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    axis.set_facecolor("white")
    axis.grid(alpha=0.2, linewidth=0.5)
    contour_styles = {
        level: dict(RATIO_LEVEL_STYLES.get(level, RATIO_FALLBACK_STYLE))
        for level in visible_levels
    }
    if visible_levels:
        log_levels = np.log10(visible_levels)
        contour = axis.contour(
            mesh_c3,
            mesh_d4,
            grid_log_values,
            levels=log_levels,
            colors=[
                contour_styles[level]["color"] for level in visible_levels
            ],
            linestyles=[
                contour_styles[level]["linestyle"]
                for level in visible_levels
            ],
            linewidths=1.8,
        )
        axis.clabel(
            contour,
            fmt={
                math.log10(level): f"{level:.2g}"
                for level in visible_levels
            },
            inline=True,
            fontsize=11,
        )
    axis.set_xlim(PLOT_C3_RANGE)
    axis.set_ylim(PLOT_D4_RANGE)
    axis.set_xlabel(r"$c_3$", fontsize=20)
    axis.set_ylabel(r"$d_4$", fontsize=20)
    axis.set_title(title + " at 14 TeV", fontsize=18)
    axis.tick_params(axis="both", labelsize=15)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    return {
        "status": "ok",
        "output_pdf": str(output_pdf),
        "output_png": str(output_png),
        "value_field": value_field,
        "points": len(values),
        "ratio_min": float(np.min(z)),
        "ratio_max": float(np.max(z)),
        "visible_levels": visible_levels,
        "contour_level_styles": {
            f"{level:.2g}": contour_styles[level]
            for level in visible_levels
        },
        "interpolation": (
            "C1 cubic triangular interpolation of log10(pointwise ratio)"
        ),
        "interpolation_kind": "matplotlib CubicTriInterpolator kind=geom",
        "interpolation_grid_points_per_axis": (
            CONTOUR_GRID_POINTS_PER_AXIS
        ),
        "interpolation_coordinate_scaling": {
            "c3_center": c3_center,
            "c3_scale": c3_scale,
            "d4_center": d4_center,
            "d4_scale": d4_scale,
        },
        "point_interpolation_max_abs_log10_residual": (
            interpolation_residual
        ),
        "extrapolation": (
            "none; interpolator is masked outside the Delaunay convex hull"
        ),
    }


def make_plots(paths: AnalysisPaths) -> dict[str, object]:
    ratio_csv = paths.results_dir / "ratio_points.csv"
    rows: list[dict[str, object]] = read_csv_rows(ratio_csv)
    primary = plot_ratio_contours(
        rows,
        "ratio_hhhh_over_hhh_plus_hhhbb",
        paths.results_dir / f"{PRIMARY_PLOT_STEM}.pdf",
        r"$\frac{\sigma(hhhh,\geq6b)}"
        r"{\sigma(hhh,\geq6b)+\sigma(hhh+b\bar b,\geq6b)}$",
    )
    diagnostic = plot_ratio_contours(
        rows,
        "ratio_hhhh_over_hhh",
        paths.results_dir / f"{DIAGNOSTIC_PLOT_STEM}.pdf",
        r"$\frac{\sigma(hhhh,\geq6b)}{\sigma(hhh,\geq6b)}$",
    )
    payload = {
        "combination_scheme": "additive_unmatched",
        "overlap_caveat": (
            "inclusive showered hhh can contain g->bb; "
            "primary denominator is not matched"
        ),
        "primary": primary,
        "diagnostic": diagnostic,
    }
    atomic_write_json(paths.results_dir / "plot_metadata.json", payload)
    return payload


def result_status(paths: AnalysisPaths) -> dict[str, object]:
    outputs = [
        "hhh_ge6b_cross_sections.csv",
        "hhhbb_ge6b_cross_sections.csv",
        "hhhh_ge6b_cross_sections.csv",
        "hhh_plus_hhhbb_ge6b_cross_sections.csv",
        "ratio_points.csv",
        f"{PRIMARY_RATIO_STEM}.csv",
        f"{DIAGNOSTIC_RATIO_STEM}.csv",
        f"{PRIMARY_PLOT_STEM}.pdf",
        f"{PRIMARY_PLOT_STEM}.png",
        f"{DIAGNOSTIC_PLOT_STEM}.pdf",
        f"{DIAGNOSTIC_PLOT_STEM}.png",
    ]
    files = {
        name: {
            "exists": (paths.results_dir / name).is_file(),
            "bytes": (
                (paths.results_dir / name).stat().st_size
                if (paths.results_dir / name).is_file()
                else 0
            ),
        }
        for name in outputs
    }
    cache_counts = {
        process: len(list((paths.results_dir / "cache" / process).glob("*.json")))
        for process in ("hhh", "hhhbb", "hhhh")
    }
    return {
        "results_dir": str(paths.results_dir),
        "cache_counts": cache_counts,
        "outputs": files,
    }


def validate_results(
    paths: AnalysisPaths, points: list[Point]
) -> dict[str, object]:
    issues: list[str] = []
    inventory = hhhbb_inventory(
        type(
            "CampaignPaths",
            (),
            {"source_repo": paths.source_repo},
        )()
    )
    if inventory["status"] != "complete":
        issues.append(f"HHHbb inventory is {inventory['status']}")
    expected_coordinates = {point.coordinate for point in points}
    table_names = (
        "hhh_ge6b_cross_sections.csv",
        "hhhbb_ge6b_cross_sections.csv",
        "hhhh_ge6b_cross_sections.csv",
        "hhh_plus_hhhbb_ge6b_cross_sections.csv",
        "ratio_points.csv",
        f"{PRIMARY_RATIO_STEM}.csv",
        f"{DIAGNOSTIC_RATIO_STEM}.csv",
    )
    table_counts = {}
    for name in table_names:
        path = paths.results_dir / name
        if not path.is_file():
            issues.append(f"missing result table {path}")
            continue
        rows = read_csv_rows(path)
        table_counts[name] = len(rows)
        coordinates = {
            (float(row["c3"]), float(row["d4"])) for row in rows
        }
        if len(rows) != EXPECTED_POINTS or coordinates != expected_coordinates:
            issues.append(
                f"{name} does not contain the exact 153 coordinates"
            )
        failed = [row for row in rows if row.get("audit_status") != "ok"]
        if failed:
            issues.append(f"{name} has {len(failed)} failed audit rows")
    for name in (
        f"{PRIMARY_PLOT_STEM}.pdf",
        f"{PRIMARY_PLOT_STEM}.png",
        f"{DIAGNOSTIC_PLOT_STEM}.pdf",
        f"{DIAGNOSTIC_PLOT_STEM}.png",
    ):
        path = paths.results_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            issues.append(f"missing or empty plot {path}")
    payload = {
        "status": "ok" if not issues else "failed",
        "expected_points": EXPECTED_POINTS,
        "table_counts": table_counts,
        "hhhbb_inventory": inventory,
        "combination_scheme": "additive_unmatched",
        "issues": issues,
    }
    atomic_write_json(paths.results_dir / "validation_summary.json", payload)
    return payload


def standard_sm_samples(
    paths: AnalysisPaths, point: Point
) -> tuple[SampleInput, SampleInput]:
    hhhh = hhhh_inputs(paths, [point])[0]
    hhhbb = hhhbb_inputs(paths, [point])[0]
    return hhhh, hhhbb


def run_smoke_analysis(paths: AnalysisPaths, points: list[Point]) -> dict[str, object]:
    smoke_samples_path = CAMPAIGN_DIR / "smoke" / "smoke_samples.json"
    smoke_samples = json.loads(smoke_samples_path.read_text())
    point = next(
        point for point in points if point.coordinate == (0.0, 0.0)
    )
    smoke_hhh = SampleInput(
        process="hhh",
        point=point,
        run_name="run_gg_hhh_smoke_ge6b_0.0_0.0",
        root_file=Path(smoke_samples["hhh_root"]),
        herwig_out=Path(smoke_samples["hhh_herwig_out"]),
        lhe_file=Path(smoke_samples["mg5_lhe"]),
        hbb_power=3,
        expected_events=100,
    )
    hhhh, hhhbb = standard_sm_samples(paths, point)
    smoke_dir = CAMPAIGN_DIR / "smoke" / "analysis"
    analysis_files = {}
    for sample in (smoke_hhh, hhhh, hhhbb):
        output = smoke_dir / f"{sample.process}.json"
        analysis_files[sample.process] = run_analyzer(
            paths, sample, force=True, output=output
        )
    rows = {
        process: make_cross_section_row(sample, analysis_files[process])
        for process, sample in (
            ("hhh", smoke_hhh),
            ("hhhh", hhhh),
            ("hhhbb", hhhbb),
        )
    }
    combined = combine_component_rows([rows["hhh"]], [rows["hhhbb"]])[0]
    ratios = make_ratio_rows(
        [rows["hhhh"]],
        [rows["hhh"]],
        [rows["hhhbb"]],
        [combined],
    )[0]
    if any(row["audit_status"] != "ok" for row in rows.values()):
        raise RuntimeError("one or more smoke normalization audits failed")
    for row in rows.values():
        if float(row["inclusive_xsec_pb"]) <= 0.0:
            raise RuntimeError("smoke sample has nonpositive cross section")
        if float(row["sigma_tag_component_closure_pb"]) > max(
            1.0e-18, 1.0e-10 * abs(float(row["sigma_ge6_pb"]))
        ):
            raise RuntimeError("smoke tag-category closure failed")

    # A deterministic non-physics mini-grid exercises both triangulated plot
    # paths without pretending that one smoke point defines a contour.
    synthetic = []
    for index, (c3, d4, primary, diagnostic) in enumerate(
        (
            (-10.0, -200.0, 0.008, 0.012),
            (0.0, -200.0, 0.04, 0.06),
            (10.0, -200.0, 0.2, 0.3),
            (-10.0, 200.0, 0.08, 0.12),
            (0.0, 200.0, 0.8, 1.2),
            (10.0, 200.0, 12.0, 15.0),
        ),
        start=1,
    ):
        synthetic.append(
            {
                "index": index,
                "c3": c3,
                "d4": d4,
                "ratio_hhhh_over_hhh_plus_hhhbb": primary,
                "ratio_hhhh_over_hhh": diagnostic,
            }
        )
    primary_plot = plot_ratio_contours(
        synthetic,
        "ratio_hhhh_over_hhh_plus_hhhbb",
        smoke_dir / "primary_fixture.pdf",
        "Smoke fixture: primary ratio",
    )
    diagnostic_plot = plot_ratio_contours(
        synthetic,
        "ratio_hhhh_over_hhh",
        smoke_dir / "diagnostic_fixture.pdf",
        "Smoke fixture: HHH-only ratio",
    )
    payload = {
        "status": "ok",
        "analysis_id": ANALYSIS_ID,
        "selection": {
            "jet_pt_cut_gev": PT_CUT_GEV,
            "jet_abs_eta_cut": ABS_ETA_CUT,
            "btag_efficiency": BTAG_EFFICIENCY,
            "smearing_seed": SMEARING_SEED,
            "extra_delta_r_cut": None,
        },
        "components": rows,
        "combined": combined,
        "ratios": ratios,
        "fixture_plots": {
            "primary": primary_plot,
            "diagnostic": diagnostic_plot,
        },
    }
    atomic_write_json(smoke_dir / "smoke_results.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("analyze", "plot", "validate", "status", "smoke-analysis"),
    )
    parser.add_argument("--cpus", type=int, default=PRODUCTION_CPUS)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--mg5-process", type=Path)
    parser.add_argument("--herwig-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--analyzer", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = analysis_paths(args)
    points = load_points(paths.points_file)
    if args.command == "analyze":
        analyze_all(paths, points, args.cpus, args.force)
        print(json.dumps(result_status(paths), indent=2, sort_keys=True))
    elif args.command == "plot":
        print(json.dumps(make_plots(paths), indent=2, sort_keys=True))
    elif args.command == "validate":
        payload = validate_results(paths, points)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "ok" else 1
    elif args.command == "status":
        print(json.dumps(result_status(paths), indent=2, sort_keys=True))
    elif args.command == "smoke-analysis":
        payload = run_smoke_analysis(paths, points)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "output": str(
                        CAMPAIGN_DIR
                        / "smoke"
                        / "analysis"
                        / "smoke_results.json"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
