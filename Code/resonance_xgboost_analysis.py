#!/usr/bin/env python3
"""Mass-aware XGBoost and expected-limit analysis for resonant four-Higgs scans.

This module is deliberately separate from the c3/d4 analysis.  It consumes the
``resonance-hybrid-v1`` trees produced by ``FourHiggsResonanceAnalysis``, trains
one fixed-configuration, five-fold classifier per topology, and evaluates the
background at the physical mass hypothesis of every signal point.

The BSM signal templates are always normalized to a 1 fb production cross
section.  Generator/LHE cross sections are retained only as diagnostics.
Tagging efficiencies are applied analytically exactly once from the persisted
true-single, double-B, charm-mistag, and light-mistag multiplicities.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


METHOD_VERSION = "resonance-mass-aware-xgboost-v1"
TREE_SCHEMA = "resonance-hybrid-v1"
EXTRACTOR_METHOD_VERSION = "resonance-hybrid-v1.2-uniform-fourvector-smearing"
EXTRACTOR_PREPROCESSING_VERSION = "resonance-preprocessing-v2"
EXTRACTOR_SMEARING_MODEL_ID = "cms-energy-uniform-fourvector-v1"
EXTRACTOR_SMEARING_ETA_PRESELECTION = "finite |eta|<2.5 before smearing"
EXTRACTOR_SMEARING_PT_THRESHOLD = "smeared pT>20 GeV"
EXTRACTOR_SMEARING_ACCEPTANCE_ORDER = "raw_abs_eta_then_smear_then_smeared_pt"
DEFAULT_TREE = "ResonanceFeatures"
DEFAULT_SEED = 12345
N_FOLDS = 5
LUMINOSITY_FB = 3000.0
HBB_BRANCHING_RATIO = 0.5824
SIGNAL_HYPOTHESIS_FB = 1.0
SIGNAL_XSEC_DEFINITION = (
    "full generated pp->4h rate after scalar-cascade branching fractions and "
    "before h->bb; scalar decay branching fractions are included"
)
BACKGROUND_NORM_UNCERTAINTY = 0.10
PYHF_LIMIT_METHOD_VERSION = "bounded-independent-cls-roots-v1"
DEFAULT_PYHF_POI_HARD_CAP = 10.0
DEFAULT_PYHF_PHYSICAL_LIMIT_HARD_CAP_FB = 1.0e6
DEFAULT_PYHF_REFERENCE_MAX_ATTEMPTS = 8
PYHF_CLS_LEVEL = 0.05
PYHF_LIMIT_CURVES = (
    "observed_asimov",
    "expected_minus2sigma",
    "expected_minus1sigma",
    "expected_median",
    "expected_plus1sigma",
    "expected_plus2sigma",
)
EPS_B = 0.85
EPS_C = 0.10
EPS_LIGHT = 0.01
EPS_BB_NOMINAL = EPS_B**2
EPS_BB_CONSERVATIVE = 0.30
PRODUCED_SIGNAL_EVENTS = 3000.0
EIGHT_B_SIGNAL_EVENTS = 345.1490798665728
NOMINAL_TAG_SIGNAL_EVENTS = 94.0498539896
PAIR_LABELS = ("12", "13", "14", "23", "24", "34")
CATEGORY_NAMES = ("resolved", "mixed", "boosted")
TAGGING_SCENARIOS = {
    "nominal": EPS_BB_NOMINAL,
    "conservative": EPS_BB_CONSERVATIVE,
}

# Fixed by construction; every fold and topology uses this recorded contract.
FIXED_XGBOOST_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": 1,
}

SCALAR_BRANCHES = (
    "event_index",
    "weight",
    "raw_bjets",
    "accepted_bjets",
    "accepted_single_bjets",
    "accepted_merged_bjets",
    "accepted_cjet_candidates",
    "accepted_lightjet_candidates",
    "accepted_tag_equivalents",
    "reco_jets_considered",
    "reco_jets_used",
    "n_configurations",
    "n_true_single",
    "n_double_b",
    "n_c_mistag",
    "n_light_mistag",
    "n_merged",
    "category",
    "best_score",
    "second_score",
    "score_gap",
    "m4h",
    "pt4h",
    "y4h",
    "ht",
    "centrality",
    "sphericity",
)
ARRAY_BRANCH_WIDTHS = {
    "jet_pt": 8,
    "higgs_e": 4,
    "higgs_px": 4,
    "higgs_py": 4,
    "higgs_pz": 4,
    "higgs_mass": 4,
    "higgs_pt": 4,
    "higgs_y": 4,
    "higgs_type": 4,
    "higgs_constituent1": 4,
    "higgs_constituent2": 4,
    "higgs_constituent1_source": 4,
    "higgs_constituent2_source": 4,
    "pair_mass": 6,
    "pair_dr": 6,
    "pair_dy": 6,
    "pair_dphi": 6,
}

BASE_SCALAR_FEATURES = (
    "n_merged",
    "category",
    "best_score",
    "second_score",
    "score_gap",
    "m4h",
    "pt4h",
    "y4h",
    "ht",
    "centrality",
    "sphericity",
)
BASE_ARRAY_FEATURES = (
    "jet_pt",
    "higgs_mass",
    "higgs_pt",
    "higgs_y",
    "higgs_type",
    "pair_mass",
    "pair_dr",
    "pair_dy",
    "pair_dphi",
)

CPP_MODEL_FEATURE_NAMES = (
    "n_merged",
    "category",
    "best_score",
    "second_score",
    "score_gap",
    *(f"jet_pt_{index}" for index in range(1, 9)),
    *(
        f"higgs_{field}_{index}"
        for field in ("e", "px", "py", "pz", "mass", "pt", "y", "type")
        for index in range(1, 5)
    ),
    *(
        f"pair_{field}_{label}"
        for field in ("mass", "dr", "dy", "dphi")
        for label in PAIR_LABELS
    ),
    "m4h",
    "pt4h",
    "y4h",
    "ht",
    "centrality",
    "sphericity",
)
FORBIDDEN_MODEL_FEATURES = frozenset(
    {
        "raw_bjets",
        "accepted_bjets",
        "accepted_single_bjets",
        "accepted_merged_bjets",
        "accepted_cjet_candidates",
        "accepted_lightjet_candidates",
        "accepted_tag_equivalents",
        "reco_jets_considered",
        "reco_jets_used",
        "n_configurations",
        "n_true_single",
        "n_double_b",
        "n_c_mistag",
        "n_light_mistag",
        "higgs_constituent1_source",
        "higgs_constituent2_source",
    }
)
if len(CPP_MODEL_FEATURE_NAMES) != 75 or FORBIDDEN_MODEL_FEATURES.intersection(
    CPP_MODEL_FEATURE_NAMES
):  # pragma: no cover - import-time contract
    raise AssertionError("invalid resonance model feature allowlist")


class AnalysisInputError(ValueError):
    """Raised when a sample violates the persisted analysis contract."""


@dataclass(frozen=True)
class MassPoint:
    topology: str
    ms: float | None = None
    m2: float | None = None
    m3: float | None = None

    def __post_init__(self) -> None:
        if self.topology == "direct":
            if self.ms is None or self.m2 is not None or self.m3 is not None:
                raise AnalysisInputError("a direct point requires only MS")
            if not math.isfinite(self.ms) or self.ms <= 500.0:
                raise AnalysisInputError(f"direct MS={self.ms!r} must satisfy MS > 4 mh")
        elif self.topology == "cascade":
            if self.m2 is None or self.m3 is None or self.ms is not None:
                raise AnalysisInputError("a cascade point requires M2 and M3")
            if not all(math.isfinite(value) for value in (self.m2, self.m3)):
                raise AnalysisInputError("cascade masses must be finite")
            if self.m2 <= 250.0 or self.m3 <= 500.0 or self.m3 <= 2.0 * self.m2:
                raise AnalysisInputError(
                    f"cascade (M2,M3)=({self.m2:g},{self.m3:g}) violates the hierarchy"
                )
        else:
            raise AnalysisInputError(f"unsupported topology {self.topology!r}")

    @property
    def point_id(self) -> str:
        if self.topology == "direct":
            return f"MS_{_mass_token(float(self.ms))}"
        return f"M2_{_mass_token(float(self.m2))}_M3_{_mass_token(float(self.m3))}"

    @property
    def sort_key(self) -> tuple[float, float]:
        if self.topology == "direct":
            return (float(self.ms), 0.0)
        return (float(self.m3), float(self.m2))

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "topology": self.topology,
            "point_id": self.point_id,
            "MS_GeV": self.ms,
            "M2_GeV": self.m2,
            "M3_GeV": self.m3,
        }


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    role: str
    root_file: Path
    summary_file: Path
    generated_events_expected: int
    cross_section_fb: float | None
    generated_cross_section_fb: float | None
    cross_section_source: str
    k_factor: float
    hbb_power: int
    rate_factor: float
    c_mistags: int
    light_mistags: int
    lhe_event_count: int | None
    hard_event_policy: str
    point: MassPoint | None = None
    lhe_file: Path | None = None
    optional: bool = False

    @property
    def is_signal(self) -> bool:
        return self.role == "signal"


@dataclass
class EventTable:
    arrays: dict[str, np.ndarray]
    input_events: int
    input_sumw: float
    input_sumw2: float
    reconstructable_events: int
    reconstructable_sumw: float
    summary: dict[str, Any]

    @property
    def entries(self) -> int:
        return len(self.arrays["weight"])


@dataclass
class LoadedSample:
    spec: SampleSpec
    table: EventTable
    folds: np.ndarray
    base_features: np.ndarray
    base_feature_names: tuple[str, ...]
    scenario_weights: dict[str, np.ndarray]


@dataclass(frozen=True)
class CrossfitScores:
    """Pointwise scores with disjoint fold roles for every rotation.

    ``test[event]`` is produced by model ``fold[event]``.  ``validation[event]``
    is produced by model ``(fold[event] - 1) mod N_FOLDS``, for which that event
    belongs to the validation fold.  Consequently a rotation's bin edges can be
    selected without ever consulting its own test events.
    """

    test: np.ndarray
    validation: np.ndarray


def _mass_token(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        return f"{int(rounded):04d}"
    return f"{value:g}".replace(".", "p")


def _float_from_row(row: Mapping[str, str], names: Sequence[str]) -> float | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            try:
                result = float(value)
            except ValueError as error:
                raise AnalysisInputError(f"{name}={value!r} is not numeric") from error
            if not math.isfinite(result):
                raise AnalysisInputError(f"{name} must be finite")
            return result
    return None


def parse_mass_point(
    topology: str, row: Mapping[str, str] | None = None, filename: str | Path | None = None
) -> MassPoint:
    """Parse physical masses from a manifest row, falling back to a filename."""

    row = row or {}
    topology = str(topology).strip().lower()
    if topology == "direct":
        ms = _float_from_row(row, ("MS_GeV", "ms_GeV", "miota_GeV", "M3_GeV"))
        if ms is None and filename is not None:
            match = re.search(r"(?:miota|MS)[_-]?(\d+(?:p\d+|\.\d+)?)", str(filename), re.I)
            if match:
                ms = float(match.group(1).replace("p", "."))
        if ms is None:
            raise AnalysisInputError("could not determine direct MS from manifest or filename")
        return MassPoint("direct", ms=ms)

    if topology == "cascade":
        m2 = _float_from_row(row, ("M2_GeV", "m2_GeV", "meta_GeV"))
        m3 = _float_from_row(row, ("M3_GeV", "m3_GeV", "miota_GeV"))
        text = str(filename or "")
        if m2 is None:
            match = re.search(r"(?:meta|M2)[_-]?(\d+(?:p\d+|\.\d+)?)", text, re.I)
            if match:
                m2 = float(match.group(1).replace("p", "."))
        if m3 is None:
            match = re.search(r"(?:miota|M3)[_-]?(\d+(?:p\d+|\.\d+)?)", text, re.I)
            if match:
                m3 = float(match.group(1).replace("p", "."))
        if m2 is None or m3 is None:
            raise AnalysisInputError("could not determine cascade M2/M3 from manifest or filename")
        return MassPoint("cascade", m2=m2, m3=m3)
    raise AnalysisInputError(f"unsupported topology {topology!r}")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisInputError(f"{name}={value!r} is not numeric") from error
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise AnalysisInputError(f"{name} must be a non-negative integer")
    return int(numeric)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AnalysisInputError(f"empty manifest: {path}")
    return rows


def _lhe_cross_section_fb(path: Path) -> float | None:
    """Read and sum XSECUP values (pb) from an LHE init block."""

    if not path.is_file():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    in_init = False
    init_lines: list[str] = []
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("<init"):
                    in_init = True
                    continue
                if in_init and stripped.startswith("</init"):
                    break
                if in_init and stripped and not stripped.startswith("#"):
                    init_lines.append(stripped)
    except OSError:
        return None
    if len(init_lines) < 2:
        return None
    try:
        n_processes = int(init_lines[0].split()[-1])
        process_lines = init_lines[1 : 1 + n_processes]
        xsec_pb = sum(float(line.split()[0]) for line in process_lines)
    except (ValueError, IndexError):
        return None
    return xsec_pb * 1000.0


def _signal_root_from_row(
    row: Mapping[str, str], manifest_dir: Path, root_dir: Path, pattern: str
) -> Path:
    for column in ("analysis_root", "resonance_root", "root_file"):
        value = row.get(column, "").strip()
        if value:
            return _resolve(manifest_dir, value)
    run_name = row.get("run_name", "").strip()
    if not run_name:
        raise AnalysisInputError("signal manifest needs run_name or an explicit feature-root column")
    rendered = pattern.format(run_name=run_name, scenario=row.get("scenario", "").strip())
    return _resolve(root_dir, rendered)


def load_signal_specs(
    manifest: Path,
    topology: str,
    signal_root_dir: Path,
    signal_root_pattern: str,
    smoke_points: int | None = None,
) -> list[SampleSpec]:
    rows = _read_csv(manifest)
    specs: list[SampleSpec] = []
    for row in rows:
        scenario = row.get("scenario", topology).strip().lower()
        if scenario != topology:
            continue
        root_file = _signal_root_from_row(
            row, manifest.parent, signal_root_dir, signal_root_pattern
        )
        point = parse_mass_point(topology, row, root_file)
        lhe_text = row.get("lhe", "").strip()
        lhe_file = _resolve(manifest.parent, lhe_text) if lhe_text else None
        diagnostic = _float_from_row(
            row, ("generated_cross_section_fb", "lhe_cross_section_fb", "xsec_fb")
        )
        source = "manifest"
        if diagnostic is None and lhe_file is not None:
            diagnostic = _lhe_cross_section_fb(lhe_file)
            source = "LHE_init" if diagnostic is not None else "unavailable"
        expected = _nonnegative_int(
            row.get("events", row.get("generated_events", "10000")), "signal events"
        )
        if expected == 0:
            raise AnalysisInputError("signal events must be positive")
        lhe_count_text = str(row.get("lhe_event_count", expected) or expected).strip()
        lhe_event_count = _nonnegative_int(lhe_count_text, "signal lhe_event_count")
        if lhe_event_count == 0:
            raise AnalysisInputError("signal lhe_event_count must be positive")
        if expected > lhe_event_count:
            raise AnalysisInputError(
                "signal generated events exceed the available unique LHE hard events; "
                "hard-event recycling is not supported"
            )
        run_name = row.get("run_name", root_file.stem).strip()
        hard_event_policy = row.get("hard_event_policy", "unique").strip() or "unique"
        if hard_event_policy.lower() in {"recycled", "reused"}:
            raise AnalysisInputError(
                f"{run_name}: hard-event recycling is not supported without persisted group IDs"
            )
        specs.append(
            SampleSpec(
                sample_id=run_name,
                role="signal",
                root_file=root_file,
                summary_file=root_file.with_suffix(".analysis_summary.json"),
                generated_events_expected=expected,
                cross_section_fb=SIGNAL_HYPOTHESIS_FB,
                generated_cross_section_fb=diagnostic,
                cross_section_source=source,
                k_factor=1.0,
                hbb_power=4,
                rate_factor=1.0,
                c_mistags=0,
                light_mistags=0,
                lhe_event_count=lhe_event_count,
                hard_event_policy=hard_event_policy,
                point=point,
                lhe_file=lhe_file,
            )
        )
    if not specs:
        raise AnalysisInputError(f"no {topology} rows found in {manifest}")
    if smoke_points is not None and len(specs) > smoke_points:
        indices = np.linspace(0, len(specs) - 1, smoke_points, dtype=int)
        specs = [specs[index] for index in sorted(set(indices.tolist()))]
    return specs


def load_background_specs(
    manifest: Path, analysis_root: Path, default_k_factor: float
) -> tuple[list[SampleSpec], list[dict[str, Any]]]:
    rows = _read_csv(manifest)
    specs: list[SampleSpec] = []
    missing_optional: list[dict[str, Any]] = []
    for row in rows:
        if not _as_bool(row.get("enabled"), True):
            continue
        optional = _as_bool(row.get("optional"), False)
        root_text = row.get("root_file", "").strip()
        if not root_text:
            if optional:
                missing_optional.append({"sample_id": row.get("sample_id"), "reason": "no root_file"})
                continue
            raise AnalysisInputError("background row is missing root_file")
        root_file = _resolve(analysis_root, root_text)
        if not root_file.is_file() and optional:
            missing_optional.append(
                {"sample_id": row.get("sample_id"), "root_file": str(root_file), "reason": "missing"}
            )
            continue
        sample_id = row.get("sample_id", root_file.stem).strip()
        role = row.get("role", "background").strip().lower() or "background"
        xsec = _float_from_row(row, ("cross_section_fb", "xsec_fb"))
        if xsec is None or not math.isfinite(xsec) or xsec <= 0.0:
            raise AnalysisInputError(f"{sample_id}: cross_section_fb must be positive")
        expected_text = row.get("generated_events", "").strip()
        if not expected_text:
            raise AnalysisInputError(f"{sample_id}: generated_events is required as an entry check")
        hbb_default = 4 if role == "sm_hhhh" else 0
        hbb_power = _nonnegative_int(
            row.get("hbb_power", hbb_default) or hbb_default,
            f"{sample_id} hbb_power",
        )
        if role == "sm_hhhh" and hbb_power != 4:
            raise AnalysisInputError(f"{sample_id}: SM hhhh must use hbb_power=4")
        k_factor = float(row.get("k_factor", default_k_factor) or default_k_factor)
        rate_factor = float(row.get("rate_factor", 1.0) or 1.0)
        if not math.isfinite(k_factor) or k_factor <= 0.0:
            raise AnalysisInputError(f"{sample_id}: k_factor must be positive and finite")
        if not math.isfinite(rate_factor) or rate_factor <= 0.0:
            raise AnalysisInputError(f"{sample_id}: rate_factor must be positive and finite")
        c_mistags = _nonnegative_int(
            row.get("c_mistags", 0) or 0, f"{sample_id} c_mistags"
        )
        light_mistags = _nonnegative_int(
            row.get("light_mistags", 0) or 0, f"{sample_id} light_mistags"
        )
        if c_mistags + light_mistags > 8:
            raise AnalysisInputError(
                f"{sample_id}: c/light mistag composition exceeds eight tag equivalents"
            )
        lhe_count_text = row.get("lhe_event_count", "").strip()
        lhe_event_count = (
            _nonnegative_int(lhe_count_text, f"{sample_id} lhe_event_count")
            if lhe_count_text
            else None
        )
        hard_event_policy = row.get("hard_event_policy", "unspecified").strip() or "unspecified"
        generated_events = _nonnegative_int(
            expected_text, f"{sample_id} generated_events"
        )
        if generated_events == 0:
            raise AnalysisInputError(f"{sample_id}: generated_events must be positive")
        if lhe_event_count is not None and generated_events > lhe_event_count:
            raise AnalysisInputError(
                f"{sample_id}: generated events exceed the available unique LHE hard events; "
                "hard-event recycling is not supported"
            )
        if hard_event_policy.lower() in {"recycled", "reused"}:
            raise AnalysisInputError(
                f"{sample_id}: hard-event recycling is not supported without persisted group IDs"
            )
        specs.append(
            SampleSpec(
                sample_id=sample_id,
                role=role,
                root_file=root_file,
                summary_file=root_file.with_suffix(".analysis_summary.json"),
                generated_events_expected=generated_events,
                cross_section_fb=xsec,
                generated_cross_section_fb=xsec,
                cross_section_source="background_manifest",
                k_factor=k_factor,
                hbb_power=hbb_power,
                rate_factor=rate_factor,
                c_mistags=c_mistags,
                light_mistags=light_mistags,
                lhe_event_count=lhe_event_count,
                hard_event_policy=hard_event_policy,
                optional=optional,
            )
        )
    present_sm = [spec for spec in specs if spec.role == "sm_hhhh"]
    if len(present_sm) != 1:
        raise AnalysisInputError(
            f"background manifest must contain exactly one available sm_hhhh row; found {len(present_sm)}"
        )
    return specs, missing_optional


def _summary_metadata(path: Path) -> tuple[int, float, float, int, float, dict[str, Any]]:
    if not path.is_file():
        raise AnalysisInputError(f"missing extractor summary: {path}")
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("schema") != TREE_SCHEMA:
        raise AnalysisInputError(f"{path}: expected schema {TREE_SCHEMA!r}")
    if summary.get("method_version") != EXTRACTOR_METHOD_VERSION:
        raise AnalysisInputError(
            f"{path}: expected extractor method_version {EXTRACTOR_METHOD_VERSION!r}"
        )
    if summary.get("preprocessing_version") != EXTRACTOR_PREPROCESSING_VERSION:
        raise AnalysisInputError(
            f"{path}: expected preprocessing_version "
            f"{EXTRACTOR_PREPROCESSING_VERSION!r}"
        )
    smearing = summary.get("smearing")
    if not isinstance(smearing, Mapping):
        raise AnalysisInputError(f"{path}: missing smearing metadata")
    expected_smearing_metadata = {
        "enabled": True,
        "preprocessing_version": EXTRACTOR_PREPROCESSING_VERSION,
        "model_id": EXTRACTOR_SMEARING_MODEL_ID,
        "eta_preselection": EXTRACTOR_SMEARING_ETA_PRESELECTION,
        "pt_threshold": EXTRACTOR_SMEARING_PT_THRESHOLD,
        "smear_before_pt_threshold": True,
        "acceptance_order": EXTRACTOR_SMEARING_ACCEPTANCE_ORDER,
    }
    incompatible_smearing = [
        name
        for name, expected in expected_smearing_metadata.items()
        if smearing.get(name) != expected
    ]
    if incompatible_smearing:
        raise AnalysisInputError(
            f"{path}: incompatible smearing metadata: "
            + ", ".join(incompatible_smearing)
        )
    if summary.get("tag_efficiencies_applied") is not False:
        raise AnalysisInputError(f"{path}: tagging must not be pre-applied")
    input_counter = summary.get("input_counter", {})
    output_counter = summary.get("reconstructable_counter", {})
    try:
        values = (
            int(input_counter["events"]),
            float(input_counter["sumw"]),
            float(input_counter["sumw2"]),
            int(output_counter["events"]),
            float(output_counter["sumw"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisInputError(f"{path}: incomplete input/reconstructable counters") from error
    if values[0] <= 0 or values[2] < 0.0 or values[1] == 0.0:
        raise AnalysisInputError(f"{path}: invalid full-sample normalization counter")
    return (*values, summary)


def _load_npz(path: Path, branches: Sequence[str], max_events: int | None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in branches if name not in payload]
        if missing:
            raise AnalysisInputError(f"{path}: missing arrays {', '.join(missing)}")
        limit = slice(None, max_events)
        return {name: np.asarray(payload[name])[limit] for name in branches}


def _load_uproot(
    path: Path, tree_name: str, branches: Sequence[str], max_events: int | None
) -> dict[str, np.ndarray]:
    import uproot  # type: ignore

    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise AnalysisInputError(f"{path}: missing {tree_name} tree")
        tree = root_file[tree_name]
        missing = [name for name in branches if name not in tree.keys()]
        if missing:
            raise AnalysisInputError(f"{path}: missing branches {', '.join(missing)}")
        arrays = tree.arrays(list(branches), entry_stop=max_events, library="np")
    return {name: np.asarray(arrays[name]) for name in branches}


def _load_pyroot(
    path: Path, tree_name: str, branches: Sequence[str], max_events: int | None
) -> dict[str, np.ndarray]:
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise AnalysisInputError(f"cannot open ROOT file {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise AnalysisInputError(f"{path}: missing {tree_name} tree")
    missing = [name for name in branches if not tree.GetBranch(name)]
    if missing:
        root_file.Close()
        raise AnalysisInputError(f"{path}: missing branches {', '.join(missing)}")
    entries = int(tree.GetEntries())
    if max_events is not None:
        entries = min(entries, int(max_events))
    rows: dict[str, list[Any]] = {name: [] for name in branches}
    for entry in range(entries):
        tree.GetEntry(entry)
        for name in branches:
            value = getattr(tree, name)
            width = ARRAY_BRANCH_WIDTHS.get(name)
            rows[name].append(
                [value[index] for index in range(width)] if width is not None else value
            )
    root_file.Close()
    return {name: np.asarray(values) for name, values in rows.items()}


def _root_feature_contract(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read and validate the C++ model/audit TNamed contracts."""

    titles: dict[str, str] = {}
    try:
        import uproot  # type: ignore

        with uproot.open(path) as root_file:
            for name in ("feature_names_json", "audit_branch_names_json"):
                if name not in root_file:
                    raise AnalysisInputError(f"{path}: missing metadata {name}")
                titles[name] = str(root_file[name].member("fTitle"))
    except ImportError:
        try:
            import ROOT  # type: ignore
        except ImportError as error:
            raise RuntimeError("reading ROOT metadata requires uproot or PyROOT") from error
        root_file = ROOT.TFile.Open(str(path))
        if not root_file or root_file.IsZombie():
            raise AnalysisInputError(f"cannot open ROOT file {path}")
        for name in ("feature_names_json", "audit_branch_names_json"):
            obj = root_file.Get(name)
            if not obj:
                root_file.Close()
                raise AnalysisInputError(f"{path}: missing metadata {name}")
            titles[name] = str(obj.GetTitle())
        root_file.Close()
    try:
        model_names = tuple(json.loads(titles["feature_names_json"]))
        audit_names = tuple(json.loads(titles["audit_branch_names_json"]))
    except (json.JSONDecodeError, TypeError) as error:
        raise AnalysisInputError(f"{path}: malformed feature metadata") from error
    if model_names != CPP_MODEL_FEATURE_NAMES:
        raise AnalysisInputError(f"{path}: model feature allowlist differs from the Python contract")
    leaked = FORBIDDEN_MODEL_FEATURES.intersection(model_names)
    if leaked:
        raise AnalysisInputError(f"{path}: truth/audit features leaked into model metadata: {sorted(leaked)}")
    if not FORBIDDEN_MODEL_FEATURES.issubset(set(audit_names)):
        missing = sorted(FORBIDDEN_MODEL_FEATURES.difference(audit_names))
        raise AnalysisInputError(f"{path}: audit metadata omits forbidden fields: {missing}")
    return model_names, audit_names


def load_event_table(
    spec: SampleSpec,
    tree_name: str = DEFAULT_TREE,
    max_events: int | None = None,
    allow_partial_input: bool = False,
) -> EventTable:
    branches = (*SCALAR_BRANCHES, *ARRAY_BRANCH_WIDTHS)
    if not spec.root_file.is_file():
        raise AnalysisInputError(f"missing feature ROOT file: {spec.root_file}")
    if spec.root_file.suffix == ".npz":
        arrays = _load_npz(spec.root_file, branches, max_events)
    else:
        try:
            arrays = _load_uproot(spec.root_file, tree_name, branches, max_events)
        except ImportError:
            try:
                arrays = _load_pyroot(spec.root_file, tree_name, branches, max_events)
            except ImportError as error:
                raise RuntimeError("reading ROOT inputs requires uproot or PyROOT") from error
        _root_feature_contract(spec.root_file)

    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1:
        raise AnalysisInputError(f"{spec.root_file}: branches have inconsistent row counts")
    for name, width in ARRAY_BRANCH_WIDTHS.items():
        if arrays[name].ndim != 2 or arrays[name].shape[1] != width:
            raise AnalysisInputError(
                f"{spec.root_file}: {name} must have shape (N,{width}), got {arrays[name].shape}"
            )
    input_events, input_sumw, input_sumw2, reco_events, reco_sumw, summary = _summary_metadata(
        spec.summary_file
    )
    if int(summary.get("c_mistags", -1)) != spec.c_mistags or int(
        summary.get("light_mistags", -1)
    ) != spec.light_mistags:
        raise AnalysisInputError(
            f"{spec.sample_id}: extractor mistag multiplicities do not match the manifest"
        )
    if not allow_partial_input and input_events != spec.generated_events_expected:
        raise AnalysisInputError(
            f"{spec.sample_id}: extractor processed {input_events} events, expected "
            f"{spec.generated_events_expected}; refusing a non-physical denominator"
        )
    if max_events is None and len(arrays["weight"]) != reco_events:
        raise AnalysisInputError(
            f"{spec.sample_id}: tree has {len(arrays['weight'])} rows but summary has {reco_events}"
        )
    _validate_event_arrays(spec, arrays)
    return EventTable(
        arrays=arrays,
        input_events=input_events,
        input_sumw=input_sumw,
        input_sumw2=input_sumw2,
        reconstructable_events=reco_events,
        reconstructable_sumw=reco_sumw,
        summary=summary,
    )


def _validate_event_arrays(spec: SampleSpec, arrays: Mapping[str, np.ndarray]) -> None:
    for name in SCALAR_BRANCHES:
        if not np.all(np.isfinite(np.asarray(arrays[name], dtype=float))):
            raise AnalysisInputError(f"{spec.sample_id}: branch {name} contains non-finite values")
    event_index = np.asarray(arrays["event_index"], dtype=np.int64)
    if len(np.unique(event_index)) != len(event_index):
        raise AnalysisInputError(f"{spec.sample_id}: duplicate event_index values")
    nmerged = np.asarray(arrays["n_merged"], dtype=int)
    ndouble = np.asarray(arrays["n_double_b"], dtype=int)
    category = np.asarray(arrays["category"], dtype=int)
    expected_category = np.where(nmerged == 0, 0, np.where(nmerged <= 2, 1, 2))
    closure = (
        2 * ndouble
        + np.asarray(arrays["n_true_single"], dtype=int)
        + np.asarray(arrays["n_c_mistag"], dtype=int)
        + np.asarray(arrays["n_light_mistag"], dtype=int)
    )
    if np.any((nmerged < 0) | (nmerged > 4)):
        raise AnalysisInputError(f"{spec.sample_id}: n_merged lies outside [0,4]")
    if not np.array_equal(ndouble, nmerged):
        raise AnalysisInputError(f"{spec.sample_id}: n_double_b does not equal n_merged")
    if not np.array_equal(category, expected_category):
        raise AnalysisInputError(f"{spec.sample_id}: category does not match n_merged")
    if np.any(closure != 8):
        raise AnalysisInputError(
            f"{spec.sample_id}: tag closure 2*n_double+n_single+n_c+n_light=8 failed"
        )
    if np.any(np.asarray(arrays["n_c_mistag"], dtype=int) != spec.c_mistags):
        raise AnalysisInputError(
            f"{spec.sample_id}: per-event n_c_mistag differs from the manifest"
        )
    if np.any(
        np.asarray(arrays["n_light_mistag"], dtype=int) != spec.light_mistags
    ):
        raise AnalysisInputError(
            f"{spec.sample_id}: per-event n_light_mistag differs from the manifest"
        )
    if np.any(np.asarray(arrays["weight"], dtype=float) == 0.0):
        raise AnalysisInputError(f"{spec.sample_id}: zero raw event weight is not supported")


def _stable_seed(identifier: Any, seed: int = DEFAULT_SEED) -> int:
    payload = f"{int(seed)}\0{identifier}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def deterministic_folds(
    source_ids: Sequence[Any], event_indices: Sequence[int], n_folds: int = N_FOLDS, seed: int = DEFAULT_SEED
) -> np.ndarray:
    sources = np.asarray(source_ids, dtype=object)
    entries = np.asarray(event_indices, dtype=np.int64)
    if sources.ndim != 1 or entries.ndim != 1 or len(sources) != len(entries):
        raise ValueError("source_ids and event_indices must be matching one-dimensional arrays")
    folds = np.empty(len(sources), dtype=np.int16)
    keys = np.asarray([str(value) for value in sources], dtype=object)
    for source in sorted(set(keys.tolist())):
        positions = np.flatnonzero(keys == source)
        local = entries[positions]
        if len(np.unique(local)) != len(local):
            raise ValueError(f"source {source!r} has duplicate event indices")
        ordered = positions[np.argsort(local, kind="mergesort")]
        rng = np.random.default_rng(_stable_seed(source, seed))
        shuffled = ordered[rng.permutation(len(ordered))]
        folds[shuffled] = np.arange(len(shuffled)) % int(n_folds)
    return folds


def base_feature_matrix(table: EventTable) -> tuple[np.ndarray, tuple[str, ...]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for name in BASE_SCALAR_FEATURES:
        values = np.asarray(table.arrays[name], dtype=float).copy()
        if name in {"second_score", "score_gap"}:
            values[values < 0.0] = np.nan
        columns.append(values)
        names.append(name)
    for name in BASE_ARRAY_FEATURES:
        matrix = np.asarray(table.arrays[name], dtype=float).copy()
        labels = PAIR_LABELS if matrix.shape[1] == 6 else tuple(str(index + 1) for index in range(matrix.shape[1]))
        for index, label in enumerate(labels):
            values = matrix[:, index]
            if name == "jet_pt":
                values = values.copy()
                values[values <= 0.0] = np.nan
            columns.append(values)
            names.append(f"{name}_{label}")
    leaked = FORBIDDEN_MODEL_FEATURES.intersection(names)
    if leaked:
        raise AnalysisInputError(f"truth/audit fields entered the classifier: {sorted(leaked)}")
    if not set(names).issubset(set(CPP_MODEL_FEATURE_NAMES)):
        unknown = sorted(set(names).difference(CPP_MODEL_FEATURE_NAMES))
        raise AnalysisInputError(f"classifier base features are absent from C++ metadata: {unknown}")
    return np.column_stack(columns), tuple(names)


def _expand_parameter(value: float | np.ndarray, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(length, float(array))
    if array.ndim != 1 or len(array) != length or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and scalar or length N")
    return array


def engineer_features(
    base: np.ndarray,
    base_names: Sequence[str],
    table: EventTable,
    topology: str,
    *,
    ms: float | np.ndarray | None = None,
    m2: float | np.ndarray | None = None,
    m3: float | np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Append the topology-specific mass hypothesis and resonance residuals."""

    count = len(base)
    pair_mass = np.asarray(table.arrays["pair_mass"], dtype=float)
    m4h = np.asarray(table.arrays["m4h"], dtype=float)
    if topology == "direct":
        if ms is None:
            raise ValueError("direct features require MS")
        ms_values = _expand_parameter(ms, count, "MS")
        extra = np.column_stack(
            [
                ms_values / 5000.0,
                (m4h - ms_values) / ms_values,
                *(pair_mass[:, index] / ms_values for index in range(6)),
            ]
        )
        names = (
            "MS_over_5000",
            "m4h_minus_MS_over_MS",
            *(f"pair_mass_{label}_over_MS" for label in PAIR_LABELS),
        )
        return np.column_stack([base, extra]), tuple(base_names) + tuple(names)

    if topology != "cascade" or m2 is None or m3 is None:
        raise ValueError("cascade features require M2 and M3")
    m2_values = _expand_parameter(m2, count, "M2")
    m3_values = _expand_parameter(m3, count, "M3")
    pairing_indices = ((0, 5), (1, 4), (2, 3))
    scores = np.column_stack(
        [
            ((pair_mass[:, first] - m2_values) / m2_values) ** 2
            + ((pair_mass[:, second] - m2_values) / m2_values) ** 2
            for first, second in pairing_indices
        ]
    )
    order = np.argsort(scores, axis=1, kind="mergesort")
    rows = np.arange(count)
    best_index = order[:, 0]
    second_index = order[:, 1]
    first_lookup = np.asarray([item[0] for item in pairing_indices], dtype=int)
    second_lookup = np.asarray([item[1] for item in pairing_indices], dtype=int)
    eta1 = pair_mass[rows, first_lookup[best_index]]
    eta2 = pair_mass[rows, second_lookup[best_index]]
    best = scores[rows, best_index]
    second = scores[rows, second_index]
    extra = np.column_stack(
        [
            m2_values / 2400.0,
            m3_values / 5000.0,
            m2_values / m3_values,
            (m3_values - 2.0 * m2_values) / m3_values,
            (m4h - m3_values) / m3_values,
            best,
            second,
            second - best,
            (eta1 - m2_values) / m2_values,
            (eta2 - m2_values) / m2_values,
            np.abs(eta1 - eta2) / m2_values,
            eta1 / m2_values,
            eta2 / m2_values,
            best_index.astype(float),
        ]
    )
    names = (
        "M2_over_2400",
        "M3_over_5000",
        "M2_over_M3",
        "M3_minus_2M2_over_M3",
        "m4h_minus_M3_over_M3",
        "cascade_best_score",
        "cascade_second_score",
        "cascade_score_gap",
        "cascade_eta1_residual",
        "cascade_eta2_residual",
        "cascade_eta_mass_balance",
        "cascade_eta1_over_M2",
        "cascade_eta2_over_M2",
        "cascade_pairing_index",
    )
    return np.column_stack([base, extra]), tuple(base_names) + names


def tag_efficiency(
    table: EventTable, eps_b: float, eps_bb: float, eps_c: float, eps_light: float
) -> np.ndarray:
    arrays = table.arrays
    result = (
        float(eps_b) ** np.asarray(arrays["n_true_single"], dtype=int)
        * float(eps_bb) ** np.asarray(arrays["n_double_b"], dtype=int)
        * float(eps_c) ** np.asarray(arrays["n_c_mistag"], dtype=int)
        * float(eps_light) ** np.asarray(arrays["n_light_mistag"], dtype=int)
    )
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("tag efficiencies must be finite and non-negative")
    return result


def physical_event_weights(
    spec: SampleSpec,
    table: EventTable,
    luminosity_fb: float,
    hbb_branching_ratio: float,
    eps_b: float,
    eps_bb: float,
    eps_c: float,
    eps_light: float,
) -> np.ndarray:
    xsec = SIGNAL_HYPOTHESIS_FB if spec.is_signal else float(spec.cross_section_fb)
    prefactor = (
        float(luminosity_fb)
        * xsec
        * float(spec.k_factor)
        * float(spec.rate_factor)
        * float(hbb_branching_ratio) ** int(spec.hbb_power)
        / float(table.input_sumw)
    )
    return (
        np.asarray(table.arrays["weight"], dtype=float)
        * prefactor
        * tag_efficiency(table, eps_b, eps_bb, eps_c, eps_light)
    )


def load_sample(
    spec: SampleSpec,
    topology: str,
    tree_name: str,
    max_events: int | None,
    allow_partial_input: bool,
    luminosity_fb: float,
    hbb_branching_ratio: float,
    eps_b: float,
    eps_c: float,
    eps_light: float,
    tagging_scenarios: Mapping[str, float],
    seed: int,
) -> LoadedSample:
    table = load_event_table(spec, tree_name, max_events, allow_partial_input)
    base, names = base_feature_matrix(table)
    folds = deterministic_folds(
        np.full(table.entries, spec.sample_id, dtype=object),
        np.asarray(table.arrays["event_index"], dtype=np.int64),
        seed=seed,
    )
    scenario_weights = {
        scenario: physical_event_weights(
            spec,
            table,
            luminosity_fb,
            hbb_branching_ratio,
            eps_b,
            eps_bb,
            eps_c,
            eps_light,
        )
        for scenario, eps_bb in tagging_scenarios.items()
    }
    return LoadedSample(spec, table, folds, base, names, scenario_weights)


def _point_features(sample: LoadedSample, point: MassPoint) -> tuple[np.ndarray, tuple[str, ...]]:
    if point.topology == "direct":
        return engineer_features(
            sample.base_features,
            sample.base_feature_names,
            sample.table,
            "direct",
            ms=float(point.ms),
        )
    return engineer_features(
        sample.base_features,
        sample.base_feature_names,
        sample.table,
        "cascade",
        m2=float(point.m2),
        m3=float(point.m3),
    )


def _equal_signal_point_weights(
    weights: np.ndarray, point_ids: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(weights), dtype=float)
    selected_points = sorted(set(point_ids[selected].tolist()))
    for point_id in selected_points:
        mask = selected & (point_ids == point_id)
        denominator = float(np.sum(np.abs(weights[mask])))
        if denominator <= 0.0:
            raise AnalysisInputError(f"signal point {point_id} has zero classifier weight")
        result[mask] = np.abs(weights[mask]) / denominator
    total = float(np.sum(result[selected]))
    if total <= 0.0:
        raise AnalysisInputError("signal class has no training weight")
    return result / total


def _background_training_assignments(
    samples: Sequence[LoadedSample], points: Sequence[MassPoint], replicas: int, seed: int
) -> dict[str, np.ndarray]:
    if replicas < 1 or replicas > len(points):
        raise ValueError("background replicas must lie between one and the point count")
    assignments: dict[str, np.ndarray] = {}
    for sample in samples:
        sample_assignments = np.empty((sample.table.entries, replicas), dtype=np.int32)
        for row, event_index in enumerate(
            np.asarray(sample.table.arrays["event_index"], dtype=np.int64)
        ):
            rng = np.random.default_rng(
                _stable_seed(f"background-mass\0{sample.spec.sample_id}\0{event_index}", seed)
            )
            sample_assignments[row] = rng.choice(
                len(points), size=replicas, replace=False
            )
        assignments[sample.spec.sample_id] = sample_assignments
    return assignments


def train_crossfit_models(
    topology: str,
    signals: Sequence[LoadedSample],
    backgrounds: Sequence[LoadedSample],
    points: Sequence[MassPoint],
    output_dir: Path,
    seed: int,
    replicas: int = 3,
) -> tuple[list[Any], tuple[str, ...]]:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as error:
        raise RuntimeError("XGBoost is required for full or smoke analysis mode") from error

    signal_features: list[np.ndarray] = []
    signal_weights: list[np.ndarray] = []
    signal_folds: list[np.ndarray] = []
    signal_points: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    for sample in signals:
        features, names = _point_features(sample, sample.spec.point)
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise AnalysisInputError("signal feature contracts differ")
        signal_features.append(features)
        signal_weights.append(sample.scenario_weights["nominal"])
        signal_folds.append(sample.folds)
        signal_points.append(np.full(sample.table.entries, sample.spec.point.point_id, dtype=object))
    sx = np.concatenate(signal_features)
    sw = np.concatenate(signal_weights)
    sf = np.concatenate(signal_folds)
    sp = np.concatenate(signal_points)

    assignments = _background_training_assignments(backgrounds, points, replicas, seed)
    models: list[Any] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for rotation in range(N_FOLDS):
        signal_train = (sf != rotation) & (sf != (rotation + 1) % N_FOLDS)
        sx_train = sx[signal_train]
        sw_balanced = _equal_signal_point_weights(sw, sp, signal_train)[signal_train]

        bx_parts: list[np.ndarray] = []
        bw_parts: list[np.ndarray] = []
        for sample in backgrounds:
            original_weight = np.abs(sample.scenario_weights["nominal"])
            train_mask = (sample.folds != rotation) & (
                sample.folds != (rotation + 1) % N_FOLDS
            )
            point_indices = assignments[sample.spec.sample_id]
            for replica in range(replicas):
                assigned_points = [points[index] for index in point_indices[:, replica]]
                if topology == "direct":
                    features, names = engineer_features(
                        sample.base_features,
                        sample.base_feature_names,
                        sample.table,
                        topology,
                        ms=np.asarray([point.ms for point in assigned_points], dtype=float),
                    )
                else:
                    features, names = engineer_features(
                        sample.base_features,
                        sample.base_feature_names,
                        sample.table,
                        topology,
                        m2=np.asarray([point.m2 for point in assigned_points], dtype=float),
                        m3=np.asarray([point.m3 for point in assigned_points], dtype=float),
                    )
                if names != feature_names:
                    raise AnalysisInputError("background feature contract differs from signal")
                bx_parts.append(features[train_mask])
                bw_parts.append(original_weight[train_mask] / replicas)
        if not bx_parts:
            raise AnalysisInputError("no background training rows")
        bx_train = np.concatenate(bx_parts)
        bw_train = np.concatenate(bw_parts)
        bw_total = float(np.sum(bw_train))
        if bw_total <= 0.0:
            raise AnalysisInputError("background class has no classifier weight")
        # Keep equal class totals while setting the mean effective training-row
        # weight to one.  A unit total per class would make the fixed
        # min_child_weight=1 contract incapable of producing any split.
        common_class_total = 0.5 * float(len(sx_train) + len(bx_train))
        sw_balanced *= common_class_total
        bw_train *= common_class_total / bw_total
        x_train = np.concatenate([sx_train, bx_train])
        y_train = np.concatenate(
            [np.ones(len(sx_train), dtype=np.int8), np.zeros(len(bx_train), dtype=np.int8)]
        )
        w_train = np.concatenate([sw_balanced, bw_train])
        params = dict(FIXED_XGBOOST_PARAMS)
        params["random_state"] = int(seed + rotation)
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train, sample_weight=w_train, verbose=False)
        model.save_model(str(output_dir / f"{topology}_fold{rotation}.json"))
        models.append(model)
    return models, tuple(feature_names or ())


def load_or_train_crossfit_models(
    topology: str,
    signals: Sequence[LoadedSample],
    backgrounds: Sequence[LoadedSample],
    points: Sequence[MassPoint],
    output_dir: Path,
    seed: int,
    replicas: int,
    analysis_fingerprint: str,
) -> tuple[list[Any], tuple[str, ...], dict[str, str], bool]:
    """Resume a verified model set, or train and atomically record a new one."""

    try:
        import xgboost as xgb  # type: ignore
    except ImportError as error:
        raise RuntimeError("XGBoost is required for full or smoke analysis mode") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "model_manifest.json"
    model_paths = [output_dir / f"{topology}_fold{fold}.json" for fold in range(N_FOLDS)]
    _, expected_feature_names = _point_features(signals[0], signals[0].spec.point)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("analysis_fingerprint") != analysis_fingerprint:
            raise AnalysisInputError(
                f"{output_dir}: model checkpoint fingerprint does not match this analysis"
            )
        if tuple(manifest.get("feature_names", [])) != expected_feature_names:
            raise AnalysisInputError(f"{output_dir}: model feature contract changed")
        recorded_hashes = manifest.get("model_sha256", {})
        models: list[Any] = []
        verified_hashes: dict[str, str] = {}
        for fold, path in enumerate(model_paths):
            if not path.is_file():
                raise AnalysisInputError(f"model checkpoint is missing {path}")
            digest = _sha256_file(path)
            if recorded_hashes.get(path.name) != digest:
                raise AnalysisInputError(f"model checkpoint hash mismatch for {path}")
            resumed_params = dict(FIXED_XGBOOST_PARAMS)
            resumed_params["random_state"] = int(seed + fold)
            resumed_params["n_jobs"] = 1
            model = xgb.XGBClassifier(**resumed_params)
            model.load_model(str(path))
            model.set_params(n_jobs=1)
            models.append(model)
            verified_hashes[path.name] = digest
        return models, expected_feature_names, verified_hashes, True
    if any(path.exists() for path in model_paths):
        raise AnalysisInputError(
            f"{output_dir}: unmanifested model files would contaminate this run"
        )
    models, feature_names = train_crossfit_models(
        topology,
        signals,
        backgrounds,
        points,
        output_dir,
        seed,
        replicas=replicas,
    )
    model_hashes = {path.name: _sha256_file(path) for path in model_paths}
    _write_json(
        manifest_path,
        {
            "analysis_fingerprint": analysis_fingerprint,
            "topology": topology,
            "feature_names": feature_names,
            "folds": N_FOLDS,
            "seed": seed,
            "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMS,
            "model_sha256": model_hashes,
        },
    )
    return models, feature_names, model_hashes, False


def _predict_point_crossfit(
    sample: LoadedSample, point: MassPoint, models: Sequence[Any]
) -> CrossfitScores:
    """Engineer one point once and cache both disjoint cross-fit score roles."""

    if len(models) != N_FOLDS:
        raise ValueError(f"expected {N_FOLDS} cross-fit models, got {len(models)}")
    features, _ = _point_features(sample, point)
    test_scores = np.full(sample.table.entries, np.nan, dtype=float)
    validation_scores = np.full(sample.table.entries, np.nan, dtype=float)
    for rotation, model in enumerate(models):
        test_mask = sample.folds == rotation
        validation_mask = sample.folds == (rotation + 1) % N_FOLDS
        positions = np.flatnonzero(test_mask | validation_mask)
        predictions = np.asarray(model.predict_proba(features[positions]), dtype=float)[:, 1]
        test_local = test_mask[positions]
        test_scores[positions[test_local]] = predictions[test_local]
        validation_scores[positions[~test_local]] = predictions[~test_local]
    if not np.all(np.isfinite(test_scores)) or not np.all(np.isfinite(validation_scores)):
        raise AnalysisInputError(f"{sample.spec.sample_id}: incomplete cross-fit score cache")
    return CrossfitScores(test=test_scores, validation=validation_scores)


def _weighted_quantile(values: np.ndarray, quantiles: Sequence[float], weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    data = values[order]
    weight = np.abs(weights[order])
    if len(data) == 0 or float(np.sum(weight)) <= 0.0:
        raise AnalysisInputError("cannot define score bins from an empty background")
    positions = (np.cumsum(weight) - 0.5 * weight) / float(np.sum(weight))
    return np.interp(np.asarray(quantiles), positions, data, left=data[0], right=data[-1])


def binned_summary(scores: np.ndarray, weights: np.ndarray, edges: Sequence[float]) -> dict[str, np.ndarray]:
    edges_array = np.asarray(edges, dtype=float)
    assignments = np.searchsorted(edges_array[1:-1], scores, side="right")
    n_bins = len(edges_array) - 1
    yields = np.bincount(assignments, weights=weights, minlength=n_bins).astype(float)
    sumw2 = np.bincount(assignments, weights=weights**2, minlength=n_bins).astype(float)
    raw = np.bincount(assignments, minlength=n_bins).astype(int)
    neff = np.divide(yields**2, sumw2, out=np.zeros_like(yields), where=sumw2 > 0.0)
    return {"yield": yields, "sumw2": sumw2, "raw": raw, "neff": neff}


def candidate_binnings(scores: np.ndarray, weights: np.ndarray) -> list[list[float]]:
    base = _weighted_quantile(scores, (0.0, 0.50, 0.75, 0.90, 0.97, 1.0), weights)
    candidates: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for n_bins in range(2, 6):
        for interior in itertools.combinations(range(1, 5), n_bins - 1):
            edges = tuple(float(base[index]) for index in (0, *interior, 5))
            if any(right <= left for left, right in zip(edges, edges[1:])) or edges in seen:
                continue
            seen.add(edges)
            candidates.append(list(edges))
    return candidates


def _bounded_cls_roots(
    cls_evaluator: Callable[[float], Sequence[float]],
    root_solver: Callable[[Callable[[float], float], float, float], float],
    *,
    model_poi_bounds: Sequence[float],
    hard_cap: float,
    level: float = PYHF_CLS_LEVEL,
) -> dict[str, Any]:
    """Find observed and expected CLs roots without leaving the POI domain.

    pyhf's automatic scan expands an insufficient upper bracket by doubling it,
    even when that trial value lies outside the model's declared POI bounds.
    Here the CLs values at the hard upper cap are evaluated first, each of the
    six curves is classified independently, and a root solver is called only
    for curves that are already bracketed inside the allowed interval.
    """

    try:
        bounds = np.asarray(model_poi_bounds, dtype=float).reshape(-1)
        if (
            len(bounds) != 2
            or np.any(~np.isfinite(bounds))
            or bounds[1] <= bounds[0]
            or not math.isfinite(hard_cap)
            or hard_cap <= bounds[0]
            or not math.isfinite(level)
            or not 0.0 < level < 1.0
        ):
            raise ValueError("invalid POI bounds, hard cap, or CLs level")
    except Exception as error:
        return {
            "status": "failed",
            "diagnostic_type": "InvalidPoiBracketConfiguration",
            "reason": "cannot construct a finite bounded POI interval",
            "cause": {"type": type(error).__name__, "message": str(error)},
        }

    lower = float(bounds[0])
    model_upper = float(bounds[1])
    upper = min(float(hard_cap), model_upper)
    cache: dict[float, np.ndarray] = {}

    def evaluate(poi: float) -> np.ndarray:
        poi = float(poi)
        if poi < lower or poi > upper:
            raise ValueError(
                f"bounded CLs evaluator refused POI={poi:g} outside [{lower:g}, {upper:g}]"
            )
        if poi not in cache:
            values = np.asarray(cls_evaluator(poi), dtype=float).reshape(-1)
            if len(values) != len(PYHF_LIMIT_CURVES) or np.any(~np.isfinite(values)):
                raise RuntimeError(
                    "pyhf hypotest must return six finite observed/expected CLs values"
                )
            cache[poi] = values
        return cache[poi]

    # The upper-cap preflight is deliberately first: an absent crossing is a
    # normal bounded diagnostic, not an invitation to ask pyhf for 2*upper.
    try:
        upper_values = evaluate(upper)
        lower_values = evaluate(lower)
    except Exception as error:
        return {
            "status": "failed",
            "diagnostic_type": "PoiPreflightFailure",
            "reason": "CLs preflight failed inside the bounded POI interval",
            "cause": {"type": type(error).__name__, "message": str(error)},
            "model_poi_bounds": [lower, model_upper],
            "configured_poi_hard_cap": float(hard_cap),
            "effective_poi_hard_cap": upper,
            "tested_poi_values": sorted(cache),
        }

    curve_diagnostics: dict[str, dict[str, Any]] = {}
    curve_roots: dict[str, float | None] = {}
    for index, name in enumerate(PYHF_LIMIT_CURVES):
        cls_lower = float(lower_values[index])
        cls_upper = float(upper_values[index])
        diagnostic: dict[str, Any] = {
            "status": "pending",
            "cls_at_lower_bound": cls_lower,
            "cls_at_upper_cap": cls_upper,
            "poi_limit": None,
        }
        if cls_lower < level:
            diagnostic.update(
                status="failed",
                reason="CLs is already below the target at the POI lower bound",
            )
            curve_roots[name] = None
        elif cls_upper > level:
            diagnostic.update(
                status="unbounded",
                reason="CLs does not cross the target before the POI hard cap",
            )
            curve_roots[name] = None
        else:
            try:
                if cls_lower == level:
                    root = lower
                elif cls_upper == level:
                    root = upper
                else:
                    root = float(
                        root_solver(
                            lambda poi, curve=index: float(evaluate(poi)[curve]) - level,
                            lower,
                            upper,
                        )
                    )
                if not math.isfinite(root) or root < lower or root > upper:
                    raise RuntimeError("root solver returned a POI outside its bracket")
                diagnostic.update(status="ok", poi_limit=root)
                curve_roots[name] = root
            except Exception as error:
                diagnostic.update(
                    status="failed",
                    reason="bounded CLs root finding failed",
                    cause={"type": type(error).__name__, "message": str(error)},
                )
                curve_roots[name] = None
        curve_diagnostics[name] = diagnostic

    observed_status = curve_diagnostics["observed_asimov"]["status"]
    median_status = curve_diagnostics["expected_median"]["status"]
    required_unbounded = [
        name
        for name in ("observed_asimov", "expected_median")
        if curve_diagnostics[name]["status"] == "unbounded"
    ]
    incomplete = [
        name
        for name in PYHF_LIMIT_CURVES
        if curve_diagnostics[name]["status"] != "ok"
    ]
    if required_unbounded:
        status = "unbounded"
        diagnostic_type = "RequiredLimitCurveNotBracketed"
        reason = (
            "required limit curves lie above the configured POI hard cap: "
            + ", ".join(required_unbounded)
        )
    elif median_status != "ok" or observed_status != "ok":
        status = "failed"
        diagnostic_type = "RequiredLimitCurveFailure"
        reason = "the observed Asimov or expected median limit could not be solved"
    elif incomplete:
        status = "partial"
        diagnostic_type = "OptionalLimitBandsIncomplete"
        reason = "the observed Asimov and median limits are finite, but some bands are not"
    else:
        status = "ok"
        diagnostic_type = "AllLimitCurvesBounded"
        reason = "all observed and expected CLs curves cross inside the POI hard cap"

    return {
        "status": status,
        "diagnostic_type": diagnostic_type,
        "reason": reason,
        "curve_roots": curve_roots,
        "curve_diagnostics": curve_diagnostics,
        "incomplete_curves": incomplete,
        "model_poi_bounds": [lower, model_upper],
        "configured_poi_hard_cap": float(hard_cap),
        "effective_poi_hard_cap": upper,
        "cls_level": float(level),
        "hard_cap_preflight": {
            name: float(upper_values[index])
            for index, name in enumerate(PYHF_LIMIT_CURVES)
        },
        "tested_poi_values": sorted(cache),
        "n_hypotest_evaluations": len(cache),
    }


def _pyhf_limit(
    channels: Sequence[Mapping[str, Any]],
    background_norm: float = BACKGROUND_NORM_UNCERTAINTY,
    poi_hard_cap: float = DEFAULT_PYHF_POI_HARD_CAP,
    physical_limit_hard_cap_fb: float = DEFAULT_PYHF_PHYSICAL_LIMIT_HARD_CAP_FB,
    max_reference_attempts: int = DEFAULT_PYHF_REFERENCE_MAX_ATTEMPTS,
) -> dict[str, Any]:
    method_metadata = {"pyhf_limit_method_version": PYHF_LIMIT_METHOD_VERSION}
    try:
        import pyhf  # type: ignore
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "pyhf is not installed",
            **method_metadata,
        }
    try:
        from scipy.optimize import toms748  # type: ignore
    except ImportError as error:
        return {
            "status": "unavailable",
            "reason": "scipy is required for bounded pyhf root finding",
            "cause": {"type": type(error).__name__, "message": str(error)},
            **method_metadata,
        }
    try:
        prepared: list[dict[str, Any]] = []
        total_signal = 0.0
        total_background = 0.0
        total_background_variance = 0.0
        for channel in channels:
            signal = np.asarray(channel["signal"], dtype=float)
            background = np.asarray(channel["background"], dtype=float)
            signal_error = np.asarray(channel["signal_staterror"], dtype=float)
            background_error = np.asarray(channel["background_staterror"], dtype=float)
            if not (
                signal.shape
                == background.shape
                == signal_error.shape
                == background_error.shape
            ):
                raise ValueError("pyhf channel template shapes do not match")
            if (
                np.any(~np.isfinite(signal))
                or np.any(~np.isfinite(background))
                or np.any(~np.isfinite(signal_error))
                or np.any(~np.isfinite(background_error))
                or np.any(signal < 0.0)
                or np.any(background <= 0.0)
                or np.any(signal_error < 0.0)
                or np.any(background_error < 0.0)
            ):
                raise ValueError(
                    "pyhf templates require finite non-negative signal/errors and "
                    "strictly positive background"
                )
            prepared.append(
                {
                    "name": str(channel["name"]),
                    "signal": signal,
                    "background": background,
                    "signal_error": signal_error,
                    "background_error": background_error,
                }
            )
            total_signal += float(np.sum(signal))
            total_background += float(np.sum(background))
            total_background_variance += float(np.sum(np.square(background_error)))
        if total_signal <= 0.0 or total_background < 0.0:
            raise ValueError("non-positive total signal or negative total background")
        if not math.isfinite(poi_hard_cap) or poi_hard_cap <= 0.0:
            raise ValueError("the pyhf POI hard cap must be positive and finite")
        if (
            not math.isfinite(physical_limit_hard_cap_fb)
            or physical_limit_hard_cap_fb <= 0.0
        ):
            raise ValueError("the pyhf physical limit hard cap must be positive and finite")
        if (
            isinstance(max_reference_attempts, bool)
            or not isinstance(max_reference_attempts, int)
            or max_reference_attempts < 1
        ):
            raise ValueError("the pyhf reference-scale attempt count must be a positive integer")
        rough_limit = (
            3.0
            + 2.0 * math.sqrt(max(0.0, total_background + total_background_variance))
            + background_norm * total_background
        ) / total_signal
        if not math.isfinite(rough_limit) or rough_limit <= 0.0:
            raise ValueError("non-finite rough cross-section limit")
    except Exception as error:
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            **method_metadata,
        }

    # Fit a dimensionless mu against a signal template scaled to an O(limit)
    # reference cross section. Both signal yields and their absolute MC-stat
    # errors receive the same scale. If the observed or median curve does not
    # cross before the dimensionless POI cap, increase only this physical
    # reference scale and rebuild the workspace. Optional expected bands never
    # trigger a retry. Every individual attempt still preflights and solves
    # strictly inside POI bounds identical to those declared by the model.
    maximum_reference_xsec_fb = float(physical_limit_hard_cap_fb) / float(poi_hard_cap)
    reference_xsec_fb = min(float(rough_limit), maximum_reference_xsec_fb)

    def evaluate_reference(reference_xsec: float) -> dict[str, Any]:
        try:
            channel_specs = []
            observations = []
            for channel in prepared:
                name = channel["name"]
                scaled_signal = channel["signal"] * reference_xsec
                scaled_signal_error = channel["signal_error"] * reference_xsec
                signal_modifiers = [{"name": "mu", "type": "normfactor", "data": None}]
                if np.any(scaled_signal_error > 0.0):
                    signal_modifiers.append(
                        {
                            "name": f"signal_stat_{name}",
                            "type": "staterror",
                            "data": scaled_signal_error.tolist(),
                        }
                    )
                background_modifiers = [
                    {
                        "name": "background_norm",
                        "type": "normsys",
                        "data": {
                            "hi": 1.0 + background_norm,
                            "lo": 1.0 - background_norm,
                        },
                    },
                    {
                        "name": f"background_stat_{name}",
                        "type": "staterror",
                        "data": channel["background_error"].tolist(),
                    },
                ]
                channel_specs.append(
                    {
                        "name": name,
                        "samples": [
                            {
                                "name": "signal",
                                "data": scaled_signal.tolist(),
                                "modifiers": signal_modifiers,
                            },
                            {
                                "name": "background",
                                "data": channel["background"].tolist(),
                                "modifiers": background_modifiers,
                            },
                        ],
                    }
                )
                observations.append(
                    {"name": name, "data": channel["background"].tolist()}
                )
            workspace = pyhf.Workspace(
                {
                    "channels": channel_specs,
                    "observations": observations,
                    "measurements": [
                        {
                            "name": "expected_limit",
                            "config": {
                                "poi": "mu",
                                "parameters": [
                                    {
                                        "name": "mu",
                                        "bounds": [[0.0, float(poi_hard_cap)]],
                                        "inits": [min(1.0, 0.5 * float(poi_hard_cap))],
                                    }
                                ],
                            },
                        }
                    ],
                    "version": "1.0.0",
                }
            )
            model = workspace.model(measurement_name="expected_limit")
            data = workspace.data(model)
            poi_slice = model.config.par_slice(model.config.poi_name)
            model_poi_bounds = model.config.suggested_bounds()[poi_slice.start]
            if (
                not math.isclose(float(model_poi_bounds[0]), 0.0, abs_tol=1.0e-12)
                or not math.isclose(
                    float(model_poi_bounds[1]),
                    float(poi_hard_cap),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ):
                return {
                    "status": "failed",
                    "diagnostic_type": "PoiBoundMismatch",
                    "reason": "the workspace POI bounds do not match the configured hard cap",
                    "configured_poi_hard_cap": float(poi_hard_cap),
                    "model_poi_bounds": [float(value) for value in model_poi_bounds],
                }

            def cls_evaluator(poi: float) -> np.ndarray:
                observed, expected = pyhf.infer.hypotest(
                    poi,
                    data,
                    model,
                    return_expected_set=True,
                )
                observed_value = float(np.asarray(observed))
                expected_values = np.asarray(expected, dtype=float).reshape(-1)
                if len(expected_values) != 5:
                    raise RuntimeError(
                        "pyhf hypotest did not return five expected CLs bands"
                    )
                return np.concatenate(([observed_value], expected_values))

            return _bounded_cls_roots(
                cls_evaluator,
                lambda function, lower, upper: float(
                    toms748(
                        function,
                        lower,
                        upper,
                        k=2,
                        xtol=2.0e-12,
                        rtol=1.0e-4,
                    )
                ),
                model_poi_bounds=model_poi_bounds,
                hard_cap=float(poi_hard_cap),
                level=PYHF_CLS_LEVEL,
            )
        except Exception as error:
            return {
                "status": "failed",
                "diagnostic_type": "PyhfWorkspaceOrFitFailure",
                "reason": "pyhf workspace construction or bounded fit setup failed",
                "cause": {"type": type(error).__name__, "message": str(error)},
                "configured_poi_hard_cap": float(poi_hard_cap),
            }

    pyhf.set_backend("numpy")
    attempt_records: list[dict[str, Any]] = []
    bounded: dict[str, Any] = {
        "status": "failed",
        "diagnostic_type": "NoPyhfFitAttempt",
        "reason": "no bounded pyhf fit attempt was made",
    }
    required_unbounded: list[str] = []
    rescaling_stop_reason = "fit_failed"
    for attempt_index in range(max_reference_attempts):
        bounded = evaluate_reference(reference_xsec_fb)
        roots = bounded.get("curve_roots", {})
        finite_roots = [
            float(root)
            for root in roots.values()
            if root is not None and math.isfinite(float(root))
        ]
        tested_poi_values = [float(value) for value in bounded.get("tested_poi_values", [])]
        attempt_record = {
            "attempt": attempt_index + 1,
            "reference_xsec_fb": reference_xsec_fb,
            "physical_poi_cap_fb": reference_xsec_fb
            * float(bounded.get("effective_poi_hard_cap", poi_hard_cap)),
            "status": bounded.get("status", "failed"),
            "diagnostic_type": bounded.get("diagnostic_type"),
            "maximum_returned_mu": max(finite_roots) if finite_roots else None,
            "incomplete_curves": bounded.get("incomplete_curves"),
            "hard_cap_preflight": bounded.get("hard_cap_preflight"),
            "tested_poi_values": tested_poi_values,
            "tested_physical_xsec_values_fb": [
                reference_xsec_fb * value for value in tested_poi_values
            ],
            "n_hypotest_evaluations": bounded.get("n_hypotest_evaluations", 0),
        }
        attempt_records.append(attempt_record)

        diagnostics = bounded.get("curve_diagnostics")
        required_unbounded = (
            [
                name
                for name in ("observed_asimov", "expected_median")
                if diagnostics.get(name, {}).get("status") == "unbounded"
            ]
            if isinstance(diagnostics, Mapping)
            else []
        )
        if not required_unbounded:
            rescaling_stop_reason = (
                "required_curves_bracketed"
                if bounded.get("status") in {"ok", "partial"}
                else "fit_failed_without_an_unbounded_required_curve"
            )
            break
        if attempt_index + 1 >= max_reference_attempts:
            rescaling_stop_reason = "maximum_reference_attempts_reached"
            break
        next_reference_xsec_fb = min(
            2.0 * reference_xsec_fb,
            maximum_reference_xsec_fb,
        )
        if next_reference_xsec_fb <= reference_xsec_fb * (1.0 + 1.0e-12):
            rescaling_stop_reason = "physical_limit_hard_cap_reached"
            break
        attempt_record["rescaled_for_unbounded_required_curves"] = required_unbounded
        attempt_record["next_reference_xsec_fb"] = next_reference_xsec_fb
        reference_xsec_fb = next_reference_xsec_fb

    if required_unbounded:
        bounded = dict(bounded)
        bounded["reason"] = (
            f"{bounded.get('reason', 'a required limit curve is not bracketed')}; "
            f"adaptive reference rescaling stopped: {rescaling_stop_reason}"
        )
        bounded["rescaling_stop_reason"] = rescaling_stop_reason

    roots = bounded.get("curve_roots", {})
    physical_limits = {
        name: (
            None
            if roots.get(name) is None
            else float(roots[name]) * reference_xsec_fb
        )
        for name in PYHF_LIMIT_CURVES
    }
    finite_roots = [float(root) for root in roots.values() if root is not None]
    maximum_mu = max(finite_roots) if finite_roots else None
    return {
        "status": bounded["status"],
        **physical_limits,
        "reason": bounded.get("reason"),
        "diagnostic_type": bounded.get("diagnostic_type"),
        "cause": bounded.get("cause"),
        **method_metadata,
        "pyhf_version": getattr(pyhf, "__version__", None),
        "poi_parameter": "mu",
        "sigma_reference_fb": reference_xsec_fb,
        "rough_limit_fb": rough_limit,
        "mu_fit_bounds": bounded.get("model_poi_bounds"),
        "configured_poi_hard_cap": float(poi_hard_cap),
        "effective_poi_hard_cap": bounded.get("effective_poi_hard_cap"),
        "configured_physical_limit_hard_cap_fb": float(physical_limit_hard_cap_fb),
        "physical_limit_cap_fb": reference_xsec_fb
        * float(bounded.get("effective_poi_hard_cap", poi_hard_cap)),
        "reference_xsec_hard_cap_fb": maximum_reference_xsec_fb,
        "max_reference_attempts": max_reference_attempts,
        "rescaling_stop_reason": bounded.get(
            "rescaling_stop_reason", rescaling_stop_reason
        ),
        "maximum_returned_mu": maximum_mu,
        "boundary_fraction_threshold": 0.80,
        "cls_level": bounded.get("cls_level", PYHF_CLS_LEVEL),
        "hard_cap_preflight": bounded.get("hard_cap_preflight"),
        "curve_diagnostics": bounded.get("curve_diagnostics"),
        "incomplete_curves": bounded.get("incomplete_curves"),
        "tested_poi_values": bounded.get("tested_poi_values", []),
        "n_hypotest_evaluations": sum(
            int(record.get("n_hypotest_evaluations", 0))
            for record in attempt_records
        ),
        "fit_attempts": attempt_records,
    }


def _select_category_edges(
    signal_scores: np.ndarray,
    signal_weights: np.ndarray,
    background_scores: np.ndarray,
    background_weights: Mapping[str, np.ndarray],
    min_raw: int,
    min_neff: float,
) -> dict[str, Any]:
    if len(signal_scores) == 0 or len(background_scores) == 0:
        return {"status": "invalid", "reason": "empty signal or background category", "candidates": []}
    if "nominal" not in background_weights or not background_weights:
        raise ValueError("background validation weights require a nominal scenario")
    for scenario, weights in background_weights.items():
        if len(weights) != len(background_scores):
            raise ValueError(f"{scenario} background scores and weights do not match")
    evaluated: list[dict[str, Any]] = []
    for edges in candidate_binnings(background_scores, background_weights["nominal"]):
        scenario_summaries = {
            scenario: binned_summary(background_scores, weights, edges)
            for scenario, weights in background_weights.items()
        }
        valid = bool(
            all(
                np.all(summary["yield"] > 0.0)
                and np.all(summary["raw"] >= min_raw)
                and np.all(summary["neff"] >= min_neff)
                for summary in scenario_summaries.values()
            )
        )
        nominal_bkg = scenario_summaries["nominal"]
        record: dict[str, Any] = {
            "edges": edges,
            "n_bins": len(edges) - 1,
            "valid_mc": valid,
            "scenario_validation": {
                scenario: {
                    "yield": summary["yield"].tolist(),
                    "raw": summary["raw"].tolist(),
                    "neff": summary["neff"].tolist(),
                }
                for scenario, summary in scenario_summaries.items()
            },
        }
        if valid:
            sig = binned_summary(signal_scores, signal_weights, edges)
            denominator = (
                nominal_bkg["yield"]
                + nominal_bkg["sumw2"]
                + np.square(BACKGROUND_NORM_UNCERTAINTY * nominal_bkg["yield"])
            )
            sensitivity_proxy = float(
                np.sum(
                    np.divide(
                        np.square(sig["yield"]),
                        denominator,
                        out=np.zeros_like(denominator),
                        where=denominator > 0.0,
                    )
                )
            )
            record["validation_sensitivity_proxy"] = sensitivity_proxy
        evaluated.append(record)
    valid_candidates = [item for item in evaluated if item["valid_mc"]]
    if valid_candidates:
        best_value = max(item["validation_sensitivity_proxy"] for item in valid_candidates)
        near_best = [
            item
            for item in valid_candidates
            if item["validation_sensitivity_proxy"] >= 0.99 * best_value
        ]
        chosen = min(near_best, key=lambda item: item["n_bins"])
        return {
            "status": "ok",
            "fallback_level": "shape_2_to_5_bins",
            "thresholds_relaxed": False,
            "edges": chosen["edges"],
            "candidates": evaluated,
        }
    inclusive_edges = [0.0, 1.0]
    inclusive_summaries = {
        scenario: binned_summary(background_scores, weights, inclusive_edges)
        for scenario, weights in background_weights.items()
    }
    inclusive_valid = all(
        np.all(summary["yield"] > 0.0)
        and np.all(summary["raw"] >= 1)
        and np.all(summary["neff"] > 0.0)
        for summary in inclusive_summaries.values()
    )
    if inclusive_valid:
        return {
            "status": "ok",
            "fallback_level": "inclusive_1_bin",
            "thresholds_relaxed": True,
            "edges": inclusive_edges,
            "candidates": evaluated,
            "inclusive_scenario_validation": {
                scenario: {
                    "yield": summary["yield"].tolist(),
                    "raw": summary["raw"].tolist(),
                    "neff": summary["neff"].tolist(),
                }
                for scenario, summary in inclusive_summaries.items()
            },
        }
    return {"status": "invalid", "reason": "no valid 2--5-bin background template", "candidates": evaluated}


def _category_mask(table: EventTable, category: int) -> np.ndarray:
    return np.asarray(table.arrays["category"], dtype=int) == int(category)


def _sum_by_nmerged(values: np.ndarray, table: EventTable, mask: np.ndarray | None = None) -> list[float]:
    selected = np.ones(table.entries, dtype=bool) if mask is None else mask
    nmerged = np.asarray(table.arrays["n_merged"], dtype=int)
    return [float(np.sum(values[selected & (nmerged == index)])) for index in range(5)]


def normalization_audit(
    signals: Sequence[LoadedSample],
    backgrounds: Sequence[LoadedSample],
    luminosity_fb: float,
    hbb_branching_ratio: float,
    eps_b: float,
    eps_c: float,
    eps_light: float,
    tagging_scenarios: Mapping[str, float],
) -> dict[str, Any]:
    # Keep the published benchmark closure immutable.  Runtime overrides are
    # audited independently and must not be compared with the fixed defaults.
    benchmark_produced = LUMINOSITY_FB * SIGNAL_HYPOTHESIS_FB
    benchmark_eight_b = benchmark_produced * HBB_BRANCHING_RATIO**4
    benchmark_nominal = benchmark_eight_b * EPS_B**8
    runtime_produced = luminosity_fb * SIGNAL_HYPOTHESIS_FB
    runtime_eight_b = runtime_produced * hbb_branching_ratio**4
    runtime_nominal = runtime_eight_b * eps_b**8
    checks = {
        "benchmark_produced_1fb_3000ifb": math.isclose(
            benchmark_produced, PRODUCED_SIGNAL_EVENTS, rel_tol=0, abs_tol=1e-12
        ),
        "benchmark_hbb4_closure": math.isclose(
            benchmark_eight_b, EIGHT_B_SIGNAL_EVENTS, rel_tol=0, abs_tol=1e-10
        ),
        "benchmark_nominal_tag_closure": math.isclose(
            benchmark_nominal, NOMINAL_TAG_SIGNAL_EVENTS, rel_tol=0, abs_tol=1e-10
        ),
    }
    point_checks: list[dict[str, Any]] = []
    for sample in signals:
        raw = np.asarray(sample.table.arrays["weight"], dtype=float)
        category_sum = sum(
            float(np.sum(raw[_category_mask(sample.table, category)])) for category in range(3)
        )
        expected_tagged = runtime_nominal * float(np.sum(raw)) / sample.table.input_sumw
        actual_tagged = float(np.sum(sample.scenario_weights["nominal"]))
        row = {
            "point_id": sample.spec.point.point_id,
            "input_events": sample.table.input_events,
            "input_sumw": sample.table.input_sumw,
            "reconstructable_events": sample.table.reconstructable_events,
            "tree_entries_loaded": sample.table.entries,
            "category_sumw_closes": math.isclose(category_sum, float(np.sum(raw)), rel_tol=1e-12, abs_tol=1e-12),
            "nominal_tagged_expected": expected_tagged,
            "nominal_tagged_actual": actual_tagged,
            "nominal_tagged_closes": math.isclose(expected_tagged, actual_tagged, rel_tol=1e-12, abs_tol=1e-12),
        }
        point_checks.append(row)
        checks[f"{sample.spec.point.point_id}_category_partition"] = row["category_sumw_closes"]
        checks[f"{sample.spec.point.point_id}_tagging"] = row["nominal_tagged_closes"]
    sample_checks: list[dict[str, Any]] = []
    for sample in (*signals, *backgrounds):
        raw = np.asarray(sample.table.arrays["weight"], dtype=float)
        categories = np.asarray(sample.table.arrays["category"], dtype=int)
        partition = sum(float(np.sum(raw[categories == index])) for index in range(3))
        row: dict[str, Any] = {
            "sample_id": sample.spec.sample_id,
            "role": sample.spec.role,
            "denominator_source": "extractor_input_counter.sumw",
            "input_sumw": sample.table.input_sumw,
            "tree_sumw": float(np.sum(raw)),
            "tree_matches_reconstructable_sumw": (
                None
                if sample.table.entries != sample.table.reconstructable_events
                else math.isclose(
                    float(np.sum(raw)),
                    sample.table.reconstructable_sumw,
                    rel_tol=1e-11,
                    abs_tol=1e-12,
                )
            ),
            "category_partition_closes": math.isclose(
                partition, float(np.sum(raw)), rel_tol=1e-12, abs_tol=1e-12
            ),
            "tag_efficiencies_applied_in_cpp": sample.table.summary.get(
                "tag_efficiencies_applied"
            ),
            "k_factor": sample.spec.k_factor,
            "hbb_power": sample.spec.hbb_power,
            "c_mistags": sample.spec.c_mistags,
            "light_mistags": sample.spec.light_mistags,
        }
        checks[f"{sample.spec.sample_id}_category_partition"] = row[
            "category_partition_closes"
        ]
        if row["tree_matches_reconstructable_sumw"] is not None:
            checks[f"{sample.spec.sample_id}_tree_sumw"] = row[
                "tree_matches_reconstructable_sumw"
            ]
        checks[f"{sample.spec.sample_id}_cpp_tagging_absent"] = (
            row["tag_efficiencies_applied_in_cpp"] is False
        )
        for scenario, eps_bb in tagging_scenarios.items():
            recomputed = physical_event_weights(
                sample.spec,
                sample.table,
                luminosity_fb,
                hbb_branching_ratio,
                eps_b,
                eps_bb,
                eps_c,
                eps_light,
            )
            closes = bool(
                np.allclose(
                    recomputed,
                    sample.scenario_weights[scenario],
                    rtol=1e-13,
                    atol=1e-15,
                )
            )
            row[f"{scenario}_tagging_closes"] = closes
            checks[f"{sample.spec.sample_id}_{scenario}_tagging"] = closes
        sample_checks.append(row)
    return {
        "benchmark_constants": {
            "signal_cross_section_hypothesis_fb": SIGNAL_HYPOTHESIS_FB,
            "signal_cross_section_definition": SIGNAL_XSEC_DEFINITION,
            "luminosity_fb_inverse": LUMINOSITY_FB,
            "hbb_branching_ratio": HBB_BRANCHING_RATIO,
            "eps_b": EPS_B,
            "eps_bb_nominal": EPS_B**2,
            "produced_events": benchmark_produced,
            "after_hbb4_events": benchmark_eight_b,
            "nominal_tag_before_reconstruction_events": benchmark_nominal,
        },
        "runtime_parameters": {
            "signal_cross_section_hypothesis_fb": SIGNAL_HYPOTHESIS_FB,
            "signal_cross_section_definition": SIGNAL_XSEC_DEFINITION,
            "luminosity_fb_inverse": luminosity_fb,
            "hbb_branching_ratio": hbb_branching_ratio,
            "eps_b": eps_b,
            "eps_bb_nominal": eps_b**2,
            "produced_events": runtime_produced,
            "after_hbb4_events": runtime_eight_b,
            "nominal_tag_before_reconstruction_events": runtime_nominal,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "points": point_checks,
        "samples": sample_checks,
    }


def _input_cross_section_rows(
    topology: str,
    signals: Sequence[LoadedSample],
    backgrounds: Sequence[LoadedSample],
    luminosity_fb: float,
    hbb_branching_ratio: float,
    eps_b: float,
    eps_c: float,
    eps_light: float,
    tagging_scenarios: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in (*signals, *backgrounds):
        spec = sample.spec
        point = spec.point.as_dict() if spec.point else {}
        xsec_used = SIGNAL_HYPOTHESIS_FB if spec.is_signal else spec.cross_section_fb
        rows.append(
            {
                "topology": topology,
                **point,
                "sample_id": spec.sample_id,
                "role": spec.role,
                "root_file": str(spec.root_file),
                "diagnostic_generated_cross_section_fb": spec.generated_cross_section_fb,
                "diagnostic_cross_section_source": spec.cross_section_source,
                "normalization_cross_section_fb": xsec_used,
                "is_one_fb_bsm_hypothesis": spec.is_signal,
                "signal_cross_section_definition": (
                    SIGNAL_XSEC_DEFINITION if spec.is_signal else None
                ),
                "k_factor": spec.k_factor,
                "rate_factor": spec.rate_factor,
                "hbb_power": spec.hbb_power,
                "hbb_factor": hbb_branching_ratio**spec.hbb_power,
                "c_mistags": spec.c_mistags,
                "light_mistags": spec.light_mistags,
                "eps_b": eps_b,
                "eps_c": eps_c,
                "eps_light": eps_light,
                "eps_bb_nominal": tagging_scenarios["nominal"],
                "eps_bb_conservative": tagging_scenarios["conservative"],
                "luminosity_fb_inverse": luminosity_fb,
                "generated_events_expected": spec.generated_events_expected,
                "lhe_event_count": spec.lhe_event_count,
                "hard_event_policy": spec.hard_event_policy,
                "hard_events_recycled": (
                    spec.hard_event_policy.strip().lower() in {"recycled", "reused"}
                ),
                "generated_events_processed": sample.table.input_events,
                "generated_sumw_from_extractor": sample.table.input_sumw,
                "generated_sumw2_from_extractor": sample.table.input_sumw2,
                "reconstructable_events": sample.table.reconstructable_events,
                "reconstructable_sumw": sample.table.reconstructable_sumw,
            }
        )
    return rows


def _category_yield_rows(
    topology: str,
    point: MassPoint,
    samples: Sequence[LoadedSample],
    valid_categories: set[int],
    tagging_scenarios: Mapping[str, float],
    luminosity_fb: float,
    hbb_branching_ratio: float,
    eps_b: float,
    eps_c: float,
    eps_light: float,
) -> list[dict[str, Any]]:
    """Write auditable production-to-limit yields for each exclusive category."""

    rows: list[dict[str, Any]] = []
    for scenario in tagging_scenarios:
        scenario_start = len(rows)
        for sample in samples:
            raw = np.asarray(sample.table.arrays["weight"], dtype=float)
            tagged = sample.scenario_weights[scenario]
            xsec_fb = (
                SIGNAL_HYPOTHESIS_FB
                if sample.spec.is_signal
                else float(sample.spec.cross_section_fb)
            )
            generated_yield = (
                float(luminosity_fb)
                * xsec_fb
                * float(sample.spec.k_factor)
                * float(sample.spec.rate_factor)
            )
            after_hbb_yield = generated_yield * float(hbb_branching_ratio) ** int(
                sample.spec.hbb_power
            )
            reconstructed = raw * after_hbb_yield / float(sample.table.input_sumw)
            tag_factors = tag_efficiency(
                sample.table,
                eps_b,
                tagging_scenarios[scenario],
                eps_c,
                eps_light,
            )
            if not np.allclose(
                tagged, reconstructed * tag_factors, rtol=1.0e-13, atol=1.0e-15
            ):
                raise AnalysisInputError(
                    f"{sample.spec.sample_id}: staged tagging weights do not close"
                )
            category_values: list[tuple[int, str, np.ndarray]] = [
                (index, name, _category_mask(sample.table, index))
                for index, name in enumerate(CATEGORY_NAMES)
            ]
            category_values.append(
                (-1, "all", np.ones(sample.table.entries, dtype=bool))
            )
            sample_rows: list[dict[str, Any]] = []
            for category, name, mask in category_values:
                category_is_used = category in valid_categories if category >= 0 else bool(
                    valid_categories
                )
                used_mask = (
                    mask
                    if category >= 0 and category in valid_categories
                    else (
                        mask
                        & np.isin(
                            np.asarray(sample.table.arrays["category"], dtype=int),
                            np.asarray(sorted(valid_categories), dtype=int),
                        )
                        if category < 0
                        else np.zeros(sample.table.entries, dtype=bool)
                    )
                )
                reconstructed_value = float(np.sum(reconstructed[mask]))
                tagged_value = float(np.sum(tagged[mask]))
                used_value = float(np.sum(tagged[used_mask]))
                row = {
                    "topology": topology,
                    **point.as_dict(),
                    "tagging_scenario": scenario,
                    "eps_bb": tagging_scenarios[scenario],
                    "sample_id": sample.spec.sample_id,
                    "role": sample.spec.role,
                    "normalization_cross_section_fb": xsec_fb,
                    "signal_cross_section_definition": (
                        SIGNAL_XSEC_DEFINITION if sample.spec.is_signal else None
                    ),
                    "category": name,
                    "category_index": category,
                    "stage_scope": "sample_total" if category < 0 else "category_with_sample_reference",
                    "category_used_in_limit": category_is_used,
                    "generated_input_events": sample.table.input_events,
                    "generated_input_sumw": sample.table.input_sumw,
                    "raw_entries": int(np.sum(mask)),
                    "raw_sumw": float(np.sum(raw[mask])),
                    "generated_yield": generated_yield,
                    "after_hbb_yield": after_hbb_yield,
                    "reconstructed_before_tag_yield": reconstructed_value,
                    "tagged_yield": tagged_value,
                    "after_xgboost_yield": used_value,
                    "used_in_limit_yield": used_value,
                    # Backward-compatible aliases with unambiguous staged names above.
                    "yield_before_tagging": reconstructed_value,
                    "yield_after_tagging": tagged_value,
                    "yield_in_limit": used_value,
                    "hbb_efficiency": (
                        after_hbb_yield / generated_yield if generated_yield != 0.0 else 0.0
                    ),
                    "reconstruction_efficiency": (
                        reconstructed_value / after_hbb_yield
                        if after_hbb_yield != 0.0
                        else 0.0
                    ),
                    "tagging_efficiency": (
                        tagged_value / reconstructed_value if reconstructed_value != 0.0 else 0.0
                    ),
                    "xgboost_used_efficiency": (
                        used_value / tagged_value if tagged_value != 0.0 else 0.0
                    ),
                    "hbb_stage_closure_pass": math.isclose(
                        after_hbb_yield,
                        generated_yield * hbb_branching_ratio ** sample.spec.hbb_power,
                        rel_tol=1.0e-13,
                        abs_tol=1.0e-15,
                    ),
                    "reconstruction_stage_closure_pass": math.isclose(
                        reconstructed_value,
                        after_hbb_yield * float(np.sum(raw[mask])) / sample.table.input_sumw,
                        rel_tol=1.0e-13,
                        abs_tol=1.0e-15,
                    ),
                    "tagging_stage_closure_pass": math.isclose(
                        tagged_value,
                        float(np.sum(reconstructed[mask] * tag_factors[mask])),
                        rel_tol=1.0e-13,
                        abs_tol=1.0e-15,
                    ),
                    "xgboost_stage_closure_pass": math.isclose(
                        used_value,
                        float(np.sum(tagged[used_mask])),
                        rel_tol=1.0e-13,
                        abs_tol=1.0e-15,
                    ),
                }
                nmerged_values = np.asarray(sample.table.arrays["n_merged"], dtype=int)
                for index in range(5):
                    nmask = mask & (nmerged_values == index)
                    nused = used_mask & (nmerged_values == index)
                    row[f"nmerged_{index}_reconstructed_before_tag_yield"] = float(
                        np.sum(reconstructed[nmask])
                    )
                    row[f"nmerged_{index}_tagged_yield"] = float(np.sum(tagged[nmask]))
                    row[f"nmerged_{index}_used_in_limit_yield"] = float(
                        np.sum(tagged[nused])
                    )
                    row[f"nmerged_{index}_yield"] = row[f"nmerged_{index}_tagged_yield"]
                rows.append(row)
                sample_rows.append(row)
            category_rows = [row for row in sample_rows if row["category_index"] >= 0]
            all_row = next(row for row in sample_rows if row["category_index"] < 0)
            for stage in (
                "reconstructed_before_tag_yield",
                "tagged_yield",
                "used_in_limit_yield",
            ):
                closure = math.isclose(
                    sum(float(row[stage]) for row in category_rows),
                    float(all_row[stage]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                all_row[f"category_partition_{stage}_closure_pass"] = closure
                if not closure:
                    raise AnalysisInputError(
                        f"{sample.spec.sample_id}: category partition failed for {stage}"
                    )
        scenario_rows = rows[scenario_start:]
        for category, name in (*tuple(enumerate(CATEGORY_NAMES)), (-1, "all")):
            components = [
                row
                for row in scenario_rows
                if row["category_index"] == category and row["role"] != "signal"
            ]
            if not components:
                continue
            total = {
                "topology": topology,
                **point.as_dict(),
                "tagging_scenario": scenario,
                "eps_bb": tagging_scenarios[scenario],
                "sample_id": "TOTAL_BACKGROUND",
                "role": "total_background",
                "normalization_cross_section_fb": None,
                "signal_cross_section_definition": None,
                "category": name,
                "category_index": category,
                "stage_scope": "sample_total" if category < 0 else "category_with_sample_reference",
                "category_used_in_limit": (
                    category in valid_categories if category >= 0 else bool(valid_categories)
                ),
                "generated_input_events": sum(
                    int(row["generated_input_events"]) for row in components
                ),
                "generated_input_sumw": sum(
                    float(row["generated_input_sumw"]) for row in components
                ),
                "raw_entries": sum(int(row["raw_entries"]) for row in components),
                "raw_sumw": sum(float(row["raw_sumw"]) for row in components),
                "generated_yield": sum(float(row["generated_yield"]) for row in components),
                "after_hbb_yield": sum(float(row["after_hbb_yield"]) for row in components),
                "reconstructed_before_tag_yield": sum(
                    float(row["reconstructed_before_tag_yield"]) for row in components
                ),
                "tagged_yield": sum(float(row["tagged_yield"]) for row in components),
                "after_xgboost_yield": sum(
                    float(row["after_xgboost_yield"]) for row in components
                ),
                "used_in_limit_yield": sum(
                    float(row["used_in_limit_yield"]) for row in components
                ),
            }
            total["yield_before_tagging"] = total["reconstructed_before_tag_yield"]
            total["yield_after_tagging"] = total["tagged_yield"]
            total["yield_in_limit"] = total["used_in_limit_yield"]
            total["hbb_efficiency"] = (
                total["after_hbb_yield"] / total["generated_yield"]
                if total["generated_yield"] != 0.0
                else 0.0
            )
            total["reconstruction_efficiency"] = (
                total["reconstructed_before_tag_yield"] / total["after_hbb_yield"]
                if total["after_hbb_yield"] != 0.0
                else 0.0
            )
            total["tagging_efficiency"] = (
                total["tagged_yield"] / total["reconstructed_before_tag_yield"]
                if total["reconstructed_before_tag_yield"] != 0.0
                else 0.0
            )
            total["xgboost_used_efficiency"] = (
                total["used_in_limit_yield"] / total["tagged_yield"]
                if total["tagged_yield"] != 0.0
                else 0.0
            )
            for closure_name in (
                "hbb_stage_closure_pass",
                "reconstruction_stage_closure_pass",
                "tagging_stage_closure_pass",
                "xgboost_stage_closure_pass",
            ):
                total[closure_name] = all(bool(row[closure_name]) for row in components)
            for index in range(5):
                for stage in (
                    "reconstructed_before_tag_yield",
                    "tagged_yield",
                    "used_in_limit_yield",
                ):
                    total[f"nmerged_{index}_{stage}"] = sum(
                        float(row[f"nmerged_{index}_{stage}"]) for row in components
                    )
                total[f"nmerged_{index}_yield"] = total[f"nmerged_{index}_tagged_yield"]
            rows.append(total)
        background_rows = [
            row
            for row in rows[scenario_start:]
            if row["role"] == "total_background"
        ]
        total_all = next(row for row in background_rows if row["category_index"] < 0)
        total_categories = [row for row in background_rows if row["category_index"] >= 0]
        for stage in (
            "reconstructed_before_tag_yield",
            "tagged_yield",
            "used_in_limit_yield",
        ):
            total_all[f"category_partition_{stage}_closure_pass"] = math.isclose(
                sum(float(row[stage]) for row in total_categories),
                float(total_all[stage]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
    return rows


def _combine_binned_summaries(
    summaries: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not summaries:
        raise ValueError("cannot combine an empty set of binned summaries")
    result = {
        key: np.sum([np.asarray(summary[key]) for summary in summaries], axis=0)
        for key in ("yield", "sumw2", "raw")
    }
    result["raw"] = np.asarray(result["raw"], dtype=int)
    result["neff"] = np.divide(
        np.square(result["yield"]),
        result["sumw2"],
        out=np.zeros_like(result["yield"], dtype=float),
        where=result["sumw2"] > 0.0,
    )
    return result


def _score_summary_rows(
    *,
    topology: str,
    point: MassPoint,
    scenario: str,
    eps_bb: float,
    category: str,
    fold: int,
    edges: Sequence[float],
    sample_id: str,
    role: str,
    aggregation_level: str,
    summary: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bin_index in range(len(edges) - 1):
        yield_value = float(summary["yield"][bin_index])
        row = {
            "topology": topology,
            **point.as_dict(),
            "tagging_scenario": scenario,
            "eps_bb": eps_bb,
            "category": category,
            "fold": fold,
            "bin": bin_index,
            "score_low": float(edges[bin_index]),
            "score_high": float(edges[bin_index + 1]),
            "score_edges_json": json.dumps(list(map(float, edges))),
            "sample_id": sample_id,
            "role": role,
            "aggregation_level": aggregation_level,
            "yield": yield_value,
            "sumw2": float(summary["sumw2"][bin_index]),
            "raw_entries": int(summary["raw"][bin_index]),
            "effective_entries": float(summary["neff"][bin_index]),
            "signal_yield_per_fb": yield_value if role == "signal" else None,
            "background_yield": yield_value if role != "signal" else None,
        }
        rows.append(row)
    total_yield = float(np.sum(summary["yield"]))
    total_sumw2 = float(np.sum(summary["sumw2"]))
    rows.append(
        {
            "topology": topology,
            **point.as_dict(),
            "tagging_scenario": scenario,
            "eps_bb": eps_bb,
            "category": category,
            "fold": fold,
            "bin": "all",
            "score_low": None,
            "score_high": None,
            "score_edges_json": json.dumps(list(map(float, edges))),
            "sample_id": sample_id,
            "role": role,
            "aggregation_level": f"{aggregation_level}_fold_total",
            "yield": total_yield,
            "sumw2": total_sumw2,
            "raw_entries": int(np.sum(summary["raw"])),
            "effective_entries": (
                total_yield**2 / total_sumw2 if total_sumw2 > 0.0 else 0.0
            ),
            "signal_yield_per_fb": total_yield if role == "signal" else None,
            "background_yield": total_yield if role != "signal" else None,
        }
    )
    return rows


def _all_fold_score_row(
    *,
    topology: str,
    point: MassPoint,
    scenario: str,
    eps_bb: float,
    category: str,
    sample_id: str,
    role: str,
    aggregation_level: str,
    summaries: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    yield_value = sum(float(np.sum(summary["yield"])) for summary in summaries)
    sumw2 = sum(float(np.sum(summary["sumw2"])) for summary in summaries)
    raw = sum(int(np.sum(summary["raw"])) for summary in summaries)
    return {
        "topology": topology,
        **point.as_dict(),
        "tagging_scenario": scenario,
        "eps_bb": eps_bb,
        "category": category,
        "fold": "all",
        "bin": "all",
        "score_low": None,
        "score_high": None,
        "score_edges_json": None,
        "sample_id": sample_id,
        "role": role,
        "aggregation_level": aggregation_level,
        "yield": yield_value,
        "sumw2": sumw2,
        "raw_entries": raw,
        "effective_entries": yield_value**2 / sumw2 if sumw2 > 0.0 else 0.0,
        "signal_yield_per_fb": yield_value if role == "signal" else None,
        "background_yield": yield_value if role != "signal" else None,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    """Atomically publish JSON so interrupted jobs never leave partial state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_input_fingerprint(spec: SampleSpec) -> dict[str, Any]:
    """Bind resumable results to the actual extractor output at each path."""

    root_stat = spec.root_file.stat()
    if not spec.summary_file.is_file():
        raise AnalysisInputError(f"missing extractor summary: {spec.summary_file}")
    return {
        "sample_id": spec.sample_id,
        "root_file": str(spec.root_file.resolve()),
        "root_size_bytes": int(root_stat.st_size),
        "root_mtime_ns": int(root_stat.st_mtime_ns),
        "summary_file": str(spec.summary_file.resolve()),
        "summary_sha256": _sha256_file(spec.summary_file),
    }


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisInputError(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisInputError(f"checkpoint {path} is not a JSON object")
    return value


def _point_shard_reuse_decision(
    shard: Mapping[str, Any],
    *,
    mode: str,
    point_id: str,
    run_fingerprint: str,
    tagging_scenarios: Sequence[str],
) -> tuple[bool, str]:
    """Decide whether a point shard is safe to reuse.

    Full-mode shards are physics checkpoints and are reusable only when they
    contain exactly one successful fit for every requested tagging scenario.
    Smoke shards are explicitly diagnostic: structurally complete scenario
    rows are reusable even when a fit is unavailable or failed.
    """

    if shard.get("run_fingerprint") != run_fingerprint:
        raise AnalysisInputError(
            f"point {point_id}: checkpoint fingerprint does not match this run"
        )
    if shard.get("point_id") != point_id:
        return False, "checkpoint ID mismatch"
    expected = list(tagging_scenarios)
    for key in ("point_category_yields", "score_bin_yields", "point_limits"):
        value = shard.get(key)
        if not isinstance(value, list):
            return False, f"{key} is not an array"
        if key == "point_category_yields" and not value:
            return False, "point_category_yields is empty"
        if any(
            not isinstance(row, Mapping) or row.get("point_id") != point_id
            for row in value
        ):
            return False, f"{key} contains a malformed or wrong-point row"
        if key != "point_limits":
            row_scenarios = [row.get("tagging_scenario") for row in value]
            if any(not isinstance(scenario, str) for scenario in row_scenarios) or set(
                row_scenarios
            ) != set(expected):
                return False, f"{key} does not cover exactly the requested scenarios"
    binning = shard.get("binning_audit")
    if not isinstance(binning, Mapping) or set(binning) != set(CATEGORY_NAMES):
        return False, "binning_audit does not contain the three categories"
    rows = shard.get("point_limits")
    assert isinstance(rows, list)  # checked above
    observed = [
        row.get("tagging_scenario") if isinstance(row, Mapping) else None
        for row in rows
    ]
    if (
        len(rows) != len(expected)
        or any(not isinstance(value, str) for value in observed)
        or len(set(observed)) != len(observed)
        or set(observed) != set(expected)
    ):
        return False, "point_limits does not contain exactly one row per scenario"
    if mode == "full" and any(
        not isinstance(row, Mapping) or row.get("status") != "ok" for row in rows
    ):
        return False, "full-mode point has a non-successful limit fit"
    fit_fields = (
        "observed_asimov",
        "expected_minus2sigma",
        "expected_minus1sigma",
        "expected_median",
        "expected_plus1sigma",
        "expected_plus2sigma",
    )
    if any(
        row.get("status") == "ok"
        and any(
            isinstance(row.get(field), bool)
            or not isinstance(row.get(field), (int, float))
            or not math.isfinite(float(row[field]))
            or float(row[field]) < 0.0
            for field in fit_fields
        )
        for row in rows
    ):
        return False, "successful limit row is missing finite fit results"
    if mode == "full" and (
        not shard.get("score_bin_yields")
        or any(
            isinstance(row.get("n_channels"), bool)
            or not isinstance(row.get("n_channels"), (int, float))
            or not math.isfinite(float(row["n_channels"]))
            or float(row["n_channels"]) <= 0.0
            for row in rows
        )
    ):
        return False, "full-mode successful limits lack score rows or channels"
    if mode == "smoke":
        return True, "complete diagnostic smoke shard"
    if mode == "full":
        return True, "complete successful full shard"
    return False, f"point shards are not reused in mode {mode!r}"


def _iter_point_shard_rows(
    shard_paths: Sequence[Path], key: str
) -> Iterable[Mapping[str, Any]]:
    """Yield rows while retaining at most one point shard in memory."""

    for path in shard_paths:
        shard = _read_json(path)
        rows = shard.get(key)
        if not isinstance(rows, list):
            raise AnalysisInputError(f"{path}: {key} is not a JSON array")
        for row in rows:
            if not isinstance(row, Mapping):
                raise AnalysisInputError(f"{path}: {key} contains a non-object row")
            yield row


def _write_sharded_csv(path: Path, shard_paths: Sequence[Path], key: str) -> None:
    """Assemble a CSV in two bounded-memory passes over point shards."""

    fields: list[str] = []
    for row in _iter_point_shard_rows(shard_paths, key):
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            handle.write("\n")
        else:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(_iter_point_shard_rows(shard_paths, key))
    os.replace(temporary, path)


def _write_sharded_json_array(
    path: Path, shard_paths: Sequence[Path], key: str
) -> None:
    """Assemble a valid JSON array without materializing campaign-wide rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for row in _iter_point_shard_rows(shard_paths, key):
            if not first:
                handle.write(",\n")
            handle.write("  ")
            json.dump(_json_safe(row), handle, sort_keys=True, separators=(",", ":"))
            first = False
        handle.write("\n]\n")
    os.replace(temporary, path)


def _write_sharded_binning_json(path: Path, shard_paths: Sequence[Path]) -> None:
    """Assemble the point-keyed binning object one shard at a time."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("{\n")
        first = True
        for shard_path in shard_paths:
            shard = _read_json(shard_path)
            point_id = shard.get("point_id")
            audit = shard.get("binning_audit")
            if not isinstance(point_id, str) or not isinstance(audit, Mapping):
                raise AnalysisInputError(
                    f"{shard_path}: malformed point_id or binning_audit"
                )
            if not first:
                handle.write(",\n")
            handle.write(f"  {json.dumps(point_id)}:")
            json.dump(_json_safe(audit), handle, sort_keys=True, separators=(",", ":"))
            first = False
        handle.write("\n}\n")
    os.replace(temporary, path)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _software_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "xgboost": _package_version("xgboost"),
        "pyhf": _package_version("pyhf"),
        "uproot": _package_version("uproot"),
        "matplotlib": _package_version("matplotlib"),
    }


def _require_pyhf_for_full_mode() -> str:
    try:
        import pyhf  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "pyhf is required in --mode full; install it before classifier training"
        ) from error
    return str(getattr(pyhf, "__version__", _package_version("pyhf")))


def _initialize_run_config(output_dir: Path, payload: Mapping[str, Any]) -> str:
    fingerprint = _json_fingerprint(payload)
    path = output_dir / "run_config.json"
    if path.exists():
        previous = _read_json(path)
        if previous.get("analysis_fingerprint") != fingerprint:
            raise AnalysisInputError(
                f"{output_dir}: existing outputs have a different analysis fingerprint; "
                "use a new output directory"
            )
    else:
        legacy_outputs = [
            output_dir / name
            for name in ("method_manifest.json", "point_limits.json", "models")
            if (output_dir / name).exists()
        ]
        if legacy_outputs:
            raise AnalysisInputError(
                f"{output_dir}: existing un-fingerprinted outputs would contaminate this run"
            )
        _write_json(
            path,
            {
                "analysis_fingerprint": fingerprint,
                "command": shlex.join(sys.argv),
                "configuration": payload,
            },
        )
    return fingerprint


def _write_plots(output_dir: Path, topology: str, limit_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    outputs: list[str] = []
    # A bounded fit may legitimately lack an outer expected band while still
    # providing a finite observed Asimov and median limit. Keep those partial
    # medians in the plots; unavailable uncertainty-band endpoints become gaps.
    valid = [
        row
        for row in limit_rows
        if row.get("status") in {"ok", "partial"}
        and isinstance(row.get("expected_median"), (int, float))
        and not isinstance(row.get("expected_median"), bool)
        and math.isfinite(float(row["expected_median"]))
        and float(row["expected_median"]) > 0.0
    ]
    if not valid:
        return outputs
    if topology == "direct":
        fig, axis = plt.subplots(figsize=(7.0, 5.2))
        for scenario, style in (("nominal", "-o"), ("conservative", "--s")):
            rows = sorted(
                [row for row in valid if row["tagging_scenario"] == scenario],
                key=lambda row: float(row["MS_GeV"]),
            )
            if not rows:
                continue
            x = np.asarray([row["MS_GeV"] for row in rows], dtype=float)
            y = np.asarray([row["expected_median"] for row in rows], dtype=float)
            axis.plot(x, y, style, label=scenario)
            if scenario == "nominal":
                low = np.asarray(
                    [
                        np.nan
                        if row.get("expected_minus1sigma") is None
                        else row["expected_minus1sigma"]
                        for row in rows
                    ],
                    dtype=float,
                )
                high = np.asarray(
                    [
                        np.nan
                        if row.get("expected_plus1sigma") is None
                        else row["expected_plus1sigma"]
                        for row in rows
                    ],
                    dtype=float,
                )
                axis.fill_between(x, low, high, alpha=0.22)
        axis.set(xlabel=r"$M_S$ [GeV]", ylabel=r"Expected 95% CL $\sigma_{95}$ [fb]", yscale="log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        stem = output_dir / "direct_sigma95"
    else:
        rows = [row for row in valid if row["tagging_scenario"] == "nominal"]
        if len(rows) < 3:
            return outputs
        x = np.asarray([row["M2_GeV"] for row in rows], dtype=float)
        y = np.asarray([row["M3_GeV"] for row in rows], dtype=float)
        z = np.asarray([row["expected_median"] for row in rows], dtype=float)
        fig, axis = plt.subplots(figsize=(7.2, 5.8))
        filled = axis.tricontourf(x, y, np.log10(z), levels=20)
        colorbar = fig.colorbar(filled, ax=axis)
        colorbar.set_label(r"$\log_{10}(\sigma_{95}/{\rm fb})$")
        axis.set(xlabel=r"$M_2$ [GeV]", ylabel=r"$M_3$ [GeV]")
        fig.tight_layout()
        stem = output_dir / "cascade_sigma95_contour"
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=180)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _write_html_report(
    output_dir: Path,
    topology: str,
    cross_rows: Sequence[Mapping[str, Any]],
    limit_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> Path:
    def table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
        header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
        body = []
        for row in rows:
            cells = "".join(html.escape(str(row.get(field, ""))) for field in fields)
            cells = "".join(
                f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields
            )
            body.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    cross_fields = (
        "sample_id",
        "role",
        "point_id",
        "diagnostic_generated_cross_section_fb",
        "normalization_cross_section_fb",
        "k_factor",
        "hbb_power",
        "generated_events_processed",
        "generated_sumw_from_extractor",
    )
    limit_fields = (
        "point_id",
        "tagging_scenario",
        "status",
        "expected_median",
        "expected_minus1sigma",
        "expected_plus1sigma",
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(topology)} resonance analysis</title>
<style>body{{font-family:sans-serif;max-width:1400px;margin:2rem auto}}table{{border-collapse:collapse;font-size:.82rem}}th,td{{border:1px solid #bbb;padding:.3rem .5rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}.ok{{color:#176b2c}}</style></head>
<body><h1>{html.escape(topology.title())} resonance analysis</h1>
<p>Method: {METHOD_VERSION}. Signal yields and limits use a 1 fb BSM production hypothesis; diagnostic generation cross sections are never substituted.</p>
<p class="{'ok' if audit.get('all_checks_pass') else ''}">Normalization audit: {audit.get('all_checks_pass')}</p>
<p><a href="input_cross_sections.csv">input cross sections</a> · <a href="point_category_yields.csv">category yields</a> · <a href="score_bin_yields.csv">score-bin yields</a> · <a href="point_limits.csv">point limits</a> · <a href="normalization_audit.json">normalization audit</a></p>
<h2>Inputs</h2>{table(cross_rows, cross_fields)}
<h2>Expected limits</h2>{table(limit_rows, limit_fields)}
</body></html>"""
    path = output_dir / "input_report.html"
    path.write_text(document, encoding="utf-8")
    return path


def run_analysis(args: argparse.Namespace) -> int:
    analysis_root = args.analysis_root.expanduser().resolve()
    signal_manifest = _resolve(analysis_root, args.signal_manifest)
    background_manifest = _resolve(analysis_root, args.background_manifest)
    signal_root_dir = _resolve(analysis_root, args.signal_root_dir)
    output_dir = _resolve(analysis_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke = args.mode == "smoke"
    pyhf_preflight_version = (
        _require_pyhf_for_full_mode() if args.mode == "full" else _package_version("pyhf")
    )
    tagging_scenarios = {
        "nominal": float(args.eps_b) ** 2,
        "conservative": float(args.eps_bb_conservative),
    }
    signal_specs = load_signal_specs(
        signal_manifest,
        args.topology,
        signal_root_dir,
        args.signal_root_pattern,
        args.smoke_points if smoke else None,
    )
    background_specs, missing_optional = load_background_specs(
        background_manifest, analysis_root, args.background_k_factor
    )
    sample_ids = [spec.sample_id for spec in (*signal_specs, *background_specs)]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted(
            sample_id for sample_id in set(sample_ids) if sample_ids.count(sample_id) > 1
        )
        raise AnalysisInputError(f"duplicate sample IDs: {duplicates}")
    point_ids = [spec.point.point_id for spec in signal_specs]
    if len(point_ids) != len(set(point_ids)):
        raise AnalysisInputError("signal manifest contains duplicate physical mass points")
    input_manifest_hashes = {
        "signal_manifest": _sha256_file(signal_manifest),
        "background_manifest": _sha256_file(background_manifest),
    }
    feature_input_fingerprints = [
        _feature_input_fingerprint(spec) for spec in (*signal_specs, *background_specs)
    ]
    extractor_source = analysis_root / "Code" / "FourHiggsResonanceAnalysis.cc"
    extractor_source_sha256 = (
        _sha256_file(extractor_source) if extractor_source.is_file() else None
    )
    analysis_source = Path(__file__).resolve()
    configuration = {
        "method_version": METHOD_VERSION,
        "tree_schema": TREE_SCHEMA,
        "extractor_method_version": EXTRACTOR_METHOD_VERSION,
        "extractor_preprocessing_version": EXTRACTOR_PREPROCESSING_VERSION,
        "extractor_smearing_model_id": EXTRACTOR_SMEARING_MODEL_ID,
        "extractor_smearing_acceptance_order": EXTRACTOR_SMEARING_ACCEPTANCE_ORDER,
        "topology": args.topology,
        "mode": args.mode,
        "tree_name": args.tree_name,
        "signal_manifest": str(signal_manifest),
        "signal_root_dir": str(signal_root_dir),
        "signal_root_pattern": args.signal_root_pattern,
        "background_manifest": str(background_manifest),
        "input_manifest_sha256": input_manifest_hashes,
        "feature_input_fingerprints": feature_input_fingerprints,
        "extractor_source": str(extractor_source),
        "extractor_source_sha256": extractor_source_sha256,
        "analysis_source": str(analysis_source),
        "analysis_source_sha256": _sha256_file(analysis_source),
        "signal_points": point_ids,
        "point_order": "signal_manifest",
        "signal_files": [str(spec.root_file) for spec in signal_specs],
        "background_samples": [
            {"sample_id": spec.sample_id, "root_file": str(spec.root_file)}
            for spec in background_specs
        ],
        "luminosity_fb_inverse": args.luminosity,
        "hbb_branching_ratio": args.hbb_branching_ratio,
        "eps_b": args.eps_b,
        "eps_bb_conservative": args.eps_bb_conservative,
        "eps_c": args.eps_c,
        "eps_light": args.eps_light,
        "background_k_factor_default": args.background_k_factor,
        "background_replicas": args.background_replicas,
        "min_background_raw": args.min_background_raw,
        "min_background_neff": args.min_background_neff,
        "background_norm_uncertainty": BACKGROUND_NORM_UNCERTAINTY,
        "pyhf_limit_method_version": PYHF_LIMIT_METHOD_VERSION,
        "pyhf_poi_hard_cap": args.pyhf_poi_hard_cap,
        "pyhf_physical_limit_hard_cap_fb": args.pyhf_physical_limit_hard_cap_fb,
        "pyhf_reference_max_attempts": args.pyhf_reference_max_attempts,
        "seed": args.seed,
        "smoke_points": args.smoke_points if smoke else None,
        "smoke_max_events": args.smoke_max_events if smoke else None,
        "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMS,
        "signal_cross_section_hypothesis_fb": SIGNAL_HYPOTHESIS_FB,
        "signal_cross_section_definition": SIGNAL_XSEC_DEFINITION,
    }
    analysis_fingerprint = _initialize_run_config(output_dir, configuration)
    max_events = args.smoke_max_events if smoke else None
    allow_partial = smoke
    signals = [
        load_sample(
            spec,
            args.topology,
            args.tree_name,
            max_events,
            allow_partial,
            args.luminosity,
            args.hbb_branching_ratio,
            args.eps_b,
            args.eps_c,
            args.eps_light,
            tagging_scenarios,
            args.seed,
        )
        for spec in signal_specs
    ]
    backgrounds = [
        load_sample(
            spec,
            args.topology,
            args.tree_name,
            max_events,
            allow_partial,
            args.luminosity,
            args.hbb_branching_ratio,
            args.eps_b,
            args.eps_c,
            args.eps_light,
            tagging_scenarios,
            args.seed,
        )
        for spec in background_specs
    ]
    background_provenance = [
        {
            "sample_id": sample.spec.sample_id,
            "generated_showers": sample.table.input_events,
            "lhe_event_count": sample.spec.lhe_event_count,
            "hard_event_policy": sample.spec.hard_event_policy,
            "hard_events_recycled": (
                sample.spec.hard_event_policy.strip().lower() in {"recycled", "reused"}
            ),
            "normalization_denominator": sample.table.input_sumw,
            "normalization_denominator_source": "extractor_input_counter.sumw",
        }
        for sample in backgrounds
    ]
    provenance_warnings = [
        f"{row['sample_id']}: manifest explicitly declares hard-event policy "
        f"{row['hard_event_policy']!r}"
        for row in background_provenance
        if row["hard_events_recycled"]
    ]
    audit = normalization_audit(
        signals,
        backgrounds,
        args.luminosity,
        args.hbb_branching_ratio,
        args.eps_b,
        args.eps_c,
        args.eps_light,
        tagging_scenarios,
    )
    if not audit["all_checks_pass"]:
        _write_json(output_dir / "normalization_audit.json", audit)
        raise AnalysisInputError("normalization closure failed; see normalization_audit.json")
    cross_rows = _input_cross_section_rows(
        args.topology,
        signals,
        backgrounds,
        args.luminosity,
        args.hbb_branching_ratio,
        args.eps_b,
        args.eps_c,
        args.eps_light,
        tagging_scenarios,
    )
    _write_csv(output_dir / "input_cross_sections.csv", cross_rows)
    _write_json(output_dir / "input_cross_sections.json", cross_rows)
    _write_json(output_dir / "normalization_audit.json", audit)

    if args.mode == "normalization-only":
        category_rows: list[dict[str, Any]] = []
        for signal in signals:
            category_rows.extend(
                _category_yield_rows(
                    args.topology,
                    signal.spec.point,
                    [signal, *backgrounds],
                    set(range(3)),
                    tagging_scenarios,
                    args.luminosity,
                    args.hbb_branching_ratio,
                    args.eps_b,
                    args.eps_c,
                    args.eps_light,
                )
            )
        _write_csv(output_dir / "point_category_yields.csv", category_rows)
        _write_json(output_dir / "point_category_yields.json", category_rows)
        _write_csv(output_dir / "score_bin_yields.csv", [])
        _write_json(output_dir / "score_bin_yields.json", [])
        manifest = {
            "method_version": METHOD_VERSION,
            "status": "normalization_only",
            "topology": args.topology,
            "extractor_method_version": EXTRACTOR_METHOD_VERSION,
            "extractor_preprocessing_version": EXTRACTOR_PREPROCESSING_VERSION,
            "extractor_smearing_model_id": EXTRACTOR_SMEARING_MODEL_ID,
            "extractor_smearing_acceptance_order": EXTRACTOR_SMEARING_ACCEPTANCE_ORDER,
            "analysis_fingerprint": analysis_fingerprint,
            "command": shlex.join(sys.argv),
            "configuration": configuration,
            "input_manifest_sha256": input_manifest_hashes,
            "software_versions": _software_versions(),
            "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMS,
            "signal_cross_section_hypothesis_fb": SIGNAL_HYPOTHESIS_FB,
            "signal_cross_section_definition": SIGNAL_XSEC_DEFINITION,
            "background_norm_uncertainty": BACKGROUND_NORM_UNCERTAINTY,
            "missing_optional_backgrounds": missing_optional,
            "background_provenance": background_provenance,
            "provenance_warnings": provenance_warnings,
            "normalization_audit_passed": True,
        }
        _write_json(output_dir / "method_manifest.json", manifest)
        _write_html_report(output_dir, args.topology, cross_rows, [], audit)
        return 0

    points = [sample.spec.point for sample in signals]
    replicas_used = min(args.background_replicas, len(points))
    models, feature_names, model_hashes, models_resumed = load_or_train_crossfit_models(
        args.topology,
        signals,
        backgrounds,
        points,
        output_dir / "models",
        args.seed,
        replicas=replicas_used,
        analysis_fingerprint=analysis_fingerprint,
    )
    run_fingerprint = _json_fingerprint(
        {"analysis_fingerprint": analysis_fingerprint, "model_sha256": model_hashes}
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    resumed_point_count = 0
    retried_point_count = 0
    retry_reasons: dict[str, str] = {}

    for signal in signals:
        point = signal.spec.point
        point_id = point.point_id
        shard_path = checkpoint_dir / f"{point_id}.json"
        if shard_path.exists():
            try:
                shard = _read_json(shard_path)
            except AnalysisInputError as error:
                reusable, reason = False, f"malformed JSON: {error}"
            else:
                reusable, reason = _point_shard_reuse_decision(
                    shard,
                    mode=args.mode,
                    point_id=point_id,
                    run_fingerprint=run_fingerprint,
                    tagging_scenarios=tuple(tagging_scenarios),
                )
            if reusable:
                shard_paths.append(shard_path)
                resumed_point_count += 1
                continue
            retried_point_count += 1
            retry_reasons[point_id] = reason
        point_category_rows: list[dict[str, Any]] = []
        point_score_rows: list[dict[str, Any]] = []
        point_limit_rows: list[dict[str, Any]] = []
        # Each sample/point is engineered and scored exactly once.  The cache
        # retains distinct validation and test roles for every source-local row.
        signal_score_cache = _predict_point_crossfit(signal, point, models)
        background_score_caches = {
            background.spec.sample_id: _predict_point_crossfit(background, point, models)
            for background in backgrounds
        }
        point_binning: dict[str, Any] = {}
        valid_categories: set[int] = set()
        selected_edges: dict[int, dict[int, list[float]]] = {}
        for category, category_name in enumerate(CATEGORY_NAMES):
            rotation_records: dict[str, Any] = {}
            category_edges: dict[int, list[float]] = {}
            for rotation in range(N_FOLDS):
                validation_fold = (rotation + 1) % N_FOLDS
                signal_validation_mask = (
                    _category_mask(signal.table, category)
                    & (signal.folds == validation_fold)
                )
                background_validation_scores: list[np.ndarray] = []
                background_validation_weights: dict[str, list[np.ndarray]] = {
                    scenario: [] for scenario in tagging_scenarios
                }
                for background in backgrounds:
                    mask = (
                        _category_mask(background.table, category)
                        & (background.folds == validation_fold)
                    )
                    cache = background_score_caches[background.spec.sample_id]
                    background_validation_scores.append(cache.validation[mask])
                    for scenario in tagging_scenarios:
                        background_validation_weights[scenario].append(
                            background.scenario_weights[scenario][mask]
                        )
                selection = _select_category_edges(
                    signal_score_cache.validation[signal_validation_mask],
                    signal.scenario_weights["nominal"][signal_validation_mask],
                    np.concatenate(background_validation_scores),
                    {
                        scenario: np.concatenate(parts)
                        for scenario, parts in background_validation_weights.items()
                    },
                    args.min_background_raw,
                    args.min_background_neff,
                )
                selection.update(
                    {
                        "rotation": rotation,
                        "validation_fold": validation_fold,
                        "test_fold": rotation,
                        "validation_test_disjoint": True,
                        "validation_signal_entries": int(np.sum(signal_validation_mask)),
                        "test_signal_entries": int(
                            np.sum(
                                _category_mask(signal.table, category)
                                & (signal.folds == rotation)
                            )
                        ),
                    }
                )
                rotation_records[str(rotation)] = selection
                if selection["status"] == "ok":
                    category_edges[rotation] = list(map(float, selection["edges"]))
            category_valid = len(category_edges) == N_FOLDS
            point_binning[category_name] = {
                "status": "ok" if category_valid else "invalid",
                "all_rotations_valid": category_valid,
                "rotations": rotation_records,
            }
            if category_valid:
                valid_categories.add(category)
                selected_edges[category] = category_edges
        point_category_rows.extend(
            _category_yield_rows(
                args.topology,
                point,
                [signal, *backgrounds],
                valid_categories,
                tagging_scenarios,
                args.luminosity,
                args.hbb_branching_ratio,
                args.eps_b,
                args.eps_c,
                args.eps_light,
            )
        )

        for scenario in tagging_scenarios:
            channels: list[dict[str, Any]] = []
            for category in sorted(valid_categories):
                category_name = CATEGORY_NAMES[category]
                signal_fold_summaries: list[dict[str, np.ndarray]] = []
                component_fold_summaries: dict[str, list[dict[str, np.ndarray]]] = {
                    background.spec.sample_id: [] for background in backgrounds
                }
                total_fold_summaries: list[dict[str, np.ndarray]] = []
                for fold in range(N_FOLDS):
                    edges = selected_edges[category][fold]
                    signal_mask = (
                        _category_mask(signal.table, category) & (signal.folds == fold)
                    )
                    signal_summary = binned_summary(
                        signal_score_cache.test[signal_mask],
                        signal.scenario_weights[scenario][signal_mask],
                        edges,
                    )
                    signal_fold_summaries.append(signal_summary)
                    point_score_rows.extend(
                        _score_summary_rows(
                            topology=args.topology,
                            point=point,
                            scenario=scenario,
                            eps_bb=tagging_scenarios[scenario],
                            category=category_name,
                            fold=fold,
                            edges=edges,
                            sample_id=signal.spec.sample_id,
                            role="signal",
                            aggregation_level="component",
                            summary=signal_summary,
                        )
                    )
                    component_summaries: list[dict[str, np.ndarray]] = []
                    for background in backgrounds:
                        mask = (
                            _category_mask(background.table, category)
                            & (background.folds == fold)
                        )
                        cache = background_score_caches[background.spec.sample_id]
                        summary = binned_summary(
                            cache.test[mask],
                            background.scenario_weights[scenario][mask],
                            edges,
                        )
                        component_summaries.append(summary)
                        component_fold_summaries[background.spec.sample_id].append(summary)
                        point_score_rows.extend(
                            _score_summary_rows(
                                topology=args.topology,
                                point=point,
                                scenario=scenario,
                                eps_bb=tagging_scenarios[scenario],
                                category=category_name,
                                fold=fold,
                                edges=edges,
                                sample_id=background.spec.sample_id,
                                role=background.spec.role,
                                aggregation_level="component",
                                summary=summary,
                            )
                        )
                    total_background = _combine_binned_summaries(component_summaries)
                    total_fold_summaries.append(total_background)
                    point_score_rows.extend(
                        _score_summary_rows(
                            topology=args.topology,
                            point=point,
                            scenario=scenario,
                            eps_bb=tagging_scenarios[scenario],
                            category=category_name,
                            fold=fold,
                            edges=edges,
                            sample_id="TOTAL_BACKGROUND",
                            role="total_background",
                            aggregation_level="total_background",
                            summary=total_background,
                        )
                    )
                    channels.append(
                        {
                            "name": f"{category_name}_fold{fold}",
                            "signal": signal_summary["yield"],
                            "background": total_background["yield"],
                            "signal_staterror": np.sqrt(signal_summary["sumw2"]),
                            "background_staterror": np.sqrt(total_background["sumw2"]),
                        }
                    )
                signal_all_row = _all_fold_score_row(
                    topology=args.topology,
                    point=point,
                    scenario=scenario,
                    eps_bb=tagging_scenarios[scenario],
                    category=category_name,
                    sample_id=signal.spec.sample_id,
                    role="signal",
                    aggregation_level="all_folds_category",
                    summaries=signal_fold_summaries,
                )
                expected_signal = float(
                    np.sum(
                        signal.scenario_weights[scenario][
                            _category_mask(signal.table, category)
                        ]
                    )
                )
                signal_all_row["tagged_category_closure_pass"] = math.isclose(
                    float(signal_all_row["yield"]),
                    expected_signal,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                point_score_rows.append(signal_all_row)
                component_all_rows: list[dict[str, Any]] = []
                for background in backgrounds:
                    row = _all_fold_score_row(
                        topology=args.topology,
                        point=point,
                        scenario=scenario,
                        eps_bb=tagging_scenarios[scenario],
                        category=category_name,
                        sample_id=background.spec.sample_id,
                        role=background.spec.role,
                        aggregation_level="all_folds_category",
                        summaries=component_fold_summaries[background.spec.sample_id],
                    )
                    expected = float(
                        np.sum(
                            background.scenario_weights[scenario][
                                _category_mask(background.table, category)
                            ]
                        )
                    )
                    row["tagged_category_closure_pass"] = math.isclose(
                        float(row["yield"]), expected, rel_tol=1.0e-12, abs_tol=1.0e-12
                    )
                    component_all_rows.append(row)
                    point_score_rows.append(row)
                total_all_row = _all_fold_score_row(
                    topology=args.topology,
                    point=point,
                    scenario=scenario,
                    eps_bb=tagging_scenarios[scenario],
                    category=category_name,
                    sample_id="TOTAL_BACKGROUND",
                    role="total_background",
                    aggregation_level="all_folds_category_total_background",
                    summaries=total_fold_summaries,
                )
                total_all_row["tagged_category_closure_pass"] = math.isclose(
                    float(total_all_row["yield"]),
                    sum(float(row["yield"]) for row in component_all_rows),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                point_score_rows.append(total_all_row)
            fit = (
                _pyhf_limit(
                    channels,
                    poi_hard_cap=args.pyhf_poi_hard_cap,
                    physical_limit_hard_cap_fb=args.pyhf_physical_limit_hard_cap_fb,
                    max_reference_attempts=args.pyhf_reference_max_attempts,
                )
                if channels
                else {"status": "invalid", "reason": "no valid category"}
            )
            point_limit_rows.append(
                {
                    "topology": args.topology,
                    **point.as_dict(),
                    "tagging_scenario": scenario,
                    "eps_bb": tagging_scenarios[scenario],
                    "valid_categories": ",".join(CATEGORY_NAMES[index] for index in sorted(valid_categories)),
                    "n_channels": len(channels),
                    **fit,
                }
            )
        _write_json(
            shard_path,
            {
                "run_fingerprint": run_fingerprint,
                "analysis_fingerprint": analysis_fingerprint,
                "model_sha256": model_hashes,
                "point_id": point_id,
                "point": point.as_dict(),
                "point_category_yields": point_category_rows,
                "score_bin_yields": point_score_rows,
                "point_limits": point_limit_rows,
                "binning_audit": point_binning,
            },
        )
        shard_paths.append(shard_path)

    # Campaign-wide products are assembled from ordered point shards.  Only
    # one point is resident while writing the potentially million-row score
    # outputs; the small limit table alone is collected for plotting/reporting.
    _write_sharded_csv(
        output_dir / "point_category_yields.csv", shard_paths, "point_category_yields"
    )
    _write_sharded_json_array(
        output_dir / "point_category_yields.json",
        shard_paths,
        "point_category_yields",
    )
    _write_sharded_csv(
        output_dir / "score_bin_yields.csv", shard_paths, "score_bin_yields"
    )
    _write_sharded_json_array(
        output_dir / "score_bin_yields.json", shard_paths, "score_bin_yields"
    )
    _write_sharded_csv(output_dir / "point_limits.csv", shard_paths, "point_limits")
    _write_sharded_json_array(
        output_dir / "point_limits.json", shard_paths, "point_limits"
    )
    _write_sharded_binning_json(output_dir / "binning_audit.json", shard_paths)
    limit_rows = list(_iter_point_shard_rows(shard_paths, "point_limits"))
    plot_outputs = _write_plots(output_dir, args.topology, limit_rows)
    report = _write_html_report(output_dir, args.topology, cross_rows, limit_rows, audit)
    limits_complete = len(limit_rows) == len(signals) * len(tagging_scenarios) and all(
        row.get("status") == "ok" for row in limit_rows
    )
    manifest = {
        "method_version": METHOD_VERSION,
        "status": "smoke" if smoke else ("complete" if limits_complete else "incomplete"),
        "physics_result_valid": not smoke and limits_complete,
        "limit_status_complete": limits_complete,
        "topology": args.topology,
        "tree_schema": TREE_SCHEMA,
        "extractor_method_version": EXTRACTOR_METHOD_VERSION,
        "extractor_preprocessing_version": EXTRACTOR_PREPROCESSING_VERSION,
        "extractor_smearing_model_id": EXTRACTOR_SMEARING_MODEL_ID,
        "extractor_smearing_acceptance_order": EXTRACTOR_SMEARING_ACCEPTANCE_ORDER,
        "analysis_fingerprint": analysis_fingerprint,
        "run_fingerprint": run_fingerprint,
        "command": shlex.join(sys.argv),
        "configuration": configuration,
        "input_manifest_sha256": input_manifest_hashes,
        "model_sha256": model_hashes,
        "software_versions": _software_versions(),
        "pyhf_preflight_version": pyhf_preflight_version,
        "feature_names": feature_names,
        "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMS,
        "folds": N_FOLDS,
        "seed": args.seed,
        "background_training_replicas": replicas_used,
        "parameterized_backgrounds_training_only": True,
        "pointwise_background_scoring": True,
        "score_cache_scope": "once_per_sample_and_physical_mass_point",
        "binning_selection": (
            "rotation-local validation fold r+1, applied only to disjoint test fold r; "
            "both tagging scenarios must pass template validation"
        ),
        "pyhf_channels": "category_by_test_fold",
        "pyhf_limit_method_version": PYHF_LIMIT_METHOD_VERSION,
        "pyhf_poi_parameterization": (
            "dimensionless mu with an explicit configured hard upper bound multiplying "
            "signal and signal-staterror templates scaled by sigma_reference_fb; "
            "observed and expected CLs roots are solved independently inside that bound"
        ),
        "pyhf_poi_hard_cap": args.pyhf_poi_hard_cap,
        "pyhf_physical_limit_hard_cap_fb": args.pyhf_physical_limit_hard_cap_fb,
        "pyhf_reference_max_attempts": args.pyhf_reference_max_attempts,
        "xgboost_used_stage_definition": (
            "sum of every score bin in categories with valid rotation-local templates; "
            "no post-training score cut is applied"
        ),
        "background_norm_uncertainty": BACKGROUND_NORM_UNCERTAINTY,
        "minimum_background_raw_per_shape_bin": args.min_background_raw,
        "minimum_background_neff_per_shape_bin": args.min_background_neff,
        "tagging_scenarios": tagging_scenarios,
        "eps_b": args.eps_b,
        "eps_c": args.eps_c,
        "eps_light": args.eps_light,
        "luminosity_fb_inverse": args.luminosity,
        "hbb_branching_ratio": args.hbb_branching_ratio,
        "signal_cross_section_hypothesis_fb": SIGNAL_HYPOTHESIS_FB,
        "signal_cross_section_definition": SIGNAL_XSEC_DEFINITION,
        "missing_optional_backgrounds": missing_optional,
        "background_provenance": background_provenance,
        "provenance_warnings": provenance_warnings,
        "future_background_roles": ["sm_hhhbb", "sm_hh4b"],
        "normalization_audit_passed": True,
        "models_resumed": models_resumed,
        "resumed_point_count": resumed_point_count,
        "retried_point_count": retried_point_count,
        "checkpoint_retry_reasons": retry_reasons,
        "full_checkpoint_reuse_policy": (
            "exactly one status=ok limit row for each tagging scenario"
        ),
        "smoke_checkpoint_reuse_policy": (
            "exactly one row for each tagging scenario; fit status may be diagnostic"
        ),
        "campaign_output_assembly": (
            "bounded-memory streaming from point shards in signal-manifest order"
        ),
        "checkpoint_directory": str(checkpoint_dir),
        "plot_outputs": plot_outputs,
        "html_report": str(report),
    }
    _write_json(output_dir / "method_manifest.json", manifest)
    if args.mode == "full" and not limits_complete:
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-set",
        choices=("resonance-hybrid-v1", "fatjet-ak8-softdrop-v1"),
        default="resonance-hybrid-v1",
    )
    parser.add_argument("--analysis-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--topology", required=True, choices=("direct", "cascade"))
    parser.add_argument("--mode", choices=("full", "smoke", "normalization-only"), default="full")
    parser.add_argument(
        "--signal-manifest", type=Path, default=Path("HerwigSignalPoints/mass_scan_10k/manifest.csv")
    )
    parser.add_argument(
        "--signal-root-dir", type=Path, default=Path("ResonanceAnalysis/features")
    )
    parser.add_argument(
        "--signal-root-pattern", default="{scenario}/{run_name}_resonance.root"
    )
    parser.add_argument(
        "--background-manifest", type=Path, default=Path("ResonanceAnalysis/background_manifest.csv")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tree-name", default=DEFAULT_TREE)
    parser.add_argument("--luminosity", type=float, default=LUMINOSITY_FB)
    parser.add_argument("--hbb-branching-ratio", type=float, default=HBB_BRANCHING_RATIO)
    parser.add_argument("--eps-b", type=float, default=EPS_B)
    parser.add_argument("--eps-bb-conservative", type=float, default=EPS_BB_CONSERVATIVE)
    parser.add_argument("--eps-c", type=float, default=EPS_C)
    parser.add_argument("--eps-light", type=float, default=EPS_LIGHT)
    parser.add_argument("--background-k-factor", type=float, default=2.0)
    parser.add_argument("--background-replicas", type=int, default=3)
    parser.add_argument("--min-background-raw", type=int, default=25)
    parser.add_argument("--min-background-neff", type=float, default=10.0)
    parser.add_argument(
        "--pyhf-poi-hard-cap",
        type=float,
        default=DEFAULT_PYHF_POI_HARD_CAP,
        help=(
            "Maximum dimensionless pyhf POI tested for any observed or expected "
            "limit curve; this is also the model POI upper bound."
        ),
    )
    parser.add_argument(
        "--pyhf-physical-limit-hard-cap-fb",
        type=float,
        default=DEFAULT_PYHF_PHYSICAL_LIMIT_HARD_CAP_FB,
        help=(
            "Maximum physical cross section tested while adaptively rescaling "
            "the pyhf signal reference."
        ),
    )
    parser.add_argument(
        "--pyhf-reference-max-attempts",
        type=int,
        default=DEFAULT_PYHF_REFERENCE_MAX_ATTEMPTS,
        help="Maximum bounded pyhf attempts with successively doubled signal references.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke-points", type=int, default=3)
    parser.add_argument("--smoke-max-events", type=int, default=250)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    feature_set = "resonance-hybrid-v1"
    for index, argument in enumerate(arguments):
        if argument == "--feature-set" and index + 1 < len(arguments):
            feature_set = arguments[index + 1]
        elif argument.startswith("--feature-set="):
            feature_set = argument.split("=", 1)[1]
    if feature_set == "fatjet-ak8-softdrop-v1":
        from resonance_fatjet_xgboost_analysis import main as fatjet_main

        return fatjet_main(arguments)
    args = build_parser().parse_args(arguments)
    if not math.isfinite(args.luminosity) or args.luminosity <= 0.0:
        raise SystemExit("--luminosity must be positive and finite")
    for name in (
        "hbb_branching_ratio",
        "eps_b",
        "eps_bb_conservative",
        "eps_c",
        "eps_light",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must lie in (0, 1]")
    if not math.isfinite(args.background_k_factor) or args.background_k_factor <= 0.0:
        raise SystemExit("--background-k-factor must be positive and finite")
    if args.min_background_raw < 0 or not math.isfinite(args.min_background_neff) or args.min_background_neff < 0.0:
        raise SystemExit("background bin thresholds must be non-negative")
    if not math.isfinite(args.pyhf_poi_hard_cap) or args.pyhf_poi_hard_cap <= 0.0:
        raise SystemExit("--pyhf-poi-hard-cap must be positive and finite")
    if (
        not math.isfinite(args.pyhf_physical_limit_hard_cap_fb)
        or args.pyhf_physical_limit_hard_cap_fb <= 0.0
    ):
        raise SystemExit("--pyhf-physical-limit-hard-cap-fb must be positive and finite")
    if args.pyhf_reference_max_attempts < 1:
        raise SystemExit("--pyhf-reference-max-attempts must be positive")
    if args.background_replicas < 1 or args.smoke_points < 1 or args.smoke_max_events < 1:
        raise SystemExit("replica and smoke counts must be positive")
    try:
        return run_analysis(args)
    except (AnalysisInputError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
