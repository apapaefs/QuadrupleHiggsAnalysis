#!/usr/bin/env python3
"""AK8-aware parameterized XGBoost resonance analysis with fast and pyhf stages."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np

import resonance_xgboost_analysis as resolved
from c3d4_xgboost_study import exact_cls_signal_upper_limit, poisson_median_observed


FEATURE_SET = "fatjet-ak8-softdrop-v1"
METHOD_VERSION = "resonance-fatjet-mass-aware-xgboost-v1"
PREPROCESSING_VERSION = "fatjet-ak8-preprocessing-v1"
SMEARING_MODEL_ID = "cms-energy-uniform-fourvector-v1"
N_FOLDS = 5
CATEGORY_NAMES = ("resolved", "mixed", "boosted")
LUMINOSITY_FB = 3000.0
HBB_BRANCHING_RATIO = 0.5824
SIGNAL_XSEC_FB = 1.0
BACKGROUND_NORM_UNCERTAINTY = 0.10
DEFAULT_SEED = 12345
FAST_WATERMARK = "FAST CUT-AND-COUNT VALIDATION — NOT FINAL PYHF"
TAGGING_SCENARIOS = {
    "nominal": {"eps_bb": 0.7225, "fake_bb": 0.10},
    "conservative": {"eps_bb": 0.30, "fake_bb": 0.01},
}

SCALAR_BRANCHES = (
    "event_index",
    "hypothesis_index",
    "weight",
    "raw_bjets",
    "accepted_bjets",
    "accepted_cjet_candidates",
    "accepted_lightjet_candidates",
    "n_ak8_eligible",
    "n_ak8_retained",
    "n_ak8_hh_diagnostic",
    "n_true_single",
    "n_c_mistag",
    "n_light_mistag",
    "n_true_fat_pass",
    "n_true_fat_fail",
    "n_fake_fat_pass",
    "n_fake_fat_fail",
    "n_merged",
    "category",
    "reco_jets_used",
    "n_configurations",
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
    "higgs_tag_mass": 4,
    "higgs_pt": 4,
    "higgs_y": 4,
    "higgs_type": 4,
    "higgs_constituent1": 4,
    "higgs_constituent2": 4,
    "higgs_constituent1_source": 4,
    "higgs_constituent2_source": 4,
    "fat_pt": 4,
    "fat_eta": 4,
    "fat_mass": 4,
    "fat_softdrop_mass": 4,
    "fat_tau21": 4,
    "fat_b_hadron_multiplicity": 4,
    "fat_c_hadron_multiplicity": 4,
    "fat_tag_kind": 4,
    "pair_mass": 6,
    "pair_dr": 6,
    "pair_dy": 6,
    "pair_dphi": 6,
}
BASE_SCALAR_FEATURES = (
    "n_ak8_retained",
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
    "higgs_tag_mass",
    "higgs_pt",
    "higgs_y",
    "higgs_type",
    "fat_pt",
    "fat_eta",
    "fat_mass",
    "fat_softdrop_mass",
    "fat_tau21",
    "pair_mass",
    "pair_dr",
    "pair_dy",
    "pair_dphi",
)
FORBIDDEN_FEATURES = frozenset(
    {
        "event_index",
        "hypothesis_index",
        "raw_bjets",
        "accepted_bjets",
        "accepted_cjet_candidates",
        "accepted_lightjet_candidates",
        "n_ak8_eligible",
        "n_ak8_hh_diagnostic",
        "n_true_single",
        "n_c_mistag",
        "n_light_mistag",
        "n_true_fat_pass",
        "n_true_fat_fail",
        "n_fake_fat_pass",
        "n_fake_fat_fail",
        "fat_b_hadron_multiplicity",
        "fat_c_hadron_multiplicity",
        "fat_tag_kind",
        "higgs_constituent1",
        "higgs_constituent2",
        "higgs_constituent1_source",
        "higgs_constituent2_source",
    }
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise resolved.AnalysisInputError(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(value, dict):
        raise resolved.AnalysisInputError(f"checkpoint {path} is not a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_input_fingerprint(spec: resolved.SampleSpec) -> dict[str, Any]:
    stat = spec.root_file.stat()
    return {
        "sample_id": spec.sample_id,
        "root_file": str(spec.root_file.resolve()),
        "root_size_bytes": int(stat.st_size),
        "root_mtime_ns": int(stat.st_mtime_ns),
        "summary_sha256": _sha256_file(spec.summary_file),
    }


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_arrays(
    path: Path, tree_name: str, branches: Sequence[str]
) -> dict[str, np.ndarray]:
    try:
        import uproot  # type: ignore

        with uproot.open(path) as root_file:
            if tree_name not in root_file:
                raise resolved.AnalysisInputError(f"{path}: missing {tree_name} tree")
            tree = root_file[tree_name]
            missing = [name for name in branches if name not in tree.keys()]
            if missing:
                raise resolved.AnalysisInputError(
                    f"{path}: missing branches {', '.join(missing)}"
                )
            arrays = tree.arrays(list(branches), library="np")
        return {name: np.asarray(arrays[name]) for name in branches}
    except ImportError:
        try:
            import ROOT  # type: ignore
        except ImportError as error:
            raise RuntimeError("reading AK8 feature files requires uproot or PyROOT") from error
        root_file = ROOT.TFile.Open(str(path))
        if not root_file or root_file.IsZombie():
            raise resolved.AnalysisInputError(f"cannot open {path}")
        tree = root_file.Get(tree_name)
        if not tree:
            root_file.Close()
            raise resolved.AnalysisInputError(f"{path}: missing {tree_name} tree")
        rows: dict[str, list[Any]] = {name: [] for name in branches}
        for entry in range(int(tree.GetEntries())):
            tree.GetEntry(entry)
            for name in branches:
                value = getattr(tree, name)
                width = ARRAY_BRANCH_WIDTHS.get(name)
                rows[name].append(
                    [value[index] for index in range(width)] if width else value
                )
        root_file.Close()
        return {name: np.asarray(values) for name, values in rows.items()}


def _summary_metadata(spec: resolved.SampleSpec) -> tuple[int, float, float, int, float, dict[str, Any]]:
    if not spec.summary_file.is_file():
        raise resolved.AnalysisInputError(f"missing extractor summary {spec.summary_file}")
    summary = _read_json(spec.summary_file)
    if summary.get("schema") != FEATURE_SET or summary.get("method_version") != FEATURE_SET:
        raise resolved.AnalysisInputError(f"{spec.summary_file}: incompatible AK8 schema")
    if summary.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise resolved.AnalysisInputError(f"{spec.summary_file}: preprocessing contract changed")
    if summary.get("tag_efficiencies_applied") is not False:
        raise resolved.AnalysisInputError(f"{spec.summary_file}: tag weights were pre-applied")
    smearing = summary.get("smearing", {})
    expected = {
        "enabled": True,
        "model_id": SMEARING_MODEL_ID,
        "fourvector_scaling": "uniform_correlated",
        "correlated_groomed_ungroomed_scaling": True,
        "gaussian_draws_per_physical_jet": 1,
    }
    bad = [key for key, value in expected.items() if smearing.get(key) != value]
    if bad:
        raise resolved.AnalysisInputError(
            f"{spec.summary_file}: incompatible smearing fields {', '.join(bad)}"
        )
    diagnostics = summary.get("diagnostics", {})
    for scenario in TAGGING_SCENARIOS:
        residual = float(
            diagnostics.get(f"max_pattern_probability_residual_{scenario}", math.inf)
        )
        if not math.isfinite(residual) or abs(residual) > 1.0e-12:
            raise resolved.AnalysisInputError(
                f"{spec.summary_file}: {scenario} AK8 probability closure failed"
            )
    try:
        input_counter = summary["input_counter"]
        reco_counter = summary["reconstructable_counter"]
        return (
            int(input_counter["events"]),
            float(input_counter["sumw"]),
            float(input_counter["sumw2"]),
            int(reco_counter["events"]),
            float(reco_counter["sumw"]),
            summary,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise resolved.AnalysisInputError(
            f"{spec.summary_file}: incomplete normalization counters"
        ) from error


def _root_feature_contract(path: Path) -> tuple[str, ...]:
    try:
        import uproot  # type: ignore

        with uproot.open(path) as root_file:
            if "feature_names_json" not in root_file:
                raise resolved.AnalysisInputError(f"{path}: missing feature metadata")
            title = str(root_file["feature_names_json"].member("fTitle"))
    except ImportError:
        import ROOT  # type: ignore

        root_file = ROOT.TFile.Open(str(path))
        obj = root_file.Get("feature_names_json") if root_file else None
        if not obj:
            if root_file:
                root_file.Close()
            raise resolved.AnalysisInputError(f"{path}: missing feature metadata")
        title = str(obj.GetTitle())
        root_file.Close()
    try:
        names = tuple(json.loads(title))
    except (TypeError, json.JSONDecodeError) as error:
        raise resolved.AnalysisInputError(f"{path}: malformed feature metadata") from error
    if FORBIDDEN_FEATURES.intersection(names):
        raise resolved.AnalysisInputError(f"{path}: audit fields leaked into model metadata")
    return names


def _validate_arrays(spec: resolved.SampleSpec, arrays: Mapping[str, np.ndarray]) -> None:
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise resolved.AnalysisInputError(f"{spec.sample_id}: inconsistent branch lengths")
    for name, width in ARRAY_BRANCH_WIDTHS.items():
        if arrays[name].ndim != 2 or arrays[name].shape[1] != width:
            raise resolved.AnalysisInputError(
                f"{spec.sample_id}: {name} must have shape (N,{width})"
            )
    for name in SCALAR_BRANCHES:
        if not np.all(np.isfinite(np.asarray(arrays[name], dtype=float))):
            raise resolved.AnalysisInputError(f"{spec.sample_id}: non-finite {name}")
    event = np.asarray(arrays["event_index"], dtype=np.int64)
    hypothesis = np.asarray(arrays["hypothesis_index"], dtype=np.int64)
    pairs = np.rec.fromarrays([event, hypothesis])
    if len(np.unique(pairs)) != len(event):
        raise resolved.AnalysisInputError(
            f"{spec.sample_id}: duplicate event/hypothesis identifiers"
        )
    retained = np.asarray(arrays["n_ak8_retained"], dtype=int)
    true_pass = np.asarray(arrays["n_true_fat_pass"], dtype=int)
    true_fail = np.asarray(arrays["n_true_fat_fail"], dtype=int)
    fake_pass = np.asarray(arrays["n_fake_fat_pass"], dtype=int)
    fake_fail = np.asarray(arrays["n_fake_fat_fail"], dtype=int)
    nmerged = np.asarray(arrays["n_merged"], dtype=int)
    category = np.asarray(arrays["category"], dtype=int)
    ak4 = (
        np.asarray(arrays["n_true_single"], dtype=int)
        + np.asarray(arrays["n_c_mistag"], dtype=int)
        + np.asarray(arrays["n_light_mistag"], dtype=int)
    )
    expected_category = np.where(nmerged == 0, 0, np.where(nmerged <= 2, 1, 2))
    if np.any(retained != true_pass + true_fail + fake_pass + fake_fail):
        raise resolved.AnalysisInputError(f"{spec.sample_id}: AK8 exponent closure failed")
    if np.any(nmerged != true_pass + fake_pass) or np.any(ak4 != 2 * (4 - nmerged)):
        raise resolved.AnalysisInputError(f"{spec.sample_id}: reconstruction/tag closure failed")
    if np.any((nmerged < 0) | (nmerged > 4)) or not np.array_equal(
        category, expected_category
    ):
        raise resolved.AnalysisInputError(f"{spec.sample_id}: category closure failed")
    order = np.argsort(event, kind="mergesort")
    sorted_event = event[order]
    sorted_weight = np.asarray(arrays["weight"], dtype=float)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_event)) + 1]
    ends = np.r_[starts[1:], len(order)]
    if any(
        not np.allclose(sorted_weight[start:end], sorted_weight[start], rtol=0.0, atol=0.0)
        for start, end in zip(starts, ends)
    ):
        raise resolved.AnalysisInputError(
            f"{spec.sample_id}: repeated hypotheses do not share one generator weight"
        )


def load_event_table(
    spec: resolved.SampleSpec,
    tree_name: str,
    allow_partial_input: bool,
) -> resolved.EventTable:
    if not spec.root_file.is_file():
        raise resolved.AnalysisInputError(f"missing feature file {spec.root_file}")
    branches = (*SCALAR_BRANCHES, *ARRAY_BRANCH_WIDTHS)
    arrays = _load_arrays(spec.root_file, tree_name, branches)
    model_names = _root_feature_contract(spec.root_file)
    input_events, input_sumw, input_sumw2, reco_events, reco_sumw, summary = _summary_metadata(spec)
    hypothesis_rows = int(summary.get("hypothesis_row_counter", {}).get("events", -1))
    if len(arrays["weight"]) != hypothesis_rows:
        raise resolved.AnalysisInputError(
            f"{spec.sample_id}: tree/summary hypothesis-row mismatch"
        )
    if not allow_partial_input and input_events != spec.generated_events_expected:
        raise resolved.AnalysisInputError(
            f"{spec.sample_id}: processed {input_events}, expected {spec.generated_events_expected}"
        )
    if input_events <= 0 or input_sumw == 0.0 or input_sumw2 < 0.0:
        raise resolved.AnalysisInputError(f"{spec.sample_id}: invalid source normalization")
    if int(summary.get("c_mistags", -1)) != spec.c_mistags or int(
        summary.get("light_mistags", -1)
    ) != spec.light_mistags:
        raise resolved.AnalysisInputError(f"{spec.sample_id}: manifest/extractor composition mismatch")
    _validate_arrays(spec, arrays)
    table = resolved.EventTable(
        arrays=dict(arrays),
        input_events=input_events,
        input_sumw=input_sumw,
        input_sumw2=input_sumw2,
        reconstructable_events=reco_events,
        reconstructable_sumw=reco_sumw,
        summary=summary,
    )
    table.summary["model_feature_names"] = list(model_names)
    return table


def base_feature_matrix(table: resolved.EventTable) -> tuple[np.ndarray, tuple[str, ...]]:
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
        if matrix.shape[1] == 6:
            labels = resolved.PAIR_LABELS
        else:
            labels = tuple(str(index + 1) for index in range(matrix.shape[1]))
        for index, label in enumerate(labels):
            values = matrix[:, index].copy()
            if name in {"jet_pt", "fat_pt", "fat_mass", "fat_softdrop_mass"}:
                values[values <= 0.0] = np.nan
            if name == "fat_tau21":
                values[values < 0.0] = np.nan
            columns.append(values)
            names.append(f"{name}_{label}")
    if FORBIDDEN_FEATURES.intersection(names):
        raise resolved.AnalysisInputError("audit-only fields entered the AK8 classifier")
    cpp_names = set(table.summary.get("model_feature_names", []))
    unknown = sorted(set(names).difference(cpp_names))
    if unknown:
        raise resolved.AnalysisInputError(f"Python/C++ feature contract differs: {unknown}")
    return np.column_stack(columns), tuple(names)


def grouped_folds(sample_id: str, event_indices: np.ndarray, seed: int) -> np.ndarray:
    events = np.asarray(event_indices, dtype=np.int64)
    unique = np.unique(events)
    unique_folds = resolved.deterministic_folds(
        np.full(len(unique), sample_id, dtype=object), unique, seed=seed
    )
    positions = np.searchsorted(unique, events)
    return unique_folds[positions]


def analysis_partition(sample_id: str, event_indices: np.ndarray, seed: int) -> np.ndarray:
    """Assign every generator event to a global validation/test partition."""

    events = np.asarray(event_indices, dtype=np.int64)
    result = np.empty(len(events), dtype=np.int8)
    for event in np.unique(events):
        digest = hashlib.sha256(f"{seed}\0{sample_id}\0{event}".encode()).digest()
        result[events == event] = int.from_bytes(digest[:8], "little") % 5
    return result


def tag_hypothesis_factor(
    table: resolved.EventTable,
    *,
    eps_bb: float,
    fake_bb: float,
    eps_b: float,
    eps_c: float,
    eps_light: float,
) -> np.ndarray:
    arrays = table.arrays
    factor = (
        eps_bb ** np.asarray(arrays["n_true_fat_pass"], dtype=int)
        * (1.0 - eps_bb) ** np.asarray(arrays["n_true_fat_fail"], dtype=int)
        * fake_bb ** np.asarray(arrays["n_fake_fat_pass"], dtype=int)
        * (1.0 - fake_bb) ** np.asarray(arrays["n_fake_fat_fail"], dtype=int)
        * eps_b ** np.asarray(arrays["n_true_single"], dtype=int)
        * eps_c ** np.asarray(arrays["n_c_mistag"], dtype=int)
        * eps_light ** np.asarray(arrays["n_light_mistag"], dtype=int)
    )
    if np.any(~np.isfinite(factor)) or np.any(factor < 0.0):
        raise resolved.AnalysisInputError("invalid analytic tag-hypothesis factor")
    return factor


def physical_weights(
    spec: resolved.SampleSpec,
    table: resolved.EventTable,
    scenario: Mapping[str, float],
    args: argparse.Namespace,
) -> np.ndarray:
    xsec = SIGNAL_XSEC_FB if spec.is_signal else float(spec.cross_section_fb)
    prefactor = (
        args.luminosity
        * xsec
        * spec.k_factor
        * spec.rate_factor
        * args.hbb_branching_ratio ** spec.hbb_power
        / table.input_sumw
    )
    return (
        np.asarray(table.arrays["weight"], dtype=float)
        * prefactor
        * tag_hypothesis_factor(
            table,
            eps_bb=float(scenario["eps_bb"]),
            fake_bb=float(scenario["fake_bb"]),
            eps_b=args.eps_b,
            eps_c=args.eps_c,
            eps_light=args.eps_light,
        )
    )


def load_sample(
    spec: resolved.SampleSpec,
    topology: str,
    args: argparse.Namespace,
) -> resolved.LoadedSample:
    table = load_event_table(spec, args.tree_name, args.mode == "smoke")
    base, names = base_feature_matrix(table)
    folds = grouped_folds(
        spec.sample_id, np.asarray(table.arrays["event_index"], dtype=np.int64), args.seed
    )
    scenario_weights = {
        name: physical_weights(spec, table, working_point, args)
        for name, working_point in args.tagging_scenarios.items()
    }
    return resolved.LoadedSample(spec, table, folds, base, names, scenario_weights)


def point_features(
    sample: resolved.LoadedSample, point: resolved.MassPoint
) -> tuple[np.ndarray, tuple[str, ...]]:
    if point.topology == "direct":
        return resolved.engineer_features(
            sample.base_features,
            sample.base_feature_names,
            sample.table,
            "direct",
            ms=float(point.ms),
        )
    return resolved.engineer_features(
        sample.base_features,
        sample.base_feature_names,
        sample.table,
        "cascade",
        m2=float(point.m2),
        m3=float(point.m3),
    )


def _event_point_assignments(
    sample: resolved.LoadedSample,
    points: Sequence[resolved.MassPoint],
    replicas: int,
    seed: int,
) -> np.ndarray:
    events = np.asarray(sample.table.arrays["event_index"], dtype=np.int64)
    unique = np.unique(events)
    rng = np.random.default_rng(resolved._stable_seed(sample.spec.sample_id, seed))
    unique_assignments = rng.integers(0, len(points), size=(len(unique), replicas))
    return unique_assignments[np.searchsorted(unique, events)]


def train_crossfit_models(
    topology: str,
    signals: Sequence[resolved.LoadedSample],
    backgrounds: Sequence[resolved.LoadedSample],
    points: Sequence[resolved.MassPoint],
    output_dir: Path,
    seed: int,
    replicas: int,
) -> tuple[list[Any], tuple[str, ...]]:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as error:
        raise RuntimeError("XGBoost is required for the AK8 analysis") from error
    signal_features: list[np.ndarray] = []
    signal_weights: list[np.ndarray] = []
    signal_folds: list[np.ndarray] = []
    signal_point_ids: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    for sample in signals:
        features, names = point_features(sample, sample.spec.point)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise resolved.AnalysisInputError("signal feature contracts differ")
        signal_features.append(features)
        signal_weights.append(np.abs(sample.scenario_weights["nominal"]))
        signal_folds.append(sample.folds)
        signal_point_ids.append(
            np.full(sample.table.entries, sample.spec.point.point_id, dtype=object)
        )
    sx = np.concatenate(signal_features)
    sw = np.concatenate(signal_weights)
    sf = np.concatenate(signal_folds)
    sp = np.concatenate(signal_point_ids)
    assignments = {
        sample.spec.sample_id: _event_point_assignments(sample, points, replicas, seed)
        for sample in backgrounds
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    models: list[Any] = []
    for rotation in range(N_FOLDS):
        signal_train = (sf != rotation) & (sf != (rotation + 1) % N_FOLDS)
        sx_train = sx[signal_train]
        sw_train = resolved._equal_signal_point_weights(sw, sp, signal_train)[signal_train]
        bx_parts: list[np.ndarray] = []
        bw_parts: list[np.ndarray] = []
        for sample in backgrounds:
            train_mask = (sample.folds != rotation) & (
                sample.folds != (rotation + 1) % N_FOLDS
            )
            for replica in range(replicas):
                indices = assignments[sample.spec.sample_id][:, replica]
                assigned = [points[index] for index in indices]
                if topology == "direct":
                    features, names = resolved.engineer_features(
                        sample.base_features,
                        sample.base_feature_names,
                        sample.table,
                        topology,
                        ms=np.asarray([point.ms for point in assigned], dtype=float),
                    )
                else:
                    features, names = resolved.engineer_features(
                        sample.base_features,
                        sample.base_feature_names,
                        sample.table,
                        topology,
                        m2=np.asarray([point.m2 for point in assigned], dtype=float),
                        m3=np.asarray([point.m3 for point in assigned], dtype=float),
                    )
                if names != feature_names:
                    raise resolved.AnalysisInputError("background feature contract differs")
                bx_parts.append(features[train_mask])
                bw_parts.append(
                    np.abs(sample.scenario_weights["nominal"][train_mask]) / replicas
                )
        bx_train = np.concatenate(bx_parts)
        bw_train = np.concatenate(bw_parts)
        if not len(sx_train) or not len(bx_train) or np.sum(bw_train) <= 0.0:
            raise resolved.AnalysisInputError("empty classifier training class")
        common_total = 0.5 * float(len(sx_train) + len(bx_train))
        sw_train *= common_total
        bw_train *= common_total / float(np.sum(bw_train))
        x_train = np.concatenate([sx_train, bx_train])
        labels = np.concatenate(
            [np.ones(len(sx_train), dtype=np.int8), np.zeros(len(bx_train), dtype=np.int8)]
        )
        weights = np.concatenate([sw_train, bw_train])
        params = dict(resolved.FIXED_XGBOOST_PARAMS)
        params.update(random_state=seed + rotation, n_jobs=1)
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, labels, sample_weight=weights, verbose=False)
        final_path = output_dir / f"{topology}_fold{rotation}.json"
        temporary = output_dir / f".{topology}_fold{rotation}.tmp-{os.getpid()}.json"
        model.save_model(str(temporary))
        os.replace(temporary, final_path)
        models.append(model)
    return models, tuple(feature_names or ())


def load_or_train_models(
    topology: str,
    signals: Sequence[resolved.LoadedSample],
    backgrounds: Sequence[resolved.LoadedSample],
    points: Sequence[resolved.MassPoint],
    model_dir: Path,
    seed: int,
    replicas: int,
    input_fingerprint: str,
) -> tuple[list[Any], tuple[str, ...], dict[str, str], bool]:
    import xgboost as xgb  # type: ignore

    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "model_manifest.json"
    paths = [model_dir / f"{topology}_fold{fold}.json" for fold in range(N_FOLDS)]
    expected_names = point_features(signals[0], signals[0].spec.point)[1]
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("input_fingerprint") != input_fingerprint:
            raise resolved.AnalysisInputError(
                f"{model_dir}: stale model fingerprint; refusing checkpoint reuse"
            )
        if tuple(manifest.get("feature_names", [])) != expected_names:
            raise resolved.AnalysisInputError(f"{model_dir}: model feature contract changed")
        models: list[Any] = []
        hashes: dict[str, str] = {}
        for fold, path in enumerate(paths):
            if not path.is_file():
                raise resolved.AnalysisInputError(f"missing model checkpoint {path}")
            digest = _sha256_file(path)
            if manifest.get("model_sha256", {}).get(path.name) != digest:
                raise resolved.AnalysisInputError(f"model hash mismatch for {path}")
            params = dict(resolved.FIXED_XGBOOST_PARAMS)
            params.update(random_state=seed + fold, n_jobs=1)
            model = xgb.XGBClassifier(**params)
            model.load_model(str(path))
            model.set_params(n_jobs=1)
            models.append(model)
            hashes[path.name] = digest
        return models, expected_names, hashes, True
    if any(path.exists() for path in paths):
        raise resolved.AnalysisInputError(f"{model_dir}: unmanifested model checkpoint exists")
    models, names = train_crossfit_models(
        topology, signals, backgrounds, points, model_dir, seed, replicas
    )
    hashes = {path.name: _sha256_file(path) for path in paths}
    _atomic_json(
        manifest_path,
        {
            "input_fingerprint": input_fingerprint,
            "feature_set": FEATURE_SET,
            "topology": topology,
            "folds": N_FOLDS,
            "seed": seed,
            "feature_names": names,
            "fixed_xgboost_parameters": resolved.FIXED_XGBOOST_PARAMS,
            "model_sha256": hashes,
        },
    )
    return models, names, hashes, False


def predict_crossfit(
    sample: resolved.LoadedSample,
    point: resolved.MassPoint,
    models: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    features, _ = point_features(sample, point)
    test = np.full(sample.table.entries, np.nan)
    validation = np.full(sample.table.entries, np.nan)
    for rotation, model in enumerate(models):
        test_mask = sample.folds == rotation
        validation_mask = sample.folds == (rotation + 1) % N_FOLDS
        positions = np.flatnonzero(test_mask | validation_mask)
        prediction = np.asarray(model.predict_proba(features[positions]))[:, 1]
        local_test = test_mask[positions]
        test[positions[local_test]] = prediction[local_test]
        validation[positions[~local_test]] = prediction[~local_test]
    if not np.all(np.isfinite(test)) or not np.all(np.isfinite(validation)):
        raise resolved.AnalysisInputError(f"{sample.spec.sample_id}: incomplete score cache")
    return test, validation


def _score_cache_path(base: Path, point_id: str, sample_id: str) -> Path:
    safe = sample_id.replace("/", "_")
    return base / point_id / f"{safe}.npz"


def load_or_predict_scores(
    sample: resolved.LoadedSample,
    point: resolved.MassPoint,
    models: Sequence[Any],
    cache_base: Path,
    core_fingerprint: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = _score_cache_path(cache_base, point.point_id, sample.spec.sample_id)
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            fingerprint = str(np.asarray(payload["core_fingerprint"]).item())
            test = np.asarray(payload["test"], dtype=float)
            validation = np.asarray(payload["validation"], dtype=float)
        if fingerprint != core_fingerprint or len(test) != sample.table.entries:
            raise resolved.AnalysisInputError(f"{path}: stale score-cache fingerprint")
        return test, validation
    test, validation = predict_crossfit(sample, point, models)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        core_fingerprint=np.asarray(core_fingerprint),
        test=test,
        validation=validation,
    )
    os.replace(temporary, path)
    return test, validation


def grouped_binned_summary(
    scores: np.ndarray,
    weights: np.ndarray,
    event_indices: np.ndarray,
    edges: Sequence[float],
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    events = np.asarray(event_indices, dtype=np.int64)
    selected = np.ones(len(scores), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    edge_array = np.asarray(edges, dtype=float)
    n_bins = len(edge_array) - 1
    if not np.any(selected):
        return {
            "yield": np.zeros(n_bins),
            "sumw2": np.zeros(n_bins),
            "raw": np.zeros(n_bins, dtype=int),
            "neff": np.zeros(n_bins),
        }
    bins = np.searchsorted(edge_array[1:-1], scores[selected], side="right")
    group_event = events[selected]
    group_weight = weights[selected]
    order = np.lexsort((bins, group_event))
    group_event = group_event[order]
    bins = bins[order]
    group_weight = group_weight[order]
    starts = np.r_[0, np.flatnonzero((np.diff(group_event) != 0) | (np.diff(bins) != 0)) + 1]
    grouped_weight = np.add.reduceat(group_weight, starts)
    grouped_bins = bins[starts]
    yields = np.bincount(grouped_bins, weights=grouped_weight, minlength=n_bins).astype(float)
    sumw2 = np.bincount(
        grouped_bins, weights=grouped_weight**2, minlength=n_bins
    ).astype(float)
    raw = np.bincount(grouped_bins, minlength=n_bins).astype(int)
    neff = np.divide(yields**2, sumw2, out=np.zeros_like(yields), where=sumw2 > 0.0)
    return {"yield": yields, "sumw2": sumw2, "raw": raw, "neff": neff}


def grouped_threshold_scan(
    scores: np.ndarray,
    weights: np.ndarray,
    event_indices: np.ndarray,
    thresholds: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    events = np.asarray(event_indices, dtype=np.int64)
    selected = np.ones(len(scores), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    scores = scores[selected]
    weights = weights[selected]
    events = events[selected]
    order = np.argsort(scores, kind="mergesort")[::-1]
    scores = scores[order]
    weights = weights[order]
    events = events[order]
    unique, inverse = np.unique(events, return_inverse=True)
    event_weights = np.zeros(len(unique), dtype=float)
    active_rows = np.zeros(len(unique), dtype=np.int32)
    descending = np.argsort(thresholds, kind="mergesort")[::-1]
    output_yield = np.zeros(len(thresholds))
    output_sumw2 = np.zeros(len(thresholds))
    output_raw = np.zeros(len(thresholds), dtype=int)
    total = 0.0
    total_square = 0.0
    raw = 0
    position = 0
    for output_index in descending:
        threshold = thresholds[output_index]
        while position < len(scores) and scores[position] >= threshold:
            group = inverse[position]
            old = event_weights[group]
            value = weights[position]
            event_weights[group] = old + value
            total += value
            total_square += 2.0 * old * value + value * value
            if active_rows[group] == 0:
                raw += 1
            active_rows[group] += 1
            position += 1
        output_yield[output_index] = total
        output_sumw2[output_index] = max(0.0, total_square)
        output_raw[output_index] = raw
    neff = np.divide(
        output_yield**2,
        output_sumw2,
        out=np.zeros_like(output_yield),
        where=output_sumw2 > 0.0,
    )
    return {"yield": output_yield, "sumw2": output_sumw2, "raw": output_raw, "neff": neff}


def combine_summaries(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("cannot combine an empty summary list")
    return {
        name: np.sum([np.asarray(part[name]) for part in parts], axis=0)
        for name in ("yield", "sumw2", "raw")
    } | {
        "neff": np.divide(
            np.sum([np.asarray(part["yield"]) for part in parts], axis=0) ** 2,
            np.sum([np.asarray(part["sumw2"]) for part in parts], axis=0),
            out=np.zeros_like(np.asarray(parts[0]["yield"], dtype=float)),
            where=np.sum([np.asarray(part["sumw2"]) for part in parts], axis=0) > 0.0,
        )
    }


def _candidate_edges(scores: np.ndarray, weights: np.ndarray) -> list[list[float]]:
    scores = np.asarray(scores, dtype=float)
    weights = np.abs(np.asarray(weights, dtype=float))
    if not len(scores) or np.sum(weights) <= 0.0:
        return []
    order = np.argsort(scores, kind="mergesort")
    data = scores[order]
    weight = weights[order]
    positions = (np.cumsum(weight) - 0.5 * weight) / float(np.sum(weight))
    quantiles = np.interp(
        np.asarray([0.0, 0.50, 0.75, 0.90, 0.97, 1.0]),
        positions,
        data,
        left=data[0],
        right=data[-1],
    )
    candidates: list[list[float]] = []
    for indices in ((0, 1, 2, 3, 4, 5), (0, 2, 3, 4, 5), (0, 2, 4, 5), (0, 3, 5)):
        edges = [float(quantiles[index]) for index in indices]
        edges[0] = min(edges[0], 0.0)
        edges[-1] = max(edges[-1], 1.0)
        if all(right > left for left, right in zip(edges, edges[1:])):
            candidates.append(edges)
    return candidates


def _select_template_edges(
    background_scores: Sequence[np.ndarray],
    backgrounds: Sequence[resolved.LoadedSample],
    masks: Sequence[np.ndarray],
    min_raw: int,
    min_neff: float,
) -> tuple[list[float] | None, dict[str, Any]]:
    nominal_scores = np.concatenate(
        [scores[mask] for scores, mask in zip(background_scores, masks)]
    )
    nominal_weights = np.concatenate(
        [
            sample.scenario_weights["nominal"][mask]
            for sample, mask in zip(backgrounds, masks)
        ]
    )
    audit: list[dict[str, Any]] = []
    for edges in _candidate_edges(nominal_scores, nominal_weights):
        scenario_audit: dict[str, Any] = {}
        valid = True
        for scenario in TAGGING_SCENARIOS:
            summaries = [
                grouped_binned_summary(
                    scores,
                    sample.scenario_weights[scenario],
                    np.asarray(sample.table.arrays["event_index"], dtype=np.int64),
                    edges,
                    mask,
                )
                for sample, scores, mask in zip(backgrounds, background_scores, masks)
            ]
            total = combine_summaries(summaries)
            scenario_valid = bool(
                np.all(total["yield"] > 0.0)
                and np.all(total["raw"] >= min_raw)
                and np.all(total["neff"] >= min_neff)
            )
            scenario_audit[scenario] = {
                "yield": total["yield"],
                "raw_unique_events": total["raw"],
                "neff": total["neff"],
                "valid": scenario_valid,
            }
            valid = valid and scenario_valid
        audit.append({"edges": edges, "scenarios": scenario_audit, "valid": valid})
        if valid:
            return edges, {"status": "ok", "candidates": audit}
    return None, {"status": "invalid", "candidates": audit}


def _sample_category_rows(
    topology: str,
    point: resolved.MassPoint,
    samples: Sequence[resolved.LoadedSample],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        events = np.asarray(sample.table.arrays["event_index"], dtype=np.int64)
        categories = np.asarray(sample.table.arrays["category"], dtype=int)
        for scenario in TAGGING_SCENARIOS:
            for index, category in enumerate(CATEGORY_NAMES):
                summary = grouped_binned_summary(
                    np.full(sample.table.entries, 0.5),
                    sample.scenario_weights[scenario],
                    events,
                    [0.0, 1.0],
                    categories == index,
                )
                rows.append(
                    {
                        "topology": topology,
                        **point.as_dict(),
                        "sample_id": sample.spec.sample_id,
                        "role": sample.spec.role,
                        "tagging_scenario": scenario,
                        "category": category,
                        "input_cross_section_fb": (
                            SIGNAL_XSEC_FB
                            if sample.spec.is_signal
                            else sample.spec.cross_section_fb
                        ),
                        "yield_3000fb": float(summary["yield"][0]),
                        "sumw2_grouped": float(summary["sumw2"][0]),
                        "raw_unique_events": int(summary["raw"][0]),
                        "neff": float(summary["neff"][0]),
                    }
                )
    return rows


def _fast_threshold_result(
    signal: resolved.LoadedSample,
    backgrounds: Sequence[resolved.LoadedSample],
    signal_scores: np.ndarray,
    background_scores: Sequence[np.ndarray],
    scenario: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    thresholds = np.linspace(0.0, 1.0, 1001)
    signal_events = np.asarray(signal.table.arrays["event_index"], dtype=np.int64)
    signal_partition = analysis_partition(signal.spec.sample_id, signal_events, args.seed)
    validation_mask = signal_partition < 2
    test_mask = signal_partition >= 2
    signal_validation = grouped_threshold_scan(
        signal_scores,
        signal.scenario_weights[scenario] / 0.4,
        signal_events,
        thresholds,
        validation_mask,
    )
    background_validation_parts: list[dict[str, np.ndarray]] = []
    for sample, scores in zip(backgrounds, background_scores):
        events = np.asarray(sample.table.arrays["event_index"], dtype=np.int64)
        partition = analysis_partition(sample.spec.sample_id, events, args.seed)
        background_validation_parts.append(
            grouped_threshold_scan(
                scores,
                sample.scenario_weights[scenario] / 0.4,
                events,
                thresholds,
                partition < 2,
            )
        )
    background_validation = combine_summaries(background_validation_parts)
    valid = (
        (background_validation["raw"] >= args.min_background_raw)
        & (background_validation["neff"] >= args.min_background_neff)
        & (background_validation["yield"] >= 0.0)
        & (signal_validation["yield"] > 0.0)
    )
    expected = np.full(len(thresholds), np.inf)
    for index in np.flatnonzero(valid):
        signal_limit = exact_cls_signal_upper_limit(
            float(background_validation["yield"][index]), 0.95
        )
        expected[index] = signal_limit / float(signal_validation["yield"][index])
    if not np.any(np.isfinite(expected)):
        return {
            "status": "invalid",
            "reason": "no threshold satisfies unique-background and Neff requirements",
        }
    minimum = float(np.nanmin(expected))
    candidates = np.flatnonzero(np.isclose(expected, minimum, rtol=0.0, atol=1.0e-14))
    chosen = int(candidates[-1])
    threshold = float(thresholds[chosen])
    signal_test = grouped_threshold_scan(
        signal_scores,
        signal.scenario_weights[scenario] / 0.6,
        signal_events,
        np.asarray([threshold]),
        test_mask,
    )
    background_test_parts: list[dict[str, np.ndarray]] = []
    for sample, scores in zip(backgrounds, background_scores):
        events = np.asarray(sample.table.arrays["event_index"], dtype=np.int64)
        partition = analysis_partition(sample.spec.sample_id, events, args.seed)
        background_test_parts.append(
            grouped_threshold_scan(
                scores,
                sample.scenario_weights[scenario] / 0.6,
                events,
                np.asarray([threshold]),
                partition >= 2,
            )
        )
    background_test = combine_summaries(background_test_parts)
    signal_yield = float(signal_test["yield"][0])
    background_yield = float(background_test["yield"][0])
    if signal_yield <= 0.0 or background_yield < 0.0:
        return {"status": "invalid", "reason": "non-positive disjoint-test yield"}
    observed = poisson_median_observed(background_yield)
    s95 = exact_cls_signal_upper_limit(background_yield, 0.95, observed)
    return {
        "status": "ok",
        "threshold": threshold,
        "validation_expected_limit_fb": minimum,
        "validation_background_yield": float(background_validation["yield"][chosen]),
        "validation_background_raw_unique": int(background_validation["raw"][chosen]),
        "validation_background_neff": float(background_validation["neff"][chosen]),
        "test_signal_yield_at_1fb": signal_yield,
        "test_background_yield": background_yield,
        "test_background_sumw2_grouped": float(background_test["sumw2"][0]),
        "test_background_raw_unique": int(background_test["raw"][0]),
        "test_background_neff": float(background_test["neff"][0]),
        "median_background_only_observation": observed,
        "exact_cls_signal_event_limit": s95,
        "expected_median_limit_fb": s95 / signal_yield,
        "validation_partition_fraction": 0.4,
        "test_partition_fraction": 0.6,
        "partitions_disjoint_by_generator_event": True,
    }


_POINT_CONTEXT: dict[str, Any] = {}


def _build_point_template(index: int) -> tuple[int, str]:
    context = _POINT_CONTEXT
    args: argparse.Namespace = context["args"]
    signals: Sequence[resolved.LoadedSample] = context["signals"]
    backgrounds: Sequence[resolved.LoadedSample] = context["backgrounds"]
    models: Sequence[Any] = context["models"]
    core_fingerprint: str = context["core_fingerprint"]
    template_dir: Path = context["template_dir"]
    fast_dir: Path = context["fast_dir"]
    score_dir: Path = context["score_dir"]
    signal = signals[index]
    point = signal.spec.point
    path = template_dir / f"{point.point_id}.json"
    fast_path = fast_dir / f"{point.point_id}.json"
    if path.exists() and fast_path.exists():
        for checkpoint in (path, fast_path):
            payload = _read_json(checkpoint)
            if payload.get("core_fingerprint") != core_fingerprint:
                raise resolved.AnalysisInputError(
                    f"{checkpoint}: stale checkpoint fingerprint"
                )
        return index, "kept_existing"
    for checkpoint in (path, fast_path):
        if checkpoint.exists() and _read_json(checkpoint).get(
            "core_fingerprint"
        ) != core_fingerprint:
            raise resolved.AnalysisInputError(
                f"{checkpoint}: stale checkpoint fingerprint"
            )
    signal_test, signal_validation = load_or_predict_scores(
        signal, point, models, score_dir, core_fingerprint
    )
    background_test: list[np.ndarray] = []
    background_validation: list[np.ndarray] = []
    for background in backgrounds:
        test, validation = load_or_predict_scores(
            background, point, models, score_dir, core_fingerprint
        )
        background_test.append(test)
        background_validation.append(validation)
    fast_limits = {
        scenario: _fast_threshold_result(
            signal,
            backgrounds,
            signal_test,
            background_test,
            scenario,
            args,
        )
        for scenario in TAGGING_SCENARIOS
    }
    category_rows = _sample_category_rows(
        args.topology, point, [signal, *backgrounds]
    )
    channels: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in TAGGING_SCENARIOS
    }
    binning_audit: dict[str, Any] = {}
    signal_categories = np.asarray(signal.table.arrays["category"], dtype=int)
    for category, category_name in enumerate(CATEGORY_NAMES):
        category_audit: dict[str, Any] = {}
        channel_starts = {
            scenario: len(channels[scenario]) for scenario in TAGGING_SCENARIOS
        }
        valid_fold_count = 0
        for fold in range(N_FOLDS):
            validation_fold = (fold + 1) % N_FOLDS
            masks = [
                (np.asarray(sample.table.arrays["category"], dtype=int) == category)
                & (sample.folds == validation_fold)
                for sample in backgrounds
            ]
            edges, audit = _select_template_edges(
                background_validation,
                backgrounds,
                masks,
                args.min_background_raw,
                args.min_background_neff,
            )
            category_audit[str(fold)] = {
                "validation_fold": validation_fold,
                "test_fold": fold,
                "validation_test_disjoint": True,
                **audit,
            }
            if edges is None:
                continue
            valid_fold_count += 1
            signal_mask = (signal_categories == category) & (signal.folds == fold)
            for scenario in TAGGING_SCENARIOS:
                signal_summary = grouped_binned_summary(
                    signal_test,
                    signal.scenario_weights[scenario],
                    np.asarray(signal.table.arrays["event_index"], dtype=np.int64),
                    edges,
                    signal_mask,
                )
                component_summaries = []
                for sample, scores in zip(backgrounds, background_test):
                    mask = (
                        np.asarray(sample.table.arrays["category"], dtype=int) == category
                    ) & (sample.folds == fold)
                    component_summaries.append(
                        grouped_binned_summary(
                            scores,
                            sample.scenario_weights[scenario],
                            np.asarray(sample.table.arrays["event_index"], dtype=np.int64),
                            edges,
                            mask,
                        )
                    )
                total = combine_summaries(component_summaries)
                channels[scenario].append(
                    {
                        "name": f"{category_name}_fold{fold}",
                        "category": category_name,
                        "fold": fold,
                        "edges": edges,
                        "signal": signal_summary["yield"],
                        "signal_sumw2": signal_summary["sumw2"],
                        "signal_raw_unique": signal_summary["raw"],
                        "background": total["yield"],
                        "background_sumw2": total["sumw2"],
                        "background_raw_unique": total["raw"],
                        "background_neff": total["neff"],
                    }
                )
        category_audit["status"] = "ok" if valid_fold_count == N_FOLDS else "invalid"
        category_audit["all_rotations_valid"] = valid_fold_count == N_FOLDS
        if valid_fold_count != N_FOLDS:
            for scenario in TAGGING_SCENARIOS:
                del channels[scenario][channel_starts[scenario] :]
        binning_audit[category_name] = category_audit
    _atomic_json(
        path,
        {
            "core_fingerprint": core_fingerprint,
            "feature_set": FEATURE_SET,
            "point": point.as_dict(),
            "category_yields": category_rows,
            "channels": channels,
            "binning_audit": binning_audit,
            "source_normalization": "generator input sumw, never hypothesis-row sumw",
            "mc_statistics": "hypotheses grouped by source event within each bin",
        },
    )
    _atomic_json(
        fast_path,
        {
            "core_fingerprint": core_fingerprint,
            "feature_set": FEATURE_SET,
            "point": point.as_dict(),
            "fast_limits": fast_limits,
            "watermark": FAST_WATERMARK,
        },
    )
    return index, "complete"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows([{key: _json_safe(value) for key, value in row.items()} for row in rows])
    os.replace(temporary, path)


def _input_cross_section_rows(
    samples: Sequence[resolved.LoadedSample],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        spec = sample.spec
        rows.append(
            {
                "sample_id": spec.sample_id,
                "role": spec.role,
                "topology": spec.point.topology if spec.point else "background",
                "MS_GeV": spec.point.ms if spec.point else None,
                "M2_GeV": spec.point.m2 if spec.point else None,
                "M3_GeV": spec.point.m3 if spec.point else None,
                "analysis_cross_section_fb": (
                    SIGNAL_XSEC_FB if spec.is_signal else spec.cross_section_fb
                ),
                "generator_cross_section_fb": spec.generated_cross_section_fb,
                "cross_section_source": spec.cross_section_source,
                "k_factor": spec.k_factor,
                "rate_factor": spec.rate_factor,
                "hbb_power": spec.hbb_power,
                "source_events": sample.table.input_events,
                "source_generator_sumw": sample.table.input_sumw,
                "hypothesis_rows": sample.table.entries,
                "normalization_denominator": "source_generator_sumw",
            }
        )
    return rows


def _collate_fast(
    output_dir: Path,
    signals: Sequence[resolved.LoadedSample],
    template_dir: Path,
    fast_dir: Path,
    core_fingerprint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    limit_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for signal in signals:
        point = signal.spec.point
        template = _read_json(template_dir / f"{point.point_id}.json")
        fast = _read_json(fast_dir / f"{point.point_id}.json")
        if template.get("core_fingerprint") != core_fingerprint or fast.get(
            "core_fingerprint"
        ) != core_fingerprint:
            raise resolved.AnalysisInputError(f"{point.point_id}: stale collation input")
        category_rows.extend(template.get("category_yields", []))
        audits[point.point_id] = template.get("binning_audit", {})
        for scenario in TAGGING_SCENARIOS:
            result = dict(fast.get("fast_limits", {}).get(scenario, {}))
            limit_rows.append(
                {
                    "topology": point.topology,
                    **point.as_dict(),
                    "tagging_scenario": scenario,
                    "eps_bb": TAGGING_SCENARIOS[scenario]["eps_bb"],
                    "fake_bb": TAGGING_SCENARIOS[scenario]["fake_bb"],
                    **result,
                    "label": FAST_WATERMARK,
                }
            )
    _write_csv(output_dir / "fast_point_limits.csv", limit_rows)
    _atomic_json(output_dir / "fast_point_limits.json", limit_rows)
    _write_csv(output_dir / "point_category_yields.csv", category_rows)
    _atomic_json(output_dir / "point_category_yields.json", category_rows)
    _atomic_json(output_dir / "binning_audit.json", audits)
    return limit_rows, category_rows


def _write_limit_plot(
    output_dir: Path,
    topology: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    fast: bool,
) -> list[str]:
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return []
    outputs: list[str] = []
    if topology == "direct":
        figure, axis = plt.subplots(figsize=(7.2, 5.2))
        for scenario in TAGGING_SCENARIOS:
            selected = sorted(
                [row for row in valid if row["tagging_scenario"] == scenario],
                key=lambda row: float(row["MS_GeV"]),
            )
            if selected:
                axis.plot(
                    [float(row["MS_GeV"]) for row in selected],
                    [float(row["expected_median_limit_fb"]) for row in selected],
                    marker="o",
                    markersize=3,
                    label=scenario,
                )
        axis.set_xlabel(r"$M_S$ [GeV]")
        axis.set_ylabel(r"expected 95% CL upper limit [fb]")
        axis.set_yscale("log")
        axis.legend()
    else:
        figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
        for axis, scenario in zip(axes, TAGGING_SCENARIOS):
            selected = [row for row in valid if row["tagging_scenario"] == scenario]
            x = np.asarray([float(row["M2_GeV"]) for row in selected])
            y = np.asarray([float(row["M3_GeV"]) for row in selected])
            z = np.asarray([float(row["expected_median_limit_fb"]) for row in selected])
            if len(selected) >= 3:
                try:
                    contour = axis.tricontourf(x, y, z, levels=20)
                    figure.colorbar(contour, ax=axis, label="expected limit [fb]")
                except (ValueError, RuntimeError):
                    axis.scatter(x, y, c=z)
            else:
                axis.scatter(x, y, c=z)
            axis.set_title(scenario)
            axis.set_xlabel(r"$M_2$ [GeV]")
            axis.set_ylabel(r"$M_3$ [GeV]")
    title = FAST_WATERMARK if fast else "Expected pyhf 95% CL exclusion"
    figure.suptitle(title, fontsize=11, color="darkred" if fast else "black")
    stem = "fast_validation_limits" if fast else "pyhf_expected_limits"
    for extension in ("pdf", "png"):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def _pyhf_worker(task: tuple[str, str, str, str]) -> tuple[str, str]:
    template_text, output_text, scenario, pyhf_fingerprint = task
    template_path = Path(template_text)
    output_path = Path(output_text)
    if output_path.exists():
        payload = _read_json(output_path)
        if payload.get("pyhf_fingerprint") != pyhf_fingerprint:
            raise resolved.AnalysisInputError(f"{output_path}: stale pyhf fingerprint")
        return output_path.stem, "kept_existing"
    template = _read_json(template_path)
    channels = []
    for channel in template.get("channels", {}).get(scenario, []):
        channels.append(
            {
                "name": channel["name"],
                "signal": channel["signal"],
                "background": channel["background"],
                "signal_staterror": np.sqrt(np.asarray(channel["signal_sumw2"], dtype=float)),
                "background_staterror": np.sqrt(
                    np.asarray(channel["background_sumw2"], dtype=float)
                ),
            }
        )
    fit = (
        resolved._pyhf_limit(channels, BACKGROUND_NORM_UNCERTAINTY)
        if channels
        else {"status": "invalid", "reason": "no complete category-by-fold templates"}
    )
    _atomic_json(
        output_path,
        {
            "core_fingerprint": template["core_fingerprint"],
            "pyhf_fingerprint": pyhf_fingerprint,
            "point": template["point"],
            "tagging_scenario": scenario,
            "n_channels": len(channels),
            **fit,
        },
    )
    return output_path.stem, "complete"


def _run_pyhf_stage(
    args: argparse.Namespace,
    signals: Sequence[resolved.LoadedSample],
    template_dir: Path,
    pyhf_dir: Path,
    core_fingerprint: str,
) -> tuple[list[dict[str, Any]], str]:
    try:
        pyhf_version = importlib.metadata.version("pyhf")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("full mode requires pyhf; fast mode does not") from error
    pyhf_fingerprint = _fingerprint(
        {
            "core_fingerprint": core_fingerprint,
            "pyhf_version": pyhf_version,
            "background_norm_uncertainty": BACKGROUND_NORM_UNCERTAINTY,
            "likelihood": "multi-bin category-by-crossfit-fold",
        }
    )
    tasks: list[tuple[str, str, str, str]] = []
    for signal in signals:
        point_id = signal.spec.point.point_id
        template = template_dir / f"{point_id}.json"
        if not template.is_file():
            raise resolved.AnalysisInputError(
                f"missing verified fast template {template}; run --mode fast first"
            )
        if _read_json(template).get("core_fingerprint") != core_fingerprint:
            raise resolved.AnalysisInputError(f"{template}: stale template fingerprint")
        for scenario in TAGGING_SCENARIOS:
            tasks.append(
                (
                    str(template),
                    str(pyhf_dir / f"{point_id}__{scenario}.json"),
                    scenario,
                    pyhf_fingerprint,
                )
            )
    if args.pyhf_jobs == 1:
        for task in tasks:
            _pyhf_worker(task)
    else:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.pyhf_jobs, mp_context=context) as pool:
            futures = [pool.submit(_pyhf_worker, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
    rows: list[dict[str, Any]] = []
    for signal in signals:
        point = signal.spec.point
        for scenario in TAGGING_SCENARIOS:
            payload = _read_json(pyhf_dir / f"{point.point_id}__{scenario}.json")
            rows.append(
                {
                    "topology": point.topology,
                    **point.as_dict(),
                    "tagging_scenario": scenario,
                    "eps_bb": TAGGING_SCENARIOS[scenario]["eps_bb"],
                    "fake_bb": TAGGING_SCENARIOS[scenario]["fake_bb"],
                    **{
                        key: value
                        for key, value in payload.items()
                        if key not in {"point", "core_fingerprint", "pyhf_fingerprint"}
                    },
                }
            )
    return rows, pyhf_fingerprint


def run_analysis(args: argparse.Namespace) -> int:
    root = args.analysis_root.expanduser().resolve()
    signal_manifest = _resolve(root, args.signal_manifest)
    background_manifest = _resolve(root, args.background_manifest)
    signal_root_dir = _resolve(root, args.signal_root_dir)
    output_dir = _resolve(root, args.output_dir) if args.output_dir else (
        root / "ResonanceAnalysis/results/ak8-v1" / args.topology
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    model_dir = checkpoints / "models"
    score_dir = checkpoints / "scores"
    template_dir = checkpoints / "templates"
    fast_dir = checkpoints / "fast"
    pyhf_dir = checkpoints / "pyhf"

    smoke_points = args.smoke_points if args.mode == "smoke" else None
    signal_specs = resolved.load_signal_specs(
        signal_manifest,
        args.topology,
        signal_root_dir,
        args.signal_root_pattern,
        smoke_points=smoke_points,
    )
    background_specs, missing_optional = resolved.load_background_specs(
        background_manifest, root, args.background_k_factor
    )
    signals = [load_sample(spec, args.topology, args) for spec in signal_specs]
    backgrounds = [load_sample(spec, args.topology, args) for spec in background_specs]
    points = [sample.spec.point for sample in signals]

    input_payload = {
        "method_version": METHOD_VERSION,
        "feature_set": FEATURE_SET,
        "preprocessing_version": PREPROCESSING_VERSION,
        "topology": args.topology,
        "mode_independent": True,
        "signal_manifest_sha256": _sha256_file(signal_manifest),
        "background_manifest_sha256": _sha256_file(background_manifest),
        "feature_inputs": [
            _feature_input_fingerprint(spec) for spec in [*signal_specs, *background_specs]
        ],
        "tagging_scenarios": args.tagging_scenarios,
        "eps_b": args.eps_b,
        "eps_c": args.eps_c,
        "eps_light": args.eps_light,
        "luminosity_fb_inverse": args.luminosity,
        "hbb_branching_ratio": args.hbb_branching_ratio,
        "signal_cross_section_fb": SIGNAL_XSEC_FB,
        "seed": args.seed,
        "folds": N_FOLDS,
        "background_replicas": args.background_replicas,
        "fixed_xgboost_parameters": resolved.FIXED_XGBOOST_PARAMS,
        "minimum_background_raw": args.min_background_raw,
        "minimum_background_neff": args.min_background_neff,
    }
    input_fingerprint = _fingerprint(input_payload)
    if args.mode == "full" and not (model_dir / "model_manifest.json").is_file():
        raise resolved.AnalysisInputError(
            "full mode will not train models; run the matching --mode fast campaign first"
        )
    models, feature_names, model_hashes, models_resumed = load_or_train_models(
        args.topology,
        signals,
        backgrounds,
        points,
        model_dir,
        args.seed,
        args.background_replicas,
        input_fingerprint,
    )
    core_payload = {
        "input_fingerprint": input_fingerprint,
        "model_sha256": model_hashes,
        "feature_names": feature_names,
    }
    core_fingerprint = _fingerprint(core_payload)
    core_manifest_path = checkpoints / "core_manifest.json"
    if core_manifest_path.exists():
        existing = _read_json(core_manifest_path)
        if existing.get("core_fingerprint") != core_fingerprint:
            raise resolved.AnalysisInputError(
                f"{core_manifest_path}: stale core fingerprint; use a new output directory"
            )
    else:
        _atomic_json(
            core_manifest_path,
            {
                "core_fingerprint": core_fingerprint,
                "input_fingerprint": input_fingerprint,
                "core_payload": core_payload,
                "input_payload": input_payload,
            },
        )

    cross_rows = _input_cross_section_rows([*signals, *backgrounds])
    _write_csv(output_dir / "input_cross_sections.csv", cross_rows)
    _atomic_json(output_dir / "input_cross_sections.json", cross_rows)

    template_status: list[dict[str, Any]] = []
    if args.mode in {"fast", "smoke"}:
        global _POINT_CONTEXT
        _POINT_CONTEXT = {
            "args": args,
            "signals": signals,
            "backgrounds": backgrounds,
            "models": models,
            "core_fingerprint": core_fingerprint,
            "score_dir": score_dir,
            "template_dir": template_dir,
            "fast_dir": fast_dir,
        }
        if args.point_jobs == 1:
            completed = [_build_point_template(index) for index in range(len(signals))]
        else:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=min(args.point_jobs, len(signals)), mp_context=context
            ) as pool:
                futures = {
                    pool.submit(_build_point_template, index): index
                    for index in range(len(signals))
                }
                completed = [future.result() for future in as_completed(futures)]
        for index, status in sorted(completed):
            template_status.append(
                {"point_id": signals[index].spec.point.point_id, "status": status}
            )
            print(f"[{status}] {signals[index].spec.point.point_id}", flush=True)
        limit_rows, _ = _collate_fast(
            output_dir, signals, template_dir, fast_dir, core_fingerprint
        )
        plot_outputs = _write_limit_plot(
            output_dir, args.topology, limit_rows, fast=True
        )
        limits_complete = len(limit_rows) == 2 * len(signals) and all(
            row.get("status") == "ok" for row in limit_rows
        )
        manifest = {
            "method_version": METHOD_VERSION,
            "feature_set": FEATURE_SET,
            "mode": "fast" if args.mode == "fast" else "smoke",
            "status": "complete" if limits_complete else "incomplete",
            "physics_result_valid": False,
            "fast_validation_result": True,
            "watermark": FAST_WATERMARK,
            "pyhf_imported_or_required": False,
            "core_fingerprint": core_fingerprint,
            "input_fingerprint": input_fingerprint,
            "topology": args.topology,
            "command": shlex.join(sys.argv),
            "models_resumed": models_resumed,
            "model_sha256": model_hashes,
            "feature_names": feature_names,
            "template_status": template_status,
            "point_jobs": args.point_jobs,
            "threads_per_process": 1,
            "tagging_scenarios": args.tagging_scenarios,
            "normalization_denominator": "source sample generator-weight sum",
            "sumw2_grouping": "sum hypotheses per source event in each bin, then square",
            "raw_count_definition": "unique source generator events",
            "fast_statistic": "exact one-bin Poisson CLs at median background-only integer observation",
            "threshold_scan": {"minimum": 0.0, "maximum": 1.0, "step": 0.001},
            "validation_test_partition": "event-hash 40/60, disjoint and inverse-fraction normalized",
            "plot_outputs": plot_outputs,
            "missing_optional_backgrounds": missing_optional,
            "future_background_roles": ["sm_hhhbb", "sm_hh4b"],
            "optuna": "not used",
        }
        _atomic_json(output_dir / "method_manifest.json", manifest)
        return 0 if limits_complete or args.mode == "smoke" else 3

    # Full mode deliberately consumes verified fast templates and performs no
    # model training, rescoring, or template reconstruction.
    for signal in signals:
        point_id = signal.spec.point.point_id
        for path in (template_dir / f"{point_id}.json", fast_dir / f"{point_id}.json"):
            if not path.is_file():
                raise resolved.AnalysisInputError(
                    f"missing fast cache {path}; run --mode fast first"
                )
            if _read_json(path).get("core_fingerprint") != core_fingerprint:
                raise resolved.AnalysisInputError(f"{path}: stale fast cache")
    _, category_rows = _collate_fast(
        output_dir, signals, template_dir, fast_dir, core_fingerprint
    )
    pyhf_rows, pyhf_fingerprint = _run_pyhf_stage(
        args, signals, template_dir, pyhf_dir, core_fingerprint
    )
    for row in pyhf_rows:
        if "expected_median" in row:
            row["expected_median_limit_fb"] = row["expected_median"]
    _write_csv(output_dir / "point_limits.csv", pyhf_rows)
    _atomic_json(output_dir / "point_limits.json", pyhf_rows)
    plot_outputs = _write_limit_plot(output_dir, args.topology, pyhf_rows, fast=False)
    limits_complete = len(pyhf_rows) == 2 * len(signals) and all(
        row.get("status") == "ok" for row in pyhf_rows
    )
    _atomic_json(
        output_dir / "method_manifest.json",
        {
            "method_version": METHOD_VERSION,
            "feature_set": FEATURE_SET,
            "mode": "full",
            "status": "complete" if limits_complete else "incomplete",
            "physics_result_valid": limits_complete,
            "limit_status_complete": limits_complete,
            "core_fingerprint": core_fingerprint,
            "pyhf_fingerprint": pyhf_fingerprint,
            "topology": args.topology,
            "command": shlex.join(sys.argv),
            "models_resumed": True,
            "rescoring_performed": False,
            "template_reconstruction_performed": False,
            "pyhf_jobs": args.pyhf_jobs,
            "threads_per_process": 1,
            "pyhf_likelihood": "multi-bin category-by-crossfit-fold",
            "background_norm_uncertainty": BACKGROUND_NORM_UNCERTAINTY,
            "normalization_denominator": "source sample generator-weight sum",
            "sumw2_grouping": "sum hypotheses per source event in each bin, then square",
            "plot_outputs": plot_outputs,
            "category_yield_rows": len(category_rows),
            "missing_optional_backgrounds": missing_optional,
            "optuna": "not used",
        },
    )
    return 0 if limits_complete else 3


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", choices=(FEATURE_SET,), default=FEATURE_SET)
    parser.add_argument("--analysis-root", type=Path, default=root)
    parser.add_argument("--topology", required=True, choices=("direct", "cascade"))
    parser.add_argument("--mode", choices=("fast", "full", "smoke"), default="fast")
    parser.add_argument(
        "--signal-manifest",
        type=Path,
        default=Path("HerwigSignalPoints/mass_scan_10k_ak8-v1/manifest.csv"),
    )
    parser.add_argument(
        "--signal-root-dir", type=Path, default=Path("ResonanceAnalysis/features/ak8-v1")
    )
    parser.add_argument(
        "--signal-root-pattern", default="{scenario}/{run_name}_fatjet.root"
    )
    parser.add_argument(
        "--background-manifest",
        type=Path,
        default=Path("ResonanceAnalysis/background_manifest_ak8-v1_features.csv"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tree-name", default="ResonanceFeatures")
    parser.add_argument("--luminosity", type=float, default=LUMINOSITY_FB)
    parser.add_argument("--hbb-branching-ratio", type=float, default=HBB_BRANCHING_RATIO)
    parser.add_argument("--eps-b", type=float, default=0.85)
    parser.add_argument("--eps-c", type=float, default=0.10)
    parser.add_argument("--eps-light", type=float, default=0.01)
    parser.add_argument("--eps-bb-nominal", type=float, default=0.7225)
    parser.add_argument("--fake-bb-nominal", type=float, default=0.10)
    parser.add_argument("--eps-bb-conservative", type=float, default=0.30)
    parser.add_argument("--fake-bb-conservative", type=float, default=0.01)
    parser.add_argument("--background-k-factor", type=float, default=2.0)
    parser.add_argument("--background-replicas", type=int, default=3)
    parser.add_argument("--min-background-raw", type=int, default=25)
    parser.add_argument("--min-background-neff", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--point-jobs", type=int, default=8)
    parser.add_argument("--pyhf-jobs", type=int, default=8)
    parser.add_argument("--smoke-points", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probabilities = (
        "hbb_branching_ratio",
        "eps_b",
        "eps_c",
        "eps_light",
        "eps_bb_nominal",
        "fake_bb_nominal",
        "eps_bb_conservative",
        "fake_bb_conservative",
    )
    for name in probabilities:
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must lie in (0,1)")
    if not math.isfinite(args.luminosity) or args.luminosity <= 0.0:
        raise SystemExit("--luminosity must be positive")
    if (
        args.point_jobs < 1
        or args.pyhf_jobs < 1
        or args.background_replicas < 1
        or args.smoke_points < 1
    ):
        raise SystemExit("worker, replica and smoke counts must be positive")
    if args.min_background_raw < 0 or args.min_background_neff < 0.0:
        raise SystemExit("background template requirements must be non-negative")
    args.tagging_scenarios = {
        "nominal": {
            "eps_bb": args.eps_bb_nominal,
            "fake_bb": args.fake_bb_nominal,
        },
        "conservative": {
            "eps_bb": args.eps_bb_conservative,
            "fake_bb": args.fake_bb_conservative,
        },
    }
    global TAGGING_SCENARIOS
    TAGGING_SCENARIOS = args.tagging_scenarios
    try:
        return run_analysis(args)
    except (resolved.AnalysisInputError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
