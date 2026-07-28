"""Auditable ROOT -> XGBoost -> pyhf helpers for the SM hhhh tutorial.

The functions in this module intentionally keep three concepts separate:

* XGBoost is trained with non-negative, class-balanced ``abs(weight)`` values.
* Expected event yields retain the signed physical Monte Carlo weights.
* The signal HistFactory template is normalized to a production cross section
  of exactly 1 fb, so the pyhf POI ``sigma_hhhh_fb`` has units of fb.

The final likelihood only sees frozen, binned, out-of-fold classifier scores.
It neither fits nor retrains XGBoost.

Heavy optional dependencies are imported inside the functions that use them.
The small ``Code`` directory bootstrap makes imports work both from the
repository root and from a notebook in this tutorial directory.
"""

from __future__ import annotations

import copy
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import logging
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TUTORIAL_METHOD_VERSION = "sm-gg8b-xgboost-pyhf-tutorial-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "Code"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("tutorial_outputs")
EXPECTED_SCHEMA = "extended-91-v2"
EXPECTED_FEATURE_PROFILE = "corrected28"
POI_NAME = "sigma_hhhh_fb"
MEASUREMENT_NAME = "sm_hhhh_measurement"


def _ensure_code_path() -> None:
    code = str(CODE_DIR)
    if code not in sys.path:
        sys.path.insert(0, code)


def _analysis_module(name: str) -> Any:
    _ensure_code_path()
    return importlib.import_module(name)


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    exact: int | None = None,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    try:
        is_integral = float(value) == result
    except (TypeError, ValueError):
        is_integral = False
    if not is_integral:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if exact is not None and result != exact:
        raise ValueError(f"{name} must be {exact} for the declared tutorial split")
    return result


def _label(value: Any, name: str) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        aliases = {
            "signal": 1,
            "sm": 1,
            "hhhh": 1,
            "background": 0,
            "gg8b": 0,
        }
        if normalized in aliases:
            return aliases[normalized]
    result = _integer(value, name)
    if result not in (0, 1):
        raise ValueError(f"{name} must be 1 (signal) or 0 (background)")
    return result


def _optional_sha256(value: Any, name: str) -> str | None:
    if value in (None, ""):
        return None
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest")
    return digest


def _resolve_repo_path(value: Any, repo_root: Path, name: str) -> Path:
    if value in (None, ""):
        raise ValueError(f"{name} is required")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


@dataclass(frozen=True)
class SampleSpec:
    """Configuration for exactly one ROOT event sample."""

    name: str
    label: int
    root_path: Path
    summary_path: Path
    cross_section_fb: float
    k_factor: float
    rate_factor_kind: str
    sha256_root: str | None = None
    sha256_summary: str | None = None
    generation_integration_error_fraction: float | None = None

    @property
    def is_signal(self) -> bool:
        return self.label == 1


@dataclass(frozen=True)
class BinningConfig:
    quantiles: tuple[float, ...] = (0.0, 0.50, 0.75, 0.90, 0.97, 1.0)
    min_bins: int = 2
    max_bins: int = 5
    min_background_raw: int = 25
    min_background_neff: float = 10.0
    within_fraction: float = 0.01


@dataclass(frozen=True)
class BackgroundNormsysConfig:
    name: str = "gg8b_norm"
    lo: float = 0.9
    hi: float = 1.1
    enabled: bool = True
    description: str = (
        "Illustrative correlated normalization uncertainty; not a measured "
        "gg->8b theory uncertainty."
    )


@dataclass(frozen=True)
class TutorialConfig:
    """Validated, path-resolved tutorial configuration."""

    schema: str
    feature_profile: str
    sqrt_s_tev: float
    luminosity_fb: float
    sm_cross_section_fb: float
    branching_ratio_hbb: float
    btag_efficiency: float
    n_folds: int
    seed: int
    binning: BinningConfig
    background_normsys: BackgroundNormsysConfig
    xgboost_params: Mapping[str, Any]
    samples: tuple[SampleSpec, ...]
    config_path: Path
    repo_root: Path = field(default=REPO_ROOT, repr=False)

    @property
    def signal(self) -> SampleSpec:
        return next(sample for sample in self.samples if sample.is_signal)

    @property
    def background(self) -> SampleSpec:
        return next(sample for sample in self.samples if not sample.is_signal)


@dataclass
class LoadedSample:
    """In-memory event arrays with all weight conventions made explicit."""

    spec: SampleSpec
    features: np.ndarray
    raw_weights: np.ndarray
    physical_weights: np.ndarray
    unit_xsec_weights: np.ndarray
    event_indices: np.ndarray
    source_entry_indices: np.ndarray
    folds: np.ndarray
    rate_factor: float
    total_weight_in: float
    summary: dict[str, Any]
    file_contract: dict[str, Any]
    hashes: dict[str, str]

    @property
    def label(self) -> int:
        return self.spec.label

    @property
    def entries(self) -> int:
        return int(self.features.shape[0])


_TOP_LEVEL_CONFIG_KEYS = {
    "schema",
    "feature_profile",
    "sqrt_s_tev",
    "luminosity_fb",
    "sm_cross_section_fb",
    "branching_ratio_hbb",
    "btag_efficiency",
    "n_folds",
    "seed",
    "binning",
    "background_normsys",
    "xgboost_params",
    "samples",
}
_SAMPLE_KEYS = {
    "name",
    "label",
    "root_path",
    "summary_path",
    "cross_section_fb",
    "k_factor",
    "rate_factor_kind",
    "sha256_root",
    "sha256_summary",
    "generation_integration_error_fraction",
}
_BINNING_KEYS = {
    "quantiles",
    "min_bins",
    "max_bins",
    "min_background_raw",
    "min_background_neff",
    "within_fraction",
}
_NORMSYS_KEYS = {"name", "lo", "hi", "enabled", "description"}
_XGBOOST_KEYS = {
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
}


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{where} contains unknown fields: {', '.join(unknown)}")


def _parse_binning(value: Any) -> BinningConfig:
    if not isinstance(value, Mapping):
        raise ValueError("binning must be a JSON object")
    _reject_unknown(value, _BINNING_KEYS, "binning")
    quantiles = tuple(
        _finite_float(item, f"binning.quantiles[{index}]")
        for index, item in enumerate(value.get("quantiles", BinningConfig.quantiles))
    )
    if len(quantiles) < 3 or quantiles[0] != 0.0 or quantiles[-1] != 1.0:
        raise ValueError("binning.quantiles must start at 0, end at 1, and have an interior value")
    if any(right <= left for left, right in zip(quantiles, quantiles[1:])):
        raise ValueError("binning.quantiles must be strictly increasing")
    min_bins = _integer(value.get("min_bins", 2), "binning.min_bins", minimum=1)
    max_bins = _integer(value.get("max_bins", 5), "binning.max_bins", minimum=min_bins)
    if max_bins > len(quantiles) - 1:
        raise ValueError("binning.max_bins exceeds the number of quantile intervals")
    min_raw = _integer(
        value.get("min_background_raw", 25),
        "binning.min_background_raw",
        minimum=0,
    )
    min_neff = _finite_float(
        value.get("min_background_neff", 10.0),
        "binning.min_background_neff",
    )
    within = _finite_float(
        value.get("within_fraction", 0.01), "binning.within_fraction"
    )
    if min_neff < 0.0 or within < 0.0:
        raise ValueError("binning MC constraints and within_fraction must be non-negative")
    return BinningConfig(
        quantiles=quantiles,
        min_bins=min_bins,
        max_bins=max_bins,
        min_background_raw=min_raw,
        min_background_neff=min_neff,
        within_fraction=within,
    )


def _parse_normsys(value: Any) -> BackgroundNormsysConfig:
    if not isinstance(value, Mapping):
        raise ValueError("background_normsys must be a JSON object")
    _reject_unknown(value, _NORMSYS_KEYS, "background_normsys")
    name = str(value.get("name", "gg8b_norm")).strip()
    if not name:
        raise ValueError("background_normsys.name must not be empty")
    lo = _finite_float(value.get("lo", 0.9), "background_normsys.lo", positive=True)
    hi = _finite_float(value.get("hi", 1.1), "background_normsys.hi", positive=True)
    if not lo < 1.0 < hi:
        raise ValueError("background_normsys requires lo < 1 < hi")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("background_normsys.enabled must be true or false")
    return BackgroundNormsysConfig(
        name=name,
        lo=lo,
        hi=hi,
        enabled=enabled,
        description=str(value.get("description", BackgroundNormsysConfig.description)),
    )


def _parse_xgboost_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("xgboost_params must be a JSON object")
    _reject_unknown(value, _XGBOOST_KEYS, "xgboost_params")
    study = _analysis_module("c3d4_xgboost_study")
    params = dict(study.CURRENT_XGBOOST_PARAMS)
    params.update(dict(value))
    integer_fields = ("n_estimators", "max_depth")
    positive_fields = ("learning_rate", "min_child_weight", "subsample", "colsample_bytree")
    nonnegative_fields = ("gamma", "reg_alpha", "reg_lambda")
    for field_name in integer_fields:
        params[field_name] = _integer(
            params[field_name], f"xgboost_params.{field_name}", minimum=1
        )
    for field_name in positive_fields:
        params[field_name] = _finite_float(
            params[field_name], f"xgboost_params.{field_name}", positive=True
        )
    for field_name in nonnegative_fields:
        params[field_name] = _finite_float(
            params[field_name], f"xgboost_params.{field_name}"
        )
        if params[field_name] < 0.0:
            raise ValueError(f"xgboost_params.{field_name} must be non-negative")
    for fraction in ("subsample", "colsample_bytree"):
        if params[fraction] > 1.0:
            raise ValueError(f"xgboost_params.{fraction} must not exceed 1")
    return params


def _parse_sample(value: Any, index: int, repo_root: Path) -> SampleSpec:
    if not isinstance(value, Mapping):
        raise ValueError(f"samples[{index}] must be a JSON object")
    _reject_unknown(value, _SAMPLE_KEYS, f"samples[{index}]")
    name = str(value.get("name", "")).strip()
    if not name:
        raise ValueError(f"samples[{index}].name must not be empty")
    label = _label(value.get("label"), f"samples[{index}].label")
    kind = str(value.get("rate_factor_kind", "")).strip()
    allowed_kinds = {"signal_hhhh_8b", "background_8b"}
    if kind not in allowed_kinds:
        raise ValueError(
            f"samples[{index}].rate_factor_kind must be one of {sorted(allowed_kinds)}"
        )
    if (label == 1) != (kind == "signal_hhhh_8b"):
        raise ValueError(f"samples[{index}] label and rate_factor_kind disagree")
    integration_error = (
        None
        if value.get("generation_integration_error_fraction") is None
        else _finite_float(
            value["generation_integration_error_fraction"],
            f"samples[{index}].generation_integration_error_fraction",
        )
    )
    if integration_error is not None and integration_error < 0.0:
        raise ValueError(
            f"samples[{index}].generation_integration_error_fraction must be non-negative"
        )
    return SampleSpec(
        name=name,
        label=label,
        root_path=_resolve_repo_path(
            value.get("root_path"), repo_root, f"samples[{index}].root_path"
        ),
        summary_path=_resolve_repo_path(
            value.get("summary_path"), repo_root, f"samples[{index}].summary_path"
        ),
        cross_section_fb=_finite_float(
            value.get("cross_section_fb"),
            f"samples[{index}].cross_section_fb",
            positive=True,
        ),
        k_factor=_finite_float(
            value.get("k_factor", 1.0), f"samples[{index}].k_factor", positive=True
        ),
        rate_factor_kind=kind,
        sha256_root=_optional_sha256(
            value.get("sha256_root"), f"samples[{index}].sha256_root"
        ),
        sha256_summary=_optional_sha256(
            value.get("sha256_summary"), f"samples[{index}].sha256_summary"
        ),
        generation_integration_error_fraction=integration_error,
    )


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repo_root: str | Path | None = None,
) -> TutorialConfig:
    """Load the exact JSON configuration used by the notebook.

    Relative sample paths are resolved against the repository root, not the
    caller's current working directory.
    """

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        cwd_candidate = (Path.cwd() / path).resolve()
        repo_candidate = (REPO_ROOT / path).resolve()
        path = cwd_candidate if cwd_candidate.exists() else repo_candidate
    path = path.resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"tutorial config does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("tutorial config must be a JSON object")
    _reject_unknown(payload, _TOP_LEVEL_CONFIG_KEYS, "tutorial config")

    root = REPO_ROOT if repo_root is None else Path(repo_root).expanduser().resolve()
    schema = str(payload.get("schema", ""))
    profile = str(payload.get("feature_profile", ""))
    if schema != EXPECTED_SCHEMA or profile != EXPECTED_FEATURE_PROFILE:
        raise ValueError(
            "this tutorial requires schema='extended-91-v2' and "
            "feature_profile='corrected28'"
        )
    luminosity = _finite_float(
        payload.get("luminosity_fb"), "luminosity_fb", positive=True
    )
    sm_xsec = _finite_float(
        payload.get("sm_cross_section_fb"), "sm_cross_section_fb", positive=True
    )
    branching_ratio = _finite_float(
        payload.get("branching_ratio_hbb"), "branching_ratio_hbb", positive=True
    )
    efficiency = _finite_float(
        payload.get("btag_efficiency"), "btag_efficiency", positive=True
    )
    if branching_ratio > 1.0 or efficiency > 1.0:
        raise ValueError("branching_ratio_hbb and btag_efficiency must not exceed 1")
    samples_payload = payload.get("samples")
    if not isinstance(samples_payload, Sequence) or isinstance(samples_payload, (str, bytes)):
        raise ValueError("samples must be a JSON array")
    samples = tuple(
        _parse_sample(sample, index, root)
        for index, sample in enumerate(samples_payload)
    )
    if len(samples) != 2 or sorted(sample.label for sample in samples) != [0, 1]:
        raise ValueError("this two-process tutorial requires exactly one signal and one background")
    if len({sample.name for sample in samples}) != len(samples):
        raise ValueError("sample names must be unique")
    signal = next(sample for sample in samples if sample.label == 1)
    if not math.isclose(
        signal.cross_section_fb, sm_xsec, rel_tol=1.0e-12, abs_tol=0.0
    ):
        raise ValueError(
            "the signal sample cross_section_fb must equal top-level "
            "sm_cross_section_fb; the unit-cross-section template is derived separately"
        )

    return TutorialConfig(
        schema=schema,
        feature_profile=profile,
        sqrt_s_tev=_finite_float(
            payload.get("sqrt_s_tev"), "sqrt_s_tev", positive=True
        ),
        luminosity_fb=luminosity,
        sm_cross_section_fb=sm_xsec,
        branching_ratio_hbb=branching_ratio,
        btag_efficiency=efficiency,
        n_folds=_integer(payload.get("n_folds"), "n_folds", exact=5),
        seed=_integer(payload.get("seed"), "seed", minimum=0),
        binning=_parse_binning(payload.get("binning", {})),
        background_normsys=_parse_normsys(payload.get("background_normsys", {})),
        xgboost_params=_parse_xgboost_params(payload.get("xgboost_params", {})),
        samples=samples,
        config_path=path,
        repo_root=root,
    )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate_factor(spec: SampleSpec, config: TutorialConfig) -> float:
    btag = config.btag_efficiency**8
    if spec.rate_factor_kind == "signal_hhhh_8b":
        return spec.k_factor * config.branching_ratio_hbb**4 * btag
    if spec.rate_factor_kind == "background_8b":
        return spec.k_factor * btag
    raise AssertionError(f"unhandled rate factor kind {spec.rate_factor_kind!r}")


def _read_summary(spec: SampleSpec, config: TutorialConfig) -> tuple[dict[str, Any], str]:
    if not spec.summary_path.is_file():
        raise FileNotFoundError(f"{spec.name}: summary sidecar does not exist: {spec.summary_path}")
    actual_hash = sha256_file(spec.summary_path)
    if spec.sha256_summary is not None and actual_hash != spec.sha256_summary:
        raise ValueError(
            f"{spec.name}: summary SHA-256 mismatch: got {actual_hash}, "
            f"expected {spec.sha256_summary}"
        )
    try:
        summary = json.loads(spec.summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{spec.summary_path}: invalid sidecar JSON: {error}") from error
    if not isinstance(summary, dict):
        raise ValueError(f"{spec.summary_path}: summary sidecar must be a JSON object")
    if summary.get("observable_schema") != config.schema:
        raise ValueError(
            f"{spec.name}: sidecar schema {summary.get('observable_schema')!r} "
            f"does not match {config.schema!r}"
        )
    total_weight_in = _finite_float(
        summary.get("total_weight_in"),
        f"{spec.name} sidecar total_weight_in",
        positive=True,
    )
    summary["total_weight_in"] = total_weight_in
    return summary, actual_hash


def load_sample(spec: SampleSpec, config: TutorialConfig) -> LoadedSample:
    """Hash, schema-check, and load one Data3/corrected28 ROOT sample."""

    if not spec.root_path.is_file():
        raise FileNotFoundError(f"{spec.name}: ROOT file does not exist: {spec.root_path}")
    root_hash = sha256_file(spec.root_path)
    if spec.sha256_root is not None and root_hash != spec.sha256_root:
        raise ValueError(
            f"{spec.name}: ROOT SHA-256 mismatch: got {root_hash}, "
            f"expected {spec.sha256_root}"
        )
    summary, summary_hash = _read_summary(spec, config)

    reader = _analysis_module("read_root_varfiles")
    features, _, _, metadata = reader.read_ROOT_varfile(
        spec.root_path,
        spec.label,
        xsec=1.0,
        observable_set=config.schema,
        feature_profile=config.feature_profile,
        return_metadata=True,
    )
    feature_array = np.asarray(features, dtype=float)
    raw = np.asarray(metadata["raw_weights"], dtype=float)
    event_indices = np.asarray(metadata["event_indices"], dtype=np.int64)
    source_entries = np.asarray(metadata["source_entry_indices"], dtype=np.int64)
    contract = dict(metadata["file_contract"])
    expected_rows = len(raw)
    if feature_array.shape != (expected_rows, 28):
        raise ValueError(
            f"{spec.name}: expected corrected28 array ({expected_rows}, 28), "
            f"got {feature_array.shape}"
        )
    if not contract.get("root_metadata_verified"):
        raise ValueError(f"{spec.name}: Data3 ROOT metadata was not verified")
    if (
        contract.get("observable_schema") != config.schema
        or contract.get("feature_profile") != config.feature_profile
        or contract.get("tree_name") != "Data3"
        or int(contract.get("feature_count", -1)) != 28
    ):
        raise ValueError(f"{spec.name}: ROOT file contract does not match Data3/corrected28")
    if expected_rows < config.n_folds:
        raise ValueError(
            f"{spec.name}: only {expected_rows} valid events for {config.n_folds} folds"
        )
    if len(np.unique(event_indices)) != len(event_indices):
        raise ValueError(f"{spec.name}: event_index contains duplicates")

    expected_tree_entries = summary.get("feature_tree_mc_events_out")
    if expected_tree_entries is not None and int(expected_tree_entries) != int(
        metadata["entries_considered"]
    ):
        raise ValueError(
            f"{spec.name}: ROOT Data3 entries {metadata['entries_considered']} do not "
            f"match sidecar feature_tree_mc_events_out={expected_tree_entries}"
        )
    expected_tree_weight = summary.get("feature_tree_weight_out")
    if expected_tree_weight is not None and not math.isclose(
        float(np.sum(raw, dtype=np.float64)),
        float(expected_tree_weight),
        rel_tol=2.0e-5,
        abs_tol=1.0e-8,
    ):
        raise ValueError(
            f"{spec.name}: loaded Data3 weight sum does not match the summary sidecar"
        )

    rate_factor = _rate_factor(spec, config)
    total_weight_in = float(summary["total_weight_in"])
    unit = config.luminosity_fb * rate_factor * raw / total_weight_in
    physical = spec.cross_section_fb * unit
    study = _analysis_module("c3d4_xgboost_study")
    source_ids = np.full(expected_rows, spec.name, dtype=object)
    folds = np.asarray(
        study.deterministic_folds(
            source_ids,
            event_indices,
            n_folds=config.n_folds,
            seed=config.seed,
        ),
        dtype=np.int16,
    )
    return LoadedSample(
        spec=spec,
        features=feature_array,
        raw_weights=raw,
        physical_weights=np.asarray(physical, dtype=float),
        unit_xsec_weights=np.asarray(unit, dtype=float),
        event_indices=event_indices,
        source_entry_indices=source_entries,
        folds=folds,
        rate_factor=rate_factor,
        total_weight_in=total_weight_in,
        summary=summary,
        file_contract=contract,
        hashes={"root": root_hash, "summary": summary_hash},
    )


def load_samples(config: TutorialConfig) -> tuple[LoadedSample, LoadedSample]:
    """Return ``(signal, background)`` in a stable order."""

    loaded = {sample.label: load_sample(sample, config) for sample in config.samples}
    return loaded[1], loaded[0]


def _combined_event_arrays(
    signal: LoadedSample, background: LoadedSample
) -> dict[str, np.ndarray]:
    return {
        "features": np.concatenate([signal.features, background.features], axis=0),
        "labels": np.concatenate(
            [
                np.ones(signal.entries, dtype=np.int8),
                np.zeros(background.entries, dtype=np.int8),
            ]
        ),
        "raw_weights": np.concatenate([signal.raw_weights, background.raw_weights]),
        "physical_weights": np.concatenate(
            [signal.physical_weights, background.physical_weights]
        ),
        "unit_xsec_weights": np.concatenate(
            [signal.unit_xsec_weights, np.zeros(background.entries, dtype=float)]
        ),
        "event_indices": np.concatenate(
            [signal.event_indices, background.event_indices]
        ),
        "source_entry_indices": np.concatenate(
            [signal.source_entry_indices, background.source_entry_indices]
        ),
        "folds": np.concatenate([signal.folds, background.folds]),
        "sample_names": np.concatenate(
            [
                np.full(signal.entries, signal.spec.name, dtype=object),
                np.full(background.entries, background.spec.name, dtype=object),
            ]
        ),
    }


def _classifier_weights(
    labels: np.ndarray,
    physical_weights: np.ndarray,
    study: Any,
) -> np.ndarray:
    """Balance classes while keeping the typical XGBoost weight O(1)."""

    base = np.abs(np.asarray(physical_weights, dtype=float))
    effective_rows = int(np.count_nonzero(base))
    if effective_rows <= 0:
        raise ValueError("classifier split has no non-zero training weights")
    return np.asarray(
        study.balanced_binary_training_weights(
            labels,
            base,
            class_total=0.5 * effective_rows,
        ),
        dtype=float,
    )


def _xgboost_parameters(config: TutorialConfig, rotation: int, study: Any) -> dict[str, Any]:
    params = dict(config.xgboost_params)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": int(study.stable_seed(f"xgboost-fold-{rotation}", config.seed)),
            "n_jobs": 1,
        }
    )
    return params


def crossfit_xgboost(
    signal: LoadedSample,
    background: LoadedSample,
    config: TutorialConfig,
    *,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train five deterministic 3/1/1 rotations and score every event OOF.

    Rotation ``r`` uses fold ``r`` only for the final template, fold
    ``(r + 1) % 5`` only for validation/binning, and the remaining folds for
    training. Hyperparameters are fixed before any validation event is read.
    """

    try:
        import xgboost as xgb
    except ImportError as error:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "XGBoost is required for crossfit_xgboost; install the tutorial requirements"
        ) from error

    study = _analysis_module("c3d4_xgboost_study")
    schemas = _analysis_module("observable_schemas")
    arrays = _combined_event_arrays(signal, background)
    features = arrays["features"]
    labels = arrays["labels"]
    physical = arrays["physical_weights"]
    folds = arrays["folds"]
    n_events = len(labels)
    oof_scores = np.full(n_events, np.nan, dtype=float)
    validation_scores = np.full(n_events, np.nan, dtype=float)
    test_multiplicity = np.zeros(n_events, dtype=np.int8)
    validation_multiplicity = np.zeros(n_events, dtype=np.int8)
    rotations: list[dict[str, Any]] = []
    models: list[Any] = []

    destination = None if model_dir is None else Path(model_dir).resolve()
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    contract = schemas.get_feature_contract(config.schema, config.feature_profile)
    feature_names = list(contract.feature_names)

    for rotation in range(config.n_folds):
        masks = study.rotation_masks(folds, rotation, n_folds=config.n_folds)
        train = np.asarray(masks["train"], dtype=bool)
        validation = np.asarray(masks["validation"], dtype=bool)
        test = np.asarray(masks["test"], dtype=bool)
        for split_name, mask in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            present = np.unique(labels[mask])
            if not np.array_equal(present, np.asarray([0, 1], dtype=np.int8)):
                raise ValueError(
                    f"rotation {rotation} {split_name} split does not contain both classes"
                )
        if np.any(train & validation) or np.any(train & test) or np.any(validation & test):
            raise AssertionError(f"rotation {rotation}: split masks overlap")

        train_weights = _classifier_weights(labels[train], physical[train], study)
        validation_weights = _classifier_weights(
            labels[validation], physical[validation], study
        )
        params = _xgboost_parameters(config, rotation, study)
        model = xgb.XGBClassifier(**params)
        model.fit(
            features[train],
            labels[train],
            sample_weight=train_weights,
            eval_set=[
                (features[train], labels[train]),
                (features[validation], labels[validation]),
            ],
            sample_weight_eval_set=[train_weights, validation_weights],
            verbose=False,
        )
        booster = model.get_booster()
        booster.feature_names = feature_names
        split_count = int(
            sum(tree.count('"split"') for tree in booster.get_dump(dump_format="json"))
        )
        if split_count <= 0:
            raise RuntimeError(
                f"rotation {rotation}: XGBoost produced no split nodes; check weight scaling"
            )
        metadata = schemas.attach_model_metadata(
            model,
            observable_set=config.schema,
            feature_profile=config.feature_profile,
            training_strategy="sm-gg8b-crossfit-3-1-1",
            method_version=TUTORIAL_METHOD_VERSION,
            fold=rotation,
            train_folds=sorted(set(folds[train].astype(int).tolist())),
            validation_fold=int(masks["validation_fold"]),
            test_fold=int(masks["test_fold"]),
            parameters=params,
            seed=int(params["random_state"]),
            xgboost_split_nodes=split_count,
        )
        schemas.validate_model_contract(model, config.schema, config.feature_profile)

        validation_prediction = np.asarray(
            model.predict_proba(features[validation])[:, 1], dtype=float
        )
        test_prediction = np.asarray(
            model.predict_proba(features[test])[:, 1], dtype=float
        )
        if not (
            np.all(np.isfinite(validation_prediction))
            and np.all(np.isfinite(test_prediction))
            and np.all((validation_prediction >= 0.0) & (validation_prediction <= 1.0))
            and np.all((test_prediction >= 0.0) & (test_prediction <= 1.0))
        ):
            raise RuntimeError(f"rotation {rotation}: XGBoost returned invalid probabilities")
        validation_scores[validation] = validation_prediction
        oof_scores[test] = test_prediction
        validation_multiplicity[validation] += 1
        test_multiplicity[test] += 1

        model_path = None
        if destination is not None:
            model_path = destination / f"fold_{rotation}.json"
            # Saving the Booster avoids an XGBoost 3.0 / very-new scikit-learn
            # wrapper compatibility issue involving ``_estimator_type``.
            model.get_booster().save_model(str(model_path))
            _write_json(destination / f"metadata_fold_{rotation}.json", metadata)
        # XGBoost returns {validation_0: {logloss: [...]}, ...}; keep that
        # structure but convert NumPy scalar types to ordinary JSON numbers.
        history = {
            dataset: {
                metric: [float(item) for item in values]
                for metric, values in metrics.items()
            }
            for dataset, metrics in model.evals_result().items()
        }
        rotations.append(
            {
                "rotation": rotation,
                "train_mask": train,
                "validation_mask": validation,
                "test_mask": test,
                "train_folds": sorted(set(folds[train].astype(int).tolist())),
                "validation_fold": int(masks["validation_fold"]),
                "test_fold": int(masks["test_fold"]),
                "n_train": int(np.sum(train)),
                "n_validation": int(np.sum(validation)),
                "n_test": int(np.sum(test)),
                "classifier_signal_weight": float(
                    np.sum(train_weights[labels[train] == 1])
                ),
                "classifier_background_weight": float(
                    np.sum(train_weights[labels[train] == 0])
                ),
                "parameters": params,
                "metadata": metadata,
                "history": history,
                "model_path": None if model_path is None else str(model_path),
            }
        )
        models.append(model)

    if not np.all(test_multiplicity == 1) or not np.all(validation_multiplicity == 1):
        raise AssertionError(
            "cross-fitting invariant failed: every event must be test and validation once"
        )
    if not np.all(np.isfinite(oof_scores)) or not np.all(np.isfinite(validation_scores)):
        raise AssertionError("cross-fitting left unscored events")
    return {
        "method_version": TUTORIAL_METHOD_VERSION,
        "arrays": arrays,
        "oof_scores": oof_scores,
        "validation_scores": validation_scores,
        "rotations": rotations,
        "models": models,
        "feature_names": feature_names,
    }


def _valid_signal_template(channel: Mapping[str, Any]) -> bool:
    signal = np.asarray(channel["signal"], dtype=float)
    return bool(
        signal.size
        and np.all(np.isfinite(signal))
        and np.all(signal >= 0.0)
        and float(np.sum(signal)) > 0.0
    )


def select_binning_and_build_channels(
    crossfit: Mapping[str, Any],
    config: TutorialConfig,
) -> dict[str, Any]:
    """Freeze bins on each validation fold, then apply them to its test fold.

    If a test template has a nonpositive signed background bin, only the
    nested coarsenings declared by validation are tried. No bin is clipped.
    """

    study = _analysis_module("c3d4_xgboost_study")
    arrays = crossfit["arrays"]
    labels = np.asarray(arrays["labels"], dtype=int)
    physical = np.asarray(arrays["physical_weights"], dtype=float)
    unit = np.asarray(arrays["unit_xsec_weights"], dtype=float)
    validation_scores = np.asarray(crossfit["validation_scores"], dtype=float)
    test_scores = np.asarray(crossfit["oof_scores"], dtype=float)
    binning = config.binning
    channels: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for rotation_record in crossfit["rotations"]:
        rotation = int(rotation_record["rotation"])
        validation = np.asarray(rotation_record["validation_mask"], dtype=bool)
        test = np.asarray(rotation_record["test_mask"], dtype=bool)
        validation_signal = validation & (labels == 1)
        validation_background = validation & (labels == 0)
        test_signal = test & (labels == 1)
        test_background = test & (labels == 0)

        # A one-in-five validation fold is deterministically scaled to the
        # full luminosity for choosing among candidate binnings. This scale is
        # predeclared and never derived from the held-out test yield. Multiplying
        # every validation weight by five also multiplies sum(w^2) by 25, so
        # the relative MC uncertainty remains that of one fold. Compared with
        # five independent folds at the same total yield, this is conservative
        # by sqrt(5) and can favor coarser binning.
        validation_scale = float(config.n_folds)
        validation_background_scores = validation_scores[validation_background]
        validation_background_weights = (
            physical[validation_background] * validation_scale
        )
        validation_signal_scores = validation_scores[validation_signal]
        validation_signal_weights = unit[validation_signal] * validation_scale

        def validation_limit_evaluator(edges: Sequence[float]) -> float:
            validation_channel = study.build_pyhf_channel(
                f"validation_fold_{rotation}",
                validation_signal_scores,
                validation_signal_weights,
                validation_background_scores,
                validation_background_weights,
                edges,
            )
            result = study.pyhf_combined_limit(
                [validation_channel],
                include_staterror=True,
                poi_bounds=(0.0, 10.0),
                raise_on_failure=True,
            )
            return float(result["expected_median"])

        selected = study.validation_binning(
            validation_background_scores,
            validation_background_weights,
            limit_evaluator=validation_limit_evaluator,
            quantiles=binning.quantiles,
            min_bins=binning.min_bins,
            max_bins=binning.max_bins,
            min_background_raw=binning.min_background_raw,
            min_background_neff=binning.min_background_neff,
            within_fraction=binning.within_fraction,
            include_staterror=True,
        )

        background_selection = study.select_test_binning(
            selected,
            test_scores[test_background],
            physical[test_background],
        )
        attempts: list[dict[str, Any]] = []
        chosen_channel = None
        chosen_level = None
        for level, candidate in enumerate(selected["fallback_hierarchy"]):
            channel = study.build_pyhf_channel(
                f"fold_{rotation}",
                test_scores[test_signal],
                unit[test_signal],
                test_scores[test_background],
                physical[test_background],
                candidate["edges"],
            )
            positive_background = bool(
                np.all(np.asarray(channel["background"], dtype=float) > 0.0)
            )
            valid_signal = _valid_signal_template(channel)
            attempts.append(
                {
                    "fallback_level": level,
                    "edges": list(map(float, candidate["edges"])),
                    "positive_background": positive_background,
                    "valid_signal": valid_signal,
                }
            )
            if positive_background and valid_signal:
                chosen_channel = channel
                chosen_level = level
                break
        if chosen_channel is None:
            raise RuntimeError(
                f"rotation {rotation}: every validation-defined test binning has a "
                "nonpositive signed signal/background template; refusing to clip"
            )
        channels.append(chosen_channel)
        records.append(
            {
                "rotation": rotation,
                "validation_scale": validation_scale,
                "validation_mcstat_convention": (
                    "one-fold weights scaled to full yield; relative MC "
                    "uncertainty remains one-fold and is conservative"
                ),
                "validation_mcstat_relative_inflation_vs_full_sample": float(
                    math.sqrt(config.n_folds)
                ),
                "validation": selected,
                "test_background_selection": background_selection,
                "test_attempts": attempts,
                "chosen_fallback_level": int(chosen_level),
                "used_fallback": bool(chosen_level),
                "edges": list(map(float, chosen_channel["edges"])),
            }
        )

    return {"channels": channels, "records": records}


def _channel_as_json(channel: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(value)
        for key, value in channel.items()
    }


def _suggest_poi_upper(
    channels: Sequence[Mapping[str, Any]], sm_cross_section_fb: float
) -> float:
    study = _analysis_module("c3d4_xgboost_study")
    signal = float(
        sum(np.sum(np.asarray(channel["signal"], dtype=float)) for channel in channels)
    )
    background = float(
        sum(np.sum(np.asarray(channel["background"], dtype=float)) for channel in channels)
    )
    if signal <= 0.0 or background < 0.0:
        raise ValueError("templates do not have positive total sensitivity")
    one_bin_estimate = study.exact_cls_signal_upper_limit(background) / signal
    return float(max(100.0, 100.0 * one_bin_estimate, 20.0 * sm_cross_section_fb))


def build_workspace(
    channels: Sequence[Mapping[str, Any]],
    config: TutorialConfig,
    *,
    include_staterror: bool = True,
    include_background_normsys: bool | None = None,
    poi_upper: float | None = None,
) -> dict[str, Any]:
    """Build a schema-valid HistFactory workspace with a cross-section POI."""

    if not channels:
        raise ValueError("at least one channel is required")
    normsys = config.background_normsys
    use_normsys = normsys.enabled if include_background_normsys is None else bool(
        include_background_normsys
    )
    upper = (
        _suggest_poi_upper(channels, config.sm_cross_section_fb)
        if poi_upper is None
        else _finite_float(poi_upper, "poi_upper", positive=True)
    )
    channel_specs: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, raw_channel in enumerate(channels):
        channel = _channel_as_json(raw_channel)
        name = str(channel["name"])
        if name in seen_names:
            raise ValueError(f"duplicate channel name {name!r}")
        seen_names.add(name)
        signal = np.asarray(channel["signal"], dtype=float)
        background = np.asarray(channel["background"], dtype=float)
        signal_error = np.asarray(channel["signal_staterror"], dtype=float)
        background_error = np.asarray(channel["background_staterror"], dtype=float)
        if signal.ndim != 1 or len(signal) == 0 or signal.shape != background.shape:
            raise ValueError(f"{name}: signal/background templates must be matching 1D arrays")
        if (
            np.any(~np.isfinite(signal))
            or np.any(~np.isfinite(background))
            or np.any(signal < 0.0)
            or np.any(background <= 0.0)
            or float(np.sum(signal)) <= 0.0
        ):
            raise ValueError(f"{name}: pyhf templates contain invalid signed yields")
        if signal_error.shape != signal.shape or background_error.shape != background.shape:
            raise ValueError(f"{name}: staterror arrays do not match their templates")
        if np.any(signal_error < 0.0) or np.any(background_error < 0.0):
            raise ValueError(f"{name}: staterrors must be non-negative")

        signal_modifiers: list[dict[str, Any]] = [
            {"name": POI_NAME, "type": "normfactor", "data": None}
        ]
        background_modifiers: list[dict[str, Any]] = []
        if include_staterror:
            if np.any(signal_error > 0.0):
                signal_modifiers.append(
                    {
                        "name": f"signal_mcstat_{index}_{name}",
                        "type": "staterror",
                        "data": signal_error.tolist(),
                    }
                )
            if np.any(background_error > 0.0):
                background_modifiers.append(
                    {
                        "name": f"background_mcstat_{index}_{name}",
                        "type": "staterror",
                        "data": background_error.tolist(),
                    }
                )
        if use_normsys:
            background_modifiers.append(
                {
                    "name": normsys.name,
                    "type": "normsys",
                    "data": {"lo": normsys.lo, "hi": normsys.hi},
                }
            )
        channel_specs.append(
            {
                "name": name,
                "samples": [
                    {
                        "name": "sm_hhhh_per_fb",
                        "data": signal.tolist(),
                        "modifiers": signal_modifiers,
                    },
                    {
                        "name": "gg8b",
                        "data": background.tolist(),
                        "modifiers": background_modifiers,
                    },
                ],
            }
        )
        # Headline expected limits use a background-only Asimov observation.
        observations.append({"name": name, "data": background.tolist()})

    return {
        "channels": channel_specs,
        "observations": observations,
        "measurements": [
            {
                "name": MEASUREMENT_NAME,
                "config": {
                    "poi": POI_NAME,
                    "parameters": [
                        {
                            "name": POI_NAME,
                            "bounds": [[0.0, upper]],
                            "inits": [
                                min(max(config.sm_cross_section_fb, 1.0e-12), upper)
                            ],
                        }
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }


def construct_channels(
    crossfit: Mapping[str, Any],
    config: TutorialConfig,
) -> dict[str, Any]:
    """Notebook-friendly alias for :func:`select_binning_and_build_channels`."""

    return select_binning_and_build_channels(crossfit, config)


def build_workspace_spec(
    channels: Sequence[Mapping[str, Any]],
    config: TutorialConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    """Notebook-friendly alias for :func:`build_workspace`."""

    return build_workspace(channels, config, **kwargs)


def _configure_pyhf(*, prefer_minuit: bool = False) -> tuple[Any, str]:
    try:
        import pyhf
    except ImportError as error:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "pyhf is required for inference; install the tutorial requirements"
        ) from error
    if prefer_minuit:
        try:
            # The default Minuit tolerance is intentionally loose for broad
            # compatibility. A deterministic Asimov closure test should recover
            # its generating parameters much more accurately, so request a
            # tighter EDM convergence criterion.
            optimizer = pyhf.optimize.minuit_optimizer(
                verbose=0,
                tolerance=1.0e-7,
            )
            pyhf.set_backend("numpy", optimizer)
            return pyhf, "numpy+iminuit"
        except (ImportError, AttributeError):
            pass
    pyhf.set_backend("numpy", pyhf.optimize.scipy_optimizer())
    return pyhf, "numpy+scipy"


def _float_scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"expected a scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _workspace_model(pyhf: Any, workspace_spec: Mapping[str, Any]) -> tuple[Any, Any]:
    workspace = pyhf.Workspace(dict(workspace_spec))
    model = workspace.model(measurement_name=MEASUREMENT_NAME)
    if model.config.poi_name != POI_NAME:
        raise ValueError(
            f"workspace POI is {model.config.poi_name!r}, expected {POI_NAME!r}"
        )
    return workspace, model


def _toms748_upper_limit(
    pyhf: Any,
    data: Sequence[float],
    model: Any,
    *,
    confidence_level: float = 0.95,
) -> tuple[Any, Any]:
    """Return asymptotic CLs limits while honoring the requested confidence.

    In pyhf 0.7.6, ``upper_limit(data, model, scan=None, level=...)`` delegates
    to ``toms748_scan`` without forwarding ``level``. Calling the root finder
    directly avoids that version-specific bug. Here ``confidence_level`` is
    the coverage (for example 0.95), while pyhf's ``level`` is the CLs
    threshold ``alpha = 1 - confidence_level``.
    """

    confidence_level = _finite_float(confidence_level, "confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    poi_bounds = model.config.suggested_bounds()[model.config.poi_index]
    return pyhf.infer.intervals.upper_limits.toms748_scan(
        data,
        model,
        float(poi_bounds[0]),
        float(poi_bounds[1]),
        level=1.0 - confidence_level,
        test_stat="qtilde",
    )


def _parameter_constraint_widths(model: Any) -> np.ndarray:
    widths = np.full(model.config.npars, np.nan, dtype=float)
    for parameter_set_name in model.config.par_order:
        parameter_set = model.config.param_set(parameter_set_name)
        if not getattr(parameter_set, "constrained", False):
            continue
        try:
            values = np.asarray(parameter_set.width(), dtype=float).reshape(-1)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            values = np.ones(parameter_set.n_parameters, dtype=float)
        widths[model.config.par_slice(parameter_set_name)] = values
    return widths


def _fit_with_uncertainties(
    workspace_spec: Mapping[str, Any],
    data: Sequence[float],
) -> dict[str, Any]:
    pyhf, backend = _configure_pyhf(prefer_minuit=True)
    _, model = _workspace_model(pyhf, workspace_spec)
    correlations = None
    uncertainties = None
    try:
        fit_parameters, correlations_raw, twice_nll = pyhf.infer.mle.fit(
            data,
            model,
            return_uncertainties=True,
            return_correlations=True,
            return_fitted_val=True,
        )
        fit_array = np.asarray(fit_parameters, dtype=float)
        if fit_array.ndim == 2 and fit_array.shape == (model.config.npars, 2):
            uncertainties = fit_array[:, 1]
            bestfit = fit_array[:, 0]
        else:
            bestfit = fit_array.reshape(-1)
        if correlations_raw is not None:
            candidate = np.asarray(correlations_raw, dtype=float)
            if candidate.shape == (model.config.npars, model.config.npars):
                correlations = candidate
    except Exception:
        # Keep the tutorial usable without iminuit. The fallback fit remains
        # statistically valid, but covariance-dependent diagnostics are absent.
        pyhf, backend = _configure_pyhf(prefer_minuit=False)
        _, model = _workspace_model(pyhf, workspace_spec)
        bestfit_raw, twice_nll = pyhf.infer.mle.fit(
            data, model, return_fitted_val=True
        )
        bestfit = np.asarray(bestfit_raw, dtype=float).reshape(-1)

    expected = np.asarray(model.expected_actualdata(bestfit), dtype=float)
    nominal = np.asarray(model.config.suggested_init(), dtype=float)
    widths = _parameter_constraint_widths(model)
    pulls = np.divide(
        bestfit - nominal,
        widths,
        out=np.full_like(bestfit, np.nan),
        where=np.isfinite(widths) & (widths > 0.0),
    )
    pull_uncertainties = None
    if uncertainties is not None:
        pull_uncertainties = np.divide(
            uncertainties,
            widths,
            out=np.full_like(uncertainties, np.nan),
            where=np.isfinite(widths) & (widths > 0.0),
        )
    return {
        "backend": backend,
        "parameter_names": list(model.config.par_names),
        "bestfit_parameters": bestfit.tolist(),
        "parameter_uncertainties": (
            None if uncertainties is None else uncertainties.tolist()
        ),
        "correlation_matrix": (
            None if correlations is None else correlations.tolist()
        ),
        "constraint_widths": widths.tolist(),
        "pulls": pulls.tolist(),
        "pull_uncertainties": (
            None if pull_uncertainties is None else pull_uncertainties.tolist()
        ),
        "twice_nll": _float_scalar(twice_nll),
        "expected_postfit": expected.tolist(),
        "poi_hat_fb": float(bestfit[model.config.poi_index]),
    }


def _postfit_expected_uncertainty(
    workspace_spec: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> list[float] | None:
    uncertainties_raw = fit.get("parameter_uncertainties")
    correlations_raw = fit.get("correlation_matrix")
    if uncertainties_raw is None or correlations_raw is None:
        return None
    pyhf, _ = _configure_pyhf(prefer_minuit=False)
    _, model = _workspace_model(pyhf, workspace_spec)
    parameters = np.asarray(fit["bestfit_parameters"], dtype=float)
    uncertainties = np.asarray(uncertainties_raw, dtype=float)
    correlations = np.asarray(correlations_raw, dtype=float)
    covariance = correlations * np.outer(uncertainties, uncertainties)
    bounds = np.asarray(model.config.suggested_bounds(), dtype=float)
    nominal_prediction = np.asarray(model.expected_actualdata(parameters), dtype=float)
    jacobian = np.zeros((len(nominal_prediction), len(parameters)), dtype=float)
    for index, (value, uncertainty) in enumerate(zip(parameters, uncertainties)):
        if not math.isfinite(float(uncertainty)) or uncertainty <= 0.0:
            continue
        step = max(0.1 * float(uncertainty), 1.0e-6 * max(1.0, abs(float(value))))
        lower = max(float(bounds[index, 0]), float(value) - step)
        upper = min(float(bounds[index, 1]), float(value) + step)
        if upper <= lower:
            continue
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] = upper
        minus[index] = lower
        jacobian[:, index] = (
            np.asarray(model.expected_actualdata(plus), dtype=float)
            - np.asarray(model.expected_actualdata(minus), dtype=float)
        ) / (upper - lower)
    variance = np.einsum("bi,ij,bj->b", jacobian, covariance, jacobian)
    return np.sqrt(np.clip(variance, 0.0, None)).tolist()


def _profile_likelihood(
    workspace_spec: Mapping[str, Any],
    data: Sequence[float],
    poi_max: float,
    *,
    points: int,
) -> dict[str, Any]:
    pyhf, backend = _configure_pyhf(prefer_minuit=False)
    _, model = _workspace_model(pyhf, workspace_spec)
    scan = np.linspace(0.0, float(poi_max), int(points))
    values = []
    for poi in scan:
        _, twice_nll = pyhf.infer.mle.fixed_poi_fit(
            float(poi), data, model, return_fitted_val=True
        )
        values.append(_float_scalar(twice_nll))
    _, free_twice_nll = pyhf.infer.mle.fit(
        data, model, return_fitted_val=True
    )
    reference = min(_float_scalar(free_twice_nll), min(values))
    delta_nll = 0.5 * (np.asarray(values, dtype=float) - reference)
    return {
        "backend": backend,
        "sigma_hhhh_fb": scan.tolist(),
        "delta_nll": np.clip(delta_nll, 0.0, None).tolist(),
        "reference_twice_nll": reference,
    }


def _cls_scan(
    workspace_spec: Mapping[str, Any],
    data: Sequence[float],
    poi_max: float,
    *,
    points: int,
) -> dict[str, Any]:
    pyhf, backend = _configure_pyhf(prefer_minuit=False)
    _, model = _workspace_model(pyhf, workspace_spec)
    scan = np.linspace(0.0, float(poi_max), int(points))
    observed: list[float] = []
    expected: list[list[float]] = []
    for index, poi in enumerate(scan):
        if index == 0:
            observed.append(1.0)
            expected.append([1.0] * 5)
            continue
        result = pyhf.infer.hypotest(
            float(poi),
            data,
            model,
            test_stat="qtilde",
            return_expected_set=True,
        )
        observed_value, expected_values = result
        observed.append(_float_scalar(observed_value))
        expected.append([_float_scalar(item) for item in expected_values])
    return {
        "backend": backend,
        "sigma_hhhh_fb": scan.tolist(),
        "observed_cls": observed,
        "expected_cls": expected,
        "expected_band_order": ["-2sigma", "-1sigma", "median", "+1sigma", "+2sigma"],
    }


def _channel_vector_map(model: Any, values: Sequence[float]) -> dict[str, list[float]]:
    array = np.asarray(values, dtype=float)
    return {
        name: array[model.config.channel_slices[name]].tolist()
        for name in model.config.channels
    }


def run_pyhf_inference(
    workspace_spec: Mapping[str, Any],
    config: TutorialConfig,
    *,
    injection_fraction_of_expected_limit: float = 0.5,
    injected_background_shift_sigma: float = 0.5,
    profile_points: int = 31,
    cls_points: int = 21,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Fit/profile a labeled injected Asimov sample and compute expected CLs.

    The headline limit is always calculated from the background-only Asimov
    observations archived in ``workspace_spec``. The injected sample is
    generated separately from the same model and is labeled ``NOT DATA``.
    """

    fraction = _finite_float(
        injection_fraction_of_expected_limit,
        "injection_fraction_of_expected_limit",
        positive=True,
    )
    shift = _finite_float(
        injected_background_shift_sigma, "injected_background_shift_sigma"
    )
    profile_points = _integer(profile_points, "profile_points", minimum=5)
    cls_points = _integer(cls_points, "cls_points", minimum=5)
    confidence_level = _finite_float(confidence_level, "confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    pyhf, limit_backend = _configure_pyhf(prefer_minuit=False)
    workspace, model = _workspace_model(pyhf, workspace_spec)
    background_data = np.asarray(workspace.data(model), dtype=float)
    observed_limit, expected_limits = _toms748_upper_limit(
        pyhf,
        background_data,
        model,
        confidence_level=confidence_level,
    )
    expected_array = np.asarray(expected_limits, dtype=float).reshape(-1)
    if len(expected_array) != 5 or np.any(~np.isfinite(expected_array)):
        raise RuntimeError("pyhf did not return the five expected CLs limit bands")
    expected_median = float(expected_array[2])
    injection_sigma = fraction * expected_median
    background_only_fit = _fit_with_uncertainties(
        workspace_spec, background_data
    )

    truth = np.asarray(model.config.suggested_init(), dtype=float)
    truth[model.config.poi_index] = injection_sigma
    applied_shift = 0.0
    normsys_name = config.background_normsys.name
    if config.background_normsys.enabled and normsys_name in model.config.parameters:
        normsys_slice = model.config.par_slice(normsys_name)
        truth[normsys_slice] = shift
        applied_shift = shift
    injected_data = np.asarray(model.expected_data(truth), dtype=float)
    injected_actual = injected_data[: model.config.nmaindata]
    label = (
        "Injected Asimov (NOT DATA): "
        f"sigma_hhhh={injection_sigma:.6g} fb, "
        f"{normsys_name}={applied_shift:+.2f} constraint sigma"
    )
    fit = _fit_with_uncertainties(workspace_spec, injected_data)
    fit["expected_postfit_uncertainty"] = _postfit_expected_uncertainty(
        workspace_spec, fit
    )

    profile_max = max(
        2.0 * injection_sigma,
        1.35 * float(expected_array[-1]),
        10.0 * config.sm_cross_section_fb,
    )
    poi_bound = float(
        workspace_spec["measurements"][0]["config"]["parameters"][0]["bounds"][0][1]
    )
    profile_max = min(profile_max, 0.98 * poi_bound)
    cls_max = min(max(1.4 * float(expected_array[-1]), expected_median), 0.98 * poi_bound)
    profile = _profile_likelihood(
        workspace_spec,
        injected_data,
        profile_max,
        points=profile_points,
    )
    cls = _cls_scan(
        workspace_spec,
        background_data,
        cls_max,
        points=cls_points,
    )

    # Recreate model after backend switches before using backend tensors.
    pyhf, _ = _configure_pyhf(prefer_minuit=False)
    _, model = _workspace_model(pyhf, workspace_spec)
    return {
        "method_version": TUTORIAL_METHOD_VERSION,
        "pyhf_version": getattr(pyhf, "__version__", None),
        "poi": {
            "name": POI_NAME,
            "unit": "fb",
            "sm_cross_section_fb": config.sm_cross_section_fb,
        },
        "headline_expected_limit": {
            "data_label": "Background-only Asimov (expected limit; not observed data)",
            "backend": limit_backend,
            "confidence_level": confidence_level,
            "observed_on_background_asimov_fb": _float_scalar(observed_limit),
            "expected_bands_fb": expected_array.tolist(),
            "expected_band_order": [
                "-2sigma",
                "-1sigma",
                "median",
                "+1sigma",
                "+2sigma",
            ],
            "expected_median_fb": expected_median,
            "expected_median_mu": expected_median / config.sm_cross_section_fb,
        },
        "background_only_fit": {
            "data_label": "Background-only Asimov (NOT OBSERVED DATA)",
            **background_only_fit,
        },
        "injected_asimov": {
            "label": label,
            "is_observed_data": False,
            "sigma_hhhh_fb": injection_sigma,
            "mu": injection_sigma / config.sm_cross_section_fb,
            "background_normsys_shift_sigma": applied_shift,
            "truth_parameters": truth.tolist(),
            "full_data_vector": injected_data.tolist(),
            "observations_by_channel": _channel_vector_map(model, injected_actual),
            "fit": fit,
            "profile_likelihood": profile,
        },
        "background_only_cls_scan": cls,
    }


def _expected_limit_summary(
    workspace_spec: Mapping[str, Any],
    *,
    data_label: str,
    config: TutorialConfig,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Evaluate a background-only Asimov upper limit for one workspace."""

    pyhf, backend = _configure_pyhf(prefer_minuit=False)
    workspace, model = _workspace_model(pyhf, workspace_spec)
    observed, expected = _toms748_upper_limit(
        pyhf,
        workspace.data(model),
        model,
        confidence_level=confidence_level,
    )
    expected_values = np.asarray(expected, dtype=float).reshape(-1)
    if len(expected_values) != 5:
        raise RuntimeError("pyhf did not return five expected limit bands")
    median = float(expected_values[2])
    return {
        "data_label": data_label,
        "backend": backend,
        "confidence_level": confidence_level,
        "observed_on_background_asimov_fb": _float_scalar(observed),
        "expected_bands_fb": expected_values.tolist(),
        "expected_band_order": [
            "-2sigma",
            "-1sigma",
            "median",
            "+1sigma",
            "+2sigma",
        ],
        "expected_median_fb": median,
        "expected_median_mu": median / config.sm_cross_section_fb,
    }


def build_one_bin_control(
    channels: Sequence[Mapping[str, Any]],
    config: TutorialConfig,
) -> dict[str, Any]:
    """Build and evaluate a genuinely inclusive one-channel/one-bin control.

    Expected yields are summed across all folds and score bins. Independent MC
    variances are added in quadrature before the same staterror/normsys model
    used by the shape workspace is constructed.
    """

    if not channels:
        raise ValueError("at least one shape channel is required")
    signal_yield = float(
        sum(np.sum(np.asarray(channel["signal"], dtype=float)) for channel in channels)
    )
    background_yield = float(
        sum(
            np.sum(np.asarray(channel["background"], dtype=float))
            for channel in channels
        )
    )
    signal_variance = float(
        sum(
            np.sum(np.square(np.asarray(channel["signal_staterror"], dtype=float)))
            for channel in channels
        )
    )
    background_variance = float(
        sum(
            np.sum(np.square(np.asarray(channel["background_staterror"], dtype=float)))
            for channel in channels
        )
    )
    signal_raw = int(
        sum(np.sum(np.asarray(channel.get("signal_raw_entries", []), dtype=int)) for channel in channels)
    )
    background_raw = int(
        sum(
            np.sum(np.asarray(channel.get("background_raw_entries", []), dtype=int))
            for channel in channels
        )
    )
    channel = {
        "name": "inclusive_one_bin",
        "edges": [0.0, 1.0],
        "signal": np.asarray([signal_yield], dtype=float),
        "background": np.asarray([background_yield], dtype=float),
        "signal_staterror": np.asarray([math.sqrt(signal_variance)], dtype=float),
        "background_staterror": np.asarray(
            [math.sqrt(background_variance)], dtype=float
        ),
        "signal_raw_entries": np.asarray([signal_raw], dtype=int),
        "background_raw_entries": np.asarray([background_raw], dtype=int),
        "signal_effective_entries": np.asarray(
            [
                signal_yield * signal_yield / signal_variance
                if signal_variance > 0.0
                else 0.0
            ]
        ),
        "background_effective_entries": np.asarray(
            [
                background_yield * background_yield / background_variance
                if background_variance > 0.0
                else 0.0
            ]
        ),
    }
    workspace_spec = build_workspace([channel], config)
    return {
        "channel": channel,
        "workspace": workspace_spec,
        "limit": _expected_limit_summary(
            workspace_spec,
            data_label="Background-only Asimov inclusive control",
            config=config,
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (TutorialConfig, SampleSpec, BinningConfig, BackgroundNormsysConfig)):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} as JSON")


def _write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    destination.write_text(encoded + "\n", encoding="utf-8")
    return destination


def _write_event_scores(
    path: str | Path,
    crossfit: Mapping[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = crossfit["arrays"]
    fields = [
        "sample",
        "label",
        "event_index",
        "source_entry_index",
        "fold",
        "test_rotation",
        "validation_rotation",
        "raw_weight",
        "physical_weight",
        "unit_xsec_weight",
        "oof_score",
        "validation_score",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(arrays["labels"])):
            fold = int(arrays["folds"][index])
            writer.writerow(
                {
                    "sample": arrays["sample_names"][index],
                    "label": int(arrays["labels"][index]),
                    "event_index": int(arrays["event_indices"][index]),
                    "source_entry_index": int(arrays["source_entry_indices"][index]),
                    "fold": fold,
                    "test_rotation": fold,
                    "validation_rotation": (fold - 1) % 5,
                    "raw_weight": f"{float(arrays['raw_weights'][index]):.17g}",
                    "physical_weight": (
                        f"{float(arrays['physical_weights'][index]):.17g}"
                    ),
                    "unit_xsec_weight": (
                        f"{float(arrays['unit_xsec_weights'][index]):.17g}"
                    ),
                    "oof_score": f"{float(crossfit['oof_scores'][index]):.17g}",
                    "validation_score": (
                        f"{float(crossfit['validation_scores'][index]):.17g}"
                    ),
                }
            )
    return destination


def _package_versions() -> dict[str, Any]:
    packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "xgboost",
        "pyhf",
        "iminuit",
        "matplotlib",
        "mplhep",
    ]
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "method_version": TUTORIAL_METHOD_VERSION,
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        root = importlib.import_module("ROOT")
        versions["ROOT"] = str(root.gROOT.GetVersion())
    except (ImportError, AttributeError):
        versions["ROOT"] = None
    return versions


def _serializable_crossfit(crossfit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method_version": crossfit["method_version"],
        "feature_names": crossfit["feature_names"],
        "rotations": [
            {
                key: value
                for key, value in rotation.items()
                if key not in {"train_mask", "validation_mask", "test_mask"}
            }
            for rotation in crossfit["rotations"]
        ],
    }


def save_tutorial_artifacts(
    *,
    config: TutorialConfig,
    samples: Sequence[LoadedSample],
    crossfit: Mapping[str, Any],
    binning: Mapping[str, Any],
    channels: Sequence[Mapping[str, Any]],
    workspace: Mapping[str, Any],
    fit_results: Mapping[str, Any],
    output_dir: str | Path,
    one_bin_workspace: Mapping[str, Any] | None = None,
    one_bin_control: Mapping[str, Any] | None = None,
    mcstat_only_workspace: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Save all machine-readable inputs, templates, scores, and fit results."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": _write_json(output / "config.resolved.json", config),
        "input_hashes": _write_json(
            output / "input_hashes.json",
            {
                sample.spec.name: {
                    "root_path": sample.spec.root_path,
                    "summary_path": sample.spec.summary_path,
                    "sha256_root": sample.hashes["root"],
                    "sha256_summary": sample.hashes["summary"],
                    "file_contract": sample.file_contract,
                    "sidecar": sample.summary,
                    "total_weight_in": sample.total_weight_in,
                    "rate_factor": sample.rate_factor,
                    "generation_integration_error_fraction": (
                        sample.spec.generation_integration_error_fraction
                    ),
                    "generation_integration_error_used_as_nuisance": False,
                }
                for sample in samples
            },
        ),
        "versions": _write_json(output / "versions.json", _package_versions()),
        "crossfit": _write_json(
            output / "crossfit_training.json", _serializable_crossfit(crossfit)
        ),
        "event_scores": _write_event_scores(output / "event_scores.csv", crossfit),
        "binning": _write_json(output / "binning.json", binning),
        "channels": _write_json(
            output / "channel_templates.json",
            [_channel_as_json(channel) for channel in channels],
        ),
        "workspace": _write_json(output / "workspace.json", workspace),
        "fit_results": _write_json(output / "fit_results.json", fit_results),
    }
    if one_bin_workspace is not None:
        paths["workspace_one_bin"] = _write_json(
            output / "workspace_one_bin.json", one_bin_workspace
        )
    if one_bin_control is not None:
        paths["one_bin_control"] = _write_json(
            output / "one_bin_control.json", one_bin_control
        )
    if mcstat_only_workspace is not None:
        paths["workspace_mcstat_only"] = _write_json(
            output / "workspace_mcstat_only.json", mcstat_only_workspace
        )
    return {name: str(path) for name, path in paths.items()}


def _plotting() -> tuple[Any, Any]:
    # Some fontTools versions emit harmless font-subsetting warnings for
    # Matplotlib's mathtext glyphs. They add pages of noise to an executed
    # notebook without indicating a malformed PDF.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    try:
        import matplotlib as mpl

        # The tutorial always writes files and then displays those files in the
        # notebook. A non-interactive backend is therefore deterministic and
        # avoids macOS GUI-backend crashes during nbconvert and pytest.
        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment dependent.
        raise RuntimeError("matplotlib is required to make tutorial plots") from error
    try:
        import mplhep

        mplhep.style.use("CMS")
    except ImportError:
        pass
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return mpl, plt


def _save_figure(fig: Any, stem: str | Path) -> dict[str, str]:
    path = Path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    png = path.with_suffix(".png")
    # Homebrew's current fontTools build prints harmless mathtext-subsetting
    # diagnostics directly to stderr while producing valid vector PDFs. Keep
    # those implementation messages out of the pedagogical notebook; actual
    # save failures still raise normally.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
    return {"pdf": str(pdf.resolve()), "png": str(png.resolve())}


def _analysis_stamp(axis: Any, config: TutorialConfig, *, extra: str = "") -> None:
    text = (
        rf"$\sqrt{{s}}={config.sqrt_s_tev:g}$ TeV, "
        rf"$\mathcal{{L}}={config.luminosity_fb:g}\,\mathrm{{fb}}^{{-1}}$"
    )
    if extra:
        text = f"{text}\n{extra}"
    axis.text(
        0.03,
        0.97,
        text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )


def plot_training_history(
    crossfit: Mapping[str, Any],
    config: TutorialConfig,
    stem: str | Path,
) -> dict[str, str]:
    _, plt = _plotting()
    fig, axis = plt.subplots(figsize=(7.0, 5.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(crossfit["rotations"])))
    for color, rotation in zip(colors, crossfit["rotations"]):
        history = rotation["history"]
        train = history.get("validation_0", {}).get("logloss", [])
        validation = history.get("validation_1", {}).get("logloss", [])
        epochs = np.arange(1, len(train) + 1)
        axis.plot(
            epochs,
            train,
            color=color,
            lw=1.4,
            alpha=0.65,
            label=f"fold {rotation['rotation']} train",
        )
        axis.plot(
            epochs,
            validation,
            color=color,
            lw=2.0,
            ls="--",
            label=f"fold {rotation['rotation']} validation",
        )
    axis.set_xlabel("Boosting iteration")
    axis.set_ylabel("Weighted binary log loss")
    axis.set_title("XGBoost training history")
    axis.legend(ncol=2, fontsize=8)
    _analysis_stamp(axis, config, extra="3/1/1 deterministic cross-fit")
    result = _save_figure(fig, stem)
    plt.close(fig)
    return result


def plot_weighted_roc(
    crossfit: Mapping[str, Any],
    config: TutorialConfig,
    stem: str | Path,
) -> dict[str, str]:
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for the ROC plot") from error
    _, plt = _plotting()
    labels = np.asarray(crossfit["arrays"]["labels"], dtype=int)
    weights = np.abs(
        np.asarray(crossfit["arrays"]["physical_weights"], dtype=float)
    )
    false_positive, true_positive, _ = roc_curve(
        labels,
        np.asarray(crossfit["oof_scores"], dtype=float),
        sample_weight=weights,
    )
    area = float(auc(false_positive, true_positive))
    fig, axis = plt.subplots(figsize=(6.2, 5.5))
    axis.plot(false_positive, true_positive, lw=2.2, label=f"OOF XGBoost (AUC={area:.3f})")
    axis.plot([0, 1], [0, 1], color="0.45", ls="--", lw=1.2, label="Random")
    axis.set(xlabel="Background efficiency", ylabel="Signal efficiency", xlim=(0, 1), ylim=(0, 1))
    axis.set_title(r"Weighted ROC: SM $hhhh$ vs. $gg\to8b$")
    axis.legend(loc="lower right")
    _analysis_stamp(axis, config, extra=r"evaluation weights: $|w_\mathrm{phys}|$")
    result = _save_figure(fig, stem)
    plt.close(fig)
    return result


def plot_feature_importance(
    crossfit: Mapping[str, Any],
    config: TutorialConfig,
    stem: str | Path,
    *,
    maximum_features: int = 15,
) -> dict[str, str]:
    _, plt = _plotting()
    names = list(crossfit["feature_names"])
    fold_values = []
    for model in crossfit["models"]:
        scores = model.get_booster().get_score(importance_type="gain")
        fold_values.append(
            np.asarray(
                [float(scores.get(name, scores.get(f"f{index}", 0.0))) for index, name in enumerate(names)]
            )
        )
    importance = np.mean(fold_values, axis=0)
    if float(np.sum(importance)) > 0.0:
        importance /= float(np.sum(importance))
    selected = np.argsort(importance)[-int(maximum_features) :]
    fig, axis = plt.subplots(figsize=(7.2, 6.1))
    axis.barh(np.arange(len(selected)), importance[selected], color="#3b78a8")
    axis.set_yticks(np.arange(len(selected)), [names[index] for index in selected])
    axis.set_xlabel("Mean normalized gain")
    axis.set_title("Cross-fold XGBoost feature importance")
    _analysis_stamp(axis, config)
    result = _save_figure(fig, stem)
    plt.close(fig)
    return result


def _histogram_summary(
    scores: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.histogram(scores, bins=edges, weights=weights)[0].astype(float)
    variance = np.histogram(scores, bins=edges, weights=np.square(weights))[0].astype(float)
    return values, variance


def plot_score_distribution(
    crossfit: Mapping[str, Any],
    config: TutorialConfig,
    stem: str | Path,
    *,
    normalized: bool,
) -> dict[str, str]:
    _, plt = _plotting()
    labels = np.asarray(crossfit["arrays"]["labels"], dtype=int)
    scores = np.asarray(crossfit["oof_scores"], dtype=float)
    physical = np.asarray(crossfit["arrays"]["physical_weights"], dtype=float)
    unit_xsec = np.asarray(crossfit["arrays"]["unit_xsec_weights"], dtype=float)
    edges = np.linspace(0.0, 1.0, 31)
    centers = 0.5 * (edges[:-1] + edges[1:])
    signal_weights = physical[labels == 1] if normalized else unit_xsec[labels == 1]
    signal, signal_variance = _histogram_summary(
        scores[labels == 1], signal_weights, edges
    )
    background, background_variance = _histogram_summary(
        scores[labels == 0], physical[labels == 0], edges
    )
    if normalized:
        for values, variance in (
            (signal, signal_variance),
            (background, background_variance),
        ):
            total = float(np.sum(values))
            if total <= 0.0:
                raise ValueError("cannot normalize a nonpositive signed score template")
            values /= total
            variance /= total * total
    fig, axis = plt.subplots(figsize=(7.0, 5.3))
    axis.stairs(background, edges, color="#2b6fbb", lw=2.0, label=r"$gg\to8b$")
    axis.fill_between(
        centers,
        background - np.sqrt(background_variance),
        background + np.sqrt(background_variance),
        step="mid",
        color="#2b6fbb",
        alpha=0.2,
        linewidth=0,
    )
    signal_label = (
        r"SM $hhhh\to8b$"
        if normalized
        else r"$hhhh\to8b$ ($\sigma_{hhhh}=1$ fb template)"
    )
    axis.stairs(signal, edges, color="#c83e4d", lw=2.2, label=signal_label)
    axis.fill_between(
        centers,
        signal - np.sqrt(signal_variance),
        signal + np.sqrt(signal_variance),
        step="mid",
        color="#c83e4d",
        alpha=0.2,
        linewidth=0,
    )
    axis.set_xlabel("Out-of-fold XGBoost signal score")
    axis.set_ylabel("Normalized events" if normalized else "Expected events / bin")
    axis.set_title(
        "Classifier shape comparison" if normalized else "Expected score templates"
    )
    if not normalized and np.all(signal > 0.0) and np.all(background > 0.0):
        axis.set_yscale("log")
    axis.legend(loc="best")
    _analysis_stamp(axis, config, extra="signed MC yields; bands are MC statistical")
    result = _save_figure(fig, stem)
    plt.close(fig)
    return result


def plot_unrolled_fit(
    channels: Sequence[Mapping[str, Any]],
    fit_results: Mapping[str, Any],
    config: TutorialConfig,
    stem: str | Path,
) -> dict[str, str]:
    _, plt = _plotting()
    injection = fit_results["injected_asimov"]
    fit = injection["fit"]
    ordered = sorted(channels, key=lambda item: str(item["name"]))
    signal = np.concatenate([np.asarray(item["signal"], dtype=float) for item in ordered])
    background = np.concatenate(
        [np.asarray(item["background"], dtype=float) for item in ordered]
    )
    observation_map = injection["observations_by_channel"]
    observations = np.concatenate(
        [np.asarray(observation_map[str(item["name"])], dtype=float) for item in ordered]
    )
    # The nominal pre-fit reference is the background-only expectation stored
    # in the workspace. ``observations`` is the full deterministic generating
    # expectation and therefore includes both the injected signal and the
    # +0.5-sigma gg8b_norm displacement. Keeping these as separate curves makes
    # the fit evolution explicit and avoids silently omitting that displacement.
    prefit = background
    generated = observations
    postfit = np.asarray(fit["expected_postfit"], dtype=float)
    postfit_error = fit.get("expected_postfit_uncertainty")
    if postfit_error is None:
        postfit_error_array = np.zeros_like(postfit)
    else:
        postfit_error_array = np.asarray(postfit_error, dtype=float)
    x = np.arange(len(prefit))

    fig, (axis, ratio) = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )
    prefit_artist = axis.stairs(
        prefit,
        np.arange(len(prefit) + 1) - 0.5,
        color="#2b6fbb",
        lw=1.8,
        label="Nominal pre-fit b only",
    )
    postfit_artist = axis.stairs(
        postfit,
        np.arange(len(postfit) + 1) - 0.5,
        color="#2a9d63",
        lw=2.2,
        label="Post-fit total",
    )
    uncertainty_artist = axis.fill_between(
        x,
        postfit - postfit_error_array,
        postfit + postfit_error_array,
        step="mid",
        color="#2a9d63",
        alpha=0.25,
        linewidth=0,
        label="Post-fit uncertainty",
    )
    # Draw the injected truth after the post-fit curve: for a successful
    # same-model Asimov closure they coincide, so the dashed overlay makes
    # their agreement visible instead of being hidden underneath the fit.
    generated_artist = axis.stairs(
        generated,
        np.arange(len(generated) + 1) - 0.5,
        color="#e09f3e",
        lw=1.8,
        ls="--",
        label="Injected truth: signal + shifted b",
        zorder=4,
    )
    asimov_artist = axis.errorbar(
        x,
        observations,
        fmt="o",
        ms=3.5,
        color="black",
        label="Injected Asimov bins (NOT DATA)",
        zorder=5,
    )
    axis.set_ylabel("Events / score bin")
    axis.set_title("Unrolled cross-fit score likelihood")
    vertical_max = max(
        float(np.max(generated)),
        float(np.max(postfit + postfit_error_array)),
        float(np.max(prefit)),
    )
    axis.set_ylim(0.0, 1.28 * vertical_max)
    axis.legend(
        handles=[
            prefit_artist,
            generated_artist,
            postfit_artist,
            uncertainty_artist,
            asimov_artist,
        ],
        loc="upper right",
        ncol=2,
        fontsize=8.5,
        frameon=False,
    )
    _analysis_stamp(axis, config, extra="Injected teaching diagnostic — NOT DATA")
    denominator = np.where(postfit > 0.0, postfit, np.nan)
    ratio.errorbar(
        x,
        observations / denominator,
        fmt="o",
        ms=3.0,
        color="black",
    )
    ratio.fill_between(
        x,
        1.0 - postfit_error_array / denominator,
        1.0 + postfit_error_array / denominator,
        step="mid",
        color="#2a9d63",
        alpha=0.25,
        linewidth=0,
    )
    ratio.axhline(1.0, color="0.3", lw=1.0)
    ratio.set_ylim(0.45, 1.55)
    ratio.set_ylabel("Asimov/fit")
    ratio.set_xlabel("Unrolled fold / XGBoost-score bin")
    offset = 0
    tick_positions = []
    tick_labels = []
    for item in ordered:
        bins = len(item["signal"])
        tick_positions.append(offset + 0.5 * (bins - 1))
        tick_labels.append(str(item["name"]).replace("_", " "))
        offset += bins
        if offset < len(signal):
            axis.axvline(offset - 0.5, color="0.75", lw=0.8)
            ratio.axvline(offset - 0.5, color="0.75", lw=0.8)
    ratio.set_xticks(tick_positions, tick_labels)
    result = _save_figure(fig, stem)
    plt.close(fig)
    return result


def plot_likelihood_scan(
    fit_results: Mapping[str, Any],
    config: TutorialConfig,
    stem: str | Path,
) -> dict[str, str]:
    _, plt = _plotting()
    profile = fit_results["injected_asimov"]["profile_likelihood"]
    x = np.asarray(profile["sigma_hhhh_fb"], dtype=float)
    y = np.asarray(profile["delta_nll"], dtype=float)
    fig, axis = plt.subplots(figsize=(6.8, 5.2))
    axis.plot(x, y, color="#3b78a8", lw=2.2)
    axis.axhline(0.5, color="0.45", ls="--", lw=1.1, label=r"$68\%$ (Wilks)")
    axis.axhline(1.92, color="0.45", ls=":", lw=1.1, label=r"$95\%$ (Wilks)")
    axis.axvline(
        fit_results["injected_asimov"]["sigma_hhhh_fb"],
        color="#c83e4d",
        ls="--",
        lw=1.3,
        label="Injected value",
    )
    axis.set(xlabel=r"$\sigma_{hhhh}$ [fb]", ylabel=r"$\Delta(-\ln\mathcal{L})$")
    axis.set_ylim(bottom=0.0)
    axis.set_title("Profile-likelihood scan")
    axis.legend()
    _analysis_stamp(axis, config, extra="Injected Asimov — NOT DATA")
    result = _save_figure(fig, stem)
    plt.close(fig)
    return result


def plot_cls_scan(
    fit_results: Mapping[str, Any],
    config: TutorialConfig,
    stem: str | Path,
) -> dict[str, str]:
    _, plt = _plotting()
    scan = fit_results["background_only_cls_scan"]
    x = np.asarray(scan["sigma_hhhh_fb"], dtype=float)
    observed = np.asarray(scan["observed_cls"], dtype=float)
    expected = np.asarray(scan["expected_cls"], dtype=float)
    fig, axis = plt.subplots(figsize=(6.8, 5.2))
    axis.fill_between(x, expected[:, 0], expected[:, 4], color="#f1c40f", alpha=0.45, label=r"Expected $\pm2\sigma$")
    axis.fill_between(x, expected[:, 1], expected[:, 3], color="#57b65f", alpha=0.55, label=r"Expected $\pm1\sigma$")
    axis.plot(x, expected[:, 2], color="black", ls="--", lw=1.8, label="Expected median")
    axis.plot(x, observed, color="#2b6fbb", lw=1.8, label="Background-Asimov curve")
    axis.axhline(0.05, color="#c83e4d", ls=":", lw=1.4, label=r"$CL_s=0.05$")
    axis.set(
        xlabel=r"$\sigma_{hhhh}$ [fb]",
        ylabel=r"$CL_s$",
        ylim=(0.0, 1.02),
    )
    axis.set_title(r"Asymptotic $CL_s$ scan")
    axis.legend(fontsize=9)
    _analysis_stamp(axis, config, extra="background-only Asimov expectation")
    result = _save_figure(fig, stem)
    plt.close(fig)
    return result


def _selected_nuisance_indices(
    fit: Mapping[str, Any],
    config: TutorialConfig,
    *,
    maximum: int = 16,
) -> list[int]:
    names = list(fit["parameter_names"])
    pulls = np.asarray(fit["pulls"], dtype=float)
    candidates = [
        index
        for index, name in enumerate(names)
        if name != POI_NAME and math.isfinite(float(pulls[index]))
    ]
    candidates.sort(key=lambda index: abs(float(pulls[index])), reverse=True)
    normsys = [
        index for index, name in enumerate(names) if name == config.background_normsys.name
    ]
    selected = normsys + [index for index in candidates if index not in normsys]
    return selected[:maximum]


def plot_nuisance_pulls(
    fit_results: Mapping[str, Any],
    config: TutorialConfig,
    stem: str | Path,
) -> dict[str, str]:
    _, plt = _plotting()
    fit = fit_results["injected_asimov"]["fit"]
    indices = _selected_nuisance_indices(fit, config)
    names = [fit["parameter_names"][index] for index in indices]
    pulls = np.asarray(fit["pulls"], dtype=float)[indices]
    errors_raw = fit.get("pull_uncertainties")
    errors = (
        None
        if errors_raw is None
        else np.asarray(errors_raw, dtype=float)[indices]
    )
    fig, axis = plt.subplots(figsize=(8.0, max(4.5, 0.34 * len(indices) + 1.8)))
    y = np.arange(len(indices))
    axis.axvspan(-2.0, 2.0, color="#f1c40f", alpha=0.18)
    axis.axvspan(-1.0, 1.0, color="#57b65f", alpha=0.22)
    axis.axvline(0.0, color="0.35", lw=1.0)
    axis.errorbar(
        pulls,
        y,
        xerr=errors,
        fmt="o",
        color="#2b6fbb",
        capsize=2,
    )
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    axis.set_xlabel("Fitted shift / nominal constraint width")
    axis.set_title("Largest fitted nuisance shifts")
    _analysis_stamp(axis, config, extra="Injected Asimov — NOT DATA")
    result = _save_figure(fig, stem)
    plt.close(fig)
    return result


def plot_reduced_correlation(
    fit_results: Mapping[str, Any],
    config: TutorialConfig,
    stem: str | Path,
) -> dict[str, str] | None:
    _, plt = _plotting()
    fit = fit_results["injected_asimov"]["fit"]
    correlation_raw = fit.get("correlation_matrix")
    if correlation_raw is None:
        return None
    names = list(fit["parameter_names"])
    poi_index = names.index(POI_NAME)
    full_correlation = np.asarray(correlation_raw, dtype=float)
    normsys_indices = [
        index
        for index, name in enumerate(names)
        if name == config.background_normsys.name
    ]
    reserved = {poi_index, *normsys_indices}
    mcstat = [
        index
        for index, name in enumerate(names)
        if index not in reserved and "mcstat" in name
    ]
    mcstat.sort(
        key=lambda index: abs(float(full_correlation[poi_index, index])),
        reverse=True,
    )
    selected = [poi_index, *normsys_indices, *mcstat[:8]]
    correlation = full_correlation[np.ix_(selected, selected)]
    labels = [names[index] for index in selected]
    fig, axis = plt.subplots(figsize=(8.0, 7.0))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title("Reduced post-fit parameter correlation")
    fig.colorbar(image, ax=axis, label="Correlation coefficient")
    result = _save_figure(fig, stem)
    plt.close(fig)
    return result


def make_tutorial_plots(
    result: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create publication-style PDF and 300-dpi PNG pairs for every stage."""

    output = (
        Path(result.get("output_dir", DEFAULT_OUTPUT_DIR)) / "plots"
        if output_dir is None
        else Path(output_dir)
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = result["config"]
    crossfit = result["crossfit"]
    channels = result["channels"]
    fit_results = result["fit_results"]
    plots: dict[str, Any] = {
        "training_history": plot_training_history(
            crossfit, config, output / "training_history"
        ),
        "weighted_roc": plot_weighted_roc(
            crossfit, config, output / "weighted_roc"
        ),
        "feature_importance": plot_feature_importance(
            crossfit, config, output / "feature_importance"
        ),
        "score_normalized": plot_score_distribution(
            crossfit, config, output / "score_normalized", normalized=True
        ),
        "score_expected_yields": plot_score_distribution(
            crossfit,
            config,
            output / "score_expected_yields",
            normalized=False,
        ),
        "unrolled_fit": plot_unrolled_fit(
            channels, fit_results, config, output / "unrolled_prefit_postfit"
        ),
        "likelihood_scan": plot_likelihood_scan(
            fit_results, config, output / "profile_likelihood"
        ),
        "cls_scan": plot_cls_scan(
            fit_results, config, output / "expected_cls"
        ),
        "nuisance_pulls": plot_nuisance_pulls(
            fit_results, config, output / "nuisance_pulls"
        ),
    }
    correlation = plot_reduced_correlation(
        fit_results, config, output / "reduced_correlation"
    )
    if correlation is not None:
        plots["reduced_correlation"] = correlation
    return plots


def run_tutorial(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    *,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Execute the complete ROOT -> XGBoost -> pyhf tutorial pipeline.

    The returned dictionary deliberately contains live arrays and fitted model
    objects for notebook inspection. JSON/CSV/model artifacts are written to
    ``output_dir`` independently.
    """

    config = load_config(config_path)
    output = (
        DEFAULT_OUTPUT_DIR.resolve()
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    signal, background = load_samples(config)
    crossfit = crossfit_xgboost(
        signal,
        background,
        config,
        model_dir=output / "fold_models",
    )
    binning = select_binning_and_build_channels(crossfit, config)
    channels = binning["channels"]
    workspace = build_workspace(channels, config)
    fit_results = run_pyhf_inference(workspace, config)
    mcstat_only_workspace = build_workspace(
        channels,
        config,
        include_background_normsys=False,
    )
    mcstat_only_limit = _expected_limit_summary(
        mcstat_only_workspace,
        data_label=(
            "Background-only Asimov five-channel comparison "
            "(MC statistical uncertainties only)"
        ),
        config=config,
    )
    fit_results["shape_mcstat_only"] = mcstat_only_limit
    one_bin_bundle = build_one_bin_control(channels, config)
    one_bin_workspace = one_bin_bundle["workspace"]
    one_bin_control = {
        "channel": one_bin_bundle["channel"],
        "limit": one_bin_bundle["limit"],
    }
    fit_results["one_bin_control"] = one_bin_bundle["limit"]
    artifacts = save_tutorial_artifacts(
        config=config,
        samples=(signal, background),
        crossfit=crossfit,
        binning={"records": binning["records"]},
        channels=channels,
        workspace=workspace,
        fit_results=fit_results,
        output_dir=output,
        one_bin_workspace=one_bin_workspace,
        one_bin_control=one_bin_control,
        mcstat_only_workspace=mcstat_only_workspace,
    )
    result: dict[str, Any] = {
        "config": config,
        "samples": (signal, background),
        "crossfit": crossfit,
        "binning": binning,
        "channels": channels,
        "workspace": workspace,
        "mcstat_only_workspace": mcstat_only_workspace,
        "one_bin_workspace": one_bin_workspace,
        "one_bin_control": one_bin_control,
        "fit_results": fit_results,
        "artifacts": artifacts,
        "output_dir": str(output),
    }
    if make_plots:
        plots = make_tutorial_plots(result, output / "plots")
        result["plots"] = plots
        plots_manifest = _write_json(output / "plots.json", plots)
        artifacts["plots"] = str(plots_manifest)
    summary_path = _write_json(
        output / "summary.json",
        {
            "method_version": TUTORIAL_METHOD_VERSION,
            "output_dir": output,
            "headline_expected_limit": fit_results["headline_expected_limit"],
            "shape_mcstat_only_expected_limit": fit_results["shape_mcstat_only"],
            "one_bin_control_expected_limit": fit_results["one_bin_control"],
            "injected_asimov": {
                "label": fit_results["injected_asimov"]["label"],
                "is_observed_data": False,
                "sigma_hhhh_fb": fit_results["injected_asimov"]["sigma_hhhh_fb"],
                "fit_poi_hat_fb": fit_results["injected_asimov"]["fit"]["poi_hat_fb"],
            },
            "used_binning_fallback": [
                bool(record["used_fallback"]) for record in binning["records"]
            ],
            "artifacts": artifacts,
        },
    )
    artifacts["summary"] = str(summary_path)
    return result


__all__ = [
    "BackgroundNormsysConfig",
    "BinningConfig",
    "LoadedSample",
    "SampleSpec",
    "TutorialConfig",
    "build_workspace",
    "build_workspace_spec",
    "build_one_bin_control",
    "construct_channels",
    "crossfit_xgboost",
    "load_config",
    "load_sample",
    "load_samples",
    "make_tutorial_plots",
    "plot_cls_scan",
    "plot_feature_importance",
    "plot_likelihood_scan",
    "plot_nuisance_pulls",
    "plot_reduced_correlation",
    "plot_score_distribution",
    "plot_training_history",
    "plot_unrolled_fit",
    "plot_weighted_roc",
    "run_pyhf_inference",
    "run_tutorial",
    "save_tutorial_artifacts",
    "select_binning_and_build_channels",
    "sha256_file",
]
