"""Immutable observable and model contracts for the resolved-8b analysis.

The legacy and v2 trees deliberately have distinct schema identifiers even
when a v2 feature has the same human-readable name as a legacy feature.  In
particular, ``corrected28`` is a projection of ``extended-91-v2`` and must not
be used to make an untagged legacy model look compatible with the corrected
pairing semantics.

This module has no ROOT, NumPy, or XGBoost import-time dependency.  It is safe
to use from metadata tools and unit tests on machines without the HEP stack.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


LEGACY_SCHEMA_ID = "legacy-28-v1"
EXTENDED_SCHEMA_ID = "extended-91-v2"
EXTENDED_FILE_TAG = "-extended-v2"
PAIRING_COUNT = 105

MODEL_METADATA_ATTRIBUTE = "fourhiggs_model_metadata"
MODEL_CONTRACT_VERSION = 1


LEGACY_FEATURE_NAMES = (
    "bjet1_pt",
    "bjet2_pt",
    "bjet3_pt",
    "bjet4_pt",
    "bjet5_pt",
    "bjet6_pt",
    "bjet7_pt",
    "bjet8_pt",
    "m8b",
    "chi8",
    "delta_m_min",
    "delta_m_med1",
    "delta_m_med2",
    "delta_m_max",
    "higgs1_pt",
    "higgs2_pt",
    "higgs3_pt",
    "higgs4_pt",
    "dr_hh_12",
    "dr_hh_13",
    "dr_hh_14",
    "dr_hh_23",
    "dr_hh_24",
    "dr_hh_34",
    "dr_bb_h1",
    "dr_bb_h2",
    "dr_bb_h3",
    "dr_bb_h4",
)


EXTENDED_FEATURE_NAMES = LEGACY_FEATURE_NAMES + (
    "m_bb_h1",
    "m_bb_h2",
    "m_bb_h3",
    "m_bb_h4",
    "chi8_second",
    "delta_chi8",
    "n_pairings_chi8_lt60",
    "m_hh_12",
    "m_hh_13",
    "m_hh_14",
    "m_hh_23",
    "m_hh_24",
    "m_hh_34",
    "m_hhh_123",
    "m_hhh_124",
    "m_hhh_134",
    "m_hhh_234",
    "z_bb_h1",
    "z_bb_h2",
    "z_bb_h3",
    "z_bb_h4",
    "pt_4h_over_m_4h",
    "abs_y_4h",
    "ht_8b",
    "mean_m_bb",
    "std_m_bb",
    "max_abs_m_bb_minus_125",
    "abs_cos_theta_star_h1",
    "abs_cos_theta_star_h2",
    "abs_cos_theta_star_h3",
    "abs_cos_theta_star_h4",
    "abs_deta_bb_h1",
    "abs_deta_bb_h2",
    "abs_deta_bb_h3",
    "abs_deta_bb_h4",
    "abs_dphi_bb_h1",
    "abs_dphi_bb_h2",
    "abs_dphi_bb_h3",
    "abs_dphi_bb_h4",
    "min_dr_bpair_1",
    "min_dr_bpair_2",
    "min_dr_bpair_3",
    "min_dr_bpair_4",
    "min_m_bpair_1",
    "min_m_bpair_2",
    "min_m_bpair_3",
    "min_m_bpair_4",
    "higgs_rapidity_span",
    "abs_dy_hh_12",
    "abs_dy_hh_13",
    "abs_dy_hh_14",
    "abs_dy_hh_23",
    "abs_dy_hh_24",
    "abs_dy_hh_34",
    "abs_dphi_hh_12",
    "abs_dphi_hh_13",
    "abs_dphi_hh_14",
    "abs_dphi_hh_23",
    "abs_dphi_hh_24",
    "abs_dphi_hh_34",
    "centrality",
    "transverse_sphericity",
    "zness",
)


LEGACY_FEATURE_UNITS = (
    ("GeV",) * 18
    + ("dimensionless",) * 10
)

EXTENDED_FEATURE_UNITS = LEGACY_FEATURE_UNITS + (
    ("GeV",) * 6
    + ("count",)
    + ("GeV",) * 10
    + ("dimensionless",) * 6
    + ("GeV",)
    + ("GeV",) * 3
    + ("dimensionless",) * 8
    + ("rad",) * 4
    + ("dimensionless",) * 4
    + ("GeV",) * 4
    + ("dimensionless",) * 7
    + ("rad",) * 6
    + ("dimensionless",) * 2
    + ("GeV",)
)


if len(LEGACY_FEATURE_NAMES) != 28:  # pragma: no cover - import-time invariant.
    raise AssertionError("legacy observable contract must contain 28 features")
if len(EXTENDED_FEATURE_NAMES) != 91:  # pragma: no cover - import-time invariant.
    raise AssertionError("extended observable contract must contain 91 features")
if len(LEGACY_FEATURE_UNITS) != len(LEGACY_FEATURE_NAMES):  # pragma: no cover
    raise AssertionError("legacy unit contract does not match its feature contract")
if len(EXTENDED_FEATURE_UNITS) != len(EXTENDED_FEATURE_NAMES):  # pragma: no cover
    raise AssertionError("extended unit contract does not match its feature contract")


class ObservableSchemaError(ValueError):
    """Base exception for unknown or inconsistent observable contracts."""


class FeatureContractMismatch(ObservableSchemaError):
    """Raised before scoring data with a semantically incompatible model."""


class ModelContractError(FeatureContractMismatch):
    """Raised when persisted model metadata does not match the requested data."""


class LegacyModelWarning(UserWarning):
    """Warning emitted when a metadata-free 28-input legacy model is inferred."""


@dataclass(frozen=True)
class ObservableSchema:
    schema_id: str
    tree_name: str
    feature_branch: str
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    stored_value_count: int
    feature_offset: int
    weight_branch: str
    required_branches: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class FeatureContract:
    observable_schema: str
    feature_profile: str
    feature_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


_LEGACY_SCHEMA = ObservableSchema(
    schema_id=LEGACY_SCHEMA_ID,
    tree_name="Data2",
    feature_branch="variables",
    feature_names=LEGACY_FEATURE_NAMES,
    feature_units=LEGACY_FEATURE_UNITS,
    stored_value_count=29,
    feature_offset=1,
    weight_branch="weight",
    # Old Data2 files predate the redundant scalar weight branch.  Their
    # variables[0] value remains the compatibility fallback.
    required_branches=("variables",),
)

_EXTENDED_SCHEMA = ObservableSchema(
    schema_id=EXTENDED_SCHEMA_ID,
    tree_name="Data3",
    feature_branch="features",
    feature_names=EXTENDED_FEATURE_NAMES,
    feature_units=EXTENDED_FEATURE_UNITS,
    stored_value_count=91,
    feature_offset=0,
    weight_branch="weight",
    required_branches=(
        "features",
        "weight",
        "event_index",
        "cut_mask",
        "passes_legacy_full_selection",
    ),
)


SCHEMA_REGISTRY: Mapping[str, ObservableSchema] = MappingProxyType(
    {
        LEGACY_SCHEMA_ID: _LEGACY_SCHEMA,
        EXTENDED_SCHEMA_ID: _EXTENDED_SCHEMA,
    }
)

FEATURE_PROFILE_INDICES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "corrected28": tuple(range(28)),
        "core52": tuple(range(52)),
        "full91": tuple(range(91)),
    }
)


def get_schema(observable_set: str) -> ObservableSchema:
    """Return an immutable schema, rejecting aliases to keep files explicit."""

    try:
        return SCHEMA_REGISTRY[str(observable_set)]
    except KeyError as exc:
        allowed = ", ".join(SCHEMA_REGISTRY)
        raise ObservableSchemaError(
            f"Unknown observable schema {observable_set!r}; expected one of: {allowed}"
        ) from exc


def get_feature_contract(
    observable_set: str,
    feature_profile: str | None = None,
) -> FeatureContract:
    """Return the exact ordered feature projection for a schema/profile pair."""

    schema = get_schema(observable_set)
    if schema.schema_id == LEGACY_SCHEMA_ID:
        profile = "corrected28" if feature_profile is None else str(feature_profile)
        if profile != "corrected28":
            raise ObservableSchemaError(
                f"Feature profile {profile!r} is not available for {LEGACY_SCHEMA_ID}; "
                "use 'corrected28' (the 28 stored legacy inputs)"
            )
        indices = tuple(range(28))
    else:
        profile = "full91" if feature_profile is None else str(feature_profile)
        try:
            indices = FEATURE_PROFILE_INDICES[profile]
        except KeyError as exc:
            allowed = ", ".join(FEATURE_PROFILE_INDICES)
            raise ObservableSchemaError(
                f"Unknown feature profile {profile!r}; expected one of: {allowed}"
            ) from exc

    return FeatureContract(
        observable_schema=schema.schema_id,
        feature_profile=profile,
        feature_indices=indices,
        feature_names=tuple(schema.feature_names[index] for index in indices),
        feature_units=tuple(schema.feature_units[index] for index in indices),
    )


def get_feature_profile(
    feature_profile: str,
    observable_set: str = EXTENDED_SCHEMA_ID,
) -> FeatureContract:
    """Convenience wrapper returning a named feature profile contract."""

    return get_feature_contract(observable_set, feature_profile)


def validate_feature_contract(
    observable_set: str,
    feature_names: Sequence[str],
    feature_profile: str | None = None,
    *,
    feature_units: Sequence[str] | None = None,
) -> FeatureContract:
    """Validate count, order, names, and optionally units; return the contract."""

    contract = get_feature_contract(observable_set, feature_profile)
    names = tuple(str(name) for name in feature_names)
    if names != contract.feature_names:
        if len(names) != contract.feature_count:
            detail = f"feature count {len(names)} != {contract.feature_count}"
        else:
            mismatch = next(
                index
                for index, (actual, expected) in enumerate(zip(names, contract.feature_names))
                if actual != expected
            )
            detail = (
                f"feature {mismatch} is {names[mismatch]!r}, expected "
                f"{contract.feature_names[mismatch]!r}"
            )
        raise FeatureContractMismatch(
            f"{contract.observable_schema}/{contract.feature_profile} mismatch: {detail}"
        )

    if feature_units is not None:
        units = tuple(str(unit) for unit in feature_units)
        if units != contract.feature_units:
            if len(units) != contract.feature_count:
                detail = f"unit count {len(units)} != {contract.feature_count}"
            else:
                mismatch = next(
                    index
                    for index, (actual, expected) in enumerate(zip(units, contract.feature_units))
                    if actual != expected
                )
                detail = (
                    f"unit {mismatch} is {units[mismatch]!r}, expected "
                    f"{contract.feature_units[mismatch]!r}"
                )
            raise FeatureContractMismatch(
                f"{contract.observable_schema}/{contract.feature_profile} mismatch: {detail}"
            )
    return contract


def strip_extended_v2_tag(path_or_name: str | Path) -> str:
    """Strip only the v2 analysis tag from the final path component."""

    path = Path(path_or_name)
    canonical_name = path.name.replace(EXTENDED_FILE_TAG, "")
    if str(path.parent) in ("", "."):
        return canonical_name
    return str(path.with_name(canonical_name))


_ANALYSIS_SUFFIXES = (
    "_var.smearCMS.root",
    ".analysis_summary.json",
    ".smearCMS.dat",
    ".smearCMS.root",
    ".root",
    ".input",
    ".out",
    ".log",
    ".top",
    ".evp2",
    ".evp",
)


def canonical_sample_basename(path_or_name: str | Path) -> str:
    """Return a tag- and analysis-suffix-free sample basename.

    For example, ``sample-extended-v2_var.smearCMS.root`` and
    ``sample_var.smearCMS.root`` both map to ``sample``.  This is the key used
    to locate the same sample's Herwig ``.out`` and analysis-summary metadata.
    """

    name = Path(strip_extended_v2_tag(Path(path_or_name).name)).name
    for suffix in _ANALYSIS_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


canonical_sample_id = canonical_sample_basename


def feature_contract_sha256(contract: FeatureContract) -> str:
    """Return a stable digest of all fields that determine model semantics."""

    payload = {
        "observable_schema": contract.observable_schema,
        "feature_profile": contract.feature_profile,
        "feature_names": list(contract.feature_names),
        "feature_units": list(contract.feature_units),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_model_metadata(
    observable_set: str,
    feature_profile: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    """Build JSON-serializable metadata containing the immutable ML contract."""

    contract = get_feature_contract(observable_set, feature_profile)
    reserved = {
        "model_contract_version",
        "observable_schema",
        "observable_set",
        "feature_profile",
        "feature_count",
        "feature_names",
        "feature_units",
        "feature_contract_sha256",
    }
    overlap = reserved.intersection(details)
    if overlap:
        raise ValueError(f"Cannot override reserved model metadata: {', '.join(sorted(overlap))}")

    metadata: dict[str, Any] = {
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "observable_schema": contract.observable_schema,
        # Keep the CLI spelling in manifests while making observable_schema
        # the authoritative persisted field.
        "observable_set": contract.observable_schema,
        "feature_profile": contract.feature_profile,
        "feature_count": contract.feature_count,
        "feature_names": list(contract.feature_names),
        "feature_units": list(contract.feature_units),
        "feature_contract_sha256": feature_contract_sha256(contract),
    }
    metadata.update(details)
    # Fail at construction, rather than when XGBoost tries to serialize it.
    json.dumps(metadata, sort_keys=True, allow_nan=False)
    return metadata


def _as_booster(model: Any) -> Any:
    if hasattr(model, "get_booster"):
        return model.get_booster()
    return model


def attach_model_metadata(
    model: Any,
    metadata: Mapping[str, Any] | None = None,
    *,
    observable_set: str | None = None,
    feature_profile: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    """Persist a contract as an XGBoost model attribute and return a copy."""

    if metadata is None:
        if observable_set is None:
            raise TypeError("observable_set is required when metadata is not supplied")
        payload = build_model_metadata(observable_set, feature_profile, **details)
    else:
        if observable_set is not None or feature_profile is not None or details:
            raise TypeError("supply either complete metadata or observable_set/details, not both")
        payload = dict(metadata)
        schema_id = payload.get("observable_schema", payload.get("observable_set"))
        validate_model_metadata(
            payload,
            str(schema_id),
            payload.get("feature_profile"),
        )

    contract = get_feature_contract(
        str(payload.get("observable_schema", payload.get("observable_set"))),
        payload.get("feature_profile"),
    )
    _validate_model_object_features(model, contract)
    booster = _as_booster(model)
    if not hasattr(booster, "set_attr"):
        raise TypeError("model does not expose XGBoost set_attr metadata support")
    booster.set_attr(
        **{
            MODEL_METADATA_ATTRIBUTE: json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        }
    )
    return dict(payload)


def _load_booster(path: str | Path) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("XGBoost is required to read model metadata from a file") from exc
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def read_model_metadata(model_or_path: Any) -> dict[str, Any] | None:
    """Read the embedded contract without imposing an expected schema."""

    model = _load_booster(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
    booster = _as_booster(model)
    if not hasattr(booster, "attr"):
        raise TypeError("model does not expose XGBoost attr metadata support")
    encoded = booster.attr(MODEL_METADATA_ATTRIBUTE)
    if encoded in (None, ""):
        return None
    try:
        metadata = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ModelContractError("model contains invalid fourhiggs metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise ModelContractError("model fourhiggs metadata must be a JSON object")
    return metadata


def validate_model_metadata(
    metadata: Mapping[str, Any],
    observable_set: str,
    feature_profile: str | None = None,
) -> FeatureContract:
    """Validate a decoded model payload against an expected data contract."""

    contract = get_feature_contract(observable_set, feature_profile)
    try:
        contract_version = int(metadata.get("model_contract_version"))
    except (TypeError, ValueError) as exc:
        raise ModelContractError("model metadata has no valid model_contract_version") from exc
    if contract_version != MODEL_CONTRACT_VERSION:
        raise ModelContractError(
            f"model contract version {contract_version} is not supported; expected "
            f"{MODEL_CONTRACT_VERSION}"
        )
    actual_schema = metadata.get("observable_schema", metadata.get("observable_set"))
    if actual_schema != contract.observable_schema:
        raise ModelContractError(
            f"model observable schema {actual_schema!r} does not match "
            f"{contract.observable_schema!r}"
        )
    if "observable_set" in metadata and metadata["observable_set"] != actual_schema:
        raise ModelContractError(
            "model observable_schema and observable_set metadata fields disagree"
        )
    actual_profile = metadata.get("feature_profile")
    if actual_profile != contract.feature_profile:
        raise ModelContractError(
            f"model feature profile {actual_profile!r} does not match "
            f"{contract.feature_profile!r}"
        )
    try:
        actual_count = int(metadata.get("feature_count"))
    except (TypeError, ValueError) as exc:
        raise ModelContractError("model metadata has no valid feature_count") from exc
    if actual_count != contract.feature_count:
        raise ModelContractError(
            f"model feature count {actual_count} does not match {contract.feature_count}"
        )
    try:
        validate_feature_contract(
            contract.observable_schema,
            metadata.get("feature_names", ()),
            contract.feature_profile,
            feature_units=metadata.get("feature_units", ()),
        )
    except FeatureContractMismatch as exc:
        raise ModelContractError(f"model {exc}") from exc
    expected_digest = feature_contract_sha256(contract)
    if metadata.get("feature_contract_sha256") != expected_digest:
        raise ModelContractError("model feature-contract digest is missing or inconsistent")
    return contract


def _model_feature_count(model: Any) -> int | None:
    booster = _as_booster(model)
    if hasattr(booster, "num_features"):
        try:
            return int(booster.num_features())
        except (TypeError, ValueError):
            pass
    if hasattr(model, "n_features_in_"):
        try:
            return int(model.n_features_in_)
        except (TypeError, ValueError):
            pass
    return None


def _model_feature_names(model: Any) -> tuple[str, ...] | None:
    booster = _as_booster(model)
    names = getattr(booster, "feature_names", None)
    if names is None:
        names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    return tuple(str(name) for name in names)


def _validate_model_object_features(model: Any, contract: FeatureContract) -> None:
    feature_count = _model_feature_count(model)
    if feature_count is not None and feature_count != contract.feature_count:
        raise ModelContractError(
            f"model object has {feature_count} inputs but its "
            f"{contract.observable_schema}/{contract.feature_profile} contract requires "
            f"{contract.feature_count}"
        )
    feature_names = _model_feature_names(model)
    if feature_names is not None and feature_names != contract.feature_names:
        try:
            validate_feature_contract(
                contract.observable_schema,
                feature_names,
                contract.feature_profile,
            )
        except FeatureContractMismatch as exc:
            raise ModelContractError(f"model object {exc}") from exc


def validate_model_contract(
    model_or_path: Any,
    observable_set: str,
    feature_profile: str | None = None,
    *,
    allow_metadata_free_legacy: bool = True,
) -> dict[str, Any]:
    """Refuse semantic mismatches, with one explicit legacy compatibility path."""

    model = _load_booster(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
    metadata = read_model_metadata(model)
    contract = get_feature_contract(observable_set, feature_profile)
    if metadata is not None:
        validate_model_metadata(metadata, contract.observable_schema, contract.feature_profile)
        _validate_model_object_features(model, contract)
        return dict(metadata)

    feature_count = _model_feature_count(model)
    if (
        allow_metadata_free_legacy
        and contract.observable_schema == LEGACY_SCHEMA_ID
        and contract.feature_profile == "corrected28"
        and feature_count == 28
    ):
        warnings.warn(
            "Loading a metadata-free 28-input model as legacy-28-v1. Its pairing "
            "semantics are legacy and it cannot be used with extended-91-v2 data.",
            LegacyModelWarning,
            stacklevel=2,
        )
        inferred = build_model_metadata(LEGACY_SCHEMA_ID, "corrected28")
        inferred["metadata_inferred"] = True
        return inferred

    if feature_count is None:
        detail = "its input feature count could not be determined"
    else:
        detail = f"it has {feature_count} input features"
    raise ModelContractError(
        f"model has no observable metadata and {detail}; only a 28-input "
        f"{LEGACY_SCHEMA_ID} model may use the legacy compatibility path"
    )


__all__ = [
    "EXTENDED_FEATURE_NAMES",
    "EXTENDED_FEATURE_UNITS",
    "EXTENDED_FILE_TAG",
    "EXTENDED_SCHEMA_ID",
    "FEATURE_PROFILE_INDICES",
    "FeatureContract",
    "FeatureContractMismatch",
    "LEGACY_FEATURE_NAMES",
    "LEGACY_FEATURE_UNITS",
    "LEGACY_SCHEMA_ID",
    "LegacyModelWarning",
    "MODEL_CONTRACT_VERSION",
    "MODEL_METADATA_ATTRIBUTE",
    "ModelContractError",
    "ObservableSchema",
    "ObservableSchemaError",
    "PAIRING_COUNT",
    "SCHEMA_REGISTRY",
    "attach_model_metadata",
    "build_model_metadata",
    "canonical_sample_basename",
    "canonical_sample_id",
    "feature_contract_sha256",
    "get_feature_contract",
    "get_feature_profile",
    "get_schema",
    "read_model_metadata",
    "strip_extended_v2_tag",
    "validate_feature_contract",
    "validate_model_contract",
    "validate_model_metadata",
]
