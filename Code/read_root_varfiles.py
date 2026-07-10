"""Read and validate variable ROOT files from the resolved-8b analysis.

The historical :func:`read_ROOT_varfile` call and its three-value return are
kept intact.  New keyword arguments select an immutable observable contract
and can return the original event identifiers required for cross-fitting.
PyROOT is imported opportunistically so schema and model metadata utilities
remain importable on machines without ROOT.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from observable_schemas import (
        EXTENDED_FEATURE_NAMES,
        EXTENDED_SCHEMA_ID,
        FeatureContractMismatch,
        LEGACY_FEATURE_NAMES,
        LEGACY_SCHEMA_ID,
        PAIRING_COUNT,
        canonical_sample_basename,
        canonical_sample_id,
        get_feature_contract,
        get_schema,
        strip_extended_v2_tag,
        validate_feature_contract,
    )
except ImportError:  # pragma: no cover - package-style import fallback.
    from .observable_schemas import (
        EXTENDED_FEATURE_NAMES,
        EXTENDED_SCHEMA_ID,
        FeatureContractMismatch,
        LEGACY_FEATURE_NAMES,
        LEGACY_SCHEMA_ID,
        PAIRING_COUNT,
        canonical_sample_basename,
        canonical_sample_id,
        get_feature_contract,
        get_schema,
        strip_extended_v2_tag,
        validate_feature_contract,
    )


try:  # PyROOT is optional until a ROOT file is actually inspected or read.
    import ROOT as _ROOT
except ImportError:  # pragma: no cover - depends on the local HEP stack.
    _ROOT = None

ROOT = _ROOT
if ROOT is not None:  # pragma: no branch - one-time environment setup.
    ROOT.gROOT.SetBatch(True)


# Backward-compatible public names used by xgboost_root_varfiles_module.py.
VARIABLE_COUNT = 29
FEATURE_NAMES = list(LEGACY_FEATURE_NAMES)


class RootVariableFileError(ValueError):
    """Raised when a ROOT file does not implement its declared contract."""


@dataclass(frozen=True)
class RootVariableFileInfo:
    path: str
    canonical_sample_id: str
    observable_schema: str
    feature_profile: str
    tree_name: str
    feature_branch: str
    stored_value_count: int
    feature_count: int
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    entries: int
    root_metadata_verified: bool
    pairing_count: int | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["feature_units"] = list(self.feature_units)
        return payload


def _root_module(root_module: Any | None = None) -> Any:
    module = ROOT if root_module is None else root_module
    if module is None:
        raise RuntimeError(
            "PyROOT is required to inspect or read 4H variable ROOT files; "
            "schema and model metadata helpers can be used without it"
        )
    return module


def _open_root_file(path: Path, root_module: Any) -> Any:
    root_file = root_module.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise OSError(f"Failed to open ROOT variable file: {path}")
    return root_file


def _get_required_object(root_file: Any, name: str, path: Path) -> Any:
    obj = root_file.Get(name)
    if not obj:
        raise RootVariableFileError(f"{path} is missing required ROOT object {name!r}")
    return obj


def _named_title(obj: Any, name: str, path: Path) -> str:
    if not hasattr(obj, "GetTitle"):
        raise RootVariableFileError(f"{path}: ROOT object {name!r} is not a TNamed value")
    return str(obj.GetTitle())


def _parameter_value(obj: Any, name: str, path: Path) -> int:
    for accessor in ("GetVal", "GetValue"):
        if hasattr(obj, accessor):
            try:
                return int(getattr(obj, accessor)())
            except (TypeError, ValueError) as exc:
                raise RootVariableFileError(
                    f"{path}: ROOT object {name!r} does not contain an integer"
                ) from exc
    raise RootVariableFileError(f"{path}: ROOT object {name!r} is not a TParameter")


def _parse_json_string_list(encoded: str, name: str, path: Path) -> tuple[str, ...]:
    try:
        values = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise RootVariableFileError(f"{path}: {name} does not contain valid JSON") from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise RootVariableFileError(f"{path}: {name} must be a JSON array of strings")
    return tuple(values)


def _branch_length(tree: Any, branch_name: str) -> int | None:
    branch = tree.GetBranch(branch_name)
    if not branch:
        return None
    leaf = None
    if hasattr(branch, "GetLeaf"):
        leaf = branch.GetLeaf(branch_name)
    if not leaf and hasattr(tree, "GetLeaf"):
        leaf = tree.GetLeaf(branch_name)
    if leaf and hasattr(leaf, "GetLenStatic"):
        length = int(leaf.GetLenStatic())
        if length > 0:
            return length
    if hasattr(branch, "GetTitle"):
        match = re.search(r"\[(\d+)\]", str(branch.GetTitle()))
        if match:
            return int(match.group(1))
    return 1


def _detect_observable_schema(root_file: Any) -> str:
    return EXTENDED_SCHEMA_ID if root_file.Get("Data3") else LEGACY_SCHEMA_ID


def _inspect_open_root_file(
    root_file: Any,
    path: Path,
    observable_set: str | None,
    feature_profile: str | None,
) -> RootVariableFileInfo:
    schema_id = _detect_observable_schema(root_file) if observable_set is None else observable_set
    schema = get_schema(schema_id)
    contract = get_feature_contract(schema.schema_id, feature_profile)

    tree = root_file.Get(schema.tree_name)
    if not tree:
        raise RootVariableFileError(
            f"{path} does not contain the {schema.tree_name} tree required by "
            f"{schema.schema_id}"
        )
    for branch_name in schema.required_branches:
        if not tree.GetBranch(branch_name):
            raise RootVariableFileError(
                f"{path}: {schema.tree_name} does not contain required branch {branch_name!r}"
            )

    stored_count = _branch_length(tree, schema.feature_branch)
    if stored_count != schema.stored_value_count:
        raise RootVariableFileError(
            f"{path}: {schema.tree_name}.{schema.feature_branch} stores {stored_count} values; "
            f"{schema.schema_id} requires {schema.stored_value_count}"
        )

    metadata_verified = False
    pairing_count = None
    if schema.schema_id == EXTENDED_SCHEMA_ID:
        schema_name = "Data3_observable_schema"
        declared_schema = _named_title(
            _get_required_object(root_file, schema_name, path), schema_name, path
        )
        if declared_schema != EXTENDED_SCHEMA_ID:
            raise RootVariableFileError(
                f"{path}: Data3 declares schema {declared_schema!r}, expected "
                f"{EXTENDED_SCHEMA_ID!r}"
            )

        names_object = "Data3_feature_names_json"
        root_feature_names = _parse_json_string_list(
            _named_title(_get_required_object(root_file, names_object, path), names_object, path),
            names_object,
            path,
        )
        units_object = "Data3_feature_units_json"
        root_feature_units = _parse_json_string_list(
            _named_title(_get_required_object(root_file, units_object, path), units_object, path),
            units_object,
            path,
        )
        try:
            validate_feature_contract(
                EXTENDED_SCHEMA_ID,
                root_feature_names,
                "full91",
                feature_units=root_feature_units,
            )
        except FeatureContractMismatch as exc:
            raise RootVariableFileError(f"{path}: Data3 metadata {exc}") from exc

        count_object = "Data3_feature_count"
        declared_count = _parameter_value(
            _get_required_object(root_file, count_object, path), count_object, path
        )
        if declared_count != len(EXTENDED_FEATURE_NAMES):
            raise RootVariableFileError(
                f"{path}: Data3_feature_count is {declared_count}, expected "
                f"{len(EXTENDED_FEATURE_NAMES)}"
            )
        pairing_object = "Data3_pairing_count"
        pairing_count = _parameter_value(
            _get_required_object(root_file, pairing_object, path), pairing_object, path
        )
        if pairing_count != PAIRING_COUNT:
            raise RootVariableFileError(
                f"{path}: Data3_pairing_count is {pairing_count}, expected {PAIRING_COUNT}"
            )
        metadata_verified = True

    return RootVariableFileInfo(
        path=str(path),
        canonical_sample_id=canonical_sample_id(path),
        observable_schema=schema.schema_id,
        feature_profile=contract.feature_profile,
        tree_name=schema.tree_name,
        feature_branch=schema.feature_branch,
        stored_value_count=schema.stored_value_count,
        feature_count=contract.feature_count,
        feature_names=contract.feature_names,
        feature_units=contract.feature_units,
        entries=int(tree.GetEntries()),
        root_metadata_verified=metadata_verified,
        pairing_count=pairing_count,
    )


def inspect_ROOT_varfile(
    filename: str | Path,
    observable_set: str | None = None,
    feature_profile: str | None = None,
    *,
    root_module: Any | None = None,
) -> RootVariableFileInfo:
    """Inspect tree branches and v2 TNamed/TParameter metadata without reading rows.

    With ``observable_set=None``, a Data3 tree takes precedence over Data2.
    Callers that need legacy semantics from a tagged file must request
    ``legacy-28-v1`` explicitly.
    """

    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"ROOT variable file does not exist: {path}")
    module = _root_module(root_module)
    root_file = _open_root_file(path, module)
    try:
        return _inspect_open_root_file(root_file, path, observable_set, feature_profile)
    finally:
        root_file.Close()


def read_ROOT_varfile(
    filename,
    sample_id,
    xsec=1.0,
    max_events=None,
    include_weight_feature=False,
    observable_set=LEGACY_SCHEMA_ID,
    feature_profile=None,
    return_metadata=False,
    *,
    root_module=None,
):
    """Return feature rows, labels, and signed cross-section-weighted weights.

    The default is byte-for-byte compatible in shape and return structure with
    the historical Data2 reader: classifier inputs are ``variables[1:]`` and
    ``variables[0]`` is used as a fallback event weight.  For Data3, the
    feature profile is selected from ``features[91]`` and weight remains a
    separate branch.

    If ``return_metadata`` is true, a fourth dictionary contains valid source
    entry indices, original analyzer event indices, cut masks, legacy-cut
    decisions, and unscaled event weights.  Invalid/non-finite rows are absent
    from every returned array, so these identifiers can be used directly for
    deterministic cross-fitting.
    """

    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"ROOT variable file does not exist: {path}")
    module = _root_module(root_module)
    root_file = _open_root_file(path, module)
    try:
        info = _inspect_open_root_file(
            root_file,
            path,
            str(observable_set),
            feature_profile,
        )
        schema = get_schema(info.observable_schema)
        contract = get_feature_contract(info.observable_schema, info.feature_profile)
        tree = root_file.Get(schema.tree_name)

        n_entries = int(tree.GetEntries())
        if max_events is not None:
            requested = int(max_events)
            if requested < 0:
                raise ValueError("max_events must be non-negative")
            n_entries = min(n_entries, requested)

        if include_weight_feature and schema.schema_id != LEGACY_SCHEMA_ID:
            raise ValueError(
                "include_weight_feature is a legacy compatibility option and cannot be "
                "used with extended-91-v2"
            )

        try:
            xsec_value = float(xsec)
        except (TypeError, ValueError) as exc:
            raise ValueError("xsec must be a finite number") from exc
        if not math.isfinite(xsec_value):
            raise ValueError("xsec must be a finite number")

        features = []
        labels = []
        weights = []
        source_entry_indices = []
        event_indices = []
        cut_masks = []
        legacy_cut_decisions = []
        raw_weights = []
        discarded_nonfinite = 0

        for entry in range(n_entries):
            tree.GetEntry(entry)
            stored_values = [
                float(getattr(tree, schema.feature_branch)[index])
                for index in range(schema.stored_value_count)
            ]
            if schema.schema_id == LEGACY_SCHEMA_ID:
                weight = float(getattr(tree, schema.weight_branch, stored_values[0]))
                row = [
                    stored_values[schema.feature_offset + index]
                    for index in contract.feature_indices
                ]
                if include_weight_feature:
                    row = [stored_values[0]] + row
                event_index = entry
                cut_mask = None
                passed_legacy = None
            else:
                weight = float(getattr(tree, schema.weight_branch))
                row = [stored_values[index] for index in contract.feature_indices]
                event_index = int(getattr(tree, "event_index"))
                cut_mask = int(getattr(tree, "cut_mask"))
                passed_legacy = bool(getattr(tree, "passes_legacy_full_selection"))

            if not math.isfinite(weight) or not all(math.isfinite(value) for value in row):
                discarded_nonfinite += 1
                continue

            features.append(row)
            labels.append(sample_id)
            weights.append(weight * xsec_value)
            source_entry_indices.append(entry)
            event_indices.append(event_index)
            cut_masks.append(cut_mask)
            legacy_cut_decisions.append(passed_legacy)
            raw_weights.append(weight)

        if not return_metadata:
            return features, labels, weights

        metadata = {
            "file_contract": info.to_dict(),
            "source_entry_indices": source_entry_indices,
            "event_indices": event_indices,
            "cut_masks": cut_masks,
            "passes_legacy_full_selection": legacy_cut_decisions,
            "raw_weights": raw_weights,
            "xsec_fb": xsec_value,
            "entries_considered": n_entries,
            "entries_loaded": len(features),
            "entries_discarded_nonfinite": discarded_nonfinite,
        }
        return features, labels, weights, metadata
    finally:
        root_file.Close()


def read_ROOT_varfile_with_metadata(*args, **kwargs):
    """Explicit four-value wrapper for cross-fitting/event-audit callers."""

    kwargs["return_metadata"] = True
    return read_ROOT_varfile(*args, **kwargs)


__all__ = [
    "EXTENDED_SCHEMA_ID",
    "FEATURE_NAMES",
    "LEGACY_SCHEMA_ID",
    "ROOT",
    "RootVariableFileError",
    "RootVariableFileInfo",
    "VARIABLE_COUNT",
    "canonical_sample_basename",
    "canonical_sample_id",
    "inspect_ROOT_varfile",
    "read_ROOT_varfile",
    "read_ROOT_varfile_with_metadata",
    "strip_extended_v2_tag",
]
