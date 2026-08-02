#!/usr/bin/env python3
"""Create an immutable background manifest with selected LHE-header normalizations."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lhe_init(path: Path) -> tuple[float, float]:
    opener = gzip.open if path.suffix == ".gz" else open
    lines: list[str] = []
    in_init = False
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("<init"):
                in_init = True
                continue
            if in_init and stripped.startswith("</init"):
                break
            if in_init and stripped and not stripped.startswith("#"):
                lines.append(stripped)
    if len(lines) < 2:
        raise ValueError(f"{path}: missing LHE init process rows")
    process_count = int(lines[0].split()[-1])
    process_rows = lines[1 : 1 + process_count]
    if len(process_rows) != process_count:
        raise ValueError(f"{path}: incomplete LHE init process rows")
    cross_sections_pb = [float(row.split()[0]) for row in process_rows]
    uncertainties_pb = [float(row.split()[1]) for row in process_rows]
    cross_section_fb = 1000.0 * sum(cross_sections_pb)
    uncertainty_fb = 1000.0 * math.sqrt(
        sum(value * value for value in uncertainties_pb)
    )
    if not math.isfinite(cross_section_fb) or cross_section_fb <= 0.0:
        raise ValueError(f"{path}: non-positive LHE init cross section")
    if not math.isfinite(uncertainty_fb) or uncertainty_fb < 0.0:
        raise ValueError(f"{path}: invalid LHE init uncertainty")
    return cross_section_fb, uncertainty_fb


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def prepare_manifest(
    analysis_root: Path,
    input_manifest: Path,
    output_manifest: Path,
    adopted_samples: Sequence[str],
    feature_row_source_manifest: Path | None = None,
    feature_row_sample: str | None = None,
) -> dict[str, Any]:
    analysis_root = analysis_root.expanduser().resolve()
    input_manifest = input_manifest.expanduser().resolve()
    output_manifest = output_manifest.expanduser().resolve()
    adopted = set(adopted_samples)
    if not adopted:
        raise ValueError("at least one --adopt-lhe-xsec sample is required")
    with input_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "sample_id" not in fields or "source_lhe" not in fields:
        raise ValueError(f"{input_manifest}: missing required manifest fields")
    if (feature_row_source_manifest is None) != (feature_row_sample is None):
        raise ValueError(
            "feature_row_source_manifest and feature_row_sample must be supplied together"
        )
    feature_override: dict[str, Any] | None = None
    if feature_row_source_manifest is not None and feature_row_sample is not None:
        source_manifest = feature_row_source_manifest.expanduser().resolve()
        with source_manifest.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        replacement = next(
            (
                row
                for row in source_rows
                if row.get("sample_id", "").strip() == feature_row_sample
            ),
            None,
        )
        target = next(
            (row for row in rows if row["sample_id"].strip() == feature_row_sample),
            None,
        )
        if replacement is None or target is None:
            raise ValueError(
                f"{feature_row_sample}: sample is absent from the input or feature manifest"
            )
        protected_fields = (
            "sample_id",
            "role",
            "source_lhe",
            "cross_section_fb",
            "k_factor",
            "hbb_power",
            "c_mistags",
            "light_mistags",
            "lhe_event_count",
            "hard_event_policy",
        )
        changed = [
            field
            for field in protected_fields
            if target.get(field, "").strip() != replacement.get(field, "").strip()
        ]
        if changed:
            raise ValueError(
                f"{feature_row_sample}: replacement changes protected fields: "
                + ", ".join(changed)
            )
        previous = {
            "root_file": target.get("root_file"),
            "generated_events": target.get("generated_events"),
        }
        for field in ("root_file", "generated_events"):
            value = replacement.get(field, "").strip()
            if not value:
                raise ValueError(f"{feature_row_sample}: replacement has no {field}")
            target[field] = value
        feature_override = {
            "sample_id": feature_row_sample,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256(source_manifest),
            "previous": previous,
            "replacement": {
                "root_file": target["root_file"],
                "generated_events": target["generated_events"],
            },
        }
    found: set[str] = set()
    audits: list[dict[str, Any]] = []
    extra_fields = [
        "normalization_source",
        "normalization_uncertainty_fb",
        "normalization_relative_uncertainty",
        "normalization_previous_cross_section_fb",
    ]
    for field in extra_fields:
        if field not in fields:
            fields.append(field)
    for row in rows:
        sample_id = row["sample_id"].strip()
        if sample_id not in adopted:
            continue
        found.add(sample_id)
        source_lhe = _resolve(analysis_root, row["source_lhe"].strip())
        if not source_lhe.is_file():
            raise ValueError(f"{sample_id}: missing source LHE {source_lhe}")
        cross_section_fb, uncertainty_fb = _lhe_init(source_lhe)
        previous = float(row["cross_section_fb"])
        row["cross_section_fb"] = f"{cross_section_fb:.12g}"
        row["normalization_source"] = "source_lhe_init"
        row["normalization_uncertainty_fb"] = f"{uncertainty_fb:.12g}"
        row["normalization_relative_uncertainty"] = (
            f"{uncertainty_fb / cross_section_fb:.12g}"
        )
        row["normalization_previous_cross_section_fb"] = f"{previous:.12g}"
        audits.append(
            {
                "sample_id": sample_id,
                "source_lhe": str(source_lhe),
                "source_lhe_sha256": _sha256(source_lhe),
                "previous_cross_section_fb": previous,
                "adopted_cross_section_fb": cross_section_fb,
                "adopted_uncertainty_fb": uncertainty_fb,
                "relative_uncertainty": uncertainty_fb / cross_section_fb,
                "relative_change": cross_section_fb / previous - 1.0,
            }
        )
    missing = sorted(adopted - found)
    if missing:
        raise ValueError(f"samples absent from manifest: {', '.join(missing)}")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_manifest.with_name(f".{output_manifest.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if output_manifest.exists():
        if temporary.read_bytes() != output_manifest.read_bytes():
            temporary.unlink()
            raise ValueError(f"{output_manifest}: existing manifest differs")
        temporary.unlink()
    else:
        os.replace(temporary, output_manifest)
    payload = {
        "schema": "resonance-background-normalization-audit-v1",
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": _sha256(input_manifest),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": _sha256(output_manifest),
        "adopted_samples": audits,
        "feature_row_override": feature_override,
    }
    audit_path = output_manifest.with_suffix(".normalization_audit.json")
    audit_text = json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    if audit_path.exists() and audit_path.read_text(encoding="utf-8") != audit_text:
        raise ValueError(f"{audit_path}: existing audit differs")
    if not audit_path.exists():
        audit_path.write_text(audit_text, encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--adopt-lhe-xsec", action="append", default=[])
    parser.add_argument("--feature-row-source-manifest", type=Path)
    parser.add_argument("--feature-row-sample")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = prepare_manifest(
        args.analysis_root,
        args.input_manifest,
        args.output_manifest,
        args.adopt_lhe_xsec,
        args.feature_row_source_manifest,
        args.feature_row_sample,
    )
    for audit in payload["adopted_samples"]:
        print(
            f"{audit['sample_id']}: adopted {audit['adopted_cross_section_fb']:.6g} fb "
            f"from LHE init ({audit['relative_uncertainty']:.3%} integration uncertainty)"
        )
    print(payload["output_manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
