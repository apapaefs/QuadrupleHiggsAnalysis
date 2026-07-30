#!/usr/bin/env python3

from pathlib import Path as _Path
import hashlib as _hashlib
import json as _json
import math as _math
import os as _os
import re as _re
import sys as _sys

_REPO_DIR = _Path(__file__).resolve().parent
_CODE_DIR = _REPO_DIR / "Code"
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from sample_report import (
    attach_poisson_event_interval,
    background_generation_rate_factor,
    background_tag_rate_factor,
    event_interval_text,
    signal_generation_rate_factor,
    signal_tag_rate_factor,
    terminal_xgboost_mc_table,
)

DEFAULT_HBB_BRANCHING_RATIO = 0.5824
DEFAULT_ZBB_BRANCHING_RATIO = 0.150998
DEFAULT_BTAGGING_RATE = 0.85
DEFAULT_SIGNAL_HBB_POWER = 4
DEFAULT_EIGHT_BTAG_POWER = 8
DEFAULT_SIGNAL_K_FACTOR = 2.0
DEFAULT_BACKGROUND_K_FACTOR = 2.0
DEFAULT_BACKGROUND_CSV = _REPO_DIR / "Backgrounds" / "processes.csv"
DEFAULT_BACKGROUND_HERWIG_TEMPLATE = _REPO_DIR / "Backgrounds" / "HW-AlpGen8Q-LHEWriter-Reweighted.in"
DEFAULT_SM_HH4B_C3_XSEC_FIT = (
    _REPO_DIR / "Signals" / "sm_hh4b_heft" / "c3_xsec_fit.json"
)
LEGACY_EXTENDED_V2_TAG = "extended-v2"
EXTENDED_V2_TAG = "extended-v2-uniform-smear-v1"
JET_SMEARING_MODEL_ID = "cms-energy-uniform-fourvector-v1"
JET_SMEARING_ACCEPTANCE_ORDER = "raw_abs_eta_then_smear_then_smeared_pt"
JET_SMEARING_FOURVECTOR_SCALING = "uniform_correlated"
JET_SMEARING_SEED = 14101983
JET_SMEARING_MIN_ENERGY_GEV = 1.0e-6
JET_SMEARING_MAX_MASS_RESIDUAL_GEV = 1.0e-8
_VERSIONED_ANALYSIS_TAGS = (EXTENDED_V2_TAG, LEGACY_EXTENDED_V2_TAG)
_SHAPE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
)


def _configure_parallel_shape_threads(shape_jobs):
    """Cap numerical-library threads before importing the v2 runner."""

    if int(shape_jobs) <= 1:
        return {name: _os.environ.get(name) for name in _SHAPE_THREAD_ENVIRONMENT}
    for name in _SHAPE_THREAD_ENVIRONMENT:
        _os.environ[name] = "1"
    return {name: _os.environ[name] for name in _SHAPE_THREAD_ENVIRONMENT}


def _configure_v2_mode_defaults(args):
    """Fill mode-dependent CLI defaults without affecting legacy workflows."""

    if args.study_outdir is None:
        suffix = "" if args.study_mode == "full" else f"_{args.study_mode}"
        if getattr(args, "no_pyhf", False) and args.study_mode in {
            "fast-sm",
            "fast-pooled",
            "fast-parameterized",
            "full",
        }:
            suffix += "_cut-only"
        args.study_outdir = (
            _REPO_DIR / f"xgboost_c3d4_study_v2_uniform-smear-v1{suffix}"
        )
    if args.training_strategy is None:
        args.training_strategy = {
            "smoke": "sm-crossfit-v2",
            "preview": "pooled-crossfit-v2",
            "fast-sm": "sm-crossfit-v2",
            "fast-pooled": "pooled-crossfit-v2",
            "fast-parameterized": "parameterized-crossfit-v1",
            "full": "pooled-crossfit-v2",
        }[args.study_mode]
    return args


def _canonical_sample_name(filename):
    """Return the production sample name, removing variable-tree and schema tags."""

    sample_name = _Path(filename).name.split("_var.smear", 1)[0]
    for analysis_tag in _VERSIONED_ANALYSIS_TAGS:
        tagged_suffix = f"-{analysis_tag}"
        if sample_name.endswith(tagged_suffix):
            sample_name = sample_name[: -len(tagged_suffix)]
            break
    return sample_name


def _var_root_matches_analysis_tag(path, analysis_tag=None):
    """Keep tagged and untagged observable files in disjoint discovery sets."""

    name = _Path(path).name
    tagged_markers = tuple(
        f"-{versioned_tag}_var.smear"
        for versioned_tag in _VERSIONED_ANALYSIS_TAGS
    )
    if analysis_tag is None:
        return not any(marker in name for marker in tagged_markers)
    return f"-{analysis_tag}_var.smear" in name


def _parse_herwig_total_xsec(out_file):
    if not out_file.exists():
        return None, None

    total_pattern = _re.compile(r"^Total:\s+(\d+)\s+\d+\s+([0-9.+\-eE()]+)")
    for line in out_file.read_text(errors="replace").splitlines():
        match = total_pattern.search(line.strip())
        if not match:
            continue
        generated = int(match.group(1))
        xsec_nb_text = _re.sub(r"\([^)]*\)", "", match.group(2))
        return float(xsec_nb_text) * 1.0e6, generated
    return None, None


def _metadata_for_root_file(root_file):
    root_file = _Path(root_file)
    sample_name = _canonical_sample_name(root_file.name)
    out_file = root_file.parent.parent / f"{sample_name}.out"
    if not out_file.exists():
        matches = sorted(root_file.parent.parent.rglob(f"{sample_name}.out"))
        if matches:
            out_file = matches[0]
    xsec_fb, generated = _parse_herwig_total_xsec(out_file)
    return xsec_fb, generated, out_file


def _discover_var_root_files(sample_dir, include_auxiliary=False, analysis_tag=None):
    files = sorted((_REPO_DIR / sample_dir / "events").glob("*_var.smear*.root"))
    files = [path for path in files if _var_root_matches_analysis_tag(path, analysis_tag)]
    if include_auxiliary:
        return files
    excluded = ("debug", "smoke")
    return [path for path in files if not any(token in path.name.lower() for token in excluded)]


def _parse_mg5_c3d4_run_name(run_dir):
    parts = _Path(run_dir).name.split("_")
    if len(parts) < 6 or parts[:3] != ["run", "gg", "4h"]:
        return "", None, None
    try:
        return parts[-3], float(parts[-2]), float(parts[-1])
    except ValueError:
        return parts[-3], None, None


def _parse_mg5_banner_metadata(banner_file):
    metadata = {"xsec_pb": None, "generated_events": None}
    if banner_file is None or not banner_file.exists():
        return metadata

    xsec_patterns = [
        _re.compile(r"Integrated weight\s*\(pb\)\s*[:=]\s*([0-9.+\-eE]+)", _re.IGNORECASE),
        _re.compile(r"cross-?section\s*[:=]\s*([0-9.+\-eE]+)\s*pb", _re.IGNORECASE),
    ]
    event_patterns = [
        _re.compile(r"Number of Events\s*[:=]\s*(\d+)", _re.IGNORECASE),
        _re.compile(r"^\s*(\d+)\s*=\s*nevents\b", _re.IGNORECASE),
    ]
    for line in banner_file.read_text(errors="replace").splitlines():
        for pattern in xsec_patterns:
            match = pattern.search(line)
            if match:
                metadata["xsec_pb"] = float(match.group(1))
        for pattern in event_patterns:
            match = pattern.search(line)
            if match:
                metadata["generated_events"] = int(match.group(1))
    return metadata


def _metadata_for_score_root(root_file, default_generated_events=None):
    root_file = _Path(root_file)
    for parent in [root_file.parent] + list(root_file.parents):
        if not (
            parent.name.startswith("run_gg_4h_")
            or parent.name.startswith("run_gg_hhhg_")
            or parent.name.startswith("run_gg_hhgg_")
            or parent.name.startswith("run_gg_hhbbbb_heft_")
        ):
            continue
        banners = sorted(parent.glob("*_banner.txt"))
        metadata = _parse_mg5_banner_metadata(banners[0] if banners else None)
        xsec_fb = metadata["xsec_pb"] * 1000.0 if metadata["xsec_pb"] is not None else None
        generated_events = metadata["generated_events"] or default_generated_events
        return xsec_fb, generated_events
    return None, default_generated_events


def _write_mg5_c3d4_manifest(process_dir, output_csv):
    import csv

    process_dir = _Path(process_dir)
    events_dir = process_dir / "Events"
    run_dirs = sorted(path for path in events_dir.glob("run_gg_4h_*") if path.is_dir())
    output_csv = _Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_dir",
        "run_group",
        "c3",
        "d4",
        "lhe_file",
        "banner_file",
        "generated_events",
        "xsec_pb",
        "xsec_fb",
        "var_root_files",
        "status",
    ]
    rows = []
    for run_dir in run_dirs:
        run_group, c3, d4 = _parse_mg5_c3d4_run_name(run_dir)
        lhe_file = run_dir / "unweighted_events.lhe.gz"
        banners = sorted(run_dir.glob("*_banner.txt"))
        banner_file = banners[0] if banners else None
        metadata = _parse_mg5_banner_metadata(banner_file)
        xsec_pb = metadata["xsec_pb"]
        var_roots = sorted(run_dir.glob("*_var.smear*.root"))

        if var_roots:
            status = "ready_to_score"
        elif lhe_file.exists():
            status = "lhe_ready_needs_var_root"
        else:
            status = "waiting_for_lhe"

        rows.append(
            {
                "run_dir": str(run_dir),
                "run_group": run_group,
                "c3": "" if c3 is None else c3,
                "d4": "" if d4 is None else d4,
                "lhe_file": str(lhe_file) if lhe_file.exists() else "",
                "banner_file": str(banner_file) if banner_file is not None else "",
                "generated_events": "" if metadata["generated_events"] is None else metadata["generated_events"],
                "xsec_pb": "" if xsec_pb is None else xsec_pb,
                "xsec_fb": "" if xsec_pb is None else xsec_pb * 1000.0,
                "var_root_files": ";".join(str(path) for path in var_roots),
                "status": status,
            }
        )

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("Prepared MG5 c3/d4 manifest:", output_csv)
    print("Run directory counts:", counts)
    return output_csv


def _stable_seed(text, base_seed=31122002):
    digest = _hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16) % 100000000


def _optional_int(row, key):
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def _count_compact_flavor(text, flavor):
    return sum(int(match.group(1)) for match in _re.finditer(rf"(?<![A-Za-z0-9])(\d+)\s*{flavor}\b", text))


def _count_named_pair_decays(text, flavor):
    if flavor == "b":
        patterns = [r"\bb\s*bbar\b", r"\bb\s+anti-?b\b", r"\bb\s+-b\b"]
    elif flavor == "c":
        patterns = [r"\bc\s*cbar\b", r"\bc\s+anti-?c\b", r"\bc\s+-c\b"]
    else:
        return 0
    return 2 * sum(len(_re.findall(pattern, text)) for pattern in patterns)


def _count_text_flavors(text):
    text = text.lower().replace("_", " ")
    return {
        "b_quarks": _count_compact_flavor(text, "b") + _count_named_pair_decays(text, "b"),
        "c_quarks": _count_compact_flavor(text, "c") + _count_named_pair_decays(text, "c"),
        "light_jets": (
            _count_compact_flavor(text, "j")
            + sum(int(match.group(1)) for match in _re.finditer(r"(?<![A-Za-z0-9])(\d+)\s*light\s+jets?\b", text))
        ),
    }


def _count_pdg_final_state(process_text):
    counts = {"b_quarks": 0, "c_quarks": 0, "light_jets": 0}
    if "->" not in process_text:
        return counts

    for segment in process_text.split(";"):
        if "->" not in segment:
            continue
        final_state = segment.split("->", 1)[1]
        for token in _re.findall(r"[-+]?\d+", final_state):
            pdg_id = abs(int(token))
            if pdg_id == 5:
                counts["b_quarks"] += 1
            elif pdg_id == 4:
                counts["c_quarks"] += 1
            elif pdg_id == 901 or pdg_id in {1, 2, 3, 21}:
                counts["light_jets"] += 1
    return counts


def _infer_background_flavor_counts(row, csv_file):
    explicit = {
        "b_quarks": _optional_int(row, "b_quarks"),
        "c_quarks": _optional_int(row, "c_quarks"),
        "light_jets": _optional_int(row, "light_jets"),
    }
    if all(value is not None for value in explicit.values()):
        return explicit

    process_text = row.get("process", "")
    pdg_counts = _count_pdg_final_state(process_text)
    if sum(pdg_counts.values()) > 0:
        inferred = pdg_counts
    else:
        candidates = [
            _count_text_flavors(str(row.get(key, "")))
            for key in ("process", "description", "process_id", "notes")
            if str(row.get(key, "")).strip()
        ]
        complete_candidates = [
            candidate
            for candidate in candidates
            if candidate["b_quarks"] + candidate["c_quarks"] + candidate["light_jets"] == 8
        ]
        inferred = complete_candidates[0] if complete_candidates else (candidates[0] if candidates else {"b_quarks": 0, "c_quarks": 0, "light_jets": 0})

    for key, value in explicit.items():
        if value is not None:
            inferred[key] = value

    if inferred["b_quarks"] + inferred["c_quarks"] + inferred["light_jets"] != 8:
        process_id = row.get("process_id", "").strip()
        raise SystemExit(
            f"Could not infer an 8-candidate flavor composition for {process_id!r} in {csv_file}: "
            f"{inferred}. Add b_quarks,c_quarks,light_jets columns for this row."
        )
    return inferred


def _background_metadata_from_sample(sample):
    return {
        "process_id": sample["process_id"],
        "description": sample["description"],
        "local_lhe": sample["local_lhe"],
        "raw_xsec_pb": sample["xsec_pb"],
        "raw_xsec_fb": sample["xsec_fb"],
        "cross_section_unc_pb": sample.get("cross_section_unc_pb"),
        "relative_uncertainty_percent": sample.get("relative_uncertainty_percent"),
        "xsec_source": sample.get("xsec_source"),
        "b_quarks": sample["b_quarks"],
        "c_quarks": sample["c_quarks"],
        "light_jets": sample["light_jets"],
        "c_mistags": sample["c_quarks"],
        "light_mistags": sample["light_jets"],
    }


def _read_background_processes(csv_file=DEFAULT_BACKGROUND_CSV):
    import csv

    csv_file = _Path(csv_file)
    if not csv_file.exists():
        raise SystemExit(f"Background process CSV does not exist: {csv_file}")

    samples = []
    with open(csv_file, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"process_id", "events", "cross_section_pb", "local_lhe"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{csv_file} is missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            process_id = row["process_id"].strip()
            local_lhe = row["local_lhe"].strip()
            if not process_id or not local_lhe:
                continue

            try:
                events = int(float(row["events"]))
                xsec_pb = float(row["cross_section_pb"])
            except ValueError as exc:
                raise SystemExit(f"Invalid events/cross_section_pb value for {process_id} in {csv_file}") from exc

            flavor_counts = _infer_background_flavor_counts(row, csv_file)
            try:
                xsec_unc_pb = (
                    None
                    if not str(row.get("cross_section_unc_pb", "")).strip()
                    else float(row["cross_section_unc_pb"])
                )
                relative_uncertainty_percent = (
                    None
                    if not str(row.get("relative_uncertainty_percent", "")).strip()
                    else float(row["relative_uncertainty_percent"])
                )
            except ValueError as exc:
                raise SystemExit(f"Invalid cross-section uncertainty for {process_id} in {csv_file}") from exc
            run_name = f"HW-{process_id}"
            local_lhe_path = _REPO_DIR / "Backgrounds" / local_lhe
            output_root = _REPO_DIR / "Backgrounds" / "events" / f"{run_name}.root"
            output_var_root = _REPO_DIR / "Backgrounds" / "events" / f"{run_name}_var.smearCMS.root"
            samples.append(
                {
                    "process_id": process_id,
                    "description": row.get("description", "").strip(),
                    "local_lhe": local_lhe,
                    "local_lhe_path": local_lhe_path,
                    "herwig_lhe": local_lhe,
                    "events": events,
                    "xsec_pb": xsec_pb,
                    "xsec_fb": xsec_pb * 1000.0,
                    "cross_section_unc_pb": xsec_unc_pb,
                    "relative_uncertainty_percent": relative_uncertainty_percent,
                    "xsec_source": row.get("xsec_source", "").strip(),
                    "run_name": run_name,
                    "raw_root": output_root,
                    "var_root": output_var_root,
                    **flavor_counts,
                }
            )
    return samples


def _run_group_order(run_group):
    try:
        return int(run_group)
    except (TypeError, ValueError):
        return -1


def _mg5_run_metadata(run_dir):
    banners = sorted(_Path(run_dir).glob("*_banner.txt"))
    return _parse_mg5_banner_metadata(banners[0] if banners else None)


def _select_unique_c3d4_run_dirs(run_dirs, required_generated_events=10000):
    grouped = {}
    unparsable = []
    for run_dir in run_dirs:
        run_group, c3, d4 = _parse_mg5_c3d4_run_name(run_dir)
        if c3 is None or d4 is None:
            unparsable.append(run_dir)
            continue
        metadata = _mg5_run_metadata(run_dir)
        grouped.setdefault((c3, d4), []).append((run_dir, run_group, metadata))

    selected = []
    duplicates = []
    nonmatching_events = []
    for candidates in grouped.values():
        eligible = [
            candidate
            for candidate in candidates
            if required_generated_events is None
            or candidate[2]["generated_events"] == required_generated_events
        ]
        if not eligible:
            nonmatching_events.extend(candidate[0] for candidate in candidates)
            continue

        candidates = sorted(
            eligible,
            key=lambda item: (
                (item[0] / "unweighted_events.lhe.gz").exists(),
                _run_group_order(item[1]),
                item[0].name,
            ),
            reverse=True,
        )
        selected.append(candidates[0][0])
        selected_point = _parse_mg5_c3d4_run_name(candidates[0][0])[1:]
        for item in grouped[selected_point]:
            if item[0] == candidates[0][0]:
                continue
            if required_generated_events is not None and item[2]["generated_events"] != required_generated_events:
                nonmatching_events.append(item[0])
            else:
                duplicates.append(item[0])

    selected.extend(unparsable)
    return sorted(selected), set(duplicates), set(nonmatching_events)


def _render_herwig_input(template_text, lhe_file, run_name, output_location, nevents, seed):
    replacements = {
        r"^set\s+theLHReader:FileName\s+.*$": f"set theLHReader:FileName {lhe_file}",
        r"^set\s+theGenerator:NumberOfEvents\s+.*$": f"set theGenerator:NumberOfEvents {nevents}",
        r"^set\s+theGenerator:RandomNumberGenerator:Seed\s+.*$": f"set theGenerator:RandomNumberGenerator:Seed {seed}",
        r"^set\s+/Herwig/Analysis/HwSim:OutputLocation\s+.*$": f"set /Herwig/Analysis/HwSim:OutputLocation {output_location}",
        r"^saverun\s+.*\s+theGenerator\s*$": f"saverun {run_name} theGenerator",
    }
    rendered = template_text
    for pattern, replacement in replacements.items():
        rendered = _re.sub(pattern, replacement, rendered, flags=_re.MULTILINE)
    return _ensure_herwig_charm_tagging_settings(rendered)


def _set_or_insert_herwig_setting(text, pattern, replacement, insert_after_pattern=None):
    if _re.search(pattern, text, flags=_re.MULTILINE):
        return _re.sub(pattern, replacement, text, flags=_re.MULTILINE)

    if insert_after_pattern is not None:
        match = _re.search(insert_after_pattern, text, flags=_re.MULTILINE)
        if match:
            insert_at = match.end()
            return text[:insert_at] + "\n" + replacement + text[insert_at:]

    suffix = "" if text.endswith("\n") else "\n"
    return text + suffix + replacement + "\n"


def _ensure_herwig_charm_tagging_settings(text):
    text = _set_or_insert_herwig_setting(
        text,
        r"^set\s+/Herwig/Analysis/HwSim:BTaggingMethod\s+.*$",
        "set /Herwig/Analysis/HwSim:BTaggingMethod GhostBHadrons",
    )
    text = _set_or_insert_herwig_setting(
        text,
        r"^set\s+/Herwig/Analysis/HwSim:CTaggingMethod\s+.*$",
        "set /Herwig/Analysis/HwSim:CTaggingMethod GhostCHadrons",
        insert_after_pattern=r"^set\s+/Herwig/Analysis/HwSim:BTaggingMethod\s+.*$",
    )
    text = _set_or_insert_herwig_setting(
        text,
        r"^set\s+/Herwig/Analysis/HwSim:CharmTagging\s+.*$",
        "set /Herwig/Analysis/HwSim:CharmTagging Yes",
        insert_after_pattern=r"^set\s+/Herwig/Analysis/HwSim:CTaggingMethod\s+.*$",
    )
    return text


def _prepare_herwig_inputs(
    process_dir,
    output_dir,
    template_file,
    manifest_file,
    overwrite=False,
    nevents=10000,
    output_location="events/",
    run_prefix="HW",
    unique_points=True,
    required_generated_events=10000,
):
    import csv

    process_dir = _Path(process_dir)
    events_dir = process_dir / "Events"
    output_dir = _Path(output_dir)
    template_file = _Path(template_file)
    manifest_file = _Path(manifest_file)
    template_text = template_file.read_text()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / output_location).mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    all_run_dirs = sorted(path for path in events_dir.glob("run_gg_4h_*") if path.is_dir())
    if unique_points:
        run_dirs, duplicate_run_dirs, nonmatching_event_run_dirs = _select_unique_c3d4_run_dirs(
            all_run_dirs,
            required_generated_events=required_generated_events,
        )
    else:
        if required_generated_events is None:
            run_dirs = all_run_dirs
            nonmatching_event_run_dirs = set()
        else:
            run_dirs = [
                run_dir
                for run_dir in all_run_dirs
                if _mg5_run_metadata(run_dir)["generated_events"] == required_generated_events
            ]
            nonmatching_event_run_dirs = set(all_run_dirs) - set(run_dirs)
        duplicate_run_dirs = set()

    fieldnames = [
        "status",
        "run_dir",
        "run_name",
        "run_group",
        "c3",
        "d4",
        "lhe_file",
        "mg5_generated_events",
        "herwig_input",
        "herwig_run",
        "herwig_output_root",
        "herwig_output_var_root",
        "nevents",
        "seed",
        "reason",
    ]
    rows = []
    selected_inputs = []

    for run_dir in all_run_dirs:
        run_group, c3, d4 = _parse_mg5_c3d4_run_name(run_dir)
        metadata = _mg5_run_metadata(run_dir)
        lhe_file = run_dir / "unweighted_events.lhe.gz"
        run_name = f"{run_prefix}-{run_dir.name}"
        seed = _stable_seed(run_name)
        herwig_input = output_dir / f"{run_name}.in"
        herwig_run = output_dir / f"{run_name}.run"
        herwig_out = output_dir / f"{run_name}.out"
        herwig_log = output_dir / f"{run_name}.log"
        herwig_output_root = output_dir / output_location / f"{run_name}.root"
        herwig_output_var_root = output_dir / output_location / f"{run_name}_var.smearCMS.root"
        existing = [path for path in [herwig_input, herwig_run, herwig_out, herwig_log, herwig_output_root, herwig_output_var_root] if path.exists()]

        base_row = {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "run_group": run_group,
            "c3": "" if c3 is None else c3,
            "d4": "" if d4 is None else d4,
            "lhe_file": str(lhe_file) if lhe_file.exists() else "",
            "mg5_generated_events": "" if metadata["generated_events"] is None else metadata["generated_events"],
            "herwig_input": str(herwig_input),
            "herwig_run": str(herwig_run),
            "herwig_output_root": str(herwig_output_root),
            "herwig_output_var_root": str(herwig_output_var_root),
            "nevents": nevents,
            "seed": seed,
        }

        if run_dir in nonmatching_event_run_dirs:
            rows.append({
                **base_row,
                "status": "skipped_nonmatching_events",
                "reason": f"MG5 banner generated_events is not {required_generated_events}",
            })
            continue

        if run_dir in duplicate_run_dirs:
            rows.append({**base_row, "status": "skipped_duplicate", "reason": "duplicate c3/d4 point; selected a preferred run directory"})
            continue

        if not lhe_file.exists():
            rows.append({**base_row, "status": "missing_lhe", "reason": "unweighted_events.lhe.gz does not exist"})
            continue

        if existing and not overwrite:
            reason = "existing target(s): " + ";".join(str(path) for path in existing)
            rows.append({**base_row, "status": "skipped_existing", "reason": reason})
            selected_inputs.append(herwig_input)
            continue

        text = _render_herwig_input(
            template_text,
            lhe_file=lhe_file,
            run_name=run_name,
            output_location=output_location,
            nevents=nevents,
            seed=seed,
        )
        herwig_input.write_text(text)
        rows.append({**base_row, "status": "written" if not existing else "overwritten", "reason": ""})
        selected_inputs.append(herwig_input)

    with open(manifest_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    input_list = output_dir / "herwig_inputs_to_run.txt"
    input_list.write_text("".join(f"{path}\n" for path in selected_inputs))
    print("Prepared Herwig input manifest:", manifest_file)
    print("Output directory:", output_dir)
    print("Unique mode:", unique_points)
    print("Required MG5 generated events:", required_generated_events)
    print("Selected Herwig input list:", input_list)
    print("Run counts:", counts)
    return manifest_file


def _prepare_background_herwig_inputs(
    csv_file=DEFAULT_BACKGROUND_CSV,
    output_dir=_REPO_DIR / "Backgrounds",
    template_file=DEFAULT_BACKGROUND_HERWIG_TEMPLATE,
    manifest_file=_REPO_DIR / "Backgrounds" / "background_herwig_inputs_manifest.csv",
    input_list_file=_REPO_DIR / "Backgrounds" / "herwig_background_inputs_to_run.txt",
    overwrite=False,
    output_location="events/",
):
    import csv

    samples = _read_background_processes(csv_file)
    output_dir = _Path(output_dir)
    template_file = _Path(template_file)
    manifest_file = _Path(manifest_file)
    input_list_file = _Path(input_list_file)
    template_text = template_file.read_text()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / output_location).mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    input_list_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "status",
        "process_id",
        "description",
        "local_lhe",
        "herwig_lhe",
        "events",
        "cross_section_pb",
        "cross_section_fb",
        "b_quarks",
        "c_quarks",
        "light_jets",
        "c_mistags",
        "light_mistags",
        "herwig_input",
        "herwig_run",
        "herwig_output_root",
        "herwig_output_var_root",
        "seed",
        "reason",
    ]
    rows = []
    selected_inputs = []

    for sample in samples:
        run_name = sample["run_name"]
        seed = _stable_seed(run_name)
        herwig_input = output_dir / f"{run_name}.in"
        herwig_run = output_dir / f"{run_name}.run"
        herwig_out = output_dir / f"{run_name}.out"
        herwig_log = output_dir / f"{run_name}.log"
        herwig_output_root = output_dir / output_location / f"{run_name}.root"
        herwig_output_var_root = output_dir / output_location / f"{run_name}_var.smearCMS.root"
        existing = [
            path
            for path in [herwig_input, herwig_run, herwig_out, herwig_log, herwig_output_root, herwig_output_var_root]
            if path.exists()
        ]

        base_row = {
            "process_id": sample["process_id"],
            "description": sample["description"],
            "local_lhe": sample["local_lhe"],
            "herwig_lhe": sample["herwig_lhe"],
            "events": sample["events"],
            "cross_section_pb": sample["xsec_pb"],
            "cross_section_fb": sample["xsec_fb"],
            "b_quarks": sample["b_quarks"],
            "c_quarks": sample["c_quarks"],
            "light_jets": sample["light_jets"],
            "c_mistags": sample["c_quarks"],
            "light_mistags": sample["light_jets"],
            "herwig_input": str(herwig_input),
            "herwig_run": str(herwig_run),
            "herwig_output_root": str(herwig_output_root),
            "herwig_output_var_root": str(herwig_output_var_root),
            "seed": seed,
        }

        if not sample["local_lhe_path"].exists():
            rows.append({**base_row, "status": "missing_lhe", "reason": f"{sample['local_lhe_path']} does not exist"})
            continue

        text = _render_herwig_input(
            template_text,
            lhe_file=sample["herwig_lhe"],
            run_name=run_name,
            output_location=output_location,
            nevents=sample["events"],
            seed=seed,
        )

        if existing and not overwrite:
            if not herwig_input.exists() or herwig_input.read_text() != text:
                herwig_input.write_text(text)
                rows.append({
                    **base_row,
                    "status": "updated_input",
                    "reason": "CSV-rendered Herwig input changed; rerun Herwig to refresh existing outputs",
                })
            else:
                reason = "existing target(s): " + ";".join(str(path) for path in existing)
                rows.append({**base_row, "status": "skipped_existing", "reason": reason})
            selected_inputs.append(herwig_input)
            continue

        herwig_input.write_text(text)
        rows.append({**base_row, "status": "written" if not existing else "overwritten", "reason": ""})
        selected_inputs.append(herwig_input)

    with open(manifest_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    input_list_file.write_text("".join(f"{path}\n" for path in selected_inputs))

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    print("Prepared background Herwig input manifest:", manifest_file)
    print("Background CSV:", csv_file)
    print("Output directory:", output_dir)
    print("Selected Herwig input list:", input_list_file)
    print("Run counts:", counts)
    return manifest_file


def _discover_score_roots(paths, analysis_tag=None):
    roots = []
    for path in paths:
        path = _Path(path)
        if path.is_file():
            if _var_root_matches_analysis_tag(path, analysis_tag):
                roots.append(path)
        elif path.is_dir():
            roots.extend(
                sorted(
                    root
                    for root in path.rglob("*_var.smear*.root")
                    if _var_root_matches_analysis_tag(root, analysis_tag)
                )
            )
        else:
            print(f"Warning: score input does not exist: {path}")
    return roots


def _unique_paths(paths):
    unique = []
    seen = set()
    for path in paths:
        path = _Path(path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _is_var_smear_root(path):
    path = _Path(path)
    return path.suffix == ".root" and "_var.smear" in path.name


def _analysis_output_root(raw_root, analysis_tag=None):
    raw_root = _Path(raw_root)
    suffix = "" if analysis_tag is None else f"-{analysis_tag}"
    return raw_root.with_name(raw_root.name.replace(".root", f"{suffix}_var.smearCMS.root"))


def _analysis_log_file(raw_root, analysis_tag=None):
    raw_root = _Path(raw_root)
    suffix = "" if analysis_tag is None else f"-{analysis_tag}"
    return raw_root.with_name(raw_root.name.replace(".root", f"{suffix}.analysis.log"))


def _analysis_summary_file(raw_root, analysis_tag=None):
    raw_root = _Path(raw_root)
    suffix = "" if analysis_tag is None else f"-{analysis_tag}"
    return raw_root.with_name(raw_root.name.replace(".root", f"{suffix}.analysis_summary.json"))


def _analysis_log_file_for_var_root(var_root):
    var_root = _Path(var_root)
    sample_name = var_root.name.split("_var.smear", 1)[0]
    return var_root.with_name(f"{sample_name}.analysis.log")


def _analysis_summary_file_for_var_root(var_root):
    var_root = _Path(var_root)
    sample_name = var_root.name.split("_var.smear", 1)[0]
    return var_root.with_name(f"{sample_name}.analysis_summary.json")


def _parse_analysis_total_weight_in(log_file):
    log_file = _Path(log_file)
    if not log_file.exists():
        return None
    pattern = _re.compile(r"total weight in\s*=\s*([0-9.+\-eE]+)")
    for line in log_file.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def _normalisation_weight_for_var_root(var_root):
    import json

    summary_file = _analysis_summary_file_for_var_root(var_root)
    if summary_file.exists():
        try:
            with open(summary_file) as handle:
                total_weight_in = json.load(handle).get("total_weight_in")
            if total_weight_in is not None:
                return float(total_weight_in)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return _parse_analysis_total_weight_in(_analysis_log_file_for_var_root(var_root))


def _parse_last_number(line):
    matches = _re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", line)
    return None if not matches else float(matches[-1])


def _read_analysis_summary_for_var_root(var_root):
    import json

    summary_file = _analysis_summary_file_for_var_root(var_root)
    if summary_file.exists():
        with open(summary_file) as handle:
            summary = json.load(handle)
        summary["summary_file"] = str(summary_file)
        summary["summary_source"] = "json"
        return summary

    log_file = _analysis_log_file_for_var_root(var_root)
    if not log_file.exists():
        return {
            "summary_file": str(summary_file),
            "summary_source": "missing",
            "status": "missing_analysis_summary",
        }

    summary = {
        "summary_file": str(log_file),
        "summary_source": "log",
        "preselection_mc_events_out": None,
    }
    for line in log_file.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("total weight in"):
            summary["total_weight_in"] = _parse_last_number(stripped)
        elif stripped.startswith("total MC events in"):
            summary["mc_events_in"] = _parse_last_number(stripped)
        elif stripped.startswith("preselection weight out"):
            summary["preselection_weight_out"] = _parse_last_number(stripped)
        elif stripped.startswith("preselection efficiency"):
            summary["preselection_efficiency"] = _parse_last_number(stripped)
        elif stripped.startswith(("feature tree MC events out", "feature-tree MC events")):
            summary["feature_tree_mc_events_out"] = _parse_last_number(stripped)
        elif stripped.startswith(("feature tree weight out", "feature-tree weight out")):
            summary["feature_tree_weight_out"] = _parse_last_number(stripped)
        elif stripped.startswith(("feature tree efficiency", "feature-tree efficiency")):
            summary["feature_tree_efficiency"] = _parse_last_number(stripped)
        elif stripped.startswith("8bs with pT") and "preselection_weight_out" not in summary:
            summary["preselection_weight_out"] = _parse_last_number(stripped)
        elif stripped.startswith("total weight out"):
            summary["analysis_weight_out"] = _parse_last_number(stripped)
        elif stripped.startswith("actual MC events"):
            summary["analysis_mc_events_out"] = _parse_last_number(stripped)
        elif stripped.startswith("efficiency"):
            summary["analysis_efficiency"] = _parse_last_number(stripped)

    total_weight_in = summary.get("total_weight_in")
    preselection_weight_out = summary.get("preselection_weight_out")
    if total_weight_in and preselection_weight_out is not None:
        summary["preselection_efficiency"] = preselection_weight_out / total_weight_in
    if "analysis_efficiency" not in summary and total_weight_in and summary.get("analysis_weight_out") is not None:
        summary["analysis_efficiency"] = summary["analysis_weight_out"] / total_weight_in
    return summary


def _discover_analysis_inputs(paths, analysis_tag=None):
    var_roots = []
    raw_roots = []
    for path in paths:
        path = _Path(path)
        if path.is_file():
            if _is_var_smear_root(path):
                if _var_root_matches_analysis_tag(path, analysis_tag):
                    var_roots.append(path)
            elif path.suffix == ".root":
                raw_roots.append(path)
            else:
                print(f"Warning: analysis input is not a ROOT file: {path}")
        elif path.is_dir():
            var_roots.extend(
                sorted(
                    root
                    for root in path.rglob("*_var.smear*.root")
                    if _var_root_matches_analysis_tag(root, analysis_tag)
                )
            )
            raw_roots.extend(sorted(root for root in path.rglob("*.root") if not _is_var_smear_root(root)))
        else:
            print(f"Warning: analysis input does not exist: {path}")
    return _unique_paths(var_roots), _unique_paths(raw_roots)


def _filter_auxiliary_roots(files, include_auxiliary=False):
    if include_auxiliary:
        return files
    excluded = ("debug", "smoke")
    return [path for path in files if not any(token in path.name.lower() for token in excluded)]


def _expand_cli_values(values, files, label):
    if values is None:
        return None
    if len(values) == 1 and len(files) > 1:
        return values * len(files)
    if len(values) != len(files):
        raise SystemExit(f"Expected {len(files)} {label} values, got {len(values)}")
    return values


def _metadata_for_scored_signal_root(root_file, default_generated_events=None):
    xsec_fb, generated, out_file = _metadata_for_root_file(root_file)
    if xsec_fb is not None:
        return xsec_fb, generated or default_generated_events, out_file

    xsec_fb, generated = _metadata_for_score_root(root_file, default_generated_events)
    return xsec_fb, generated, None if xsec_fb is not None else out_file


def _metadata_for_hhhbb_scored_signal_root(
    root_file, default_generated_events=None
):
    """Prefer the exact forced-splitting merged-LHE cross section."""

    root_file = _Path(root_file)
    sample_name = _canonical_sample_name(root_file.name)
    suffix = "_hhhbb_stage2"
    if sample_name.endswith(suffix):
        run_name = sample_name[: -len(suffix)]
        workdir = root_file.parent.parent
        summary_file = workdir / run_name / "merge_summary.json"
        if not summary_file.is_file():
            matches = sorted(workdir.rglob(f"{run_name}/merge_summary.json"))
            if matches:
                summary_file = matches[0]
        if summary_file.is_file():
            try:
                summary = _json.loads(summary_file.read_text())
                xsec_fb = float(summary["merged_xsec_pb"]) * 1.0e3
                generated = int(summary["total_events"])
                if (
                    _math.isfinite(xsec_fb)
                    and xsec_fb > 0.0
                    and generated > 0
                ):
                    return xsec_fb, generated, summary_file
            except (KeyError, TypeError, ValueError, _json.JSONDecodeError):
                pass
    return _metadata_for_scored_signal_root(
        root_file, default_generated_events
    )


def _metadata_for_sm_hh4b_scored_signal_root(
    root_file, default_generated_events=None
):
    """Read the trusted normalized-LHE metadata for the SM hh+4b sample."""

    root_file = _Path(root_file)
    workdir = root_file.parent.parent
    metadata_file = workdir / "sample_metadata.json"
    if not metadata_file.is_file():
        matches = sorted(workdir.rglob("sample_metadata.json"))
        if matches:
            metadata_file = matches[0]
    if metadata_file.is_file():
        try:
            metadata = _json.loads(metadata_file.read_text())
            xsec_fb = float(metadata["cross_section_pb"]) * 1.0e3
            generated = int(metadata["event_count"])
            if (
                _math.isfinite(xsec_fb)
                and xsec_fb > 0.0
                and generated > 0
            ):
                return xsec_fb, generated, metadata_file
        except (KeyError, TypeError, ValueError, _json.JSONDecodeError):
            pass
    return _metadata_for_scored_signal_root(
        root_file, default_generated_events
    )


def _infer_scored_signal_metadata(
    files,
    xsec_values,
    generated_values,
    default_generated_events,
    label,
    xsec_option,
    metadata_resolver=None,
):
    metadata_resolver = metadata_resolver or _metadata_for_scored_signal_root
    normalisation_weights = [_normalisation_weight_for_var_root(path) for path in files]
    if xsec_values is None or generated_values is None:
        inferred_xsecs = []
        inferred_generated = []
        missing_xsec_sources = []
        for path in files:
            xsec_fb, generated_events, source_file = metadata_resolver(
                path, default_generated_events
            )
            if xsec_values is None:
                if xsec_fb is None:
                    missing_xsec_sources.append(source_file or path)
                else:
                    inferred_xsecs.append(xsec_fb)
            if generated_values is None:
                inferred_generated.append(generated_events)

        if missing_xsec_sources:
            missing_text = "\n  ".join(str(path) for path in missing_xsec_sources)
            raise SystemExit(
                f"Could not infer {label} cross section from its campaign, Herwig .out, "
                "or MG5 banner metadata. "
                f"Missing metadata source(s):\n  {missing_text}\n"
                f"Copy the matching Herwig .out file(s), or pass {xsec_option} once per signal file."
            )

        signal_xsecs = (
            inferred_xsecs
            if xsec_values is None
            else _expand_cli_values(xsec_values, files, f"{label} cross-section")
        )
        signal_generated = (
            inferred_generated
            if generated_values is None
            else _expand_cli_values(generated_values, files, f"{label} generated-event")
        )
    else:
        signal_xsecs = _expand_cli_values(xsec_values, files, f"{label} cross-section")
        signal_generated = _expand_cli_values(generated_values, files, f"{label} generated-event")

    return signal_xsecs, signal_generated, normalisation_weights


def _ensure_background_csv_var_roots(
    samples, args, analysis_tag=None, progress_callback=None
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    accepted_var_roots = set()
    missing_jobs = []
    for sample in samples:
        var_root = _analysis_output_root(sample["raw_root"], analysis_tag)
        raw_root = sample["raw_root"]
        if var_root.exists() and not args.force_analysis:
            current = analysis_tag != EXTENDED_V2_TAG or _extended_v2_output_is_current(
                var_root,
                raw_root=raw_root,
                source_file=args.analysis_source,
                expected_c_mistags=sample["c_quarks"],
                expected_light_mistags=sample["light_jets"],
            )
            if current:
                accepted_var_roots.add(str(var_root.resolve()))
                continue
        if raw_root.exists():
            missing_jobs.append(sample)
            continue
        if var_root.exists() and analysis_tag != EXTENDED_V2_TAG:
            print(
                f"Warning: cannot force-regenerate {var_root} because the raw ROOT "
                f"is unavailable; retaining the existing legacy output"
            )
            accepted_var_roots.add(str(var_root.resolve()))
            continue
        if var_root.exists() and analysis_tag == EXTENDED_V2_TAG:
            raise SystemExit(
                "Tagged v2 background output is stale, incomplete, schema-invalid, or "
                "uses the wrong mistag composition, and its raw ROOT is unavailable: "
                f"{var_root}"
            )
        print(f"Warning: missing background ROOT for {sample['process_id']}: {raw_root}")

    if missing_jobs and args.no_run_missing_analysis:
        if analysis_tag == EXTENDED_V2_TAG:
            paths = "\n  ".join(str(sample["raw_root"]) for sample in missing_jobs)
            raise SystemExit(
                f"{len(missing_jobs)} tagged v2 CSV background output(s) require "
                "regeneration, but --no-run-missing-analysis is active:\n  " + paths
            )
        print(
            f"Warning: {len(missing_jobs)} CSV background raw ROOT file(s) need "
            "analysis, but auto-analysis is disabled."
        )
    elif missing_jobs:
        executable = _ensure_analysis_executable(args.analysis_exe, args.analysis_source, rebuild=True)
        print(f"Running C++ analysis for {len(missing_jobs)} CSV background ROOT file(s)")
        print(f"  background analysis progress 0/{len(missing_jobs)}", flush=True)
        jobs = max(1, int(args.analysis_jobs))

        def run_sample(sample):
            return _run_one_cpp_analysis(
                sample["raw_root"],
                executable,
                max_events=args.analysis_max_events,
                force=True,
                c_mistags=sample["c_quarks"],
                light_mistags=sample["light_jets"],
                analysis_tag=analysis_tag,
            )

        if jobs == 1:
            for index, sample in enumerate(missing_jobs, start=1):
                run_sample(sample)
                print(
                    f"  background analysis progress {index}/{len(missing_jobs)}: "
                    f"{sample['process_id']}",
                    flush=True,
                )
                if progress_callback is not None:
                    progress_callback(
                        index,
                        len(missing_jobs),
                        sample["process_id"],
                    )
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(run_sample, sample): sample for sample in missing_jobs
                }
                for index, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    sample = futures[future]
                    print(
                        f"  background analysis progress {index}/{len(missing_jobs)}: "
                        f"{sample['process_id']}",
                        flush=True,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            index,
                            len(missing_jobs),
                            sample["process_id"],
                        )

    for sample in missing_jobs:
        var_root = _analysis_output_root(sample["raw_root"], analysis_tag)
        if not var_root.exists():
            continue
        if analysis_tag == EXTENDED_V2_TAG and not _extended_v2_output_is_current(
            var_root,
            raw_root=sample["raw_root"],
            source_file=args.analysis_source,
            expected_c_mistags=sample["c_quarks"],
            expected_light_mistags=sample["light_jets"],
        ):
            raise RuntimeError(
                "Regenerated tagged v2 background output failed completeness/schema/"
                f"composition validation: {var_root}"
            )
        accepted_var_roots.add(str(var_root.resolve()))

    ordered_var_roots = [
        _analysis_output_root(sample["raw_root"], analysis_tag)
        for sample in samples
        if str(
            _analysis_output_root(sample["raw_root"], analysis_tag).resolve()
        ) in accepted_var_roots
    ]
    return _unique_paths(
        _filter_auxiliary_roots(
            ordered_var_roots, args.include_auxiliary_samples
        )
    )


def _background_inputs_from_csv(
    args, ensure_analysis=False, analysis_tag=None, progress_callback=None
):
    samples = _read_background_processes(args.background_csv)
    if ensure_analysis:
        background_files = _ensure_background_csv_var_roots(
            samples,
            args,
            analysis_tag=analysis_tag,
            progress_callback=progress_callback,
        )
    else:
        background_files = [
            _analysis_output_root(sample["raw_root"], analysis_tag)
            for sample in samples
            if _analysis_output_root(sample["raw_root"], analysis_tag).exists()
        ]
        missing = [
            sample
            for sample in samples
            if not _analysis_output_root(sample["raw_root"], analysis_tag).exists()
        ]
        for sample in missing:
            missing_root = _analysis_output_root(sample["raw_root"], analysis_tag)
            print(f"Warning: missing background variable ROOT for {sample['process_id']}: {missing_root}")

    by_var_root = {
        str(_analysis_output_root(sample["raw_root"], analysis_tag)): sample
        for sample in samples
    }
    selected_samples = [by_var_root[str(path)] for path in background_files if str(path) in by_var_root]
    background_xsecs = [sample["xsec_fb"] for sample in selected_samples]
    background_generated = [sample["events"] for sample in selected_samples]
    background_metadata = [_background_metadata_from_sample(sample) for sample in selected_samples]
    background_normalisation_weights = [_normalisation_weight_for_var_root(path) for path in background_files]
    return background_files, background_xsecs, background_generated, background_normalisation_weights, background_metadata


def _metadata_for_background_files(background_files, csv_file=DEFAULT_BACKGROUND_CSV):
    samples = {}
    csv_file = _Path(csv_file)
    if csv_file.exists():
        try:
            samples = {
                sample["run_name"]: sample
                for sample in _read_background_processes(csv_file)
            }
        except (SystemExit, ValueError):
            samples = {}

    metadata = []
    for path in background_files:
        sample = samples.get(_canonical_sample_name(path))
        if sample is not None:
            metadata.append(_background_metadata_from_sample(sample))
        else:
            metadata.append({})
    return metadata


def _validate_explicit_background_composition(
    background_inputs,
    csv_file,
    c_mistags,
    light_mistags,
    *,
    analysis_tag=None,
):
    """Reject CSV-known explicit inputs that one global selection cannot model."""

    var_roots, raw_roots = _discover_analysis_inputs(
        background_inputs, analysis_tag=analysis_tag
    )
    candidates = _unique_paths(
        [
            *var_roots,
            *(
                _analysis_output_root(path, analysis_tag)
                for path in raw_roots
            ),
        ]
    )
    metadata = _metadata_for_background_files(candidates, csv_file)
    compositions = {
        (int(sample["c_mistags"]), int(sample["light_mistags"]))
        for sample in metadata
        if "c_mistags" in sample and "light_mistags" in sample
    }
    expected = (int(c_mistags), int(light_mistags))
    if len(compositions) > 1:
        raise SystemExit(
            "Explicit --background inputs have heterogeneous mistag compositions "
            f"{sorted(compositions)}, but explicit inputs use one global analyzer "
            "selection. Omit --background to use the per-process Backgrounds CSV "
            "selection, or supply only samples with one composition."
        )
    if compositions and compositions != {expected}:
        observed = next(iter(compositions))
        raise SystemExit(
            "Explicit --background input composition "
            f"{observed} does not match the global analyzer composition {expected}. "
            "Set --analysis-c-mistags/--analysis-light-mistags consistently, or omit "
            "--background to use the per-process Backgrounds CSV selection."
        )


def _training_inputs_from_cli(args, ensure_analysis=False):
    if ensure_analysis:
        signal_inputs = args.signal or [_REPO_DIR / "Signals" / "events"]
        signal_files = _ensure_analysis_var_roots(
            signal_inputs,
            executable=args.analysis_exe,
            source_file=args.analysis_source,
            include_auxiliary=args.include_auxiliary_samples,
            jobs=args.analysis_jobs,
            max_events=args.analysis_max_events,
            force=args.force_analysis,
            run_missing=not args.no_run_missing_analysis,
            c_mistags=0,
            light_mistags=0,
        )
        if args.background:
            _validate_explicit_background_composition(
                args.background,
                args.background_csv,
                args.analysis_c_mistags,
                args.analysis_light_mistags,
            )
            background_files = _ensure_analysis_var_roots(
                args.background,
                executable=args.analysis_exe,
                source_file=args.analysis_source,
                include_auxiliary=args.include_auxiliary_samples,
                jobs=args.analysis_jobs,
                max_events=args.analysis_max_events,
                force=args.force_analysis,
                run_missing=not args.no_run_missing_analysis,
                c_mistags=args.analysis_c_mistags,
                light_mistags=args.analysis_light_mistags,
            )
            background_metadata = _metadata_for_background_files(background_files, args.background_csv)
        else:
            (
                background_files,
                background_xsecs,
                background_generated,
                background_normalisation_weights,
                background_metadata,
            ) = _background_inputs_from_csv(args, ensure_analysis=True)
    else:
        signal_files = args.signal or _discover_var_root_files("Signals", args.include_auxiliary_samples)
        if args.background:
            background_files = args.background
            background_metadata = _metadata_for_background_files(background_files, args.background_csv)
        else:
            (
                background_files,
                background_xsecs,
                background_generated,
                background_normalisation_weights,
                background_metadata,
            ) = _background_inputs_from_csv(args, ensure_analysis=False)

    if not signal_files:
        raise SystemExit("No signal ROOT variable files found. Pass --signal or add files under Signals/events.")
    if not background_files:
        raise SystemExit(
            "No background ROOT variable files found. Run the CSV background Herwig/analysis steps, "
            "or pass --background explicitly."
        )

    signal_xsecs = _expand_cli_values(args.signal_xsec_fb, signal_files, "signal cross-section")
    if args.background:
        background_xsecs = _expand_cli_values(args.background_xsec_fb, background_files, "background cross-section")
    else:
        if args.background_xsec_fb is not None:
            background_xsecs = _expand_cli_values(args.background_xsec_fb, background_files, "background cross-section")
    signal_generated = []
    if args.background:
        background_generated = []

    if signal_xsecs is None:
        signal_xsecs = []
        for path in signal_files:
            xsec_fb, generated, out_file = _metadata_for_root_file(path)
            if xsec_fb is None:
                print(f"Warning: could not read signal cross section from {out_file}; using 1 fb")
                xsec_fb = 1.0
            signal_xsecs.append(xsec_fb)
            signal_generated.append(generated)
    else:
        for path in signal_files:
            _, generated, _ = _metadata_for_root_file(path)
            signal_generated.append(generated)

    if args.background and background_xsecs is None:
        background_xsecs = []
        for path in background_files:
            xsec_fb, generated, out_file = _metadata_for_root_file(path)
            if xsec_fb is None:
                print(f"Warning: could not read background cross section from {out_file}; using 1 fb")
                xsec_fb = 1.0
            background_xsecs.append(xsec_fb)
            background_generated.append(generated)
    elif args.background:
        for path in background_files:
            _, generated, _ = _metadata_for_root_file(path)
            background_generated.append(generated)

    signal_normalisation_weights = [_normalisation_weight_for_var_root(path) for path in signal_files]
    if args.background:
        background_normalisation_weights = [_normalisation_weight_for_var_root(path) for path in background_files]

    return (
        signal_files,
        background_files,
        signal_xsecs,
        background_xsecs,
        signal_generated,
        background_generated,
        signal_normalisation_weights,
        background_normalisation_weights,
        background_metadata,
    )


def _format_weight(value):
    return "unavailable" if value is None else f"{float(value):g}"


def _print_training_inputs(
    signal_files,
    background_files,
    signal_xsecs,
    background_xsecs,
    signal_generated,
    background_generated,
    signal_normalisation_weights=None,
    background_normalisation_weights=None,
    signal_rate_factors=None,
    background_rate_factors=None,
    background_metadata=None,
):
    if signal_normalisation_weights is None:
        signal_normalisation_weights = [None for _ in signal_files]
    if background_normalisation_weights is None:
        background_normalisation_weights = [None for _ in background_files]
    if signal_rate_factors is None:
        signal_rate_factors = [None for _ in signal_files]
    if background_rate_factors is None:
        background_rate_factors = [None for _ in background_files]
    if background_metadata is None:
        background_metadata = [{} for _ in background_files]

    print("Signal files:")
    for path, xsec, generated, normalisation_weight, rate_factor in zip(
        signal_files,
        signal_xsecs,
        signal_generated,
        signal_normalisation_weights,
        signal_rate_factors,
    ):
        rate_text = "" if rate_factor is None else f"  rate_factor={float(rate_factor):g}"
        print(
            f"  {path}  xsec={xsec:g} fb  generated={generated}  "
            f"normalisation_weight={_format_weight(normalisation_weight)}{rate_text}"
        )
    print("Background files:")
    for path, xsec, generated, normalisation_weight, rate_factor, metadata in zip(
        background_files,
        background_xsecs,
        background_generated,
        background_normalisation_weights,
        background_rate_factors,
        background_metadata,
    ):
        process_text = ""
        if metadata.get("process_id"):
            process_text = (
                f"  process={metadata['process_id']}"
                f"  flavors={metadata.get('b_quarks', 0)}b"
                f"+{metadata.get('c_quarks', 0)}c"
                f"+{metadata.get('light_jets', 0)}j"
            )
        rate_text = "" if rate_factor is None else f"  rate_factor={float(rate_factor):g}"
        print(
            f"  {path}  xsec={xsec:g} fb  generated={generated}  "
            f"normalisation_weight={_format_weight(normalisation_weight)}"
            f"{process_text}{rate_text}"
        )


def _format_count(value):
    return "unavailable" if value is None else str(value)


def _print_sm_background_mc_counts(metrics):
    counts = metrics.get("mc_event_counts", {})
    if not counts:
        return
    print("SM/background MC event counts")
    print("  SM entries read =", _format_count(counts.get("signal_entries_read")))
    print("  SM generated events =", _format_count(counts.get("signal_generated_events")))
    print("  Background entries read =", _format_count(counts.get("background_entries_read")))
    print("  Background generated events =", _format_count(counts.get("background_generated_events")))


def _signal_metadata_for_files(signal_files):
    metadata = []
    for path in signal_files:
        path = _Path(path)
        metadata.append(
            {
                "process_id": _canonical_sample_name(path),
                "description": "SM gg -> hhhh, h -> b bbar",
                "local_lhe": path.name,
            }
        )
    return metadata


def _sample_report_dir(args, default_parent):
    if args.sample_report_dir is not None:
        return args.sample_report_dir
    return _Path(default_parent) / "sample_report"


def _score_rows_summary(rows, luminosity):
    rows = rows or []
    entries = sum(int(row.get("entries", 0)) for row in rows)
    selected_entries = sum(int(row.get("selected_entries", 0)) for row in rows)
    preselected_events = sum(float(row.get("expected_preselected_events", 0.0)) for row in rows)
    selected_events = sum(float(row.get("expected_selected_events", 0.0)) for row in rows)
    selected_error = sum(float(row.get("expected_selected_error", 0.0)) ** 2 for row in rows) ** 0.5
    initial_events = sum(
        float(luminosity) * float(row.get("effective_xsec_fb", 0.0))
        for row in rows
    )
    analysis_efficiency = preselected_events / initial_events if initial_events > 0.0 else 0.0
    xgboost_efficiency = selected_events / preselected_events if preselected_events > 0.0 else 0.0
    final_efficiency = selected_events / initial_events if initial_events > 0.0 else 0.0
    summary = {
        "entries": entries,
        "selected_entries": selected_entries,
        "expected_preselected_events": preselected_events,
        "expected_selected_events": selected_events,
        "expected_selected_error": selected_error,
        "analysis_efficiency": analysis_efficiency,
        "xgboost_efficiency": xgboost_efficiency,
        "final_efficiency": final_efficiency,
    }
    attach_poisson_event_interval(
        summary,
        selected_entries_key="selected_entries",
        expected_events_key="expected_selected_events",
        input_entries_key="entries",
        expected_input_events_key="expected_preselected_events",
        output_prefix="expected_selected_events",
        confidence_level=0.95,
    )
    summary["expected_selected_error"] = summary["expected_selected_events_error_high_95cl"]
    return summary


def _print_xgboost_threshold_summary(threshold, sm_signal_rows, background_rows, luminosity):
    sm_summary = _score_rows_summary(sm_signal_rows, luminosity)
    background_summary = _score_rows_summary(background_rows, luminosity)

    print("XGBoost threshold event summary")
    print(f"  threshold = {threshold:g}")
    print(
        "  SM signal MC entries after threshold = "
        f"{sm_summary['selected_entries']} / {sm_summary['entries']}"
    )
    print(
        "  SM signal expected events after threshold = "
        f"{event_interval_text(sm_summary, 'expected_selected_events')} (95% CL)"
    )
    print(f"  SM signal analysis efficiency = {sm_summary['analysis_efficiency']}")
    print(f"  SM signal XGBoost efficiency = {sm_summary['xgboost_efficiency']}")
    print(f"  SM signal final efficiency = {sm_summary['final_efficiency']}")
    print(
        "  Background MC entries after threshold = "
        f"{background_summary['selected_entries']} / {background_summary['entries']}"
    )
    print(
        "  Background expected events after threshold = "
        f"{event_interval_text(background_summary, 'expected_selected_events')} (95% CL)"
    )
    print(f"  Background analysis efficiency = {background_summary['analysis_efficiency']}")
    print(f"  Background XGBoost efficiency = {background_summary['xgboost_efficiency']}")
    print(f"  Background final efficiency = {background_summary['final_efficiency']}")
    print()
    print(
        terminal_xgboost_mc_table(
            sm_signal_rows,
            title="Per-sample SM XGBoost MC event counts",
            threshold=threshold,
        )
    )
    print()
    print(
        terminal_xgboost_mc_table(
            background_rows,
            title="Per-sample background XGBoost MC event counts",
            threshold=threshold,
        )
    )


def _print_sm_hhhbb_summary(hhhbb_rows):
    sm_rows = [
        row
        for row in (hhhbb_rows or [])
        if row.get("c3") is not None
        and row.get("d4") is not None
        and abs(float(row.get("c3"))) < 1.0e-9
        and abs(float(row.get("d4"))) < 1.0e-9
    ]
    if not sm_rows:
        print("SM hhhbb forced-splitting contribution: no (c3,d4)=(0,0) row was found.")
        return
    row = sm_rows[0]
    print("SM hhhbb forced-splitting contribution")
    print(f"  xsec = {float(row.get('xsec_fb', 0.0)):g} fb")
    print(f"  effective xsec = {float(row.get('effective_xsec_fb', 0.0)):g} fb")
    print(f"  selected events = {float(row.get('expected_selected_events', 0.0)):g}")
    print(f"  analysis efficiency = {float(row.get('analysis_efficiency', 0.0)):g}")
    print(f"  XGBoost efficiency = {float(row.get('xgboost_efficiency', 0.0)):g}")
    print(f"  selected MC count = {int(row.get('selected_entries', 0) or 0)} / {int(row.get('entries', 0) or 0)}")


def _print_sm_hhbbbb_summary(hhbbbb_rows):
    sm_rows = [
        row
        for row in (hhbbbb_rows or [])
        if row.get("c3") is not None
        and abs(float(row.get("c3"))) < 1.0e-9
    ]
    if not sm_rows:
        print("SM hhbbbb forced-splitting contribution: no c3=0 row was found.")
        return
    row = sm_rows[0]
    print("SM hhbbbb forced-splitting contribution")
    print(f"  xsec = {float(row.get('xsec_fb', 0.0)):g} fb")
    print(f"  effective xsec = {float(row.get('effective_xsec_fb', 0.0)):g} fb")
    print(f"  selected events = {float(row.get('expected_selected_events', 0.0)):g}")
    print(f"  analysis efficiency = {float(row.get('analysis_efficiency', 0.0)):g}")
    print(f"  XGBoost efficiency = {float(row.get('xgboost_efficiency', 0.0)):g}")
    print(f"  selected MC count = {int(row.get('selected_entries', 0) or 0)} / {int(row.get('entries', 0) or 0)}")


def _physics_rate_factor(hbb_branching_ratio, hbb_power, btagging_rate, btag_power):
    return float(hbb_branching_ratio) ** int(hbb_power) * float(btagging_rate) ** int(btag_power)


def _signal_generation_rate_factor_for_cli(args):
    return signal_generation_rate_factor(
        args.hbb_branching_ratio,
        args.signal_hbb_power,
        args.signal_k_factor,
    )


def _signal_tag_rate_factor_for_cli(args):
    return signal_tag_rate_factor(args.btagging_rate, args.signal_btag_power)


def _signal_final_rate_factor_for_cli(args):
    return _signal_generation_rate_factor_for_cli(args) * _signal_tag_rate_factor_for_cli(args)


def _hhhbb_signal_rate_factor_for_cli(args):
    hhhbb_generation_factor = signal_generation_rate_factor(
        args.hbb_branching_ratio,
        3,
        args.signal_k_factor,
    )
    return hhhbb_generation_factor * _signal_tag_rate_factor_for_cli(args)


def _hhbbbb_signal_rate_factor_for_cli(args):
    hhbbbb_generation_factor = signal_generation_rate_factor(
        args.hbb_branching_ratio,
        2,
        args.signal_k_factor,
    )
    return hhbbbb_generation_factor * _signal_tag_rate_factor_for_cli(args)


def _background_rate_factor_from_metadata(
    metadata,
    btagging_rate,
    c_mistag_rate,
    light_mistag_rate,
    k_factor,
    hbb_branching_ratio=DEFAULT_HBB_BRANCHING_RATIO,
):
    generation_factor = background_generation_rate_factor(
        metadata,
        k_factor,
        DEFAULT_ZBB_BRANCHING_RATIO,
        hbb_branching_ratio,
    )
    tag_factor = background_tag_rate_factor(
        metadata,
        btagging_rate,
        c_mistag_rate,
        light_mistag_rate,
    )
    return generation_factor * tag_factor


def _background_generation_rate_factor_from_metadata(metadata, args):
    return background_generation_rate_factor(
        metadata,
        args.background_k_factor,
        args.zbb_branching_ratio,
        args.hbb_branching_ratio,
    )


def _background_tag_rate_factor_from_metadata(metadata, args):
    return background_tag_rate_factor(
        metadata,
        args.btagging_rate,
        args.c_mistag_rate,
        args.light_mistag_rate,
    )


def _background_generation_rate_factors_for_cli(background_metadata, args):
    if background_metadata and all(metadata.get("process_id") for metadata in background_metadata):
        return [_background_generation_rate_factor_from_metadata(metadata, args) for metadata in background_metadata]
    return float(args.background_k_factor) * float(args.hbb_branching_ratio) ** int(args.background_hbb_power)


def _background_tag_rate_factors_for_cli(background_metadata, args):
    if background_metadata and all(metadata.get("process_id") for metadata in background_metadata):
        return [_background_tag_rate_factor_from_metadata(metadata, args) for metadata in background_metadata]
    return float(args.btagging_rate) ** int(args.background_btag_power)


def _multiply_rate_factors(left, right, count):
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list):
            left = [left for _ in range(count)]
        if not isinstance(right, list):
            right = [right for _ in range(count)]
        return [float(a) * float(b) for a, b in zip(left, right)]
    return float(left) * float(right)


def _background_rate_factors_for_cli(background_metadata, args):
    generation_factors = _background_generation_rate_factors_for_cli(background_metadata, args)
    tag_factors = _background_tag_rate_factors_for_cli(background_metadata, args)
    return _multiply_rate_factors(generation_factors, tag_factors, len(background_metadata or []))


def _format_optional_float(value):
    if value is None:
        return ""
    return f"{float(value):.10g}"


def _summarize_background_analysis(args):
    import csv

    samples = _read_background_processes(args.background_csv)
    if not args.no_run_missing_analysis:
        _ensure_background_csv_var_roots(samples, args)

    background_metadata = [_background_metadata_from_sample(sample) for sample in samples]
    rate_factors = _background_rate_factors_for_cli(background_metadata, args)
    if not isinstance(rate_factors, list):
        rate_factors = [rate_factors for _ in samples]

    rows = []
    for sample, metadata, rate_factor in zip(samples, background_metadata, rate_factors):
        summary = _read_analysis_summary_for_var_root(sample["var_root"])
        effective_xsec_fb = sample["xsec_fb"] * float(rate_factor)
        preselection_efficiency = summary.get("preselection_efficiency")
        feature_tree_efficiency = summary.get("feature_tree_efficiency")
        analysis_efficiency = summary.get("analysis_efficiency")
        preselection_xsec_fb = (
            effective_xsec_fb * float(preselection_efficiency)
            if preselection_efficiency is not None
            else None
        )
        output_xsec_fb = (
            effective_xsec_fb * float(analysis_efficiency)
            if analysis_efficiency is not None
            else None
        )
        feature_tree_xsec_fb = (
            effective_xsec_fb * float(feature_tree_efficiency)
            if feature_tree_efficiency is not None
            else None
        )

        rows.append(
            {
                "process_id": sample["process_id"],
                "description": sample["description"],
                "local_lhe": sample["local_lhe"],
                "raw_root": str(sample["raw_root"]),
                "var_root": str(sample["var_root"]),
                "raw_xsec_pb": sample["xsec_pb"],
                "raw_xsec_fb": sample["xsec_fb"],
                "b_quarks": metadata["b_quarks"],
                "c_quarks": metadata["c_quarks"],
                "light_jets": metadata["light_jets"],
                "rate_factor": float(rate_factor),
                "effective_xsec_fb": effective_xsec_fb,
                "mc_events_in": summary.get("mc_events_in"),
                "total_weight_in": summary.get("total_weight_in"),
                "preselection_mc_events_out": summary.get("preselection_mc_events_out"),
                "preselection_weight_out": summary.get("preselection_weight_out"),
                "preselection_efficiency": preselection_efficiency,
                "preselection_xsec_fb": preselection_xsec_fb,
                "feature_tree_mc_events_out": summary.get("feature_tree_mc_events_out"),
                "feature_tree_weight_out": summary.get("feature_tree_weight_out"),
                "feature_tree_efficiency": feature_tree_efficiency,
                "feature_tree_xsec_fb": feature_tree_xsec_fb,
                "analysis_mc_events_out": summary.get("analysis_mc_events_out"),
                "analysis_weight_out": summary.get("analysis_weight_out"),
                "analysis_efficiency": analysis_efficiency,
                "output_xsec_fb": output_xsec_fb,
                "summary_source": summary.get("summary_source"),
                "summary_file": summary.get("summary_file"),
                "status": summary.get("status", "ok" if analysis_efficiency is not None else "missing_analysis_summary"),
            }
        )

    output_csv = _Path(args.background_analysis_summary)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "process_id",
        "description",
        "local_lhe",
        "raw_root",
        "var_root",
        "raw_xsec_pb",
        "raw_xsec_fb",
        "b_quarks",
        "c_quarks",
        "light_jets",
        "rate_factor",
        "effective_xsec_fb",
        "mc_events_in",
        "total_weight_in",
        "preselection_mc_events_out",
        "preselection_weight_out",
        "preselection_efficiency",
        "preselection_xsec_fb",
        "feature_tree_mc_events_out",
        "feature_tree_weight_out",
        "feature_tree_efficiency",
        "feature_tree_xsec_fb",
        "analysis_mc_events_out",
        "analysis_weight_out",
        "analysis_efficiency",
        "output_xsec_fb",
        "summary_source",
        "summary_file",
        "status",
    ]
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_effective_xsec = sum(row["effective_xsec_fb"] for row in rows)
    total_preselection_xsec = sum(row["preselection_xsec_fb"] or 0.0 for row in rows)
    total_output_xsec = sum(row["output_xsec_fb"] or 0.0 for row in rows)

    print("Background analysis cross-section summary")
    print(f"  CSV: {args.background_csv}")
    print(f"  Output: {output_csv}")
    print(f"  Total effective input cross section = {total_effective_xsec:g} fb")
    print(f"  Total pTb/eta/dR preselection cross section = {total_preselection_xsec:g} fb")
    print(f"  Total final output cross section = {total_output_xsec:g} fb")
    print("  Per-process:")
    for row in rows:
        print(
            f"    {row['process_id']}: "
            f"MC {row['mc_events_in']} -> {row['analysis_mc_events_out']}, "
            f"eff={_format_optional_float(row['analysis_efficiency']) or 'missing'}, "
            f"preselection_xsec={_format_optional_float(row['preselection_xsec_fb']) or 'missing'} fb, "
            f"output_xsec={_format_optional_float(row['output_xsec_fb']) or 'missing'} fb"
        )
    return rows


def _ensure_analysis_executable(executable, source_file, rebuild=True):
    import subprocess

    executable = _Path(executable)
    source_file = _Path(source_file) if source_file is not None else None
    needs_build = not executable.exists()
    if source_file is not None and source_file.exists() and executable.exists():
        dependencies = [source_file]
        extended_header = source_file.parent / "Extended91Observables.h"
        if extended_header.exists():
            dependencies.append(extended_header)
        needs_build = any(
            dependency.stat().st_mtime > executable.stat().st_mtime
            for dependency in dependencies
        )

    if rebuild and needs_build and source_file is not None and (source_file.parent / "Makefile").exists():
        print("Building analysis executable:", executable)
        subprocess.run(["make", "-C", str(source_file.parent), executable.name], check=True)

    if not executable.exists():
        raise SystemExit(f"Analysis executable does not exist: {executable}")
    return executable


def _run_one_cpp_analysis(
    raw_root,
    executable,
    max_events=None,
    force=False,
    c_mistags=0,
    light_mistags=0,
    analysis_tag=None,
):
    import subprocess

    raw_root = _Path(raw_root)
    output_root = _analysis_output_root(raw_root, analysis_tag)
    log_file = _analysis_log_file(raw_root, analysis_tag)
    if output_root.exists() and not force:
        return output_root

    command = [str(executable), str(raw_root)]
    if analysis_tag is not None:
        command.extend(["-t", str(analysis_tag)])
    if max_events is not None:
        command.extend(["-n", str(max_events)])
    if c_mistags:
        command.extend(["--c-mistags", str(int(c_mistags))])
    if light_mistags:
        command.extend(["--light-mistags", str(int(light_mistags))])

    print("Running analysis:", " ".join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(_REPO_DIR))
    log_file.write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Analysis failed for {raw_root}; see {log_file}")
    if not output_root.exists():
        raise RuntimeError(f"Analysis completed but did not create {output_root}; see {log_file}")
    return output_root


def _extended_v2_completion_evidence(
    output_root,
    summary,
    *,
    raw_root=None,
    expected_generated_events=None,
    root_module=None,
):
    """Return auditable evidence that the C++ feature tree saw the full source."""

    import math

    try:
        observed_float = float(summary["mc_events_in"])
    except (KeyError, TypeError, ValueError):
        return {
            "verified": False,
            "method": "missing-mc-events-in",
            "observed_events": None,
            "expected_events": None,
        }
    if not math.isfinite(observed_float) or observed_float < 0.0:
        return {
            "verified": False,
            "method": "invalid-mc-events-in",
            "observed_events": observed_float,
            "expected_events": None,
        }
    observed = int(round(observed_float))
    if not math.isclose(observed_float, observed, rel_tol=0.0, abs_tol=1.0e-9):
        return {
            "verified": False,
            "method": "nonintegral-mc-events-in",
            "observed_events": observed_float,
            "expected_events": None,
        }

    candidate_raw = None
    if raw_root is not None:
        candidate_raw = _Path(raw_root)
    elif summary.get("input_file"):
        candidate_raw = _Path(str(summary["input_file"]))
        if not candidate_raw.is_absolute():
            candidate_raw = (_REPO_DIR / candidate_raw).resolve()

    expected = None
    method = None
    evidence_source = None
    if candidate_raw is not None and candidate_raw.exists():
        module = root_module
        if module is None:
            import ROOT as module
        raw_file = module.TFile.Open(str(candidate_raw), "READ")
        if not raw_file or raw_file.IsZombie():
            return {
                "verified": False,
                "method": "raw-root-open-failed",
                "observed_events": observed,
                "expected_events": None,
                "source": str(candidate_raw),
            }
        try:
            raw_tree = raw_file.Get("Data")
            if not raw_tree:
                return {
                    "verified": False,
                    "method": "raw-root-data-tree-missing",
                    "observed_events": observed,
                    "expected_events": None,
                    "source": str(candidate_raw),
                }
            expected = int(raw_tree.GetEntries())
        finally:
            raw_file.Close()
        method = "raw-root-data-entries"
        evidence_source = str(candidate_raw)
    else:
        if expected_generated_events is None:
            _, expected_generated_events, metadata_source = _metadata_for_root_file(
                output_root
            )
        else:
            metadata_source = None
        if expected_generated_events is not None:
            expected = int(expected_generated_events)
            method = "generated-event-metadata"
            evidence_source = (
                None if metadata_source is None else str(metadata_source)
            )

    if expected is None:
        return {
            "verified": False,
            "method": "no-completion-reference",
            "observed_events": observed,
            "expected_events": None,
            "source": None,
        }
    return {
        "verified": observed == expected,
        "method": method,
        "observed_events": observed,
        "expected_events": expected,
        "source": evidence_source,
    }


def _extended_v2_output_is_current(
    output_root,
    raw_root=None,
    source_file=None,
    *,
    expected_c_mistags=None,
    expected_light_mistags=None,
):
    """Validate a tagged output before allowing the study to reuse it."""

    import json
    import math

    output_root = _Path(output_root)
    if not output_root.exists():
        return False
    if not _var_root_matches_analysis_tag(output_root, EXTENDED_V2_TAG):
        return False
    summary_file = _analysis_summary_file_for_var_root(output_root)
    if not summary_file.exists():
        return False
    if summary_file.stat().st_mtime < output_root.stat().st_mtime:
        return False

    dependencies = []
    if raw_root is not None:
        dependencies.append(_Path(raw_root))
    if source_file is not None:
        source_file = _Path(source_file)
        dependencies.append(source_file)
        dependencies.append(source_file.parent / "Extended91Observables.h")
    newest_output_time = min(output_root.stat().st_mtime, summary_file.stat().st_mtime)
    if any(
        dependency.exists() and dependency.stat().st_mtime > newest_output_time
        for dependency in dependencies
    ):
        return False

    try:
        import ROOT
        from observable_schemas import (
            EXTENDED_FEATURE_NAMES,
            EXTENDED_FEATURE_UNITS,
            EXTENDED_SCHEMA_ID,
            PAIRING_COUNT,
        )

        root_file = ROOT.TFile.Open(str(output_root), "READ")
        if not root_file or root_file.IsZombie():
            return False
        try:
            data2 = root_file.Get("Data2")
            data3 = root_file.Get("Data3")
            if not data2 or not data3 or data2.GetEntries() != data3.GetEntries():
                return False
            data3_entries = int(data3.GetEntries())
            if any(
                not data3.GetBranch(branch)
                for branch in (
                    "features",
                    "weight",
                    "event_index",
                    "cut_mask",
                    "passes_legacy_full_selection",
                )
            ):
                return False
            schema = root_file.Get("Data3_observable_schema")
            count = root_file.Get("Data3_feature_count")
            names = root_file.Get("Data3_feature_names_json")
            units = root_file.Get("Data3_feature_units_json")
            pairing_count = root_file.Get("Data3_pairing_count")
            output_tag = root_file.Get("analysis_output_tag")
            smearing_model = root_file.Get("jet_smearing_model_id")
            smearing_acceptance = root_file.Get("jet_smearing_acceptance_order")
            smearing_scaling = root_file.Get("jet_smearing_fourvector_scaling")
            smearing_seed = root_file.Get("jet_smearing_seed")
            smearing_floor = root_file.Get("jet_smearing_min_energy_gev")
            smearing_draws = root_file.Get("jet_smearing_gaussian_draws_per_jet")
            smearing_correlated_mass = root_file.Get(
                "jet_smearing_correlated_mass_scaling"
            )
            smearing_mass_residual = root_file.Get(
                "max_smearing_mass_scaling_residual_gev"
            )
            feature_leaf = data3.GetLeaf("features")
            if not schema or str(schema.GetTitle()) != EXTENDED_SCHEMA_ID:
                return False
            if not count or int(count.GetVal()) != len(EXTENDED_FEATURE_NAMES):
                return False
            if not names or tuple(json.loads(str(names.GetTitle()))) != EXTENDED_FEATURE_NAMES:
                return False
            if not units or tuple(json.loads(str(units.GetTitle()))) != EXTENDED_FEATURE_UNITS:
                return False
            if not pairing_count or int(pairing_count.GetVal()) != PAIRING_COUNT:
                return False
            if not output_tag or str(output_tag.GetTitle()) != EXTENDED_V2_TAG:
                return False
            if not smearing_model or str(smearing_model.GetTitle()) != JET_SMEARING_MODEL_ID:
                return False
            if (
                not smearing_acceptance
                or str(smearing_acceptance.GetTitle())
                != JET_SMEARING_ACCEPTANCE_ORDER
            ):
                return False
            if (
                not smearing_scaling
                or str(smearing_scaling.GetTitle())
                != JET_SMEARING_FOURVECTOR_SCALING
            ):
                return False
            if not smearing_seed or int(smearing_seed.GetVal()) != JET_SMEARING_SEED:
                return False
            if not smearing_floor or not math.isclose(
                float(smearing_floor.GetVal()),
                JET_SMEARING_MIN_ENERGY_GEV,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                return False
            if not smearing_draws or int(smearing_draws.GetVal()) != 1:
                return False
            if (
                not smearing_correlated_mass
                or int(smearing_correlated_mass.GetVal()) != 1
            ):
                return False
            if not smearing_mass_residual:
                return False
            root_mass_residual = float(smearing_mass_residual.GetVal())
            if (
                not math.isfinite(root_mass_residual)
                or root_mass_residual < 0.0
                or root_mass_residual > JET_SMEARING_MAX_MASS_RESIDUAL_GEV
            ):
                return False
            if not feature_leaf or int(feature_leaf.GetLenStatic()) != len(EXTENDED_FEATURE_NAMES):
                return False
        finally:
            root_file.Close()

        summary = json.loads(summary_file.read_text())
        if summary.get("observable_schema") != EXTENDED_SCHEMA_ID:
            return False
        if summary.get("analysis_output_tag") != EXTENDED_V2_TAG:
            return False
        if summary.get("jet_smearing_model_id") != JET_SMEARING_MODEL_ID:
            return False
        if (
            summary.get("jet_smearing_acceptance_order")
            != JET_SMEARING_ACCEPTANCE_ORDER
        ):
            return False
        if (
            summary.get("jet_smearing_fourvector_scaling")
            != JET_SMEARING_FOURVECTOR_SCALING
        ):
            return False
        if summary.get("jet_smearing_correlated_mass_scaling") is not True:
            return False
        if summary.get("jet_smearing_preserves_jet_mass") is not False:
            return False
        if int(summary.get("jet_smearing_gaussian_draws_per_jet", -1)) != 1:
            return False
        if int(summary.get("jet_smearing_seed", -1)) != JET_SMEARING_SEED:
            return False
        summary_smearing_floor = float(summary["jet_smearing_min_energy_gev"])
        if not math.isclose(
            summary_smearing_floor,
            JET_SMEARING_MIN_ENERGY_GEV,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            return False
        summary_mass_residual = float(
            summary["max_smearing_mass_scaling_residual_gev"]
        )
        if (
            not math.isfinite(summary_mass_residual)
            or summary_mass_residual < 0.0
            or summary_mass_residual > JET_SMEARING_MAX_MASS_RESIDUAL_GEV
            or not math.isclose(
                summary_mass_residual,
                root_mass_residual,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            return False
        if raw_root is not None:
            summary_input = _Path(summary["input_file"]).expanduser()
            if not summary_input.is_absolute():
                summary_input = _REPO_DIR / summary_input
            if summary_input.resolve() != _Path(raw_root).expanduser().resolve():
                return False
        if expected_c_mistags is not None and int(summary.get("c_mistags", -1)) != int(
            expected_c_mistags
        ):
            return False
        if expected_light_mistags is not None and int(
            summary.get("light_mistags", -1)
        ) != int(expected_light_mistags):
            return False
        if expected_c_mistags is not None or expected_light_mistags is not None:
            c_mistags = int(expected_c_mistags or 0)
            light_mistags = int(expected_light_mistags or 0)
            if int(summary.get("required_true_bjets", -1)) != 8 - c_mistags - light_mistags:
                return False

        migration_names = (
            "true_b_upward_pt_migrations",
            "true_b_downward_pt_migrations",
            "non_b_upward_pt_migrations",
            "non_b_downward_pt_migrations",
            "true_b_upward_pt_migrations_raw_pt_10_12_gev",
            "true_b_upward_pt_migrations_raw_pt_12_15_gev",
            "true_b_upward_pt_migrations_raw_pt_15_20_gev",
            "non_b_upward_pt_migrations_raw_pt_10_12_gev",
            "non_b_upward_pt_migrations_raw_pt_12_15_gev",
            "non_b_upward_pt_migrations_raw_pt_15_20_gev",
        )
        migrations = {}
        for name in migration_names:
            value = summary.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return False
            migrations[name] = value
        if migrations["true_b_upward_pt_migrations"] != sum(
            migrations[f"true_b_upward_pt_migrations_raw_pt_{pt_bin}_gev"]
            for pt_bin in ("10_12", "12_15", "15_20")
        ):
            return False
        if migrations["non_b_upward_pt_migrations"] != sum(
            migrations[f"non_b_upward_pt_migrations_raw_pt_{pt_bin}_gev"]
            for pt_bin in ("10_12", "12_15", "15_20")
        ):
            return False

        total_weight_in = float(summary["total_weight_in"])
        if not math.isfinite(total_weight_in) or total_weight_in <= 0.0:
            return False
        stage_values = {}
        for stage in ("preselection", "feature_tree", "analysis"):
            entries = float(summary[f"{stage}_mc_events_out"])
            weight = float(summary[f"{stage}_weight_out"])
            efficiency = float(summary[f"{stage}_efficiency"])
            if (
                not math.isfinite(entries)
                or entries < 0.0
                or not entries.is_integer()
                or not math.isfinite(weight)
                or not math.isfinite(efficiency)
                or not math.isclose(
                    efficiency,
                    weight / total_weight_in,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
            ):
                return False
            stage_values[stage] = (int(entries), weight, efficiency)
        if stage_values["feature_tree"][0] != data3_entries:
            return False
        completion = _extended_v2_completion_evidence(
            output_root,
            summary,
            raw_root=raw_root,
            root_module=ROOT,
        )
        return bool(completion["verified"])
    except Exception:
        return False


def _ensure_analysis_var_roots(
    inputs,
    executable,
    source_file,
    include_auxiliary=False,
    jobs=1,
    max_events=None,
    force=False,
    run_missing=True,
    c_mistags=0,
    light_mistags=0,
    analysis_tag=None,
    progress_callback=None,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    existing_var_roots, raw_roots = _discover_analysis_inputs(inputs, analysis_tag=analysis_tag)
    existing_var_roots = _filter_auxiliary_roots(existing_var_roots, include_auxiliary)
    raw_roots = _filter_auxiliary_roots(raw_roots, include_auxiliary)
    print(
        f"Discovered {len(existing_var_roots)} current variable ROOT file(s) and "
        f"{len(raw_roots)} raw ROOT file(s) for analysis",
        flush=True,
    )

    expected_var_roots = [_analysis_output_root(path, analysis_tag) for path in raw_roots]
    raw_by_output = {
        str(var_root.resolve()): raw_root
        for raw_root, var_root in zip(raw_roots, expected_var_roots)
    }
    accepted_var_roots = set()
    missing_raw_roots = []
    for raw_root, var_root in zip(raw_roots, expected_var_roots):
        regenerate = force or not var_root.exists()
        if not regenerate and analysis_tag == EXTENDED_V2_TAG:
            regenerate = not _extended_v2_output_is_current(
                var_root,
                raw_root=raw_root,
                source_file=source_file,
                expected_c_mistags=c_mistags,
                expected_light_mistags=light_mistags,
            )
        if regenerate:
            missing_raw_roots.append(raw_root)
        else:
            accepted_var_roots.add(str(var_root.resolve()))

    invalid_standalone = []
    for var_root in existing_var_roots:
        if str(var_root.resolve()) in raw_by_output:
            continue
        if analysis_tag != EXTENDED_V2_TAG:
            accepted_var_roots.add(str(var_root.resolve()))
            continue
        if force:
            invalid_standalone.append(var_root)
            continue
        if analysis_tag == EXTENDED_V2_TAG and not _extended_v2_output_is_current(
            var_root,
            source_file=source_file,
            expected_c_mistags=c_mistags,
            expected_light_mistags=light_mistags,
        ):
            invalid_standalone.append(var_root)
            continue
        accepted_var_roots.add(str(var_root.resolve()))

    if invalid_standalone:
        paths = "\n  ".join(str(path) for path in invalid_standalone)
        raise SystemExit(
            "Tagged v2 variable ROOT file(s) are stale, incomplete, or schema-invalid, "
            "and no matching raw ROOT input was supplied for regeneration:\n  " + paths
        )

    if missing_raw_roots and not run_missing:
        if analysis_tag == EXTENDED_V2_TAG:
            paths = "\n  ".join(str(path) for path in missing_raw_roots)
            raise SystemExit(
                f"{len(missing_raw_roots)} tagged v2 variable ROOT file(s) require "
                "regeneration, but --no-run-missing-analysis is active:\n  " + paths
            )
        print(
            f"Warning: {len(missing_raw_roots)} raw ROOT files are missing variable "
            "outputs, but auto-analysis is disabled."
        )
        accepted_var_roots.update(
            str(_analysis_output_root(raw_root, analysis_tag).resolve())
            for raw_root in missing_raw_roots
            if _analysis_output_root(raw_root, analysis_tag).exists()
        )
    elif missing_raw_roots:
        executable = _ensure_analysis_executable(executable, source_file, rebuild=True)
        print(
            f"Running C++ analysis for {len(missing_raw_roots)} missing, stale, "
            "or schema-invalid variable ROOT file(s)"
        )
        print(f"  analysis progress 0/{len(missing_raw_roots)}", flush=True)
        jobs = max(1, int(jobs))
        if jobs == 1:
            for index, raw_root in enumerate(missing_raw_roots, start=1):
                _run_one_cpp_analysis(
                    raw_root,
                    executable,
                    max_events=max_events,
                    force=True,
                    c_mistags=c_mistags,
                    light_mistags=light_mistags,
                    analysis_tag=analysis_tag,
                )
                print(
                    f"  analysis progress {index}/{len(missing_raw_roots)}: {raw_root.name}",
                    flush=True,
                )
                if progress_callback is not None:
                    progress_callback(index, len(missing_raw_roots), raw_root.name)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(
                        _run_one_cpp_analysis,
                        raw_root,
                        executable,
                        max_events,
                        True,
                        c_mistags,
                        light_mistags,
                        analysis_tag,
                    ): raw_root
                    for raw_root in missing_raw_roots
                }
                for index, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    raw_root = futures[future]
                    print(
                        f"  analysis progress {index}/{len(missing_raw_roots)}: {raw_root.name}",
                        flush=True,
                    )
                    if progress_callback is not None:
                        progress_callback(index, len(missing_raw_roots), raw_root.name)

    for raw_root in missing_raw_roots:
        var_root = _analysis_output_root(raw_root, analysis_tag)
        if not var_root.exists():
            continue
        if analysis_tag == EXTENDED_V2_TAG and not _extended_v2_output_is_current(
            var_root,
            raw_root=raw_root,
            source_file=source_file,
            expected_c_mistags=c_mistags,
            expected_light_mistags=light_mistags,
        ):
            raise RuntimeError(
                f"Regenerated tagged v2 output failed completeness/schema validation: {var_root}"
            )
        accepted_var_roots.add(str(var_root.resolve()))

    ordered_candidates = [*existing_var_roots, *expected_var_roots]
    final_var_roots = [
        path
        for path in ordered_candidates
        if str(path.resolve()) in accepted_var_roots
    ]
    return _unique_paths(_filter_auxiliary_roots(final_var_roots, include_auxiliary))


def _study_specs(
    files,
    xsecs,
    generated_events,
    normalisation_weights,
    rate_factors,
    metadata=None,
    require_complete_feature_sources=False,
):
    if not isinstance(rate_factors, list):
        rate_factors = [rate_factors for _ in files]
    metadata = metadata or [{} for _ in files]
    specs = []
    for path, xsec, generated, normalisation, rate_factor, sample_metadata in zip(
        files,
        xsecs,
        generated_events,
        normalisation_weights,
        rate_factors,
        metadata,
    ):
        sample_metadata = dict(sample_metadata or {})
        if require_complete_feature_sources:
            summary = _read_analysis_summary_for_var_root(path)
            evidence = _extended_v2_completion_evidence(
                path,
                summary,
                expected_generated_events=generated,
            )
            sample_metadata["feature_source_completion"] = evidence
            if not evidence["verified"]:
                raise SystemExit(
                    "Could not verify that the tagged v2 feature tree uses the complete "
                    f"event source for {path}: {evidence}"
                )
        specs.append({
            "path": path,
            "xsec_fb": xsec,
            "generated_events": generated,
            "normalisation_weight": normalisation,
            "rate_factor": rate_factor,
            "metadata": sample_metadata,
        })
    return specs


def _run_c3d4_xgboost_study_cli_impl(args):
    if int(args.shape_jobs) < 1:
        raise SystemExit("--shape-jobs must be at least one")
    if not _math.isfinite(float(args.progress_interval)) or not float(
        args.progress_interval
    ) > 0.0:
        raise SystemExit("--progress-interval must be finite and positive")
    if args.analysis_max_events is not None:
        raise SystemExit(
            "The v2 study does not allow --analysis-max-events because it can create "
            "truncated files under the shared tagged ROOT names. For a smoke test, use "
            "--study-mode smoke with --max-events or --smoke-max-events."
        )
    reuse_sm_optuna_from = getattr(args, "reuse_sm_optuna_from", None)
    if reuse_sm_optuna_from is not None:
        if args.study_mode != "fast-sm":
            raise SystemExit("--reuse-sm-optuna-from requires --study-mode fast-sm")
        manifest = reuse_sm_optuna_from / "method_manifest.json"
        if not manifest.is_file():
            raise SystemExit(
                "--reuse-sm-optuna-from does not contain method_manifest.json: "
                f"{reuse_sm_optuna_from}"
            )
    if int(args.shape_jobs) > 1:
        import multiprocessing as _multiprocessing

        if _os.name != "posix" or "fork" not in _multiprocessing.get_all_start_methods():
            raise SystemExit(
                "--shape-jobs greater than one requires a POSIX host with the 'fork' "
                "multiprocessing method"
            )
    thread_environment = _configure_parallel_shape_threads(args.shape_jobs)
    if int(args.shape_jobs) > 1:
        print(
            "Capping numerical-library threads to one per pyhf worker:",
            ", ".join(f"{key}={value}" for key, value in thread_environment.items()),
            flush=True,
        )
    from c3d4_xgboost_runner import (
        StudyProgress,
        _resolve_study_mode,
        _validate_study_output_mode,
        run_c3d4_study,
    )

    run_shape_override = False if getattr(args, "no_pyhf", False) else None
    try:
        mode_policy = _resolve_study_mode(
            study_mode=args.study_mode,
            observable_set=args.observable_set,
            feature_profile=args.feature_profile,
            training_strategy=args.training_strategy,
            optuna_trials=args.optuna_trials,
            max_events=args.max_events,
            smoke_max_events=args.smoke_max_events,
            run_shape=run_shape_override,
            hash_inputs=True,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    analysis_tag = EXTENDED_V2_TAG if args.observable_set == "extended-91-v2" else None
    if args.study_outdir.resolve() == args.c3d4_scan_outdir.resolve():
        raise SystemExit("--study-outdir must be separate from the legacy --c3d4-scan-outdir")
    try:
        _validate_study_output_mode(args.study_outdir, mode_policy.name)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    input_progress = StudyProgress(
        args.study_outdir, interval_seconds=float(args.progress_interval)
    )
    input_progress.emit(
        "input-discovery",
        "Starting v2 input discovery and tagged ROOT validation",
        observable_set=args.observable_set,
        study_mode=mode_policy.name,
        result_level=mode_policy.result_level,
        analysis_tag=analysis_tag,
    )

    def root_progress(sample_kind):
        def report(index, total, sample_id):
            input_progress.emit(
                "root-regeneration",
                "Completed tagged ROOT analysis",
                sample_kind=sample_kind,
                sample_id=sample_id,
                completed=index,
                total=total,
            )

        return report

    sm_inputs = args.signal or [_REPO_DIR / "Signals" / "events"]
    sm_files = _ensure_analysis_var_roots(
        sm_inputs,
        executable=args.analysis_exe,
        source_file=args.analysis_source,
        include_auxiliary=args.include_auxiliary_samples,
        jobs=args.analysis_jobs,
        max_events=args.analysis_max_events,
        force=args.force_analysis,
        run_missing=not args.no_run_missing_analysis,
        analysis_tag=analysis_tag,
        progress_callback=root_progress("SM signal"),
    )
    input_progress.emit(
        "input-discovery",
        "Discovered dedicated SM variable ROOT inputs",
        sample_kind="SM signal",
        discovered=len(sm_files),
    )
    exact_sm = [path for path in sm_files if _canonical_sample_name(path) == "HW-gg_hhhh_SM"]
    if exact_sm:
        sm_files = exact_sm
    if len(sm_files) != 1:
        raise SystemExit(
            "The v2 study requires exactly one dedicated production SM sample; "
            f"found {len(sm_files)}: {sm_files}"
        )
    sm_xsecs, sm_generated, sm_normalisation = _infer_scored_signal_metadata(
        sm_files,
        args.signal_xsec_fb,
        None,
        args.c3d4_default_generated_events,
        "dedicated SM signal",
        "--signal-xsec-fb",
    )

    grid_inputs = []
    if args.c3d4_signal_root:
        grid_inputs.extend(args.c3d4_signal_root)
    if args.c3d4_signal_dir:
        grid_inputs.extend(args.c3d4_signal_dir)
    if not grid_inputs:
        grid_inputs.append(_REPO_DIR / "HerwigSignalPoints" / "c3d4_10k" / "events")
    grid_files = _ensure_analysis_var_roots(
        grid_inputs,
        executable=args.analysis_exe,
        source_file=args.analysis_source,
        include_auxiliary=args.include_auxiliary_samples,
        jobs=args.analysis_jobs,
        max_events=args.analysis_max_events,
        force=args.force_analysis,
        run_missing=not args.no_run_missing_analysis,
        analysis_tag=analysis_tag,
        progress_callback=root_progress("c3/d4 signal"),
    )
    input_progress.emit(
        "input-discovery",
        "Discovered c3/d4 variable ROOT inputs",
        sample_kind="c3/d4 signal",
        discovered=len(grid_files),
    )
    grid_xsecs, grid_generated, grid_normalisation = _infer_scored_signal_metadata(
        grid_files,
        args.c3d4_signal_xsec_fb,
        args.c3d4_signal_generated_events,
        args.c3d4_default_generated_events,
        "c3/d4 signal",
        "--c3d4-signal-xsec-fb",
    )

    hhhbb_inputs = []
    if args.hhhbb_signal_root:
        hhhbb_inputs.extend(args.hhhbb_signal_root)
    if args.hhhbb_signal_dir:
        hhhbb_inputs.extend(args.hhhbb_signal_dir)
    hhhbb_files = []
    hhhbb_xsecs = []
    hhhbb_generated = []
    hhhbb_normalisation = []
    hhhbb_metadata = []
    if hhhbb_inputs:
        hhhbb_files = _ensure_analysis_var_roots(
            hhhbb_inputs,
            executable=args.analysis_exe,
            source_file=args.analysis_source,
            include_auxiliary=args.include_auxiliary_samples,
            jobs=args.analysis_jobs,
            max_events=args.analysis_max_events,
            force=args.force_analysis,
            run_missing=not args.no_run_missing_analysis,
            analysis_tag=analysis_tag,
            progress_callback=root_progress("post-fit hhhbb signal"),
        )
        if not hhhbb_files:
            raise SystemExit(
                "No hhhbb ROOT variable files found. Pass the completed "
                "forced-splitting production directories with "
                "--hhhbb-signal-dir."
            )
        (
            hhhbb_xsecs,
            hhhbb_generated,
            hhhbb_normalisation,
        ) = _infer_scored_signal_metadata(
            hhhbb_files,
            args.hhhbb_signal_xsec_fb,
            args.hhhbb_signal_generated_events,
            args.hhhbb_default_generated_events,
            "post-fit hhhbb signal",
            "--hhhbb-signal-xsec-fb",
            metadata_resolver=_metadata_for_hhhbb_scored_signal_root,
        )
        for path, xsec_fb, generated in zip(
            hhhbb_files, hhhbb_xsecs, hhhbb_generated
        ):
            exact_xsec_fb, exact_generated, source = (
                _metadata_for_hhhbb_scored_signal_root(
                    path, args.hhhbb_default_generated_events
                )
            )
            if (
                source is None
                or _Path(source).name != "merge_summary.json"
                or exact_xsec_fb is None
                or exact_generated is None
            ):
                raise SystemExit(
                    "The v2 hhhbb contribution requires the exact weighted "
                    f"merged-LHE metadata for {path}; merge_summary.json was "
                    "not found."
                )
            if not _math.isclose(
                float(xsec_fb),
                float(exact_xsec_fb),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise SystemExit(
                    f"The hhhbb cross section for {path} ({float(xsec_fb):.16g} "
                    "fb) does not match its exact weighted merged-LHE value "
                    f"({float(exact_xsec_fb):.16g} fb)."
                )
            if int(generated) != int(exact_generated):
                raise SystemExit(
                    f"The hhhbb generated-event count for {path} ({generated}) "
                    "does not match merge_summary.json "
                    f"({exact_generated})."
                )
            hhhbb_metadata.append(
                {
                    "process_id": _canonical_sample_name(path),
                    "description": (
                        "full-loop gg -> hhhg with weighted forced "
                        "g -> b bbar splitting"
                    ),
                    "cross_section_source": str(source),
                    "cross_section_source_kind": (
                        "weighted-merged-lhe-merge-summary"
                    ),
                    "cross_section_fb": float(exact_xsec_fb),
                    "generated_events": int(exact_generated),
                    "included_in_training": False,
                    "included_in_threshold_optimization": False,
                    "postfit_signal_component": "hhhbb",
                }
            )
        input_progress.emit(
            "input-discovery",
            "Discovered post-fit hhhbb variable ROOT inputs",
            sample_kind="post-fit hhhbb signal",
            discovered=len(hhhbb_files),
            role="excluded from training and threshold optimization",
        )

    sm_hh4b_inputs = []
    if args.sm_hh4b_signal_root:
        sm_hh4b_inputs.extend(args.sm_hh4b_signal_root)
    if args.sm_hh4b_signal_dir:
        sm_hh4b_inputs.extend(args.sm_hh4b_signal_dir)
    sm_hh4b_files = []
    sm_hh4b_xsecs = []
    sm_hh4b_generated = []
    sm_hh4b_normalisation = []
    sm_hh4b_metadata = []
    if sm_hh4b_inputs:
        sm_hh4b_files = _ensure_analysis_var_roots(
            sm_hh4b_inputs,
            executable=args.analysis_exe,
            source_file=args.analysis_source,
            include_auxiliary=args.include_auxiliary_samples,
            jobs=args.analysis_jobs,
            max_events=args.analysis_max_events,
            force=args.force_analysis,
            run_missing=not args.no_run_missing_analysis,
            analysis_tag=analysis_tag,
            progress_callback=root_progress("post-fit SM hh+4b signal"),
        )
        if len(sm_hh4b_files) != 1:
            raise SystemExit(
                "The v2 SM hh+4b diagnostic requires exactly one completed "
                f"SM variable ROOT file; found {len(sm_hh4b_files)}: "
                f"{sm_hh4b_files}"
            )
        (
            sm_hh4b_xsecs,
            sm_hh4b_generated,
            sm_hh4b_normalisation,
        ) = _infer_scored_signal_metadata(
            sm_hh4b_files,
            args.sm_hh4b_signal_xsec_fb,
            args.sm_hh4b_signal_generated_events,
            args.sm_hh4b_default_generated_events,
            "post-fit SM hh+4b signal",
            "--sm-hh4b-signal-xsec-fb",
            metadata_resolver=_metadata_for_sm_hh4b_scored_signal_root,
        )
        path = sm_hh4b_files[0]
        exact_xsec_fb, exact_generated, source = (
            _metadata_for_sm_hh4b_scored_signal_root(
                path, args.sm_hh4b_default_generated_events
            )
        )
        if (
            source is None
            or _Path(source).name != "sample_metadata.json"
            or exact_xsec_fb is None
            or exact_generated is None
        ):
            raise SystemExit(
                "The v2 SM hh+4b diagnostic requires its trusted "
                f"sample_metadata.json next to the Herwig campaign for {path}."
            )
        if not _math.isclose(
            float(sm_hh4b_xsecs[0]),
            float(exact_xsec_fb),
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise SystemExit(
                "The SM hh+4b cross section does not match its normalized-LHE "
                f"metadata: {float(sm_hh4b_xsecs[0]):.16g} fb versus "
                f"{float(exact_xsec_fb):.16g} fb."
            )
        if int(sm_hh4b_generated[0]) != int(exact_generated):
            raise SystemExit(
                "The SM hh+4b generated-event count does not match its "
                f"normalized-LHE metadata: {sm_hh4b_generated[0]} versus "
                f"{exact_generated}."
            )
        sm_hh4b_c3_fit = None
        fit_option = getattr(
            args,
            "sm_hh4b_c3_xsec_fit",
            DEFAULT_SM_HH4B_C3_XSEC_FIT,
        )
        if fit_option is not None:
            fit_path = _Path(fit_option)
            if fit_path.is_file():
                from hh4b_c3_xsec import (
                    evaluate_hh4b_c3_fit,
                    load_hh4b_c3_fit,
                )

                try:
                    sm_hh4b_c3_fit = load_hh4b_c3_fit(fit_path)
                    evaluate_hh4b_c3_fit(sm_hh4b_c3_fit, 0.0)
                except ValueError as exc:
                    raise SystemExit(
                        f"Invalid SM hh+4b c3 cross-section fit {fit_path}: "
                        f"{exc}"
                    ) from exc
            elif fit_path != DEFAULT_SM_HH4B_C3_XSEC_FIT:
                raise SystemExit(
                    "The requested SM hh+4b c3 cross-section fit does not "
                    f"exist: {fit_path}"
                )
            else:
                print(
                    "Warning: no completed SM hh+4b c3 cross-section fit was "
                    f"found at {fit_path}; retaining the singleton SM-only "
                    "table row."
                )
        sm_hh4b_metadata.append(
            {
                "process_id": "sm_hh4b_heft",
                "description": (
                    "SM HEFT gg -> hh + b bbar b bbar with stable Higgs "
                    "bosons forced to h -> b bbar in Herwig"
                ),
                "cross_section_source": str(source),
                "cross_section_source_kind": "normalized-lhe-sample-metadata",
                "cross_section_fb": float(exact_xsec_fb),
                "generated_events": int(exact_generated),
                "included_in_training": False,
                "included_in_threshold_optimization": False,
                "included_in_shape_binning_optimization": False,
                "included_in_background": False,
                "included_in_limits": False,
                "cross_section_fit_applied": sm_hh4b_c3_fit is not None,
                "c3_cross_section_fit": sm_hh4b_c3_fit,
                "postfit_signal_component": "sm_hh4b",
            }
        )
        input_progress.emit(
            "input-discovery",
            "Discovered the post-fit SM hh+4b variable ROOT input",
            sample_kind="post-fit SM hh+4b signal",
            discovered=1,
            role=(
                "single SM efficiency result, excluded from training, "
                "backgrounds, shape optimization, and limits"
            ),
        )

    if args.background:
        _validate_explicit_background_composition(
            args.background,
            args.background_csv,
            args.analysis_c_mistags,
            args.analysis_light_mistags,
            analysis_tag=analysis_tag,
        )
        background_files = _ensure_analysis_var_roots(
            args.background,
            executable=args.analysis_exe,
            source_file=args.analysis_source,
            include_auxiliary=args.include_auxiliary_samples,
            jobs=args.analysis_jobs,
            max_events=args.analysis_max_events,
            force=args.force_analysis,
            run_missing=not args.no_run_missing_analysis,
            c_mistags=args.analysis_c_mistags,
            light_mistags=args.analysis_light_mistags,
            analysis_tag=analysis_tag,
            progress_callback=root_progress("background"),
        )
        background_metadata = _metadata_for_background_files(background_files, args.background_csv)
        background_xsecs = _expand_cli_values(
            args.background_xsec_fb, background_files, "background cross-section"
        )
        if background_xsecs is None:
            background_xsecs = []
            background_generated = []
            for path in background_files:
                xsec, generated, out_file = _metadata_for_root_file(path)
                if xsec is None:
                    raise SystemExit(
                        f"Could not infer background cross section from {out_file}; "
                        "pass --background-xsec-fb once per file"
                    )
                background_xsecs.append(xsec)
                background_generated.append(generated)
        else:
            background_generated = [
                _metadata_for_root_file(path)[1] for path in background_files
            ]
        background_normalisation = [
            _normalisation_weight_for_var_root(path) for path in background_files
        ]
    else:
        (
            background_files,
            background_xsecs,
            background_generated,
            background_normalisation,
            background_metadata,
        ) = _background_inputs_from_csv(
            args,
            ensure_analysis=True,
            analysis_tag=analysis_tag,
            progress_callback=root_progress("background"),
        )
    input_progress.emit(
        "input-discovery",
        "Discovered background variable ROOT inputs",
        sample_kind="background",
        discovered=len(background_files),
    )

    signal_rate_factor = _signal_final_rate_factor_for_cli(args)
    hhhbb_rate_factor = _hhhbb_signal_rate_factor_for_cli(args)
    sm_hh4b_rate_factor = _hhbbbb_signal_rate_factor_for_cli(args)
    background_rate_factors = _background_rate_factors_for_cli(background_metadata, args)
    sm_specs = _study_specs(
        sm_files,
        sm_xsecs,
        sm_generated,
        sm_normalisation,
        signal_rate_factor,
        _signal_metadata_for_files(sm_files),
        require_complete_feature_sources=args.observable_set == "extended-91-v2",
    )
    grid_specs = _study_specs(
        grid_files,
        grid_xsecs,
        grid_generated,
        grid_normalisation,
        signal_rate_factor,
        require_complete_feature_sources=args.observable_set == "extended-91-v2",
    )
    hhhbb_specs = _study_specs(
        hhhbb_files,
        hhhbb_xsecs,
        hhhbb_generated,
        hhhbb_normalisation,
        hhhbb_rate_factor,
        hhhbb_metadata,
        require_complete_feature_sources=args.observable_set == "extended-91-v2",
    )
    sm_hh4b_specs = _study_specs(
        sm_hh4b_files,
        sm_hh4b_xsecs,
        sm_hh4b_generated,
        sm_hh4b_normalisation,
        sm_hh4b_rate_factor,
        sm_hh4b_metadata,
        require_complete_feature_sources=args.observable_set == "extended-91-v2",
    )
    background_specs = _study_specs(
        background_files,
        background_xsecs,
        background_generated,
        background_normalisation,
        background_rate_factors,
        background_metadata,
        require_complete_feature_sources=args.observable_set == "extended-91-v2",
    )
    print("Resolved-8b c3/d4 XGBoost v2 inputs")
    print("  observable set:", args.observable_set)
    print("  study mode:", mode_policy.name)
    print("  result level:", mode_policy.result_level)
    print("  feature profile:", mode_policy.feature_profile or "validation-selected")
    print("  Optuna trials per fold:", mode_policy.optuna_trials)
    print("  reused SM Optuna study:", args.reuse_sm_optuna_from)
    print("  Python event cap per source:", mode_policy.max_events)
    print("  pyhf score shapes:", mode_policy.run_shape)
    print("  dedicated SM samples:", len(sm_specs))
    print("  c3/d4 samples:", len(grid_specs))
    print("  post-fit hhhbb samples:", len(hhhbb_specs))
    if hhhbb_specs:
        print(
            "  post-fit hhhbb role: excluded from training/threshold/binning "
            "optimization; added only to the final cut and pyhf signal templates"
        )
    print("  post-fit SM hh+4b samples:", len(sm_hh4b_specs))
    if sm_hh4b_specs:
        role = (
            "one frozen SM efficiency evaluated with the c3 cross-section fit "
            "at every hhhbb table point"
            if sm_hh4b_metadata[0].get("cross_section_fit_applied")
            else "one SM efficiency result"
        )
        print(
            f"  post-fit SM hh+4b role: {role} after the "
            "classifier/threshold are frozen; excluded from training, "
            "backgrounds, shape optimization, and limits"
        )
    print("  background samples:", len(background_specs))
    print("  output:", args.study_outdir)
    print("  shape workers:", args.shape_jobs)
    print("  progress interval [s]:", args.progress_interval)
    input_progress.emit(
        "input-discovery",
        "Completed v2 input discovery and normalization lookup",
        dedicated_sm_samples=len(sm_specs),
        c3d4_samples=len(grid_specs),
        postfit_hhhbb_samples=len(hhhbb_specs),
        postfit_sm_hh4b_samples=len(sm_hh4b_specs),
        background_samples=len(background_specs),
    )
    summary = run_c3d4_study(
        sm_signal_specs=sm_specs,
        grid_signal_specs=grid_specs,
        hhhbb_signal_specs=hhhbb_specs,
        sm_hh4b_signal_specs=sm_hh4b_specs,
        background_specs=background_specs,
        output_dir=args.study_outdir,
        observable_set=args.observable_set,
        feature_profile=args.feature_profile,
        training_strategy=args.training_strategy,
        cv_folds=args.cv_folds,
        optuna_trials=args.optuna_trials,
        luminosity=args.luminosity,
        seed=args.seed,
        max_events=args.max_events,
        legacy_scan_csv=args.c3d4_scan_outdir / "c3d4_limit_scan.csv",
        repo_dir=_REPO_DIR,
        run_shape=mode_policy.run_shape,
        shape_jobs=args.shape_jobs,
        progress_interval=args.progress_interval,
        study_mode=args.study_mode,
        smoke_max_events=args.smoke_max_events,
        reuse_sm_optuna_from=args.reuse_sm_optuna_from,
        contour_c3_range=(args.c3d4_plot_c3_min, args.c3d4_plot_c3_max),
        contour_d4_range=(args.c3d4_plot_d4_min, args.c3d4_plot_d4_max),
        contour_grid_bins=args.c3d4_plot_nbins,
        contour_interpolation=args.c3d4_contour_interpolation,
        xsec_source_dir=args.c3d4_xsec_source_dir,
        xsec_overlay=not args.no_c3d4_xsec_overlay,
        write_input_report=not args.no_sample_report,
    )
    print("v2 study complete; selected profile =", summary["selected_feature_profile"])
    return 0


def _run_c3d4_xgboost_study_cli(args):
    """Run the v2 CLI while keeping preprocessing failures visible to monitors."""

    try:
        return _run_c3d4_xgboost_study_cli_impl(args)
    except BaseException as error:
        try:
            if args.study_outdir.resolve() != args.c3d4_scan_outdir.resolve():
                from c3d4_xgboost_runner import _record_study_failure

                _record_study_failure(
                    _Path(args.study_outdir),
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error,
                    study_mode=getattr(args, "study_mode", "full"),
                )
        except Exception:
            pass
        raise


def _run_local_xgboost_cli():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Train a 4H XGBoost signal-vs-background classifier from local ROOT variable files."
    )
    parser.add_argument("--signal", action="append", type=_Path, help="Signal Data2 ROOT file. May be repeated.")
    parser.add_argument("--background", action="append", type=_Path, help="Background Data2 ROOT file. May be repeated.")
    parser.add_argument("--include-auxiliary-samples", action="store_true", help="Include debug/smoke ROOT files in discovery.")
    parser.add_argument("--outdir", type=_Path, default=_REPO_DIR / "xgboost_results", help="Directory for model and plots.")
    parser.add_argument("--sample-report-dir", type=_Path, default=None, help="Directory for the LaTeX table and observable-plot webpage.")
    parser.add_argument("--no-sample-report", action="store_true", help="Disable the SM/background cutflow table and observable-plot webpage.")
    parser.add_argument("--luminosity", type=float, default=3000.0, help="Integrated luminosity in fb^-1.")
    parser.add_argument("--systematics", type=float, default=0.0, help="Fractional background systematic for threshold scan.")
    parser.add_argument("--hbb-branching-ratio", type=float, default=DEFAULT_HBB_BRANCHING_RATIO, help="Higgs to b bbar branching ratio used in the c3/d4 limit scan.")
    parser.add_argument("--zbb-branching-ratio", type=float, default=DEFAULT_ZBB_BRANCHING_RATIO, help="Z to b bbar branching ratio applied only to Z backgrounds not already generated with Z -> b bbar.")
    parser.add_argument("--btagging-rate", type=float, default=DEFAULT_BTAGGING_RATE, help="Per-b b-tagging rate used in the c3/d4 limit scan.")
    parser.add_argument("--c-mistag-rate", type=float, default=0.1, help="Per-c-jet charm mistag rate applied to CSV background rates.")
    parser.add_argument("--light-mistag-rate", type=float, default=0.01, help="Per-light-jet mistag rate applied to CSV background rates.")
    parser.add_argument("--signal-hbb-power", type=int, default=DEFAULT_SIGNAL_HBB_POWER, help="Power of BR(h->bb) applied to signal rates.")
    parser.add_argument("--signal-btag-power", type=int, default=DEFAULT_EIGHT_BTAG_POWER, help="Power of the b-tagging rate applied to signal rates.")
    parser.add_argument("--background-hbb-power", type=int, default=0, help="Power of BR(h->bb) applied to background rates.")
    parser.add_argument("--background-btag-power", type=int, default=DEFAULT_EIGHT_BTAG_POWER, help="Power of the b-tagging rate applied to background rates.")
    parser.add_argument("--signal-k-factor", type=float, default=DEFAULT_SIGNAL_K_FACTOR, help="Multiplicative K-factor applied to signal cross sections.")
    parser.add_argument("--background-k-factor", type=float, default=DEFAULT_BACKGROUND_K_FACTOR, help="Multiplicative K-factor applied to background cross sections.")
    parser.add_argument("--test-size", type=float, default=0.35, help="Held-out test fraction.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional maximum events read per file.")
    parser.add_argument(
        "--min-xgb-background-mc",
        type=int,
        default=25,
        help="Minimum raw held-out background MC entries required after the optimized XGBoost threshold. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-xgb-background-effective-mc",
        type=float,
        default=10.0,
        help="Minimum effective held-out background MC entries required after the optimized XGBoost threshold. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-xgb-background-mc-per-sample",
        type=int,
        default=0,
        help="Optional minimum held-out background MC entries per background source after the optimized XGBoost threshold.",
    )
    parser.add_argument("--signal-xsec-fb", action="append", type=float, help="Signal cross section in fb. May be repeated.")
    parser.add_argument("--background-xsec-fb", action="append", type=float, help="Background cross section in fb. May be repeated.")
    parser.add_argument(
        "--background-csv",
        type=_Path,
        default=DEFAULT_BACKGROUND_CSV,
        help="CSV file defining default background samples. Used when --background is not supplied.",
    )
    parser.add_argument(
        "--summarize-background-analysis",
        action="store_true",
        help="Write a CSV summary of CSV-background analysis efficiencies and cross sections, then exit.",
    )
    parser.add_argument(
        "--background-analysis-summary",
        type=_Path,
        default=_REPO_DIR / "Backgrounds" / "background_analysis_summary.csv",
        help="CSV output path for --summarize-background-analysis.",
    )
    parser.add_argument("--prepare-mg5-dir", type=_Path, help="Write a manifest for an MG5 gg_4h_c3d4 Events directory and exit.")
    parser.add_argument(
        "--mg5-manifest",
        type=_Path,
        default=_REPO_DIR / "xgboost_results" / "mg5_c3d4_signal_manifest.csv",
        help="Manifest path used with --prepare-mg5-dir.",
    )
    parser.add_argument(
        "--prepare-herwig-inputs",
        type=_Path,
        help="Prepare Herwig .in files for MG5 gg_4h_c3d4 run directories and exit.",
    )
    parser.add_argument(
        "--prepare-background-herwig-inputs",
        action="store_true",
        help="Prepare Backgrounds/HW-<process_id>.in files from --background-csv and exit.",
    )
    parser.add_argument(
        "--herwig-template",
        type=_Path,
        default=_REPO_DIR / "Signals" / "HW-gg_hhhh_SM.in",
        help="Template Herwig input file used by --prepare-herwig-inputs.",
    )
    parser.add_argument(
        "--herwig-outdir",
        type=_Path,
        default=_REPO_DIR / "HerwigSignalPoints" / "c3d4",
        help="Directory where prepared Herwig .in files and future run outputs live.",
    )
    parser.add_argument(
        "--herwig-manifest",
        type=_Path,
        default=_REPO_DIR / "HerwigSignalPoints" / "c3d4" / "herwig_inputs_manifest.csv",
        help="CSV manifest written by --prepare-herwig-inputs.",
    )
    parser.add_argument(
        "--background-herwig-template",
        type=_Path,
        default=DEFAULT_BACKGROUND_HERWIG_TEMPLATE,
        help="Template Herwig input file used by --prepare-background-herwig-inputs.",
    )
    parser.add_argument(
        "--background-herwig-outdir",
        type=_Path,
        default=_REPO_DIR / "Backgrounds",
        help="Directory where background Herwig .in files and future run outputs live.",
    )
    parser.add_argument(
        "--background-herwig-manifest",
        type=_Path,
        default=_REPO_DIR / "Backgrounds" / "background_herwig_inputs_manifest.csv",
        help="CSV manifest written by --prepare-background-herwig-inputs.",
    )
    parser.add_argument(
        "--background-herwig-input-list",
        type=_Path,
        default=_REPO_DIR / "Backgrounds" / "herwig_background_inputs_to_run.txt",
        help="Text file of background Herwig inputs written by --prepare-background-herwig-inputs.",
    )
    parser.add_argument(
        "--overwrite-herwig-inputs",
        action="store_true",
        help="Overwrite Herwig .in files even when prior .in/.run/.out/.log/root targets exist.",
    )
    parser.add_argument(
        "--include-duplicate-herwig-points",
        action="store_true",
        help="Prepare every MG5 run directory instead of selecting one run per unique c3/d4 point.",
    )
    parser.add_argument("--herwig-nevents", type=int, default=10000, help="NumberOfEvents value written to prepared Herwig inputs.")
    parser.add_argument(
        "--herwig-required-generated-events",
        type=int,
        default=10000,
        help="Only prepare/select MG5 run directories whose banner reports this generated-event count.",
    )
    parser.add_argument(
        "--herwig-output-location",
        default="events/",
        help="HwSim OutputLocation written to prepared Herwig inputs, relative to --herwig-outdir.",
    )
    parser.add_argument("--herwig-run-prefix", default="HW", help="Prefix for generated Herwig run names.")
    parser.add_argument("--score-signal-root", action="append", type=_Path, help="Additional signal-point _var.smear*.root file to score.")
    parser.add_argument("--score-signal-dir", action="append", type=_Path, help="Directory searched recursively for signal-point _var.smear*.root files.")
    parser.add_argument(
        "--score-outdir",
        type=_Path,
        default=_REPO_DIR / "xgboost_signal_scores",
        help="Directory for additional signal-point score summaries.",
    )
    parser.add_argument(
        "--model-file",
        type=_Path,
        default=_REPO_DIR / "xgboost_results" / "signal_background_xgboost.json",
        help="Trained XGBoost model used for --score-signal-*.",
    )
    parser.add_argument(
        "--metrics-file",
        type=_Path,
        default=_REPO_DIR / "xgboost_results" / "metrics.json",
        help="Metrics file used to read the best threshold when --threshold is omitted.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Signal score threshold for --score-signal-*.")
    parser.add_argument("--score-signal-xsec-fb", action="append", type=float, help="Cross section in fb for scored signal files.")
    parser.add_argument("--score-signal-generated-events", action="append", type=int, help="Generated event counts for scored signal files.")
    parser.add_argument(
        "--score-default-generated-events",
        type=int,
        default=None,
        help="Fallback generated-event count for scored signal files when it cannot be read from the MG5 banner.",
    )
    parser.add_argument("--write-event-scores", action="store_true", help="Also write per-event score CSV for --score-signal-*.")
    parser.add_argument(
        "--run-c3d4-limit-scan",
        action="store_true",
        help="Train the SM signal-vs-background XGBoost model, score c3/d4 signal points, and plot the Poisson 95%% CL region.",
    )
    parser.add_argument(
        "--run-c3d4-xgboost-study",
        action="store_true",
        help="Run the versioned five-fold c3/d4 XGBoost and CLs study without modifying legacy outputs.",
    )
    parser.add_argument(
        "--replot-c3d4-study-contours",
        action="store_true",
        help=(
            "Generate the legacy-style cut/shape exclusion contour family from existing "
            "v2 JSON tables without retraining or rerunning pyhf."
        ),
    )
    parser.add_argument(
        "--write-c3d4-v2-input-report",
        action="store_true",
        help=(
            "Add normalized and legacy-style stacked input-observable plots to an "
            "already completed v2 study without retraining."
        ),
    )
    parser.add_argument(
        "--study-mode",
        choices=(
            "smoke",
            "preview",
            "fast-sm",
            "fast-pooled",
            "fast-parameterized",
            "full",
        ),
        default="full",
        help=(
            "v2 execution level: smoke uses truncated feature-tree reads and is non-physics; "
            "preview uses all events with fixed parameters and cut-only limits; fast-sm "
            "fast-pooled and fast-parameterized use all events, one fixed-parameter "
            "cross-fit strategy, and pyhf score shapes unless --no-pyhf is passed; "
            "fast-parameterized also runs a coupling-point holdout diagnostic; full "
            "runs the complete tuning and gated parameterized workflow."
        ),
    )
    parser.add_argument(
        "--observable-set",
        choices=("legacy-28-v1", "extended-91-v2"),
        default="extended-91-v2",
        help="Immutable observable schema used by the v2 study.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=("corrected28", "core52", "full91"),
        default=None,
        help=(
            "Force one v2 feature profile. If omitted, full mode selects globally from "
            "all three on validation folds, fast-sm/fast-pooled/fast-parameterized "
            "use full91, preview uses core52, and smoke uses corrected28."
        ),
    )
    parser.add_argument(
        "--training-strategy",
        choices=("sm-crossfit-v2", "pooled-crossfit-v2", "parameterized-crossfit-v1"),
        default=None,
        help=(
            "Primary v2 classifier strategy. Defaults to SM for smoke/fast-sm, pooled "
            "for preview/fast-pooled/full, and parameterized for fast-parameterized. "
            "The fast modes evaluate only their named strategy; preview/full also "
            "evaluate the SM baseline."
        ),
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of deterministic rotating cross-fit folds.")
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=None,
        help="Sequential Optuna trials per fold. Defaults to 40 in full mode and 0 otherwise.",
    )
    parser.add_argument(
        "--reuse-sm-optuna-from",
        type=_Path,
        default=None,
        help=(
            "Completed v2 study directory whose five fold-specific SM Optuna best "
            "trials are reused as frozen XGBoost parameters in fast-sm mode."
        ),
    )
    parser.add_argument(
        "--smoke-max-events",
        type=int,
        default=2000,
        help="Maximum Data3 entries read per source in smoke mode when --max-events is omitted.",
    )
    parser.add_argument(
        "--shape-jobs",
        type=int,
        default=1,
        help="Independent pyhf score-shape worker processes; default 1 preserves serial resource use.",
    )
    parser.add_argument(
        "--no-pyhf",
        "--no-shape-limits",
        dest="no_pyhf",
        action="store_true",
        help=(
            "Skip the pyhf score-shape stage and write exact single-bin cut limits "
            "only. In fast-sm, fast-pooled, fast-parameterized, or full mode the "
            "result is physically normalized but preliminary and watermarked."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between v2 progress heartbeats while pyhf workers are running.",
    )
    parser.add_argument(
        "--study-outdir",
        type=_Path,
        default=None,
        help=(
            "Output directory for the versioned study. Mode-specific defaults keep smoke and "
            "preview products separate from "
            "xgboost_c3d4_study_v2_uniform-smear-v1."
        ),
    )
    parser.add_argument(
        "--c3d4-signal-root",
        action="append",
        type=_Path,
        help="c3/d4 signal-point _var.smear*.root file for --run-c3d4-limit-scan. May be repeated.",
    )
    parser.add_argument(
        "--c3d4-signal-dir",
        action="append",
        type=_Path,
        help="Directory searched recursively for c3/d4 _var.smear*.root files. Defaults to HerwigSignalPoints/c3d4_10k/events.",
    )
    parser.add_argument(
        "--c3d4-scan-outdir",
        type=_Path,
        default=_REPO_DIR / "xgboost_c3d4_scan",
        help="Directory for SM optimization, c3/d4 score summaries, and limit plots.",
    )
    parser.add_argument(
        "--c3d4-sm-outdir",
        type=_Path,
        default=None,
        help="Directory for the SM-trained XGBoost model. Defaults to --c3d4-scan-outdir/sm_optimization.",
    )
    parser.add_argument(
        "--c3d4-cl-target",
        type=float,
        default=2.0,
        help="Gaussian S/sqrt(B) reference threshold kept in the c3/d4 scan outputs.",
    )
    parser.add_argument(
        "--poisson-cl",
        type=float,
        default=0.95,
        help="Poisson confidence level for the c3/d4 expected exclusion target.",
    )
    parser.add_argument(
        "--poisson-limit-method",
        choices=("cls", "classical"),
        default="cls",
        help="Poisson upper-limit construction for the c3/d4 target.",
    )
    parser.add_argument(
        "--poisson-observed-events",
        type=int,
        default=None,
        help="Observed event count for the Poisson target. Defaults to the median expected background count.",
    )
    parser.add_argument(
        "--background-variation-factor",
        type=float,
        default=4.0,
        help="Multiplicative background normalization factor used for the shaded c3/d4 exclusion band.",
    )
    parser.add_argument(
        "--no-background-variation-band",
        action="store_true",
        help="Disable the shaded c3/d4 exclusion band from varying the total background normalization.",
    )
    parser.add_argument("--c3d4-signal-xsec-fb", action="append", type=float, help="Cross section in fb for c3/d4 scan files.")
    parser.add_argument("--c3d4-signal-generated-events", action="append", type=int, help="Generated event counts for c3/d4 scan files.")
    parser.add_argument(
        "--c3d4-default-generated-events",
        type=int,
        default=10000,
        help="Fallback generated-event count for c3/d4 scan files.",
    )
    parser.add_argument(
        "--hhhbb-signal-root",
        action="append",
        type=_Path,
        help="hhhbb forced-splitting signal _var.smear*.root or raw ROOT file. May be repeated.",
    )
    parser.add_argument(
        "--hhhbb-signal-dir",
        action="append",
        type=_Path,
        help=(
            "Directory searched recursively for hhhbb ROOT files. They are excluded "
            "from XGBoost training and threshold optimization, then scored and added "
            "only to final c3/d4 cut and pyhf signal templates."
        ),
    )
    parser.add_argument("--hhhbb-signal-xsec-fb", action="append", type=float, help="Cross section in fb for hhhbb signal files.")
    parser.add_argument("--hhhbb-signal-generated-events", action="append", type=int, help="Generated event counts for hhhbb signal files.")
    parser.add_argument(
        "--hhhbb-default-generated-events",
        type=int,
        default=10000,
        help="Fallback generated-event count for hhhbb signal files.",
    )
    parser.add_argument(
        "--hhbbbb-signal-root",
        action="append",
        type=_Path,
        help="hhbbbb forced-splitting signal _var.smear*.root or raw ROOT file. May be repeated.",
    )
    parser.add_argument(
        "--hhbbbb-signal-dir",
        action="append",
        type=_Path,
        help="Directory searched recursively for c3-only hhbbbb ROOT files scored and added only to final c3/d4 limits.",
    )
    parser.add_argument("--hhbbbb-signal-xsec-fb", action="append", type=float, help="Cross section in fb for hhbbbb signal files.")
    parser.add_argument("--hhbbbb-signal-generated-events", action="append", type=int, help="Generated event counts for hhbbbb signal files.")
    parser.add_argument(
        "--hhbbbb-default-generated-events",
        type=int,
        default=10000,
        help="Fallback generated-event count for hhbbbb signal files.",
    )
    parser.add_argument(
        "--sm-hh4b-signal-root",
        action="append",
        type=_Path,
        help=(
            "SM hh+4b HEFT _var.smear*.root or raw ROOT file. The v2 study "
            "requires one (c3,d4)=(0,0) sample and reports it only as a "
            "post-training signal-efficiency diagnostic."
        ),
    )
    parser.add_argument(
        "--sm-hh4b-signal-dir",
        action="append",
        type=_Path,
        help=(
            "Directory searched recursively for the singleton SM hh+4b ROOT "
            "sample. It is excluded from training, backgrounds, shape "
            "optimization, and limits."
        ),
    )
    parser.add_argument(
        "--sm-hh4b-signal-xsec-fb",
        action="append",
        type=float,
        help="Cross section in fb for the singleton SM hh+4b signal file.",
    )
    parser.add_argument(
        "--sm-hh4b-signal-generated-events",
        action="append",
        type=int,
        help="Generated-event count for the singleton SM hh+4b signal file.",
    )
    parser.add_argument(
        "--sm-hh4b-default-generated-events",
        type=int,
        default=10000,
        help="Fallback generated-event count for the singleton SM hh+4b file.",
    )
    parser.add_argument(
        "--sm-hh4b-c3-xsec-fit",
        type=_Path,
        default=DEFAULT_SM_HH4B_C3_XSEC_FIT,
        help=(
            "Quadratic raw-generator cross-section fit used to rescale the "
            "singleton SM hh+4b efficiency at the same c3/d4 table points as "
            "hhh+bb. The standard fit is used automatically when present."
        ),
    )
    parser.add_argument("--no-c3d4-chebyshev-fit", action="store_true", help="Disable the Chebyshev-Lobatto sigma*eff fit and plot only scored points.")
    parser.add_argument("--c3d4-fit-k3-min", type=float, default=-29.0, help="Minimum k3=1+c3 used to scale the Chebyshev fit.")
    parser.add_argument("--c3d4-fit-k3-max", type=float, default=31.0, help="Maximum k3=1+c3 used to scale the Chebyshev fit.")
    parser.add_argument("--c3d4-fit-k4-min", type=float, default=-699.0, help="Minimum k4=1+d4 used to scale the Chebyshev fit.")
    parser.add_argument("--c3d4-fit-k4-max", type=float, default=701.0, help="Maximum k4=1+d4 used to scale the Chebyshev fit.")
    parser.add_argument("--c3d4-plot-c3-min", type=float, default=-20.0, help="Minimum c3 shown in the fitted limit plot.")
    parser.add_argument("--c3d4-plot-c3-max", type=float, default=20.0, help="Maximum c3 shown in the fitted limit plot.")
    parser.add_argument("--c3d4-plot-d4-min", type=float, default=-300.0, help="Minimum d4 shown in the fitted limit plot.")
    parser.add_argument("--c3d4-plot-d4-max", type=float, default=300.0, help="Maximum d4 shown in the fitted limit plot.")
    parser.add_argument("--c3d4-plot-nbins", type=int, default=301, help="Number of bins per axis for fitted c3/d4 plots.")
    parser.add_argument(
        "--c3d4-contour-interpolation",
        choices=("linear", "clough-tocher"),
        default="linear",
        help=(
            "Interpolation of log10(sigma/sigma95) for v2 paper-style exclusion "
            "contours. Linear is the assumption-minimal default; clough-tocher "
            "uses a smooth piecewise-cubic surface inside the sampled convex hull."
        ),
    )
    parser.add_argument(
        "--c3d4-xsec-source-dir",
        type=_Path,
        default=_Path("/mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_4h_c3d4"),
        help="MG5 gg_4h_c3d4 directory used for the hhhh cross-section plot with the 95%% CL overlay.",
    )
    parser.add_argument(
        "--hhh-xsec-source-dir",
        type=_Path,
        default=_Path("/mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_hhh_c3d4"),
        help="MG5 gg_hhh_c3d4 directory used for the sigma(hhhh)/sigma(hhh) contour plot.",
    )
    parser.add_argument(
        "--no-c3d4-xsec-overlay",
        action="store_true",
        help="Do not write the hhhh cross-section plot or hhhh/hhh ratio contours.",
    )
    parser.add_argument(
        "--analysis-exe",
        type=_Path,
        default=_CODE_DIR / "FourHiggs8bAnalysis_smear_CMS",
        help="C++ analysis executable used to create missing *_var.smearCMS.root files.",
    )
    parser.add_argument(
        "--analysis-source",
        type=_Path,
        default=_CODE_DIR / "FourHiggs8bAnalysis_smear_CMS.cc",
        help="C++ analysis source used to rebuild --analysis-exe when it is stale or missing.",
    )
    parser.add_argument("--analysis-jobs", type=int, default=1, help="Concurrent C++ analysis jobs for missing variable ROOT files.")
    parser.add_argument("--analysis-max-events", type=int, default=None, help="Optional max events passed to the C++ analysis with -n.")
    parser.add_argument("--analysis-c-mistags", type=int, default=0, help="C++ analyzer c-mistag count for explicit --background raw ROOT files.")
    parser.add_argument("--analysis-light-mistags", type=int, default=0, help="C++ analyzer light-mistag count for explicit --background raw ROOT files.")
    parser.add_argument("--force-analysis", action="store_true", help="Rerun the C++ analysis even when *_var.smearCMS.root exists.")
    parser.add_argument(
        "--no-run-missing-analysis",
        action="store_true",
        help="Do not create missing *_var.smearCMS.root files before scoring.",
    )

    args = parser.parse_args()

    _configure_v2_mode_defaults(args)

    if args.prepare_mg5_dir is not None:
        _write_mg5_c3d4_manifest(args.prepare_mg5_dir, args.mg5_manifest)
        return 0

    if args.prepare_herwig_inputs is not None:
        _prepare_herwig_inputs(
            process_dir=args.prepare_herwig_inputs,
            output_dir=args.herwig_outdir,
            template_file=args.herwig_template,
            manifest_file=args.herwig_manifest,
            overwrite=args.overwrite_herwig_inputs,
            nevents=args.herwig_nevents,
            output_location=args.herwig_output_location,
            run_prefix=args.herwig_run_prefix,
            unique_points=not args.include_duplicate_herwig_points,
            required_generated_events=args.herwig_required_generated_events,
        )
        return 0

    if args.prepare_background_herwig_inputs:
        _prepare_background_herwig_inputs(
            csv_file=args.background_csv,
            output_dir=args.background_herwig_outdir,
            template_file=args.background_herwig_template,
            manifest_file=args.background_herwig_manifest,
            input_list_file=args.background_herwig_input_list,
            overwrite=args.overwrite_herwig_inputs,
            output_location=args.herwig_output_location,
        )
        return 0

    if args.summarize_background_analysis:
        _summarize_background_analysis(args)
        return 0

    if args.write_c3d4_v2_input_report:
        conflicts = []
        if args.run_c3d4_xgboost_study:
            conflicts.append("--run-c3d4-xgboost-study")
        if args.replot_c3d4_study_contours:
            conflicts.append("--replot-c3d4-study-contours")
        if args.run_c3d4_limit_scan:
            conflicts.append("--run-c3d4-limit-scan")
        if conflicts:
            raise SystemExit(
                "--write-c3d4-v2-input-report is mutually exclusive with "
                + " and ".join(conflicts)
            )
        from c3d4_xgboost_runner import write_c3d4_input_report_from_manifest

        try:
            report = write_c3d4_input_report_from_manifest(args.study_outdir)
        except (OSError, ValueError, RuntimeError) as error:
            raise SystemExit(f"Cannot write v2 input-observable report: {error}") from None
        print("v2 input-observable report:", report["index"])
        print("plots written:", report["plot_count"])
        return 0

    if args.replot_c3d4_study_contours:
        conflicts = []
        if args.run_c3d4_xgboost_study:
            conflicts.append("--run-c3d4-xgboost-study")
        if args.run_c3d4_limit_scan:
            conflicts.append("--run-c3d4-limit-scan")
        if conflicts:
            raise SystemExit(
                "--replot-c3d4-study-contours is mutually exclusive with "
                + " and ".join(conflicts)
            )
        from c3d4_xgboost_runner import replot_c3d4_study_contours

        cli_tokens = _sys.argv[1:]

        def option_was_given(name):
            return any(
                token == name or token.startswith(name + "=")
                for token in cli_tokens
            )

        c3_range_given = option_was_given("--c3d4-plot-c3-min") or option_was_given(
            "--c3d4-plot-c3-max"
        )
        d4_range_given = option_was_given("--c3d4-plot-d4-min") or option_was_given(
            "--c3d4-plot-d4-max"
        )
        try:
            summary = replot_c3d4_study_contours(
                args.study_outdir,
                luminosity=(
                    args.luminosity if option_was_given("--luminosity") else None
                ),
                contour_c3_range=(
                    (args.c3d4_plot_c3_min, args.c3d4_plot_c3_max)
                    if c3_range_given
                    else None
                ),
                contour_d4_range=(
                    (args.c3d4_plot_d4_min, args.c3d4_plot_d4_max)
                    if d4_range_given
                    else None
                ),
                contour_grid_bins=(
                    args.c3d4_plot_nbins
                    if option_was_given("--c3d4-plot-nbins")
                    else None
                ),
                contour_interpolation=(
                    args.c3d4_contour_interpolation
                    if option_was_given("--c3d4-contour-interpolation")
                    else None
                ),
                xsec_source_dir=(
                    args.c3d4_xsec_source_dir
                    if option_was_given("--c3d4-xsec-source-dir")
                    else None
                ),
                xsec_overlay=(False if args.no_c3d4_xsec_overlay else None),
            )
        except (ValueError, RuntimeError) as error:
            raise SystemExit(f"Cannot replot c3/d4 contours: {error}") from None
        print(
            f"Legacy-style v2 contour replot status: {summary['status']}; strategies:",
            ", ".join(summary["strategies"]),
        )
        print("Contour manifest:", args.study_outdir / "contour_replot_manifest.json")
        return 0

    if args.run_c3d4_xgboost_study:
        return _run_c3d4_xgboost_study_cli(args)

    if any(
        (
            args.sm_hh4b_signal_root,
            args.sm_hh4b_signal_dir,
            args.sm_hh4b_signal_xsec_fb,
            args.sm_hh4b_signal_generated_events,
        )
    ):
        raise SystemExit(
            "The --sm-hh4b-* options are supported only with "
            "--run-c3d4-xgboost-study. The legacy limit scan must not include "
            "this singleton before a c3 cross-section fit is available."
        )

    # Keep the legacy XGBoost stack out of the v2 startup path.  In
    # particular, this lets --shape-jobs configure BLAS/OpenMP before NumPy,
    # SciPy, pyhf or XGBoost is imported for the parallel study.
    from xgboost_root_varfiles_module import (
        combine_signal_component_rows,
        run_signal_background_analysis,
        score_background_files,
        score_signal_files,
        write_sample_report,
        write_c3d4_limit_scan,
    )

    if args.run_c3d4_limit_scan:
        (
            signal_files,
            background_files,
            signal_xsecs,
            background_xsecs,
            signal_generated,
            background_generated,
            signal_normalisation_weights,
            background_normalisation_weights,
            background_metadata,
        ) = _training_inputs_from_cli(args, ensure_analysis=True)

        signal_generation_factor = _signal_generation_rate_factor_for_cli(args)
        signal_tag_factor = _signal_tag_rate_factor_for_cli(args)
        signal_rate_factor = signal_generation_factor * signal_tag_factor
        background_generation_factor = _background_generation_rate_factors_for_cli(background_metadata, args)
        background_tag_factor = _background_tag_rate_factors_for_cli(background_metadata, args)
        background_rate_factor = _background_rate_factors_for_cli(background_metadata, args)
        signal_metadata = _signal_metadata_for_files(signal_files)
        _print_training_inputs(
            signal_files,
            background_files,
            signal_xsecs,
            background_xsecs,
            signal_generated,
            background_generated,
            signal_normalisation_weights,
            background_normalisation_weights,
            signal_rate_factors=[signal_rate_factor for _ in signal_files],
            background_rate_factors=(
                background_rate_factor
                if isinstance(background_rate_factor, list)
                else [background_rate_factor for _ in background_files]
            ),
            background_metadata=background_metadata,
        )
        rate_metadata = {
            "hbb_branching_ratio": args.hbb_branching_ratio,
            "btagging_rate": args.btagging_rate,
            "c_mistag_rate": args.c_mistag_rate,
            "light_mistag_rate": args.light_mistag_rate,
            "signal_hbb_power": args.signal_hbb_power,
            "signal_btag_power": args.signal_btag_power,
            "background_hbb_power": args.background_hbb_power,
            "background_btag_power": args.background_btag_power,
            "zbb_branching_ratio": args.zbb_branching_ratio,
            "signal_k_factor": float(args.signal_k_factor),
            "background_k_factor": float(args.background_k_factor),
            "signal_generation_rate_factor": signal_generation_factor,
            "signal_tag_rate_factor": signal_tag_factor,
            "signal_rate_factor": signal_rate_factor,
            "background_generation_rate_factor": background_generation_factor,
            "background_tag_rate_factor": background_tag_factor,
            "background_rate_factor": background_rate_factor,
        }
        print(f"Using luminosity {args.luminosity:g} fb^-1")
        print(f"K-factors: signal = {args.signal_k_factor:g}, background = {args.background_k_factor:g}")
        print(
            f"Signal generation factor = {signal_generation_factor:g} "
            f"(K_signal * BR_hbb^{args.signal_hbb_power}); "
            f"final tag factor = {signal_tag_factor:g} (btag^{args.signal_btag_power})"
        )
        if isinstance(background_rate_factor, list):
            print(
                "Background rate factors use K_background, optional BR_zbb for undecayed Z samples, "
                "and btag^b * c_mistag^c * light_mistag^j per CSV process"
            )
        else:
            print(
                f"Background rate factor = {background_rate_factor:g} "
                f"(K_background * BR_hbb^{args.background_hbb_power} * btag^{args.background_btag_power})"
            )

        sm_outdir = args.c3d4_sm_outdir or (args.c3d4_scan_outdir / "sm_optimization")
        print("Training SM-optimized XGBoost model in", sm_outdir)
        analysis = run_signal_background_analysis(
            signal_files=signal_files,
            background_files=background_files,
            output_dir=sm_outdir,
            signal_xsecs_fb=signal_xsecs,
            background_xsecs_fb=background_xsecs,
            signal_rate_factors=signal_rate_factor,
            background_rate_factors=background_rate_factor,
            signal_generated_events=signal_generated,
            background_generated_events=background_generated,
            signal_normalisation_weights=signal_normalisation_weights,
            background_normalisation_weights=background_normalisation_weights,
            signal_metadata=signal_metadata,
            background_metadata=background_metadata,
            luminosity=args.luminosity,
            test_size=args.test_size,
            seed=args.seed,
            systematics=args.systematics,
            min_background_mc_entries=args.min_xgb_background_mc,
            min_background_effective_entries=args.min_xgb_background_effective_mc,
            min_background_mc_entries_per_sample=args.min_xgb_background_mc_per_sample,
            max_events=args.max_events,
        )
        metrics = analysis["metrics"]
        best_threshold = metrics["best_threshold"]
        threshold = best_threshold["threshold"]
        model_file = _Path(metrics["outputs"]["model"])
        metrics_file = _Path(metrics["outputs"]["metrics"])

        background_scores = score_background_files(
            background_files=background_files,
            model_file=model_file,
            output_dir=args.c3d4_scan_outdir / "background_scores",
            threshold=threshold,
            background_xsecs_fb=background_xsecs,
            background_rate_factors=background_rate_factor,
            background_generated_events=background_generated,
            background_normalisation_weights=background_normalisation_weights,
            background_metadata=background_metadata,
            luminosity=args.luminosity,
            max_events=args.max_events,
        )
        sm_signal_scores = score_signal_files(
            signal_files=signal_files,
            model_file=model_file,
            output_dir=args.c3d4_scan_outdir / "sm_signal_scores",
            threshold=threshold,
            signal_xsecs_fb=signal_xsecs,
            signal_rate_factors=signal_rate_factor,
            signal_generated_events=signal_generated,
            signal_normalisation_weights=signal_normalisation_weights,
            signal_metadata=signal_metadata,
            luminosity=args.luminosity,
            max_events=args.max_events,
        )
        _print_xgboost_threshold_summary(
            threshold,
            sm_signal_scores,
            background_scores["backgrounds"],
            args.luminosity,
        )
        full_sample_background_events = background_scores["metadata"]["expected_selected_events_total"]
        background_events = float(best_threshold.get("background_events", 0.0))
        print(
            "Background yield used for the limit from held-out evaluation =",
            background_events,
            "(full training+test rescore diagnostic =",
            full_sample_background_events,
            ")",
        )
        if background_events <= 0.0:
            background_events = full_sample_background_events
            print("Warning: held-out background estimate is zero; using the full-sample diagnostic estimate.")

        scan_inputs = []
        if args.c3d4_signal_root:
            scan_inputs.extend(args.c3d4_signal_root)
        if args.c3d4_signal_dir:
            scan_inputs.extend(args.c3d4_signal_dir)
        if not scan_inputs:
            scan_inputs.append(_REPO_DIR / "HerwigSignalPoints" / "c3d4_10k" / "events")

        scan_files = _ensure_analysis_var_roots(
            scan_inputs,
            executable=args.analysis_exe,
            source_file=args.analysis_source,
            include_auxiliary=args.include_auxiliary_samples,
            jobs=args.analysis_jobs,
            max_events=args.analysis_max_events,
            force=args.force_analysis,
            run_missing=not args.no_run_missing_analysis,
        )
        if not scan_files:
            raise SystemExit(
                "No c3/d4 ROOT variable files found. Run the Herwig analysis step first, "
                "or pass --c3d4-signal-root/--c3d4-signal-dir with raw Herwig ROOT files."
            )

        scan_xsecs, scan_generated, scan_normalisation_weights = _infer_scored_signal_metadata(
            scan_files,
            args.c3d4_signal_xsec_fb,
            args.c3d4_signal_generated_events,
            args.c3d4_default_generated_events,
            "c3/d4 signal",
            "--c3d4-signal-xsec-fb",
        )
        print("c3/d4 signal files:")
        for path, xsec, generated, normalisation_weight in zip(
            scan_files,
            scan_xsecs,
            scan_generated,
            scan_normalisation_weights,
        ):
            print(
                f"  {path}  xsec={xsec:g} fb  generated={generated}  "
                f"normalisation_weight={_format_weight(normalisation_weight)}"
            )

        score_outdir = args.c3d4_scan_outdir / "signal_scores"
        scored_rows = score_signal_files(
            signal_files=scan_files,
            model_file=model_file,
            output_dir=score_outdir,
            threshold=threshold,
            signal_xsecs_fb=scan_xsecs,
            signal_rate_factors=signal_rate_factor,
            signal_generated_events=scan_generated,
            signal_normalisation_weights=scan_normalisation_weights,
            luminosity=args.luminosity,
            max_events=args.max_events,
            write_event_scores=args.write_event_scores,
        )
        rows_for_limit = scored_rows
        hhhbb_score_rows = []
        hhhbb_inputs = []
        if args.hhhbb_signal_root:
            hhhbb_inputs.extend(args.hhhbb_signal_root)
        if args.hhhbb_signal_dir:
            hhhbb_inputs.extend(args.hhhbb_signal_dir)
        if hhhbb_inputs:
            hhhbb_files = _ensure_analysis_var_roots(
                hhhbb_inputs,
                executable=args.analysis_exe,
                source_file=args.analysis_source,
                include_auxiliary=args.include_auxiliary_samples,
                jobs=args.analysis_jobs,
                max_events=args.analysis_max_events,
                force=args.force_analysis,
                run_missing=not args.no_run_missing_analysis,
            )
            if not hhhbb_files:
                raise SystemExit(
                    "No hhhbb ROOT variable files found. Pass --hhhbb-signal-root/--hhhbb-signal-dir "
                    "with campaign ROOT outputs or precomputed *_var.smear*.root files."
                )

            hhhbb_xsecs, hhhbb_generated, hhhbb_normalisation_weights = _infer_scored_signal_metadata(
                hhhbb_files,
                args.hhhbb_signal_xsec_fb,
                args.hhhbb_signal_generated_events,
                args.hhhbb_default_generated_events,
                "hhhbb signal",
                "--hhhbb-signal-xsec-fb",
            )
            hhhbb_rate_factor = _hhhbb_signal_rate_factor_for_cli(args)
            rate_metadata["hhhbb_signal_hbb_power"] = 3
            rate_metadata["hhhbb_signal_rate_factor"] = hhhbb_rate_factor
            print("hhhbb signal files:")
            for path, xsec, generated, normalisation_weight in zip(
                hhhbb_files,
                hhhbb_xsecs,
                hhhbb_generated,
                hhhbb_normalisation_weights,
            ):
                print(
                    f"  {path}  xsec={xsec:g} fb  generated={generated}  "
                    f"normalisation_weight={_format_weight(normalisation_weight)}  "
                    f"rate_factor={hhhbb_rate_factor:g}"
                )
            hhhbb_score_rows = score_signal_files(
                signal_files=hhhbb_files,
                model_file=model_file,
                output_dir=args.c3d4_scan_outdir / "hhhbb_signal_scores",
                threshold=threshold,
                signal_xsecs_fb=hhhbb_xsecs,
                signal_rate_factors=hhhbb_rate_factor,
                signal_generated_events=hhhbb_generated,
                signal_normalisation_weights=hhhbb_normalisation_weights,
                luminosity=args.luminosity,
                max_events=args.max_events,
                write_event_scores=args.write_event_scores,
            )
            _print_sm_hhhbb_summary(hhhbb_score_rows)
            rows_for_limit = combine_signal_component_rows(scored_rows, hhhbb_score_rows)
        hhbbbb_score_rows = []
        hhbbbb_inputs = []
        if args.hhbbbb_signal_root:
            hhbbbb_inputs.extend(args.hhbbbb_signal_root)
        if args.hhbbbb_signal_dir:
            hhbbbb_inputs.extend(args.hhbbbb_signal_dir)
        if hhbbbb_inputs:
            hhbbbb_files = _ensure_analysis_var_roots(
                hhbbbb_inputs,
                executable=args.analysis_exe,
                source_file=args.analysis_source,
                include_auxiliary=args.include_auxiliary_samples,
                jobs=args.analysis_jobs,
                max_events=args.analysis_max_events,
                force=args.force_analysis,
                run_missing=not args.no_run_missing_analysis,
            )
            if not hhbbbb_files:
                raise SystemExit(
                    "No hhbbbb ROOT variable files found. Pass --hhbbbb-signal-root/--hhbbbb-signal-dir "
                    "with campaign ROOT outputs or precomputed *_var.smear*.root files."
                )

            hhbbbb_xsecs, hhbbbb_generated, hhbbbb_normalisation_weights = _infer_scored_signal_metadata(
                hhbbbb_files,
                args.hhbbbb_signal_xsec_fb,
                args.hhbbbb_signal_generated_events,
                args.hhbbbb_default_generated_events,
                "hhbbbb signal",
                "--hhbbbb-signal-xsec-fb",
            )
            hhbbbb_rate_factor = _hhbbbb_signal_rate_factor_for_cli(args)
            rate_metadata["hhbbbb_signal_hbb_power"] = 2
            rate_metadata["hhbbbb_signal_rate_factor"] = hhbbbb_rate_factor
            print("hhbbbb signal files:")
            for path, xsec, generated, normalisation_weight in zip(
                hhbbbb_files,
                hhbbbb_xsecs,
                hhbbbb_generated,
                hhbbbb_normalisation_weights,
            ):
                print(
                    f"  {path}  xsec={xsec:g} fb  generated={generated}  "
                    f"normalisation_weight={_format_weight(normalisation_weight)}  "
                    f"rate_factor={hhbbbb_rate_factor:g}"
                )
            hhbbbb_score_rows = score_signal_files(
                signal_files=hhbbbb_files,
                model_file=model_file,
                output_dir=args.c3d4_scan_outdir / "hhbbbb_signal_scores",
                threshold=threshold,
                signal_xsecs_fb=hhbbbb_xsecs,
                signal_rate_factors=hhbbbb_rate_factor,
                signal_generated_events=hhbbbb_generated,
                signal_normalisation_weights=hhbbbb_normalisation_weights,
                luminosity=args.luminosity,
                max_events=args.max_events,
                write_event_scores=args.write_event_scores,
            )
            _print_sm_hhbbbb_summary(hhbbbb_score_rows)
            rows_for_limit = combine_signal_component_rows(scored_rows, hhhbb_score_rows, hhbbbb_score_rows)
        write_c3d4_limit_scan(
            rows_for_limit,
            output_dir=args.c3d4_scan_outdir,
            background_events=background_events,
            threshold=threshold,
            luminosity=args.luminosity,
            cl_target=args.c3d4_cl_target,
            poisson_confidence_level=args.poisson_cl,
            poisson_method=args.poisson_limit_method,
            poisson_observed_events=args.poisson_observed_events,
            background_variation_band=not args.no_background_variation_band,
            background_variation_factor=args.background_variation_factor,
            systematics=args.systematics,
            model_file=model_file,
            metrics_file=metrics_file,
            fit_signal=not args.no_c3d4_chebyshev_fit,
            fit_k3_range=(args.c3d4_fit_k3_min, args.c3d4_fit_k3_max),
            fit_k4_range=(args.c3d4_fit_k4_min, args.c3d4_fit_k4_max),
            plot_c3_range=(args.c3d4_plot_c3_min, args.c3d4_plot_c3_max),
            plot_d4_range=(args.c3d4_plot_d4_min, args.c3d4_plot_d4_max),
            plot_n_c3=args.c3d4_plot_nbins,
            plot_n_d4=args.c3d4_plot_nbins,
            xsec_overlay=not args.no_c3d4_xsec_overlay,
            xsec_source_dir=args.c3d4_xsec_source_dir,
            hhh_xsec_source_dir=args.hhh_xsec_source_dir,
            rate_metadata=rate_metadata,
        )
        _print_sm_background_mc_counts(metrics)
        if not args.no_sample_report:
            write_sample_report(
                signal_files=signal_files,
                background_files=background_files,
                output_dir=_sample_report_dir(args, args.c3d4_scan_outdir),
                model_file=model_file,
                threshold=threshold,
                signal_xsecs_fb=signal_xsecs,
                background_xsecs_fb=background_xsecs,
                signal_generation_rate_factors=signal_generation_factor,
                background_generation_rate_factors=background_generation_factor,
                signal_tag_rate_factors=signal_tag_factor,
                background_tag_rate_factors=background_tag_factor,
                signal_generated_events=signal_generated,
                background_generated_events=background_generated,
                signal_normalisation_weights=signal_normalisation_weights,
                background_normalisation_weights=background_normalisation_weights,
                signal_metadata=signal_metadata,
                background_metadata=background_metadata,
                luminosity=args.luminosity,
                max_events=args.max_events,
            )
        return 0

    score_inputs = []
    if args.score_signal_root:
        score_inputs.extend(args.score_signal_root)
    if args.score_signal_dir:
        score_inputs.extend(args.score_signal_dir)
    if score_inputs:
        score_files = _discover_score_roots(score_inputs)
        if not score_files:
            raise SystemExit("No signal-point ROOT variable files found for scoring.")

        threshold = args.threshold
        if threshold is None:
            with open(args.metrics_file) as handle:
                threshold = json.load(handle)["best_threshold"]["threshold"]

        signal_xsecs, signal_generated, signal_normalisation_weights = _infer_scored_signal_metadata(
            score_files,
            args.score_signal_xsec_fb,
            args.score_signal_generated_events,
            args.score_default_generated_events,
            "scored signal",
            "--score-signal-xsec-fb",
        )

        signal_rate_factor = _signal_final_rate_factor_for_cli(args)
        print(
            f"Signal rate factor = {signal_rate_factor:g} "
            f"(K_signal * BR_hbb^{args.signal_hbb_power} * btag^{args.signal_btag_power})"
        )
        score_signal_files(
            signal_files=score_files,
            model_file=args.model_file,
            output_dir=args.score_outdir,
            threshold=threshold,
            signal_xsecs_fb=signal_xsecs,
            signal_rate_factors=signal_rate_factor,
            signal_generated_events=signal_generated,
            signal_normalisation_weights=signal_normalisation_weights,
            luminosity=args.luminosity,
            max_events=args.max_events,
            write_event_scores=args.write_event_scores,
        )
        return 0

    (
        signal_files,
        background_files,
        signal_xsecs,
        background_xsecs,
        signal_generated,
        background_generated,
        signal_normalisation_weights,
        background_normalisation_weights,
        background_metadata,
    ) = _training_inputs_from_cli(args)

    signal_generation_factor = _signal_generation_rate_factor_for_cli(args)
    signal_tag_factor = _signal_tag_rate_factor_for_cli(args)
    signal_rate_factor = signal_generation_factor * signal_tag_factor
    background_generation_factor = _background_generation_rate_factors_for_cli(background_metadata, args)
    background_tag_factor = _background_tag_rate_factors_for_cli(background_metadata, args)
    background_rate_factor = _background_rate_factors_for_cli(background_metadata, args)
    signal_metadata = _signal_metadata_for_files(signal_files)
    _print_training_inputs(
        signal_files,
        background_files,
        signal_xsecs,
        background_xsecs,
        signal_generated,
        background_generated,
        signal_normalisation_weights,
        background_normalisation_weights,
        signal_rate_factors=[signal_rate_factor for _ in signal_files],
        background_rate_factors=(
            background_rate_factor
            if isinstance(background_rate_factor, list)
            else [background_rate_factor for _ in background_files]
        ),
        background_metadata=background_metadata,
    )
    print(f"K-factors: signal = {args.signal_k_factor:g}, background = {args.background_k_factor:g}")
    print(
        f"Signal generation factor = {signal_generation_factor:g} "
        f"(K_signal * BR_hbb^{args.signal_hbb_power}); "
        f"final tag factor = {signal_tag_factor:g} (btag^{args.signal_btag_power})"
    )

    analysis = run_signal_background_analysis(
        signal_files=signal_files,
        background_files=background_files,
        output_dir=args.outdir,
        signal_xsecs_fb=signal_xsecs,
        background_xsecs_fb=background_xsecs,
        signal_rate_factors=signal_rate_factor,
        background_rate_factors=background_rate_factor,
        signal_generated_events=signal_generated,
        background_generated_events=background_generated,
        signal_normalisation_weights=signal_normalisation_weights,
        background_normalisation_weights=background_normalisation_weights,
        signal_metadata=signal_metadata,
        background_metadata=background_metadata,
        luminosity=args.luminosity,
        test_size=args.test_size,
        seed=args.seed,
        systematics=args.systematics,
        min_background_mc_entries=args.min_xgb_background_mc,
        min_background_effective_entries=args.min_xgb_background_effective_mc,
        min_background_mc_entries_per_sample=args.min_xgb_background_mc_per_sample,
        max_events=args.max_events,
    )
    metrics = analysis["metrics"]
    threshold = metrics["best_threshold"]["threshold"]
    model_file = _Path(metrics["outputs"]["model"])
    background_scores = score_background_files(
        background_files=background_files,
        model_file=model_file,
        output_dir=args.outdir / "background_scores",
        threshold=threshold,
        background_xsecs_fb=background_xsecs,
        background_rate_factors=background_rate_factor,
        background_generated_events=background_generated,
        background_normalisation_weights=background_normalisation_weights,
        background_metadata=background_metadata,
        luminosity=args.luminosity,
        max_events=args.max_events,
    )
    sm_signal_scores = score_signal_files(
        signal_files=signal_files,
        model_file=model_file,
        output_dir=args.outdir / "sm_signal_scores",
        threshold=threshold,
        signal_xsecs_fb=signal_xsecs,
        signal_rate_factors=signal_rate_factor,
        signal_generated_events=signal_generated,
        signal_normalisation_weights=signal_normalisation_weights,
        signal_metadata=signal_metadata,
        luminosity=args.luminosity,
        max_events=args.max_events,
    )
    _print_xgboost_threshold_summary(
        threshold,
        sm_signal_scores,
        background_scores["backgrounds"],
        args.luminosity,
    )
    if not args.no_sample_report:
        write_sample_report(
            signal_files=signal_files,
            background_files=background_files,
            output_dir=_sample_report_dir(args, args.outdir),
            model_file=model_file,
            threshold=threshold,
            signal_xsecs_fb=signal_xsecs,
            background_xsecs_fb=background_xsecs,
            signal_generation_rate_factors=signal_generation_factor,
            background_generation_rate_factors=background_generation_factor,
            signal_tag_rate_factors=signal_tag_factor,
            background_tag_rate_factors=background_tag_factor,
            signal_generated_events=signal_generated,
            background_generated_events=background_generated,
            signal_normalisation_weights=signal_normalisation_weights,
            background_normalisation_weights=background_normalisation_weights,
            signal_metadata=signal_metadata,
            background_metadata=background_metadata,
            luminosity=args.luminosity,
            max_events=args.max_events,
        )
    return 0


if __name__ == "__main__" and "--legacy" not in _sys.argv:
    raise SystemExit(_run_local_xgboost_cli())

import numpy as np
import math
import random
from math import log10, floor
import os
import string
import subprocess
from scipy.optimize import curve_fit
from functools import partial
from lheinfo import get_xsec_witherror
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, brentq
import matplotlib.lines as mlines
import threading
from threading import Thread
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from joblib import Parallel, delayed

# xgboost stuff
from xgboost_root_varfiles_module import *


############################
# LOCATIONS AND PARAMETERS #
############################
print('HEFT Higgs XSEC Fitting and Analysis -- Global Version')

#################
# RUN FLAGS:
#################

# read the fit or perform it and write it?
DoFit = True

# Rerun?
ReRunHerwig = False

# Run herwig/analysis in the first place?
RunHerwig = False

# Rerun Analysis?
ReRunAnalysis = False

# Rerun XGBOOST Analysis?
ReRunAnalysisXGBOOST = False

# do the training and write it or not?
DoTraining = False

###############
# PARAMS
###############

# choose the model - HEFT2/HEFT3 or C3-D4 model only 
#MODEL = 'HEFT2' # HEFT2 or C3D4ONLY or HEFT3
#MODEL = 'HEFT3'
#MODEL = 'C3D4ONLY'
#MODEL = 'HEFT4'
MODEL = 'HEFT6'
#MODEL = 'HEFT4C3D4'
print('MODEL=', MODEL)

# choose the type of smearing: NONE, ATLAS, CMS
#SMEARING = 'NONE'
#SMEARING = 'ATLAS'
SMEARING = 'CMS'

# Systematics
Systematics = 0.0 # the alpha value for the systematics
# b-tagging rate
btagging = 0.85

# energy and luminosity
Energy = 13.6 # energy
Luminosity = 3000 # integrated luminosity in /fb to calculate signif
#Energy = 100
#Luminosity = 20000

# ENERGY RESCALING HERE:
DoRescaling = False
EnergyToRescale = 10000
ERESCALE = 1 # not a switch
RESCALETAG = ''
if DoRescaling is True:
    ERESCALE = EnergyToRescale**2/100**2
    RESCALETAG = '_RescaleE' + str(EnergyToRescale)


# K-factors for signal and backgrounds
KFAC_SIGNAL = 2.0
KFAC_BACKGROUNDS = 2.0

# change the KFACTOR ON THE BACKGROUND
CHANGE_KFAC = False
KFACTAG = ''
KFAC_BACKGROUNDS_NEW = 3.0
if CHANGE_KFAC is True:
    KFAC_BACKGROUNDS = KFAC_BACKGROUNDS_NEW
    KFACTAG = '_KFACBKG' + str(KFAC_BACKGROUNDS)

# change the KFACTOR ON THE SIGNL
CHANGE_KFAC_SIGNAL = True
KFACTAG = ''
KFAC_SIGNAL_NEW = 2.5
if CHANGE_KFAC_SIGNAL is True:
    KFAC_SIGNAL = KFAC_SIGNAL_NEW
    KFACTAG = '_KFACSIG' + str(KFAC_SIGNAL)


###############
# END OF PARAMS
###############

    
# array of variables:
variables = {}
variables[0] = 'c3'
variables[1] = 'ct2'
variables[2] = 'ct3'
variables[3] = 'd4'
variables[4] = 'ct1'
variables_latex = {}
variables_latex[0] = 'c_3'
variables_latex[1] = 'c_{t2}'
variables_latex[2] = 'c_{t3}'
variables_latex[3] = 'd_4'
variables_latex[4] = 'c_{t1}'

# constraints on these (fractional):
constraints = {}
constraints[100] = {}
constraints[100][0] = 5/100
constraints[100][1] = 0.1
constraints[100][2] = -1
constraints[100][3] = -1


# Input file templates for LO, MC@NLO and FxFx:
# the real files have a .in.template extension
HW_template = ['','', '']
HW_template[0] = 'Templates/HW-LO.in' # 0th element is LO

# The reduction factor of the number of events between the LHE file and the actual HW run for each process:
Reduction_Fac = [ '', '', '' ]
Reduction_Fac[0] = 0.999

# Branching ratios:
BR_z_ellell = 3.3632E-2 #  Z -> lepton lepton (one flavour)
BR_w_ellnu = 10.86E-2 # W -> lepton+neutrino (one flavour)
BR_z_vv = 0.2 # Z -> neutrino neutrino (all flavours)
BR_z_qq = 0.116 + 0.156 + 0.1203 + 0.1512 # Z -> qq
BR_z_bb = 0.150998
BR_h_bb = 0.5824
BR_h_gamgam = 0.00229

# chi-sq values in 2D for one and two sigma:
onesigma = 2.278868566376729
twosigma = 5.99

# debug flag
debug = True

# define the process under investigation:
Process = 'gg_hhh'

# the number of runs and tests for fitting
Nruns = 205

# The number of free coefficients to fit in the ME for each process
NCoeffs = {}
if MODEL == 'HEFT2':
    NCoeffs['gg_hhh'] = 18
elif MODEL == 'HEFT3':
    NCoeffs['gg_hhh'] = 8
elif MODEL == 'C3D4ONLY' or MODEL == 'HEFT4C3D4':
    NCoeffs['gg_hhh'] = 9
elif MODEL == 'HEFT4':
    NCoeffs['gg_hhh'] = 25
elif MODEL == 'HEFT6':
    NCoeffs['gg_hhh'] = 80
    
# directory for plots:
plot_dir = 'plots/'

# directory for fits:
fit_dir = 'fits/'

# Dictionaries to hold the fit coefficients and their covariance:
popt = {}
pcov = {}

# Directory for the pickle results
ResultsDir = '/mnt/hdd/Projects/GlobalHHH100/PickleResults/'

# Constraints directory
ConstraintsDir = 'Constraints/'

# MG5_aMC sub-dir:
if MODEL == 'HEFT2':
    MGLocation = '/home/apapaefs/Projects/GlobalHHH100/MG5_aMC_v2_9_22/' # hhh with 2 insertions in the HEFT
elif MODEL == 'C3D4ONLY' or MODEL == 'HEFT3' or MODEL == "HEFT4":
    MGLocation = '/home/apapaefs/Projects/GlobalHHH100/MG5_aMC_v2_9_24/' # hhh with 2 insertions in the HEFT
elif MODEL == 'HEFT4C3D4' or MODEL == "HEFT6":
    MGLocation = '/home/apapaefs/Projects/GlobalHHH100/MG5_aMC_v2_9_26/' # hhh with 2 insertions in the HEFT

# Analysis executable:
ExecutableSmear = {}
#ExecutableSmear[100] = 'Code/HwSimPostAnalysis_smear_100_example' # to be replaced with the full analysis including smearing
smearing_tag = ''
if SMEARING == 'NONE':
    ExecutableSmear[100] = 'Code/HwSimPostAnalysis_smear_100_variables'
    smearing_tag = ''
elif SMEARING == 'ATLAS':
    ExecutableSmear[100] = 'Code/HwSimPostAnalysis_smear_100_variables_ATLAS'
    smearing_tag = 'ATLAS'
elif SMEARING == 'CMS':
    ExecutableSmear[100] = 'Code/HwSimPostAnalysis_smear_100_variables_CMS'
    ExecutableSmear[13.6] = 'Code/HwSimPostAnalysis_smear_100_variables_CMS'
    smearing_tag = 'CMS' 



# the MG5 subdirectory for each process
ProcLocations = {}
if MODEL == 'HEFT2':
    ProcLocations['gg_hhh'] = 'gg_hhh_mheft2l2_restricted/' # hhh with squared truncation
elif MODEL == 'HEFT3':
    ProcLocations['gg_hhh'] = 'gg_hhh_mheft2l3_morerestricted/' # hhh with cubic truncation
elif MODEL == 'C3D4ONLY': 
    ProcLocations['gg_hhh'] = 'gg_hhh_c3d4/' # hhh with 2 insertions in the HEFT (no truncation)
elif MODEL == 'HEFT4C3D4':
    ProcLocations['gg_hhh'] = 'gg_hhh_restricted5new_heft4/' # hhh with 2 insertions in the HEFT (no truncation)
    #ProcLocations['gg_hhh'] = 'gg_hhh_full_mheft4/'
elif MODEL == 'HEFT4': 
    ProcLocations['gg_hhh'] = 'gg_hhh_restricted_mheft4/' # hhh with 2 insertions in the HEFT (no truncation)
elif MODEL == 'HEFT6': 
    ProcLocations['gg_hhh'] = 'gg_hhh_restricted5_mheft6_new/' # hhh with 2 insertions in the HEFT (no truncation)



# The numbering tag for the run:
if MODEL == 'HEFT2':
    RunNum = '11' # 100 TeV event generation # NEW FOR GLOBAL HHH - HEFT
elif (MODEL == 'C3D4ONLY' and Energy==100) or (MODEL == 'HEFT4C3D4' and Energy==13.6): # C3D4ONLY was 100 TeV, HEFT4C3D4 is 13.6 TeV
    RunNum = '10' # 100 TeV event generation # NEW FOR GLOBAL HHH - C3-D4 MODEL ONLY
elif MODEL == 'HEFT3':
    RunNum = '12'
elif MODEL == 'HEFT4': # 13.6 event generation - HEFT4 restricted (c3,d4,ct2,ct3)
    RunNum = '13'
elif MODEL == 'HEFT6': # 13.6 event generation - HEFT6 restricted (c3,d4,ct2,ct3,ct1)
    RunNum = '14'
elif MODEL == 'C3D4ONLY' and Energy==13.6:
    RunNum = '15' # 13.6 TeV event generation # NEW FOR GLOBAL HHH - C3-D4 MODEL ONLY


# SELECT FINAL STATE HERE:
FinalState = '6b'
if FinalState == '6b':
    FinalState6b = ''
    FinalStatebtau = '#'
    FinalStatebgamma = '#'

# Background Location:
BackgroundLocation = 'Backgrounds/events/'
Backgrounds = []
Backgrounds.append('all_events_6b')
Backgrounds.append('pp_zbbbb')
Backgrounds.append('pp_zzbb')
Backgrounds.append('pp_hzbb')
Backgrounds.append('pp_hhbb')
Backgrounds.append('pp_hbbbb')
Backgrounds.append('pp_hzz')
Backgrounds.append('pp_hhz')
Backgrounds.append('pp_zzz')
Backgrounds.append('gg_hzz')
Backgrounds.append('gg_zzz')
Backgrounds.append('gg_hhz')
Backgrounds_xsec = {}

Backgrounds_xsec[(100, 'all_events_6b')] = 28.328254252903694E3 # cross section for 6b background in fb (100 TeV)
Backgrounds_xsec[(100, 'pp_zbbbb')] = 958.3291282 # cross section for zbbbb background in fb (100 TeV) # 
Backgrounds_xsec[(100, 'pp_zzbb')] = 30.18859257 # cross section for pp_zzbb background in fb (100 TeV) # 
Backgrounds_xsec[(100, 'pp_hzbb')] = 5.417507336 # cross section for pp_hzbb background in fb (100 TeV) #
Backgrounds_xsec[(100, 'pp_zzz')] = 0.4773830264  # cross section for gg_zzz background in fb (100 TeV) # 
Backgrounds_xsec[(100, 'pp_hzz')] = 0.392990544 # cross section for pp_hzz background in fb (100 TeV)
Backgrounds_xsec[(100, 'pp_hhz')] = 0.2149781325 # cross section for pp_hhbb background in fb (100 TeV) # 
Backgrounds_xsec[(100, 'pp_hhbb')] = 0.04761220149 # cross section for pp_hhbb background in fb (100 TeV) #
Backgrounds_xsec[(100, 'pp_hbbbb')] = 1.92239859 # cross section for pp_hbbbb background in fb (100 TeV) # 
Backgrounds_xsec[(100, 'gg_hzz')] = 0.09506002389 # cross section for gg_hzz background in fb (100 TeV) #
Backgrounds_xsec[(100, 'gg_zzz')] = 0.01372856589  # cross section for gg_zzz background in fb (100 TeV) # 
Backgrounds_xsec[(100, 'gg_hhz')] = 0.1700475286  # cross section for gg_hhz background in fb (100 TeV) #
# initial total weight of events (before the analysis that created the _var.root files):
initial_S_SM = 100000
initial_S = 9990

if Energy == 100:
    #xsS=0.0028783E3 # signal cross section at 100 TeV in fb (SM)
    xsS=0.0028783
elif Energy == 13.6:
    xsS = 5.7563e-05 # signal cross section at 13.6 TeV in PB (SM)

signal_SM_file = './Herwig/events/HW-8_SM_6b_var.smear' + smearing_tag + '.root'

# location of the _var root files for the backgrounds:
Background_files = {}
Background_files[(100, 'all_events_6b')] = './Backgrounds/events/HW-all_events_6b_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'pp_zbbbb')] = './Backgrounds/events/HW-pp_zbbbb_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'pp_zzbb')] = './Backgrounds/events/HW-pp_zzbb_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'pp_hzbb')] = './Backgrounds/events/HW-pp_hzbb_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'pp_hhbb')] = './Backgrounds/events/HW-pp_hhbb_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'pp_hbbbb')] = './Backgrounds/events/HW-pp_hbbbb_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'pp_hzz')] = './Backgrounds/events/HW-pp_hzz_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'pp_zzz')] = './Backgrounds/events/HW-pp_zzz_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'pp_hhz')] = './Backgrounds/events/HW-pp_hhz_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'gg_hzz')] = './Backgrounds/events/HW-gg_hzz_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'gg_zzz')] = './Backgrounds/events/HW-gg_zzz_100_var.smear' + smearing_tag + '.root'
Background_files[(100, 'gg_hhz')] = './Backgrounds/events/HW-gg_hhz_100_var.smear' + smearing_tag + '.root'

# initial weight of Monte Carlo events (at the start of the analysis that generated the var root files):
initial_B = {}
initial_B['all_events_6b'] = 864960
initial_B['pp_zbbbb'] = 200000
initial_B['pp_zzbb'] = 200000
initial_B['pp_hzbb'] = 200000
initial_B['pp_hzz'] = 200000
initial_B['pp_zzz'] = 200000
initial_B['pp_hhz'] = 200000
initial_B['pp_hhbb'] = 200000
initial_B['pp_hbbbb'] = 200000
initial_B['gg_zzz'] = 100000
initial_B['gg_hzz'] = 200000
initial_B['gg_hhz'] = 200000

# initial actual (i.e. at luminosity) number of events for backgrounds
initial_NB = {}

# background ids:
idB = {}
idB['all_events_6b'] = 1
idB['pp_zbbbb'] = 2
idB['pp_zzbb'] = 3
idB['pp_hzbb'] = 4
idB['pp_hhz'] = 5
idB['pp_hzz'] = 6
idB['pp_zzz'] = 7
idB['pp_hhbb'] = 8
idB['pp_hbbbb'] = 9
idB['gg_hzz'] = 10
idB['gg_zzz'] = 11
idB['gg_hhz'] = 12
    


# factors to apply to signal and background (K-factors and BRs)
sig_factors = KFAC_SIGNAL * BR_h_bb**3 * btagging**6 * ERESCALE
bkg_factors = KFAC_BACKGROUNDS * btagging**6 * ERESCALE # BRs already applied. The k-factor is uniform



# Herwig input file sub-dir and output for the events
HerwigLocation = 'Herwig/'
HerwigOutputLocation = HerwigLocation + 'events/'
HerwigOutputDirectory = HerwigOutputLocation



#########################################################
# FUNCTIONS                                             # 
#########################################################

# function to get template
def getTemplate(basename):
    with open('%s.template' % basename, 'r') as f:
        templateText = f.read()
    return string.Template( templateText )

# write a filename
def writeFile(filename, text):
    with open(filename,'w') as f:
        f.write(text)

# round to a certain number of significant figures
def round_sig(x, sig=4):
    if x == 0.:
        return 0.
    if math.isnan(x) is True:
        print('Warning, NaN!')
        return 0.
    return round(x, sig-int(floor(log10(abs(x))))-1)

# gaussian function
def gaussian(x, mu, delta):
    return 1./(np.sqrt(2.*np.pi)*delta)*np.exp(-np.power((x - mu)/delta, 2.)/2)

# function for Higgs boson triple production in the HEFT:
# only c3, d4, ct2, ct3 are assumed to be relevant
def func_CX(couplings=[], *coeffs, procname):
    #print('couplings=', couplings)
    Msq = 0
    if procname == 'gg_hhh':
        if MODEL == 'HEFT2':
            S1, S2, A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, L1, L2, N1, N2 = [float(coef) for coef in coeffs]
            c3, d4, cg1, cg2, ct1, cb1, ct2, cb2, ct3, cb3 = couplings
            Msq = A1**2*c3**2 + 2*A1*B1*c3*d4 + 2*A1*L1*ct2*c3 + 2*A1*N1*ct3*c3 + 2*A1*S1*c3 + A2**2*c3**2 + 2*A2*B2*c3*d4 + 2*A2*L2*ct2*c3 + 2*A2*N2*ct3*c3 + 2*A2*S2*c3 + B1**2*d4**2 + 2*B1*L1*ct2*d4 + 2*B1*N1*ct3*d4 + 2*B1*S1*d4 + B2**2*d4**2 + 2*B2*L2*ct2*d4 + 2*B2*N2*ct3*d4 + 2*B2*S2*d4 + 2*C1*S1*c3**2 + 2*C2*S2*c3**2 + 2*D1*S1*d4**2 + 2*D2*S2*d4**2 + 2*E1*S1*ct2**2 + 2*E2*S2*ct2**2 + 2*F1*S1*ct3**2 + 2*F2*S2*ct3**2 + L1**2*ct2**2 + 2*L1*N1*ct2*ct3 + 2*L1*S1*ct2 + L2**2*ct2**2 + 2*L2*N2*ct2*ct3 + 2*L2*S2*ct2 + N1**2*ct3**2 + 2*N1*S1*ct3 + N2**2*ct3**2 + 2*N2*S2*ct3 + S1**2 + S2**2
        elif MODEL == 'HEFT3':
            S1, B1, C1, D1, E1, F1, L1, N1 = [float(coef) for coef in coeffs]
            c3, d4, cg1, cg2, ct1, cb1, ct2, cb2, ct3, cb3 = couplings
            Msq = S1 + B1 * c3**3 + C1 * c3**2 * d4 + D1 * c3**2 + E1 * d4**2 + F1 * c3 * d4 + L1 * d4 + N1 * c3
        elif MODEL == 'C3D4ONLY' or MODEL == 'HEFT4C3D4': 
            S1, A1, B1, C1, D1, E1, F1, L1, N1 = [float(coef) for coef in coeffs]
            c3, d4, cg1, cg2, ct1, cb1, ct2, cb2, ct3, cb3 = couplings
            Msq = S1 + A1 * c3**4 + B1 * c3**3 + C1 * c3**2 * d4 + D1 * c3**2 + E1 * d4**2 + F1 * c3 * d4 + L1 * d4 + N1 * c3
        elif MODEL == 'HEFT4':
            A, B, C, D, E, F, G, H, J, K, L, M, N, O, P, Q, R, S, T, W, X, Y, Z, ZZ, WW= [float(coef) for coef in coeffs]
            c3, d4, cg1, cg2, ct1, cb1, ct2, cb2, ct3, cb3 = couplings
            Msq = A*c3**2*d4 + B*c3**2*ct2**2 + C*c3**2*ct2 + D * c3**2*ct3 + E*c3**2 + F*c3**4 + G*c3**3*ct2 + H*c3**3 + J*c3*ct2*d4 + K*c3*d4 + L*c3*ct2*ct3 + M*c3*ct2 + N*c3*ct2**2 + O*c3*ct3 + P*c3 + Q*d4**2 + R*ct2*d4 + S*ct3*d4 + T*d4 + W*ct2**2 + X*ct2*ct3 + Y*ct2 + Z*ct3**2 + ZZ*ct3 + WW
        elif MODEL == 'HEFT6':
            A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, AA, AB, AC, AD, AE, AF, AG, AH, AI, AJ, AK, AL, AM, AN, AO, AP, AQ, AR, AS, AT, AU, AV, AW, AX, AY, AZ, BA, BB, BC, BD, BE, BF, BG, BH, BI, BJ, BK, BL, BM, BN, BO, BP, BQ, BR, BS, BT, BU, BV, BW, BX, BY, BZ, CA, CB = [float(coef) for coef in coeffs]
            c3, d4, cg1, cg2, cb3, cb1, ct2, cb2, ct3, ct1 = couplings # notice change of order here
            Msq = A*c3**2*ct1*d4 + B*c3**2*ct1**2*d4 + C*c3**2*d4 + D*c3**2*ct1**4 + E*c3**2*ct1**2*ct2 + F*c3**2*ct1**2 + G*c3**2*ct1*ct2 + H*c3**2*ct1*ct3 + I*c3**2*ct1 + J*c3**2*ct1**3 + K*c3**2*ct2**2 + L*c3**2*ct2 + M*c3**2*ct3 + N*c3**2 + O*c3**4*ct1**2 + P*c3**4*ct1 + Q*c3**4 + R*c3**3*ct1*ct2 + S*c3**3*ct1 + T*c3**3*ct1**2 + U*c3**3*ct1**3 + V*c3**3*ct2 + W*c3**3 + X*c3*ct1*ct2*d4 + Y*c3*ct1*d4 + Z*c3*ct1**2*d4 + AA*c3*ct1**3*d4 + AB*c3*ct2*d4 + AC*c3*d4 + AD*c3*ct1*ct2 + AE*c3*ct1*ct2**2 + AF*c3*ct1*ct3 + AG*c3*ct1 + AH*c3*ct1**2*ct2 + AI*c3*ct1**2*ct3 + AJ*c3*ct1**2 + AK*c3*ct1**3*ct2 + AL*c3*ct1**3 + AM*c3*ct1**4 + AN*c3*ct1**5 + AO*c3*ct2*ct3 + AP*c3*ct2 + AQ*c3*ct2**2 + AR*c3*ct3 + AS*c3 + AT*ct1**2*d4**2 + AU*ct1*d4**2 + AV*d4**2 + AW*ct1*ct2*d4 + AX*ct1*ct3*d4 + AY*ct1*d4 + AZ*ct1**2*ct2*d4 + BA*ct1**2*d4 + BB*ct1**3*d4 + BC*ct1**4*d4 + BD*ct2*d4 + BE*ct3*d4 + BF*d4 + BG*ct1**2*ct2**2 + BH*ct1**2*ct2 + BI*ct1**2*ct3 + BJ*ct1**2 + BK*ct1**4*ct2 + BL*ct1**4 + BM*ct1**6 + BN*ct1**3*ct2 + BO*ct1**3*ct3 + BP*ct1**3 + BQ*ct1*ct2*ct3 + BR*ct1*ct2 + BS*ct1*ct2**2 + BT*ct1*ct3 + BU*ct1 + BV*ct1**5 + BW*ct2**2 + BX*ct2*ct3 + BY*ct2 + BZ*ct3**2 + CA*ct3 + CB
    return Msq


# function for Higgs boson triple production in the HEFT (PLOT VERSION)
def func_t_CX(c3, d4, ct2, ct3, coeffs, procname):
    if procname == 'gg_hhh':
        if MODEL == 'HEFT2':
            S1, S2, A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, L1, L2, N1, N2 = [float(coef) for coef in coeffs]
            Msq = A1**2*c3**2 + 2*A1*B1*c3*d4 + 2*A1*L1*ct2*c3 + 2*A1*N1*ct3*c3 + 2*A1*S1*c3 + A2**2*c3**2 + 2*A2*B2*c3*d4 + 2*A2*L2*ct2*c3 + 2*A2*N2*ct3*c3 + 2*A2*S2*c3 + B1**2*d4**2 + 2*B1*L1*ct2*d4 + 2*B1*N1*ct3*d4 + 2*B1*S1*d4 + B2**2*d4**2 + 2*B2*L2*ct2*d4 + 2*B2*N2*ct3*d4 + 2*B2*S2*d4 + 2*C1*S1*c3**2 + 2*C2*S2*c3**2 + 2*D1*S1*d4**2 + 2*D2*S2*d4**2 + 2*E1*S1*ct2**2 + 2*E2*S2*ct2**2 + 2*F1*S1*ct3**2 + 2*F2*S2*ct3**2 + L1**2*ct2**2 + 2*L1*N1*ct2*ct3 + 2*L1*S1*ct2 + L2**2*ct2**2 + 2*L2*N2*ct2*ct3 + 2*L2*S2*ct2 + N1**2*ct3**2 + 2*N1*S1*ct3 + N2**2*ct3**2 + 2*N2*S2*ct3 + S1**2 + S2**2
        elif  MODEL == 'HEFT3':
            S1, B1, C1, D1, E1, F1, L1, N1 = [float(coef) for coef in coeffs]
            Msq = S1 + B1 * c3**3 + C1 * c3**2 * d4 + D1 * c3**2 + E1 * d4**2 + F1 * c3 * d4 + L1 * d4 + N1 * c3
        elif MODEL == 'C3D4ONLY' or MODEL == 'HEFT4C3D4': 
            S1, A1, B1, C1, D1, E1, F1, L1, N1 = [float(coef) for coef in coeffs]
            Msq = S1 + A1 * c3**4 + B1 * c3**3 + C1 * c3**2 * d4 + D1 * c3**2 + E1 * d4**2 + F1 * c3 * d4 + L1 * d4 + N1 * c3
        elif MODEL == 'HEFT4':
            A, B, C, D, E, F, G, H, J, K, L, M, N, O, P, Q, R, S, T, W, X, Y, Z, ZZ, WW = [float(coef) for coef in coeffs]
            Msq = A*c3**2*d4 + B*c3**2*ct2**2 + C*c3**2*ct2 + D * c3**2*ct3 + E*c3**2 + F*c3**4 + G*c3**3*ct2 + H*c3**3 + J*c3*ct2*d4 + K*c3*d4 + L*c3*ct2*ct3 + M*c3*ct2 + N*c3*ct2**2 + O*c3*ct3 + P*c3 + Q*d4**2 + R*ct2*d4 + S*ct3*d4 + T*d4 + W*ct2**2 + X*ct2*ct3 + Y*ct2 + Z*ct3**2 + ZZ*ct3 + WW
        elif MODEL == 'HEFT6':
            A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, AA, AB, AC, AD, AE, AF, AG, AH, AI, AJ, AK, AL, AM, AN, AO, AP, AQ, AR, AS, AT, AU, AV, AW, AX, AY, AZ, BA, BB, BC, BD, BE, BF, BG, BH, BI, BJ, BK, BL, BM, BN, BO, BP, BQ, BR, BS, BT, BU, BV, BW, BX, BY, BZ, CA, CB = [float(coef) for coef in coeffs]
            Msq = A*c3**2*ct1*d4 + B*c3**2*ct1**2*d4 + C*c3**2*d4 + D*c3**2*ct1**4 + E*c3**2*ct1**2*ct2 + F*c3**2*ct1**2 + G*c3**2*ct1*ct2 + H*c3**2*ct1*ct3 + I*c3**2*ct1 + J*c3**2*ct1**3 + K*c3**2*ct2**2 + L*c3**2*ct2 + M*c3**2*ct3 + N*c3**2 + O*c3**4*ct1**2 + P*c3**4*ct1 + Q*c3**4 + R*c3**3*ct1*ct2 + S*c3**3*ct1 + T*c3**3*ct1**2 + U*c3**3*ct1**3 + V*c3**3*ct2 + W*c3**3 + X*c3*ct1*ct2*d4 + Y*c3*ct1*d4 + Z*c3*ct1**2*d4 + AA*c3*ct1**3*d4 + AB*c3*ct2*d4 + AC*c3*d4 + AD*c3*ct1*ct2 + AE*c3*ct1*ct2**2 + AF*c3*ct1*ct3 + AG*c3*ct1 + AH*c3*ct1**2*ct2 + AI*c3*ct1**2*ct3 + AJ*c3*ct1**2 + AK*c3*ct1**3*ct2 + AL*c3*ct1**3 + AM*c3*ct1**4 + AN*c3*ct1**5 + AO*c3*ct2*ct3 + AP*c3*ct2 + AQ*c3*ct2**2 + AR*c3*ct3 + AS*c3 + AT*ct1**2*d4**2 + AU*ct1*d4**2 + AV*d4**2 + AW*ct1*ct2*d4 + AX*ct1*ct3*d4 + AY*ct1*d4 + AZ*ct1**2*ct2*d4 + BA*ct1**2*d4 + BB*ct1**3*d4 + BC*ct1**4*d4 + BD*ct2*d4 + BE*ct3*d4 + BF*d4 + BG*ct1**2*ct2**2 + BH*ct1**2*ct2 + BI*ct1**2*ct3 + BJ*ct1**2 + BK*ct1**4*ct2 + BL*ct1**4 + BM*ct1**6 + BN*ct1**3*ct2 + BO*ct1**3*ct3 + BP*ct1**3 + BQ*ct1*ct2*ct3 + BR*ct1*ct2 + BS*ct1*ct2**2 + BT*ct1*ct3 + BU*ct1 + BV*ct1**5 + BW*ct2**2 + BX*ct2*ct3 + BY*ct2 + BZ*ct3**2 + CA*ct3 + CB
    return Msq


# function to read the mg5 event cross sections                
def read_files(runnum, mgloc, procloc, procname, CouplingsArray, nruns):
    X = []
    Z = []
    ZERR = []
    XSEC = {}
    for coups in CouplingsArray:
        #print(coups)
        lhe = 'run_' + procname + '_' + str(runnum) + '_' + '_'.join((coups)) + '/unweighted_events.lhe.gz'
        lhefile = mgloc + '/' + procloc + 'Events/' + lhe
        print('lhefile read=', lhefile)
        #TestBool = True
        #if TestBool is False:
        if os.path.exists(lhefile) is False:
            print('Error, lhe file or summary file:', lhefile, 'does not exist!')
            exit()
        else:
            #zgrepcommand = 'zgrep "Integrated weight" ' + lhefile
            #p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
            #for line in iter(p.stdout.readline, b''):
            #    xsec = float(line.split()[5])
            #print(coups, xsec)
            xsec, xsecerr = get_xsec_witherror(lhefile)
            print(coups, xsec)
            #xsec = 0
            coups_tuple = []
            for mm in range(len(coups)):
                coups_tuple.append(float(coups[mm]))
            X.append(tuple(coups_tuple))
            Z.append(float(xsec))
            ZERR.append(float(xsecerr))
            XSEC[tuple(coups_tuple)] = float(xsec)
            #print(X)
    return np.transpose(X), Z, ZERR, XSEC

def gen_coupbdasarray_dim_rand_range(coup_min, coup_max, nruns, randseed):
    random.seed(randseed)
    
    CouplingsArray_R = []
    CouplingsArrayF_R = []
    random_choice = 0
    # NOTE: legacy zeroes to comply with previous code! 
    while random_choice < nruns:
        coup1 = coup_min[0] + (coup_max[0] - coup_min[0]) * random.random()
        coup2 = coup_min[1] + (coup_max[1] - coup_min[1]) * random.random()
        coup3 = 0.0 * random.random()
        coup4 = 0.0 * random.random()
        coup5 = 0.0 * random.random()
        coup6 = 0.0 * random.random()
        coup7 = coup_min[2] + (coup_max[2] - coup_min[2]) * random.random()
        coup8 = 0.0 * random.random()
        coup9 = coup_min[3] + (coup_max[3] - coup_min[3]) * random.random()
        if MODEL == 'HEFT6':
            coup10 = coup_min[4] + (coup_max[4] - coup_min[4]) * random.random()
        else:
            coup10 = 0.0 * random.random()
        CouplingsArray = [str(round_sig(coup1,4)), str(round_sig(coup2,4)), str(round_sig(coup3,4)), str(round_sig(coup4,4)), str(round_sig(coup5,4)), str(round_sig(coup6,4)), str(round_sig(coup7,4)), str(round_sig(coup8,4)), str(round_sig(coup9,4)), str(round_sig(coup10,4))]
        CouplingsArrayF = tuple([round_sig(coup1,4), round_sig(coup2,4), round_sig(coup3,4), round_sig(coup4,4), round_sig(coup5,4), round_sig(coup6,4), round_sig(coup7,4), round_sig(coup8,4), round_sig(coup9,4), round_sig(coup10,4)])
        #print('CouplingsArray RANDOM=', CouplingsArray)
        CouplingsArray_R.append(CouplingsArray)
        CouplingsArrayF_R.append(CouplingsArrayF)
        random_choice = random_choice + 1
    print('Generated random arrays for Nruns=', nruns)
    return CouplingsArray_R, CouplingsArrayF_R


# function to read the mg5 event cross sections and compare to the fit              
def test_fit(runnum, mgloc, procloc, procname, CouplingsArray, ntotal, popt):
    X = []
    Z = []
    XSEC = {}
    ZERR = []
    func_CX_proc = partial(func_CX, procname=Process)
    fracdiff_avg = 0.
    for coups in CouplingsArray:
        lhe = 'run_' + procname + '_' + str(runnum) + '_' + '_'.join((coups)) + '/unweighted_events.lhe.gz'
        lhefile = mgloc + '/' + procloc + 'Events/' + lhe
        #TestBool = True
        #if TestBool is False:
        if os.path.exists(lhefile) is False:
            print('Error, lhe file or summary file:', lhefile, 'does not exist!')
            exit()
        else:
            #zgrepcommand = 'zgrep "Integrated weight" ' + lhefile
            #p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
            #for line in iter(p.stdout.readline, b''):
            #    xsec = float(line.split()[5])
            xsec, xsecerr = get_xsec_witherror(lhefile)
            coups_tuple = []
            for mm in range(len(coups)):
                coups_tuple.append(float(coups[mm]))
            X.append(tuple(coups_tuple))
            Z.append(float(xsec))
            ZERR.append(float(xsecerr))
            # get the fitted XSEC
            xsec_fit = func_CX_proc(coups_tuple, *popt)
            fracdiff = abs(xsec-xsec_fit)/xsec
            if fracdiff > 0.2:
                print(coups, xsec)
                print('!!! lhefile=', lhefile)
                print('!!! xsec: real, fitted, frac diff =', xsec, xsec_fit, fracdiff)
            fracdiff_avg = fracdiff_avg + fracdiff
            XSEC[tuple(coups_tuple)] = float(xsec)
            #print(X)
    print('average fractional difference =', fracdiff_avg/ntotal)
    return np.transpose(X), Z, ZERR, XSEC




# 2D contour plot 
def contour_xsec(procname, plotname, plottitle, fit_coeffs, var1, var2, xlim, ylim, axext='', figext='', smtext=True, starsize=15, setxlabel=True, setylabel=True, nbins=100, savefig=True,variables=variables, variables_latex=variables_latex, labelsize=20, normalbar=True, contours=np.arange(0, 10, 0.5),norm_to_zeroth=True):
    output = procname + '_' + plotname + '_' + var1 + '_' + var2
    print('Plotting', output)
    nvar1 = [key for key, value in variables.items() if value == var1][0]
    nvar2 = [key for key, value in variables.items() if value == var2][0]
    #print(var1, var2)
    #print(nvar1, nvar2)
    # construct the axes for the plot
    # no need to modify this if you just need one plot
    gs = gridspec.GridSpec(4, 4)
    if figext == '':
        fig = plt.figure()
    else:
        fig = figext
    if axext == '':
        ax = fig.add_subplot(111)
    else:
        ax=axext
    ax.grid(False)
    ax.set_title(plottitle)
    # create legend and plot/font size
    #ax.legend()
    #ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8})
    # set the ticks, labels and limits etc.
    xlab = '$' + variables_latex[nvar1] + '$'
    ylab = '$' + variables_latex[nvar2] + '$'
    if setylabel == True:
        ax.set_ylabel(ylab, fontsize=labelsize)
    if setxlabel == True:
        ax.set_xlabel(xlab, fontsize=labelsize)
    
    # choose x and y log scales
    #if ylog:
    #    ax.set_yscale('log')
    #else:
    #    ax.set_yscale('linear')
    #if xlog:
    #    ax.set_xscale('log')
    #else:
    #    ax.set_xscale('linear')
    # set the limits on the x and y axes if required below:
    ymin = ylim[0]
    ymax = ylim[1]
    xmin = xlim[0]
    xmax = xlim[1]
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    ctexts = []
    cvartexts = []
    for i in range(0, len(variables.keys())):
        if i != nvar1 and i != nvar2:
            ctext = variables[i] + '=0'
            ctexts.append(ctext)
        else:
            cvartexts.append(variables[i])
    #print(ctexts)
    fstr = 'partial(func_t_CX, ' + ','.join([ct for ct in ctexts]) + ', procname=Process)'
    global func_CX_partial
    func_CX_partial = eval(fstr)
    #print(func_CX_partial)
    #print(fit_coeffs)
    global fit_coeffs_g
    fit_coeffs_g = fit_coeffs
    #print(cvartexts[0], cvartexts[1])
    if norm_to_zeroth is True:
        feval = 'func_CX_partial(' + cvartexts[0] +'=x1,' + cvartexts[1] + '=x2,coeffs=fit_coeffs_g)/func_CX_partial(' + cvartexts[0] +'=0,' + cvartexts[1] + '=0,coeffs=fit_coeffs_g)'
    else:
        feval = 'func_CX_partial(' + cvartexts[0] +'=x1,' + cvartexts[1] + '=x2,coeffs=fit_coeffs_g)'
    func_fin = lambda x1, x2: eval(feval)
    #print(func_fin(0.05, -0.05))
    x = np.linspace(xlim[0], xlim[1], nbins)
    y = np.linspace(ylim[0], ylim[1], nbins)
    X, Y = np.meshgrid(x,y)
    Z = func_fin(X,Y)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    cont = ax.contourf(X, Y, Z, contours, cmap='Spectral', extend='max')
    ax.plot(0,0,marker='*',ms=starsize, color='black')
    if smtext == True:
        ax.text(0.53, 0.53,"SM", transform=ax.transAxes)
    if normalbar == True:
        plt.colorbar(cont)
    if savefig == True:
        # save the figure
        print('saving the figure')
        # save the figure in PDF format
        infile = output + '.dat'
        print('---')
        print('output in', infile.replace('.dat','.pdf'))
        plt.savefig(infile.replace('.dat','.pdf'), bbox_inches='tight')
        plt.close(fig)
    return cont

# 1D plot of the xsec
def oned_xsec(procname, plotname, plottitle, fit_coeffs, var1, xlim, ylim, axext='', figext='', smtext=True, starsize=15, setxlabel=True, setylabel=True, nbins=100, savefig=True,variables=variables, variables_latex=variables_latex, labelsize=20, normalbar=True, contours=np.arange(0, 20, 0.5),norm_to_zeroth=True):
    output = procname + '_' + plotname + '_' + var1 
    print('Plotting', output)
    nvar1 = [key for key, value in variables.items() if value == var1][0]
    #print(var1, var2)
    #print(nvar1, nvar2)
    # construct the axes for the plot
    # no need to modify this if you just need one plot
    gs = gridspec.GridSpec(4, 4)
    if figext == '':
        fig = plt.figure()
    else:
        fig = figext
    if axext == '':
        ax = fig.add_subplot(111)
    else:
        ax=axext
    ax.grid(False)
    ax.set_title(plottitle)
    # create legend and plot/font size
    #ax.legend()
    #ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8})
    # set the ticks, labels and limits etc.
    xlab = '$' + variables_latex[nvar1] + '$'
    ylab = r'$\sigma/\sigma_\mathrm{SM}$'
    if setylabel == True:
        ax.set_ylabel(ylab, fontsize=labelsize)
    if setxlabel == True:
        ax.set_xlabel(xlab, fontsize=labelsize)
    
    # choose x and y log scales
    #if ylog:
    #    ax.set_yscale('log')
    #else:
    #    ax.set_yscale('linear')
    #if xlog:
    #    ax.set_xscale('log')
    #else:
    #    ax.set_xscale('linear')
    # set the limits on the x and y axes if required below:
    ymin = ylim[0]
    ymax = ylim[1]
    xmin = xlim[0]
    xmax = xlim[1]
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    ctexts = []
    cvartexts = []
    for i in range(0, len(variables.keys())):
        if i != nvar1:
            ctext = variables[i] + '=0'
            ctexts.append(ctext)
        else:
            cvartexts.append(variables[i])
    #print(ctexts)
    fstr = 'partial(func_t_CX, ' + ','.join([ct for ct in ctexts]) + ', procname=Process)'
    global func_CX_partial
    func_CX_partial = eval(fstr)
    #print(func_CX_partial)
    #print(fit_coeffs)
    global fit_coeffs_g
    fit_coeffs_g = fit_coeffs
    #print(cvartexts[0], cvartexts[1])
    if norm_to_zeroth is True:
        feval = 'func_CX_partial(' + cvartexts[0] +'=x1,coeffs=fit_coeffs_g)/func_CX_partial(' + cvartexts[0] +'=0,coeffs=fit_coeffs_g)'
    else:
        feval = 'func_CX_partial(' + cvartexts[0] +'=x1,coeffs=fit_coeffs_g)'
    func_fin = lambda x1: eval(feval)
    #print(func_fin(0.05, -0.05))
    X = np.linspace(xlim[0], xlim[1], nbins)
    Z = func_fin(X)
    
    line = ax.plot(X, Z, marker='', ls='--', color='blue', lw=3)
    ax.axhline(y=1.0,  linewidth=0.5, color = 'k', ls='--')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if savefig == True:
        # save the figure
        print('saving the figure')
        # save the figure in PDF format
        infile = output + '.dat'
        print('---')
        print('output in', plot_dir + infile.replace('.dat','.pdf'))
        plt.savefig(plot_dir + infile.replace('.dat','.pdf'), bbox_inches='tight')
        plt.close(fig)
    return line


def correlation_plot(procname, plotname, popt, varnames,plottitle='',contours=np.arange(-2, 32, 2),norm_to_zeroth=True):
    ###################################################################################
    # correlation plots for cross section
    ###################################################################################
    print('---')
    print('plotting correlation plots for', procname, plotname)
    # plot settings ########
    output = procname + '_' + plotname + '_correlation'

    fig2 = plt.figure(figsize=(9,9))
    spec2 = gridspec.GridSpec(ncols=len(variables), nrows=len(variables),wspace=0, hspace=0, figure=fig2)

    f2_ax_array = []
    cc = 0
    for i in range(len(varnames)):
        for j in range(len(varnames)):
            if i > j:
                if procname == 'gg_hh' and (varnames[i] == 'd4' or varnames[j] == 'd4' or varnames[i] == 'ct3' or varnames[j] == 'ct3' or varnames[i] == 'cb3' or varnames[j] == 'cb3'):
                    continue
                f2_ax = fig2.add_subplot(spec2[i, j])
                f2_ax.set_box_aspect(1)
                f2_ax.xaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
                f2_ax.yaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
                f2_ax.tick_params(axis='both', labelsize=5)
                f2_ax_array.append(f2_ax)
                cc = cc+1
    spec2.update(wspace=0,hspace=0)

    nplots = len(varnames)**2
    cc = 0
    for i in range(len(varnames)):
        for j in range(len(varnames)):
            if i > j:
                if procname == 'gg_hh' and (varnames[i] == 'd4' or varnames[j] == 'd4' or varnames[i] == 'ct3' or varnames[j] == 'ct3' or varnames[i] == 'cb3' or varnames[j] == 'cb3'):
                    continue
                labelx=False
                labely=False
                if i == len(varnames)-1 or (procname=='gg_hh' and i==len(varnames)-4):
                    labelx=True
                else:
                    f2_ax_array[cc].set(xticks=[])
                if j == 0:
                    labely = True
                else:
                    f2_ax_array[cc].set(yticks=[])
                if varnames[j] == 'c3' and varnames[i] != 'd4':
                     cont = contour_xsec(Process, 'xsec', '', popt, varnames[j], varnames[i], [-10.0, 10.0], [-1.0, 1.0], smtext=False, starsize=2, setxlabel=labelx, setylabel=labely, figext=fig2, axext=f2_ax_array[cc], savefig=False, labelsize=15, normalbar=False,contours=contours, norm_to_zeroth=norm_to_zeroth)
                elif varnames[j] == 'c3' and varnames[i] == 'd4':
                    cont = contour_xsec(Process, 'xsec', '', popt, varnames[j], varnames[i], [-10.0, 10.0], [-40.0, 40.0], smtext=False, starsize=2, setxlabel=labelx, setylabel=labely, figext=fig2, axext=f2_ax_array[cc], savefig=False, labelsize=15, normalbar=False,contours=contours, norm_to_zeroth=norm_to_zeroth)
                elif varnames[j] != 'c3' and varnames[i] == 'd4':
                    cont = contour_xsec(Process, 'xsec', '', popt, varnames[j], varnames[i], [-1.0, 1.0], [-40.0, 40.0], smtext=False, starsize=2, setxlabel=labelx, setylabel=labely, figext=fig2, axext=f2_ax_array[cc], savefig=False, labelsize=15, normalbar=False,contours=contours, norm_to_zeroth=norm_to_zeroth)
                else:
                    cont = contour_xsec(Process, 'xsec', '', popt, varnames[j], varnames[i], [-1.0, 1.0], [-1.0, 1.0], smtext=False, starsize=2, setxlabel=labelx, setylabel=labely, figext=fig2, axext=f2_ax_array[cc], savefig=False, labelsize=15, normalbar=False,contours=contours, norm_to_zeroth=norm_to_zeroth)
                cc = cc + 1
    #fig2.tight_layout()
    #plt.subplots_adjust(wspace=0, hspace=0)
    #fig2.colorbar(cont, ax=f2_ax_array[-1])
    if procname == 'gg_hhh':
        axins = inset_axes(f2_ax_array[-1], # here using axis of the lowest plot
                width="20%",  # width = 5% of parent_bbox width
                height="280%",  # height : 340% good for a (4x4) Grid
                loc='lower left',
                    bbox_to_anchor=(1.08, 0.15, 1, 1),
                    bbox_transform=f2_ax_array[-1].transAxes,
                borderpad=0,
                )
    elif procname == 'gg_hh': 
        axins = inset_axes(f2_ax_array[-1], # here using axis of the lowest plot
                width="28%",  # width = 5% of parent_bbox width
                height="550%",  # height : 340% good for a (4x4) Grid
                loc='lower left',
                    bbox_to_anchor=(1.04, 0.1, 1, 1),
                    bbox_transform=f2_ax_array[-1].transAxes,
                borderpad=0,
                )
        
    cb = fig2.colorbar(cont, cax=axins)
    if procname == 'gg_hhh':
        fig2.suptitle(plottitle,y=0.72,fontsize=15)
    elif procname == 'gg_hh':
        fig2.suptitle(plottitle,x=0.4,y=0.8,fontsize=10)
    # save the figure
    print('saving the figure')
    # save the figure in PDF format
    infile = output + '.dat'
    print('---')
    print('output in', plot_dir + infile.replace('.dat','.pdf'))
    plt.savefig(plot_dir + infile.replace('.dat','.pdf'), bbox_inches='tight')
    plt.close(fig2)

    ####################


# function to save the fit for Process in the fit_dir for a specific RunNum:
def saveFit(popt, pcov, Process, RunNum):
    filename = fit_dir + 'fit_' + Process + '_run' + str(RunNum) + smearing_tag + '.dat'
    f = open(filename,'w')
    f.write('\t'.join((str(x) for x in popt)))
    f.write('\n')
    f.write('\t'.join((str(x) for x in pcov)))
    f.close()
# function to read the fit for Process in the fit_dir for a specific RunNum:
def readFit(Process, RunNum):
    filename = fit_dir + 'fit_' + Process + '_run' + str(RunNum) + smearing_tag + '.dat'
    print('Reading fit from', filename)
    f = open(filename, 'r')
    for i,line in enumerate(f):
        if i == 0:
            if len(line.split())!= NCoeffs[Process]:
                print('Error: the number of coefficients found is insufficient: expected:', NCoeffs[Process], 'got:', len(line.split()))
                exit()
            else:
                popt = [float(x) for x in line.split()]
    pcov = [] # WARNING: COVARIANCE IS EMPTY HERE!
    return popt, pcov


def drive_mg_proc(runnum, mgloc, procloc, procname, CouplingsArray, nevents, nruns, ecm=14):
    filename = mgloc + '/' + procname + '_coupvar_run' + str(runnum) + '.dcmd'
    print('generating mg5input:', filename)
    ebeam1 = ecm*1000/2
    ebeam2 = ebeam1
    counter = 0
    for coups in CouplingsArray:
        if counter > nruns:
            break
        lhe = 'run_' + procname + '_' + str(RunNum) + '_' + '_'.join((coups)) + '/unweighted_events.lhe.gz'
        lhefile = mgloc + '/' + procloc + 'Events/' + lhe
        if os.path.exists(lhefile) is False:
            filestream = open(filename,'w')
            filestream.write('launch run_' + procname + '_' + str(RunNum) + '_' + '_'.join((coups)) + ' --accuracy=0.25 --points=300 --iterations=1\n0\n')
            filestream.write('set ebeam1 ' + str(ebeam1) + '\n')
            filestream.write('set ebeam2 ' + str(ebeam2) + '\n')
            filestream.write('set d3 ' + str(coups[0]) + '\n')
            filestream.write('set d4 ' + str(coups[1]) + '\n')
            #filestream.write('set cg1 ' + str(coups[2]) + '\n')
            #filestream.write('set cg2 ' + str(coups[3]) + '\n')
            #filestream.write('set ct1 ' + str(coups[4]) + '\n')
            #filestream.write('set cb1 ' + str(coups[5]) + '\n')
            filestream.write('set ct2 ' + str(coups[6]) + '\n')
            #filestream.write('set cb2 ' + str(coups[7]) + '\n')
            filestream.write('set ct3 ' + str(coups[8]) + '\n')
            if MODEL == 'HEFT6':
                filestream.write('set ct1 ' + str(coups[9]) + '\n')
            filestream.write('set nevents ' + str(nevents) + '\n')
            filestream.write('0')
            filestream.close()
            # run mg5 with the file generated
            runcommand = 'cat ' + filename
            p = subprocess.run(runcommand, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=mgloc + '/' + procloc)
            runcommand = mgloc + '/' + procloc + '/bin/madevent ' + filename
            p = subprocess.Popen(runcommand, shell=True, text=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=mgloc + '/' + procloc)
            for line in iter(p.stdout.readline, b''):
                print(line)
            print(p.stdout)
            print(p.stderr)
                
            counter = counter + 1
    return counter

# function that runs herwig for specific final states
def run_herwig_proc(runnum, mgloc, hwloc, procloc, procname, CouplingsArray, nevents, nruns, ecm=100):
    print('Running Herwig from the input files previously generated, for:', procname, 'at Energy=', Energy)
    for coups in CouplingsArray:
        #print(lams)
        lhe = 'run_' + procname + '_' + str(RunNum) + '_' + '_'.join((coups)) + '/unweighted_events.lhe.gz'
        lhefile = mgloc + '/' + procloc + 'Events/' + lhe
        if os.path.exists(lhefile) is False:
            print('File', lhefile, 'does not exist, cannot run Herwig!')
            exit()
        # get the template and write the input file:
        # Signal is LO
        HerwigInputTemplate = getTemplate(HW_template[0])
        processname = 'HW-' + str(RunNum) + '_' + '_'.join((coups)) + '_' + FinalState
        hwinputfile = processname + '.in'
        parmtextsubs = {
            'PROCESSNAME' : processname, 
            'LHEFILE' : lhefile,
            'OUTPUTLOCATION' : 'events/',
            'FatAnalysis' : '#',
            'HwSimLibrary' : 'HwSim',
            'FinalState6b' : FinalState6b,
            'FinalStatebtau' : FinalStatebtau,
            'FinalStatebgamma' : FinalStatebgamma
            
        }
        print('\t\twriting', hwinputfile)
        writeFile(HerwigLocation + hwinputfile, HerwigInputTemplate.substitute(parmtextsubs) )

        # check if the root file already exists. if it does, only run if ReRun is set to True
        hwrunfile = processname + '.run'
        outputlocation = HerwigOutputLocation
        rootfile = outputlocation + processname + '.root'
        print("Checking rootfile:", rootfile)
        
        if os.path.exists(rootfile) is True:
            print('File', rootfile, 'exists')
        if os.path.exists(rootfile) is False or (os.path.exists(rootfile) is True and ReRunHerwig is True): # if the root file exists, do not proceed except if ReRun is true
                if os.path.exists(rootfile) is True and ReRunHerwig is True:
                    print('File', rootfile, 'exists, but have chosen to re-run!')
                # get the number of events in the corresponding lhe file:
                zgrepcommand = 'zgrep "= nevents" ' + lhefile
                print(zgrepcommand)
                p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=HerwigLocation)
                for line in iter(p.stdout.readline, b''):
                    nevents = float(line.split()[0])
                print('\t\tHerwig reading:', hwinputfile)
                readcommand = 'Herwig read ' + hwinputfile
                print(readcommand)
                p = subprocess.Popen(readcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=HerwigLocation)
                for line in iter(p.stdout.readline, b''):
                    print('\t\t', line, end=' ')
                out, err = p.communicate()
                #print out, err
                print('\t\tHerwig running:', hwrunfile, 'for', nevents, 'events')
                runcommand = 'Herwig run ' + hwrunfile + ' -N' + str(int(nevents*Reduction_Fac[0]))
                p = subprocess.Popen(runcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=HerwigLocation)
                for line in iter(p.stdout.readline, b''):
                    print('\t\t', line, end=' ')
                out, err = p.communicate()
                #print out, err


def run_herwig_proc_parallel(runnum, mgloc, hwloc, procloc, procname, CouplingsArray, nevents, nruns, ecm=100):
    print('Running Herwig from the input files previously generated, for:', procname, 'at Energy=', ecm)
    
    def worker(coups):
        lhe = 'run_' + procname + '_' + str(runnum) + '_' + '_'.join((coups)) + '/unweighted_events.lhe.gz'
        lhefile = mgloc + '/' + procloc + 'Events/' + lhe
        if not os.path.exists(lhefile):
            print('File', lhefile, 'does not exist, cannot run Herwig!')
            return  # Skip this job

        HerwigInputTemplate = getTemplate(HW_template[0])
        processname = 'HW-' + str(runnum) + '_' + '_'.join((coups)) + '_' + FinalState
        hwinputfile = processname + '.in'
        parmtextsubs = {
            'PROCESSNAME' : processname, 
            'LHEFILE' : lhefile,
            'OUTPUTLOCATION' : 'events/',
            'FatAnalysis' : '#',
            'HwSimLibrary' : 'HwSim',
            'FinalState6b' : FinalState6b,
            'FinalStatebtau' : FinalStatebtau,
            'FinalStatebgamma' : FinalStatebgamma
        }
        print('\t\twriting', hwinputfile)
        writeFile(HerwigLocation + hwinputfile, HerwigInputTemplate.substitute(parmtextsubs))

        hwrunfile = processname + '.run'
        outputlocation = HerwigOutputLocation
        rootfile = outputlocation + processname + '.root'
        print("Checking rootfile:", rootfile)

        rerun = (not os.path.exists(rootfile)) or (os.path.exists(rootfile) and ReRunHerwig)
        if os.path.exists(rootfile):
            print('File', rootfile, 'exists')
        if rerun:
            if os.path.exists(rootfile) and ReRunHerwig:
                print('File', rootfile, 'exists, but have chosen to re-run!')
            zgrepcommand = f'zgrep "= nevents" {lhefile}'
            print(zgrepcommand)
            p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=HerwigLocation)
            output, _ = p.communicate()
            try:
                nevents_local = float(output.decode().split()[0])
            except Exception:
                print('Could not parse nevents from LHE file, skipping', lhefile)
                return
            print('\t\tHerwig reading:', hwinputfile)
            readcommand = f'Herwig read {hwinputfile}'
            print(readcommand)
            p = subprocess.Popen(readcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=HerwigLocation)
            for line in iter(p.stdout.readline, b''):
                print('\t\t', line.decode(), end=' ')
            p.communicate()

            print('\t\tHerwig running:', hwrunfile, 'for', nevents_local, 'events')
            runcommand = f'Herwig run {hwrunfile} -N{int(nevents_local * Reduction_Fac[0])}'
            print(runcommand)
            p = subprocess.Popen(runcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=HerwigLocation)
            for line in iter(p.stdout.readline, b''):
                print('\t\t', line.decode(), end=' ')
            p.communicate()

    # Launch all Herwig runs in parallel
    Parallel(n_jobs=-1, backend="loky")(
        delayed(worker)(coups) for coups in CouplingsArray
    )

# function to read the analysis results and test the fit          
def test_fit_analysis(runnum, mgloc, procloc, procname, CouplingsArray, ntotal, popt):
    X = []
    Z = []
    EFFICIENCY = {}
    ZERR = []
    func_CX_proc = partial(func_CX, procname=Process)
    fracdiff_avg = 0.
    for coups in CouplingsArray:
        outputlocation = HerwigOutputLocation
        processname = 'HW-' + str(RunNum) + '_' + '_'.join((coups))
        rootfile = outputlocation + processname + '_' + FinalState + '.root'
        print('rootfile=', rootfile)
        analysisOutputfile = outputlocation + processname + '.smear' + smearing_tag + '.dat'
        if os.path.exists(analysisOutputfile)is False:
            print('File', analysisOutputfile, 'does not exist!')
            exit()
        else:
            print('File', analysisOutputfile, ' exists, reading results')
            zgrepcommand = 'cat ' + analysisOutputfile
            p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
            for line in iter(p.stdout.readline, b''):
                efficiency = float(line.split()[0])
            #print('efficiency=', efficiency)
            coups_tuple = []
            for mm in range(len(coups)):
                coups_tuple.append(float(coups[mm]))
            X.append(tuple(coups_tuple))
            Z.append(float(efficiency))
            EFFICIENCY[tuple(coups_tuple)] = float(efficiency)
            # get the fitted XSEC
            eff_fit = func_CX_proc(coups_tuple, *popt)
            if efficiency != 0:
                fracdiff = abs(efficiency-eff_fit)/efficiency
            else:
                fracdiff = 0
            if fracdiff > 0.5:
                print(coups, efficiency)
                print('!!! xsec: real, fitted, frac diff =', efficiency, eff_fit, fracdiff)
            fracdiff_avg = fracdiff_avg + fracdiff
            #print(X)
    print('average fractional difference =', fracdiff_avg/ntotal)

def run_analysis(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
    for line in iter(p.stdout.readline, b''):
        print('\t\t', line.decode(), end=' ')
    out, err = p.communicate()
    print('\n')

# run the xgboost analysis - chatgpt modification using joblib
def run_analysis_xgboost(runnum, mgloc, hwloc, procloc, procname, CouplingsArray, nevents, nruns,
                        model_file, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S, 
                        sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, training_seed, ecm=14):
    print('Running Analysis on the root files, for:', procname, 'at Energy=', Energy)
    X = []
    Z = []
    EFFICIENCY = {}
    EFFICIENCY_BKG = {}
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    jobs = []
    for coups in CouplingsArray:
        outputlocation = HerwigOutputLocation
        processname = 'HW-' + str(runnum) + '_' + '_'.join((coups))
        rootfile = outputlocation + processname + '_' + FinalState + '.root'
        analysisOutputfile = outputlocation + processname + smearing_tag + '.XGBOOST.dat'
        analysisInputfile = outputlocation + processname + '_var.smear' + smearing_tag + '.root'
        #print("Checking analysis output:", analysisOutputfile)

        if os.path.exists(analysisOutputfile) and not ReRunAnalysisXGBOOST:
            #print('File', analysisOutputfile, ' already exists, reading results')
            p = subprocess.Popen(f"cat {analysisOutputfile}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(p.stdout.readline, b''):
                if not line:
                    break
                efficiency = float(line.split()[0])
            #print('efficiency=', efficiency)
            coups_tuple = tuple(float(c) for c in coups)
            X.append(coups_tuple)
            Z.append(float(efficiency))
            EFFICIENCY[coups_tuple] = float(efficiency)
        else:
            if os.path.exists(analysisOutputfile) and ReRunAnalysisXGBOOST:
                print('File', analysisOutputfile, 'exists, but have chosen to re-run analysis!')
            if not os.path.exists(rootfile):
                print('Error: ROOT file:', rootfile, 'does not exist!')
                exit()
            print('running the XGBOOST analysis on the input file', analysisInputfile)
            jobs.append((
                model_file, analysisInputfile, Backgrounds, Background_files, Backgrounds_xsec, xsS,
                initial_S, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, training_seed, smearing_tag
            ))

    # Run all XGBoost analyses in parallel
    if jobs:
        Parallel(n_jobs=-1, backend="loky")(
            delayed(apply_xgboost_write)(*args) for args in jobs
        )

    for bkg in Backgrounds:  # background loop
        processname = 'HW-' + str(bkg) + '_' + str(Energy)
        rootfile = BackgroundLocation + processname + '.root'
        analysisOutputfile = BackgroundLocation + processname + smearing_tag + '.XGBOOST.dat'
        #print("Checking analysis output:", analysisOutputfile) 
        if os.path.exists(analysisOutputfile):
            print('File', analysisOutputfile, ' exists, reading results')
            p = subprocess.Popen(f"cat {analysisOutputfile}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(p.stdout.readline, b''):
                if not line:
                    break
                efficiency = float(line.split()[0])
            print(bkg, 'efficiency=', efficiency)
            EFFICIENCY_BKG[bkg] = float(efficiency)
            continue
        else:
            print('Error, analysis for bkg', bkg, 'does not exist!', analysisOutputfile)
            exit()
    return np.transpose(X), Z, EFFICIENCY, EFFICIENCY_BKG

# run the analysis on signal and background USING XGBOOST             
def run_analysis_xgboost_threads(runnum, mgloc, hwloc, procloc, procname, CouplingsArray, nevents, nruns, trained_model, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, training_seed, ecm=14):
    print('Running Analysis on the root files, for:', procname, 'at Energy=', Energy)
    X = []
    Z = []
    EFFICIENCY = {}
    EFFICIENCY_BKG = {}
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
    #print(Max_Jobs)
    threads = list()
    for coups in CouplingsArray:
        #  write the analysis input file:
        outputlocation = HerwigOutputLocation
        processname = 'HW-' + str(RunNum) + '_' + '_'.join((coups))
        rootfile = outputlocation + processname + '_' + FinalState + '.root'
        analysisOutputfile = outputlocation + processname + smearing_tag + '.XGBOOST.dat'
        analysisInputfile = outputlocation + processname + '_var.smear' + smearing_tag + '.root'
        #print("Checking analysis output:", analysisOutputfile)
        if os.path.exists(analysisOutputfile) is True and ReRunAnalysis is False:
            print('File', analysisOutputfile, ' already exists, reading results')
            zgrepcommand = 'cat ' + analysisOutputfile
            p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
            for line in iter(p.stdout.readline, b''):
                efficiency = float(line.split()[0])
            print('efficiency=', efficiency)
            coups_tuple = []
            for mm in range(len(coups)):
                coups_tuple.append(float(coups[mm]))
            X.append(tuple(coups_tuple))
            Z.append(float(efficiency))
            EFFICIENCY[tuple(coups_tuple)] = float(efficiency)
        elif (os.path.exists(analysisOutputfile) is False) or (os.path.exists(analysisOutputfile) is True and ReRunAnalysisXGBOOST is True): # if the root file exists, do not proceed except if ReRun is true
                if os.path.exists(analysisOutputfile) is True and ReRunAnalysisXGBOOST is True:
                    print('File', analysisOutputfile, 'exists, but have chosen to re-run analysis!')
                if os.path.exists(rootfile) is False:
                    print('Error: ROOT file:', rootfile, 'does not exist!')
                    exit()
                print('running the XGBOOST analysis on the input file', analysisInputfile)
                print('Launching: apply_xgboost_write with:', trained_model, analysisInputfile, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, training_seed, smearing_tag)
                
                x = threading.Thread(target=apply_xgboost_write, args=(trained_model, analysisInputfile, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, training_seed, smearing_tag))
                #x = multiprocessing.Process(target=apply_xgboost_write, args=(trained_model, analysisInputfile, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, training_seed,))
                x.start()
                x.join()
                print(x.exitcode) 
                #threads.append(x)
    #for index, thread in enumerate(threads):
    #    logging.info("Main    : before joining thread %d.", index)
    #    thread.join()
    #    logging.info("Main    : thread %d done", index)
    for bkg in Backgrounds: # background loop
        processname = 'HW-' + str(bkg) + '_' + str(Energy)
        rootfile = BackgroundLocation + processname + '.root'
        analysisOutputfile = BackgroundLocation + processname + smearing_tag + '.XGBOOST.dat'
        print("Checking analysis output:", analysisOutputfile) 
        if os.path.exists(analysisOutputfile) is True:
            print('File', analysisOutputfile, ' exists, reading results')
            zgrepcommand = 'cat ' + analysisOutputfile
            p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
            for line in iter(p.stdout.readline, b''):
                efficiency = float(line.split()[0])
            print(bkg, 'efficiency=', efficiency)
            EFFICIENCY_BKG[bkg] = float(efficiency)
            continue
        if os.path.exists(analysisOutputfile) is False: # if the root file exists, do not proceed except if ReRun is true
                print('Error, analysis for bkg', bkg, 'does not exist!')
                exit()
    return np.transpose(X), Z, EFFICIENCY, EFFICIENCY_BKG


    
# run the analysis on signal and background               
def run_analysis_proc(runnum, mgloc, hwloc, procloc, procname, CouplingsArray, nevents, nruns, ecm=14):
    print('Running Analysis on the root files, for:', procname, 'at Energy=', Energy)
    X = []
    Z = []
    EFFICIENCY = {}
    EFFICIENCY_BKG = {}
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
    #print(Max_Jobs)
    threads = list()
    for coups in CouplingsArray:
        #  write the analysis input file:
        outputlocation = HerwigOutputLocation
        processname = 'HW-' + str(RunNum) + '_' + '_'.join((coups))
        rootfile = outputlocation + processname + '_' + FinalState + '.root'
        analysisOutputfile = outputlocation + processname + '.smear' + smearing_tag + '.dat'
        analysisInputfile = outputlocation + processname + '.input'
        analysisInputstream = open(analysisInputfile,'w') 
        print("Checking analysis output:", analysisOutputfile)
        if os.path.exists(analysisOutputfile) is True and ReRunAnalysis is False:
            print('File', analysisOutputfile, ' already exists, reading results')
            zgrepcommand = 'cat ' + analysisOutputfile
            p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
            for line in iter(p.stdout.readline, b''):
                efficiency = float(line.split()[0])
            print('efficiency=', efficiency)
            coups_tuple = []
            for mm in range(len(coups)):
                coups_tuple.append(float(coups[mm]))
            X.append(tuple(coups_tuple))
            Z.append(float(efficiency))
            EFFICIENCY[tuple(coups_tuple)] = float(efficiency)
        if os.path.exists(analysisOutputfile) is False or (os.path.exists(analysisOutputfile) is True and ReRunAnalysis is True): # if the root file exists, do not proceed except if ReRun is true
                if os.path.exists(analysisOutputfile) is True and ReRunAnalysis is True:
                    print('File', analysisOutputfile, 'exists, but have chosen to re-run analysis!')
                if os.path.exists(rootfile) is False:
                    print('Error: ROOT file:', rootfile, 'does not exist!')
                    exit()
                elif os.path.exists(rootfile) is True:
                    analysisInputstream.write(rootfile + '\n')
                    analysisInputstream.close()
                print('running the analysis', ExecutableSmear[Energy], 'on the input file', analysisInputfile)
                analysiscommand = ExecutableSmear[Energy] + ' ' + analysisInputfile
                print('Launching:', analysiscommand)

                #p = subprocess.Popen(analysiscommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
                #for line in iter(p.stdout.readline, b''):
                #print('\t\t', line, end=' ')
                #out, err = p.communicate()
                #print('\n')
                x = threading.Thread(target=run_analysis, args=(analysiscommand,))
                threads.append(x)
                x.start()
    for index, thread in enumerate(threads):
        #logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)
    for bkg in Backgrounds: # background loop
        processname = 'HW-' + str(bkg) + '_' + str(Energy)
        rootfile = BackgroundLocation + processname + '.root'
        analysisOutputfile = BackgroundLocation + processname + '.smear' + smearing_tag + '.dat'
        analysisInputfile = BackgroundLocation + processname + '.input'
        analysisInputstream = open(analysisInputfile,'w') 
        print("Checking analysis output:", analysisOutputfile) 
        if os.path.exists(analysisOutputfile) is True and ReRunAnalysis is False:
            print('File', analysisOutputfile, ' already exists, reading results')
            zgrepcommand = 'cat ' + analysisOutputfile
            p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
            for line in iter(p.stdout.readline, b''):
                efficiency = float(line.split()[0])
            print(bkg, 'efficiency=', efficiency)
            EFFICIENCY_BKG[bkg] = float(efficiency)
            continue
        if os.path.exists(analysisOutputfile) is False or (os.path.exists(analysisOutputfile) is True and ReRunAnalysis is True): # if the root file exists, do not proceed except if ReRun is true
                if os.path.exists(analysisOutputfile) is True and ReRunAnalysis is True:
                    print('File', analysisOutputfile, 'exists, but have chosen to re-run analysis!')
                if os.path.exists(rootfile) is False:
                    print('Error: ROOT file:', rootfile, 'does not exist!')
                    exit()
                elif os.path.exists(rootfile) is True:
                    analysisInputstream.write(rootfile + '\n')
                    analysisInputstream.close()
                print('running the analysis', ExecutableSmear[Energy], 'on the input file', analysisInputfile)
                analysiscommand = ExecutableSmear[Energy] + ' ' + analysisInputfile
                p = subprocess.Popen(analysiscommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
                for line in iter(p.stdout.readline, b''):
                        print('\t\t', line, end=' ')
                out, err = p.communicate()
                print('\n')
    return np.transpose(X), Z, EFFICIENCY, EFFICIENCY_BKG


def contour_pvalue_ct3d4_marginalized(procname, plotname, plottitle, fit_coeffs_xsec, fit_coeffs_eff, sigma_bkg, var1, var2, xlim, ylim, axext='', figext='', smtext=True, starsize=15, setxlabel=True, setylabel=True, nbins=200, savefig=True,variables=variables, variables_latex=variables_latex, labelsize=20, normalbar=True, contours=np.arange(0, 10, 0.5),norm_to_zeroth=True, lumi=Luminosity):
    output = procname + '_' + plotname + '_' + var1 + '_' + var2 + '_including_constraints_marginalized'
    print('Plotting', output)
    nvar1 = [key for key, value in variables.items() if value == var1][0]
    nvar2 = [key for key, value in variables.items() if value == var2][0]
    #print(var1, var2)
    #print(nvar1, nvar2)
    # construct the axes for the plot
    # no need to modify this if you just need one plot
    gs = gridspec.GridSpec(4, 4)
    if figext == '':
        fig = plt.figure()
    else:
        fig = figext
    if axext == '':
        ax = fig.add_subplot(111)
    else:
        ax=axext
    ax.grid(False)
    ax.set_title(plottitle)
    
    # set the ticks, labels and limits etc.
    xlab = '$' + variables_latex[nvar1] + '$'
    ylab = '$' + variables_latex[nvar2] + '$'
    if setylabel == True:
        ax.set_ylabel(ylab, fontsize=labelsize)
    if setxlabel == True:
        ax.set_xlabel(xlab, fontsize=labelsize)
        
    # set the limits on the x and y axes if required below:
    ymin = ylim[0]
    ymax = ylim[1]
    xmin = xlim[0]
    xmax = xlim[1]
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    ctexts = []
    cvartexts = []
    for i in range(0, len(variables.keys())):
        #if i != nvar1 and i != nvar2:
        #   ctext = variables[i] + '=0'
        #    ctexts.append(ctext)
        #else:
        cvartexts.append(variables[i])
    #print(ctexts)
    #fstr = 'partial(func_t_CX, ' + ','.join([ct for ct in ctexts]) + ', procname=Process)'
    fstr = 'partial(func_t_CX, procname=Process)'
    global func_CX_partial
    func_CX_partial = eval(fstr)
    #print('func_CX_partial=', func_CX_partial)
    #print(fit_coeffs)
    global fit_coeffs_g_xsec
    fit_coeffs_g_xsec = fit_coeffs_xsec
    global fit_coeffs_g_eff
    fit_coeffs_g_eff = fit_coeffs_eff
    #print(cvartexts[0], cvartexts[1])

    # functions for xsec and significance:
    print(cvartexts)
    # 0     1      2      3     
    #['c3', 'ct2', 'ct3', 'd4']
    feval_xsec = 'func_CX_partial(' + cvartexts[0] +'=0,' + cvartexts[1] + '=0,' + cvartexts[2] + '=x1,' + cvartexts[3] + '=x2,coeffs=fit_coeffs_g_xsec)'
    feval_eff = 'func_CX_partial(' + cvartexts[0] +'=0,' + cvartexts[1] + '=0,'  + cvartexts[2] + '=x1,' + cvartexts[3] + '=x2,coeffs=fit_coeffs_g_eff)'

    feval_xsec_g = 'func_CX_partial(' + cvartexts[0] +'=x3,' + cvartexts[1] + '=x4,' + cvartexts[2] + '=x1,' + cvartexts[3] + '=x2,coeffs=fit_coeffs_g_xsec)'
    feval_eff_g  = 'func_CX_partial(' + cvartexts[0] +'=x3,' + cvartexts[1] + '=x4,' + cvartexts[2] + '=x1,' + cvartexts[3] + '=x2,coeffs=fit_coeffs_g_eff)'
  
    
    func_fin = lambda x1, x2: eval(feval_xsec) * sig_factors * eval(feval_eff) * lumi * 1000. / math.sqrt(sigma_bkg * bkg_factors * lumi + Systematics**2 * (sigma_bkg*bkg_factors)**2 * lumi**2)
    #func_fin = lambda x1, x2: significance(eval(feval_xsec) * eval(feval_eff) * lumi * 1000., sigma_bkg * lumi, Systematics)

    pfunc_fin = lambda x1, x2: (1 - 2 * scipy.special.ndtr(-(eval(feval_xsec) * sig_factors * eval(feval_eff) * lumi * 1000. / math.sqrt(sigma_bkg * bkg_factors * lumi + Systematics**2 * (sigma_bkg*bkg_factors)**2 * lumi**2))))

    pfunc_fin_gaussrw = lambda x1, x2, x3, x4: (1 - 2 * scipy.special.ndtr(-(eval(feval_xsec_g) * sig_factors * eval(feval_eff_g) * lumi * 1000. / math.sqrt(sigma_bkg * bkg_factors * lumi + Systematics**2 * (sigma_bkg*bkg_factors)**2 * lumi**2)))) * gaussian(x3, 0, constraints[Energy][0]) * gaussian(x4, 0, constraints[Energy][1])

    # SM significance: 
    feval_xsec_sm = 'func_CX_partial(' + cvartexts[0] +'=0,' + cvartexts[1] + '=0,' + cvartexts[2] + '=0,' + cvartexts[3] + '=0,coeffs=fit_coeffs_g_xsec)'
    feval_eff_sm = 'func_CX_partial(' + cvartexts[0] +'=0,' + cvartexts[1] + '=0,' + cvartexts[2] + '=0,' + cvartexts[3] + '=0,coeffs=fit_coeffs_g_eff)'
    
    sm_signif= eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * 1000. * lumi / math.sqrt(sigma_bkg * bkg_factors * lumi + Systematics**2 * (sigma_bkg*bkg_factors)**2 * lumi**2)

    # if SM is the "null" hypothesis:
    # SM number of events:
    S_SM = eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * 1000. * lumi
    # SM total uncertainty, including the background uncertainty:
    delta_SM =  math.sqrt(S_SM + sigma_bkg * bkg_factors * lumi + Systematics**2 * (sigma_bkg*bkg_factors)**2 * lumi**2) 
    # {c_i} number of events in the 4D model:
    S_i_4D = lambda x1, x2, x3, x4: eval(feval_xsec_g) * sig_factors * eval(feval_eff_g) * lumi * 1000.
    # {c_i} number of events in 2D model:
    S_i_2D = lambda x1, x2: eval(feval_xsec) * sig_factors * eval(feval_eff) * lumi * 1000.
    # significance versus the SM in the 4D model:
    func_fin_SM_4D = lambda x1, x2, x3, x4: np.power( (S_SM - S_i_4D(x1, x2, x3, x4))/delta_SM, 2)
    # significance versus the SM in the 2D mode: 
    func_fin_SM_2D = lambda x1, x2: np.power( (S_SM - S_i_2D(x1, x2))/delta_SM,2)
    
    # p-value in the 4D model (NO gaussian RW):
    pfunc_fin_SM_4D = lambda x1, x2, x3, x4: 1/(np.sqrt(2.*np.pi)*delta_SM)*np.exp(-func_fin_SM_4D(x1, x2, x3, x4)/2)
    # p-value in the 4D model (WITH gaussian RW):
    pfunc_fin_SM_4D_g = lambda x1, x2, x3, x4: 1/(np.sqrt(2.*np.pi)*delta_SM)*np.exp(-func_fin_SM_4D(x1, x2, x3, x4)/2) * gaussian(x3, 0, constraints[Energy][0]) * gaussian(x4, 0, constraints[Energy][1])
    # p-value in the 2D model:
    pfunc_fin_SM_2D = lambda x1, x2:  1/(np.sqrt(2.*np.pi)*delta_SM)*np.exp(-func_fin_SM_2D(x1, x2)/2)

    print('pfunc_fin_SM_4D_g(0,0,0,0)=',pfunc_fin_SM_4D_g(0,0,0,0))
    print('pfunc_fin_SM_2D(0,0)=',pfunc_fin_SM_2D(0,0))
    print("sigma_sig before anal. [fb]=", eval(feval_xsec_sm)*1000*sig_factors)
    print("analysis eff. on signal=", eval(feval_eff_sm))
    print("sigma_bkg after anal. [fb]=", sigma_bkg * bkg_factors)
    print("sigma sig SM after anal. [fb]=",eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * 1000)
    print("N(bkg)@lumi=", sigma_bkg * bkg_factors * lumi)
    print("N(sig SM)@lumi=", eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * lumi * 1000.) 
    print("SM significance=", sm_signif)
    
    # The two-dimensional p-value (all other coefficients zero):
    x = np.linspace(xlim[0], xlim[1], nbins)
    y = np.linspace(ylim[0], ylim[1], nbins)
    X, Y = np.meshgrid(x,y)
    P = pfunc_fin_SM_2D(X,Y) #func_fin(X,Y)
    #P = P/pfunc_fin_SM_2D(0,0)
    #print(np.amax(P))
    # convert to chi-sq.:
    chisq = stats.chi2.isf(P,2)
    chisq_sub = chisq - np.amin(chisq)
    print('np.amin(chisq)=',np.amin(chisq))
    
    # The four-dimensional p-value: (ct3, d4, c3, ct2)
    x1 = np.linspace(xlim[0], xlim[1], nbins) # ct3
    x2 = np.linspace(ylim[0], ylim[1], nbins) # d4
    nsigma = 10 # number of standard deviations away from the central value
    x3 = np.linspace(-nsigma*constraints[Energy][0],nsigma*constraints[Energy][0], nbins) # c3 limits
    x4 = np.linspace(-nsigma*constraints[Energy][1],nsigma*constraints[Energy][1], nbins) # ct2 limits
    #x3 = np.zeros(nbins)
    #x4 = np.zeros(nbins)
    X1, X2, X3, X4 = np.meshgrid(x1,x2,x3,x4)
    P_g = pfunc_fin_SM_4D_g(X1,X2,X3,X4)
    P_g_marg = np.apply_over_axes(np.sum, P_g, [2,3])
    P_g_marg_s = P_g_marg.reshape(P_g_marg.shape[0], P_g_marg.shape[1])
    P_g_marg_bar = P_g_marg_s*2*nsigma*constraints[Energy][0]*2*nsigma*constraints[Energy][1]/nbins/nbins
    # convert to chi-sq.:
    chisq_marg = stats.chi2.isf(P_g_marg_bar,2)
    chisq_marg_sub = chisq_marg - np.amin(chisq_marg)
    print('np.amin(chisq_marg)=',np.amin(chisq_marg))

    
    #cont = ax.contourf(X, Y, P, contours, cmap='Spectral', extend='max')

    # do the one-dimensional marginalizations:
    P_g_marg_d4 = np.apply_over_axes(np.sum, P_g, [0, 2, 3])
    P_g_marg_ct3 = np.apply_over_axes(np.sum, P_g, [1, 2, 3])
    P_g_marg_d4_s = P_g_marg_d4.reshape(P_g_marg_d4.shape[1])
    P_g_marg_ct3_s = P_g_marg_ct3.reshape(P_g_marg_ct3.shape[0])
    P_g_marg_d4_bar = P_g_marg_d4_s*2*nsigma*constraints[Energy][0]*2*nsigma*constraints[Energy][1]*(xlim[1]-xlim[0])/nbins/nbins/nbins
    P_g_marg_ct3_bar = P_g_marg_d4_s*2*nsigma*constraints[Energy][0]*2*nsigma*constraints[Energy][1]*(ylim[1]-ylim[0])/nbins/nbins/nbins
    chisq_marg_d4 = stats.chi2.isf(P_g_marg_d4_bar,1)
    chisq_marg_d4_sub = chisq_marg_d4 - np.amin(chisq_marg_d4)
    chisq_marg_ct3 = stats.chi2.isf(P_g_marg_ct3_bar,1)
    chisq_marg_ct3_sub = chisq_marg_ct3 - np.amin(chisq_marg_ct3)

    # remove inf and nans if necessary:
    #chisq_marg_d4_sub[np.isinf(chisq_marg_d4_sub)] = np.nan
    #chisq_marg_d4_sub[np.isnan(chisq_marg_d4_sub)] = np.nanmax(chisq_marg_d4_sub, axis=0)
    #chisq_marg_ct3_sub[np.isinf(chisq_marg_ct3_sub)] = np.nan
    #chisq_marg_ct3_sub[np.isnan(chisq_marg_ct3_sub)] = np.nanmax(chisq_marg_ct3_sub, axis=0)
    #print(chisq_marg_d4_sub)
    #print(chisq_marg_ct3_sub)

    # interpolate the 1D functions: 
    func_chisq_1D_d4 = interp1d(x2,chisq_marg_d4_sub, fill_value="extrapolate")
    func_chisq_1D_ct3 = interp1d(x1,chisq_marg_ct3_sub, fill_value="extrapolate")
    #func_chisq_1D_d4 =  make_interp_spline(x2,chisq_marg_d4_sub, k=3)
    #func_chisq_1D_ct3 =  make_interp_spline(x1,chisq_marg_ct3_sub, k=3)
    
    # construction functions to find 1 and 2 sigma limits on d4 and ct3 (from chi-sq min). 
    def func_d4_1sigma(x): return (func_chisq_1D_d4(x) - 0.99)
    def func_d4_2sigma(x): return (func_chisq_1D_d4(x) - 3.84)
    def func_ct3_1sigma(x): return (func_chisq_1D_ct3(x) - 0.99)
    def func_ct3_2sigma(x): return (func_chisq_1D_ct3(x) - 3.84)
        
    # guesses for the locations of the solutions in 1D [change with energy]:
    d4_min_1 = {}
    d4_max_1 = {}
    d4_min_2 = {}
    d4_max_2 = {}
    d4_min_1[13.6] = -10
    d4_max_1[13.6] = 10
    d4_min_2[13.6] = -35 # triple-ins
    d4_max_2[13.6] = 80 # triple-ins
    #d4_min_2[13.6] = -50 # double-ins
    #d4_max_2[13.6] = 20 # double-ins
    
    d4_min_1[100] = -5
    d4_max_1[100] = 32
    d4_min_2[100] = -5
    d4_max_2[100] = 32

    ct3_min_1 = {}
    ct3_max_1 = {}
    ct3_min_2 = {}
    ct3_max_2 = {}
    ct3_min_1[13.6] = -1
    ct3_max_1[13.6] = 2
    ct3_min_2[13.6] = -2
    ct3_max_2[13.6] = 4
    
    ct3_min_1[100] = -0.1
    ct3_max_1[100] = 0.6
    ct3_min_2[100] = -0.8
    ct3_max_2[100] = 0.5

    # calculate and print out the solutions:
    #print('d4@68% CL:', fsolve(func_d4_1sigma, d4_min_1[Energy]), fsolve(func_d4_1sigma, d4_max_1[Energy]))
    #print('d4@95% CL:', fsolve(func_d4_2sigma, d4_max_2[Energy]), fsolve(func_d4_2sigma, d4_max_2[Energy]))
    #print('ct3@68% CL:', fsolve(func_ct3_1sigma, ct3_min_1[Energy]), fsolve(func_ct3_1sigma, ct3_max_1[Energy]))
    #print('ct3@95% CL:', fsolve(func_ct3_2sigma, ct3_max_2[Energy]), fsolve(func_ct3_2sigma, ct3_max_2[Energy]))
    print('d4@68% CL:', fsolve(func_d4_1sigma, [d4_min_1[Energy], d4_max_1[Energy]]))
    print('d4@95% CL:', fsolve(func_d4_2sigma, [d4_min_2[Energy], d4_max_2[Energy]]))
    print('ct3@68% CL:', fsolve(func_ct3_1sigma, [ct3_min_1[Energy], ct3_max_1[Energy]]))
    print('ct3@95% CL:', fsolve(func_ct3_2sigma, [ct3_min_2[Energy], ct3_max_2[Energy]]))

    # plot the contours:
    #ax.clabel(cont)#, inline=True)
    ax.plot(0,0,marker='*',ms=starsize, color='black')
    #cont2 = ax.contour(X, Y, P_g_marg_bar, contours, extend='max', colors=('black'), label='4D')
    #cont = ax.contour(X, Y, P, contours, extend='max', colors=('red'), linestyles=('--'), label='2D')
    cont = ax.contour(X, Y, chisq_marg_sub, contours, extend='max', colors=('black', 'red'), linestyles=('-','--'))
    labels = ['$1\\sigma$', '$2\\sigma$']
    for i in range(len(labels)):
        cont.collections[i].set_label(labels[i])

    #cont = ax.contour(X, Y, chisq_sub, contours, extend='max', colors=('red'), linestyles=('--'), label='2D')

    # add constraints:
    if constraints[Energy][nvar1] != -1:
        ax.axvline(x=constraints[Energy][nvar1],  linewidth=0.5, color = 'k', ls='--')
        ax.axvline(x=-constraints[Energy][nvar1],  linewidth=0.5, color = 'k', ls='--')
    if constraints[Energy][nvar2] != -1:
        ax.axhline(y=constraints[Energy][nvar2],  linewidth=0.5, color = 'k', ls='--')
        ax.axhline(y=-constraints[Energy][nvar2],  linewidth=0.5, color = 'k', ls='--')
    
    if smtext == True:
        ax.text(0.53, 0.53,"SM", transform=ax.transAxes)
    if normalbar == True:
        plt.colorbar(cont)
    #handles, labels = cs.legend_elements()

    # after you’ve done your contour call…
    black_line = mlines.Line2D([], [], color='black', linestyle='-',
                            label='$1\\sigma$')
    red_line   = mlines.Line2D([], [], color='red',   linestyle='--',
                           label='$2\\sigma$')

    ax.legend(handles=[black_line, red_line],
          loc="upper right", frameon=False, prop={'size':8})
        
    #ax.legend()
    #ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8}, handles=[cont, cont2])
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    if Energy == 100:
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    elif Energy == 13.6:
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    if savefig == True:
        # save the figure
        print('saving the figure')
        # save the figure in PDF format
        infile = output + '.dat'
        print('---')
        print('output in', infile.replace('.dat','.pdf'))
        plt.savefig(plot_dir + infile.replace('.dat','.pdf'), bbox_inches='tight')
        plt.close(fig)
        
    return cont




def contour_pvalue_only_old(procname, plotname, plottitle, fit_coeffs_xsec, fit_coeffs_eff, sigma_bkg, var1, var2, plotlimits, searchlimits, deltac3=-1, axext='', figext='', smtext=True, starsize=15, setxlabel=True, setylabel=True, nbins=400, savefig=True,variables=variables, variables_latex=variables_latex, labelsize=20, normalbar=True, contours=np.arange(0, 10, 0.5),norm_to_zeroth=True, lumi=Luminosity):
    output = procname + '_' + plotname + '_' + var1 + '_' + var2
    print('Plotting', output)
    nvar1 = [key for key, value in variables.items() if value == var1][0]
    nvar2 = [key for key, value in variables.items() if value == var2][0]
    nvar3 = [key for key, value in variables.items() if value != var1 and value != var2][0]
    nvar4 = [key for key, value in variables.items() if value != var1 and value != var2][1]

    #print(var1, var2)
    print('nvar1, nvar2=', nvar1, nvar2)
    #print(nvar3, nvar4)
   
        
    # set the limits on the x and y axes if required below:
    ymin = plotlimits[Energy][nvar2][0]
    ymax = plotlimits[Energy][nvar2][1]
    xmin = plotlimits[Energy][nvar1][0]
    xmax = plotlimits[Energy][nvar1][1]
  
    ctexts = []
    cvartexts = []
    for i in range(0, len(variables.keys())):
        cvartexts.append(variables[i])
    fstr = 'partial(func_t_CX, procname=Process)'
    global func_CX_partial
    func_CX_partial = eval(fstr)
    #print('func_CX_partial=', func_CX_partial)
    #print(fit_coeffs)
    global fit_coeffs_g_xsec
    fit_coeffs_g_xsec = fit_coeffs_xsec
    global fit_coeffs_g_eff
    fit_coeffs_g_eff = fit_coeffs_eff
    #print(cvartexts[0], cvartexts[1])

    # functions for xsec and significance:
    print(cvartexts)
    # 0     1      2      3     
    #['c3', 'ct2', 'ct3', 'd4']
    print(cvartexts[nvar3], cvartexts[nvar4], cvartexts[nvar1], cvartexts[nvar2])
    feval_xsec = 'func_CX_partial(' + cvartexts[nvar3] +'=0,' + cvartexts[nvar4] + '=0,' + cvartexts[nvar1] + '=x1,' + cvartexts[nvar2] + '=x2,coeffs=fit_coeffs_g_xsec)'
    feval_eff = 'func_CX_partial(' + cvartexts[nvar3] +'=0,' + cvartexts[nvar4] + '=0,'  + cvartexts[nvar1] + '=x1,' + cvartexts[nvar2] + '=x2,coeffs=fit_coeffs_g_eff)'
      
    func_fin = lambda x1, x2: eval(feval_xsec) * sig_factors * eval(feval_eff) * lumi * 1000. / math.sqrt(sigma_bkg * lumi + Systematics**2 * (sigma_bkg)**2 * lumi**2)

    pfunc_fin = lambda x1, x2: (1 - 2 * scipy.special.ndtr(-(eval(feval_xsec) * sig_factors * eval(feval_eff) * lumi * 1000. / math.sqrt(sigma_bkg * lumi + Systematics**2 * (sigma_bkg)**2 * lumi**2))))

    # SM significance: 
    feval_xsec_sm = 'func_CX_partial(' + cvartexts[0] +'=0,' + cvartexts[1] + '=0,' + cvartexts[2] + '=0,' + cvartexts[3] + '=0,coeffs=fit_coeffs_g_xsec)'
    feval_eff_sm = 'func_CX_partial(' + cvartexts[0] +'=0,' + cvartexts[1] + '=0,' + cvartexts[2] + '=0,' + cvartexts[3] + '=0,coeffs=fit_coeffs_g_eff)'
    
    sm_signif= eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * 1000. * lumi / math.sqrt(sigma_bkg * lumi + Systematics**2 * (sigma_bkg)**2 * lumi**2)

    # if SM is the "null" hypothesis:
    # SM number of events:
    S_SM = eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * 1000. * lumi
    # SM total uncertainty, including the background uncertainty:
    delta_SM =  math.sqrt(S_SM + sigma_bkg * lumi + Systematics**2 * (sigma_bkg)**2 * lumi**2) 
    # {c_i} number of events in 2D model:
    S_i_2D = lambda x1, x2: eval(feval_xsec) * sig_factors * eval(feval_eff) * lumi * 1000.
    # significance versus the SM in the 2D mode: 
    func_fin_SM_2D = lambda x1, x2: np.power( (S_SM - S_i_2D(x1, x2))/delta_SM,2)

    # p-value in the 2D model:
    #pfunc_fin_SM_2D = lambda x1, x2:  1/(np.sqrt(2.*np.pi)*delta_SM)*np.exp(-func_fin_SM_2D(x1, x2)/2)
    # print('pfunc_fin_SM_2D(0,0)=',pfunc_fin_SM_2D(0,0))
    
    print("sigma_sig before anal. [fb]=", eval(feval_xsec_sm)*1000*sig_factors)
    print("analysis eff. on signal=", eval(feval_eff_sm))
    print("sigma_bkg after anal. [fb]=", sigma_bkg)
    print("sigma sig SM after anal. [fb]=",eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * 1000)
    print("N(bkg)@lumi=", sigma_bkg * lumi)
    print("N(sig SM)@lumi=", eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * lumi * 1000.) 
    print("SM significance=", sm_signif)
    

    
    x = np.linspace(xmin, xmax, nbins)
    y = np.linspace(ymin, ymax, nbins)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x,y)
    #P = pfunc_fin_SM_2D(X,Y)  #func_fin(X,Y)
    chisq_prior = 0
    if deltac3 > 0:
        #P=P*gaussian(X, 0, deltac3) # REWEIGH BY C3 PRIOR if deltac3 > 0
        chisq_prior = (X - 0)**2 / deltac3**2
    chisq = func_fin_SM_2D(X,Y) + chisq_prior

    #P = P/pfunc_fin_SM_2D(0,0)
    #print(np.amax(P))
    # convert to chi-sq.:
    #chisq = stats.chi2.isf(P,2)
        
    chisq_sub = chisq - np.amin(chisq)
    print('np.amin(chisq)=',np.amin(chisq))

    # MARGINALIZATION ATTEMPTS BELOW
    # do the one-dimensional marginalizations:
    # sum over the marginalized direction
    #P_marg_nvar1 = np.apply_over_axes(np.sum, P, [1]) * dy
    #P_marg_nvar2 = np.apply_over_axes(np.sum, P, [0]) * dx
    # change the shape:
    #P_marg_nvar1_s = P_marg_nvar1.reshape(P_marg_nvar1.shape[0])
    #P_marg_nvar2_s = P_marg_nvar2.reshape(P_marg_nvar2.shape[1])
    #print('P_marg_nvar1_s=',P_marg_nvar1_s)
    #print('P_marg_nvar2_s=',P_marg_nvar2_s)
    # convert each probability to chisq:
    #chisq_marg_nvar1 = stats.chi2.isf(P_marg_nvar1_s,1)
    #chisq_marg_nvar2 = stats.chi2.isf(P_marg_nvar2_s,1)
    #print('chisq_marg_nvar1=', chisq_marg_nvar1)
    #print('chisq_marg_nvar2=', chisq_marg_nvar2)
    # remove infinities and nans:
    #chisq_marg_nvar1 = np.nan_to_num(chisq_marg_nvar1)
    #chisq_marg_nvar2 = np.nan_to_num(chisq_marg_nvar2)
    # subtract the minimum of chisq
    #chisq_marg_nvar1_sub = chisq_marg_nvar1 - np.amin(chisq_marg_nvar1)
    #chisq_marg_nvar2_sub = chisq_marg_nvar2 - np.amin(chisq_marg_nvar2)
    #print('chisq_marg_nvar1_sub=', chisq_marg_nvar1_sub)
    #print('chisq_marg_nvar2_sub=', chisq_marg_nvar2_sub)
    # remove infinities and nans:
    #chisq_marg_nvar1_sub = np.nan_to_num(chisq_marg_nvar1_sub)
    #chisq_marg_nvar2_sub = np.nan_to_num(chisq_marg_nvar2_sub)
    #print('chisq_marg_nvar1_sub=', chisq_marg_nvar1_sub)
    #print('chisq_marg_nvar2_sub=', chisq_marg_nvar2_sub)

    # profiling attempt:
    chisq_marg_nvar1_sub = np.min(chisq_sub, axis=1) 
    chisq_marg_nvar2_sub = np.min(chisq_sub, axis=0)

    x1 = np.linspace(xmin, xmax, nbins) # nvar1
    x2 = np.linspace(ymin, ymax, nbins) # nvar2
    
    # interpolate the 1D functions: 
    func_chisq_1D_nvar1 = interp1d(x1,chisq_marg_nvar1_sub, fill_value="extrapolate")
    func_chisq_1D_nvar2 = interp1d(x2,chisq_marg_nvar2_sub, fill_value="extrapolate")
    
    # construction functions to find 1 and 2 sigma limits on nvar1 and nvar2 (from chi-sq min). 
    def func_nvar1_1sigma(x): return (func_chisq_1D_nvar1(x) - 0.99)
    def func_nvar1_2sigma(x): return (func_chisq_1D_nvar1(x) - 3.84)
    def func_nvar2_1sigma(x): return (func_chisq_1D_nvar2(x) - 0.99)
    def func_nvar2_2sigma(x): return (func_chisq_1D_nvar2(x) - 3.84)
        
    # guesses for the locations of the solutions in 1D [change with energy]:
    nvar1_min_1 = {}
    nvar1_max_1 = {}
    nvar1_min_2 = {}
    nvar1_max_2 = {}

    nvar1_min_1[100] = searchlimits[Energy][nvar1][0]
    nvar1_max_1[100] = searchlimits[Energy][nvar1][1]
    nvar1_min_2[100] = searchlimits[Energy][nvar1][0]
    nvar1_max_2[100] = searchlimits[Energy][nvar1][1]

    nvar2_min_1 = {}
    nvar2_max_1 = {}
    nvar2_min_2 = {}
    nvar2_max_2 = {}
    
    nvar2_min_1[100] = searchlimits[Energy][nvar2][0]
    nvar2_max_1[100] = searchlimits[Energy][nvar2][1]
    nvar2_min_2[100] = searchlimits[Energy][nvar2][0]
    nvar2_max_2[100] = searchlimits[Energy][nvar2][1]

    CL_threshold = 3.84  # 95% CL
    allowed = np.where(chisq_marg_nvar1_sub <= CL_threshold)[0]
    x_limits = x1[allowed]
    x_lower, x_upper = x_limits[0], x_limits[-1]
    print(f"95% CL for c3: {x_lower:.3f} to {x_upper:.3f} (c3)")
    allowed = np.where(chisq_marg_nvar2_sub <= CL_threshold)[0]
    x_limits = x2[allowed]
    x_lower, x_upper = x_limits[0], x_limits[-1]
    print(f"95% CL for d4: {x_lower:.3f} to {x_upper:.3f} (d4)")
    
    
    # calculate and print out the solutions:
    #print(variables[nvar1] + '@68% CL:', fsolve(func_nvar1_1sigma, [nvar1_min_1[Energy], nvar1_max_1[Energy]]))
    print(variables[nvar1] + '@95% CL:', fsolve(func_nvar1_2sigma, [nvar1_min_2[Energy], nvar1_max_2[Energy]]))
    #print(variables[nvar2] + '@68% CL:', fsolve(func_nvar2_1sigma, [nvar2_min_1[Energy], nvar2_max_1[Energy]]))
    print(variables[nvar2] + '@95% CL:', fsolve(func_nvar2_2sigma, [nvar2_min_2[Energy], nvar2_max_2[Energy]]))

    # TEST FUNCTIONS TO SOLVE HERE:
    plt.clf()
    x2 = np.linspace(ymin, ymax, nbins) # nvar2
    y2 = func_nvar2_2sigma(x2)
    y1 = func_nvar2_1sigma(x2)
    plt.plot(x2, y1)
    plt.plot(x2, y2)
    plt.axhline(0, color='k', linestyle='--')
    plt.savefig(plot_dir + output + 'test_d4.pdf', bbox_inches='tight')
    plt.clf()
    x2 = np.linspace(xmin, xmax, nbins) # nvar2
    y2 = func_nvar1_2sigma(x2)
    y1 = func_nvar1_1sigma(x2)
    plt.plot(x2, y1)
    plt.plot(x2, y2)
    plt.axhline(0, color='k', linestyle='--')
    plt.savefig(plot_dir + output + 'test_c3.pdf', bbox_inches='tight')
    plt.clf()
    # END OF TEST FUNCTIONS TO SOLVE

    # construct the axes for the plot
    # no need to modify this if you just need one plot
    gs = gridspec.GridSpec(4, 4)
    if figext == '':
        fig = plt.figure()
    else:
        fig = figext
    if axext == '':
        ax = fig.add_subplot(111)
    else:
        ax=axext
    ax.grid(False)
    ax.set_title(plottitle, fontsize=10)
    
    # set the ticks, labels and limits etc.
    xlab = '$' + variables_latex[nvar1] + '$'
    ylab = '$' + variables_latex[nvar2] + '$'
    if setylabel == True:
        ax.set_ylabel(ylab, fontsize=labelsize)
    if setxlabel == True:
        ax.set_xlabel(xlab, fontsize=labelsize)
    # plot the contours:
    ax.plot(0,0,marker='*',ms=starsize, color='black')
    
    #cont = ax.contour(X, Y, chisq_sub, contours, extend='max', colors=('black', 'red'), linestyles=('-','--'))
    #labels = ['$1\\sigma$', '$2\\sigma$']
    #for i in range(len(labels)):
    #    cont.collections[i].set_label(labels[i])
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    cont = ax.contour(
    X, Y, chisq_sub, contours,
    extend='max',
    colors=('black', 'red'),
    linestyles=('-', '--')
        )
    labels = ['$1\\sigma$', '$2\\sigma$']

    # Create a dictionary mapping each level to its label
    label_dict = {level: label for level, label in zip(cont.levels, labels)}
    
    # Add the legend labels via ax.clabel with a formatter
    ax.clabel(cont, fmt=label_dict, manual=False)  # Remove manual=... for automatic labeling

    
    if smtext == True:
        ax.text(0.53, 0.40,"SM", transform=ax.transAxes)
    if normalbar == True:
        plt.colorbar(cont)
    #handles, labels = cs.legend_elements()

    # after you’ve done your contour call…
    black_line = mlines.Line2D([], [], color='black', linestyle='-',
                            label='$1\\sigma$')
    red_line   = mlines.Line2D([], [], color='red',   linestyle='--',
                           label='$2\\sigma$')

    ax.legend(handles=[black_line, red_line],
          loc="upper right", frameon=False, prop={'size':8})
        
    #ax.legend()
    #ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8}, handles=[cont, cont2])
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if savefig == True:
        # save the figure
        print('saving the figure')
        # save the figure in PDF format
        infile = output + '.dat'
        print('---')
        print('output in', infile.replace('.dat','.pdf'))
        plt.savefig(plot_dir + infile.replace('.dat','.pdf'), bbox_inches='tight')
        plt.close(fig)
        
    return cont, X, Y, chisq_sub


def contour_pvalue_only(procname, plotname, plottitle, fit_coeffs_xsec, fit_coeffs_eff, sigma_bkg, var1, var2, plotlimits, searchlimits, deltac3=-1, axext='', figext='', smtext=True, starsize=15, setxlabel=True, setylabel=True, nbins=400, savefig=True,variables=variables, variables_latex=variables_latex, labelsize=20, normalbar=True, contours=np.arange(0, 10, 0.5),norm_to_zeroth=True, lumi=Luminosity):
    output = procname + '_' + plotname + '_' + var1 + '_' + var2
    print('Plotting', output)
    nvar1 = [key for key, value in variables.items() if value == var1][0]
    nvar2 = [key for key, value in variables.items() if value == var2][0]
    nvar3 = [key for key, value in variables.items() if value != var1 and value != var2][0]
    nvar4 = [key for key, value in variables.items() if value != var1 and value != var2][1]

    #print(var1, var2)
    print('nvar1, nvar2=', nvar1, nvar2)
    #print(nvar3, nvar4)
   
        
    # set the limits on the x and y axes if required below:
    ymin = plotlimits[Energy][nvar2][0]
    ymax = plotlimits[Energy][nvar2][1]
    xmin = plotlimits[Energy][nvar1][0]
    xmax = plotlimits[Energy][nvar1][1]
  
    ctexts = []
    cvartexts = []
    for i in range(0, len(variables.keys())):
        cvartexts.append(variables[i])
    fstr = 'partial(func_t_CX, procname=Process)'
    global func_CX_partial
    func_CX_partial = eval(fstr)
    #print('func_CX_partial=', func_CX_partial)
    #print(fit_coeffs)
    global fit_coeffs_g_xsec
    fit_coeffs_g_xsec = fit_coeffs_xsec
    global fit_coeffs_g_eff
    fit_coeffs_g_eff = fit_coeffs_eff
    #print(cvartexts[0], cvartexts[1])

    # functions for xsec and significance:
    print(cvartexts)
    # 0     1      2      3     
    #['c3', 'ct2', 'ct3', 'd4']
    print(cvartexts[nvar3], cvartexts[nvar4], cvartexts[nvar1], cvartexts[nvar2])
    feval_xsec = 'func_CX_partial(' + cvartexts[nvar3] +'=0,' + cvartexts[nvar4] + '=0,' + cvartexts[nvar1] + '=x1,' + cvartexts[nvar2] + '=x2,coeffs=fit_coeffs_g_xsec)'
    feval_eff = 'func_CX_partial(' + cvartexts[nvar3] +'=0,' + cvartexts[nvar4] + '=0,'  + cvartexts[nvar1] + '=x1,' + cvartexts[nvar2] + '=x2,coeffs=fit_coeffs_g_eff)'
      
    func_fin = lambda x1, x2: eval(feval_xsec) * sig_factors * eval(feval_eff) * lumi * 1000. / math.sqrt(sigma_bkg * lumi + Systematics**2 * (sigma_bkg)**2 * lumi**2)

    pfunc_fin = lambda x1, x2: (1 - 2 * scipy.special.ndtr(-(eval(feval_xsec) * sig_factors * eval(feval_eff) * lumi * 1000. / math.sqrt(sigma_bkg * lumi + Systematics**2 * (sigma_bkg)**2 * lumi**2))))

    # SM significance: 
    feval_xsec_sm = 'func_CX_partial(' + cvartexts[0] +'=0,' + cvartexts[1] + '=0,' + cvartexts[2] + '=0,' + cvartexts[3] + '=0,coeffs=fit_coeffs_g_xsec)'
    feval_eff_sm = 'func_CX_partial(' + cvartexts[0] +'=0,' + cvartexts[1] + '=0,' + cvartexts[2] + '=0,' + cvartexts[3] + '=0,coeffs=fit_coeffs_g_eff)'
    
    sm_signif= eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * 1000. * lumi / math.sqrt(sigma_bkg * lumi + Systematics**2 * (sigma_bkg)**2 * lumi**2)

    # if SM is the "null" hypothesis:
    # SM number of events:
    S_SM = eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * 1000. * lumi
    # SM total uncertainty, including the background uncertainty:
    delta_SM =  math.sqrt(S_SM + sigma_bkg * lumi + Systematics**2 * (sigma_bkg)**2 * lumi**2) 
    # {c_i} number of events in 2D model:
    S_i_2D = lambda x1, x2: eval(feval_xsec) * sig_factors * eval(feval_eff) * lumi * 1000.
    # significance versus the SM in the 2D mode: 
    func_fin_SM_2D = lambda x1, x2: np.power( (S_SM - S_i_2D(x1, x2))/delta_SM,2)

    # p-value in the 2D model:
    #pfunc_fin_SM_2D = lambda x1, x2:  1/(np.sqrt(2.*np.pi)*delta_SM)*np.exp(-func_fin_SM_2D(x1, x2)/2)
    # print('pfunc_fin_SM_2D(0,0)=',pfunc_fin_SM_2D(0,0))
    
    print("sigma_sig before anal. [fb]=", eval(feval_xsec_sm)*1000*sig_factors)
    print("analysis eff. on signal=", eval(feval_eff_sm))
    print("sigma_bkg after anal. [fb]=", sigma_bkg)
    print("sigma sig SM after anal. [fb]=",eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * 1000)
    print("N(bkg)@lumi=", sigma_bkg * lumi)
    print("N(sig SM)@lumi=", eval(feval_xsec_sm) * sig_factors * eval(feval_eff_sm) * lumi * 1000.) 
    print("SM significance=", sm_signif)
    

    
    x = np.linspace(xmin, xmax, nbins)
    y = np.linspace(ymin, ymax, nbins)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x,y)

    # ---- New Bayesian & Frequentist interval calculation clearly added here ----
    chisq_prior = (X)**2 / deltac3**2 if deltac3 > 0 else 0.0
        
    # Combine chi-squared with prior
    chisq_total = func_fin_SM_2D(X,Y) + chisq_prior
    chisq_sub = chisq_total - np.amin(chisq_total)
    
    # Bayesian posterior
    posterior = np.exp(-chisq_total / 2)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    posterior /= np.sum(posterior) * dx * dy  # normalized posterior
    
    # Bayesian marginalization over each parameter:
    posterior_nvar1 = np.sum(posterior, axis=0) * dy  # marginalized over Y (nvar2)
    posterior_nvar2 = np.sum(posterior, axis=1) * dx  # marginalized over X (nvar1)
    
    cdf_nvar1 = np.cumsum(posterior_nvar1) * dx
    cdf_nvar2 = np.cumsum(posterior_nvar2) * dy
    
    cdf_nvar1 /= cdf_nvar1[-1]
    cdf_nvar2 /= cdf_nvar2[-1]
    
    # 95% Bayesian credible intervals (central 95%)
    nvar1_low95 = np.interp(0.025, cdf_nvar1, x)
    nvar1_high95 = np.interp(0.975, cdf_nvar1, x)
    
    nvar2_low95 = np.interp(0.025, cdf_nvar2, y)
    nvar2_high95 = np.interp(0.975, cdf_nvar2, y)
    
    print("Bayesian 95% credible interval for", variables[nvar1], f": {nvar1_low95:.3f} to {nvar1_high95:.3f}")
    print("Bayesian 95% credible interval for", variables[nvar2], f": {nvar2_low95:.3f} to {nvar2_high95:.3f}")

    # --- Frequentist profiling clearly included here ---
    chisq_profile_nvar1 = np.min(chisq_sub, axis=0)  # profile over nvar2 (y)
    chisq_profile_nvar2 = np.min(chisq_sub, axis=1)  # profile over nvar1 (x)
    
    # Frequentist 95% confidence intervals (Δχ²=3.84 for 1 parameter)
    allowed_nvar1 = x[chisq_profile_nvar1 <= 3.84]
    allowed_nvar2 = y[chisq_profile_nvar2 <= 3.84]

    freq_nvar1_low, freq_nvar1_high = allowed_nvar1[0], allowed_nvar1[-1]
    freq_nvar2_low, freq_nvar2_high = allowed_nvar2[0], allowed_nvar2[-1]

    print("Frequentist (profile) 95% CL for", variables[nvar1], f": {freq_nvar1_low:.3f} \t {freq_nvar1_high:.3f}")
    print("Frequentist (profile) 95% CL for", variables[nvar2], f": {freq_nvar2_low:.3f} \t {freq_nvar2_high:.3f}")
    
    # write frequentist results to files:
    filewrite_frequentist_c3 = ConstraintsDir + output + 'frequentist_c3.out'
    filewrite_frequentist_d4 = ConstraintsDir + output + 'frequentist_d4.out'
    with open(filewrite_frequentist_c3,'w') as f:
        f.write(str(f"{freq_nvar1_low:.3f} \t {freq_nvar1_high:.3f}"))
    with open(filewrite_frequentist_d4,'w') as f:
        f.write(str(f"{freq_nvar2_low:.3f} \t {freq_nvar2_high:.3f}"))
    
    # interpolate the 1D functions: 
    func_chisq_1D_nvar1 = interp1d(x,chisq_profile_nvar1, fill_value="extrapolate")
    func_chisq_1D_nvar2 = interp1d(y,chisq_profile_nvar2, fill_value="extrapolate")
    
    # construction functions to find 1 and 2 sigma limits on nvar1 and nvar2 (from chi-sq min). 
    def func_nvar1_1sigma(x): return (func_chisq_1D_nvar1(x) - 0.99)
    def func_nvar1_2sigma(x): return (func_chisq_1D_nvar1(x) - 3.84)
    def func_nvar2_1sigma(x): return (func_chisq_1D_nvar2(x) - 0.99)
    def func_nvar2_2sigma(x): return (func_chisq_1D_nvar2(x) - 3.84)
    
    # TEST FUNCTIONS TO SOLVE HERE:
    plt.clf()
    x2 = np.linspace(ymin, ymax, nbins) # nvar2
    y2 = func_nvar2_2sigma(y)
    y1 = func_nvar2_1sigma(y)
    plt.plot(y, y1)
    plt.plot(y, y2)
    plt.axhline(0, color='k', linestyle='--')
    plt.savefig(plot_dir + output + 'test_d4.pdf', bbox_inches='tight')
    plt.clf()
    x2 = np.linspace(xmin, xmax, nbins) # nvar2
    y2 = func_nvar1_2sigma(x)
    y1 = func_nvar1_1sigma(x)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.axhline(0, color='k', linestyle='--')
    plt.savefig(plot_dir + output + 'test_c3.pdf', bbox_inches='tight')
    plt.clf()
    # END OF TEST FUNCTIONS TO SOLVE

    # construct the axes for the plot
    # no need to modify this if you just need one plot
    gs = gridspec.GridSpec(4, 4)
    if figext == '':
        fig = plt.figure()
    else:
        fig = figext
    if axext == '':
        ax = fig.add_subplot(111)
    else:
        ax=axext
    ax.grid(False)
    ax.set_title(plottitle, fontsize=10)
    
    # set the ticks, labels and limits etc.
    xlab = '$' + variables_latex[nvar1] + '$'
    ylab = '$' + variables_latex[nvar2] + '$'
    if setylabel == True:
        ax.set_ylabel(ylab, fontsize=labelsize)
    if setxlabel == True:
        ax.set_xlabel(xlab, fontsize=labelsize)
    # plot the contours:
    ax.plot(0,0,marker='*',ms=starsize, color='black')
    
    #cont = ax.contour(X, Y, chisq_sub, contours, extend='max', colors=('black', 'red'), linestyles=('-','--'))
    #labels = ['$1\\sigma$', '$2\\sigma$']
    #for i in range(len(labels)):
    #    cont.collections[i].set_label(labels[i])
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    cont = ax.contour(
    X, Y, chisq_sub, contours,
    extend='max',
    colors=('black', 'red'),
    linestyles=('-', '--')
        )
    labels = ['$1\\sigma$', '$2\\sigma$']

    # Create a dictionary mapping each level to its label
    label_dict = {level: label for level, label in zip(cont.levels, labels)}
    
    # Add the legend labels via ax.clabel with a formatter
    ax.clabel(cont, fmt=label_dict, manual=False)  # Remove manual=... for automatic labeling

    
    if smtext == True:
        ax.text(0.53, 0.40,"SM", transform=ax.transAxes)
    if normalbar == True:
        plt.colorbar(cont)
    #handles, labels = cs.legend_elements()

    # after you’ve done your contour call…
    black_line = mlines.Line2D([], [], color='black', linestyle='-',
                            label='$1\\sigma$')
    red_line   = mlines.Line2D([], [], color='red',   linestyle='--',
                           label='$2\\sigma$')

    ax.legend(handles=[black_line, red_line],
          loc="upper right", frameon=False, prop={'size':8})
        
    #ax.legend()
    #ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8}, handles=[cont, cont2])
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if savefig == True:
        # save the figure
        print('saving the figure')
        # save the figure in PDF format
        infile = output + '.dat'
        print('---')
        print('output in', infile.replace('.dat','.pdf'))
        plt.savefig(plot_dir + infile.replace('.dat','.pdf'), bbox_inches='tight')
        plt.close(fig)
        
    return cont, X, Y, chisq_sub


# save the data:
def save_data(data, filename):
    with open(filename,'wb') as f:
        pickle.dump(data,f)

# load the data:
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(filename)
    return data


################################c#########################
# RUN THE CODE HERE                                     # 
#########################################################


############################################
# GENERATE MG5 LHE FILES FOR SIGNAL:       #
############################################

if MODEL != 'HEFT4C3D4' and MODEL != 'C3D4ONLY':
    # reduced set for Global HHH (100 TeV runs):
    # [c3, d4, ct2, ct3 ] -> 4 couplings
    couplings_min = [-5.0, -5.0, -0.1, -0.1]
    couplings_max = [5.0, 5.0, 0.1, 0.1]
    if MODEL == 'HEFT6':
        couplings_min = couplings_min + [-0.1]
        couplings_max = couplings_max + [0.1]
elif MODEL == 'HEFT4C3D4' or MODEL== 'C3D4ONLY':
    couplings_min = [-5.0, -5.0, 0.0, 0.0]
    couplings_max = [5.0, 5.0, 0.0, 0.0]

# generate random coupling arrays:
randseed = 999
CouplingsArray_R, CouplingsArrayF_R = gen_coupbdasarray_dim_rand_range(couplings_min, couplings_max, Nruns, randseed)

# additional set:
Nadditional=560
couplings_min = [-0.5, -0.5, -0.001, -0.001]
couplings_max = [0.5, 0.5, 0.001, 0.001]
if MODEL == 'HEFT6':
    couplings_min = couplings_min + [-0.1]
    couplings_max = couplings_max + [0.1]
CouplingsArray_R_add, CouplingsArrayF_R_add = gen_coupbdasarray_dim_rand_range(couplings_min, couplings_max, Nadditional, randseed+31)

# additional set 2:
Nadditional2=200
couplings_min = [-5.0, -50, 0, 0]
couplings_max = [5.0, 50, 0, 0]
if MODEL == 'HEFT6':
    couplings_min = couplings_min + [-0.1]
    couplings_max = couplings_max + [0.1]
CouplingsArray_R_add2, CouplingsArrayF_R_add2 = gen_coupbdasarray_dim_rand_range(couplings_min, couplings_max, Nadditional2, randseed+29)

# additional set 3:
Nadditional3=300
couplings_min = [-100.0, -100, 0, 0]
couplings_max = [100.0, 100, 0, 0]
if MODEL == 'HEFT6':
    couplings_min = couplings_min + [-0.1]
    couplings_max = couplings_max + [0.1]
CouplingsArray_R_add3, CouplingsArrayF_R_add3 = gen_coupbdasarray_dim_rand_range(couplings_min, couplings_max, Nadditional3, randseed+27)

# additional set 4:
Nadditional4=290
couplings_min = [-10.0, -100, -2, -2]
couplings_max = [10.0, 100, 2, 2]
if MODEL == 'HEFT6':
    couplings_min = couplings_min + [-2]
    couplings_max = couplings_max + [2]
CouplingsArray_R_add4, CouplingsArrayF_R_add4 = gen_coupbdasarray_dim_rand_range(couplings_min, couplings_max, Nadditional4, randseed+33)

# additional set 5:
Nadditional5=500
couplings_min = [-20.0, -100, -4, -4]
couplings_max = [20.0, 100, 4, 4]
if MODEL == 'HEFT6':
    couplings_min = couplings_min + [-5]
    couplings_max = couplings_max + [5]
CouplingsArray_R_add5, CouplingsArrayF_R_add5 = gen_coupbdasarray_dim_rand_range(couplings_min, couplings_max, Nadditional5, randseed+99)

# additional set 6:
Nadditional6=100
couplings_min = [-40.0, -100, 0, 0]
couplings_max = [40.0, 100, 0, 0]
if MODEL == 'HEFT6':
    couplings_min = couplings_min + [-0.1]
    couplings_max = couplings_max + [0.1]
CouplingsArray_R_add6, CouplingsArrayF_R_add6 = gen_coupbdasarray_dim_rand_range(couplings_min, couplings_max, Nadditional6, randseed+4)

# additional set 7:
#Nadditional7=0
#CouplingsArray_R_add7, CouplingsArrayF_R_add7 = [], []
Nadditional7=200
couplings_min = [-50.0, -800, -2, -2] 
couplings_max = [50.0, 800, 2, 2]
if MODEL == 'HEFT6':
    couplings_min = couplings_min + [-0.1]
    couplings_max = couplings_max + [0.1]
CouplingsArray_R_add7, CouplingsArrayF_R_add7 = gen_coupbdasarray_dim_rand_range(couplings_min, couplings_max, Nadditional7, randseed+3)

# additional set 8: 
#Nadditional8=0
#CouplingsArray_R_add8, CouplingsArrayF_R_add8 = [], []
Nadditional8=200
couplings_min = [-20.0, -600, -2, -2]
couplings_max = [20.0, 600, 2, 2]
if MODEL == 'HEFT6':
    couplings_min = couplings_min + [-1.0]
    couplings_max = couplings_max + [1.0]
CouplingsArray_R_add8, CouplingsArrayF_R_add8 = gen_coupbdasarray_dim_rand_range(couplings_min, couplings_max, Nadditional8, randseed+2)

# for testing: reset some to zero:
#CouplingsArray_R_add, CouplingsArrayF_R_add = [], []
#CouplingsArray_R_add2, CouplingsArrayF_R_add2 = [], []
#CouplingsArray_R_add3, CouplingsArrayF_R_add3 = [], []
#CouplingsArray_R_add4, CouplingsArrayF_R_add4 = [], []
#CouplingsArray_R_add5, CouplingsArrayF_R_add5 = [], []
#CouplingsArray_R_add6, CouplingsArrayF_R_add6 = [], []
#CouplingsArray_R_add7, CouplingsArrayF_R_add7 = [], []
#CouplingsArray_R_add8, CouplingsArrayF_R_add8 = [], []
#Nadditional, Nadditional2, Nadditional3, Nadditional4,
#Nadditional5, Nadditional6, Nadditional7, Nadditional8 = 0,0,0,0

# concatenate
CouplingsArray_R+=CouplingsArray_R_add+CouplingsArray_R_add2+CouplingsArray_R_add3+CouplingsArray_R_add4+CouplingsArray_R_add5+CouplingsArray_R_add6+CouplingsArray_R_add7+CouplingsArray_R_add8
CouplingsArrayF_R+=CouplingsArrayF_R_add+CouplingsArrayF_R_add2+CouplingsArrayF_R_add3+CouplingsArrayF_R_add4+CouplingsArrayF_R_add5+CouplingsArrayF_R_add6+CouplingsArrayF_R_add7+CouplingsArrayF_R_add8

# Launch MG5 event generation
nevents=1
drive_mg_proc(RunNum, MGLocation, ProcLocations[Process], Process, CouplingsArray_R, nevents, Nruns+Nadditional+Nadditional2+Nadditional3+Nadditional4+Nadditional5+Nadditional6+Nadditional7+Nadditional8, ecm=Energy)

###################################
# PERFORM THE FIT OR READ THE FIT #
###################################

# read the generated MG5 files:
if DoFit is True:
    print('reading in generated files')
    print('CouplingsArray_R=',CouplingsArray_R)
    X, Z, ZERR, XSEC = read_files(RunNum, MGLocation, ProcLocations[Process], Process, CouplingsArray_R, Nruns+Nadditional+Nadditional2+Nadditional3+Nadditional4+Nadditional5+Nadditional6+Nadditional7+Nadditional8)
    print(X)
else:
    print('Not reading in files, will read fit!')


# generate the list of initial guesses:
p0_i = []
p0_iE = []
for i in range(0,NCoeffs[Process]):
    p0_i.append(0.01)
    p0_iE.append(0.1)
# get the partial function with the process fixed:
func_CX_proc = partial(func_CX, procname=Process)
# perform the fit:

if DoFit is True:
    popt[Process], pcov[Process] = curve_fit(func_CX_proc, tuple(X) , Z, sigma=ZERR, method='lm', maxfev=2000, p0=p0_i)
    saveFit(popt[Process], pcov[Process], Process, RunNum)
    # test the fit:
    test_fit(RunNum, MGLocation, ProcLocations[Process], Process, CouplingsArray_R,  Nruns+Nadditional+Nadditional2+Nadditional3+Nadditional4+Nadditional5+Nadditional6+Nadditional7+Nadditional8, popt[Process])
else:
    popt[Process], pcov[Process] = readFit(Process, RunNum)

if debug:
    print('fitted parameters:')
    print(popt[Process])
    
if RunHerwig is False:
    print('Fit coefficients for MODEL=', MODEL, '=',  popt[Process]/popt[Process][-1])
    print('Errors=', np.sqrt(np.diag(pcov[Process]))/popt[Process][-1])
    print('RunHerwig is False: Not running Herwig or analysis, exiting')
    exit()

####################################
# RUN HERWIG ON LHE FILES          #
# AND PERFORM THE ANALYSIS         #
# FIT THE EFFICIENCY               # 
####################################


print('Running Herwig on generated MG5 LHEs')
run_herwig_proc_parallel(RunNum, MGLocation, HerwigOutputDirectory, ProcLocations[Process], Process, CouplingsArray_R, nevents,  Nruns+Nadditional+Nadditional2+Nadditional3+Nadditional4+Nadditional5+Nadditional6, ecm=Energy)

print('Running analysis on signal and background')
XE, ZE, EFFICIENCY, EFFICIENCY_BKG = run_analysis_proc(RunNum, MGLocation, HerwigOutputDirectory, ProcLocations[Process], Process, CouplingsArray_R, nevents,  Nruns+Nadditional+Nadditional2+Nadditional3+Nadditional4+Nadditional5+Nadditional6, ecm=Energy)
# to fix the issue of not reading it the first time ReRunAnalysis is True
#if ReRunAnalysis is True:
#    ReRunAnalysis = False
#    XE, ZE, EFFICIENCY, EFFICIENCY_BKG = run_analysis_proc(RunNum, MGLocation, HerwigOutputDirectory, ProcLocations[Process], Process, CouplingsArray_R, nevents,  Nruns+Nadditional+Nadditional2+Nadditional3+Nadditional4+Nadditional5+Nadditional6, ecm=Energy)

print(ZE)
popt_eff, pcov = curve_fit(func_CX_proc, tuple(XE), ZE, method='lm', maxfev=10000, p0=p0_iE)
test_fit_analysis(RunNum, MGLocation, ProcLocations[Process], Process, CouplingsArray_R,  Nruns+Nadditional+Nadditional2+Nadditional3+Nadditional4+Nadditional5+Nadditional6, popt_eff)

# get the SM efficiency:
analysisInputfile = './Herwig/events/HW-8_SM_6b.root' 
print('running the analysis', ExecutableSmear[Energy], 'on the input file', analysisInputfile)
analysiscommand = ExecutableSmear[Energy] + ' ' + analysisInputfile
print('Launching:', analysiscommand)
p = subprocess.Popen(analysiscommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
for line in iter(p.stdout.readline, b''):
    print('\t\t', line, end=' ')
out, err = p.communicate()
analysisOutputfile = analysisInputfile.replace('.root', '.smear' + smearing_tag + '.dat')
if os.path.exists(analysisOutputfile)is False:
    print('File', analysisOutputfile, 'does not exist!')
    exit()
else:
    print('File', analysisOutputfile, ' exists, reading results')
    zgrepcommand = 'cat ' + analysisOutputfile
    p = subprocess.Popen(zgrepcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
    for line in iter(p.stdout.readline, b''):
        SM_efficiency = float(line.split()[0])
        
# print the SM
print('SM RESULTS:')
print('Signal cross section BEFORE cuts (no b-tagging/BRs/k-factors)=', xsS)
print('Signal cross section BEFORE cuts (WITH b-tagging/BRs/k-factors)=', xsS*sig_factors)
print('Signal cut efficiency=', SM_efficiency)
sigma_SM_after_cuts = xsS * sig_factors * SM_efficiency
print('Signal cross section AFTRER cuts=', sigma_SM_after_cuts)
print('NSM(EVENTS)=', Luminosity*sigma_SM_after_cuts)

# Print the backgrounds
sigma_bkg = 0
print('Background cross sections BEFORE cuts:') 
for bkg in Backgrounds:
    print(bkg, 'sigma=', Backgrounds_xsec[(Energy, bkg)])
print('Background cut efficiency:') 
for bkg in Backgrounds:
    print(bkg, 'eff=', EFFICIENCY_BKG[bkg])
print('Background cross sections AFTER cuts:') 
for bkg in Backgrounds:
    sigma_bkg = sigma_bkg + EFFICIENCY_BKG[bkg] * Backgrounds_xsec[(Energy, bkg)]*bkg_factors
    print(bkg, 'sigma=', EFFICIENCY_BKG[bkg] * Backgrounds_xsec[(Energy, bkg)])
print('Background EXPECTED NUMBER OF EVENTS AFTER cuts:') 
for bkg in Backgrounds:
    print(bkg, 'N(EVENTS)=', bkg_factors*Luminosity*EFFICIENCY_BKG[bkg] * Backgrounds_xsec[(Energy, bkg)])
print('sigma_bkg total (fb) = ', sigma_bkg)

print("EXPECTED SM SIGNIFICANCE (CUTS)=", Luminosity*sigma_SM_after_cuts/np.sqrt(sigma_bkg*Luminosity + (Systematics*sigma_bkg*Luminosity)**2))

########################################
# XGBOOST ANALYSIS HERE
########################################
# do the training on the SM
training_seed = 12345
if DoTraining is True:
    trained_model = train_xgboost(signal_SM_file, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S_SM, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, training_seed)
    trained_model_file = 'trained_model' + str(RunNum) + smearing_tag + '.json'
    save_model(trained_model, trained_model_file)
else:
    trained_model_file = 'trained_model' + str(RunNum) + smearing_tag + '.json'
    trained_model = load_model(trained_model_file)
    
# apply the model on the SM (testing):
apply_xgboost(trained_model, signal_SM_file, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S_SM, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, training_seed)
time.sleep(10)
# apply to all points, get the efficiencies for signal and backgrounds
print('Running XGBOOST on all points')
XE, ZE, EFFICIENCY, EFFICIENCY_BKG = run_analysis_xgboost(RunNum, MGLocation, HerwigOutputDirectory, ProcLocations[Process], Process, CouplingsArray_R, nevents,  Nruns+Nadditional+Nadditional2+Nadditional3+Nadditional4+Nadditional5+Nadditional6, trained_model_file, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, training_seed, ecm=Energy)
popt_eff_XGBOOST, pcov_XGBOOST = curve_fit(func_CX_proc, tuple(XE), ZE, method='lm', maxfev=1000000, p0=p0_iE)
# calculate the background cross section after the xgboost analysis: 
sigma_bkg_xgboost = 0
for bkg in Backgrounds:
    sigma_bkg_xgboost = sigma_bkg_xgboost + EFFICIENCY_BKG[bkg] * Backgrounds_xsec[(Energy, bkg)]*bkg_factors

########################################
# PLOTTING STARTS HERE                 #
########################################
    
# Plot "correlation" plot of the cross section
if MODEL != 'C3D4ONLY' and MODEL !='HEFT3':
    correlation_plot(Process, 'xsec'+ str(Energy), popt[Process], variables, plottitle='$\\sigma(gg\\rightarrow hhh)$@' + str(Energy) + ' TeV, normalized to SM value')

# plot 1D plots of the variation of the cross section with coefficient
oned_xsec(Process, 'xsec' + str(Energy), r'$\sigma(gg\rightarrow hhh)$@' + str(Energy) + ' TeV, normalized to SM value', popt[Process], 'c3',[-5.0, 5.0], [0.5, 10])
oned_xsec(Process, 'xsec' + str(Energy), r'$\sigma(gg\rightarrow hhh)$@' + str(Energy) + ' TeV, normalized to SM value', popt[Process], 'd4',[-40.0, 40.0], [0.5, 10])

if MODEL != 'C3D4ONLY' and MODEL !='HEFT3':
    oned_xsec(Process, 'xsec' + str(Energy), r'$\sigma(gg\rightarrow hhh)$@' + str(Energy) + ' TeV, normalized to SM value', popt[Process], 'ct2',[-1.0, 1.0], [0.5, 25.0])
    oned_xsec(Process, 'xsec' + str(Energy), r'$\sigma(gg\rightarrow hhh)$@' + str(Energy) + ' TeV, normalized to SM value', popt[Process], 'ct3',[-1.0, 1.0], [0.5, 25.0])

# plot the "correlation plot" of the efficiency
if MODEL != 'C3D4ONLY' and MODEL !='HEFT3':
    correlation_plot(Process, 'eff'+ str(Energy), popt_eff, variables, plottitle='$\\epsilon(gg\\rightarrow hhh)$@' + str(Energy) + ' TeV', contours=np.arange(0.005, 0.02, 0.0005), norm_to_zeroth=False)

# for the XGBOOST case:
if MODEL != 'C3D4ONLY' and MODEL !='HEFT3':
    correlation_plot(Process, 'eff_XGBOOST'+ str(Energy), popt_eff_XGBOOST, variables, plottitle='$\\epsilon_\\mathrm{XG}(gg\\rightarrow hhh)$@' + str(Energy) + ' TeV', contours=np.arange(0.005, 0.02, 0.0005), norm_to_zeroth=False)


#########################################
# p-value contours and calculations
#########################################
nbinsdist=5000

# limits on the plots (exclusion)
plotlimits = {}
plotlimits[100] = {}
searchlimits = {}
searchlimits[100] = {}
if Systematics == 0.0:
    plotlimits[100][0] = [-5.0, 5.0]
    plotlimits[100][1] = [-1.0, 1.0]
    plotlimits[100][2] = [-0.5, 0.5]
    plotlimits[100][3] = [-30.0, 40.0]
    # search limits for the exclusion
    searchlimits[100][0] = [-8.0, 8.0]
    searchlimits[100][1] = [-1.0, 1.0]
    searchlimits[100][2] = [-0.5, 0.5]
    searchlimits[100][3] = [-30.0, 40.0]    
else:
    plotlimits[100][0] = [-10.0, 12.0]
    plotlimits[100][1] = [-1.0, 1.0]
    plotlimits[100][2] = [-0.5, 0.5]
    plotlimits[100][3] = [-180.0, 110.0]
    searchlimits[100][0] = [-10.0, 12.0]
    searchlimits[100][1] = [-1.0, 1.0]
    searchlimits[100][2] = [-0.5, 0.5]
    searchlimits[100][3] = [-180.0, 90.0]

if EnergyToRescale == 13 and DoRescaling is True:
    plotlimits[100][0] = [-20.0, 20.0]
    plotlimits[100][1] = [-1.0, 1.0]
    plotlimits[100][2] = [-0.5, 0.5]
    plotlimits[100][3] = [-100.0, 100.0]
    # search limits for the exclusion
    searchlimits[100][0] = [-20.0, 20.0]
    searchlimits[100][1] = [-1.0, 1.0]
    searchlimits[100][2] = [-0.5, 0.5]
    searchlimits[100][3] = [-100.0, 100.0]

if Luminosity < 1000:
    plotlimits[100][0] = [-10.0, 10.0]
    plotlimits[100][1] = [-1.0, 1.0]
    plotlimits[100][2] = [-0.5, 0.5]
    plotlimits[100][3] = [-60.0, 60.0]
    # search limits for the exclusion
    searchlimits[100][0] = [-8.0, 10.0]
    searchlimits[100][1] = [-1.0, 1.0]
    searchlimits[100][2] = [-0.5, 0.5]
    searchlimits[100][3] = [-60.0, 60.0]    
    

#contour_pvalue_ct3d4_marginalized(Process, 'pvalue'+ str(Energy) + '_L' + str(Luminosity), '$gg\\rightarrow hhh$@' + str(Energy) + ' TeV, L=' + str(Luminosity) + ' fb$^{-1}$, $\\alpha_\\mathrm{syst.} = ' + str(100*Systematics) +  '\%$', popt[Process], popt_eff, sigma_bkg, 'ct3', 'd4', plotlimits[Energy][2], plotlimits[Energy][3], contours=[2.278868566376729, 5.99], nbins=nbinsdist, normalbar=False)


# the tag for the output PDFs:
fulltag = str(Energy) + '_L' + str(Luminosity) + '_Syst' + str(Systematics) + '_pb' + str(btagging) + smearing_tag + '_' + MODEL + RESCALETAG + KFACTAG

# ct3, d4 (all others zero)
if MODEL != 'C3D4ONLY' and MODEL !='HEFT3':
    cont, X, Y, chisq_sub = contour_pvalue_only(Process, 'pvalue'+ fulltag, '$gg\\rightarrow hhh$@' + str(Energy) + ' TeV, L=' + str(Luminosity) + ' fb$^{-1}$, $\\mathcal{P}(b \\rightarrow b ) =' + str(btagging) + ' $' + ', $\\alpha_\\mathrm{syst.} = ' + str(100*Systematics) +  '\%$', popt[Process], popt_eff, sigma_bkg, 'ct3', 'd4', plotlimits, searchlimits,contours=[onesigma, twosigma], nbins=nbinsdist, normalbar=False)
    save_data([cont, X, Y, chisq_sub], ResultsDir + 'contourdata'+ fulltag + '_ct3_d4.pkl')
    

# c3, d4 (all others zero) 
cont, X, Y, chisq_sub = contour_pvalue_only(Process, 'pvalue'+ fulltag, '$gg\\rightarrow hhh$@' + str(Energy) + ' TeV, L=' + str(Luminosity) + ' fb$^{-1}$, $\\mathcal{P}(b \\rightarrow b ) =' + str(btagging) + ' $' + ', $\\alpha_\\mathrm{syst.} = ' + str(100*Systematics) +  '\%$', popt[Process], popt_eff, sigma_bkg, 'c3', 'd4', plotlimits, searchlimits,contours=[onesigma, twosigma], nbins=nbinsdist, normalbar=False)
save_data([cont, X, Y, chisq_sub], ResultsDir + 'contourdata'+ fulltag + '_c3_d4.pkl')


# XGBOOST:
#searchlimits[100][0] = [-3.0, 4.0]
#searchlimits[100][1] = [-1.0, 1.0]
#searchlimits[100][2] = [-0.5, 0.5]
#ssearchlimits[100][3] = [-10.0, 21.0]
# ct3, d4 (all others zero)
if MODEL != 'C3D4ONLY' and MODEL !='HEFT3':
    cont, X, Y, chisq_sub = contour_pvalue_only(Process, 'pvalueXGBOOST' + fulltag, '$gg\\rightarrow hhh$@' + str(Energy) + ' TeV, L=' + str(Luminosity) + ' fb$^{-1}$, $\\mathcal{P}(b \\rightarrow b ) =' + str(btagging) + ' $' + ', $\\alpha_\\mathrm{syst.} = ' + str(100*Systematics) +  '\%$', popt[Process], popt_eff_XGBOOST, sigma_bkg_xgboost, 'ct3', 'd4', plotlimits, searchlimits,contours=[onesigma, twosigma], nbins=nbinsdist, normalbar=False)
    save_data([cont, X, Y, chisq_sub], ResultsDir + 'contourdataXGBOOST'+ fulltag + '_ct3_d4.pkl')


# c3, d4 (all others zero) 
cont, X, Y, chisq_sub = contour_pvalue_only(Process, 'pvalueXGBOOST'+ fulltag, '$gg\\rightarrow hhh$@' + str(Energy) + ' TeV, L=' + str(Luminosity) + ' fb$^{-1}$, $\\mathcal{P}(b \\rightarrow b ) =' + str(btagging) + ' $' + ', $\\alpha_\\mathrm{syst.} = ' + str(100*Systematics) +  '\%$', popt[Process], popt_eff_XGBOOST, sigma_bkg_xgboost, 'c3', 'd4', plotlimits, searchlimits, contours=[onesigma, twosigma], nbins=nbinsdist, normalbar=False)
save_data([cont, X, Y, chisq_sub], ResultsDir + 'contourdataXGBOOST'+ fulltag + '_c3_d4.pkl')

########################
# MARGINALIZE OVER C3
########################
# c3, d4 (all others zero)
deltac3 = 0.05

plotlimits[100][0] = [-2.0, 2.0]
plotlimits[100][1] = [-1.0, 1.0]
plotlimits[100][2] = [-0.5, 0.5]
plotlimits[100][3] = [-60.0, 80.0]
searchlimits[100][0] = [-0.3, 0.3]
searchlimits[100][1] = [-1.0, 1.0]
searchlimits[100][2] = [-0.5, 0.5]
searchlimits[100][3] = [-8.0, 8.0]

# CUTS
cont, X, Y, chisq_sub = contour_pvalue_only(Process, 'pvalue_deltac3' + str(deltac3) + '_' + fulltag, '$gg\\rightarrow hhh$@' + str(Energy) + ' TeV, L=' + str(Luminosity) + ' fb$^{-1}$, $\\mathcal{P}(b \\rightarrow b ) =' + str(btagging) + ' $' + ', $\\alpha_\\mathrm{syst.} = ' + str(100*Systematics) +  '\%$', popt[Process], popt_eff, sigma_bkg, 'c3', 'd4', plotlimits, searchlimits,contours=[onesigma, twosigma], nbins=nbinsdist, normalbar=False, deltac3=deltac3)
save_data([cont, X, Y, chisq_sub], ResultsDir + 'contourdata_deltac3' + str(deltac3) + fulltag + '_c3_d4.pkl')

# XGBOOST 
cont, X, Y, chisq_sub = contour_pvalue_only(Process, 'pvalueXGBOOST_deltac3' + str(deltac3) + '_' + fulltag, '$gg\\rightarrow hhh$@' + str(Energy) + ' TeV, L=' + str(Luminosity) + ' fb$^{-1}$, $\\mathcal{P}(b \\rightarrow b ) =' + str(btagging) + ' $' + ', $\\alpha_\\mathrm{syst.} = ' + str(100*Systematics) +  '\%$', popt[Process], popt_eff_XGBOOST, sigma_bkg_xgboost, 'c3', 'd4', plotlimits, searchlimits, contours=[onesigma, twosigma], nbins=nbinsdist, normalbar=False, deltac3=deltac3)
save_data([cont, X, Y, chisq_sub], ResultsDir + 'contourdataXGBOOST_deltac3_' + str(deltac3) + fulltag + '_c3_d4.pkl')

####################################
# PRINT COEFFICIENTS OF XSEC FIT:
####################################

print('Fit coefficients for MODEL=', MODEL, '=',  popt[Process])
