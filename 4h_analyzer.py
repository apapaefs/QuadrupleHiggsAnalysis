#!/usr/bin/env python3

from pathlib import Path as _Path
import hashlib as _hashlib
import re as _re
import sys as _sys

_REPO_DIR = _Path(__file__).resolve().parent
_CODE_DIR = _REPO_DIR / "Code"
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

DEFAULT_HBB_BRANCHING_RATIO = 0.5824
DEFAULT_BTAGGING_RATE = 0.85
DEFAULT_SIGNAL_HBB_POWER = 4
DEFAULT_EIGHT_BTAG_POWER = 8
DEFAULT_SIGNAL_K_FACTOR = 2.0
DEFAULT_BACKGROUND_K_FACTOR = 2.0
DEFAULT_BACKGROUND_CSV = _REPO_DIR / "Backgrounds" / "processes.csv"
DEFAULT_BACKGROUND_HERWIG_TEMPLATE = _REPO_DIR / "Backgrounds" / "HW-AlpGen8Q-LHEWriter-Reweighted.in"


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
    sample_name = root_file.name.split("_var.smear", 1)[0]
    out_file = root_file.parent.parent / f"{sample_name}.out"
    xsec_fb, generated = _parse_herwig_total_xsec(out_file)
    return xsec_fb, generated, out_file


def _discover_var_root_files(sample_dir, include_auxiliary=False):
    files = sorted((_REPO_DIR / sample_dir / "events").glob("*_var.smear*.root"))
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
        if not parent.name.startswith("run_gg_4h_"):
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


def _discover_score_roots(paths):
    roots = []
    for path in paths:
        path = _Path(path)
        if path.is_file():
            roots.append(path)
        elif path.is_dir():
            roots.extend(sorted(path.rglob("*_var.smear*.root")))
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


def _analysis_output_root(raw_root):
    raw_root = _Path(raw_root)
    return raw_root.with_name(raw_root.name.replace(".root", "_var.smearCMS.root"))


def _analysis_log_file(raw_root):
    raw_root = _Path(raw_root)
    return raw_root.with_name(raw_root.name.replace(".root", ".analysis.log"))


def _analysis_summary_file(raw_root):
    raw_root = _Path(raw_root)
    return raw_root.with_name(raw_root.name.replace(".root", ".analysis_summary.json"))


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


def _discover_analysis_inputs(paths):
    var_roots = []
    raw_roots = []
    for path in paths:
        path = _Path(path)
        if path.is_file():
            if _is_var_smear_root(path):
                var_roots.append(path)
            elif path.suffix == ".root":
                raw_roots.append(path)
            else:
                print(f"Warning: analysis input is not a ROOT file: {path}")
        elif path.is_dir():
            var_roots.extend(sorted(path.rglob("*_var.smear*.root")))
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
    return xsec_fb, generated, None


def _infer_scored_signal_metadata(files, xsec_values, generated_values, default_generated_events, label):
    normalisation_weights = [_normalisation_weight_for_var_root(path) for path in files]
    if xsec_values is None or generated_values is None:
        inferred_xsecs = []
        inferred_generated = []
        for path in files:
            xsec_fb, generated_events, source_file = _metadata_for_scored_signal_root(path, default_generated_events)
            if xsec_fb is None:
                source_text = f" from {source_file}" if source_file is not None else ""
                print(f"Warning: could not infer {label} cross section{source_text}; using 1 fb for {path}")
                xsec_fb = 1.0
            inferred_xsecs.append(xsec_fb)
            inferred_generated.append(generated_events)

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


def _ensure_background_csv_var_roots(samples, args):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    var_roots = []
    missing_jobs = []
    for sample in samples:
        var_root = sample["var_root"]
        raw_root = sample["raw_root"]
        if var_root.exists() and not args.force_analysis:
            var_roots.append(var_root)
            continue
        if raw_root.exists():
            missing_jobs.append(sample)
            continue
        print(f"Warning: missing background ROOT for {sample['process_id']}: {raw_root}")

    if missing_jobs and args.no_run_missing_analysis:
        print(f"Warning: {len(missing_jobs)} CSV background raw ROOT file(s) need analysis, but auto-analysis is disabled.")
    elif missing_jobs:
        executable = _ensure_analysis_executable(args.analysis_exe, args.analysis_source, rebuild=True)
        print(f"Running C++ analysis for {len(missing_jobs)} CSV background ROOT file(s)")
        jobs = max(1, int(args.analysis_jobs))

        def run_sample(sample):
            return _run_one_cpp_analysis(
                sample["raw_root"],
                executable,
                max_events=args.analysis_max_events,
                force=args.force_analysis,
                c_mistags=sample["c_quarks"],
                light_mistags=sample["light_jets"],
            )

        if jobs == 1:
            for sample in missing_jobs:
                run_sample(sample)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(run_sample, sample) for sample in missing_jobs]
                for future in as_completed(futures):
                    future.result()

    for sample in samples:
        if sample["var_root"].exists():
            var_roots.append(sample["var_root"])

    return _unique_paths(_filter_auxiliary_roots(var_roots, args.include_auxiliary_samples))


def _background_inputs_from_csv(args, ensure_analysis=False):
    samples = _read_background_processes(args.background_csv)
    if ensure_analysis:
        background_files = _ensure_background_csv_var_roots(samples, args)
    else:
        background_files = [
            sample["var_root"]
            for sample in samples
            if sample["var_root"].exists()
        ]
        missing = [sample for sample in samples if not sample["var_root"].exists()]
        for sample in missing:
            print(f"Warning: missing background variable ROOT for {sample['process_id']}: {sample['var_root']}")

    by_var_root = {str(sample["var_root"]): sample for sample in samples}
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
                sample["var_root"].name: sample
                for sample in _read_background_processes(csv_file)
            }
        except (SystemExit, ValueError):
            samples = {}

    metadata = []
    for path in background_files:
        sample = samples.get(_Path(path).name)
        if sample is not None:
            metadata.append(_background_metadata_from_sample(sample))
        else:
            metadata.append({})
    return metadata


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
    return {
        "entries": entries,
        "selected_entries": selected_entries,
        "expected_preselected_events": preselected_events,
        "expected_selected_events": selected_events,
        "expected_selected_error": selected_error,
        "analysis_efficiency": analysis_efficiency,
        "xgboost_efficiency": xgboost_efficiency,
        "final_efficiency": final_efficiency,
    }


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
        f"{sm_summary['expected_selected_events']} +/- {sm_summary['expected_selected_error']}"
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
        f"{background_summary['expected_selected_events']} +/- {background_summary['expected_selected_error']}"
    )
    print(f"  Background analysis efficiency = {background_summary['analysis_efficiency']}")
    print(f"  Background XGBoost efficiency = {background_summary['xgboost_efficiency']}")
    print(f"  Background final efficiency = {background_summary['final_efficiency']}")


def _physics_rate_factor(hbb_branching_ratio, hbb_power, btagging_rate, btag_power):
    return float(hbb_branching_ratio) ** int(hbb_power) * float(btagging_rate) ** int(btag_power)


def _background_rate_factor_from_metadata(metadata, btagging_rate, c_mistag_rate, light_mistag_rate, k_factor):
    return (
        float(k_factor)
        * float(btagging_rate) ** int(metadata.get("b_quarks", 0))
        * float(c_mistag_rate) ** int(metadata.get("c_quarks", 0))
        * float(light_mistag_rate) ** int(metadata.get("light_jets", 0))
    )


def _background_rate_factors_for_cli(background_metadata, args):
    if background_metadata and all(metadata.get("process_id") for metadata in background_metadata):
        return [
            _background_rate_factor_from_metadata(
                metadata,
                args.btagging_rate,
                args.c_mistag_rate,
                args.light_mistag_rate,
                args.background_k_factor,
            )
            for metadata in background_metadata
        ]

    return _physics_rate_factor(
        args.hbb_branching_ratio,
        args.background_hbb_power,
        args.btagging_rate,
        args.background_btag_power,
    ) * float(args.background_k_factor)


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
        needs_build = source_file.stat().st_mtime > executable.stat().st_mtime

    if rebuild and needs_build and source_file is not None and (source_file.parent / "Makefile").exists():
        print("Building analysis executable:", executable)
        subprocess.run(["make", "-C", str(source_file.parent), executable.name], check=True)

    if not executable.exists():
        raise SystemExit(f"Analysis executable does not exist: {executable}")
    return executable


def _run_one_cpp_analysis(raw_root, executable, max_events=None, force=False, c_mistags=0, light_mistags=0):
    import subprocess

    raw_root = _Path(raw_root)
    output_root = _analysis_output_root(raw_root)
    log_file = _analysis_log_file(raw_root)
    if output_root.exists() and not force:
        return output_root

    command = [str(executable), str(raw_root)]
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
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    existing_var_roots, raw_roots = _discover_analysis_inputs(inputs)
    existing_var_roots = _filter_auxiliary_roots(existing_var_roots, include_auxiliary)
    raw_roots = _filter_auxiliary_roots(raw_roots, include_auxiliary)

    expected_var_roots = [_analysis_output_root(path) for path in raw_roots]
    missing_raw_roots = [
        raw_root
        for raw_root, var_root in zip(raw_roots, expected_var_roots)
        if force or not var_root.exists()
    ]

    if missing_raw_roots and not run_missing:
        print(f"Warning: {len(missing_raw_roots)} raw ROOT files are missing variable outputs, but auto-analysis is disabled.")
    elif missing_raw_roots:
        executable = _ensure_analysis_executable(executable, source_file, rebuild=True)
        print(f"Running C++ analysis for {len(missing_raw_roots)} missing variable ROOT file(s)")
        jobs = max(1, int(jobs))
        if jobs == 1:
            for raw_root in missing_raw_roots:
                _run_one_cpp_analysis(
                    raw_root,
                    executable,
                    max_events=max_events,
                    force=force,
                    c_mistags=c_mistags,
                    light_mistags=light_mistags,
                )
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = [
                    executor.submit(_run_one_cpp_analysis, raw_root, executable, max_events, force, c_mistags, light_mistags)
                    for raw_root in missing_raw_roots
                ]
                for future in as_completed(futures):
                    future.result()

    final_var_roots = list(existing_var_roots)
    for var_root in expected_var_roots:
        if var_root.exists():
            final_var_roots.append(var_root)
    return _unique_paths(_filter_auxiliary_roots(final_var_roots, include_auxiliary))


def _run_local_xgboost_cli():
    import argparse
    import json

    from xgboost_root_varfiles_module import (
        run_signal_background_analysis,
        score_background_files,
        score_signal_files,
        write_c3d4_limit_scan,
    )

    parser = argparse.ArgumentParser(
        description="Train a 4H XGBoost signal-vs-background classifier from local ROOT variable files."
    )
    parser.add_argument("--signal", action="append", type=_Path, help="Signal Data2 ROOT file. May be repeated.")
    parser.add_argument("--background", action="append", type=_Path, help="Background Data2 ROOT file. May be repeated.")
    parser.add_argument("--include-auxiliary-samples", action="store_true", help="Include debug/smoke ROOT files in discovery.")
    parser.add_argument("--outdir", type=_Path, default=_REPO_DIR / "xgboost_results", help="Directory for model and plots.")
    parser.add_argument("--luminosity", type=float, default=3000.0, help="Integrated luminosity in fb^-1.")
    parser.add_argument("--systematics", type=float, default=0.0, help="Fractional background systematic for threshold scan.")
    parser.add_argument("--hbb-branching-ratio", type=float, default=DEFAULT_HBB_BRANCHING_RATIO, help="Higgs to b bbar branching ratio used in the c3/d4 limit scan.")
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
    parser.add_argument("--c3d4-signal-xsec-fb", action="append", type=float, help="Cross section in fb for c3/d4 scan files.")
    parser.add_argument("--c3d4-signal-generated-events", action="append", type=int, help="Generated event counts for c3/d4 scan files.")
    parser.add_argument(
        "--c3d4-default-generated-events",
        type=int,
        default=10000,
        help="Fallback generated-event count for c3/d4 scan files.",
    )
    parser.add_argument("--no-c3d4-chebyshev-fit", action="store_true", help="Disable the Chebyshev-Lobatto sigma*eff fit and plot only scored points.")
    parser.add_argument("--c3d4-fit-k3-min", type=float, default=-29.0, help="Minimum k3=1+c3 used to scale the Chebyshev fit.")
    parser.add_argument("--c3d4-fit-k3-max", type=float, default=31.0, help="Maximum k3=1+c3 used to scale the Chebyshev fit.")
    parser.add_argument("--c3d4-fit-k4-min", type=float, default=-699.0, help="Minimum k4=1+d4 used to scale the Chebyshev fit.")
    parser.add_argument("--c3d4-fit-k4-max", type=float, default=701.0, help="Maximum k4=1+d4 used to scale the Chebyshev fit.")
    parser.add_argument("--c3d4-plot-c3-min", type=float, default=-30.0, help="Minimum c3 shown in the fitted limit plot.")
    parser.add_argument("--c3d4-plot-c3-max", type=float, default=30.0, help="Maximum c3 shown in the fitted limit plot.")
    parser.add_argument("--c3d4-plot-d4-min", type=float, default=-700.0, help="Minimum d4 shown in the fitted limit plot.")
    parser.add_argument("--c3d4-plot-d4-max", type=float, default=700.0, help="Maximum d4 shown in the fitted limit plot.")
    parser.add_argument("--c3d4-plot-nbins", type=int, default=301, help="Number of bins per axis for fitted c3/d4 plots.")
    parser.add_argument(
        "--c3d4-xsec-source-dir",
        type=_Path,
        default=_Path("/mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_4h_c3d4"),
        help="MG5 gg_4h_c3d4 directory used for the hhhh cross-section plot with the 95%% CL overlay.",
    )
    parser.add_argument(
        "--no-c3d4-xsec-overlay",
        action="store_true",
        help="Do not write the hhhh cross-section plot with the 95%% CL contour overlay.",
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

        signal_decay_btag_rate_factor = _physics_rate_factor(
            args.hbb_branching_ratio,
            args.signal_hbb_power,
            args.btagging_rate,
            args.signal_btag_power,
        )
        signal_rate_factor = signal_decay_btag_rate_factor * float(args.signal_k_factor)
        background_rate_factor = _background_rate_factors_for_cli(background_metadata, args)
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
            "signal_k_factor": float(args.signal_k_factor),
            "background_k_factor": float(args.background_k_factor),
            "signal_decay_btag_rate_factor": signal_decay_btag_rate_factor,
            "signal_rate_factor": signal_rate_factor,
            "background_rate_factor": background_rate_factor,
        }
        print(f"Using luminosity {args.luminosity:g} fb^-1")
        print(f"K-factors: signal = {args.signal_k_factor:g}, background = {args.background_k_factor:g}")
        print(
            f"Signal rate factor = {signal_rate_factor:g} "
            f"(K_signal * BR_hbb^{args.signal_hbb_power} * btag^{args.signal_btag_power})"
        )
        if isinstance(background_rate_factor, list):
            print(
                "Background rate factors use "
                "K_background * btag^b * c_mistag^c * light_mistag^j per CSV process"
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
            background_metadata=background_metadata,
            luminosity=args.luminosity,
            test_size=args.test_size,
            seed=args.seed,
            systematics=args.systematics,
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
            luminosity=args.luminosity,
            max_events=args.max_events,
        )
        _print_xgboost_threshold_summary(
            threshold,
            sm_signal_scores,
            background_scores["backgrounds"],
            args.luminosity,
        )
        background_events = background_scores["metadata"]["expected_selected_events_total"]
        if background_events <= 0.0:
            background_events = metrics["expected_preselected_background_events"] * best_threshold["background_efficiency"]
            print("Warning: full-sample background score is zero; using training-metric background estimate.")

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
        write_c3d4_limit_scan(
            scored_rows,
            output_dir=args.c3d4_scan_outdir,
            background_events=background_events,
            threshold=threshold,
            luminosity=args.luminosity,
            cl_target=args.c3d4_cl_target,
            poisson_confidence_level=args.poisson_cl,
            poisson_method=args.poisson_limit_method,
            poisson_observed_events=args.poisson_observed_events,
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
            rate_metadata=rate_metadata,
        )
        _print_sm_background_mc_counts(metrics)
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
        )

        print(f"Signal K-factor = {args.signal_k_factor:g}")
        score_signal_files(
            signal_files=score_files,
            model_file=args.model_file,
            output_dir=args.score_outdir,
            threshold=threshold,
            signal_xsecs_fb=signal_xsecs,
            signal_rate_factors=float(args.signal_k_factor),
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

    signal_rate_factor = float(args.signal_k_factor)
    background_rate_factor = _background_rate_factors_for_cli(background_metadata, args)
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

    run_signal_background_analysis(
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
        background_metadata=background_metadata,
        luminosity=args.luminosity,
        test_size=args.test_size,
        seed=args.seed,
        systematics=args.systematics,
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
