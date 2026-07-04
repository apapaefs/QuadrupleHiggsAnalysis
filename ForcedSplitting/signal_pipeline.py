"""Prepare forced-splitting Herwig cards from MG5 signal-point runs."""

import argparse
import csv
import hashlib
import re
from pathlib import Path

from .herwig_cards import PROCESS_CONFIGS, stage1_lhewriter_card, stage2_hwsim_card


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_C3_RE = re.compile(r"^\s*4\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+#\s*c3\b", re.MULTILINE)
_D4_RE = re.compile(r"^\s*6\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+#\s*d4\b", re.MULTILINE)


def _stable_seed(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 900000000 + 1


def _format_float(value):
    if value is None:
        return ""
    return str(float(value))


def _point_key(c3, d4):
    if c3 in (None, "") or d4 in (None, ""):
        return None
    return (round(float(c3), 9), round(float(d4), 9))


def _parse_c3d4_from_run_name(name):
    numeric_tokens = [token for token in name.split("_") if _NUMBER_RE.match(token)]
    if len(numeric_tokens) < 2:
        return None, None
    return float(numeric_tokens[-2]), float(numeric_tokens[-1])


def _parse_c3d4_from_text(text):
    c3_match = _C3_RE.search(text)
    d4_match = _D4_RE.search(text)
    if not c3_match or not d4_match:
        return None, None
    return float(c3_match.group(1)), float(d4_match.group(1))


def _parse_c3d4_from_cards(run_dir):
    candidates = [
        run_dir / "param_card.dat",
        run_dir / "Cards" / "param_card.dat",
    ]
    candidates.extend(sorted(run_dir.glob("*_banner.txt")))
    for candidate in candidates:
        if not candidate.exists():
            continue
        c3, d4 = _parse_c3d4_from_text(candidate.read_text(errors="replace"))
        if c3 is not None and d4 is not None:
            return c3, d4
    return None, None


def parse_c3d4(run_dir):
    c3, d4 = _parse_c3d4_from_run_name(run_dir.name)
    if c3 is not None and d4 is not None:
        return c3, d4
    return _parse_c3d4_from_cards(run_dir)


def _find_lhe(run_dir):
    for name in ("unweighted_events.lhe.gz", "unweighted_events.lhe"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return run_dir / "unweighted_events.lhe.gz"


def load_reference_grid(manifest_file):
    if manifest_file is None:
        return None
    points = set()
    with Path(manifest_file).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = _point_key(row.get("c3"), row.get("d4"))
            if key is not None:
                points.add(key)
    return points


def _run_dirs(mg5_dir):
    events_dir = Path(mg5_dir) / "Events"
    return sorted(path for path in events_dir.glob("run*") if path.is_dir())


def _write_input_list(path, inputs):
    path.write_text("".join("%s\n" % item for item in inputs))


def prepare_forced_splitting_inputs(
    process,
    mg5_dir,
    output_dir,
    events,
    manifest_file=None,
    output_location="events/",
    run_prefix="FS",
    probe_trials=0,
    reference_grid_manifest=None,
    overwrite=False,
):
    process_config = PROCESS_CONFIGS[process]
    mg5_dir = Path(mg5_dir)
    output_dir = Path(output_dir)
    if manifest_file is None:
        manifest_file = output_dir / "forced_splitting_manifest.csv"
    else:
        manifest_file = Path(manifest_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / output_location).mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    reference_grid = load_reference_grid(reference_grid_manifest)
    rows = []
    stage1_inputs = []
    stage1_outputs = []
    stage2_inputs = []
    stage1_reweight_inputs = []

    for run_dir in _run_dirs(mg5_dir):
        c3, d4 = parse_c3d4(run_dir)
        point_key = _point_key(c3, d4)
        lhe_file = _find_lhe(run_dir)
        stage1_run_name = "%s1-%s" % (run_prefix, run_dir.name)
        stage2_run_name = "%s2-%s" % (run_prefix, run_dir.name)
        stage1_input = output_dir / ("%s.in" % stage1_run_name)
        stage2_input = output_dir / ("%s.in" % stage2_run_name)
        stage1_output_lhe = "%s.lhe" % stage1_run_name
        stage1_weighted_lhe = "%s.weighted.lhe" % stage1_run_name
        stage1_correction = "%s.force_split.weights" % stage1_run_name
        stage2_lhe_file = stage1_weighted_lhe if probe_trials > 0 else stage1_output_lhe
        seed_stage1 = _stable_seed(stage1_run_name)
        seed_stage2 = _stable_seed(stage2_run_name)
        stage2_root = output_dir / output_location / ("%s.root" % stage2_run_name)

        row = {
            "status": "",
            "process": process,
            "run_dir": str(run_dir),
            "c3": _format_float(c3),
            "d4": _format_float(d4),
            "lhe_file": str(lhe_file) if lhe_file.exists() else "",
            "stage1_run_name": stage1_run_name,
            "stage1_input": str(stage1_input),
            "stage1_run": str(output_dir / ("%s.run" % stage1_run_name)),
            "stage1_output_lhe": str(output_dir / stage1_output_lhe),
            "stage1_weighted_lhe": str(output_dir / stage1_weighted_lhe) if probe_trials > 0 else "",
            "stage1_correction_file": str(output_dir / stage1_correction),
            "stage2_run_name": stage2_run_name,
            "stage2_lhe_file": str(output_dir / stage2_lhe_file),
            "stage2_input": str(stage2_input),
            "stage2_run": str(output_dir / ("%s.run" % stage2_run_name)),
            "stage2_output_root": str(stage2_root),
            "events": int(events),
            "probe_trials": int(probe_trials),
            "seed_stage1": seed_stage1,
            "seed_stage2": seed_stage2,
            "reason": "",
        }

        if reference_grid is not None and point_key not in reference_grid:
            row["status"] = "skipped_not_in_reference_grid"
            row["reason"] = "c3/d4 point is not present in reference grid manifest"
            rows.append(row)
            continue

        if not lhe_file.exists():
            row["status"] = "missing_lhe"
            row["reason"] = "unweighted_events.lhe(.gz) does not exist"
            rows.append(row)
            continue

        existing = [path for path in (stage1_input, stage2_input) if path.exists()]
        if existing and not overwrite:
            row["status"] = "skipped_existing"
            row["reason"] = "existing target(s): " + ";".join(str(path) for path in existing)
            rows.append(row)
            stage1_inputs.append(stage1_input.name)
            stage1_outputs.append(stage1_output_lhe)
            if probe_trials > 0:
                stage1_reweight_inputs.append("%s %s %s" % (stage1_output_lhe, stage1_correction, stage1_weighted_lhe))
            stage2_inputs.append(stage2_input.name)
            continue

        stage1_text = stage1_lhewriter_card(
            process_config,
            input_lhe=lhe_file,
            output_prefix=stage1_run_name,
            events=events,
            seed=seed_stage1,
            probe_trials=probe_trials,
            correction_file=stage1_correction,
        )
        stage2_text = stage2_hwsim_card(
            input_lhe=stage2_lhe_file,
            output_location=output_location,
            events=events,
            run_name=stage2_run_name,
            seed=seed_stage2,
        )
        stage1_input.write_text(stage1_text)
        stage2_input.write_text(stage2_text)
        row["status"] = "overwritten" if existing else "written"
        rows.append(row)
        stage1_inputs.append(stage1_input.name)
        stage1_outputs.append(stage1_output_lhe)
        if probe_trials > 0:
            stage1_reweight_inputs.append("%s %s %s" % (stage1_output_lhe, stage1_correction, stage1_weighted_lhe))
        stage2_inputs.append(stage2_input.name)

    fieldnames = [
        "status",
        "process",
        "run_dir",
        "c3",
        "d4",
        "lhe_file",
        "stage1_run_name",
        "stage1_input",
        "stage1_run",
        "stage1_output_lhe",
        "stage1_weighted_lhe",
        "stage1_correction_file",
        "stage2_run_name",
        "stage2_lhe_file",
        "stage2_input",
        "stage2_run",
        "stage2_output_root",
        "events",
        "probe_trials",
        "seed_stage1",
        "seed_stage2",
        "reason",
    ]
    with manifest_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_input_list(output_dir / "stage1_inputs_to_run.txt", stage1_inputs)
    _write_input_list(output_dir / "stage1_outputs_to_normalize.txt", stage1_outputs)
    _write_input_list(output_dir / "stage1_outputs_to_reweight.txt", stage1_reweight_inputs)
    _write_input_list(output_dir / "stage2_inputs_to_run.txt", stage2_inputs)

    return manifest_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("process", choices=sorted(PROCESS_CONFIGS))
    parser.add_argument("--mg5-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--events", required=True, type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-location", default="events/")
    parser.add_argument("--run-prefix", default="FS")
    parser.add_argument("--probe-trials", type=int, default=0)
    parser.add_argument("--reference-grid-manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest = prepare_forced_splitting_inputs(
        process=args.process,
        mg5_dir=args.mg5_dir,
        output_dir=args.outdir,
        events=args.events,
        manifest_file=args.manifest,
        output_location=args.output_location,
        run_prefix=args.run_prefix,
        probe_trials=args.probe_trials,
        reference_grid_manifest=args.reference_grid_manifest,
        overwrite=args.overwrite,
    )
    print("Prepared forced-splitting manifest:", manifest)
    print("Stage-1 input list:", Path(args.outdir) / "stage1_inputs_to_run.txt")
    print("Stage-1 split-LHE normalization list:", Path(args.outdir) / "stage1_outputs_to_normalize.txt")
    print("Stage-1 split-LHE reweighting list:", Path(args.outdir) / "stage1_outputs_to_reweight.txt")
    print("Stage-2 input list:", Path(args.outdir) / "stage2_inputs_to_run.txt")


if __name__ == "__main__":
    main()
