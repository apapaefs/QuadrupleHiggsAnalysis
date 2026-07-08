"""Production campaign for gg -> hhhg with forced g -> b bbar."""

import argparse
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .herwig_cards import DEFAULT_HERWIG_PDF_NAME, PROCESS_CONFIGS, stage1_lhewriter_card, stage2_hwsim_card
from .lhe_merge import event_ranges, merge_weighted_lhe_chunks, write_lhe_event_slice
from .lhe_validation import normalize_lhe_file_process_ids
from .lhe_weights import apply_weights, verify_weighted_lhe
from .mg5_grid import DEFAULT_MG5_CORES, MG5_PROCESS_CONFIGS, _find_lhe, _run_name, load_signal_grid, prepare_mg5_grid
from .run_chain import count_lhe_events
from .validation_hbb import _run_herwig


@dataclass
class HHHBBCampaignConfig(object):
    mg5_dir: Path
    reference_grid_manifest: Path
    workdir: Path
    events: int
    jobs: int = 32
    probe_trials: int = 99999
    run_name: str = "hhhbb_campaign"
    herwig: str = "Herwig"
    output_location: str = "events"
    seed_stage1: int = 31122002
    seed_stage2: int = 89968250
    pdf_name: str = DEFAULT_HERWIG_PDF_NAME
    input_xsec_error: float = None
    allow_zero_probe_successes: bool = False
    overwrite: bool = False
    dry_run: bool = False
    skip_missing: bool = False


def _default_runner(command, cwd):
    subprocess.run(command, cwd=str(cwd), check=True)


def _write_text(path, text, overwrite):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError("%s already exists; pass --overwrite to replace it" % path)
    path.write_text(text)


def _write_csv(path, rows):
    fieldnames = [
        "status",
        "run_name",
        "run_group",
        "c3",
        "d4",
        "source_lhe",
        "input_events",
        "requested_events",
        "jobs",
        "active_jobs",
        "probe_trials",
        "zero_success_rows",
        "nonzero_weight_rows",
        "mean_p_hat",
        "merged_lhe",
        "merged_events",
        "merged_xsec_pb",
        "stage2_root",
        "stage2_root_exists",
        "summary",
        "reason",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _point_key(c3, d4):
    return (round(float(c3), 9), round(float(d4), 9))


def _combine_weight_checks(job_summaries):
    checks = [summary.get("weight_check") for summary in job_summaries if summary.get("weight_check")]
    if not checks:
        return None
    rows = sum(int(check.get("correction_rows", 0)) for check in checks)
    mean_p_hat = (
        sum(float(check.get("mean_p_hat", 0.0)) * int(check.get("correction_rows", 0)) for check in checks) / rows
        if rows
        else 0.0
    )
    return {
        "parallel_jobs": len(job_summaries),
        "correction_rows": rows,
        "zero_success_rows": sum(int(check.get("zero_success_rows", 0)) for check in checks),
        "nonzero_weight_rows": sum(int(check.get("nonzero_weight_rows", 0)) for check in checks),
        "mean_p_hat": float(mean_p_hat),
        "jobs": checks,
    }


def _run_stage1_chunk(job):
    commands = []
    commands.append(_run_herwig(job["herwig"], "read", Path(job["stage1_card"]).name, job["job_dir"], job["runner"], job["dry_run"]))
    commands.append(_run_herwig(job["herwig"], "run", Path(job["stage1_run"]).name, job["job_dir"], job["runner"], job["dry_run"]))

    weight_check = None
    normalization_message = None
    if not job["dry_run"]:
        stage1_lhe = Path(job["stage1_lhe"])
        correction_file = Path(job["correction_file"])
        weighted_lhe = Path(job["weighted_lhe"])
        if not stage1_lhe.exists():
            raise FileNotFoundError("Stage-1 LHE was not produced: %s" % stage1_lhe)
        _, normalization_message = normalize_lhe_file_process_ids(stage1_lhe)
        if job["probe_trials"] > 0:
            if not correction_file.exists():
                raise FileNotFoundError("ProbeTrials was nonzero but no correction sidecar exists: %s" % correction_file)
            apply_weights(
                stage1_lhe,
                correction_file,
                weighted_lhe,
                input_xsec_error=job["input_xsec_error"],
            )
            weight_check = verify_weighted_lhe(stage1_lhe, correction_file, weighted_lhe)
            if not weight_check["ok"]:
                raise RuntimeError("Weighted LHE verification failed for %s: %s" % (job["run_name"], weight_check))
            if weight_check["zero_success_rows"] and not job["allow_zero_probe_successes"]:
                raise RuntimeError(
                    "Correction sidecar for %s has %d rows with probe_successes = 0; "
                    "increase --probe-trials or pass --allow-zero-probe-successes"
                    % (job["run_name"], weight_check["zero_success_rows"])
                )
        else:
            weighted_lhe = stage1_lhe

    return {
        "job_index": job["job_index"],
        "run_name": job["run_name"],
        "events": int(job["events"]),
        "input_lhe": str(job["input_lhe"]),
        "stage1_card": str(job["stage1_card"]),
        "stage1_run": str(job["stage1_run"]),
        "stage1_lhe": str(job["stage1_lhe"]),
        "correction_file": str(job["correction_file"]),
        "weighted_lhe": str(job["weighted_lhe"] if job["probe_trials"] > 0 else job["stage1_lhe"]),
        "normalization_message": normalization_message,
        "weight_check": weight_check,
        "commands": [" ".join(command) for command in commands],
    }


def _prepare_point_jobs(config, point, source_lhe, input_event_count, point_dir):
    active_jobs = min(int(config.jobs), int(config.events))
    ranges = event_ranges(config.events, active_jobs)
    jobs_dir = point_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    chunks = []
    for job_index, (start, stop) in enumerate(ranges, start=1):
        job_events = stop - start
        job_name = "%s_job%03d" % (_run_name(MG5_PROCESS_CONFIGS["gg_hhhg"], point), job_index)
        job_dir = jobs_dir / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        chunk_lhe = job_dir / ("%s_input.lhe" % job_name)
        write_lhe_event_slice(source_lhe, chunk_lhe, start, stop, overwrite=config.overwrite)

        stage1_name = "%s_stage1" % job_name
        stage1_card = job_dir / ("%s.in" % stage1_name)
        stage1_run = job_dir / ("%s.run" % stage1_name)
        stage1_lhe = job_dir / ("%s.lhe" % stage1_name)
        correction_file = job_dir / ("%s.force_split.weights" % stage1_name)
        weighted_lhe = job_dir / ("%s.weighted.lhe" % stage1_name)
        stage1_text = stage1_lhewriter_card(
            PROCESS_CONFIGS["gg_hhhg"],
            input_lhe=chunk_lhe,
            output_prefix=stage1_name,
            events=job_events,
            seed=config.seed_stage1 + 1000003 * job_index,
            probe_trials=config.probe_trials,
            correction_file=correction_file.name,
            pdf_name=config.pdf_name,
        )
        _write_text(stage1_card, stage1_text, config.overwrite)
        job = {
            "job_index": job_index,
            "run_name": job_name,
            "job_dir": job_dir,
            "events": job_events,
            "input_lhe": chunk_lhe,
            "stage1_card": stage1_card,
            "stage1_run": stage1_run,
            "stage1_lhe": stage1_lhe,
            "correction_file": correction_file,
            "weighted_lhe": weighted_lhe,
            "probe_trials": int(config.probe_trials),
            "input_xsec_error": config.input_xsec_error,
            "allow_zero_probe_successes": bool(config.allow_zero_probe_successes),
            "herwig": config.herwig,
            "dry_run": bool(config.dry_run),
            "runner": None,
        }
        jobs.append(job)
        chunks.append(
            {
                "job_index": job_index,
                "events": job_events,
                "input_lhe": str(chunk_lhe),
                "stage1_card": str(stage1_card),
                "stage1_run": str(stage1_run),
                "stage1_lhe": str(stage1_lhe),
                "correction_file": str(correction_file),
                "weighted_lhe": str(weighted_lhe),
            }
        )

    return active_jobs, jobs, chunks


def _run_point(config, point, source_lhe, input_event_count, events_dir, runner):
    run_name = _run_name(MG5_PROCESS_CONFIGS["gg_hhhg"], point)
    point_dir = Path(config.workdir) / run_name
    point_dir.mkdir(parents=True, exist_ok=True)
    active_jobs, jobs, chunks = _prepare_point_jobs(config, point, source_lhe, input_event_count, point_dir)
    for job in jobs:
        job["runner"] = runner

    job_summaries = []
    if config.dry_run or active_jobs == 1:
        for job in jobs:
            job_summaries.append(_run_stage1_chunk(job))
    else:
        with ThreadPoolExecutor(max_workers=active_jobs) as executor:
            futures = {executor.submit(_run_stage1_chunk, job): job for job in jobs}
            for future in as_completed(futures):
                job_summaries.append(future.result())
        job_summaries.sort(key=lambda item: item["job_index"])

    merged_lhe = point_dir / ("%s_split.weighted.merged.lhe.gz" % run_name)
    merge_summary_path = point_dir / "merge_summary.json"
    merge_summary = None
    if not config.dry_run:
        weighted_inputs = [summary["weighted_lhe"] for summary in job_summaries]
        merge_summary = merge_weighted_lhe_chunks(
            weighted_inputs,
            merged_lhe,
            summary_path=merge_summary_path,
            overwrite=config.overwrite,
        )

    stage2_name = "%s_hhhbb_stage2" % run_name
    stage2_card = point_dir / ("%s.in" % stage2_name)
    stage2_run = point_dir / ("%s.run" % stage2_name)
    stage2_root = events_dir / ("%s.root" % stage2_name)
    stage2_text = stage2_hwsim_card(
        input_lhe=merged_lhe,
        output_location=events_dir,
        events=config.events,
        run_name=stage2_name,
        seed=config.seed_stage2,
        pdf_name=config.pdf_name,
    )
    _write_text(stage2_card, stage2_text, config.overwrite)

    stage2_commands = []
    stage2_commands.append(_run_herwig(config.herwig, "read", stage2_card.name, point_dir, runner, config.dry_run))
    stage2_commands.append(_run_herwig(config.herwig, "run", stage2_run.name, point_dir, runner, config.dry_run))
    if not config.dry_run and not stage2_root.exists():
        raise FileNotFoundError("Stage-2 ROOT output was not produced: %s" % stage2_root)

    weight_check = _combine_weight_checks(job_summaries)
    summary = {
        "status": "dry_run" if config.dry_run else "complete",
        "process": "gg_hhhg",
        "run_name": run_name,
        "run_group": point["run_group"],
        "c3": point["c3"],
        "d4": point["d4"],
        "source_lhe": str(source_lhe),
        "input_events": int(input_event_count),
        "requested_events": int(config.events),
        "jobs": int(config.jobs),
        "active_jobs": int(active_jobs),
        "probe_trials": int(config.probe_trials),
        "allow_zero_probe_successes": bool(config.allow_zero_probe_successes),
        "point_dir": str(point_dir),
        "chunks": chunks,
        "job_summaries": job_summaries,
        "weight_check": weight_check,
        "merged_weighted_lhe": str(merged_lhe),
        "merge_summary": str(merge_summary_path),
        "merge": merge_summary,
        "stage2_card": str(stage2_card),
        "stage2_run": str(stage2_run),
        "stage2_root": str(stage2_root),
        "stage2_commands": [" ".join(command) for command in stage2_commands],
        "dry_run": bool(config.dry_run),
    }
    summary_path = point_dir / "point_summary.json"
    summary["summary"] = str(summary_path)
    _write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n", True)
    return summary


def _manifest_row(point_summary):
    weight_check = point_summary.get("weight_check") or {}
    merge = point_summary.get("merge") or {}
    return {
        "status": point_summary.get("status", ""),
        "run_name": point_summary.get("run_name", ""),
        "run_group": point_summary.get("run_group", ""),
        "c3": point_summary.get("c3", ""),
        "d4": point_summary.get("d4", ""),
        "source_lhe": point_summary.get("source_lhe", ""),
        "input_events": point_summary.get("input_events", ""),
        "requested_events": point_summary.get("requested_events", ""),
        "jobs": point_summary.get("jobs", ""),
        "active_jobs": point_summary.get("active_jobs", ""),
        "probe_trials": point_summary.get("probe_trials", ""),
        "zero_success_rows": weight_check.get("zero_success_rows", ""),
        "nonzero_weight_rows": weight_check.get("nonzero_weight_rows", ""),
        "mean_p_hat": weight_check.get("mean_p_hat", ""),
        "merged_lhe": point_summary.get("merged_weighted_lhe", ""),
        "merged_events": merge.get("total_events", ""),
        "merged_xsec_pb": merge.get("merged_xsec_pb", ""),
        "stage2_root": point_summary.get("stage2_root", ""),
        "stage2_root_exists": Path(point_summary.get("stage2_root", "")).exists(),
        "summary": point_summary.get("summary", ""),
        "reason": point_summary.get("reason", ""),
    }


def run_hhhbb_campaign(config, runner=None):
    """Run the hhhbb forced-splitting production campaign over the reference grid."""

    if runner is None:
        runner = _default_runner
    if config.jobs < 1:
        raise ValueError("--jobs must be positive")
    if config.events < 1:
        raise ValueError("--events must be positive")

    config.workdir = Path(config.workdir)
    config.mg5_dir = Path(config.mg5_dir)
    config.reference_grid_manifest = Path(config.reference_grid_manifest)
    config.workdir.mkdir(parents=True, exist_ok=True)
    events_dir = config.workdir / config.output_location
    events_dir.mkdir(parents=True, exist_ok=True)

    points = load_signal_grid(config.reference_grid_manifest)
    process_config = MG5_PROCESS_CONFIGS["gg_hhhg"]
    point_inputs = []
    missing = []
    for point in points:
        run_name = _run_name(process_config, point)
        run_dir = config.mg5_dir / "Events" / run_name
        source_lhe = _find_lhe(run_dir)
        if source_lhe is None:
            missing_summary = {
                "status": "missing_lhe",
                "run_name": run_name,
                "run_group": point["run_group"],
                "c3": point["c3"],
                "d4": point["d4"],
                "source_lhe": str(run_dir / "unweighted_events.lhe.gz"),
                "reason": "MG5 unweighted_events.lhe(.gz) not found",
            }
            missing.append(missing_summary)
            continue

        input_event_count = count_lhe_events(source_lhe)
        if input_event_count < config.events:
            raise RuntimeError(
                "%s contains %d events but --events requested %d. Generate more MG5 events or lower --events."
                % (source_lhe, input_event_count, config.events)
            )
        point_inputs.append((point, source_lhe.resolve(), input_event_count))

    if missing and not config.skip_missing:
        example = missing[0]
        raise FileNotFoundError(
            "Missing %d MG5 LHE file(s), first is %s at %s; run prepare-mg5/MG5 first or pass --skip-missing"
            % (len(missing), example["run_name"], example["source_lhe"])
        )

    point_summaries = list(missing)
    for point, source_lhe, input_event_count in point_inputs:
        point_summaries.append(_run_point(config, point, source_lhe, input_event_count, events_dir.resolve(), runner))

    manifest = config.workdir / "hhhbb_campaign_manifest.csv"
    _write_csv(manifest, [_manifest_row(point) for point in point_summaries])
    summary = {
        "status": "dry_run" if config.dry_run else "complete",
        "process": "gg_hhhg",
        "reference_grid_manifest": str(config.reference_grid_manifest),
        "mg5_dir": str(config.mg5_dir),
        "workdir": str(config.workdir),
        "events": int(config.events),
        "jobs": int(config.jobs),
        "probe_trials": int(config.probe_trials),
        "allow_zero_probe_successes": bool(config.allow_zero_probe_successes),
        "points_requested": len(points),
        "processed_points": sum(1 for point in point_summaries if point.get("status") in {"complete", "dry_run"}),
        "missing_points": len(missing),
        "manifest": str(manifest),
        "points": point_summaries,
    }
    summary_path = config.workdir / "hhhbb_campaign_summary.json"
    summary["summary"] = str(summary_path)
    _write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n", True)
    return summary


def check_campaign(workdir):
    workdir = Path(workdir)
    manifest = workdir / "hhhbb_campaign_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError("Campaign manifest not found: %s" % manifest)
    rows = list(csv.DictReader(manifest.open()))
    status_counts = {}
    for row in rows:
        status_counts[row.get("status", "")] = status_counts.get(row.get("status", ""), 0) + 1
    summary = {
        "workdir": str(workdir),
        "manifest": str(manifest),
        "rows": len(rows),
        "status_counts": status_counts,
        "complete_roots": sum(1 for row in rows if str(row.get("stage2_root_exists", "")).lower() == "true"),
        "zero_success_rows": sum(int(float(row.get("zero_success_rows") or 0)) for row in rows),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _tail_lines(path, limit):
    path = Path(path)
    if not path.exists() or limit <= 0:
        return []
    lines = path.read_text(errors="replace").splitlines()
    return lines[-int(limit) :]


def _interesting_log_lines(path, limit=8):
    wanted = ("error", "failed", "traceback", "madgraph5error", "command not executed")
    lines = []
    for line in Path(path).read_text(errors="replace").splitlines():
        lowered = line.lower()
        if any(token in lowered for token in wanted):
            lines.append(line)
    return lines[-int(limit) :]


def _current_mg5_processes(mg5_dir):
    try:
        proc = subprocess.run(
            ["pgrep", "-af", str(Path(mg5_dir).resolve())],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return []
    lines = []
    for line in proc.stdout.splitlines():
        if "pgrep -af" in line or "monitor-mg5" in line:
            continue
        lines.append(line)
    return lines


def _debug_log_run_name(path):
    name = Path(path).name
    marker = "_tag_"
    if marker in name:
        return name.split(marker, 1)[0]
    return name.rsplit("_debug.log", 1)[0]


def _latest_mg5_progress_line(lines):
    progress = ""
    for line in lines:
        if "Idle:" in line and "Running:" in line and "Completed:" in line:
            progress = line
    return progress


def _configured_cores_from_latest_deck(deck_dir, process):
    deck_dir = Path(deck_dir)
    decks = sorted(deck_dir.glob("%s_*events.mg5cmd" % process), key=lambda path: path.stat().st_mtime, reverse=True)
    if not decks:
        return None
    configured_cores = None
    for raw_line in decks[0].read_text(errors="replace").splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 3 and parts[0] == "set" and parts[1] == "nb_core":
            try:
                configured_cores = int(parts[2])
            except ValueError:
                configured_cores = None
    return configured_cores


def monitor_mg5_grid(
    mg5_dir,
    reference_grid_manifest,
    process="gg_hhhg",
    count_events=False,
    tail=25,
    show_points=8,
):
    """Summarize MG5 hard-process grid progress for the hhhbb campaign."""

    if process not in MG5_PROCESS_CONFIGS:
        raise ValueError("Unknown MG5 forced-splitting process %r" % process)
    mg5_dir = Path(mg5_dir)
    process_config = MG5_PROCESS_CONFIGS[process]
    points = load_signal_grid(reference_grid_manifest)
    rows = []
    counts = {"complete": 0, "incomplete": 0, "pending": 0}
    total_events = 0
    for point in points:
        run_name = _run_name(process_config, point)
        run_dir = mg5_dir / "Events" / run_name
        lhe = _find_lhe(run_dir)
        event_count = None
        if lhe is not None:
            status = "complete"
            counts["complete"] += 1
            if count_events:
                event_count = count_lhe_events(lhe)
                total_events += event_count
        elif run_dir.exists():
            status = "incomplete"
            counts["incomplete"] += 1
        else:
            status = "pending"
            counts["pending"] += 1
        rows.append(
            {
                "status": status,
                "run_name": run_name,
                "run_group": point["run_group"],
                "c3": point["c3"],
                "d4": point["d4"],
                "run_dir": str(run_dir),
                "lhe_file": "" if lhe is None else str(lhe),
                "event_count": event_count,
            }
        )

    deck_dir = mg5_dir / "ForcedSplittingDecks"
    grid_manifest = deck_dir / "mg5_grid_manifest.csv"
    grid_log = deck_dir / "mg5_grid.log"
    debug_logs = sorted(mg5_dir.glob("run_*_tag_*_debug.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    debug_summaries = [
        {
            "run_name": _debug_log_run_name(path),
            "path": str(path),
            "mtime": path.stat().st_mtime,
            "errors": _interesting_log_lines(path, limit=8),
        }
        for path in debug_logs[: max(0, int(show_points))]
    ]
    current_processes = _current_mg5_processes(mg5_dir)
    grid_log_tail = _tail_lines(grid_log, tail)
    summary = {
        "process": process,
        "mg5_dir": str(mg5_dir),
        "reference_grid_manifest": str(reference_grid_manifest),
        "grid_points": len(points),
        "complete_lhes": counts["complete"],
        "incomplete_run_dirs": counts["incomplete"],
        "pending_run_dirs": counts["pending"],
        "debug_logs": len(debug_logs),
        "grid_manifest": str(grid_manifest) if grid_manifest.exists() else "",
        "grid_log": str(grid_log) if grid_log.exists() else "",
        "grid_log_mtime": grid_log.stat().st_mtime if grid_log.exists() else None,
        "configured_cores": _configured_cores_from_latest_deck(deck_dir, process),
        "current_processes": current_processes,
        "current_process_count": len(current_processes),
        "latest_progress_line": _latest_mg5_progress_line(grid_log_tail),
        "total_counted_events": total_events if count_events else None,
        "points": rows,
        "recent_debug_logs": debug_summaries,
        "grid_log_tail": grid_log_tail,
    }
    return summary


def _print_mg5_monitor(summary, show_points=8):
    print("MG5 hhhbb hard-process monitor")
    print("  process dir:", summary["mg5_dir"])
    print("  grid points:", summary["grid_points"])
    print("  completed LHEs:", "%d/%d" % (summary["complete_lhes"], summary["grid_points"]))
    print("  incomplete run dirs:", summary["incomplete_run_dirs"])
    print("  pending run dirs:", summary["pending_run_dirs"])
    print("  debug logs:", summary["debug_logs"])
    if summary["configured_cores"] is not None:
        print("  configured MG5 cores:", summary["configured_cores"])
    print("  matching processes:", summary["current_process_count"])
    if summary["latest_progress_line"]:
        print("  latest MG5 progress:", summary["latest_progress_line"])
    if summary["grid_log"]:
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(summary["grid_log_mtime"]))
        print("  grid log:", summary["grid_log"], "(updated %s)" % mtime)
    if summary["total_counted_events"] is not None:
        print("  counted LHE events:", summary["total_counted_events"])

    for status, label in (
        ("complete", "recent/completed points"),
        ("incomplete", "incomplete run dirs"),
        ("pending", "pending points"),
    ):
        selected = [row for row in summary["points"] if row["status"] == status][: int(show_points)]
        if not selected:
            continue
        print()
        print("  %s:" % label)
        for row in selected:
            suffix = "" if row["event_count"] is None else " events=%s" % row["event_count"]
            print("    {run_name} c3={c3} d4={d4}{suffix}".format(suffix=suffix, **row))

    if summary["recent_debug_logs"]:
        print()
        print("  recent debug-log errors:")
        for debug in summary["recent_debug_logs"]:
            print("   ", debug["run_name"], "->", debug["path"])
            for line in debug["errors"][-3:]:
                print("      ", line)

    if summary["current_processes"]:
        print()
        print("  matching processes:")
        for line in summary["current_processes"][: int(show_points)]:
            print("   ", line)

    if summary["grid_log_tail"]:
        print()
        print("  grid log tail:")
        for line in summary["grid_log_tail"]:
            print("   ", line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare-mg5", help="write/run the gg_hhhg MG5 grid deck")
    prepare.add_argument("--mg5-root", required=True, type=Path)
    prepare.add_argument("--process-dir", type=Path, default=None)
    prepare.add_argument("--reference-grid-manifest", required=True, type=Path)
    prepare.add_argument("--events", type=int, required=True)
    prepare.add_argument("--signal-run-card", type=Path, default=None)
    prepare.add_argument("--deck-dir", type=Path, default=None)
    prepare.add_argument("--accuracy", default="0.02")
    prepare.add_argument("--points", type=int, default=3000)
    prepare.add_argument("--iterations", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--cores", type=int, default=DEFAULT_MG5_CORES)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run", help="run the forced-splitting hhhbb campaign")
    run.add_argument("--mg5-dir", required=True, type=Path)
    run.add_argument("--reference-grid-manifest", required=True, type=Path)
    run.add_argument("--workdir", required=True, type=Path)
    run.add_argument("--events", required=True, type=int)
    run.add_argument("--jobs", type=int, default=32)
    run.add_argument("--probe-trials", type=int, default=99999)
    run.add_argument("--run-name", default="hhhbb_campaign")
    run.add_argument("--herwig", default="Herwig")
    run.add_argument("--output-location", default="events")
    run.add_argument("--pdf-name", default=DEFAULT_HERWIG_PDF_NAME)
    run.add_argument("--input-xsec-error", type=float, default=None)
    run.add_argument("--allow-zero-probe-successes", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-missing", action="store_true")

    check = subparsers.add_parser("check", help="summarize an existing campaign workdir")
    check.add_argument("--workdir", required=True, type=Path)

    monitor = subparsers.add_parser("monitor-mg5", help="summarize MG5 hard-process grid progress")
    monitor.add_argument("--mg5-dir", required=True, type=Path)
    monitor.add_argument("--reference-grid-manifest", required=True, type=Path)
    monitor.add_argument("--process", choices=sorted(MG5_PROCESS_CONFIGS), default="gg_hhhg")
    monitor.add_argument("--count-events", action="store_true", help="Count events inside completed LHE files; slower.")
    monitor.add_argument("--tail", type=int, default=25, help="Number of mg5_grid.log tail lines to print.")
    monitor.add_argument("--show-points", type=int, default=8, help="Number of point/debug/process rows to show per section.")
    monitor.add_argument("--json", action="store_true", help="Write the full monitor summary as JSON.")

    args = parser.parse_args(argv)
    if args.command == "prepare-mg5":
        process_dir = args.process_dir or (args.mg5_root / "gg_hhhg")
        summary = prepare_mg5_grid(
            process="gg_hhhg",
            process_dir=process_dir,
            reference_grid_manifest=args.reference_grid_manifest,
            events=args.events,
            signal_run_card=args.signal_run_card,
            deck_dir=args.deck_dir,
            accuracy=args.accuracy,
            points=args.points,
            iterations=args.iterations,
            seed=args.seed,
            cores=args.cores,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "run":
        summary = run_hhhbb_campaign(
            HHHBBCampaignConfig(
                mg5_dir=args.mg5_dir,
                reference_grid_manifest=args.reference_grid_manifest,
                workdir=args.workdir,
                events=args.events,
                jobs=args.jobs,
                probe_trials=args.probe_trials,
                run_name=args.run_name,
                herwig=args.herwig,
                output_location=args.output_location,
                pdf_name=args.pdf_name,
                input_xsec_error=args.input_xsec_error,
                allow_zero_probe_successes=args.allow_zero_probe_successes,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                skip_missing=args.skip_missing,
            )
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "check":
        check_campaign(args.workdir)
    elif args.command == "monitor-mg5":
        summary = monitor_mg5_grid(
            mg5_dir=args.mg5_dir,
            reference_grid_manifest=args.reference_grid_manifest,
            process=args.process,
            count_events=args.count_events,
            tail=args.tail,
            show_points=args.show_points,
        )
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            _print_mg5_monitor(summary, show_points=args.show_points)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
