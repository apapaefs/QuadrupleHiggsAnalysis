"""Production campaign for direct HEFT gg -> hh b bbar b bbar samples."""

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .herwig_cards import DEFAULT_HERWIG_PDF_NAME, stage2_hwsim_card
from .hhhbb_campaign import _print_mg5_monitor, monitor_mg5_grid as _monitor_mg5_grid
from .mg5_grid import DEFAULT_MG5_CORES, MG5_PROCESS_CONFIGS, _find_lhe, _run_name, load_signal_grid, prepare_mg5_grid
from .run_chain import count_lhe_events
from .validation_hbb import _run_herwig


@dataclass
class HHBBBBHEFTCampaignConfig(object):
    mg5_dir: Path
    reference_grid_manifest: Path
    workdir: Path
    events: int
    run_name: str = "hhbbbb_heft_campaign"
    herwig: str = "Herwig"
    output_location: str = "events"
    seed_stage2: int = 89968250
    pdf_name: str = DEFAULT_HERWIG_PDF_NAME
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


def _run_point(config, point, source_lhe, input_event_count, events_dir, runner):
    process_config = MG5_PROCESS_CONFIGS["gg_hhbbbb_heft"]
    run_name = _run_name(process_config, point)
    point_dir = Path(config.workdir) / run_name
    point_dir.mkdir(parents=True, exist_ok=True)

    stage2_name = "%s_hhbbbb_heft_stage2" % run_name
    stage2_card = point_dir / ("%s.in" % stage2_name)
    stage2_run = point_dir / ("%s.run" % stage2_name)
    stage2_root = events_dir / ("%s.root" % stage2_name)
    stage2_text = stage2_hwsim_card(
        input_lhe=source_lhe,
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

    summary = {
        "status": "dry_run" if config.dry_run else "complete",
        "process": "gg_hhbbbb_heft",
        "run_name": run_name,
        "run_group": point["run_group"],
        "c3": point["c3"],
        "d4": point["d4"],
        "source_lhe": str(source_lhe),
        "input_events": int(input_event_count),
        "requested_events": int(config.events),
        "point_dir": str(point_dir),
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
    return {
        "status": point_summary.get("status", ""),
        "run_name": point_summary.get("run_name", ""),
        "run_group": point_summary.get("run_group", ""),
        "c3": point_summary.get("c3", ""),
        "d4": point_summary.get("d4", ""),
        "source_lhe": point_summary.get("source_lhe", ""),
        "input_events": point_summary.get("input_events", ""),
        "requested_events": point_summary.get("requested_events", ""),
        "stage2_root": point_summary.get("stage2_root", ""),
        "stage2_root_exists": Path(point_summary.get("stage2_root", "")).exists(),
        "summary": point_summary.get("summary", ""),
        "reason": point_summary.get("reason", ""),
    }


def run_hhbbbb_heft_campaign(config, runner=None):
    """Run direct HEFT hhbbbb production over the unique-c3 projection."""

    if runner is None:
        runner = _default_runner
    if config.events < 1:
        raise ValueError("--events must be positive")

    config.workdir = Path(config.workdir)
    config.mg5_dir = Path(config.mg5_dir)
    config.reference_grid_manifest = Path(config.reference_grid_manifest)
    config.workdir.mkdir(parents=True, exist_ok=True)
    events_dir = config.workdir / config.output_location
    events_dir.mkdir(parents=True, exist_ok=True)

    points = load_signal_grid(config.reference_grid_manifest, c3_only=True, c3_only_d4="0.0")
    process_config = MG5_PROCESS_CONFIGS["gg_hhbbbb_heft"]
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

    manifest = config.workdir / "hhbbbb_heft_campaign_manifest.csv"
    _write_csv(manifest, [_manifest_row(point) for point in point_summaries])
    summary = {
        "status": "dry_run" if config.dry_run else "complete",
        "process": "gg_hhbbbb_heft",
        "reference_grid_manifest": str(config.reference_grid_manifest),
        "mg5_dir": str(config.mg5_dir),
        "workdir": str(config.workdir),
        "events": int(config.events),
        "points_requested": len(points),
        "processed_points": sum(1 for point in point_summaries if point.get("status") in {"complete", "dry_run"}),
        "missing_points": len(missing),
        "manifest": str(manifest),
        "points": point_summaries,
    }
    summary_path = config.workdir / "hhbbbb_heft_campaign_summary.json"
    summary["summary"] = str(summary_path)
    _write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n", True)
    return summary


def check_campaign(workdir):
    workdir = Path(workdir)
    manifest = workdir / "hhbbbb_heft_campaign_manifest.csv"
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
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def monitor_mg5_grid(mg5_dir, reference_grid_manifest, count_events=False, tail=25, show_points=8):
    return _monitor_mg5_grid(
        mg5_dir=mg5_dir,
        reference_grid_manifest=reference_grid_manifest,
        process="gg_hhbbbb_heft",
        c3_only=True,
        c3_only_d4="0.0",
        count_events=count_events,
        tail=tail,
        show_points=show_points,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare-mg5", help="write/run the c3-only gg_hhbbbb_heft MG5 grid deck")
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

    run = subparsers.add_parser("run", help="run direct HEFT hhbbbb Stage-2 Herwig/HwSim production")
    run.add_argument("--mg5-dir", required=True, type=Path)
    run.add_argument("--reference-grid-manifest", required=True, type=Path)
    run.add_argument("--workdir", required=True, type=Path)
    run.add_argument("--events", required=True, type=int)
    run.add_argument("--run-name", default="hhbbbb_heft_campaign")
    run.add_argument("--herwig", default="Herwig")
    run.add_argument("--output-location", default="events")
    run.add_argument("--pdf-name", default=DEFAULT_HERWIG_PDF_NAME)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-missing", action="store_true")

    check = subparsers.add_parser("check", help="summarize an existing campaign workdir")
    check.add_argument("--workdir", required=True, type=Path)

    monitor = subparsers.add_parser("monitor-mg5", help="summarize MG5 hard-process grid progress")
    monitor.add_argument("--mg5-dir", required=True, type=Path)
    monitor.add_argument("--reference-grid-manifest", required=True, type=Path)
    monitor.add_argument("--count-events", action="store_true", help="Count events inside completed LHE files; slower.")
    monitor.add_argument("--tail", type=int, default=25, help="Number of mg5_grid.log tail lines to print.")
    monitor.add_argument("--show-points", type=int, default=8, help="Number of point/debug/process rows to show per section.")
    monitor.add_argument("--json", action="store_true", help="Write the full monitor summary as JSON.")

    args = parser.parse_args(argv)
    if args.command == "prepare-mg5":
        process_dir = args.process_dir or (args.mg5_root / "gg_hhbbbb_heft")
        summary = prepare_mg5_grid(
            process="gg_hhbbbb_heft",
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
            c3_only=True,
            c3_only_d4="0.0",
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "run":
        summary = run_hhbbbb_heft_campaign(
            HHBBBBHEFTCampaignConfig(
                mg5_dir=args.mg5_dir,
                reference_grid_manifest=args.reference_grid_manifest,
                workdir=args.workdir,
                events=args.events,
                run_name=args.run_name,
                herwig=args.herwig,
                output_location=args.output_location,
                pdf_name=args.pdf_name,
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
