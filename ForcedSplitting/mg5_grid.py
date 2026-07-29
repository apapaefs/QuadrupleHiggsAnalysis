"""Launch MG5 hard-process grids for the forced-splitting samples."""

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


PRODUCTION_SIGNAL_STATUSES = {"written", "skipped_existing", "complete", "skipped_complete"}
DEFAULT_MG5_CORES = 324


@dataclass
class MG5ProcessConfig(object):
    process: str
    run_prefix: str
    default_process_dir: str
    process_card_line: str


MG5_PROCESS_CONFIGS = {
    "gg_hhh": MG5ProcessConfig(
        process="gg_hhh",
        run_prefix="run_gg_hhh",
        default_process_dir="gg_hhh",
        process_card_line="generate g g > h h h [noborn=QCD]",
    ),
    "gg_hhhg": MG5ProcessConfig(
        process="gg_hhhg",
        run_prefix="run_gg_hhhg",
        default_process_dir="gg_hhhg",
        process_card_line="generate g g > h h h g [noborn=QCD]",
    ),
    "gg_hhgg": MG5ProcessConfig(
        process="gg_hhgg",
        run_prefix="run_gg_hhgg",
        default_process_dir="gg_hhgg",
        process_card_line="generate g g > h h g g [noborn=QCD]",
    ),
    "gg_hggg": MG5ProcessConfig(
        process="gg_hggg",
        run_prefix="run_gg_hggg",
        default_process_dir="gg_hggg",
        process_card_line="generate g g > h g g g [noborn=QCD]",
    ),
    "gg_hhbbbb_heft": MG5ProcessConfig(
        process="gg_hhbbbb_heft",
        run_prefix="run_gg_hhbbbb_heft",
        default_process_dir="gg_hhbbbb_heft",
        process_card_line="generate g g > h h b b~ b b~",
    ),
}


MG5_PROCESS_RUN_SETTINGS = {
    "gg_hhbbbb_heft": [
        "set ptb 15.0",
        "set etab 3.0",
        "set drbb 0.3",
    ],
}


RUN_CARD_KEYS = [
    "ebeam1",
    "ebeam2",
    "pdlabel",
    "lhaid",
    "fixed_ren_scale",
    "fixed_fac_scale",
    "scale",
    "dsqrt_q2fact1",
    "dsqrt_q2fact2",
    "dynamical_scale_choice",
    "scalefact",
    "event_norm",
    "nhel",
    "sde_strategy",
    "bwcutoff",
    "maxjetflavor",
    "use_syst",
]


DEFAULT_RUN_CARD_VALUES = {
    "ebeam1": "7000.0",
    "ebeam2": "7000.0",
    "pdlabel": "nn23lo1",
    "lhaid": "230000",
    "fixed_ren_scale": "False",
    "fixed_fac_scale": "False",
    "scale": "91.188",
    "dsqrt_q2fact1": "91.188",
    "dsqrt_q2fact2": "91.188",
    "dynamical_scale_choice": "-1",
    "scalefact": "1.0",
    "event_norm": "average",
    "nhel": "1",
    "sde_strategy": "1",
    "bwcutoff": "15.0",
    "maxjetflavor": "4",
    "use_syst": "False",
}


def _madloop_library_env(cwd, base_env=None):
    """Return an environment that can load MG5-root HEPTools loop libraries."""
    env = dict(os.environ if base_env is None else base_env)
    cwd = Path(cwd)
    candidates = [
        cwd / "HEPTools" / "lib",
        cwd / "HEPTools" / "collier",
        cwd.parent / "HEPTools" / "lib",
        cwd.parent / "HEPTools" / "collier",
    ]
    library_paths = [str(path) for path in candidates if path.exists()]
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        library_paths.extend(path for path in existing.split(":") if path)
    if library_paths:
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(library_paths))
    return env


def _run_command(command, cwd, log_path):
    env = _madloop_library_env(cwd)
    with Path(log_path).open("a") as log:
        with subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        ) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            proc.wait()
    return proc.returncode


def _strip_comment(text):
    return text.split("!", 1)[0].strip()


def parse_run_card_values(run_card, keys=RUN_CARD_KEYS):
    values = {}
    if run_card is None:
        return values
    run_card = Path(run_card)
    if not run_card.exists():
        return values
    wanted = set(keys)
    with run_card.open(errors="replace") as handle:
        for raw_line in handle:
            if "=" not in raw_line:
                continue
            left, right = raw_line.split("=", 1)
            key = _strip_comment(right).split()[0] if _strip_comment(right) else ""
            if key in wanted:
                values[key] = left.strip()
    return values


def _point_key(c3, d4):
    return (round(float(c3), 9), round(float(d4), 9))


def load_signal_grid(manifest_file, statuses=None, c3_only=False, c3_only_d4="0.0"):
    statuses = set(statuses or PRODUCTION_SIGNAL_STATUSES)
    points = []
    seen = set()
    with Path(manifest_file).open(newline="") as handle:
        for row in csv.DictReader(handle):
            status = row.get("status")
            # Authoritative point manifests such as points_153.csv describe
            # coordinates rather than preparation state and therefore have no
            # status column.  Preserve filtering for campaign manifests while
            # accepting those status-less point lists.
            if statuses and status not in (None, "") and status not in statuses:
                continue
            c3 = row.get("c3")
            d4 = row.get("d4")
            if c3 in (None, "") or d4 in (None, ""):
                continue
            if c3_only:
                d4 = str(c3_only_d4)
                key = (round(float(c3), 9),)
            else:
                key = _point_key(c3, d4)
            if key in seen:
                continue
            seen.add(key)
            points.append(
                {
                    "run_group": row.get("run_group") or "4",
                    "c3": c3,
                    "d4": d4,
                    "seed": row.get("seed"),
                }
            )
    return points


def _run_name(process_config, point):
    return "%s_%s_%s_%s" % (
        process_config.run_prefix,
        point["run_group"],
        point["c3"],
        point["d4"],
    )


def _find_lhe(run_dir):
    for name in ("unweighted_events.lhe.gz", "unweighted_events.lhe"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def _execution_settings(cores):
    cores = int(cores)
    if cores < 1:
        raise ValueError("MG5 core count must be positive, got %d" % cores)
    run_mode = 2 if cores > 1 else 0
    return ["set run_mode %d" % run_mode, "set nb_core %d" % cores]


def _launch_block(run_name, point, events, run_settings, accuracy, points, iterations, seed, extra_run_settings=None):
    lines = [
        "launch %s --accuracy=%s --points=%s --iterations=%s" % (run_name, accuracy, points, iterations),
        "0",
    ]
    for key in RUN_CARD_KEYS:
        if key in run_settings:
            lines.append("set %s %s" % (key, run_settings[key]))
    lines.extend(str(setting) for setting in (extra_run_settings or []))
    lines.extend(
        [
            "set c3 %s" % point["c3"],
            "set d4 %s" % point["d4"],
            "set nevents %d" % int(events),
            "set iseed %d" % int(seed),
            "0",
            "",
        ]
    )
    return "\n".join(lines)


def _write_manifest(path, rows):
    fieldnames = [
        "status",
        "process",
        "run_name",
        "run_group",
        "c3",
        "d4",
        "events",
        "seed",
        "cores",
        "run_dir",
        "lhe_file",
        "deck",
        "reason",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_mg5_grid(
    process,
    process_dir,
    reference_grid_manifest,
    events,
    signal_run_card=None,
    deck_dir=None,
    manifest_file=None,
    accuracy="0.02",
    points=3000,
    iterations=5,
    seed=0,
    cores=DEFAULT_MG5_CORES,
    c3_only=False,
    c3_only_d4="0.0",
    overwrite=False,
    dry_run=False,
    runner=None,
):
    if process not in MG5_PROCESS_CONFIGS:
        raise ValueError("Unknown MG5 forced-splitting process %r" % process)
    if runner is None:
        runner = _run_command

    cores = int(cores)
    _execution_settings(cores)
    process_config = MG5_PROCESS_CONFIGS[process]
    process_dir = Path(process_dir)
    deck_dir = Path(deck_dir) if deck_dir is not None else process_dir / "ForcedSplittingDecks"
    deck_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_file) if manifest_file is not None else deck_dir / "mg5_grid_manifest.csv"
    deck = deck_dir / ("%s_%sevents.mg5cmd" % (process, int(events)))
    log_path = deck_dir / "mg5_grid.log"

    run_settings = dict(DEFAULT_RUN_CARD_VALUES)
    run_settings.update(parse_run_card_values(signal_run_card))

    rows = []
    blocks = []
    for point in load_signal_grid(reference_grid_manifest, c3_only=c3_only, c3_only_d4=c3_only_d4):
        run_name = _run_name(process_config, point)
        point_seed = int(point.get("seed") or seed)
        run_dir = process_dir / "Events" / run_name
        existing_lhe = _find_lhe(run_dir)
        row = {
            "status": "",
            "process": process,
            "run_name": run_name,
            "run_group": point["run_group"],
            "c3": point["c3"],
            "d4": point["d4"],
            "events": int(events),
            "seed": point_seed,
            "cores": int(cores),
            "run_dir": str(run_dir),
            "lhe_file": str(existing_lhe) if existing_lhe is not None else str(run_dir / "unweighted_events.lhe.gz"),
            "deck": str(deck),
            "reason": "",
        }
        if existing_lhe is not None and not overwrite:
            row["status"] = "skipped_existing_lhe"
            row["reason"] = "unweighted_events.lhe(.gz) already exists"
            rows.append(row)
            continue

        row["status"] = "scheduled"
        rows.append(row)
        blocks.append(
            _launch_block(
                run_name,
                point,
                events,
                run_settings,
                accuracy,
                points,
                iterations,
                point_seed,
                extra_run_settings=MG5_PROCESS_RUN_SETTINGS.get(process, []),
            )
        )

    deck_blocks = ["\n".join(_execution_settings(cores))] + blocks if blocks else []
    deck.write_text("\n\n".join(deck_blocks))
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_file, rows)

    run_status = "dry_run" if dry_run else "no_scheduled_points"
    exit_code = None
    if blocks and not dry_run:
        madevent = process_dir / "bin" / "madevent"
        if not madevent.exists():
            raise FileNotFoundError("MadEvent executable does not exist: %s" % madevent)
        exit_code = runner([str(madevent), str(deck)], process_dir, log_path)
        run_status = "complete" if exit_code == 0 else "failed:%d" % exit_code
        if exit_code != 0:
            raise RuntimeError("MG5/MadEvent grid launch failed with exit code %d; see %s" % (exit_code, log_path))

    return {
        "process": process,
        "process_dir": str(process_dir),
        "deck": str(deck),
        "manifest": str(manifest_file),
        "scheduled_points": len(blocks),
        "total_points": len(rows),
        "cores": int(cores),
        "c3_only": bool(c3_only),
        "c3_only_d4": str(c3_only_d4),
        "run_status": run_status,
        "exit_code": exit_code,
    }


def _default_signal_run_card(mg5_root):
    if mg5_root is None:
        return None
    return Path(mg5_root) / "gg_4h_c3d4" / "Cards" / "run_card.dat"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("process", choices=sorted(MG5_PROCESS_CONFIGS))
    parser.add_argument("--process-dir", type=Path, help="Existing MG5 process directory to launch.")
    parser.add_argument("--mg5-root", type=Path, help="MG5 root; used to infer --process-dir and signal run card.")
    parser.add_argument("--reference-grid-manifest", required=True, type=Path)
    parser.add_argument("--events", required=True, type=int, help="MG5 hard-process events requested per c3/d4 point.")
    parser.add_argument("--signal-run-card", type=Path)
    parser.add_argument("--deck-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--accuracy", default="0.02")
    parser.add_argument("--points", type=int, default=3000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cores", type=int, default=DEFAULT_MG5_CORES, help="MadEvent multicore worker count.")
    parser.add_argument("--c3-only", action="store_true", help="Schedule one run per unique c3 value, with d4 fixed by --c3-only-d4.")
    parser.add_argument("--c3-only-d4", default="0.0", help="d4 value written into c3-only run names and param cards.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    process_config = MG5_PROCESS_CONFIGS[args.process]
    process_dir = args.process_dir
    if process_dir is None:
        if args.mg5_root is None:
            parser.error("provide either --process-dir or --mg5-root")
        process_dir = args.mg5_root / process_config.default_process_dir
    signal_run_card = args.signal_run_card or _default_signal_run_card(args.mg5_root)

    summary = prepare_mg5_grid(
        process=args.process,
        process_dir=process_dir,
        reference_grid_manifest=args.reference_grid_manifest,
        events=args.events,
        signal_run_card=signal_run_card,
        deck_dir=args.deck_dir,
        manifest_file=args.manifest,
        accuracy=args.accuracy,
        points=args.points,
        iterations=args.iterations,
        seed=args.seed,
        cores=args.cores,
        c3_only=args.c3_only,
        c3_only_d4=args.c3_only_d4,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    print("MG5 deck:", summary["deck"])
    print("MG5 manifest:", summary["manifest"])
    print("Scheduled points:", summary["scheduled_points"], "of", summary["total_points"])
    print("MG5 cores:", summary["cores"])
    if summary["c3_only"]:
        print("c3-only projection: yes, d4 =", summary["c3_only_d4"])
    print("Run status:", summary["run_status"])


if __name__ == "__main__":
    main()
