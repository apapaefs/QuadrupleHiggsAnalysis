"""Run the full forced-splitting Herwig chain from one MG LHE file."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess

from .herwig_cards import PROCESS_CONFIGS, stage1_lhewriter_card, stage2_hwsim_card
from .lhe_validation import normalize_lhe_file_process_ids
from .lhe_weights import apply_weights, verify_weighted_lhe


@dataclass
class ChainConfig(object):
    process: str
    input_lhe: Path
    workdir: Path
    events: int
    probe_trials: int = 0
    run_name: str = None
    stage2_events: int = None
    herwig: str = "Herwig"
    output_location: str = "events"
    seed_stage1: int = 31122002
    seed_stage2: int = 89968250
    input_xsec_error: float = None
    allow_zero_probe_successes: bool = False
    overwrite: bool = False
    dry_run: bool = False


def _default_runner(command, cwd):
    subprocess.run(command, cwd=str(cwd), check=True)


def _write_text(path, text, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("%s already exists; pass --overwrite to replace it" % path)
    path.write_text(text)


def _run_herwig(herwig, subcommand, filename, cwd, runner, dry_run):
    command = [herwig, subcommand, filename]
    if dry_run:
        return command
    runner(command, cwd)
    return command


def _relative_output_location(path_text):
    path_text = str(path_text)
    return path_text.rstrip("/")


def run_chain(config, runner=None):
    """Run Stage 1, optional reweighting, and Stage 2 for one LHE input."""
    if runner is None:
        runner = _default_runner
    if config.process not in PROCESS_CONFIGS:
        raise ValueError("Unknown process %r" % config.process)

    input_lhe = Path(config.input_lhe)
    if not input_lhe.exists():
        raise FileNotFoundError("Input LHE file does not exist: %s" % input_lhe)
    input_lhe = input_lhe.resolve()

    run_name = config.run_name or config.process
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    output_location = _relative_output_location(config.output_location)
    (workdir / output_location).mkdir(parents=True, exist_ok=True)

    stage1_name = "%s_stage1" % run_name
    stage2_name = "%s_stage2" % run_name
    stage1_card = workdir / ("%s.in" % stage1_name)
    stage1_run = workdir / ("%s.run" % stage1_name)
    stage1_lhe = workdir / ("%s.lhe" % stage1_name)
    correction_file = workdir / ("%s.force_split.weights" % stage1_name)
    weighted_lhe = workdir / ("%s.weighted.lhe" % stage1_name)
    stage2_card = workdir / ("%s.in" % stage2_name)
    stage2_run = workdir / ("%s.run" % stage2_name)
    stage2_root = workdir / output_location / ("%s.root" % stage2_name)
    summary_file = workdir / ("%s_summary.json" % run_name)

    stage1_text = stage1_lhewriter_card(
        PROCESS_CONFIGS[config.process],
        input_lhe=input_lhe,
        output_prefix=stage1_name,
        events=config.events,
        seed=config.seed_stage1,
        probe_trials=config.probe_trials,
        correction_file=correction_file.name,
    )
    _write_text(stage1_card, stage1_text, config.overwrite)

    commands = []
    commands.append(_run_herwig(config.herwig, "read", stage1_card.name, workdir, runner, config.dry_run))
    commands.append(_run_herwig(config.herwig, "run", stage1_run.name, workdir, runner, config.dry_run))

    stage2_lhe = stage1_lhe
    normalization_message = None
    weight_check = None

    if not config.dry_run:
        if not stage1_lhe.exists():
            raise FileNotFoundError("Stage-1 LHE was not produced: %s" % stage1_lhe)
        _, normalization_message = normalize_lhe_file_process_ids(stage1_lhe)

        if config.probe_trials > 0:
            if not correction_file.exists():
                raise FileNotFoundError("ProbeTrials was nonzero but no correction sidecar exists: %s" % correction_file)
            apply_weights(
                stage1_lhe,
                correction_file,
                weighted_lhe,
                input_xsec_error=config.input_xsec_error,
            )
            weight_check = verify_weighted_lhe(stage1_lhe, correction_file, weighted_lhe)
            if not weight_check["ok"]:
                raise RuntimeError("Weighted LHE verification failed: %s" % weight_check)
            if weight_check["zero_success_rows"] and not config.allow_zero_probe_successes:
                raise RuntimeError(
                    "Correction sidecar has %d rows with probe_successes = 0; "
                    "increase --probe-trials or pass --allow-zero-probe-successes for a diagnostic run"
                    % weight_check["zero_success_rows"]
                )
            stage2_lhe = weighted_lhe
    elif config.probe_trials > 0:
        stage2_lhe = weighted_lhe

    stage2_events = config.stage2_events if config.stage2_events is not None else config.events
    stage2_text = stage2_hwsim_card(
        input_lhe=stage2_lhe.name,
        output_location=output_location,
        events=stage2_events,
        run_name=stage2_name,
        seed=config.seed_stage2,
    )
    _write_text(stage2_card, stage2_text, config.overwrite)

    commands.append(_run_herwig(config.herwig, "read", stage2_card.name, workdir, runner, config.dry_run))
    commands.append(_run_herwig(config.herwig, "run", stage2_run.name, workdir, runner, config.dry_run))

    if not config.dry_run and not stage2_root.exists():
        raise FileNotFoundError("Stage-2 ROOT output was not produced: %s" % stage2_root)

    summary = {
        "process": config.process,
        "input_lhe": str(input_lhe),
        "workdir": str(workdir),
        "events": int(config.events),
        "stage2_events": int(stage2_events),
        "probe_trials": int(config.probe_trials),
        "run_name": run_name,
        "stage1_card": stage1_card.name,
        "stage1_run": stage1_run.name,
        "stage1_lhe": stage1_lhe.name,
        "normalization_message": normalization_message,
        "correction_file": correction_file.name if config.probe_trials > 0 else "",
        "weighted_lhe": weighted_lhe.name if config.probe_trials > 0 else "",
        "stage2_lhe": stage2_lhe.name,
        "stage2_card": stage2_card.name,
        "stage2_run": stage2_run.name,
        "stage2_root": str(stage2_root),
        "weight_check": weight_check,
        "commands": [" ".join(command) for command in commands],
        "dry_run": bool(config.dry_run),
    }
    _write_text(summary_file, json.dumps(summary, indent=2, sort_keys=True) + "\n", True)
    return summary


def _default_workdir(run_name):
    return Path.cwd() / ("%s_forced_splitting" % run_name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("process", choices=sorted(PROCESS_CONFIGS))
    parser.add_argument("--input-lhe", type=Path, default=Path("unweighted_events.lhe.gz"))
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--probe-trials", type=int, default=0)
    parser.add_argument("--run-name")
    parser.add_argument("--stage2-events", type=int)
    parser.add_argument("--herwig", default="Herwig")
    parser.add_argument("--output-location", default="events")
    parser.add_argument("--seed-stage1", type=int, default=31122002)
    parser.add_argument("--seed-stage2", type=int, default=89968250)
    parser.add_argument("--input-xsec-error", type=float)
    parser.add_argument("--allow-zero-probe-successes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_name = args.run_name or args.process
    workdir = args.workdir or _default_workdir(run_name)
    summary = run_chain(
        ChainConfig(
            process=args.process,
            input_lhe=args.input_lhe,
            workdir=workdir,
            events=args.events,
            probe_trials=args.probe_trials,
            run_name=run_name,
            stage2_events=args.stage2_events,
            herwig=args.herwig,
            output_location=args.output_location,
            seed_stage1=args.seed_stage1,
            seed_stage2=args.seed_stage2,
            input_xsec_error=args.input_xsec_error,
            allow_zero_probe_successes=args.allow_zero_probe_successes,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    )

    print("Forced-splitting chain summary:", Path(summary["workdir"]) / ("%s_summary.json" % summary["run_name"]))
    print("Stage-2 input LHE:", summary["stage2_lhe"])
    print("Stage-2 ROOT:", summary["stage2_root"])
    if summary["weight_check"] is not None:
        check = summary["weight_check"]
        print(
            "Weight check: rows={rows} zero_success_rows={zero} mean_p_hat={mean:.9g} "
            "weighted_mean_xwgtup={xwgt:.9g} ok={ok}".format(
                rows=check["correction_rows"],
                zero=check["zero_success_rows"],
                mean=check["mean_p_hat"],
                xwgt=check["weighted_mean_xwgtup"],
                ok=check["ok"],
            )
        )


if __name__ == "__main__":
    main()
