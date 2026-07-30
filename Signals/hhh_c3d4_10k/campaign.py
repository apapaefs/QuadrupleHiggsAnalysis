#!/usr/bin/env python3
"""Operate the 153-point inclusive gg -> hhh MG5 and Herwig campaign."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import gzip
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO_DIR = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(REPO_DIR))

from ForcedSplitting.mg5_grid import prepare_mg5_grid  # noqa: E402


EXPECTED_POINTS = 153
PRODUCTION_EVENTS = 10_000
PRODUCTION_CPUS = 64
ANALYSIS_ID = "hhh-hhhh-ge6b-pairing-v3"
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
INTEGRATED_WEIGHT = re.compile(
    rf"Integrated weight\s*\(pb\)\s*:\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)
C3_VALUE = re.compile(
    rf"^\s*(?:\d+\s+)*({FLOAT_PATTERN})\s+#\s*c3\s*$",
    re.IGNORECASE,
)
D4_VALUE = re.compile(
    rf"^\s*(?:\d+\s+)*({FLOAT_PATTERN})\s+#\s*d4\s*$",
    re.IGNORECASE,
)
HERWIG_COMPLETE_MARKER = "Number of events that pass basic cuts"


@dataclass(frozen=True)
class Point:
    index: int
    c3: str
    d4: str
    source: str
    seed: int

    @property
    def coordinate(self) -> tuple[float, float]:
        return (float(self.c3), float(self.d4))

    @property
    def run_name(self) -> str:
        return f"run_gg_hhh_5_{self.c3}_{self.d4}"


@dataclass(frozen=True)
class Paths:
    source_repo: Path
    mg5_process: Path
    points_file: Path
    herwig_dir: Path
    state_dir: Path
    smoke_dir: Path


def default_source_repo() -> Path:
    configured = os.environ.get("QHA_SOURCE_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    canonical = Path("/home/apapaefs/Projects/QuadrupleHiggsAnalysis")
    return canonical if canonical.is_dir() else REPO_DIR


def paths_from_args(args: argparse.Namespace) -> Paths:
    source_repo = (
        args.source_repo.expanduser().resolve()
        if args.source_repo
        else default_source_repo()
    )
    mg5_process = (
        args.mg5_process.expanduser().resolve()
        if args.mg5_process
        else source_repo / "MG5_aMC_v3_5_16" / "gg_hhh"
    )
    herwig_dir = (
        args.herwig_dir.expanduser().resolve()
        if args.herwig_dir
        else REPO_DIR / "HerwigSignalPoints" / "hhh_c3d4_10k"
    )
    return Paths(
        source_repo=source_repo,
        mg5_process=mg5_process,
        points_file=REPO_DIR
        / "Signals"
        / "c3d4_40k"
        / "metadata"
        / "points_153.csv",
        herwig_dir=herwig_dir,
        state_dir=herwig_dir / "state",
        smoke_dir=CAMPAIGN_DIR / "smoke",
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"campaign lock is already held: {path}") from error
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        yield


def require_cpu_budget(cpus: int) -> None:
    if cpus < 1:
        raise ValueError("CPU count must be positive")
    online = os.cpu_count() or 1
    if cpus > online:
        raise ValueError(
            f"refusing {cpus} logical CPUs on a host with {online} online CPUs"
        )


def load_points(path: Path) -> list[Point]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"index", "source", "c3", "d4", "seed"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(
            "authoritative point manifest is missing columns: "
            + ", ".join(sorted(missing))
        )
    points = [
        Point(
            index=int(row["index"]),
            source=row["source"],
            c3=row["c3"],
            d4=row["d4"],
            seed=int(row["seed"]),
        )
        for row in rows
    ]
    if len(points) != EXPECTED_POINTS:
        raise ValueError(
            f"authoritative manifest has {len(points)} points, "
            f"expected {EXPECTED_POINTS}"
        )
    if len({point.coordinate for point in points}) != len(points):
        raise ValueError("authoritative manifest contains duplicate coordinates")
    if len({point.seed for point in points}) != len(points):
        raise ValueError("authoritative manifest contains duplicate seeds")
    return points


def open_lhe(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def find_lhe(run_dir: Path) -> Path | None:
    for name in ("unweighted_events.lhe.gz", "unweighted_events.lhe"):
        candidate = run_dir / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def inspect_lhe(
    path: Path, expected_events: int, expected_c3: float, expected_d4: float
) -> dict[str, object]:
    event_count = 0
    integrated_weight_pb: float | None = None
    embedded_c3: float | None = None
    embedded_d4: float | None = None
    with open_lhe(path) as stream:
        for line in stream:
            if line.strip() == "<event>":
                event_count += 1
            if integrated_weight_pb is None:
                match = INTEGRATED_WEIGHT.search(line)
                if match:
                    integrated_weight_pb = float(match.group(1))
            if embedded_c3 is None:
                match = C3_VALUE.match(line)
                if match:
                    embedded_c3 = float(match.group(1))
            if embedded_d4 is None:
                match = D4_VALUE.match(line)
                if match:
                    embedded_d4 = float(match.group(1))
    issues: list[str] = []
    if event_count != expected_events:
        issues.append(f"contains {event_count} events, expected {expected_events}")
    if integrated_weight_pb is None or integrated_weight_pb <= 0.0:
        issues.append("has no positive Integrated weight (pb)")
    if embedded_c3 is None or abs(embedded_c3 - expected_c3) > 1.0e-9:
        issues.append(f"embedded c3={embedded_c3}, expected {expected_c3}")
    if embedded_d4 is None or abs(embedded_d4 - expected_d4) > 1.0e-9:
        issues.append(f"embedded d4={embedded_d4}, expected {expected_d4}")
    return {
        "event_count": event_count,
        "integrated_weight_pb": integrated_weight_pb,
        "embedded_c3": embedded_c3,
        "embedded_d4": embedded_d4,
        "status": "ok" if not issues else "invalid",
        "issues": issues,
    }


def mg5_status(paths: Paths, points: list[Point], deep: bool) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    complete = 0
    incomplete = 0
    invalid = 0
    for point in points:
        run_dir = paths.mg5_process / "Events" / point.run_name
        lhe = find_lhe(run_dir)
        if lhe is None:
            status = "incomplete" if run_dir.exists() else "missing"
            if status == "incomplete":
                incomplete += 1
            rows.append(
                {
                    "index": point.index,
                    "c3": point.c3,
                    "d4": point.d4,
                    "run_name": point.run_name,
                    "status": status,
                    "lhe_file": "",
                    "issues": [],
                }
            )
            continue
        inspection: dict[str, object] = {
            "status": "present",
            "issues": [],
        }
        if deep:
            inspection = inspect_lhe(
                lhe, PRODUCTION_EVENTS, float(point.c3), float(point.d4)
            )
        status = str(inspection["status"])
        if status in {"ok", "present"}:
            complete += 1
        else:
            invalid += 1
        rows.append(
            {
                "index": point.index,
                "c3": point.c3,
                "d4": point.d4,
                "run_name": point.run_name,
                "status": status,
                "lhe_file": str(lhe),
                "issues": inspection.get("issues", []),
            }
        )
    return {
        "process": "gg_hhh",
        "expected_points": len(points),
        "complete_points": complete,
        "missing_points": len(points) - complete - incomplete - invalid,
        "incomplete_points": incomplete,
        "invalid_points": invalid,
        "deep_validation": deep,
        "production_events_per_point": PRODUCTION_EVENTS,
        "mg5_process": str(paths.mg5_process),
        "points": rows,
    }


def assert_no_invalid_mg5_state(status: dict[str, object]) -> None:
    invalid = [
        row
        for row in status["points"]  # type: ignore[index]
        if row["status"] in {"invalid", "incomplete"}
    ]
    if invalid:
        preview = ", ".join(str(row["run_name"]) for row in invalid[:5])
        raise RuntimeError(
            f"{len(invalid)} existing MG5 run directories are incomplete or "
            f"invalid ({preview}); archive or repair them before restarting"
        )


def run_mg5(paths: Paths, points: list[Point], cpus: int) -> None:
    require_cpu_budget(cpus)
    if not (paths.mg5_process / "bin" / "madevent").is_file():
        raise FileNotFoundError(
            f"missing MadEvent process executable: "
            f"{paths.mg5_process / 'bin' / 'madevent'}"
        )
    with exclusive_lock(paths.state_dir / "mg5.lock"):
        current = mg5_status(paths, points, deep=True)
        assert_no_invalid_mg5_state(current)
        summary = prepare_mg5_grid(
            process="gg_hhh",
            process_dir=paths.mg5_process,
            reference_grid_manifest=paths.points_file,
            events=PRODUCTION_EVENTS,
            signal_run_card=paths.mg5_process / "Cards" / "run_card.dat",
            deck_dir=paths.state_dir / "mg5",
            manifest_file=paths.state_dir / "mg5" / "manifest.csv",
            cores=cpus,
        )
        atomic_write_json(paths.state_dir / "mg5" / "launch_summary.json", summary)
        final = mg5_status(paths, points, deep=True)
        atomic_write_json(paths.state_dir / "mg5" / "status.json", final)
        if final["complete_points"] != EXPECTED_POINTS:
            raise RuntimeError(
                f"MG5 returned without a complete grid: "
                f"{final['complete_points']}/{EXPECTED_POINTS}"
            )


def replace_required(
    text: str, pattern: str, replacement: str, description: str
) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(
            f"expected one {description} line in Herwig template, found {count}"
        )
    return updated


def herwig_seed(point_seed: int) -> int:
    return ((int(point_seed) + 31_122_002) % 900_000_000) + 1


def render_herwig_card(
    template: str,
    lhe_file: Path,
    run_name: str,
    events: int,
    seed: int,
) -> str:
    text = replace_required(
        template,
        r"^set\s+theLHReader:FileName\s+.*$",
        f"set theLHReader:FileName {lhe_file}",
        "LHE filename",
    )
    text = replace_required(
        text,
        r"^set\s+theGenerator:NumberOfEvents\s+\d+\s*$",
        f"set theGenerator:NumberOfEvents {events}",
        "event count",
    )
    text = replace_required(
        text,
        r"^set\s+theGenerator:RandomNumberGenerator:Seed\s+\d+\s*$",
        f"set theGenerator:RandomNumberGenerator:Seed {seed}",
        "random seed",
    )
    text = replace_required(
        text,
        r"^set\s+/Herwig/Analysis/HwSim:OutputLocation\s+.*$",
        "set /Herwig/Analysis/HwSim:OutputLocation events/",
        "HwSim output location",
    )
    text = replace_required(
        text,
        r"^saverun\s+\S+\s+theGenerator\s*$",
        f"saverun {run_name} theGenerator",
        "saverun",
    )
    return text


def prepare_herwig_cards(
    paths: Paths, points: list[Point]
) -> tuple[Path, list[dict[str, object]]]:
    template_path = REPO_DIR / "Signals" / "HW-gg_hhhh_SM.in"
    template = template_path.read_text()
    paths.herwig_dir.mkdir(parents=True, exist_ok=True)
    (paths.herwig_dir / "events").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cards: list[Path] = []
    for point in points:
        lhe = find_lhe(paths.mg5_process / "Events" / point.run_name)
        if lhe is None:
            raise FileNotFoundError(f"missing production LHE for {point.run_name}")
        herwig_run_name = f"HW-{point.run_name}"
        card = paths.herwig_dir / f"{herwig_run_name}.in"
        rendered = render_herwig_card(
            template=template,
            lhe_file=lhe.resolve(),
            run_name=herwig_run_name,
            events=PRODUCTION_EVENTS,
            seed=herwig_seed(point.seed),
        )
        if not card.exists() or card.read_text() != rendered:
            atomic_write_text(card, rendered)
        cards.append(card)
        rows.append(
            {
                "index": point.index,
                "c3": point.c3,
                "d4": point.d4,
                "run_name": point.run_name,
                "seed": point.seed,
                "herwig_seed": herwig_seed(point.seed),
                "lhe_file": str(lhe.resolve()),
                "herwig_input": str(card.resolve()),
                "root_file": str(
                    (paths.herwig_dir / "events" / f"{herwig_run_name}.root")
                    .resolve()
                ),
            }
        )
    manifest = paths.herwig_dir / "herwig_inputs_manifest.csv"
    atomic_write_csv(manifest, rows, list(rows[0]))
    input_list = paths.herwig_dir / "herwig_inputs_to_run.txt"
    atomic_write_text(
        input_list, "".join(f"{card.resolve()}\n" for card in cards)
    )
    return input_list, rows


def herwig_output_complete(
    root_file: Path, run_log: Path, input_card: Path
) -> bool:
    if not root_file.is_file() or root_file.stat().st_size <= 0:
        return False
    if not run_log.is_file():
        return False
    if (
        root_file.stat().st_mtime < input_card.stat().st_mtime
        or run_log.stat().st_mtime < input_card.stat().st_mtime
    ):
        return False
    return HERWIG_COMPLETE_MARKER in run_log.read_text(errors="replace")


def herwig_status(
    paths: Paths, points: list[Point], prepare: bool
) -> dict[str, object]:
    if prepare:
        _, rows = prepare_herwig_cards(paths, points)
    else:
        rows = []
        for point in points:
            herwig_run_name = f"HW-{point.run_name}"
            rows.append(
                {
                    "index": point.index,
                    "c3": point.c3,
                    "d4": point.d4,
                    "run_name": point.run_name,
                    "herwig_input": str(
                        paths.herwig_dir / f"{herwig_run_name}.in"
                    ),
                    "root_file": str(
                        paths.herwig_dir / "events" / f"{herwig_run_name}.root"
                    ),
                }
            )
    complete = 0
    details: list[dict[str, object]] = []
    for row in rows:
        input_card = Path(str(row["herwig_input"]))
        root_file = Path(str(row["root_file"]))
        run_log = (
            paths.herwig_dir
            / "logs"
            / f"{input_card.stem}.run.log"
        )
        is_complete = (
            input_card.is_file()
            and herwig_output_complete(root_file, run_log, input_card)
        )
        complete += int(is_complete)
        details.append(
            {
                "index": row["index"],
                "c3": row["c3"],
                "d4": row["d4"],
                "run_name": row["run_name"],
                "status": "complete" if is_complete else "missing",
                "root_file": str(root_file),
                "run_log": str(run_log),
            }
        )
    return {
        "process": "gg_hhh",
        "expected_points": len(points),
        "complete_points": complete,
        "missing_points": len(points) - complete,
        "herwig_dir": str(paths.herwig_dir),
        "points": details,
    }


def run_herwig(paths: Paths, points: list[Point], cpus: int) -> None:
    require_cpu_budget(cpus)
    mg5 = mg5_status(paths, points, deep=True)
    if mg5["complete_points"] != EXPECTED_POINTS:
        raise RuntimeError(
            f"Herwig requires a complete MG5 grid; found "
            f"{mg5['complete_points']}/{EXPECTED_POINTS}"
        )
    with exclusive_lock(paths.state_dir / "herwig.lock"):
        input_list, _ = prepare_herwig_cards(paths, points)
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = "1"
        command = [
            sys.executable,
            str(REPO_DIR / "run_herwig_signal_inputs.py"),
            "--list",
            str(input_list),
            "--jobs",
            str(cpus),
        ]
        subprocess.run(
            command,
            cwd=REPO_DIR,
            env=environment,
            check=True,
        )
        status = herwig_status(paths, points, prepare=False)
        atomic_write_json(paths.state_dir / "herwig" / "status.json", status)
        if status["complete_points"] != EXPECTED_POINTS:
            raise RuntimeError(
                f"Herwig returned without a complete grid: "
                f"{status['complete_points']}/{EXPECTED_POINTS}"
            )


def smoke_point() -> Point:
    return Point(
        index=1,
        source="smoke",
        c3="0.0",
        d4="0.0",
        seed=246_813_579,
    )


def smoke_run_name() -> str:
    return "run_gg_hhh_smoke_ge6b_0.0_0.0"


def write_smoke_manifest(path: Path, point: Point) -> None:
    rows = [
        {
            "index": 1,
            "source": "smoke",
            "c3": point.c3,
            "d4": point.d4,
            "run_group": "smoke_ge6b",
            "run_name": smoke_run_name(),
            "target_events": 100,
            "seed": point.seed,
        }
    ]
    atomic_write_csv(path, rows, list(rows[0]))


def prepare_smoke_herwig_card(
    paths: Paths, point: Point, lhe: Path
) -> Path:
    smoke_herwig = paths.smoke_dir / "herwig"
    smoke_herwig.mkdir(parents=True, exist_ok=True)
    (smoke_herwig / "events").mkdir(parents=True, exist_ok=True)
    template = (REPO_DIR / "Signals" / "HW-gg_hhhh_SM.in").read_text()
    herwig_run_name = f"HW-{smoke_run_name()}"
    card = smoke_herwig / f"{herwig_run_name}.in"
    atomic_write_text(
        card,
        render_herwig_card(
            template,
            lhe.resolve(),
            herwig_run_name,
            events=100,
            seed=herwig_seed(point.seed),
        ),
    )
    input_list = smoke_herwig / "herwig_inputs.txt"
    atomic_write_text(input_list, f"{card.resolve()}\n")
    return input_list


def run_smoke_samples(paths: Paths, cpus: int) -> Path:
    require_cpu_budget(cpus)
    point = smoke_point()
    paths.smoke_dir.mkdir(parents=True, exist_ok=True)
    manifest = paths.smoke_dir / "point.csv"
    write_smoke_manifest(manifest, point)
    run_dir = paths.mg5_process / "Events" / smoke_run_name()
    existing_lhe = find_lhe(run_dir)
    if existing_lhe is not None:
        inspection = inspect_lhe(existing_lhe, 100, 0.0, 0.0)
        if inspection["status"] != "ok":
            raise RuntimeError(
                f"existing smoke LHE is invalid: {inspection['issues']}"
            )
    elif run_dir.exists():
        raise RuntimeError(
            f"incomplete smoke MG5 directory exists: {run_dir}; "
            "archive it before rerunning"
        )
    else:
        with exclusive_lock(paths.smoke_dir / "mg5.lock"):
            summary = prepare_mg5_grid(
                process="gg_hhh",
                process_dir=paths.mg5_process,
                reference_grid_manifest=manifest,
                events=100,
                signal_run_card=paths.mg5_process / "Cards" / "run_card.dat",
                deck_dir=paths.smoke_dir / "mg5",
                manifest_file=paths.smoke_dir / "mg5" / "manifest.csv",
                cores=cpus,
            )
            atomic_write_json(
                paths.smoke_dir / "mg5" / "launch_summary.json", summary
            )
        existing_lhe = find_lhe(run_dir)
        if existing_lhe is None:
            raise RuntimeError("smoke MadEvent run did not produce an LHE")
        inspection = inspect_lhe(existing_lhe, 100, 0.0, 0.0)
        if inspection["status"] != "ok":
            raise RuntimeError(
                f"new smoke LHE failed validation: {inspection['issues']}"
            )

    input_list = prepare_smoke_herwig_card(paths, point, existing_lhe)
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    subprocess.run(
        [
            sys.executable,
            str(REPO_DIR / "run_herwig_signal_inputs.py"),
            "--list",
            str(input_list),
            "--jobs",
            "1",
        ],
        cwd=REPO_DIR,
        env=environment,
        check=True,
    )
    root_file = (
        paths.smoke_dir
        / "herwig"
        / "events"
        / f"HW-{smoke_run_name()}.root"
    )
    run_log = (
        paths.smoke_dir
        / "herwig"
        / "logs"
        / f"HW-{smoke_run_name()}.run.log"
    )
    card = (
        paths.smoke_dir / "herwig" / f"HW-{smoke_run_name()}.in"
    )
    if not herwig_output_complete(root_file, run_log, card):
        raise RuntimeError("smoke Herwig output is not complete")
    payload = {
        "status": "ok",
        "events": 100,
        "c3": 0.0,
        "d4": 0.0,
        "mg5_lhe": str(existing_lhe.resolve()),
        "hhh_root": str(root_file.resolve()),
        "hhh_herwig_out": str(
            (paths.smoke_dir / "herwig" / f"HW-{smoke_run_name()}.out")
            .resolve()
        ),
    }
    output = paths.smoke_dir / "smoke_samples.json"
    atomic_write_json(output, payload)
    return output


def hhhbb_inventory(paths: Paths) -> dict[str, object]:
    workdir = (
        paths.source_repo
        / "HerwigForcedSplitting"
        / "gg_hhhg_c3d4_10k_hhhbb_153"
    )
    consolidated = workdir / "consolidated_sources.json"
    if not consolidated.is_file():
        return {
            "status": "missing",
            "workdir": str(workdir),
            "points": 0,
        }
    payload = json.loads(consolidated.read_text())
    point_count = int(payload.get("point_count", 0))
    raw_roots = list((workdir / "events").glob("*_hhhbb_stage2.root"))
    return {
        "status": (
            "complete"
            if point_count == EXPECTED_POINTS
            and len(raw_roots) == EXPECTED_POINTS
            else "incomplete"
        ),
        "workdir": str(workdir),
        "points": point_count,
        "raw_roots": len(raw_roots),
    }


def print_summary(payload: dict[str, object], include_points: bool) -> None:
    if not include_points:
        payload = {key: value for key, value in payload.items() if key != "points"}
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "run-mg5",
            "status-mg5",
            "run-herwig",
            "status-herwig",
            "status",
            "smoke-samples",
        ),
    )
    parser.add_argument("--cpus", type=int, default=PRODUCTION_CPUS)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--mg5-process", type=Path)
    parser.add_argument("--herwig-dir", type=Path)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Count and validate every event when reporting MG5 status.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-point status rows in JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = paths_from_args(args)
    points = load_points(paths.points_file)
    if args.command == "run-mg5":
        run_mg5(paths, points, args.cpus)
        print_summary(mg5_status(paths, points, deep=True), args.details)
    elif args.command == "status-mg5":
        print_summary(mg5_status(paths, points, deep=args.deep), args.details)
    elif args.command == "run-herwig":
        run_herwig(paths, points, args.cpus)
        print_summary(
            herwig_status(paths, points, prepare=False), args.details
        )
    elif args.command == "status-herwig":
        print_summary(
            herwig_status(paths, points, prepare=False), args.details
        )
    elif args.command == "status":
        print(
            json.dumps(
                {
                    "mg5": {
                        key: value
                        for key, value in mg5_status(
                            paths, points, deep=args.deep
                        ).items()
                        if key != "points"
                    },
                    "herwig": {
                        key: value
                        for key, value in herwig_status(
                            paths, points, prepare=False
                        ).items()
                        if key != "points"
                    },
                    "hhhbb": hhhbb_inventory(paths),
                    "default_logical_cpus": PRODUCTION_CPUS,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "smoke-samples":
        print(run_smoke_samples(paths, args.cpus))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
