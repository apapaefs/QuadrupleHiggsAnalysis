#!/usr/bin/env python3
"""Run and fit the three 14 TeV ``gg -> hh`` coupling points.

The three points use ``c3 = -2, -1, 0``, corresponding to
``kappa_lambda = -1, 0, +1``.  Only process initialization and integration
are run: the driver always invokes Sherpa with ``-e 0`` and never generates
production events.

Each completed point has an immutable ``integration_result.json`` marker.
Matching completed points are reused on a later invocation; a card, parameter
card, or model-library hash mismatch is rejected instead of overwriting the
existing grids.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "Examples" / "GluonFusion_UFO_HEFT_GG_HH_LHE"
DEFAULT_WORK_DIR = ROOT / "runs" / "gg_hh_c3_fit_14tev"
PARAM_CARD = "param_heft_c3d4_sherpa.dat"
RESULT_FILENAME = "integration_result.json"
POINTS_CSV = "gg_hh_c3_points.csv"
FIT_JSON = "gg_hh_c3_fit.json"
INTEGRATION_ERROR_TARGET = 0.005
INTEGRATION_ERROR_TOLERANCE = 1.0e-6
POINTS = (
    (-2.0, -1.0, "c3_m2", 1000003),
    (-1.0, 0.0, "c3_m1", 2000003),
    (0.0, 1.0, "c3_0", 3000003),
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
SHERPA_XSEC_RE = re.compile(
    r":\s*(%s)\s*pb\s*\+-\s*\(\s*(%s)\s*pb" % (NUMBER_RE, NUMBER_RE)
)
TAGS_RE = re.compile(r"^TAGS:\s*\{C3:\s*[^}]+\}\s*$", re.MULTILINE)
RANDOM_SEED_RE = re.compile(r"^RANDOM_SEED:\s*\d+\s*$", re.MULTILINE)


class FitDriverError(RuntimeError):
    """Raised for an unsafe resume or an invalid Sherpa result."""


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix="." + path.name + ".",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def c3_text(c3: float) -> str:
    if c3.is_integer():
        return str(int(c3))
    return repr(c3)


def render_card(source_text: str, c3: float, random_seed: int) -> str:
    if random_seed < 1:
        raise FitDriverError("random seed must be positive")
    replacement = "TAGS: {C3: %s}" % c3_text(c3)
    if len(TAGS_RE.findall(source_text)) != 1:
        raise FitDriverError("example Sherpa.yaml must contain exactly one TAGS: {C3: ...} line")
    rendered = TAGS_RE.sub(replacement, source_text, count=1)
    seed_line = "RANDOM_SEED: %d" % random_seed
    seed_matches = RANDOM_SEED_RE.findall(rendered)
    if len(seed_matches) > 1:
        raise FitDriverError("example Sherpa.yaml contains more than one RANDOM_SEED line")
    if seed_matches:
        return RANDOM_SEED_RE.sub(seed_line, rendered, count=1)
    return rendered.replace(replacement, replacement + "\n" + seed_line, 1)


def parse_sherpa_cross_sections(text: str) -> List[Tuple[float, float]]:
    clean = ANSI_RE.sub("", text)
    values = []
    for line in clean.splitlines():
        match = SHERPA_XSEC_RE.search(line)
        if match:
            values.append((float(match.group(1)), float(match.group(2))))
    return values


def parse_sherpa_cross_section_log(path: Path) -> Tuple[float, float]:
    values = parse_sherpa_cross_sections(path.read_text(encoding="utf-8", errors="replace"))
    if not values:
        raise FitDriverError("no Sherpa cross-section line found in %s" % path)
    cross_section, uncertainty = values[-1]
    if not math.isfinite(cross_section) or cross_section <= 0.0:
        raise FitDriverError("Sherpa returned a non-positive or non-finite cross section in %s" % path)
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise FitDriverError("Sherpa returned an invalid uncertainty in %s" % path)
    validate_integration_precision(cross_section, uncertainty, path)
    return cross_section, uncertainty


def validate_integration_precision(
    cross_section: float,
    uncertainty: float,
    source: Path,
) -> None:
    relative = uncertainty / cross_section
    allowed = INTEGRATION_ERROR_TARGET * (1.0 + INTEGRATION_ERROR_TOLERANCE)
    if relative > allowed:
        raise FitDriverError(
            "Sherpa integration in %s has relative uncertainty %.8g, above the %.8g target"
            % (source, relative, INTEGRATION_ERROR_TARGET)
        )


def quadratic_coefficients(
    sigma_minus: float,
    sigma_zero: float,
    sigma_plus: float,
) -> Dict[str, float]:
    return {
        "A": 0.5 * (sigma_plus + sigma_minus) - sigma_zero,
        "B": 0.5 * (sigma_plus - sigma_minus),
        "C": sigma_zero,
    }


def coefficient_covariance(
    error_minus: float,
    error_zero: float,
    error_plus: float,
) -> List[List[float]]:
    """Propagate independent point uncertainties to the ``(A, B, C)`` basis."""

    transform = (
        (0.5, -1.0, 0.5),
        (-0.5, 0.0, 0.5),
        (0.0, 1.0, 0.0),
    )
    variances = (error_minus ** 2, error_zero ** 2, error_plus ** 2)
    return [
        [
            sum(transform[row][point] * variances[point] * transform[col][point] for point in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def next_attempt_log(log_dir: Path, stem: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = log_dir / ("%s.attempt%02d.log" % (stem, index))
        if not candidate.exists():
            return candidate
        index += 1


def run_logged(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    printable = " ".join(command)
    print("[%s] %s" % (cwd, printable), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# started_utc: %s\n" % now_utc())
        log.write("# cwd: %s\n" % cwd)
        log.write("# command: %s\n" % printable)
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
        log.write("# finished_utc: %s\n" % now_utc())
        log.write("# return_code: %d\n" % return_code)
    if return_code != 0:
        raise FitDriverError("command failed with status %d; see %s" % (return_code, log_path))


def find_model_library(sherpa: str) -> Path:
    explicit = os.environ.get("SHERPA_UFO_LIBRARY")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FitDriverError("SHERPA_UFO_LIBRARY is not a file: %s" % candidate)

    executable = shutil.which(sherpa)
    if executable:
        prefix = Path(executable).resolve().parent.parent
        candidates = (
            prefix / "lib64" / "SHERPA-MC" / "libSherpaheft_c3d4_sherpa.so",
            prefix / "lib" / "SHERPA-MC" / "libSherpaheft_c3d4_sherpa.so",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FitDriverError(
        "could not locate libSherpaheft_c3d4_sherpa.so; pass --model-library or set SHERPA_UFO_LIBRARY"
    )


def point_expected_hashes(point_dir: Path, model_library: Path) -> Dict[str, str]:
    return {
        "sherpa_card_sha256": sha256_file(point_dir / "Sherpa.yaml"),
        "parameter_card_sha256": sha256_file(point_dir / PARAM_CARD),
        "model_library_sha256": sha256_file(model_library),
    }


def validate_completed_result(
    result_path: Path,
    c3: float,
    kappa_lambda: float,
    random_seed: int,
    expected_hashes: Mapping[str, str],
    ranks: int,
    sherpa: str,
    mpirun: str,
) -> Dict[str, object]:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FitDriverError("could not read completed result %s: %s" % (result_path, error))
    if payload.get("c3") != c3 or payload.get("kappa_lambda") != kappa_lambda:
        raise FitDriverError("completed result has the wrong coupling point: %s" % result_path)
    expected_execution = {
        "random_seed": random_seed,
        "mpi_ranks": ranks,
        "sherpa_executable": sherpa,
        "mpirun_executable": mpirun,
    }
    execution_mismatches = [
        key for key, expected in expected_execution.items() if payload.get(key) != expected
    ]
    if execution_mismatches:
        raise FitDriverError(
            "refusing to reuse completed grids in %s; execution metadata mismatch for %s"
            % (result_path.parent, ", ".join(execution_mismatches))
        )
    hashes = payload.get("hashes")
    if not isinstance(hashes, dict):
        raise FitDriverError("completed result has no hash record: %s" % result_path)
    mismatches = [
        key for key, expected in expected_hashes.items() if hashes.get(key) != expected
    ]
    if mismatches:
        raise FitDriverError(
            "refusing to overwrite completed grids in %s; hash mismatch for %s"
            % (result_path.parent, ", ".join(mismatches))
        )
    cross_section = payload.get("cross_section_pb")
    uncertainty = payload.get("uncertainty_pb")
    if not isinstance(cross_section, (int, float)) or not math.isfinite(cross_section) or cross_section <= 0:
        raise FitDriverError("completed result has an invalid cross section: %s" % result_path)
    if not isinstance(uncertainty, (int, float)) or not math.isfinite(uncertainty) or uncertainty < 0:
        raise FitDriverError("completed result has an invalid uncertainty: %s" % result_path)
    validate_integration_precision(float(cross_section), float(uncertainty), result_path)
    return payload


def prepare_point(point_dir: Path, c3: float, random_seed: int) -> None:
    source_card = EXAMPLE_DIR / "Sherpa.yaml"
    source_param = EXAMPLE_DIR / PARAM_CARD
    expected_card = render_card(source_card.read_text(encoding="utf-8"), c3, random_seed)
    point_dir.mkdir(parents=True, exist_ok=True)

    card = point_dir / "Sherpa.yaml"
    param = point_dir / PARAM_CARD
    if card.exists() and card.read_text(encoding="utf-8") != expected_card:
        raise FitDriverError("refusing to replace an existing modified card: %s" % card)
    if param.exists() and param.read_bytes() != source_param.read_bytes():
        raise FitDriverError("refusing to replace an existing modified parameter card: %s" % param)
    if not card.exists():
        card.write_text(expected_card, encoding="utf-8")
    if not param.exists():
        shutil.copy2(str(source_param), str(param))


def command_record(command: Sequence[str]) -> List[str]:
    return [str(item) for item in command]


def run_point(
    work_dir: Path,
    c3: float,
    kappa_lambda: float,
    label: str,
    random_seed: int,
    sherpa: str,
    mpirun: str,
    ranks: int,
    model_library: Path,
    provenance: Optional[Path],
) -> Dict[str, object]:
    point_dir = work_dir / label
    prepare_point(point_dir, c3, random_seed)
    hashes = point_expected_hashes(point_dir, model_library)
    if provenance is not None:
        hashes["model_provenance_sha256"] = sha256_file(provenance)

    result_path = point_dir / RESULT_FILENAME
    if result_path.exists():
        result = validate_completed_result(
            result_path,
            c3,
            kappa_lambda,
            random_seed,
            hashes,
            ranks,
            sherpa,
            mpirun,
        )
        print("Reusing completed %s (c3=%s)" % (point_dir, c3_text(c3)), flush=True)
        return result

    initialize_command = [sherpa, "-I", "Sherpa.yaml"]
    integration_command = [
        mpirun,
        "--use-hwthread-cpus",
        "-np",
        str(ranks),
        "--bind-to",
        "hwthread",
        "--map-by",
        "hwthread",
        sherpa,
        "-e",
        "0",
        "Sherpa.yaml",
    ]
    logs = point_dir / "logs"
    initialize_log = None
    if not (point_dir / "Process").exists() and not (point_dir / "Process.zip").exists():
        initialize_log = next_attempt_log(logs, "initialize")
        run_logged(initialize_command, point_dir, initialize_log)
    else:
        print("Reusing initialized process in %s" % point_dir, flush=True)

    integration_log = next_attempt_log(logs, "integrate")
    run_logged(integration_command, point_dir, integration_log)
    cross_section, uncertainty = parse_sherpa_cross_section_log(integration_log)

    result = {
        "schema_version": 1,
        "completed_utc": now_utc(),
        "c3": c3,
        "kappa_lambda": kappa_lambda,
        "random_seed": random_seed,
        "mpi_ranks": ranks,
        "sherpa_executable": sherpa,
        "mpirun_executable": mpirun,
        "cross_section_pb": cross_section,
        "uncertainty_pb": uncertainty,
        "relative_uncertainty": uncertainty / cross_section,
        "point_directory": str(point_dir.resolve()),
        "commands": {
            "initialize": command_record(initialize_command),
            "integrate": command_record(integration_command),
        },
        "logs": {
            "initialize": str(initialize_log.resolve()) if initialize_log else None,
            "integrate": str(integration_log.resolve()),
        },
        "hashes": hashes,
        "model_library": str(model_library),
        "model_provenance": str(provenance) if provenance else None,
    }
    atomic_write_json(result_path, result)
    return result


def ordered_results(results: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    by_kappa = {float(item["kappa_lambda"]): item for item in results}
    return [by_kappa[-1.0], by_kappa[0.0], by_kappa[1.0]]


def write_points_csv(work_dir: Path, results: Iterable[Mapping[str, object]]) -> None:
    rows = sorted(results, key=lambda item: float(item["kappa_lambda"]))
    path = work_dir / POINTS_CSV
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=str(work_dir),
        prefix="." + path.name + ".",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "c3",
                    "kappa_lambda",
                    "random_seed",
                    "cross_section_pb",
                    "uncertainty_pb",
                    "relative_uncertainty",
                    "sherpa_card_sha256",
                    "parameter_card_sha256",
                    "model_library_sha256",
                    "point_directory",
                ),
            )
            writer.writeheader()
            for result in rows:
                hashes = result["hashes"]
                assert isinstance(hashes, dict)
                writer.writerow(
                    {
                        "c3": result["c3"],
                        "kappa_lambda": result["kappa_lambda"],
                        "random_seed": result["random_seed"],
                        "cross_section_pb": result["cross_section_pb"],
                        "uncertainty_pb": result["uncertainty_pb"],
                        "relative_uncertainty": result["relative_uncertainty"],
                        "sherpa_card_sha256": hashes["sherpa_card_sha256"],
                        "parameter_card_sha256": hashes["parameter_card_sha256"],
                        "model_library_sha256": hashes["model_library_sha256"],
                        "point_directory": result["point_directory"],
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_fit_payload(
    work_dir: Path,
    results: Iterable[Mapping[str, object]],
    ranks: int,
    sherpa: str,
    mpirun: str,
    model_library: Path,
    provenance: Optional[Path],
) -> Dict[str, object]:
    minus, zero, plus = ordered_results(results)
    cross_sections = [float(item["cross_section_pb"]) for item in (minus, zero, plus)]
    uncertainties = [float(item["uncertainty_pb"]) for item in (minus, zero, plus)]
    coefficients = quadratic_coefficients(*cross_sections)
    covariance = coefficient_covariance(*uncertainties)
    coefficient_errors = {
        name: math.sqrt(covariance[index][index])
        for index, name in enumerate(("A", "B", "C"))
    }
    return {
        "schema_version": 1,
        "created_utc": now_utc(),
        "process": "21 21 -> 25 25",
        "collider_energy_gev": 14000.0,
        "beam_energies_gev": [7000.0, 7000.0],
        "model": "heft_c3d4_sherpa",
        "model_library": str(model_library),
        "model_library_sha256": sha256_file(model_library),
        "model_provenance": str(provenance) if provenance else None,
        "model_provenance_sha256": sha256_file(provenance) if provenance else None,
        "pdf_set": "NNPDF23_nlo_as_0119",
        "alpha_s_source": "PDF",
        "scale": "VAR{0.25*(H_T2+sqr(2*125.0))}{0.25*(H_T2+sqr(2*125.0))}",
        "integration_error_target": INTEGRATION_ERROR_TARGET,
        "mpi_ranks": ranks,
        "sherpa_executable": sherpa,
        "mpirun_executable": mpirun,
        "event_generation_performed": False,
        "fit_basis": {"c3": [-2.0, -1.0, 0.0], "kappa_lambda": [-1.0, 0.0, 1.0]},
        "point_random_seeds": [int(item["random_seed"]) for item in (minus, zero, plus)],
        "formula": "sigma(kappa_lambda) = A*kappa_lambda^2 + B*kappa_lambda + C",
        "coefficient_order": ["A", "B", "C"],
        "coefficients_pb": coefficients,
        "coefficient_uncertainties_pb": coefficient_errors,
        "coefficient_covariance_pb2": covariance,
        "point_results": list(ordered_results(results)),
        "work_directory": str(work_dir.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--np", type=int, default=32, help="MPI ranks for each sequential integration")
    parser.add_argument("--sherpa", default="Sherpa")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--model-library", type=Path)
    parser.add_argument("--model-provenance", type=Path)
    args = parser.parse_args()

    if args.np < 1:
        raise SystemExit("--np must be positive")
    if not (EXAMPLE_DIR / "Sherpa.yaml").is_file() or not (EXAMPLE_DIR / PARAM_CARD).is_file():
        raise SystemExit("missing gg_hh_ufo example files under %s" % EXAMPLE_DIR)

    model_library = args.model_library.expanduser().resolve() if args.model_library else find_model_library(args.sherpa)
    if not model_library.is_file():
        raise SystemExit("model library is not a file: %s" % model_library)
    provenance = args.model_provenance.expanduser().resolve() if args.model_provenance else None
    if provenance is not None and not provenance.is_file():
        raise SystemExit("model provenance is not a file: %s" % provenance)

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    for c3, kappa_lambda, label, random_seed in POINTS:
        result = run_point(
            work_dir,
            c3,
            kappa_lambda,
            label,
            random_seed,
            args.sherpa,
            args.mpirun,
            args.np,
            model_library,
            provenance,
        )
        results.append(result)
        write_points_csv(work_dir, results)

    fit = build_fit_payload(
        work_dir,
        results,
        args.np,
        args.sherpa,
        args.mpirun,
        model_library,
        provenance,
    )
    atomic_write_json(work_dir / FIT_JSON, fit)
    print("Wrote %s" % (work_dir / POINTS_CSV), flush=True)
    print("Wrote %s" % (work_dir / FIT_JSON), flush=True)
    for name in ("A", "B", "C"):
        print(
            "%s = %.12g +- %.12g pb"
            % (name, fit["coefficients_pb"][name], fit["coefficient_uncertainties_pb"][name]),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FitDriverError as error:
        print("error: %s" % error, file=sys.stderr)
        raise SystemExit(2)
