"""Apply forced-splitting sidecar acceptance factors to LHE event weights."""

import argparse
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev
import sys


@dataclass(frozen=True)
class Correction(object):
    factor: float
    probe_trials: int
    probe_successes: int


@dataclass
class ProcessStats(object):
    weights: list = field(default_factory=list)
    factors: list = field(default_factory=list)

    def add(self, weight, factor):
        self.weights.append(float(weight))
        self.factors.append(float(factor))

    @property
    def scale(self):
        weight_sum = sum(self.weights)
        if weight_sum != 0.0:
            return sum(w * f for w, f in zip(self.weights, self.factors)) / weight_sum
        return fmean(self.factors)

    @property
    def factor_stderr(self):
        if len(self.factors) < 2:
            return 0.0
        return stdev(self.factors) / sqrt(len(self.factors))


@dataclass(frozen=True)
class InitUpdate(object):
    old_xsec: float
    new_xsec: float
    new_xerr: float
    scale: float
    event_count: int
    old_lprup: int
    new_lprup: int
    event_weight_norm: float


def read_corrections(path):
    corrections = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 4:
            raise ValueError("Malformed correction line: %r" % line)
        corrections.append(
            Correction(
                factor=float(fields[3]),
                probe_trials=int(fields[1]),
                probe_successes=int(fields[2]),
            )
        )
    return corrections


def _read_init_and_event_headers(path):
    process_lines = None
    event_headers = []
    with Path(path).open() as src:
        line_iter = iter(src)
        for line in line_iter:
            stripped = line.strip()
            if stripped == "<init>":
                beam_fields = next(line_iter).split()
                if len(beam_fields) < 10:
                    raise ValueError("Malformed LHE init beam line")
                nprup = int(beam_fields[-1])
                process_lines = [next(line_iter).split() for _ in range(nprup)]
                continue
            if stripped == "<event>":
                header = next(line_iter).split()
                if len(header) < 3:
                    raise ValueError("Malformed LHE event header: %r" % " ".join(header))
                event_headers.append(header)
    if process_lines is None:
        raise ValueError("LHE file does not contain an init block")
    return process_lines, event_headers


def verify_weighted_lhe(input_lhe, corrections, output_lhe, tolerance=1.0e-9):
    """Check that a weighted LHE reflects the forced-splitting sidecar factors."""
    correction_rows = read_corrections(corrections)
    process_lines, process_stats, process_ids, _ = scan_lhe(input_lhe, correction_rows)
    _, init_updates = update_process_lines(
        process_lines,
        process_stats,
        process_ids,
        input_xsec_error=None,
        update_init=True,
    )
    if not init_updates:
        raise ValueError("Cannot verify weighted LHE without init updates")

    _, raw_headers = _read_init_and_event_headers(input_lhe)
    weighted_process_lines, weighted_headers = _read_init_and_event_headers(output_lhe)
    if len(raw_headers) != len(weighted_headers):
        raise ValueError(
            "Raw LHE has %d events but weighted LHE has %d events"
            % (len(raw_headers), len(weighted_headers))
        )
    if len(raw_headers) != len(correction_rows):
        raise ValueError(
            "Correction file has %d rows but LHE has %d events"
            % (len(correction_rows), len(raw_headers))
        )

    max_event_weight_delta = 0.0
    for index, raw_header in enumerate(raw_headers):
        idprup = int(raw_header[1])
        process = process_index(process_lines, idprup)
        expected = (
            float(raw_header[2])
            * correction_rows[index].factor
            * init_updates[process].event_weight_norm
        )
        observed = float(weighted_headers[index][2])
        max_event_weight_delta = max(max_event_weight_delta, abs(observed - expected))

    weighted_weights = [float(header[2]) for header in weighted_headers]
    weighted_mean = sum(weighted_weights) / len(weighted_weights) if weighted_weights else 0.0
    weighted_init_xsec = sum(float(fields[0]) for fields in weighted_process_lines)
    zero_success_rows = sum(1 for row in correction_rows if row.probe_successes == 0)
    mean_p_hat = (
        sum(row.factor for row in correction_rows) / len(correction_rows)
        if correction_rows
        else 0.0
    )
    init_mean_delta = abs(weighted_mean - weighted_init_xsec)
    ok = max_event_weight_delta <= tolerance and init_mean_delta <= tolerance

    return {
        "ok": ok,
        "correction_rows": len(correction_rows),
        "zero_success_rows": zero_success_rows,
        "nonzero_weight_rows": sum(1 for weight in weighted_weights if weight != 0.0),
        "mean_p_hat": mean_p_hat,
        "weighted_mean_xwgtup": weighted_mean,
        "weighted_init_xsec": weighted_init_xsec,
        "max_event_weight_delta": max_event_weight_delta,
        "init_mean_delta": init_mean_delta,
    }


def process_index(process_lines, idprup):
    if len(process_lines) == 1:
        return 0

    for index, fields in enumerate(process_lines):
        if len(fields) >= 4 and int(fields[3]) == idprup:
            return index

    raise ValueError("Event IDPRUP=%d does not match any LPRUP in the init block" % idprup)


def scan_lhe(input_lhe, corrections):
    process_lines = None
    process_stats = []
    process_ids = []
    event_index = 0
    has_header = False

    with Path(input_lhe).open() as src:
        line_iter = iter(src)
        for line in line_iter:
            stripped = line.strip()
            if stripped == "<header>":
                has_header = True

            if stripped == "<init>":
                beam_fields = next(line_iter).split()
                if len(beam_fields) < 10:
                    raise ValueError("Malformed LHE init beam line")
                nprup = int(beam_fields[-1])
                process_lines = [next(line_iter).split() for _ in range(nprup)]
                process_stats = [ProcessStats() for _ in range(nprup)]
                process_ids = [set() for _ in range(nprup)]
                continue

            if stripped == "<event>":
                if process_lines is None:
                    raise ValueError("Encountered an event before the init block")
                if event_index >= len(corrections):
                    raise ValueError("Correction file has fewer entries than LHE events")

                header = next(line_iter).split()
                if len(header) < 3:
                    raise ValueError("Malformed LHE event header: %r" % " ".join(header))

                idprup = int(header[1])
                weight = float(header[2])
                index = process_index(process_lines, idprup)
                process_stats[index].add(weight, corrections[event_index].factor)
                process_ids[index].add(idprup)
                event_index += 1

    if process_lines is None:
        raise ValueError("LHE file does not contain an init block")
    if event_index != len(corrections):
        raise ValueError(
            "Correction file has %d entries but LHE has %d events"
            % (len(corrections), event_index)
        )

    return process_lines, process_stats, process_ids, has_header


def update_process_lines(process_lines, process_stats, process_ids, input_xsec_error, update_init):
    if not update_init:
        return process_lines, []
    if input_xsec_error is not None and len(process_lines) != 1:
        raise ValueError("--input-xsec-error can only be used with one init process")

    updated_lines = []
    updates = []

    for index, fields in enumerate(process_lines):
        if len(fields) < 4:
            raise ValueError("Malformed LHE init process line: %r" % " ".join(fields))

        old_xsec = float(fields[0])
        old_xerr = float(fields[1])
        old_lprup = int(fields[3])
        observed_ids = process_ids[index]
        if len(observed_ids) > 1:
            raise ValueError(
                "A single init process received multiple event IDPRUP values: "
                + ", ".join(str(value) for value in sorted(observed_ids))
            )
        new_lprup = next(iter(observed_ids), old_lprup)
        stats = process_stats[index]

        if stats.factors:
            scale = stats.scale
            factor_stderr = stats.factor_stderr
            event_count = len(stats.factors)
        else:
            scale = 1.0
            factor_stderr = 0.0
            event_count = 0

        xerr_input = input_xsec_error if input_xsec_error is not None else old_xerr
        new_xsec = old_xsec * scale
        new_xerr = sqrt((abs(xerr_input) * abs(scale)) ** 2 + (abs(old_xsec) * factor_stderr) ** 2)
        input_weight_sum = sum(stats.weights)
        event_weight_norm = (
            old_xsec * event_count / input_weight_sum
            if event_count > 0 and input_weight_sum != 0.0
            else 1.0
        )

        updated = list(fields)
        updated[0] = "%.9e" % new_xsec
        updated[1] = "%.9e" % new_xerr
        updated[3] = str(new_lprup)
        updated_lines.append(updated)
        updates.append(
            InitUpdate(
                old_xsec=old_xsec,
                new_xsec=new_xsec,
                new_xerr=new_xerr,
                scale=scale,
                event_count=event_count,
                old_lprup=old_lprup,
                new_lprup=new_lprup,
                event_weight_norm=event_weight_norm,
            )
        )

    return updated_lines, updates


def write_header_comments(dst, updates):
    dst.write(
        "<!-- Reweighted with MinBShowerVeto sidecar factors: "
        "each event XWGTUP has been multiplied by p_hat and the LHE weight normalization. -->\n"
    )
    for index, update in enumerate(updates, start=1):
        dst.write(
            "<!-- Process {index}: XSECUP {old:.9e} pb * "
            "weighted_mean(p_hat) {scale:.9e} = {new:.9e} pb; "
            "event weights normalized by {norm:.9e}. -->\n".format(
                index=index,
                old=update.old_xsec,
                scale=update.scale,
                new=update.new_xsec,
                norm=update.event_weight_norm,
            )
        )


def write_header_note(dst, updates):
    dst.write("<header>\n")
    write_header_comments(dst, updates)
    dst.write("</header>\n")


def apply_weights(input_lhe, corrections, output_lhe, input_xsec_error=None, update_init=True):
    input_lhe = Path(input_lhe)
    corrections = Path(corrections)
    output_lhe = Path(output_lhe)

    correction_rows = read_corrections(corrections)
    process_lines, process_stats, process_ids, has_header = scan_lhe(input_lhe, correction_rows)
    updated_process_lines, init_updates = update_process_lines(
        process_lines, process_stats, process_ids, input_xsec_error, update_init
    )
    event_index = 0
    expecting_header = False
    inserted_header = False

    with input_lhe.open() as src, output_lhe.open("w") as dst:
        line_iter = iter(src)
        for line in line_iter:
            if (
                update_init
                and init_updates
                and not has_header
                and not inserted_header
                and line.strip().startswith("<LesHouchesEvents")
            ):
                dst.write(line)
                write_header_note(dst, init_updates)
                inserted_header = True
                continue

            if update_init and init_updates and has_header and line.strip() == "</header>":
                write_header_comments(dst, init_updates)
                dst.write(line)
                inserted_header = True
                continue

            if update_init and init_updates and line.strip() == "<init>":
                dst.write(line)
                beam_line = next(line_iter)
                dst.write(beam_line)
                nprup = int(beam_line.split()[-1])
                if nprup != len(updated_process_lines):
                    raise ValueError("Init process count changed between scan and write")
                for fields in updated_process_lines:
                    next(line_iter)
                    dst.write("  " + "  ".join(fields) + "\n")
                continue

            if expecting_header:
                if event_index >= len(correction_rows):
                    raise ValueError("Correction file has fewer entries than LHE events")
                fields = line.split()
                if len(fields) < 3:
                    raise ValueError("Malformed LHE event header: %r" % line)
                event_weight_norm = 1.0
                if update_init and init_updates:
                    idprup = int(fields[1])
                    index = process_index(process_lines, idprup)
                    event_weight_norm = init_updates[index].event_weight_norm
                fields[2] = "%.16e" % (
                    float(fields[2]) * correction_rows[event_index].factor * event_weight_norm
                )
                dst.write(" ".join(fields) + "\n")
                event_index += 1
                expecting_header = False
                continue

            dst.write(line)
            if line.strip() == "<event>":
                expecting_header = True

    if event_index != len(correction_rows):
        raise ValueError(
            "Correction file has %d entries but LHE has %d events"
            % (len(correction_rows), event_index)
        )

    for index, update in enumerate(init_updates, start=1):
        print(
            "Updated init process {index}: XSECUP {old:.9e} -> {new:.9e} pb, "
            "XERRUP -> {err:.9e} pb, LPRUP {old_lprup} -> {new_lprup} "
            "from {events} events".format(
                index=index,
                old=update.old_xsec,
                new=update.new_xsec,
                err=update.new_xerr,
                old_lprup=update.old_lprup,
                new_lprup=update.new_lprup,
                events=update.event_count,
            ),
            file=sys.stderr,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-xsec-error",
        type=float,
        default=None,
        help=(
            "Override the init-block XERRUP value, in pb, before propagating "
            "the corrected uncertainty. Useful when the input LHEWriter file "
            "contains a placeholder uncertainty."
        ),
    )
    parser.add_argument(
        "--no-update-init",
        action="store_true",
        help="Only reweight event headers; leave the LHE init block unchanged.",
    )
    parser.add_argument("input_lhe", type=Path)
    parser.add_argument("corrections", type=Path)
    parser.add_argument("output_lhe", type=Path)
    args = parser.parse_args(argv)
    apply_weights(
        args.input_lhe,
        args.corrections,
        args.output_lhe,
        input_xsec_error=args.input_xsec_error,
        update_init=not args.no_update_init,
    )


if __name__ == "__main__":
    main()
