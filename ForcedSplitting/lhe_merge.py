"""Small LHE helpers for chunking and merging forced-splitting samples."""

import gzip
import json
import math
import re
from pathlib import Path
from statistics import stdev


_RAW_EVENT_RE = re.compile(r"<event>.*?</event>\s*", re.DOTALL)


def _open_text_read(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def _open_text_write(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt")
    return path.open("w")


def read_lhe_text(path):
    with _open_text_read(path) as handle:
        return handle.read()


def write_lhe_text(path, text, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("%s already exists; pass overwrite=True to replace it" % path)
    with _open_text_write(path) as handle:
        handle.write(text)


def lhe_event_parts(path):
    text = read_lhe_text(path)
    matches = list(_RAW_EVENT_RE.finditer(text))
    if not matches:
        raise RuntimeError("Input LHE contains no <event> blocks: %s" % path)
    preamble = text[: matches[0].start()]
    events = [match.group(0) for match in matches]
    footer = text[matches[-1].end() :]
    return preamble, events, footer


def event_ranges(total, chunks):
    total = int(total)
    chunks = int(chunks)
    if chunks < 1:
        raise ValueError("number of chunks must be positive")
    if total < 1:
        raise ValueError("cannot split zero events")
    chunks = min(chunks, total)
    base, extra = divmod(total, chunks)
    ranges = []
    start = 0
    for index in range(chunks):
        count = base + (1 if index < extra else 0)
        stop = start + count
        ranges.append((start, stop))
        start = stop
    return ranges


def write_lhe_event_slice(input_lhe, output_lhe, start, stop, overwrite=False):
    preamble, events, footer = lhe_event_parts(input_lhe)
    selected = events[int(start) : int(stop)]
    if not selected:
        raise RuntimeError("Requested empty LHE event slice %s:%s from %s" % (start, stop, input_lhe))
    write_lhe_text(output_lhe, preamble + "".join(selected) + footer, overwrite=overwrite)
    return len(selected)


def event_weight(event_block):
    lines = event_block.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "<event>":
            if index + 1 >= len(lines):
                raise ValueError("Malformed LHE event block without an event header")
            fields = lines[index + 1].split()
            if len(fields) < 3:
                raise ValueError("Malformed LHE event header: %r" % lines[index + 1])
            return float(fields[2])
    raise ValueError("Event block does not contain an <event> tag")


def event_weights_from_lhe(path):
    _, events, _ = lhe_event_parts(path)
    return [event_weight(event) for event in events]


def _replace_single_process_init(preamble, xsec_pb, xerr_pb, xmax_weight):
    lines = preamble.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() != "<init>":
            continue
        beam_index = index + 1
        process_index = beam_index + 1
        if process_index >= len(lines):
            raise ValueError("Malformed LHE init block")
        beam_fields = lines[beam_index].split()
        if not beam_fields:
            raise ValueError("Malformed LHE init beam line")
        nprup = int(beam_fields[-1])
        if nprup != 1:
            raise ValueError("Weighted LHE chunk merging currently expects exactly one init process")
        fields = lines[process_index].split()
        if len(fields) < 4:
            raise ValueError("Malformed LHE init process line: %r" % lines[process_index])
        fields[0] = "%.9e" % float(xsec_pb)
        fields[1] = "%.9e" % float(xerr_pb)
        fields[2] = "%.9e" % float(xmax_weight)
        lines[process_index] = "  " + "  ".join(fields) + "\n"
        return "".join(lines)
    raise ValueError("LHE preamble does not contain an init block")


def _input_xsec(path):
    preamble, _, _ = lhe_event_parts(path)
    lines = preamble.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "<init>":
            continue
        if index + 2 >= len(lines):
            return None
        fields = lines[index + 2].split()
        return float(fields[0]) if fields else None
    return None


def merge_weighted_lhe_chunks(input_lhes, output_lhe, summary_path=None, overwrite=False):
    """Merge complete LHE event blocks and set XSECUP to mean merged XWGTUP.

    The input chunks are expected to have already had per-event forced-splitting
    weights applied.  The merged init cross section is therefore the mean event
    weight of the merged event ensemble, not the sum of per-chunk init entries.
    """

    input_lhes = [Path(path) for path in input_lhes]
    if not input_lhes:
        raise ValueError("At least one LHE chunk is required")
    for path in input_lhes:
        if not path.exists():
            raise FileNotFoundError("Input LHE chunk does not exist: %s" % path)

    first_preamble = None
    footer = None
    merged_events = []
    input_event_counts = []
    input_xsecs = []
    for path in input_lhes:
        preamble, events, chunk_footer = lhe_event_parts(path)
        if first_preamble is None:
            first_preamble = preamble
            footer = chunk_footer
        merged_events.extend(events)
        input_event_counts.append(len(events))
        input_xsecs.append(_input_xsec(path))

    weights = [event_weight(event) for event in merged_events]
    if not weights:
        raise RuntimeError("No events found while merging LHE chunks")
    weight_sum = sum(weights)
    merged_xsec = weight_sum / len(weights)
    merged_xerr = stdev(weights) / math.sqrt(len(weights)) if len(weights) > 1 else 0.0
    xmax = max(abs(weight) for weight in weights)
    merged_preamble = _replace_single_process_init(first_preamble, merged_xsec, merged_xerr, xmax)
    write_lhe_text(output_lhe, merged_preamble + "".join(merged_events) + footer, overwrite=overwrite)

    summary = {
        "input_files": [str(path) for path in input_lhes],
        "input_file_count": len(input_lhes),
        "input_event_counts": input_event_counts,
        "input_xsec_pb": input_xsecs,
        "total_events": len(merged_events),
        "weight_sum": float(weight_sum),
        "merged_xsec_pb": float(merged_xsec),
        "merged_xsec_error_pb": float(merged_xerr),
        "event_weight_min": float(min(weights)),
        "event_weight_max": float(max(weights)),
        "event_weight_mean": float(merged_xsec),
        "zero_weight_events": sum(1 for weight in weights if weight == 0.0),
        "output_lhe": str(output_lhe),
    }
    if summary_path is not None:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
