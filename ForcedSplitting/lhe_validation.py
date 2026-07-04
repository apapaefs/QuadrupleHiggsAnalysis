"""Small LHE parser and repair helpers for forced-splitting LHE files."""

import argparse
import gzip
import re
from pathlib import Path


class LHEParticle(object):
    def __init__(
        self,
        pid,
        status,
        mother1,
        mother2,
        color1,
        color2,
        px,
        py,
        pz,
        energy,
        mass,
    ):
        self.pid = pid
        self.status = status
        self.mother1 = mother1
        self.mother2 = mother2
        self.color1 = color1
        self.color2 = color2
        self.px = px
        self.py = py
        self.pz = pz
        self.energy = energy
        self.mass = mass


class LHEEvent(object):
    def __init__(self, header, particles):
        self.header = header
        self.particles = particles


_EVENT_RE = re.compile(r"<event>(.*?)</event>", re.DOTALL)


def _clean_event_lines(event_text):
    lines = []
    for line in event_text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    return lines


def parse_particle_row(line):
    columns = line.split()
    if len(columns) < 11:
        raise ValueError("LHE particle row has fewer than 11 columns: %s" % line)
    return LHEParticle(
        pid=int(columns[0]),
        status=int(columns[1]),
        mother1=int(columns[2]),
        mother2=int(columns[3]),
        color1=int(columns[4]),
        color2=int(columns[5]),
        px=float(columns[6]),
        py=float(columns[7]),
        pz=float(columns[8]),
        energy=float(columns[9]),
        mass=float(columns[10]),
    )


def parse_lhe_events(lhe_text):
    events = []
    for match in _EVENT_RE.finditer(lhe_text):
        lines = _clean_event_lines(match.group(1))
        if not lines:
            continue
        header = lines[0].split()
        if len(header) < 1:
            raise ValueError("Malformed LHE event header")
        nup = int(header[0])
        particle_lines = lines[1 : 1 + nup]
        if len(particle_lines) != nup:
            raise ValueError("Event declares %d particles but has %d rows" % (nup, len(particle_lines)))
        events.append(LHEEvent(header=header, particles=[parse_particle_row(line) for line in particle_lines]))
    return events


def _read_lhe_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def _write_lhe_text(path, text):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(str(path), "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)


def _line_columns(line):
    return line.split("#", 1)[0].split()


def _find_init_line_indices(lines):
    init_start = None
    init_end = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "<init>":
            init_start = index
        elif stripped == "</init>" and init_start is not None:
            init_end = index
            break
    if init_start is None or init_end is None:
        raise ValueError("LHE file does not contain a complete <init> block")
    return init_start, init_end


def event_process_ids(lhe_text):
    """Return the sorted non-comment event process IDs used in an LHE file."""
    process_ids = set()
    for event in parse_lhe_events(lhe_text):
        if len(event.header) < 2:
            raise ValueError("Malformed LHE event header: missing IDPRUP")
        process_ids.add(int(event.header[1]))
    return sorted(process_ids)


def declared_process_ids(lhe_text):
    """Return the process IDs declared by the LHE <init> block."""
    lines = lhe_text.splitlines()
    init_start, init_end = _find_init_line_indices(lines)
    data_indices = [
        index
        for index in range(init_start + 1, init_end)
        if _line_columns(lines[index])
    ]
    if not data_indices:
        raise ValueError("LHE <init> block is empty")

    beam_columns = _line_columns(lines[data_indices[0]])
    if len(beam_columns) < 10:
        raise ValueError("Malformed LHE <init> beam/process-count line")
    n_processes = int(beam_columns[9])

    process_indices = data_indices[1 : 1 + n_processes]
    if len(process_indices) != n_processes:
        raise ValueError("LHE <init> block declares %d processes but only has %d process rows" % (n_processes, len(process_indices)))

    process_ids = []
    for index in process_indices:
        columns = _line_columns(lines[index])
        if len(columns) < 4:
            raise ValueError("Malformed LHE process declaration: %s" % lines[index])
        process_ids.append(int(columns[-1]))
    return process_ids


def normalize_single_process_lprup(lhe_text):
    """Normalize a single-process LHE init declaration to match event IDPRUP.

    Herwig's LHEWriter can write one process with LPRUP = 0 while the event
    headers use IDPRUP = 1.  Herwig can continue after warning about an
    undeclared process, but Stage-2 reads are cleaner if the declaration is
    internally consistent.  This helper only rewrites the unambiguous case:
    one declared process and one event process id.
    """
    lines = lhe_text.splitlines()
    trailing_newline = lhe_text.endswith("\n")
    init_start, init_end = _find_init_line_indices(lines)
    data_indices = [
        index
        for index in range(init_start + 1, init_end)
        if _line_columns(lines[index])
    ]
    if not data_indices:
        raise ValueError("LHE <init> block is empty")

    beam_columns = _line_columns(lines[data_indices[0]])
    if len(beam_columns) < 10:
        raise ValueError("Malformed LHE <init> beam/process-count line")
    n_processes = int(beam_columns[9])
    process_indices = data_indices[1 : 1 + n_processes]
    if len(process_indices) != n_processes:
        raise ValueError("LHE <init> block declares %d processes but only has %d process rows" % (n_processes, len(process_indices)))

    event_ids = event_process_ids(lhe_text)
    declared_ids = declared_process_ids(lhe_text)
    if len(event_ids) != 1:
        return lhe_text, False, "not changed: found %d event process ids: %s" % (len(event_ids), event_ids)
    if event_ids[0] in declared_ids:
        return lhe_text, False, "not changed: event process id %d is already declared" % event_ids[0]
    if n_processes != 1:
        return lhe_text, False, "not changed: cannot repair %d declared processes automatically" % n_processes

    process_index = process_indices[0]
    original_line = lines[process_index]
    data_part, separator, comment = original_line.partition("#")
    columns = data_part.split()
    if len(columns) < 4:
        raise ValueError("Malformed LHE process declaration: %s" % original_line)
    columns[-1] = str(event_ids[0])

    leading = re.match(r"\s*", data_part).group(0)
    replacement = leading + " ".join(columns)
    if separator:
        replacement += " #" + comment
    lines[process_index] = replacement

    normalized = "\n".join(lines)
    if trailing_newline:
        normalized += "\n"
    return normalized, True, "changed: declared process id %s -> %d" % (declared_ids[0], event_ids[0])


def normalize_lhe_file_process_ids(path, output=None):
    """Normalize process declarations in an LHE file and write the result."""
    path = Path(path)
    output = Path(output) if output is not None else path
    text = _read_lhe_text(path)
    normalized, changed, message = normalize_single_process_lprup(text)
    if changed or output != path:
        _write_lhe_text(output, normalized)
    return changed, message


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    normalize = subparsers.add_parser(
        "normalize-process-ids",
        help="rewrite a single-process LHE LPRUP declaration to match event IDPRUP",
    )
    normalize.add_argument("lhe_file", type=Path)
    normalize.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "normalize-process-ids":
        changed, message = normalize_lhe_file_process_ids(args.lhe_file, output=args.output)
        print("%s: %s" % (args.output or args.lhe_file, message))
        return 0 if changed else 0
    parser.error("choose a command: normalize-process-ids")


if __name__ == "__main__":
    main()
