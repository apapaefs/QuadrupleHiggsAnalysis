#!/usr/bin/env python3
"""Merge LHE source groups and normalize their event weights to one cross section."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from merge_lhe_shards import (
    InitInfo,
    LHE_FOOTER,
    ProcessInit,
    expand_inputs,
    extract_preamble,
    iter_complete_events,
    open_text_input,
    open_text_output,
    parse_init_info,
    read_file,
)


@dataclass(frozen=True)
class SourceStats:
    label: str
    input: str
    files: list[str]
    event_count: int
    raw_weight_sum: float
    raw_weight_square_sum: float
    raw_weight_min: float
    raw_weight_max: float
    fraction: float
    scale: float


def format_float(value: float) -> str:
    return f"{value:.16g}"


def format_init_process_line(process: ProcessInit) -> str:
    return f"{process.xsec:.12g} {process.xerr:.12g} {process.xmax:.12g} {process.lprup:d}"


def source_label(path_text: str, index: int, used: set[str]) -> str:
    label = Path(path_text).name or f"source_{index}"
    if label not in used:
        used.add(label)
        return label
    suffix = 2
    while f"{label}_{suffix}" in used:
        suffix += 1
    label = f"{label}_{suffix}"
    used.add(label)
    return label


def event_payload_lines(event: str, path: Path) -> list[str]:
    start = event.find("<event")
    start = event.find(">", start)
    end = event.rfind("</event>")
    if start < 0 or end < 0:
        raise ValueError(f"{path}: incomplete event block")
    return [
        line.strip()
        for line in event[start + 1:end].splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def event_weight(event: str, path: Path) -> float:
    lines = event_payload_lines(event, path)
    if not lines:
        raise ValueError(f"{path}: empty event block")
    tokens = lines[0].split()
    if len(tokens) < 3:
        raise ValueError(f"{path}: short LHE event header line: {lines[0]}")
    try:
        return float(tokens[2])
    except ValueError as exc:
        raise ValueError(f"{path}: invalid XWGTUP value in event header: {lines[0]}") from exc


def replace_event_weight(event: str, path: Path, new_weight: float) -> str:
    start = event.find("<event")
    start_tag_end = event.find(">", start)
    end = event.rfind("</event>")
    if start_tag_end < 0 or end < 0:
        raise ValueError(f"{path}: incomplete event block")

    body = event[start_tag_end + 1:end]
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        content = line[:-1] if line.endswith("\n") else line
        newline = "\n" if line.endswith("\n") else ""
        stripped = content.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 3:
            raise ValueError(f"{path}: short LHE event header line: {stripped}")
        tokens[2] = format_float(new_weight)
        indent = content[: len(content) - len(content.lstrip())]
        lines[index] = indent + " ".join(tokens) + newline
        return event[: start_tag_end + 1] + "".join(lines) + event[end:]

    raise ValueError(f"{path}: event block has no event header line")


def set_init_idwtup(init_info: InitInfo, idwtup: int) -> InitInfo:
    tokens = init_info.beam_line.split()
    if len(tokens) < 10:
        raise ValueError("short <init> beam/process-count line")
    tokens[-2] = str(idwtup)
    return InitInfo(
        beam_line=" ".join(tokens),
        nprup=init_info.nprup,
        processes=init_info.processes,
    )


def update_preamble_init_info(preamble: str, init_info: InitInfo) -> str:
    init_start = preamble.find("<init")
    if init_start < 0:
        raise ValueError("merged preamble has no <init> block")
    payload_start = preamble.find(">", init_start)
    init_end = preamble.find("</init>", payload_start)
    if payload_start < 0 or init_end < 0:
        raise ValueError("merged preamble has incomplete <init> block")

    payload = preamble[payload_start + 1:init_end]
    lines = payload.splitlines()
    data_line_count = 0
    process_index = 0
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            data_line_count += 1
            indent = line[: len(line) - len(line.lstrip())]
            if data_line_count == 1:
                new_lines.append(indent + init_info.beam_line)
                continue
            if process_index < len(init_info.processes):
                new_lines.append(indent + format_init_process_line(init_info.processes[process_index]))
                process_index += 1
                continue
        new_lines.append(line)

    if process_index != len(init_info.processes):
        raise ValueError("could not rewrite all <init> process cross-section lines")

    payload = "\n".join(new_lines)
    if payload and not payload.endswith("\n"):
        payload += "\n"
    return preamble[: payload_start + 1] + payload + preamble[init_end:]


def compatible_init(reference: InitInfo, current: InitInfo, path: Path) -> None:
    reference_tokens = reference.beam_line.split()
    current_tokens = current.beam_line.split()
    if len(reference_tokens) != len(current_tokens):
        raise ValueError(f"{path}: <init> beam/process-count line differs from first input")
    if reference_tokens[:-2] != current_tokens[:-2]:
        raise ValueError(f"{path}: beam/PDF part of <init> differs from first input")
    if reference.nprup != current.nprup:
        raise ValueError(f"{path}: NPRUP differs from first input")
    if reference.nprup != 1:
        raise ValueError("only single-process LHE files are supported")
    if reference.processes[0].lprup != current.processes[0].lprup:
        raise ValueError(f"{path}: LPRUP differs from first input")


def collect_group_raw_stats(
    label: str,
    input_text: str,
    files: list[Path],
    strict: bool,
) -> SourceStats:
    if not files:
        raise ValueError(f"{input_text}: no LHE input files found")

    event_count = 0
    raw_sum = 0.0
    raw_square_sum = 0.0
    raw_min = math.inf
    raw_max = -math.inf

    for path in files:
        text = read_file(path)
        for event in iter_complete_events(text, path, strict):
            weight = event_weight(event, path)
            event_count += 1
            raw_sum += weight
            raw_square_sum += weight * weight
            raw_min = min(raw_min, weight)
            raw_max = max(raw_max, weight)

    if event_count == 0:
        raise ValueError(f"{input_text}: no complete events found")
    if raw_sum == 0.0:
        raise ValueError(f"{input_text}: raw weight sum is zero")

    return SourceStats(
        label=label,
        input=input_text,
        files=[str(path) for path in files],
        event_count=event_count,
        raw_weight_sum=raw_sum,
        raw_weight_square_sum=raw_square_sum,
        raw_weight_min=raw_min,
        raw_weight_max=raw_max,
        fraction=0.0,
        scale=0.0,
    )


def source_basis(stats: SourceStats, fraction_mode: str) -> float:
    if fraction_mode == "count":
        return float(stats.event_count)
    if fraction_mode == "equal":
        return 1.0
    if fraction_mode == "effective-events":
        if stats.raw_weight_square_sum == 0.0:
            return 0.0
        return stats.raw_weight_sum * stats.raw_weight_sum / stats.raw_weight_square_sum
    raise ValueError(f"unknown fraction mode {fraction_mode}")


def assign_source_fractions(
    stats: list[SourceStats],
    total_xsec: float,
    fraction_mode: str,
) -> list[SourceStats]:
    bases = [source_basis(item, fraction_mode) for item in stats]
    basis_total = sum(bases)
    if basis_total == 0.0:
        raise ValueError(f"{fraction_mode}: source-fraction basis sums to zero")

    scaled: list[SourceStats] = []
    for item, basis in zip(stats, bases):
        fraction = basis / basis_total
        scaled.append(
            SourceStats(
                label=item.label,
                input=item.input,
                files=item.files,
                event_count=item.event_count,
                raw_weight_sum=item.raw_weight_sum,
                raw_weight_square_sum=item.raw_weight_square_sum,
                raw_weight_min=item.raw_weight_min,
                raw_weight_max=item.raw_weight_max,
                fraction=fraction,
                scale=total_xsec * fraction / item.raw_weight_sum,
            )
        )
    return scaled


def first_input_file(groups: list[list[Path]]) -> Path:
    for files in groups:
        if files:
            return files[0]
    raise ValueError("no LHE input files found")


def normalized_init_info(
    first_file: Path,
    total_xsec: float,
    total_xerr: float,
    idwtup: int,
) -> InitInfo:
    text = read_file(first_file)
    init_info = parse_init_info(text, first_file)
    if init_info.nprup != 1:
        raise ValueError("only single-process LHE files are supported")
    process = init_info.processes[0]
    return set_init_idwtup(
        InitInfo(
            beam_line=init_info.beam_line,
            nprup=init_info.nprup,
            processes=(
                ProcessInit(
                    xsec=total_xsec,
                    xerr=total_xerr,
                    xmax=process.xmax,
                    lprup=process.lprup,
                    raw_line=process.raw_line,
                ),
            ),
        ),
        idwtup,
    )


def check_all_inits(groups: list[list[Path]], reference_file: Path) -> None:
    reference = parse_init_info(read_file(reference_file), reference_file)
    for files in groups:
        for path in files:
            compatible_init(reference, parse_init_info(read_file(path), path), path)


def write_manifest(
    manifest_path: Path,
    output: Path,
    total_xsec: float,
    total_xerr: float,
    fraction_mode: str,
    event_count: int,
    total_weight_sum: float,
    sources: list[SourceStats],
) -> None:
    payload = {
        "output": str(output),
        "total_xsec_pb": total_xsec,
        "total_xerr_pb": total_xerr,
        "fraction_mode": fraction_mode,
        "event_count": event_count,
        "total_weight_sum_pb": total_weight_sum,
        "sources": [asdict(source) for source in sources],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def merge_normalized_lhe(
    source_inputs: list[str],
    output: Path,
    total_xsec: float,
    total_xerr: float,
    prefix: str | None,
    expected_events: int | None,
    strict: bool,
    fraction_mode: str,
    idwtup: int,
    manifest_path: Path | None,
) -> tuple[int, float, list[SourceStats]]:
    output.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    used_labels: set[str] = set()
    for index, input_text in enumerate(source_inputs, start=1):
        labels.append(source_label(input_text, index, used_labels))

    groups = [expand_inputs([input_text], prefix, output) for input_text in source_inputs]
    first_file = first_input_file(groups)
    check_all_inits(groups, first_file)

    raw_stats = [
        collect_group_raw_stats(label, input_text, files, strict)
        for label, input_text, files in zip(labels, source_inputs, groups)
    ]
    stats = assign_source_fractions(raw_stats, total_xsec, fraction_mode)

    init_info = normalized_init_info(first_file, total_xsec, total_xerr, idwtup)
    preamble = extract_preamble(read_file(first_file), first_file)
    preamble = update_preamble_init_info(preamble, init_info)

    event_count = 0
    total_weight_sum = 0.0
    with open_text_output(output) as out:
        out.write(preamble)
        for source, files in zip(stats, groups):
            for path in files:
                with open_text_input(path) as handle:
                    text = handle.read()
                for event in iter_complete_events(text, path, strict):
                    weight = event_weight(event, path) * source.scale
                    out.write(replace_event_weight(event, path, weight))
                    event_count += 1
                    total_weight_sum += weight
        out.write(f"{LHE_FOOTER}\n")

    if expected_events is not None and event_count != expected_events:
        raise ValueError(f"expected {expected_events} events, merged {event_count}")

    if manifest_path is not None:
        write_manifest(
            manifest_path,
            output,
            total_xsec,
            total_xerr,
            fraction_mode,
            event_count,
            total_weight_sum,
            stats,
        )

    return event_count, total_weight_sum, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="+",
        help="LHE files or directories; each positional source is one normalization group",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="normalized merged LHE output path")
    parser.add_argument("--total-xsec", type=float, required=True, help="target total cross section in pb")
    parser.add_argument("--total-xerr", type=float, default=0.0, help="target cross-section uncertainty in pb")
    parser.add_argument("--prefix", help="when reading directories, only include PREFIX*.lhe[.gz]")
    parser.add_argument("--expected-events", type=int, help="fail if the merged event count differs")
    parser.add_argument("--strict", action="store_true", help="fail on incomplete trailing event blocks")
    parser.add_argument(
        "--fraction-mode",
        choices=("count", "effective-events", "equal"),
        default="count",
        help=(
            "how to apportion the total cross section among source groups: "
            "count=N_source/N_total, effective-events=(sum w)^2/sum w^2, equal=1/N_sources"
        ),
    )
    parser.add_argument("--idwtup", type=int, default=3, help="IDWTUP value to write in the merged <init> block")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="write a JSON manifest; defaults to OUTPUT.manifest.json unless --no-manifest is set",
    )
    parser.add_argument("--no-manifest", action="store_true", help="do not write a JSON manifest")
    args = parser.parse_args()

    try:
        if args.total_xsec == 0.0:
            raise ValueError("--total-xsec must be nonzero")
        manifest_path = None
        if not args.no_manifest:
            manifest_path = args.manifest or args.output.with_name(args.output.name + ".manifest.json")
        event_count, total_weight_sum, stats = merge_normalized_lhe(
            args.sources,
            args.output,
            args.total_xsec,
            args.total_xerr,
            args.prefix,
            args.expected_events,
            args.strict,
            args.fraction_mode,
            args.idwtup,
            manifest_path,
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    for source in stats:
        print(
            f"source {source.label}: events={source.event_count} "
            f"raw_sum={source.raw_weight_sum:.12g} fraction={source.fraction:.12g} "
            f"scale={source.scale:.12g}"
        )
    print(
        f"merged {event_count} events from {len(stats)} source groups into {args.output}; "
        f"sum(XWGTUP)={total_weight_sum:.12g} pb target={args.total_xsec:.12g} pb"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
