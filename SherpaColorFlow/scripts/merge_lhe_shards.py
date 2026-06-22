#!/usr/bin/env python3
"""Merge sharded LHE files into one valid LHE file."""

from __future__ import annotations

import argparse
import gzip
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


LHE_FOOTER = "</LesHouchesEvents>"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SHERPA_XSEC_RE = re.compile(
    r":\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*pb\s*"
    r"\+-\s*\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*pb"
)


@dataclass(frozen=True)
class ProcessInit:
    xsec: float
    xerr: float
    xmax: float
    lprup: int
    raw_line: str


@dataclass(frozen=True)
class InitInfo:
    beam_line: str
    nprup: int
    processes: tuple[ProcessInit, ...]


def open_text_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def open_text_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt")
    return path.open("w")


def expand_inputs(inputs: Iterable[str], prefix: str | None, output: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    output_resolved = output.resolve()

    for item in inputs:
        path = Path(item)
        candidates: list[Path]
        if path.is_dir():
            patterns = [f"{prefix}*.lhe", f"{prefix}*.lhe.gz"] if prefix else ["*.lhe", "*.lhe.gz"]
            candidates = []
            for pattern in patterns:
                candidates.extend(path.rglob(pattern))
        else:
            candidates = [path]

        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved == output_resolved or resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)

    return sorted(files)


def read_file(path: Path) -> str:
    with open_text_input(path) as handle:
        return handle.read()


def write_file(path: Path, text: str) -> None:
    with open_text_output(path) as handle:
        handle.write(text)


def backup_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


def has_lhe_footer(text: str) -> bool:
    return LHE_FOOTER in text


def close_lhe_text(text: str) -> str:
    last_event_end = text.rfind("</event>")
    if last_event_end >= 0:
        end = last_event_end + len("</event>")
        return text[:end].rstrip() + f"\n{LHE_FOOTER}\n"
    return text.rstrip() + f"\n{LHE_FOOTER}\n"


def load_inputs(
    files: list[Path],
    fix_unclosed_inputs: bool,
    make_backup: bool,
    backup_suffix: str,
) -> dict[Path, str]:
    texts: dict[Path, str] = {}
    for path in files:
        text = read_file(path)
        if not has_lhe_footer(text):
            print(f"WARNING: {path}: missing {LHE_FOOTER} footer", file=sys.stderr)
            if fix_unclosed_inputs:
                if make_backup:
                    shutil.copy2(path, backup_path(path, backup_suffix))
                text = close_lhe_text(text)
                write_file(path, text)
                print(f"WARNING: {path}: fixed missing LHE footer", file=sys.stderr)
        texts[path] = text
    return texts


def extract_tag_block(text: str, path: Path, tag: str) -> str:
    start = text.find(f"<{tag}")
    if start < 0:
        raise ValueError(f"{path}: no <{tag}> block found")
    start_tag_end = text.find(">", start)
    end = text.find(f"</{tag}>", start_tag_end)
    if start_tag_end < 0 or end < 0:
        raise ValueError(f"{path}: incomplete <{tag}> block")
    return text[start:end + len(f"</{tag}>")]


def init_payload_lines(init_block: str) -> list[str]:
    start_tag_end = init_block.find(">")
    end = init_block.rfind("</init>")
    payload = init_block[start_tag_end + 1:end]
    return [
        line.strip()
        for line in payload.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_init_info(text: str, path: Path) -> InitInfo:
    lines = init_payload_lines(extract_tag_block(text, path, "init"))
    if not lines:
        raise ValueError(f"{path}: empty <init> block")

    beam_tokens = lines[0].split()
    if len(beam_tokens) < 10:
        raise ValueError(f"{path}: short <init> beam/process-count line")
    try:
        nprup = int(beam_tokens[-1])
    except ValueError as exc:
        raise ValueError(f"{path}: invalid NPRUP value in <init> block") from exc

    process_lines = lines[1:1 + nprup]
    if len(process_lines) != nprup:
        raise ValueError(f"{path}: <init> declares {nprup} process line(s), found {len(process_lines)}")

    processes: list[ProcessInit] = []
    for index, line in enumerate(process_lines, start=1):
        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError(f"{path}: short process line {index} in <init> block")
        try:
            processes.append(
                ProcessInit(
                    xsec=float(tokens[0]),
                    xerr=float(tokens[1]),
                    xmax=float(tokens[2]),
                    lprup=int(tokens[3]),
                    raw_line=line,
                )
            )
        except ValueError as exc:
            raise ValueError(f"{path}: invalid numeric value in <init> process line {index}") from exc

    return InitInfo(beam_line=" ".join(beam_tokens), nprup=nprup, processes=tuple(processes))


def close_enough(left: float, right: float, abs_tol: float, rel_tol: float) -> bool:
    return math.isclose(left, right, abs_tol=abs_tol, rel_tol=rel_tol)


def validate_cross_sections(
    init_infos: dict[Path, InitInfo],
    abs_tol: float,
    rel_tol: float,
) -> InitInfo:
    if not init_infos:
        raise ValueError("no <init> blocks found")

    reference_path = next(iter(init_infos))
    reference = init_infos[reference_path]
    for path, info in list(init_infos.items())[1:]:
        if info.beam_line != reference.beam_line:
            raise ValueError(f"{path}: <init> beam/process-count line differs from {reference_path}")
        if info.nprup != reference.nprup:
            raise ValueError(f"{path}: NPRUP differs from {reference_path}")
        for index, (left, right) in enumerate(zip(reference.processes, info.processes), start=1):
            if left.lprup != right.lprup:
                raise ValueError(f"{path}: LPRUP mismatch for process {index}")
            if not close_enough(left.xsec, right.xsec, abs_tol, rel_tol):
                raise ValueError(
                    f"{path}: cross-section mismatch for process {index}: "
                    f"{right.xsec} vs {left.xsec} in {reference_path}"
                )
            if not close_enough(left.xerr, right.xerr, abs_tol, rel_tol):
                raise ValueError(
                    f"{path}: cross-section uncertainty mismatch for process {index}: "
                    f"{right.xerr} vs {left.xerr} in {reference_path}"
                )
    return reference


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_sherpa_log_cross_sections(path: Path) -> tuple[tuple[float, float], ...]:
    values: list[tuple[float, float]] = []
    text = read_file(path)
    for line in strip_ansi(text).splitlines():
        match = SHERPA_XSEC_RE.search(line)
        if match:
            values.append((float(match.group(1)), float(match.group(2))))
    return tuple(values)


def discover_log_cross_sections(files: list[Path], log_glob: str) -> dict[Path, tuple[tuple[float, float], ...]]:
    discovered: dict[Path, tuple[tuple[float, float], ...]] = {}
    seen: set[Path] = set()
    for path in files:
        for log_path in sorted(path.parent.glob(log_glob)):
            resolved = log_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            values = parse_sherpa_log_cross_sections(log_path)
            if values:
                discovered[log_path] = values
    return discovered


def init_info_with_log_cross_section(
    reference: InitInfo,
    logs: dict[Path, tuple[tuple[float, float], ...]],
    abs_tol: float,
    rel_tol: float,
) -> tuple[InitInfo, str]:
    if not logs:
        return reference, "LHE init"

    first_log = next(iter(logs))
    first_values = logs[first_log]
    if len(first_values) != reference.nprup:
        if reference.nprup == 1 and first_values:
            first_values = first_values[:1]
        else:
            raise ValueError(
                f"{first_log}: found {len(first_values)} Sherpa cross-section line(s), "
                f"but <init> declares {reference.nprup} process(es)"
            )

    for path, values in list(logs.items())[1:]:
        if len(values) != len(first_values):
            raise ValueError(f"{path}: Sherpa log cross-section process count differs from {first_log}")
        for index, ((ref_xsec, ref_xerr), (xsec, xerr)) in enumerate(zip(first_values, values), start=1):
            if not close_enough(ref_xsec, xsec, abs_tol, rel_tol):
                raise ValueError(
                    f"{path}: Sherpa log cross-section mismatch for process {index}: "
                    f"{xsec} vs {ref_xsec} in {first_log}"
                )
            if not close_enough(ref_xerr, xerr, abs_tol, rel_tol):
                raise ValueError(
                    f"{path}: Sherpa log cross-section uncertainty mismatch for process {index}: "
                    f"{xerr} vs {ref_xerr} in {first_log}"
                )

    processes = []
    for process, (xsec, xerr) in zip(reference.processes, first_values):
        processes.append(
            ProcessInit(
                xsec=xsec,
                xerr=xerr,
                xmax=process.xmax,
                lprup=process.lprup,
                raw_line=process.raw_line,
            )
        )
    return (
        InitInfo(
            beam_line=reference.beam_line,
            nprup=reference.nprup,
            processes=tuple(processes),
        ),
        f"Sherpa log ({first_log})",
    )


def format_init_process_line(process: ProcessInit) -> str:
    return f"{process.xsec:.12g} {process.xerr:.12g} {process.xmax:.12g} {process.lprup:d}"


def update_preamble_init_processes(preamble: str, init_info: InitInfo) -> str:
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
            if data_line_count > 1 and process_index < len(init_info.processes):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(indent + format_init_process_line(init_info.processes[process_index]))
                process_index += 1
                continue
        new_lines.append(line)

    if process_index != len(init_info.processes):
        raise ValueError("could not rewrite all <init> process cross-section lines")

    payload = "\n".join(new_lines)
    if payload and not payload.endswith("\n"):
        payload += "\n"
    return preamble[:payload_start + 1] + payload + preamble[init_end:]


def extract_preamble(text: str, path: Path) -> str:
    event_start = text.find("<event")
    if event_start >= 0:
        preamble = text[:event_start]
    else:
        footer_start = text.find(LHE_FOOTER)
        preamble = text[:footer_start] if footer_start >= 0 else text

    if "<LesHouchesEvents" not in preamble:
        raise ValueError(f"{path}: no <LesHouchesEvents> header before first event")
    if "<init" not in preamble:
        raise ValueError(f"{path}: no <init> block before first event")

    return preamble.rstrip() + "\n"


def iter_complete_events(text: str, path: Path, strict: bool):
    pos = 0
    while True:
        start = text.find("<event", pos)
        if start < 0:
            return
        start_tag_end = text.find(">", start)
        end = text.find("</event>", start_tag_end)
        if start_tag_end < 0 or end < 0:
            message = f"{path}: incomplete trailing event block skipped"
            if strict:
                raise ValueError(message)
            print(f"WARNING: {message}", file=sys.stderr)
            return
        end += len("</event>")
        yield text[start:end].rstrip() + "\n"
        pos = end


def merge_lhe_files(
    files: list[Path],
    output: Path,
    expected_events: int | None,
    strict: bool,
    fix_unclosed_inputs: bool,
    make_backup: bool,
    backup_suffix: str,
    xsec_abs_tol: float,
    xsec_rel_tol: float,
    use_sherpa_log_xsec: bool,
    xsec_log_glob: str,
) -> tuple[int, InitInfo, str]:
    if not files:
        raise ValueError("no LHE input files found")

    texts = load_inputs(files, fix_unclosed_inputs, make_backup, backup_suffix)
    init_info = validate_cross_sections(
        {path: parse_init_info(texts[path], path) for path in files},
        xsec_abs_tol,
        xsec_rel_tol,
    )
    xsec_source = "LHE init"
    if use_sherpa_log_xsec:
        init_info, xsec_source = init_info_with_log_cross_section(
            init_info,
            discover_log_cross_sections(files, xsec_log_glob),
            xsec_abs_tol,
            xsec_rel_tol,
        )

    first_text = texts[files[0]]
    preamble = extract_preamble(first_text, files[0])
    if xsec_source != "LHE init":
        preamble = update_preamble_init_processes(preamble, init_info)

    event_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_text_output(output) as out:
        out.write(preamble)
        for path in files:
            text = texts[path]
            for event in iter_complete_events(text, path, strict):
                out.write(event)
                event_count += 1
        out.write(f"{LHE_FOOTER}\n")

    if expected_events is not None and event_count != expected_events:
        raise ValueError(f"expected {expected_events} events, merged {event_count}")
    return event_count, init_info, xsec_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="LHE files or directories containing LHE shards")
    parser.add_argument("-o", "--output", type=Path, required=True, help="merged LHE output path")
    parser.add_argument("--prefix", help="when reading directories, only include PREFIX*.lhe[.gz]")
    parser.add_argument("--expected-events", type=int, help="fail if the merged event count differs")
    parser.add_argument("--strict", action="store_true", help="fail on incomplete trailing event blocks")
    parser.add_argument(
        "--fix-unclosed-inputs",
        action="store_true",
        help="repair input LHE files missing the final LesHouchesEvents footer; use only after jobs finish",
    )
    parser.add_argument("--backup-suffix", default=".bak", help="suffix for backups made by --fix-unclosed-inputs")
    parser.add_argument("--no-backup", action="store_true", help="do not keep backups when fixing inputs")
    parser.add_argument("--xsec-abs-tol", type=float, default=0.0, help="absolute tolerance for XSECUP/XERRUP checks")
    parser.add_argument(
        "--xsec-rel-tol",
        type=float,
        default=1.0e-12,
        help="relative tolerance for XSECUP/XERRUP checks",
    )
    parser.add_argument(
        "--xsec-log-glob",
        default="sherpa_*.log",
        help="sibling Sherpa log glob used to recover physical cross sections when LHE init has placeholders",
    )
    parser.add_argument(
        "--no-sherpa-log-xsec",
        action="store_true",
        help="do not replace merged init cross sections with values parsed from sibling Sherpa logs",
    )
    args = parser.parse_args()

    try:
        files = expand_inputs(args.inputs, args.prefix, args.output)
        event_count, init_info, xsec_source = merge_lhe_files(
            files,
            args.output,
            args.expected_events,
            args.strict,
            args.fix_unclosed_inputs,
            not args.no_backup,
            args.backup_suffix,
            args.xsec_abs_tol,
            args.xsec_rel_tol,
            not args.no_sherpa_log_xsec,
            args.xsec_log_glob,
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    for index, process in enumerate(init_info.processes, start=1):
        print(
            f"cross section: process {index} XSECUP={process.xsec:g} pb "
            f"XERRUP={process.xerr:g} pb LPRUP={process.lprup} source={xsec_source}"
        )
    print(f"merged {event_count} events from {len(files)} files into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
