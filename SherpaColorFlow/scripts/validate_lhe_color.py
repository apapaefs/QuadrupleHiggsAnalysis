#!/usr/bin/env python3
"""Validate LHE mass shells and colour-flow topology."""

from __future__ import annotations

import argparse
import glob
import gzip
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


COLORED = set(range(1, 7)) | {21}


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def expand_inputs(items: Iterable[str], prefix: str | None) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        matches = [Path(p) for p in glob.glob(item)] or [Path(item)]
        for path in matches:
            if path.is_dir():
                patterns = [f"{prefix}*.lhe", f"{prefix}*.lhe.gz"] if prefix else ["*.lhe", "*.lhe.gz"]
                for pattern in patterns:
                    for candidate in path.rglob(pattern):
                        resolved = candidate.resolve()
                        if resolved not in seen:
                            seen.add(resolved)
                            files.append(candidate)
            elif path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(path)
    return sorted(files)


def parse_events(path: Path):
    with open_text(path) as handle:
        text = handle.read()
    pos = 0
    while True:
        start = text.find("<event", pos)
        if start < 0:
            return
        start = text.find(">", start)
        end = text.find("</event>", start)
        if start < 0 or end < 0:
            raise ValueError(f"incomplete event block in {path}")
        lines = [
            line.strip()
            for line in text[start + 1:end].splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if lines:
            yield lines
        pos = end + len("</event>")


def parse_row(line: str) -> dict:
    parts = line.split()
    if len(parts) < 13:
        raise ValueError(f"short LHE particle row: {line}")
    return {
        "pid": int(parts[0]),
        "status": int(parts[1]),
        "mother1": int(parts[2]),
        "mother2": int(parts[3]),
        "c1": int(parts[4]),
        "c2": int(parts[5]),
        "px": float(parts[6]),
        "py": float(parts[7]),
        "pz": float(parts[8]),
        "e": float(parts[9]),
        "m": float(parts[10]),
        "line": line,
    }


def expected_single_quark_flow(row: dict) -> list[str]:
    pid = row["pid"]
    status = row["status"]
    c1 = row["c1"]
    c2 = row["c2"]
    if abs(pid) > 6:
        return []
    if (c1 != 0) == (c2 != 0):
        return [f"quark row must have exactly one nonzero colour tag: {row['line']}"]
    if status == 1:
        if pid > 0 and not (c1 != 0 and c2 == 0):
            return [f"final quark should carry colour1 only: {row['line']}"]
        if pid < 0 and not (c1 == 0 and c2 != 0):
            return [f"final antiquark should carry colour2 only: {row['line']}"]
    elif status == -1:
        if pid > 0 and not (c1 != 0 and c2 == 0):
            return [f"incoming quark should carry colour1 only: {row['line']}"]
        if pid < 0 and not (c1 == 0 and c2 != 0):
            return [f"incoming antiquark should carry colour2 only: {row['line']}"]
    return []


def check_mass(row: dict, label: str, idx: int, abs_tol: float, rel_tol: float) -> list[str]:
    m2 = row["e"] ** 2 - row["px"] ** 2 - row["py"] ** 2 - row["pz"] ** 2
    tol = max(abs_tol, rel_tol * max(1.0, abs(row["e"])))
    if m2 < -tol:
        return [f"{label}: row {idx} has negative mass^2 {m2}: {row['line']}"]
    mass = math.sqrt(max(0.0, m2))
    if abs(mass - abs(row["m"])) > tol:
        return [f"{label}: row {idx} mass mismatch: LHE {row['m']} vs kinematic {mass}"]
    return []


def isolated_final_qqbar_pairs(final: list[dict], endpoints: dict[int, list[tuple[str, int, dict]]], flav: int):
    pairs = []
    for q_index, q in enumerate(final, start=1):
        if q["pid"] != flav or q["c1"] == 0 or q["c2"] != 0:
            continue
        for aq_index, aq in enumerate(final, start=1):
            if aq["pid"] != -flav or aq["c1"] != 0 or aq["c2"] != q["c1"]:
                continue
            end_rows = {id(entry[2]) for entry in endpoints.get(q["c1"], [])}
            if end_rows == {id(q), id(aq)}:
                pairs.append((q_index, aq_index, q["c1"]))
    return pairs


def validate_event(rows: list[dict], label: str, args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    incoming = [r for r in rows if r["status"] == -1]
    final = [r for r in rows if r["status"] == 1]

    if args.expect_incoming is not None and len(incoming) != args.expect_incoming:
        errors.append(f"{label}: expected {args.expect_incoming} incoming rows, found {len(incoming)}")
    if args.expect_final_count is not None and len(final) != args.expect_final_count:
        errors.append(f"{label}: expected {args.expect_final_count} final rows, found {len(final)}")
    if args.expect_final_abs_pdg is not None:
        bad = [r["pid"] for r in final if abs(r["pid"]) != args.expect_final_abs_pdg]
        if bad:
            errors.append(f"{label}: final rows outside abs(PDG)={args.expect_final_abs_pdg}: {bad}")
    for pid in args.forbid_final_pdg:
        if any(r["status"] == 1 and r["pid"] == pid for r in rows):
            errors.append(f"{label}: forbidden final-state PDG {pid} is present")

    for idx, row in enumerate(rows, start=1):
        if row["status"] in args.mass_status:
            errors.extend(check_mass(row, label, idx, args.mass_abs_tol, args.mass_rel_tol))

    endpoints: dict[int, list[tuple[str, int, dict]]] = defaultdict(list)
    for idx, row in enumerate(rows, start=1):
        if row["status"] not in (-1, 1):
            continue
        pid = row["pid"]
        c1 = row["c1"]
        c2 = row["c2"]
        if abs(pid) in COLORED:
            if c1 == 0 and c2 == 0:
                errors.append(f"{label}: coloured row has zero colour flow: {row['line']}")
            if abs(pid) == 21:
                if c1 == 0 or c2 == 0:
                    errors.append(f"{label}: gluon has a missing colour tag: {row['line']}")
                if c1 == c2:
                    errors.append(f"{label}: gluon has identical colour/anticolour tags: {row['line']}")
            else:
                errors.extend(f"{label}: {err}" for err in expected_single_quark_flow(row))
        elif c1 != 0 or c2 != 0:
            errors.append(f"{label}: colourless row has nonzero colour flow: {row['line']}")
        if row["status"] == -1:
            if c1:
                endpoints[c1].append(("a", idx, row))
            if c2:
                endpoints[c2].append(("c", idx, row))
        else:
            if c1:
                endpoints[c1].append(("c", idx, row))
            if c2:
                endpoints[c2].append(("a", idx, row))

    for tag, ends in sorted(endpoints.items()):
        if len(ends) != 2:
            errors.append(f"{label}: colour tag {tag} appears {len(ends)} times, not twice")
            continue
        kinds = sorted(e[0] for e in ends)
        if kinds != ["a", "c"]:
            errors.append(f"{label}: colour tag {tag} connects {kinds}, not one colour and one anticolour")
        if ends[0][1] == ends[1][1]:
            errors.append(f"{label}: colour tag {tag} connects a particle to itself")

    if args.require_first_qqbar_singlet is not None:
        flav = args.require_first_qqbar_singlet
        first_q = next((r for r in final if r["pid"] == flav), None)
        first_aq = next((r for r in final if r["pid"] == -flav), None)
        if first_q is None or first_aq is None:
            errors.append(f"{label}: missing first final {flav}/{-flav} pair for singlet check")
        else:
            tag = first_q["c1"]
            ok = tag != 0 and first_q["c2"] == 0 and first_aq["c1"] == 0 and first_aq["c2"] == tag
            end_rows = {id(entry[2]) for entry in endpoints.get(tag, [])}
            if not ok or end_rows != {id(first_q), id(first_aq)}:
                errors.append(
                    f"{label}: first final {flav}/{-flav} pair is not an isolated singlet: "
                    f"q=({first_q['c1']},{first_q['c2']}), aq=({first_aq['c1']},{first_aq['c2']})"
                )

    if args.require_isolated_qqbar_singlet is not None:
        flav = args.require_isolated_qqbar_singlet
        pairs = isolated_final_qqbar_pairs(final, endpoints, flav)
        if not pairs:
            errors.append(f"{label}: no isolated final {flav}/{-flav} singlet line found")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="LHE files, directories, or shell globs")
    parser.add_argument("--prefix", help="when an input is a directory, only read files matching PREFIX*.lhe[.gz]")
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--expect-incoming", type=int, default=2)
    parser.add_argument("--expect-final-count", type=int)
    parser.add_argument("--expect-final-abs-pdg", type=int)
    parser.add_argument("--forbid-final-pdg", type=int, action="append", default=[])
    parser.add_argument("--require-first-qqbar-singlet", type=int)
    parser.add_argument("--require-isolated-qqbar-singlet", type=int)
    parser.add_argument("--mass-status", type=int, nargs="+", default=[-1, 1])
    parser.add_argument("--mass-abs-tol", type=float, default=1.0e-5)
    parser.add_argument("--mass-rel-tol", type=float, default=1.0e-6)
    parser.add_argument("--max-errors", type=int, default=80)
    args = parser.parse_args()

    files = expand_inputs(args.inputs, args.prefix)
    if not files:
        raise SystemExit("FAIL: no LHE files found")

    total = 0
    all_errors: list[str] = []
    for path in files:
        try:
            event_iter = parse_events(path)
            for lines in event_iter:
                total += 1
                try:
                    nup = int(lines[0].split()[0])
                    rows = [parse_row(line) for line in lines[1:1 + nup]]
                except Exception as exc:
                    all_errors.append(f"{path.name} event {total}: parse error: {exc}")
                    continue
                if len(rows) != nup:
                    all_errors.append(f"{path.name} event {total}: expected {nup} particle rows, parsed {len(rows)}")
                    continue
                all_errors.extend(validate_event(rows, f"{path.name} event {total}", args))
        except Exception as exc:
            all_errors.append(f"{path}: {exc}")

    if args.expected_events is not None and total != args.expected_events:
        all_errors.append(f"expected {args.expected_events} complete events, found {total}")

    if all_errors:
        print(f"FAIL: checked {total} events from {len(files)} file(s)")
        for err in all_errors[:args.max_errors]:
            print(" -", err)
        if len(all_errors) > args.max_errors:
            print(f" - ... {len(all_errors) - args.max_errors} more error(s)")
        return 1

    print(f"PASS: checked {total} events from {len(files)} file(s)")
    print("PASS: masses and colour-flow topology are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
