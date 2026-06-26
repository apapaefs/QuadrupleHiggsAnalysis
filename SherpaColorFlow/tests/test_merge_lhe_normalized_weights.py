#!/usr/bin/env python3
"""Tests for source-aware LHE weight normalization."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_lhe_normalized_weights.py"


def lhe_header(xsec: float = 1.0, xerr: float = 0.1, idwtup: int = 3) -> str:
    return f"""<LesHouchesEvents version="3.0">
<header>
  <generator name="Sherpa"/>
</header>
<init>
  2212 2212 7.0000000000e+03 7.0000000000e+03 0 0 0 0 {idwtup} 1
  {xsec} {xerr} 1.0 1
</init>
"""


def event_block(label: int, weight: float) -> str:
    return f"""<event>
  2 1 {weight} 91.188 0.007297 0.118
  21 -1 0 0 501 502 0 0 7000 7000 0 0 9
  21 -1 0 0 503 504 0 0 -7000 7000 0 0 9
  # event {label}
</event>
"""


def write_lhe(path: Path, weight: float, label: int) -> None:
    path.write_text(lhe_header() + event_block(label, weight) + "</LesHouchesEvents>\n")


def write_lhe_with_idwtup(path: Path, weight: float, label: int, idwtup: int) -> None:
    path.write_text(lhe_header(idwtup=idwtup) + event_block(label, weight) + "</LesHouchesEvents>\n")


def event_weights(text: str) -> list[float]:
    weights: list[float] = []
    pos = 0
    while True:
        start = text.find("<event", pos)
        if start < 0:
            return weights
        start = text.find(">", start)
        end = text.find("</event>", start)
        lines = [
            line.strip()
            for line in text[start + 1:end].splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        weights.append(float(lines[0].split()[2]))
        pos = end + len("</event>")


class MergeLheNormalizedWeightsTests(unittest.TestCase):
    def test_count_fraction_rescales_each_source_and_total_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_a = base / "source_a"
            source_b = base / "source_b"
            source_a.mkdir()
            source_b.mkdir()
            write_lhe(source_a / "a1.lhe", 2.0, 1)
            write_lhe(source_a / "a2.lhe", 4.0, 2)
            write_lhe(source_b / "b1.lhe", 10.0, 3)

            output = base / "merged.lhe"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source_a),
                    str(source_b),
                    "--total-xsec",
                    "0.9",
                    "--total-xerr",
                    "0.0123",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            merged = output.read_text()
            weights = event_weights(merged)
            self.assertEqual(len(weights), 3)
            self.assertEqual([round(weight, 12) for weight in weights], [0.2, 0.4, 0.3])
            self.assertAlmostEqual(sum(weights), 0.9, places=12)
            self.assertIn("0.9 0.0123 1 1\n</init>", merged)
            self.assertIn("merged 3 events from 2 source groups", result.stdout)
            self.assertIn("source_a", result.stdout)
            self.assertIn("source_b", result.stdout)

    def test_zero_source_weight_sum_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            write_lhe(source / "zero.lhe", 0.0, 1)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    "--total-xsec",
                    "0.9",
                    "--output",
                    str(base / "merged.lhe"),
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("raw weight sum is zero", result.stderr)

    def test_writes_requested_idwtup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            write_lhe_with_idwtup(source / "weighted.lhe", 2.0, 1, idwtup=1)

            output = base / "merged.lhe"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    "--total-xsec",
                    "0.9",
                    "--output",
                    str(output),
                    "--idwtup",
                    "3",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            init_lines = [
                line.strip()
                for line in output.read_text().splitlines()
                if line.strip() and not line.strip().startswith("<")
            ]
            self.assertEqual(init_lines[0].split()[-2:], ["3", "1"])


if __name__ == "__main__":
    unittest.main()
