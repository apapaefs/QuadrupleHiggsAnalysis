#!/usr/bin/env python3
"""Tests for merging sharded LHE output."""

from __future__ import annotations

import gzip
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_lhe_shards.py"


def lhe_header(xsec: float = 1.0, xerr: float = 0.1) -> str:
    return f"""<LesHouchesEvents version="3.0">
<header>
  <generator name="Sherpa"/>
</header>
<init>
  2212 2212 7.0000000000e+03 7.0000000000e+03 0 0 0 0 3 1
  {xsec} {xerr} 1.0 1
</init>
"""


def event_block(label: int) -> str:
    return f"""<event>
  2 1 1.0 91.188 0.007297 0.118
  21 -1 0 0 501 502 0 0 7000 7000 0 0 9
  21 -1 0 0 503 504 0 0 -7000 7000 0 0 9
  # event {label}
</event>
"""


class MergeLheShardsTests(unittest.TestCase):
    def test_merges_complete_events_with_single_header_and_footer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            events = base / "events"
            events.mkdir()
            (events / "a.lhe").write_text(
                lhe_header() + event_block(1) + "</LesHouchesEvents>\n"
            )
            with gzip.open(events / "b.lhe.gz", "wt") as handle:
                handle.write(lhe_header() + event_block(2) + event_block(3))
                handle.write("<event>\n  incomplete\n")

            output = base / "merged.lhe"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(events),
                    "--output",
                    str(output),
                    "--expected-events",
                    "3",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            merged = output.read_text()
            self.assertIn("merged 3 events from 2 files", result.stdout)
            self.assertIn("WARNING", result.stderr)
            self.assertEqual(merged.count("<LesHouchesEvents"), 1)
            self.assertEqual(merged.count("<header>"), 1)
            self.assertEqual(merged.count("<init>"), 1)
            self.assertEqual(merged.count("<event>"), 3)
            self.assertEqual(merged.count("</event>"), 3)
            self.assertEqual(merged.count("</LesHouchesEvents>"), 1)
            self.assertTrue(merged.rstrip().endswith("</LesHouchesEvents>"))

    def test_expected_event_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "a.lhe").write_text(lhe_header() + event_block(1))

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(base),
                    "--output",
                    str(base / "merged.lhe"),
                    "--expected-events",
                    "2",
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 2 events, merged 1", result.stderr)

    def test_keeps_first_cross_section_and_validates_matching_inits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            events = base / "events"
            events.mkdir()
            (events / "a.lhe").write_text(
                lhe_header(2.5, 0.25) + event_block(1) + "</LesHouchesEvents>\n"
            )
            (events / "b.lhe").write_text(
                lhe_header(2.5, 0.25) + event_block(2) + "</LesHouchesEvents>\n"
            )

            output = base / "merged.lhe"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(events),
                    "--output",
                    str(output),
                    "--expected-events",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            merged = output.read_text()
            self.assertIn("cross section: process 1 XSECUP=2.5", result.stdout)
            self.assertIn("  2.5 0.25 1.0 1", merged)
            self.assertNotIn("  5.0 0.5", merged)

    def test_uses_sherpa_log_cross_section_when_lhe_init_is_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for idx, seed in enumerate((1234, 2468), start=1):
                job = base / f"job_{idx:04d}"
                job.mkdir()
                (job / f"events_{seed}.lhe").write_text(
                    lhe_header(1.0, 1.0) + event_block(idx) + "</LesHouchesEvents>\n"
                )
                (job / f"sherpa_{seed}.log").write_text(
                    "proc : 0.00108292 pb +- ( 2.83736e-05 pb = 2.6201 % ) exp. eff: 0.00040487 %\n"
                )

            output = base / "merged.lhe"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(base),
                    "--output",
                    str(output),
                    "--expected-events",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            merged = output.read_text()
            self.assertIn("cross section: process 1 XSECUP=0.00108292", result.stdout)
            self.assertIn("source=Sherpa log", result.stdout)
            self.assertIn("0.00108292", merged)
            self.assertIn("2.83736e-05", merged)
            self.assertIn("0.00108292 2.83736e-05 1 1\n</init>", merged)
            self.assertNotIn("  1.0 1.0 1.0 1", merged)

    def test_cross_section_mismatch_fails_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            events = base / "events"
            events.mkdir()
            (events / "a.lhe").write_text(
                lhe_header(2.5, 0.25) + event_block(1) + "</LesHouchesEvents>\n"
            )
            (events / "b.lhe").write_text(
                lhe_header(3.0, 0.30) + event_block(2) + "</LesHouchesEvents>\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(events),
                    "--output",
                    str(base / "merged.lhe"),
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cross-section mismatch", result.stderr)

    def test_fix_unclosed_inputs_repairs_missing_footer_and_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "unclosed.lhe"
            source.write_text(lhe_header() + event_block(1) + "<event>\n  incomplete\n")

            output = base / "merged.lhe"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    "--output",
                    str(output),
                    "--expected-events",
                    "1",
                    "--fix-unclosed-inputs",
                    "--no-backup",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            repaired = source.read_text()
            self.assertIn("WARNING", result.stderr)
            self.assertIn("missing </LesHouchesEvents> footer", result.stderr)
            self.assertIn("fixed", result.stderr)
            self.assertNotIn("incomplete", repaired)
            self.assertTrue(repaired.rstrip().endswith("</LesHouchesEvents>"))
            self.assertEqual(output.read_text().count("<event>"), 1)


if __name__ == "__main__":
    unittest.main()
