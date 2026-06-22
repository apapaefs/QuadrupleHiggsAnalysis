#!/usr/bin/env python3
"""Tests for the Sherpa run preparation helper."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_sherpa_run.py"


class PrepareSherpaRunTests(unittest.TestCase):
    def test_seeded_jobs_writes_runner_and_prints_adjustable_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "gg8b_seeded"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "gg8b",
                    str(run_dir),
                    "--total-events",
                    "120",
                    "--np",
                    "8",
                    "--output-prefix",
                    "gg_4bbbar_120evt",
                    "--seeded-jobs",
                    "6",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            runner = run_dir / "run_seeded_generation.sh"
            self.assertTrue(runner.exists())
            self.assertTrue(os.access(runner, os.X_OK))

            runner_text = runner.read_text()
            self.assertIn('TOTAL_EVENTS="${1:-120}"', runner_text)
            self.assertIn('JOBS="${2:-6}"', runner_text)
            self.assertIn('OUTPUT_STEM="${OUTPUT_STEM:-gg_4bbbar_120evt}"', runner_text)
            self.assertIn("Results_PartiallyUnweighted", runner_text)
            self.assertIn('"MPI_EVENT_MODE: 0"', runner_text)

            self.assertIn("Integration command before seeded generation:", result.stdout)
            self.assertIn("Sherpa -e 0 Sherpa.yaml", result.stdout)
            self.assertIn("Seeded single-rank generation command:", result.stdout)
            self.assertIn("./run_seeded_generation.sh 120 6", result.stdout)


if __name__ == "__main__":
    unittest.main()
