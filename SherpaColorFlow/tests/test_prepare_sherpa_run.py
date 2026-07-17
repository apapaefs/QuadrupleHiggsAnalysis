#!/usr/bin/env python3
"""Tests for the Sherpa run preparation helper."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_sherpa_run.py"


class PrepareSherpaRunTests(unittest.TestCase):
    def test_script_can_start_on_python36(self) -> None:
        self.assertNotIn("from __future__ import annotations", SCRIPT.read_text())

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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
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

    def test_seeded_runner_refuses_nonempty_outbase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "gg8b_seeded"
            subprocess.run(
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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            (run_dir / "Process").mkdir()
            (run_dir / "Results_PartiallyUnweighted").mkdir()
            existing_job = run_dir / "events_more_20k" / "job_0001"
            existing_job.mkdir(parents=True)
            (existing_job / "gg_4bbbar_120evt_4321.lhe").write_text("</LesHouchesEvents>\n")

            fake_bin = Path(tmp) / "bin"
            fake_bin.mkdir()
            fake_sherpa = fake_bin / "Sherpa"
            fake_sherpa.write_text("#!/bin/sh\necho Sherpa should not run >&2\nexit 99\n")
            fake_sherpa.chmod(0o755)

            env = os.environ.copy()
            env["OUTBASE"] = "events_more_20k"
            env["BASE_SEED"] = "4321"
            env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(
                [str(run_dir / "run_seeded_generation.sh"), "20000", "164"],
                cwd=str(run_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("Refusing to use non-empty OUTBASE", result.stderr)
            self.assertFalse((existing_job / "sherpa_4321.log").exists())

    def test_seeded_runner_copies_ufo_parameter_card_to_each_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "gg8b_seeded"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "gg8b",
                    str(run_dir),
                    "--total-events",
                    "2",
                    "--seeded-jobs",
                    "2",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            parameter_card = run_dir / "param_heft_c3d4_sherpa.dat"
            parameter_card.write_text("BLOCK SMINPUTS\n  3  1.190000e-01\n")
            with (run_dir / "Sherpa.yaml").open("a") as card:
                card.write("\nUFO_PARAM_CARD: param_heft_c3d4_sherpa.dat\n")
            (run_dir / "Process").mkdir()
            (run_dir / "Results_PartiallyUnweighted").mkdir()

            fake_bin = Path(tmp) / "bin"
            fake_bin.mkdir()
            fake_sherpa = fake_bin / "Sherpa"
            fake_sherpa.write_text(
                "#!/bin/sh\n"
                "test -f param_heft_c3d4_sherpa.dat || exit 42\n"
                "exit 0\n"
            )
            fake_sherpa.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(
                [str(run_dir / "run_seeded_generation.sh"), "2", "2"],
                cwd=str(run_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for job in ("job_0001", "job_0002"):
                copied = run_dir / "events" / job / parameter_card.name
                self.assertEqual(copied.read_bytes(), parameter_card.read_bytes())

    def test_seeded_runner_fails_before_launch_when_ufo_parameter_card_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "gg8b_seeded"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "gg8b",
                    str(run_dir),
                    "--total-events",
                    "1",
                    "--seeded-jobs",
                    "1",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            with (run_dir / "Sherpa.yaml").open("a") as card:
                card.write("\nUFO_PARAM_CARD: param_heft_c3d4_sherpa.dat\n")
            (run_dir / "Process").mkdir()
            (run_dir / "Results_PartiallyUnweighted").mkdir()

            result = subprocess.run(
                [str(run_dir / "run_seeded_generation.sh"), "1", "1"],
                cwd=str(run_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "Missing UFO_PARAM_CARD 'param_heft_c3d4_sherpa.dat' referenced by Sherpa.yaml",
                result.stderr,
            )
            self.assertFalse((run_dir / "events").exists())


if __name__ == "__main__":
    unittest.main()
