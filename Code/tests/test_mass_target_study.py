from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "mass_target_study.py"
SPEC = importlib.util.spec_from_file_location("mass_target_study", MODULE_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


class TargetPointTests(unittest.TestCase):
    def test_small_preset_is_frozen_and_contains_shared_baseline(self) -> None:
        points = study.small_target_points()
        values = [point.values for point in points]
        self.assertEqual(len(values), 19)
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(
            study.BASELINE_TARGETS["resonant"],
            study.BASELINE_TARGETS["nonresonant"],
        )
        self.assertIn(study.BASELINE_TARGETS["resonant"], values)

    def test_none_preset_still_keeps_shared_comparison_baseline(self) -> None:
        points = study.resolve_target_points("none", [])
        self.assertEqual(
            {point.values for point in points},
            set(study.BASELINE_TARGETS.values()),
        )

    def test_target_parser_enforces_candidate_rank_contract(self) -> None:
        point = study.parse_target_point("120, 115, 110, 105")
        self.assertEqual(point.target_id, "m120_115_110_105")
        self.assertEqual(point.cli_value, "120,115,110,105")
        for invalid in (
            "120,115,110",
            "120,115,110,0",
            "120,115,nan,105",
            "105,110,115,120",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(study.StudyInputError):
                    study.parse_target_point(invalid)


class MetricTests(unittest.TestCase):
    def test_weighted_auc_handles_ordering_and_ties(self) -> None:
        weights = np.ones(2)
        self.assertEqual(
            study.weighted_auc(
                np.asarray([2.0, 3.0]),
                np.asarray([0.0, 1.0]),
                weights,
                weights,
            ),
            1.0,
        )
        self.assertEqual(
            study.weighted_auc(
                np.asarray([0.0, 1.0]),
                np.asarray([2.0, 3.0]),
                weights,
                weights,
            ),
            0.0,
        )
        self.assertEqual(
            study.weighted_auc(
                np.asarray([1.0, 1.0]),
                np.asarray([1.0, 1.0]),
                weights,
                weights,
            ),
            0.5,
        )

    def test_split_is_deterministic_and_source_local(self) -> None:
        first = [
            study._is_validation_event(
                "sample-a", index, fraction=0.35, seed=8128
            )
            for index in range(100)
        ]
        repeated = [
            study._is_validation_event(
                "sample-a", index, fraction=0.35, seed=8128
            )
            for index in range(100)
        ]
        other_sample = [
            study._is_validation_event(
                "sample-b", index, fraction=0.35, seed=8128
            )
            for index in range(100)
        ]
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_sample)
        self.assertGreater(sum(first), 0)
        self.assertLess(sum(first), len(first))


class ManifestAndJobTests(unittest.TestCase):
    def test_manifest_expansion_builds_both_extractor_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "samples.csv"
            manifest.write_text(
                "sample_id,class,workflow,raw_root\n"
                "rsig,signal,resonant,rsig.root\n"
                "nsig,signal,nonresonant,nsig.root\n"
                "bkg,background,both,bkg.root\n",
                encoding="utf-8",
            )
            samples = study.load_sample_manifest(
                manifest, root, study.WORKFLOWS
            )
            targets = study.resolve_target_points("none", [])
            jobs = study.build_extraction_jobs(
                output_dir=root / "output",
                targets=targets,
                samples=samples,
                workflows=study.WORKFLOWS,
                resonant_executable=root / "resonant",
                nonresonant_executable=root / "nonresonant",
                max_events=100,
            )
            self.assertEqual(len(jobs), 4)
            resonant = next(job for job in jobs if job.workflow == "resonant")
            nonresonant = next(
                job for job in jobs if job.workflow == "nonresonant"
            )
            self.assertIn("--output", resonant.command)
            self.assertIn("--max-reco-jets", resonant.command)
            self.assertIn("--higgs-mass-targets", resonant.command)
            self.assertIsNone(resonant.input_list)
            self.assertEqual(nonresonant.command[2], "-t")
            self.assertIn("--higgs-mass-targets", nonresonant.command)
            self.assertIsNotNone(nonresonant.input_list)
            self.assertTrue(
                nonresonant.output_root.name.endswith(
                    "_var.smearCMS.root"
                )
            )

    def test_manifest_requires_both_classes_for_each_requested_workflow(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "samples.csv"
            manifest.write_text(
                "sample_id,class,workflow,raw_root\n"
                "sig,signal,resonant,signal.root\n",
                encoding="utf-8",
            )
            with self.assertRaises(study.StudyInputError):
                study.load_sample_manifest(
                    manifest, root, ("resonant",)
                )

    def test_stale_executable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "extractor"
            source = root / "extractor.cc"
            executable.write_bytes(b"binary")
            source.write_text("// source\n", encoding="utf-8")
            os.utime(executable, ns=(1_000_000_000, 1_000_000_000))
            os.utime(source, ns=(2_000_000_000, 2_000_000_000))
            with self.assertRaisesRegex(
                SystemExit, "executable is older"
            ):
                study._validate_executable(
                    executable, (source,), "resonant"
                )


class EvaluationTests(unittest.TestCase):
    def test_selection_uses_tune_split_and_reports_validation_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = study.TargetPoint(
                study.BASELINE_TARGETS["nonresonant"]
            )
            alternative = study.TargetPoint((125.0, 125.0, 125.0, 125.0))
            signal = study.SampleSpec(
                "signal",
                "signal",
                "nonresonant",
                root / "signal.raw.root",
                0,
                0,
                8,
                1.0,
            )
            background = study.SampleSpec(
                "background",
                "background",
                "nonresonant",
                root / "background.raw.root",
                0,
                0,
                8,
                1.0,
            )
            jobs = study.build_extraction_jobs(
                output_dir=root / "output",
                targets=(baseline, alternative),
                samples=(signal, background),
                workflows=("nonresonant",),
                resonant_executable=root / "resonant",
                nonresonant_executable=root / "nonresonant",
                max_events=100,
            )
            for job in jobs:
                job.output_root.parent.mkdir(parents=True, exist_ok=True)
                job.output_root.touch()

            event_index = np.arange(100, dtype=np.int64)
            masses = np.tile(
                np.asarray([120.0, 115.0, 110.0, 105.0]), (100, 1)
            )

            def fake_load(job, _root_module):
                if job.target == alternative:
                    score = (
                        np.ones(100)
                        if job.sample.label == "signal"
                        else np.zeros(100)
                    )
                else:
                    score = np.full(100, 0.5)
                return study.Observations(
                    sample_id=job.sample.sample_id,
                    label=job.sample.label,
                    event_index=event_index,
                    score=score,
                    weight=np.ones(100),
                    candidate_masses=masses,
                )

            class Root:
                class gROOT:
                    @staticmethod
                    def SetBatch(_value):
                        return None

            with patch.object(
                study, "load_root_observations", side_effect=fake_load
            ):
                summary = study.evaluate_jobs(
                    jobs,
                    output_dir=root / "output",
                    validation_fraction=0.35,
                    split_seed=8128,
                    bootstrap_repetitions=20,
                    bootstrap_seed=123,
                    make_plots=False,
                    root_module=Root,
                )

            result = summary["recommendations"]["nonresonant"]
            self.assertEqual(
                result["selected_targets_gev"],
                [125.0, 125.0, 125.0, 125.0],
            )
            self.assertEqual(result["validation_auc"], 1.0)
            self.assertEqual(result["baseline_validation_auc"], 0.5)
            self.assertEqual(
                result["validation_auc_difference_from_baseline"], 0.5
            )
            self.assertEqual(
                result["decision"], "shortlist_for_full_analysis"
            )
            self.assertTrue((root / "output" / "metrics.csv").is_file())
            self.assertTrue(
                (root / "output" / "sample_metrics.csv").is_file()
            )
            report = (root / "output" / "REPORT.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("not a final sensitivity optimization", report)
            persisted = json.loads(
                (root / "output" / "study_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                persisted["recommendations"]["nonresonant"][
                    "selected_target_id"
                ],
                alternative.target_id,
            )


if __name__ == "__main__":
    unittest.main()
