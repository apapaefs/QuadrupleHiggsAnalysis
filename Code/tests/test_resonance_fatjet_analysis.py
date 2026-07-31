from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import resonance_fatjet_xgboost_analysis as fat
import resonance_xgboost_analysis as resolved


class FatJetStatisticsTests(unittest.TestCase):
    @staticmethod
    def _template_background(
        nominal: np.ndarray, conservative: np.ndarray
    ) -> SimpleNamespace:
        events = np.arange(len(nominal), dtype=np.int64)
        return SimpleNamespace(
            table=SimpleNamespace(arrays={"event_index": events}),
            scenario_weights={
                "nominal": np.asarray(nominal, dtype=float),
                "conservative": np.asarray(conservative, dtype=float),
            },
        )

    def test_grouped_sumw2_combines_hypotheses_before_squaring(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.7, 0.6])
        weights = np.asarray([0.2, 0.3, 0.4, 0.5])
        events = np.asarray([10, 10, 20, 20])
        summary = fat.grouped_binned_summary(scores, weights, events, [0.0, 1.0])
        self.assertAlmostEqual(float(summary["yield"][0]), 1.4)
        self.assertAlmostEqual(float(summary["sumw2"][0]), 0.5**2 + 0.9**2)
        self.assertEqual(int(summary["raw"][0]), 2)

    def test_threshold_scan_groups_each_event_at_each_threshold(self) -> None:
        result = fat.grouped_threshold_scan(
            np.asarray([0.9, 0.8, 0.7, 0.6]),
            np.asarray([0.2, 0.3, 0.4, 0.5]),
            np.asarray([1, 1, 2, 2]),
            np.asarray([0.0, 0.65, 0.85, 1.0]),
        )
        np.testing.assert_allclose(result["yield"], [1.4, 0.9, 0.2, 0.0])
        np.testing.assert_allclose(result["sumw2"], [1.06, 0.41, 0.04, 0.0])
        np.testing.assert_array_equal(result["raw"], [2, 2, 1, 0])

    @staticmethod
    def _threshold_inputs(
        neff: list[float], raw: list[int] | None = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        thresholds = np.asarray([0.2, 0.8])
        signal = {
            "yield": np.asarray([5.0, 4.0]),
            "sumw2": np.asarray([1.0, 1.0]),
            "raw": np.asarray([40, 30]),
            "neff": np.asarray([25.0, 16.0]),
        }
        background = {
            "yield": np.asarray([10.0, 10.0]),
            "sumw2": np.asarray([10.0, 10.0]),
            "raw": np.asarray(raw if raw is not None else [30, 30]),
            "neff": np.asarray(neff),
        }
        return thresholds, signal, background

    def test_selective_neff_uses_primary_when_available(self) -> None:
        thresholds, signal, background = self._threshold_inputs([12.0, 6.0])
        result = fat._select_fast_threshold(
            thresholds, signal, background, 25, 10.0, 5.0
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["threshold_selection_tier"], "primary")
        self.assertFalse(result["used_neff_fallback"])
        self.assertEqual(result["required_background_neff"], 10.0)
        self.assertEqual(result["primary_valid_threshold_count"], 1)
        self.assertFalse(result["fallback_threshold_scan_attempted"])
        self.assertIsNone(result["fallback_valid_threshold_count"])

    def test_selective_neff_retries_at_fallback_only_after_primary_fails(self) -> None:
        thresholds, signal, background = self._threshold_inputs([8.0, 6.0])
        result = fat._select_fast_threshold(
            thresholds, signal, background, 25, 10.0, 5.0
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["threshold_selection_tier"], "fallback")
        self.assertTrue(result["used_neff_fallback"])
        self.assertEqual(result["required_background_neff"], 5.0)
        self.assertEqual(result["primary_valid_threshold_count"], 0)
        self.assertTrue(result["fallback_threshold_scan_attempted"])
        self.assertEqual(result["fallback_valid_threshold_count"], 2)

    def test_selective_neff_remains_invalid_when_both_tiers_fail(self) -> None:
        thresholds, signal, background = self._threshold_inputs([4.0, 3.0])
        result = fat._select_fast_threshold(
            thresholds, signal, background, 25, 10.0, 5.0
        )
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["threshold_selection_tier"], "none")
        self.assertFalse(result["used_neff_fallback"])
        self.assertEqual(result["primary_valid_threshold_count"], 0)
        self.assertTrue(result["fallback_threshold_scan_attempted"])
        self.assertEqual(result["fallback_valid_threshold_count"], 0)

    def test_selective_neff_does_not_relax_unique_event_requirement(self) -> None:
        thresholds, signal, background = self._threshold_inputs(
            [8.0, 6.0], raw=[24, 24]
        )
        result = fat._select_fast_threshold(
            thresholds, signal, background, 25, 10.0, 5.0
        )
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["fallback_valid_threshold_count"], 0)

    def test_template_binning_uses_audited_inclusive_fallback(self) -> None:
        scores = np.linspace(0.1, 0.9, 5)
        background = self._template_background(np.ones(5), np.ones(5))
        edges, audit = fat._select_template_edges(
            [scores],
            [background],
            [np.ones(5, dtype=bool)],
            min_raw=25,
            min_neff=10.0,
            allow_inclusive_fallback=True,
        )
        self.assertEqual(edges, [0.0, 1.0])
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["fallback_level"], "inclusive_1_bin")
        self.assertTrue(audit["thresholds_relaxed"])
        self.assertTrue(audit["inclusive_fallback_attempted"])
        self.assertTrue(audit["inclusive_fallback_used"])
        self.assertEqual(audit["selected_bin_count"], 1)
        self.assertEqual(
            audit["shape_requirements"]["minimum_raw_unique_events_per_bin"], 25
        )
        self.assertEqual(
            audit["shape_requirements"]["minimum_background_neff_per_bin"],
            10.0,
        )
        self.assertEqual(
            audit["inclusive_fallback_requirements"]["minimum_raw_unique_events"],
            1,
        )
        self.assertTrue(
            audit["inclusive_scenario_validation"]["nominal"]["valid"]
        )
        self.assertTrue(
            audit["inclusive_scenario_validation"]["conservative"]["valid"]
        )

    def test_template_binning_excludes_low_mc_category_by_default(self) -> None:
        scores = np.linspace(0.1, 0.9, 5)
        background = self._template_background(np.ones(5), np.ones(5))
        edges, audit = fat._select_template_edges(
            [scores],
            [background],
            [np.ones(5, dtype=bool)],
            min_raw=25,
            min_neff=10.0,
        )
        self.assertIsNone(edges)
        self.assertEqual(audit["status"], "invalid")
        self.assertFalse(audit["inclusive_fallback_attempted"])
        self.assertIn("inclusive fallback disabled", audit["reason"])

    def test_supported_likelihood_uses_resolved_category_at_every_point(self) -> None:
        self.assertTrue(fat._category_in_likelihood("resolved", "exclude"))
        self.assertFalse(fat._category_in_likelihood("mixed", "exclude"))
        self.assertFalse(fat._category_in_likelihood("boosted", "exclude"))
        for category in fat.CATEGORY_NAMES:
            self.assertTrue(
                fat._category_in_likelihood(category, "inclusive-diagnostic")
            )

    def test_template_binning_rejects_invalid_inclusive_scenario(self) -> None:
        scores = np.linspace(0.1, 0.9, 5)
        background = self._template_background(np.ones(5), -np.ones(5))
        edges, audit = fat._select_template_edges(
            [scores],
            [background],
            [np.ones(5, dtype=bool)],
            min_raw=25,
            min_neff=10.0,
            allow_inclusive_fallback=True,
        )
        self.assertIsNone(edges)
        self.assertEqual(audit["status"], "invalid")
        self.assertEqual(audit["fallback_level"], "none")
        self.assertTrue(audit["inclusive_fallback_attempted"])
        self.assertFalse(audit["inclusive_fallback_used"])
        self.assertTrue(
            audit["inclusive_scenario_validation"]["nominal"]["valid"]
        )
        self.assertFalse(
            audit["inclusive_scenario_validation"]["conservative"]["valid"]
        )

    def test_template_statistics_marks_complete_low_neff_fallback_diagnostic(self) -> None:
        point = resolved.MassPoint("direct", ms=1000.0)
        binning_audit = {
            category: {
                **{
                    str(fold): {
                        "status": "ok",
                        "fallback_level": "inclusive_1_bin",
                        "thresholds_relaxed": True,
                    }
                    for fold in range(fat.N_FOLDS)
                },
                "status": "ok",
                "all_rotations_valid": True,
            }
            for category in fat.CATEGORY_NAMES
        }
        channels = [
            {
                "name": f"{category}_fold{fold}",
                "category": category,
                "fold": fold,
                "edges": [0.0, 1.0],
                "background": [2.0],
                "background_sumw2": [2.0],
                "background_raw_unique": [2],
                "background_neff": [2.0],
            }
            for category in fat.CATEGORY_NAMES
            for fold in range(fat.N_FOLDS)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            templates = output / "templates"
            templates.mkdir()
            (templates / f"{point.point_id}.json").write_text(
                json.dumps(
                    {
                        "core_fingerprint": "core",
                        "template_binning_version": fat.TEMPLATE_BINNING_VERSION,
                        "pyhf_low_mc_policy": "inclusive-diagnostic",
                        "binning_audit": binning_audit,
                        "channels": {
                            "nominal": channels,
                            "conservative": channels,
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = fat._collate_template_mc_statistics(
                output,
                [SimpleNamespace(spec=SimpleNamespace(point=point))],
                templates,
                "core",
                min_raw=25,
                min_neff=10.0,
                low_mc_policy="inclusive-diagnostic",
            )
        self.assertTrue(summary["all_categories_retained"])
        self.assertFalse(summary["primary_template_statistics_satisfied"])
        self.assertEqual(summary["rotation_counts"]["inclusive_1_bin"], 15)
        self.assertEqual(summary["channel_bin_count"], 30)
        self.assertEqual(
            summary["statistical_interpretation"], "diagnostic_existing_mc_only"
        )

    def test_hypothesis_probabilities_close_for_both_working_points(self) -> None:
        # Two retained candidates: one genuine and one fake.  The four rows are
        # their complete pass/fail bitmask enumeration.
        arrays = {
            "n_true_fat_pass": np.asarray([0, 1, 0, 1]),
            "n_true_fat_fail": np.asarray([1, 0, 1, 0]),
            "n_fake_fat_pass": np.asarray([0, 0, 1, 1]),
            "n_fake_fat_fail": np.asarray([1, 1, 0, 0]),
            "n_true_single": np.zeros(4, dtype=int),
            "n_c_mistag": np.zeros(4, dtype=int),
            "n_light_mistag": np.zeros(4, dtype=int),
        }
        table = resolved.EventTable(arrays, 1, 1.0, 1.0, 1, 1.0, {})
        for working_point in fat.TAGGING_SCENARIOS.values():
            factors = fat.tag_hypothesis_factor(
                table,
                eps_bb=working_point["eps_bb"],
                fake_bb=working_point["fake_bb"],
                eps_b=0.85,
                eps_c=0.10,
                eps_light=0.01,
            )
            self.assertAlmostEqual(float(np.sum(factors)), 1.0, places=14)

    def test_duplicate_hypotheses_share_fold_and_partition(self) -> None:
        events = np.asarray([2, 2, 2, 5, 5, 9])
        folds = fat.grouped_folds("sample", events, 12345)
        partitions = fat.analysis_partition("sample", events, 12345)
        for event in np.unique(events):
            self.assertEqual(len(np.unique(folds[events == event])), 1)
            self.assertEqual(len(np.unique(partitions[events == event])), 1)

    def test_score_cache_rejects_stale_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            point = resolved.MassPoint("direct", ms=600.0)
            path = fat._score_cache_path(base, point.point_id, "sample")
            path.parent.mkdir(parents=True)
            np.savez_compressed(
                path,
                core_fingerprint=np.asarray("old"),
                test=np.asarray([0.5]),
                validation=np.asarray([0.5]),
            )
            spec = resolved.SampleSpec(
                "sample", "signal", Path("unused"), Path("unused"), 1,
                1.0, 1.0, "test", 1.0, 4, 1.0, 0, 0, 1, "unique", point,
            )
            table = resolved.EventTable(
                {"weight": np.asarray([1.0])}, 1, 1.0, 1.0, 1, 1.0, {}
            )
            sample = resolved.LoadedSample(
                spec, table, np.asarray([0]), np.zeros((1, 1)), ("x",), {"nominal": np.ones(1)}
            )
            with self.assertRaises(resolved.AnalysisInputError):
                fat.load_or_predict_scores(sample, point, [], base, "new")

    def test_limit_completion_keeps_partial_medians_but_not_missing_limits(self) -> None:
        complete = fat._limit_completion_summary(
            [
                {"status": "ok", "expected_median": 1.0},
                {"status": "partial", "expected_median": 2.0},
            ],
            2,
        )
        self.assertTrue(complete["computationally_complete"])
        self.assertTrue(complete["median_limits_complete"])
        self.assertFalse(complete["all_expected_bands_complete"])
        incomplete = fat._limit_completion_summary(
            [
                {"status": "ok", "expected_median": 1.0},
                {"status": "unbounded", "expected_median": None},
            ],
            2,
        )
        self.assertTrue(incomplete["computationally_complete"])
        self.assertFalse(incomplete["median_limits_complete"])

    def test_background_normalization_audit_is_required_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "backgrounds.csv"
            manifest.write_text(
                "sample_id,cross_section_fb,normalization_source\n"
                "HW-gg_to_4b_2c_2j,2751.78,source_lhe_init\n",
                encoding="utf-8",
            )
            provisional = fat._background_normalization_provenance(manifest)
            self.assertEqual(provisional["status"], "provisional")
            audit = {
                "schema": "resonance-background-normalization-audit-v1",
                "output_manifest_sha256": fat._sha256_file(manifest),
                "adopted_samples": [
                    {
                        "sample_id": "HW-gg_to_4b_2c_2j",
                        "adopted_cross_section_fb": 2751.78,
                        "relative_uncertainty": 0.0747327,
                        "source_lhe": "sample.lhe",
                        "source_lhe_sha256": "abc",
                    }
                ],
            }
            manifest.with_suffix(".normalization_audit.json").write_text(
                json.dumps(audit), encoding="utf-8"
            )
            accepted = fat._background_normalization_provenance(manifest)
            self.assertEqual(accepted["status"], "accepted")
            self.assertEqual(
                accepted["adopted_samples"][0]["sample_id"],
                "HW-gg_to_4b_2c_2j",
            )

    def test_fast_module_has_no_pyhf_import(self) -> None:
        source = inspect.getsource(fat)
        self.assertNotIn("import pyhf", source)


if __name__ == "__main__":
    unittest.main()
