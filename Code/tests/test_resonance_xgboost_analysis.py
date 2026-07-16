from __future__ import annotations

import importlib.util
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "resonance_xgboost_analysis.py"
SPEC = importlib.util.spec_from_file_location("resonance_xgboost_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def synthetic_table(nmerged: np.ndarray | None = None) -> analysis.EventTable:
    nmerged = np.asarray([0, 1, 2, 3, 4] if nmerged is None else nmerged, dtype=int)
    count = len(nmerged)
    category = np.where(nmerged == 0, 0, np.where(nmerged <= 2, 1, 2))
    arrays: dict[str, np.ndarray] = {}
    for name in analysis.SCALAR_BRANCHES:
        arrays[name] = np.ones(count, dtype=float)
    arrays.update(
        event_index=np.arange(count, dtype=np.int64),
        weight=np.ones(count),
        n_merged=nmerged,
        n_double_b=nmerged.copy(),
        n_true_single=8 - 2 * nmerged,
        n_c_mistag=np.zeros(count, dtype=int),
        n_light_mistag=np.zeros(count, dtype=int),
        category=category,
        second_score=np.full(count, -1.0),
        score_gap=np.full(count, -1.0),
        m4h=np.full(count, 1000.0),
    )
    for name, width in analysis.ARRAY_BRANCH_WIDTHS.items():
        arrays[name] = np.ones((count, width), dtype=float)
    arrays["pair_mass"] = np.tile(np.asarray([300.0, 420.0, 510.0, 490.0, 380.0, 310.0]), (count, 1))
    arrays["jet_pt"] = np.tile(np.arange(8, 0, -1, dtype=float), (count, 1))
    return analysis.EventTable(
        arrays=arrays,
        input_events=count,
        input_sumw=float(count),
        input_sumw2=float(count),
        reconstructable_events=count,
        reconstructable_sumw=float(count),
        summary={"tag_efficiencies_applied": False},
    )


def synthetic_loaded_sample(
    sample_id: str,
    role: str,
    *,
    point: analysis.MassPoint | None = None,
    nmerged: np.ndarray | None = None,
    luminosity: float = analysis.LUMINOSITY_FB,
    hbb_branching_ratio: float = analysis.HBB_BRANCHING_RATIO,
    eps_b: float = analysis.EPS_B,
    eps_c: float = analysis.EPS_C,
    eps_light: float = analysis.EPS_LIGHT,
) -> analysis.LoadedSample:
    table = synthetic_table(nmerged)
    spec = analysis.SampleSpec(
        sample_id=sample_id,
        role=role,
        root_file=Path(f"{sample_id}.root"),
        summary_file=Path(f"{sample_id}.analysis_summary.json"),
        generated_events_expected=table.input_events,
        cross_section_fb=1.0 if role == "signal" else 2.0,
        generated_cross_section_fb=1.0 if role == "signal" else 2.0,
        cross_section_source="synthetic",
        k_factor=1.0 if role == "signal" else 2.0,
        hbb_power=4,
        rate_factor=1.0,
        c_mistags=0,
        light_mistags=0,
        lhe_event_count=table.input_events,
        hard_event_policy="unique",
        point=point,
    )
    base, names = analysis.base_feature_matrix(table)
    folds = np.arange(table.entries, dtype=int) % analysis.N_FOLDS
    scenarios = {
        "nominal": eps_b**2,
        "conservative": analysis.EPS_BB_CONSERVATIVE,
    }
    weights = {
        scenario: analysis.physical_event_weights(
            spec,
            table,
            luminosity,
            hbb_branching_ratio,
            eps_b,
            eps_bb,
            eps_c,
            eps_light,
        )
        for scenario, eps_bb in scenarios.items()
    }
    return analysis.LoadedSample(spec, table, folds, base, names, weights)


class MassPointTests(unittest.TestCase):
    def test_manifest_and_filename_parsing(self) -> None:
        direct = analysis.parse_mass_point("direct", {"miota_GeV": "1250"})
        cascade = analysis.parse_mass_point(
            "cascade", filename="HW-miota_1600-meta_0550_resonance.root"
        )
        self.assertEqual(direct.point_id, "MS_1250")
        self.assertEqual(cascade.point_id, "M2_0550_M3_1600")

    def test_hierarchy_is_enforced(self) -> None:
        with self.assertRaises(analysis.AnalysisInputError):
            analysis.MassPoint("cascade", m2=400.0, m3=800.0)

    def test_signal_manifest_rejects_hard_event_recycling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text(
                "scenario,run_name,miota_GeV,events,lhe_event_count,hard_event_policy\n"
                "direct,signal_1000,1000,11,10,unique\n",
                encoding="utf-8",
            )
            with self.assertRaises(analysis.AnalysisInputError):
                analysis.load_signal_specs(
                    manifest,
                    "direct",
                    Path(directory),
                    "{run_name}.root",
                )


class FeatureTests(unittest.TestCase):
    def test_truth_audit_branches_are_not_classifier_features(self) -> None:
        table = synthetic_table()
        _, names = analysis.base_feature_matrix(table)
        forbidden = {
            "raw_bjets",
            "accepted_bjets",
            "accepted_single_bjets",
            "accepted_merged_bjets",
            "accepted_cjet_candidates",
            "accepted_lightjet_candidates",
            "n_true_single",
            "n_double_b",
            "n_c_mistag",
            "n_light_mistag",
        }
        self.assertTrue(forbidden.isdisjoint(names))
        self.assertIn("n_merged", names)

    def test_direct_features_contain_mass_residual_and_six_pair_ratios(self) -> None:
        table = synthetic_table()
        base, names = analysis.base_feature_matrix(table)
        matrix, feature_names = analysis.engineer_features(
            base, names, table, "direct", ms=1000.0
        )
        self.assertEqual(matrix.shape[1], len(names) + 8)
        self.assertTrue(np.allclose(matrix[:, feature_names.index("m4h_minus_MS_over_MS")], 0.0))
        self.assertTrue(all(f"pair_mass_{label}_over_MS" in feature_names for label in analysis.PAIR_LABELS))

    def test_cascade_selects_best_of_three_pairings(self) -> None:
        table = synthetic_table(np.asarray([0]))
        base, names = analysis.base_feature_matrix(table)
        matrix, feature_names = analysis.engineer_features(
            base, names, table, "cascade", m2=300.0, m3=1000.0
        )
        # Pairing (12,34) has masses 300 and 310 and is the best pairing.
        self.assertAlmostEqual(matrix[0, feature_names.index("cascade_pairing_index")], 0.0)
        self.assertAlmostEqual(matrix[0, feature_names.index("cascade_eta1_residual")], 0.0)
        self.assertGreaterEqual(
            matrix[0, feature_names.index("cascade_second_score")],
            matrix[0, feature_names.index("cascade_best_score")],
        )


class FoldAndNormalizationTests(unittest.TestCase):
    def test_source_local_folds_are_balanced_and_order_independent(self) -> None:
        entries = np.arange(23)
        sources = np.full(23, "sample", dtype=object)
        folds = analysis.deterministic_folds(sources, entries)
        counts = np.bincount(folds, minlength=5)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        order = np.arange(22, -1, -1)
        reordered = analysis.deterministic_folds(sources[order], entries[order])
        restored = np.empty_like(reordered)
        restored[order] = reordered
        self.assertTrue(np.array_equal(folds, restored))

    def test_nominal_double_tag_factor_closes_to_eight_single_tags(self) -> None:
        table = synthetic_table()
        factors = analysis.tag_efficiency(
            table,
            analysis.EPS_B,
            analysis.EPS_B**2,
            analysis.EPS_C,
            analysis.EPS_LIGHT,
        )
        self.assertTrue(np.allclose(factors, analysis.EPS_B**8, rtol=0.0, atol=1e-14))

    def test_exact_one_fb_normalization_constants(self) -> None:
        produced = analysis.LUMINOSITY_FB * analysis.SIGNAL_HYPOTHESIS_FB
        eight_b = produced * analysis.HBB_BRANCHING_RATIO**4
        nominal = eight_b * analysis.EPS_B**8
        self.assertEqual(produced, 3000.0)
        self.assertTrue(math.isclose(eight_b, 345.1490798665728, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(nominal, 94.0498539896, rel_tol=0.0, abs_tol=1e-10))

    def test_charm_and_light_factors_are_applied_once(self) -> None:
        table = synthetic_table(np.asarray([0]))
        table.arrays["n_true_single"][:] = 4
        table.arrays["n_c_mistag"][:] = 2
        table.arrays["n_light_mistag"][:] = 2
        factor = analysis.tag_efficiency(table, 0.85, 0.85**2, 0.10, 0.01)[0]
        self.assertAlmostEqual(factor, 0.85**4 * 0.10**2 * 0.01**2)

    def test_per_event_mistag_composition_must_match_manifest(self) -> None:
        sample = synthetic_loaded_sample(
            "background", "sm_hhhh", point=None, nmerged=np.asarray([0])
        )
        sample.table.arrays["n_c_mistag"][:] = 1
        sample.table.arrays["n_true_single"][:] = 7
        with self.assertRaises(analysis.AnalysisInputError):
            analysis._validate_event_arrays(sample.spec, sample.table.arrays)

    def test_runtime_overrides_do_not_mutate_benchmark_closure(self) -> None:
        point = analysis.MassPoint("direct", ms=1000.0)
        signal = synthetic_loaded_sample(
            "signal",
            "signal",
            point=point,
            luminosity=100.0,
            hbb_branching_ratio=0.5,
            eps_b=0.8,
        )
        background = synthetic_loaded_sample(
            "sm_hhhh",
            "sm_hhhh",
            luminosity=100.0,
            hbb_branching_ratio=0.5,
            eps_b=0.8,
        )
        audit = analysis.normalization_audit(
            [signal],
            [background],
            100.0,
            0.5,
            0.8,
            analysis.EPS_C,
            analysis.EPS_LIGHT,
            {"nominal": 0.8**2, "conservative": analysis.EPS_BB_CONSERVATIVE},
        )
        self.assertTrue(audit["all_checks_pass"])
        self.assertEqual(audit["runtime_parameters"]["produced_events"], 100.0)
        self.assertEqual(
            audit["benchmark_constants"]["produced_events"],
            analysis.PRODUCED_SIGNAL_EVENTS,
        )


class CrossfitAndBinningTests(unittest.TestCase):
    class DummyModel:
        def __init__(self, rotation: int) -> None:
            self.rotation = rotation
            self.calls = 0

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            self.calls += 1
            probability = np.full(len(features), 0.1 * (self.rotation + 1))
            return np.column_stack([1.0 - probability, probability])

    def test_point_scores_are_cached_once_with_disjoint_fold_roles(self) -> None:
        point = analysis.MassPoint("direct", ms=1000.0)
        sample = synthetic_loaded_sample(
            "signal", "signal", point=point, nmerged=np.zeros(10, dtype=int)
        )
        models = [self.DummyModel(rotation) for rotation in range(analysis.N_FOLDS)]
        scores = analysis._predict_point_crossfit(sample, point, models)
        for event, fold in enumerate(sample.folds):
            self.assertAlmostEqual(scores.test[event], 0.1 * (fold + 1))
            validation_model = (int(fold) - 1) % analysis.N_FOLDS
            self.assertAlmostEqual(
                scores.validation[event], 0.1 * (validation_model + 1)
            )
        self.assertTrue(all(model.calls == 1 for model in models))
        for rotation in range(analysis.N_FOLDS):
            validation = sample.folds == (rotation + 1) % analysis.N_FOLDS
            test = sample.folds == rotation
            self.assertFalse(np.any(validation & test))

    def test_binning_checks_both_scenarios_and_has_inclusive_fallback(self) -> None:
        scores = np.linspace(0.1, 0.9, 5)
        selection = analysis._select_category_edges(
            scores,
            np.ones(5),
            scores,
            {
                "nominal": np.ones(5),
                "conservative": np.asarray([-1.0, -1.0, -1.0, -1.0, 10.0]),
            },
            min_raw=1,
            min_neff=0.0,
        )
        self.assertEqual(selection["status"], "ok")
        self.assertEqual(selection["fallback_level"], "inclusive_1_bin")
        self.assertIn("nominal", selection["inclusive_scenario_validation"])
        self.assertIn("conservative", selection["inclusive_scenario_validation"])


class YieldOutputTests(unittest.TestCase):
    def test_staged_category_yields_and_background_total_close(self) -> None:
        point = analysis.MassPoint("direct", ms=1000.0)
        signal = synthetic_loaded_sample("signal", "signal", point=point)
        background = synthetic_loaded_sample("sm_hhhh", "sm_hhhh")
        rows = analysis._category_yield_rows(
            "direct",
            point,
            [signal, background],
            {0, 1, 2},
            analysis.TAGGING_SCENARIOS,
            analysis.LUMINOSITY_FB,
            analysis.HBB_BRANCHING_RATIO,
            analysis.EPS_B,
            analysis.EPS_C,
            analysis.EPS_LIGHT,
        )
        signal_all = next(
            row
            for row in rows
            if row["tagging_scenario"] == "nominal"
            and row["sample_id"] == "signal"
            and row["category"] == "all"
        )
        self.assertAlmostEqual(signal_all["generated_yield"], 3000.0)
        self.assertAlmostEqual(
            signal_all["after_hbb_yield"], analysis.EIGHT_B_SIGNAL_EVENTS
        )
        self.assertAlmostEqual(
            signal_all["reconstructed_before_tag_yield"],
            analysis.EIGHT_B_SIGNAL_EVENTS,
        )
        self.assertAlmostEqual(
            signal_all["tagged_yield"], analysis.NOMINAL_TAG_SIGNAL_EVENTS
        )
        self.assertAlmostEqual(
            signal_all["used_in_limit_yield"], analysis.NOMINAL_TAG_SIGNAL_EVENTS
        )
        for stage in (
            "reconstructed_before_tag_yield",
            "tagged_yield",
            "used_in_limit_yield",
        ):
            self.assertTrue(signal_all[f"category_partition_{stage}_closure_pass"])
        total_background = next(
            row
            for row in rows
            if row["tagging_scenario"] == "nominal"
            and row["sample_id"] == "TOTAL_BACKGROUND"
            and row["category"] == "all"
        )
        self.assertAlmostEqual(
            total_background["tagged_yield"],
            sum(
                row["tagged_yield"]
                for row in rows
                if row["tagging_scenario"] == "nominal"
                and row["role"] != "signal"
                and row["role"] != "total_background"
                and row["category"] == "all"
            ),
        )


@unittest.skipUnless(
    os.environ.get("RUN_PYHF_INTEGRATION") == "1"
    and importlib.util.find_spec("pyhf") is not None,
    "set RUN_PYHF_INTEGRATION=1 with pyhf installed",
)
class PyhfIntegrationTests(unittest.TestCase):
    def test_reference_scaled_poi_avoids_wide_bound_minimizer_failure(self) -> None:
        fit = analysis._pyhf_limit(
            [
                {
                    "name": "resolved_fold0",
                    "signal": np.asarray([1.0, 2.0]),
                    "background": np.asarray([10.0, 5.0]),
                    "signal_staterror": np.asarray([0.1, 0.2]),
                    "background_staterror": np.asarray([1.0, 0.7]),
                }
            ]
        )
        self.assertEqual(fit["status"], "ok", fit)
        self.assertAlmostEqual(fit["expected_median"], 2.7769926, places=4)
        self.assertLessEqual(
            fit["maximum_returned_mu"],
            fit["boundary_fraction_threshold"] * fit["mu_fit_bounds"][1],
        )
        self.assertAlmostEqual(
            fit["expected_plus2sigma"],
            6.1742253,
            places=4,
        )


class CheckpointTests(unittest.TestCase):
    @staticmethod
    def point_shard(statuses: tuple[str, str] = ("ok", "ok")) -> dict[str, object]:
        point_id = "MS_1000"
        return {
            "run_fingerprint": "fingerprint",
            "point_id": point_id,
            "point_category_yields": [
                {"point_id": point_id, "tagging_scenario": scenario, "yield": 1.0}
                for scenario in ("nominal", "conservative")
            ],
            "score_bin_yields": [
                {"point_id": point_id, "tagging_scenario": scenario, "yield": 1.0}
                for scenario in ("nominal", "conservative")
            ],
            "point_limits": [
                {
                    "point_id": point_id,
                    "tagging_scenario": scenario,
                    "status": status,
                    "n_channels": 1,
                    "observed_asimov": 1.0,
                    "expected_minus2sigma": 0.7,
                    "expected_minus1sigma": 0.8,
                    "expected_median": 1.0,
                    "expected_plus1sigma": 1.2,
                    "expected_plus2sigma": 1.4,
                }
                for scenario, status in zip(
                    ("nominal", "conservative"), statuses, strict=True
                )
            ],
            "binning_audit": {
                category: {"status": "ok"} for category in analysis.CATEGORY_NAMES
            },
        }

    def test_run_config_refuses_mismatched_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            output.mkdir()
            first = analysis._initialize_run_config(output, {"mass": 1000})
            second = analysis._initialize_run_config(output, {"mass": 1000})
            self.assertEqual(first, second)
            with self.assertRaises(analysis.AnalysisInputError):
                analysis._initialize_run_config(output, {"mass": 1200})

    def test_full_retries_failed_or_malformed_shards_but_smoke_is_explicit(self) -> None:
        scenarios = ("nominal", "conservative")
        reusable, _ = analysis._point_shard_reuse_decision(
            self.point_shard(),
            mode="full",
            point_id="MS_1000",
            run_fingerprint="fingerprint",
            tagging_scenarios=scenarios,
        )
        self.assertTrue(reusable)
        failed = self.point_shard(("ok", "failed"))
        reusable_full, reason = analysis._point_shard_reuse_decision(
            failed,
            mode="full",
            point_id="MS_1000",
            run_fingerprint="fingerprint",
            tagging_scenarios=scenarios,
        )
        self.assertFalse(reusable_full)
        self.assertIn("non-successful", reason)
        reusable_smoke, _ = analysis._point_shard_reuse_decision(
            failed,
            mode="smoke",
            point_id="MS_1000",
            run_fingerprint="fingerprint",
            tagging_scenarios=scenarios,
        )
        self.assertTrue(reusable_smoke)
        malformed = self.point_shard()
        malformed["point_limits"] = malformed["point_limits"][:1]  # type: ignore[index]
        reusable_malformed, _ = analysis._point_shard_reuse_decision(
            malformed,
            mode="full",
            point_id="MS_1000",
            run_fingerprint="fingerprint",
            tagging_scenarios=scenarios,
        )
        self.assertFalse(reusable_malformed)

    def test_streamed_shard_outputs_are_valid_and_manifest_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_paths: list[Path] = []
            for index in range(3):
                path = root / f"point_{index}.json"
                analysis._write_json(
                    path,
                    {
                        "point_id": f"point_{index}",
                        "rows": [
                            {
                                "point_id": f"point_{index}",
                                "order": index,
                                f"field_{index}": index,
                            }
                        ],
                        "binning_audit": {"order": index},
                    },
                )
                shard_paths.append(path)
            iterator = analysis._iter_point_shard_rows(shard_paths, "rows")
            self.assertFalse(isinstance(iterator, list))
            self.assertEqual([row["order"] for row in iterator], [0, 1, 2])
            csv_path = root / "rows.csv"
            json_path = root / "rows.json"
            binning_path = root / "binning.json"
            analysis._write_sharded_csv(csv_path, shard_paths, "rows")
            analysis._write_sharded_json_array(json_path, shard_paths, "rows")
            analysis._write_sharded_binning_json(binning_path, shard_paths)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["order"]) for row in csv_rows], [0, 1, 2])
            self.assertTrue(all(f"field_{index}" in csv_rows[0] for index in range(3)))
            with json_path.open(encoding="utf-8") as handle:
                json_rows = json.load(handle)
            self.assertEqual([row["order"] for row in json_rows], [0, 1, 2])
            with binning_path.open(encoding="utf-8") as handle:
                binning = json.load(handle)
            self.assertEqual(list(binning), ["point_0", "point_1", "point_2"])


if __name__ == "__main__":
    unittest.main()
