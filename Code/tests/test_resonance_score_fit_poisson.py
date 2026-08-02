from __future__ import annotations

import inspect
import importlib.util
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
import prepare_resonance_fatjet_features as prepare_features
import resonance_score_fit_poisson as scorefit
import resonance_xgboost_analysis as resolved

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


def summary(yields: list[float], raw: list[int], neff: list[float]) -> dict[str, np.ndarray]:
    values = np.asarray(yields, dtype=float)
    effective = np.asarray(neff, dtype=float)
    sumw2 = np.divide(
        values**2,
        effective,
        out=np.zeros_like(values),
        where=effective > 0.0,
    )
    return {
        "yield": values,
        "sumw2": sumw2,
        "raw": np.asarray(raw, dtype=int),
        "neff": effective,
    }


class PoissonLimitTests(unittest.TestCase):
    def test_zero_counts_and_q_zero(self) -> None:
        self.assertEqual(scorefit.poisson_q([0.0, 3.0], [2.0, 1.0], 0.0), 0.0)
        self.assertAlmostEqual(scorefit.poisson_q([0.0], [1.0], 2.0), 4.0)

    def test_q_is_monotonic_and_crosses_declared_level(self) -> None:
        counts = np.asarray([20.0, 5.0, 0.0])
        signal = np.asarray([1.0, 2.0, 0.4])
        values = [scorefit.poisson_q(counts, signal, value) for value in (0, 1, 2, 4)]
        self.assertTrue(np.all(np.diff(values) > 0.0))
        limit = scorefit.solve_sigma95(counts, signal)
        self.assertAlmostEqual(
            scorefit.poisson_q(counts, signal, limit), scorefit.Q95, places=11
        )

    def test_one_fb_template_scaling(self) -> None:
        counts = [100.0, 12.0]
        signal = np.asarray([3.0, 1.0])
        nominal = scorefit.solve_sigma95(counts, signal)
        doubled_template = scorefit.solve_sigma95(counts, 2.0 * signal)
        self.assertAlmostEqual(doubled_template, nominal / 2.0, places=12)


class CrossfitAndBinningTests(unittest.TestCase):
    def test_fold_roles_use_three_train_one_validation_one_test(self) -> None:
        folds = np.arange(5)
        for rotation in range(5):
            masks = scorefit.fold_role_masks(folds, rotation)
            self.assertEqual(int(np.sum(masks["train"])), 3)
            self.assertEqual(int(np.sum(masks["validation"])), 1)
            self.assertEqual(int(np.sum(masks["test"])), 1)
            self.assertFalse(np.any(masks["train"] & masks["validation"]))
            self.assertFalse(np.any(masks["train"] & masks["test"]))

    def test_tag_hypotheses_from_one_event_share_a_fold(self) -> None:
        events = np.asarray([3, 3, 3, 10, 10, 21])
        folds = fat.grouped_folds("sample", events, scorefit.SEED)
        for event in np.unique(events):
            self.assertEqual(len(np.unique(folds[events == event])), 1)

    def test_tail_merging_is_deterministic_and_downward(self) -> None:
        initial = [np.asarray([0.0, 0.5, 0.8, 0.95, 1.0])] * 5

        def summarize(edges: list[np.ndarray]) -> dict[str, np.ndarray]:
            bins = len(edges[0]) - 1
            if bins == 4:
                return summary([100, 50, 20, 1], [100, 80, 40, 2], [80, 50, 20, 1])
            return summary([100, 50, 21], [100, 80, 42], [80, 50, 21])

        edges, history, result = scorefit.merge_failing_tail_edges(initial, summarize)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["failed_bin"], 3)
        self.assertEqual(history[0]["removed_boundary_index"], 3)
        self.assertEqual(len(edges[0]) - 1, 3)
        self.assertEqual(scorefit.occupancy_failures(result), [])

    def test_held_out_failure_is_coarsened_without_signal_information(self) -> None:
        initial = [np.asarray([0.0, 0.5, 0.8, 1.0])] * 5

        def validation(edges: list[np.ndarray]) -> dict[str, np.ndarray]:
            bins = len(edges[0]) - 1
            return summary([100.0] * bins, [100] * bins, [20.0] * bins)

        def test(edges: list[np.ndarray]) -> dict[str, np.ndarray]:
            bins = len(edges[0]) - 1
            if bins == 3:
                return summary([100, 50, 2], [100, 80, 30], [20, 15, 2])
            return summary([100, 52], [100, 110], [20, 8])

        edges, history, partitions = scorefit.merge_partition_background_edges(
            initial, {"validation": validation, "test": test}
        )
        self.assertEqual(len(edges[0]) - 1, 2)
        self.assertEqual(history[0]["partition"], "test")
        self.assertEqual(history[0]["failed_bin"], 2)
        self.assertEqual(scorefit.occupancy_failures(partitions["validation"]), [])
        self.assertEqual(scorefit.occupancy_failures(partitions["test"]), [])

    def test_global_selection_prefers_four_bins_within_two_percent(self) -> None:
        points = [
            {
                "schemes": {
                    "background_quantile_4bin": {
                        "status": "ok",
                        "validation_sigma95_fb": 10.1,
                    },
                    "background_quantile_5bin": {
                        "status": "ok",
                        "validation_sigma95_fb": 10.0,
                    },
                }
            },
            {
                "schemes": {
                    "background_quantile_4bin": {
                        "status": "ok",
                        "validation_sigma95_fb": 20.2,
                    },
                    "background_quantile_5bin": {
                        "status": "ok",
                        "validation_sigma95_fb": 20.0,
                    },
                }
            },
        ]
        selected, audit = scorefit.select_binning_scheme(points)
        self.assertEqual(selected, "background_quantile_4bin")
        self.assertAlmostEqual(audit["median_validation_limit_ratio_four_over_five"], 1.01)

    def test_weighted_quantiles_are_fixed_by_background_weight(self) -> None:
        values = np.asarray([0.1, 0.2, 0.8, 0.9])
        edges = scorefit.weighted_quantile(values, [0.0, 0.5, 1.0], [1, 1, 8, 1])
        self.assertGreater(edges[1], 0.5)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is not installed")
    def test_cascade_display_interpolation_is_exact_for_affine_surface(self) -> None:
        coordinates = [
            (100.0, 300.0),
            (100.0, 500.0),
            (200.0, 500.0),
            (200.0, 700.0),
            (300.0, 700.0),
            (300.0, 900.0),
        ]
        rows = [
            {
                "M2_GeV": m2,
                "M3_GeV": m3,
                "sigma95_fb": 10.0 ** (0.001 * m2 + 0.0002 * m3),
            }
            for m2, m3 in coordinates
        ]
        _, _, display, clough, audit = scorefit._cascade_interpolation(
            rows, grid_size=30
        )
        self.assertFalse(audit["fallback_applied"])
        self.assertTrue(audit["paper_ready"])
        self.assertEqual(
            audit["display_method"],
            "coordinate-rescaled CloughTocher2DInterpolator on log10(sigma95)",
        )
        self.assertEqual(display.count(), clough.count())


class AsimovAndTaggingTests(unittest.TestCase):
    @staticmethod
    def sample(sample_id: str, role: str) -> SimpleNamespace:
        return SimpleNamespace(spec=SimpleNamespace(sample_id=sample_id, role=role))

    def test_asimov_includes_all_three_sm_multihiggs_components(self) -> None:
        samples = [
            self.sample("b", "background"),
            self.sample("h4", "sm_hhhh"),
            self.sample("h3bb", "sm_hhhbb"),
            self.sample("h2bb", "sm_hh4b"),
        ]
        summaries = {
            "b": summary([10.0, 20.0], [100, 100], [100, 100]),
            "h4": summary([1.0, 2.0], [100, 100], [100, 100]),
            "h3bb": summary([3.0, 4.0], [100, 100], [100, 100]),
            "h2bb": summary([5.0, 6.0], [100, 100], [100, 100]),
        }
        np.testing.assert_allclose(
            scorefit.build_asimov_counts(samples, summaries), [19.0, 32.0]
        )
        with self.assertRaises(scorefit.ScoreFitError):
            scorefit.build_asimov_counts(samples[:-1], {k: v for k, v in summaries.items() if k != "h2bb"})

    def test_nominal_ak8_probability_closure_uses_epsilon_b_squared(self) -> None:
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
        factors = fat.tag_hypothesis_factor(
            table,
            eps_bb=scorefit.EPS_BB,
            fake_bb=scorefit.FAKE_BB,
            eps_b=scorefit.EPS_B,
            eps_c=scorefit.EPS_C,
            eps_light=scorefit.EPS_LIGHT,
        )
        self.assertAlmostEqual(scorefit.EPS_BB, scorefit.EPS_B**2)
        self.assertAlmostEqual(float(np.sum(factors)), 1.0, places=14)

    def test_cli_exposes_only_inputs_and_parallelism(self) -> None:
        args = scorefit.build_parser().parse_args(
            ["--topology", "direct", "--output-dir", "out"]
        )
        self.assertEqual(args.topology, "direct")
        for forbidden in (
            "background_norm_fraction",
            "pyhf_jobs",
            "eps_bb_conservative",
            "min_background_neff",
            "q95",
        ):
            self.assertFalse(hasattr(args, forbidden))
        self.assertEqual(scorefit.MIN_BACKGROUND_NEFF, 5.0)
        self.assertEqual(scorefit.COLLIDER_ENERGY_TEV, 14.0)

    def test_driver_has_no_pyhf_or_optuna_dependency(self) -> None:
        source = inspect.getsource(scorefit)
        self.assertNotIn("import pyhf", source)
        self.assertNotIn("import optuna", source.lower())
        self.assertIn("{output_path.suffix}", source)

    def test_feature_manifest_uses_persisted_input_event_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            source.write_text(
                "sample_id,root_file,generated_events\nexample,old.root,29616\n",
                encoding="utf-8",
            )
            output = root / "features"
            output.mkdir()
            feature = output / "example_fatjet.root"
            feature.touch()
            feature.with_suffix(".analysis_summary.json").write_text(
                '{"input_counter":{"events":16953}}', encoding="utf-8"
            )
            rendered = prepare_features._versioned_manifest_text(source, root, output)
            self.assertTrue(rendered.rstrip().endswith(",16953"))


if __name__ == "__main__":
    unittest.main()
