from __future__ import annotations

import sys
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import c3d4_xgboost_runner as runner  # noqa: E402
from observable_schemas import EXTENDED_FEATURE_NAMES  # noqa: E402


def sample(sample_id, kind, raw, physical, c3=None, d4=None):
    raw = np.asarray(raw, dtype=float)
    physical = np.asarray(physical, dtype=float)
    entries = len(raw)
    return runner.EventSample(
        path=Path(f"/{sample_id}.root"),
        sample_id=sample_id,
        kind=kind,
        features=np.arange(entries * 91, dtype=float).reshape(entries, 91),
        raw_weights=raw,
        physical_weights=physical,
        unit_xsec_weights=physical,
        event_indices=np.arange(entries),
        source_entry_indices=np.arange(entries),
        folds=np.arange(entries) % 5,
        xsec_fb=1.0,
        rate_factor=1.0,
        normalisation_weight=float(np.sum(raw)),
        normalisation_source="test",
        generated_events=entries,
        c3=c3,
        d4=d4,
        metadata={},
    )


class C3D4XGBoostRunnerTests(unittest.TestCase):
    def test_analysis_document_lists_the_complete_ordered_schema(self):
        document = (CODE.parent / "docs" / "c3d4_xgboost_analysis.md").read_text()
        for index, name in enumerate(EXTENDED_FEATURE_NAMES):
            self.assertIn(f"| {index} | `{name}` |", document)

    def test_legacy_schema_rejects_extended_profiles(self):
        with self.assertRaisesRegex(ValueError, "supports only corrected28"):
            runner.run_c3d4_study(
                sm_signal_specs=[],
                grid_signal_specs=[],
                background_specs=[],
                output_dir="/tmp/unused-qha-test",
                observable_set="legacy-28-v1",
                feature_profile="full91",
            )

    def test_pooled_training_arrays_equalize_points_and_classes(self):
        grid = [
            sample("p0", "grid_signal", [1, 2, 3, 4, 5], [10, 20, 30, 40, 50], 0, 0),
            sample("p1", "grid_signal", [5, 4, 3, 2, 1], [500, 400, 300, 200, 100], 1, 0),
        ]
        background = [
            sample("b0", "background", [1] * 5, [1, 2, 3, 4, 5]),
            sample("b1", "background", [1] * 5, [10, 20, 30, 40, 50]),
        ]

        _, labels, weights = runner._training_arrays(
            [],
            grid,
            background,
            strategy="pooled-crossfit-v2",
            profile_indices=np.arange(28),
            rotation=0,
            n_folds=5,
        )

        signal_weights = weights[labels == 1]
        background_weights = weights[labels == 0]
        self.assertAlmostEqual(float(np.sum(signal_weights)), 1.0)
        self.assertAlmostEqual(float(np.sum(background_weights)), 1.0)
        # Three training folds give three rows from each point, in sample order.
        self.assertAlmostEqual(float(np.sum(signal_weights[:3])), 0.5)
        self.assertAlmostEqual(float(np.sum(signal_weights[3:])), 0.5)
        # Physical process ratios are preserved within the background class.
        self.assertAlmostEqual(float(np.sum(background_weights[3:])), 10.0 / 11.0)

    def test_crossfit_test_aggregation_uses_each_fold_without_rescaling(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        rotations = []
        for fold in range(5):
            rotations.append(
                {
                    "points": {
                        point.point_id: {
                            "threshold": 0.1 * fold,
                            "signal_unit_yield": 0.2,
                            "signal_sumw2_unit": 0.04,
                            "signal_raw_entries": 1,
                            "signal_feature_unit_yield": 1.0,
                            "background_yield": 0.4,
                            "background_sumw2": 0.16,
                            "background_raw_entries": 1,
                            "background_effective_entries": 1.0,
                            "cut_sigma95_fb": runner.exact_cls_signal_upper_limit(0.4) / 0.2,
                        }
                    }
                }
            )

        result = runner._aggregate_cut_results([point], rotations)[0]

        self.assertAlmostEqual(result["selected_signal_yield_per_fb"], 1.0)
        self.assertAlmostEqual(result["background_yield"], 2.0)
        self.assertEqual(result["background_raw_entries"], 5)
        self.assertAlmostEqual(result["threshold_mean"], 0.2)
        self.assertAlmostEqual(
            result["cut_sigma95_fb"],
            runner.exact_cls_signal_upper_limit(2.0),
        )

    def test_validation_scaling_closes_each_source_to_full_yield(self):
        weights = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.asarray([False, True, False, False, False])
        scale = runner._partition_scale(weights, mask)
        self.assertAlmostEqual(float(np.sum(weights[mask] * scale)), float(np.sum(weights)))

    def test_optuna_fingerprint_binds_schema_and_inputs(self):
        common = {
            "profile": "corrected28",
            "strategy": "pooled-crossfit-v2",
            "rotation": 0,
            "n_folds": 5,
            "seed": 12345,
            "source_commit": "abc",
            "fold_digest": "folds",
            "normalization_inputs": {"luminosity_fb_inverse": 3000.0},
            "input_hashes": {"sample.root": "hash-a"},
        }
        extended = runner._run_fingerprint(observable_set="extended-91-v2", **common)
        legacy = runner._run_fingerprint(observable_set="legacy-28-v1", **common)
        changed_input = runner._run_fingerprint(
            observable_set="extended-91-v2",
            **{**common, "input_hashes": {"sample.root": "hash-b"}},
        )
        self.assertNotEqual(extended, legacy)
        self.assertNotEqual(extended, changed_input)

    def test_csv_writer_keeps_nested_bin_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.csv"
            runner._write_rows(
                output,
                [{"point_id": "p", "fold_bin_edges": [[0.0, 0.5, 1.0]]}],
            )
            with output.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["fold_bin_edges"], "[[0.0,0.5,1.0]]")


if __name__ == "__main__":
    unittest.main()
