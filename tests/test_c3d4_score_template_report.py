from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import c3d4_score_template_report as report  # noqa: E402
import c3d4_xgboost_runner as runner  # noqa: E402


def workspace_sample(name, values, errors, *, signal):
    modifiers = []
    if signal:
        modifiers.append({"name": "sigma_fb", "type": "normfactor", "data": None})
    modifiers.append(
        {"name": f"{name}_mcstat", "type": "staterror", "data": list(errors)}
    )
    return {"name": name, "data": list(values), "modifiers": modifiers}


def sm_shape_row():
    edges = [[0.0, 0.5, 1.0], [0.0, 0.4, 1.0]]
    signal = [[2.0, 3.0], [1.5, 4.0]]
    signal_error = [[0.4, 0.6], [0.3, 0.8]]
    background = [[4.0, 1.0], [3.0, 2.0]]
    background_error = [[0.5, 0.2], [0.4, 0.3]]
    channels = []
    observations = []
    for fold in range(2):
        name = f"test_fold{fold}"
        channels.append(
            {
                "name": name,
                "samples": [
                    workspace_sample(
                        "signal", signal[fold], signal_error[fold], signal=True
                    ),
                    workspace_sample(
                        "background",
                        background[fold],
                        background_error[fold],
                        signal=False,
                    ),
                ],
            }
        )
        observations.append({"name": name, "data": background[fold]})
    workspace = {
        "channels": channels,
        "observations": observations,
        "measurements": [
            {
                "name": "expected_limit",
                "config": {
                    "poi": "sigma_fb",
                    "parameters": [
                        {
                            "name": "sigma_fb",
                            "inits": [1.0],
                            "bounds": [[0.0, 10.0]],
                        }
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }
    return {
        "point_id": "c3=0,d4=0",
        "c3": 0.0,
        "d4": 0.0,
        "status": "ok",
        "bin_count": 2,
        "fold_bin_edges": edges,
        "hhhh_xsec_fb": 1.24288e-4,
        "limit_cross_section_basis": "equivalent-hhhh-fb",
        "signal_components": "hhhh,hhhbb",
        "shape_sigma95_fb": 0.0957,
        "pyhf_shape_with_mcstat": {
            "status": "ok",
            "expected_median": 0.0957,
            "workspace_spec": workspace,
        },
    }


def event_sample(sample_id, kind, folds, *, xsec=1.0):
    folds = np.asarray(folds, dtype=np.int16)
    entries = len(folds)
    unit = np.linspace(0.2, 0.2 * entries, entries)
    physical = xsec * unit
    features = np.linspace(0.05, 0.95, entries).reshape(-1, 1)
    return runner.EventSample(
        path=Path(f"/{sample_id}.root"),
        sample_id=sample_id,
        kind=kind,
        features=features,
        raw_weights=np.ones(entries),
        physical_weights=physical,
        unit_xsec_weights=unit,
        event_indices=np.arange(entries, dtype=np.int64),
        source_entry_indices=np.arange(entries, dtype=np.int64),
        folds=folds,
        xsec_fb=xsec,
        rate_factor=1.0,
        normalisation_weight=float(entries),
        normalisation_source="test",
        generated_events=entries,
        c3=0.0 if kind != "background" else None,
        d4=0.0 if kind != "background" else None,
        metadata={},
    )


class FakeModel:
    def predict_proba(self, features):
        score = np.asarray(features[:, 0], dtype=float)
        return np.column_stack((1.0 - score, score))


class ScoreTemplateReportTests(unittest.TestCase):
    def test_extracts_exact_workspace_templates_and_asimov_observations(self):
        channels = report.extract_workspace_templates(sm_shape_row())
        self.assertEqual([channel["fold"] for channel in channels], [0, 1])
        np.testing.assert_allclose(channels[0]["signal"], [2.0, 3.0])
        np.testing.assert_allclose(channels[1]["background"], [3.0, 2.0])
        np.testing.assert_allclose(
            channels[1]["observation"], channels[1]["background"]
        )

    def test_rejects_non_asimov_workspace_observation(self):
        row = sm_shape_row()
        row["pyhf_shape_with_mcstat"]["workspace_spec"]["observations"][0][
            "data"
        ][0] += 1.0
        with self.assertRaisesRegex(ValueError, "background-only Asimov"):
            report.extract_workspace_templates(row)

    def test_reconstruction_scores_each_event_once(self):
        folds = [0, 1, 2, 0, 1, 2]
        hhhh = event_sample("hhhh", "grid_signal", folds, xsec=0.1)
        hhhbb = event_sample("hhhbb", "postfit_hhhbb_signal", folds, xsec=0.2)
        background = event_sample("background", "background", folds, xsec=2.0)
        reconstructed = report.reconstruct_oof_scores(
            models=[FakeModel(), FakeModel(), FakeModel()],
            hhhh_samples=[hhhh],
            hhhbb_samples=[hhhbb],
            background_samples=[background],
            n_folds=3,
            profile_indices=np.asarray([0], dtype=int),
            hhhh_xsec_fb=0.1,
        )
        self.assertEqual(len(reconstructed["event_rows"]), 18)
        identities = [
            (row["sample_id"], row["source_entry_index"])
            for row in reconstructed["event_rows"]
        ]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(
            [len(fold["background_scores"]) for fold in reconstructed["folds"]],
            [2, 2, 2],
        )

    def test_template_closure_reports_roundoff_and_rejects_changes(self):
        saved = report.extract_workspace_templates(sm_shape_row())
        reconstructed = [
            {
                "fold": 0,
                "signal_scores": np.asarray([0.2, 0.8]),
                "signal_template_weights": np.asarray([2.0, 3.0]),
                "background_scores": np.asarray([0.2, 0.8]),
                "background_weights": np.asarray([4.0, 1.0]),
            },
            {
                "fold": 1,
                "signal_scores": np.asarray([0.2, 0.8]),
                "signal_template_weights": np.asarray([1.5, 4.0]),
                "background_scores": np.asarray([0.2, 0.8]),
                "background_weights": np.asarray([3.0, 2.0]),
            },
        ]
        # One event per bin means each staterror equals the absolute yield.
        for channel in saved:
            channel["signal_staterror"] = np.abs(channel["signal"])
            channel["background_staterror"] = np.abs(channel["background"])
        closure = report.validate_template_closure(reconstructed, saved)
        self.assertEqual(closure["status"], "passed")
        saved[0]["signal"][0] += 0.01
        with self.assertRaisesRegex(AssertionError, "does not close"):
            report.validate_template_closure(reconstructed, saved)

    def test_event_score_gzip_is_byte_reproducible(self):
        rows = [
            {
                "role": "signal",
                "component": "hhhh",
                "sample_id": "sample",
                "test_fold": 0,
                "event_index": 7,
                "source_entry_index": 11,
                "xgboost_score": 0.75,
                "raw_weight": 1.0,
                "unit_xsec_weight": 2.0,
                "physical_weight": 0.01,
                "template_weight_per_equivalent_hhhh_fb": 2.0,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.csv.gz"
            second = Path(temporary) / "second.csv.gz"
            report._write_event_scores(first, rows)
            report._write_event_scores(second, rows)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "matplotlib is required for plot artifact checks",
    )
    def test_skip_rescore_report_writes_exact_template_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary) / "study"
            strategy = study / "sm-crossfit-v2"
            strategy.mkdir(parents=True)
            manifest = {
                "status": "complete",
                "method_version": "test",
                "source_commit": "abc123",
                "study_mode": "fast-sm",
                "paper_ready": True,
                "physics_result_valid": True,
                "strategies_completed": ["sm-crossfit-v2"],
                "observable_set": "extended-91-v2",
                "selected_feature_profile": "full91",
                "luminosity_fb_inverse": 3000.0,
                "cv_folds": 2,
                "seed": 12345,
                "mode_policy": {"max_events_per_source": None},
                "inputs": [],
            }
            (study / "method_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (strategy / "shape_results.json").write_text(
                json.dumps([sm_shape_row()]), encoding="utf-8"
            )
            output = Path(temporary) / "report"
            payload = report.write_sm_score_template_report(
                study,
                output_dir=output,
                rescore=False,
                dpi=72,
            )
            self.assertEqual(payload["status"], "complete")
            for name in (
                "workspace_sm.json",
                "sm_pyhf_templates.csv",
                "sm_pyhf_unrolled_templates.pdf",
                "sm_pyhf_unrolled_templates.png",
                "README.md",
                "report.json",
            ):
                path = output / name
                self.assertTrue(path.is_file(), name)
                self.assertGreater(path.stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()
