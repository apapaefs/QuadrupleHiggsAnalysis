import csv
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "Code"))
sys.modules.setdefault("xgboost", types.SimpleNamespace(XGBClassifier=object))
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
sys.modules.setdefault(
    "sklearn.model_selection",
    types.SimpleNamespace(train_test_split=lambda *args, **kwargs: args),
)
sys.modules.setdefault(
    "sklearn.metrics",
    types.SimpleNamespace(
        accuracy_score=lambda *args, **kwargs: 0.0,
        confusion_matrix=lambda *args, **kwargs: [],
        RocCurveDisplay=object,
        roc_auc_score=lambda *args, **kwargs: 0.0,
        roc_curve=lambda *args, **kwargs: ([], [], []),
    ),
)
sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
sys.modules.setdefault("tqdm.auto", types.SimpleNamespace(tqdm=lambda iterable=None, *args, **kwargs: iterable))

from ForcedSplitting.hhhbb_campaign import HHHBBCampaignConfig, monitor_mg5_grid, run_hhhbb_campaign  # noqa: E402
from ForcedSplitting.lhe_merge import merge_weighted_lhe_chunks  # noqa: E402
from xgboost_root_varfiles_module import (  # noqa: E402
    _point_metadata_from_path,
    combine_signal_component_rows,
)


def _event(weight):
    return """<event>
    0      1     {weight:.9e}        91.188    0.007546771     0.09944864
</event>
""".format(weight=weight)


def _lhe_document(weights, xsec_pb=1.0):
    return """<LesHouchesEvents version="1.0">
<header>
</header>
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   {xsec:.9e}   5.000000000e-01            100      1
</init>
{events}</LesHouchesEvents>
""".format(xsec=xsec_pb, events="".join(_event(weight) for weight in weights))


def _write_manifest(path, points):
    fieldnames = ["status", "run_group", "c3", "d4"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow({"status": "written", "run_group": "4", "c3": point[0], "d4": point[1]})


class HHHBBCampaignTests(unittest.TestCase):
    def test_weighted_lhe_merge_uses_mean_event_weight_for_init_xsec(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            chunk_a = tmpdir / "chunk_a.lhe"
            chunk_b = tmpdir / "chunk_b.lhe"
            merged = tmpdir / "merged.lhe"
            summary_json = tmpdir / "merge_summary.json"
            chunk_a.write_text(_lhe_document([2.0, 4.0], xsec_pb=99.0))
            chunk_b.write_text(_lhe_document([6.0], xsec_pb=123.0))

            summary = merge_weighted_lhe_chunks(
                [chunk_a, chunk_b],
                merged,
                summary_path=summary_json,
                overwrite=True,
            )

            text = merged.read_text()
            self.assertEqual(summary["input_file_count"], 2)
            self.assertEqual(summary["input_event_counts"], [2, 1])
            self.assertEqual(summary["total_events"], 3)
            self.assertTrue(math.isclose(summary["weight_sum"], 12.0))
            self.assertTrue(math.isclose(summary["merged_xsec_pb"], 4.0))
            self.assertIn("4.000000000e+00", text)
            self.assertEqual(text.count("<event>"), 3)
            self.assertEqual(json.loads(summary_json.read_text())["total_events"], 3)

    def test_campaign_dry_run_splits_point_and_writes_capped_stage1_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            mg5_dir = tmpdir / "MG5" / "gg_hhhg"
            run_dir = mg5_dir / "Events" / "run_gg_hhhg_4_0.0_0.0"
            run_dir.mkdir(parents=True)
            (run_dir / "unweighted_events.lhe").write_text(_lhe_document([1.0, 1.0, 1.0, 1.0], xsec_pb=0.25))
            manifest = tmpdir / "reference_manifest.csv"
            _write_manifest(manifest, [("0.0", "0.0")])

            summary = run_hhhbb_campaign(
                HHHBBCampaignConfig(
                    mg5_dir=mg5_dir,
                    reference_grid_manifest=manifest,
                    workdir=tmpdir / "campaign",
                    events=4,
                    jobs=2,
                    probe_trials=99999,
                    allow_zero_probe_successes=True,
                    overwrite=True,
                    dry_run=True,
                )
            )

            self.assertEqual(summary["processed_points"], 1)
            point = summary["points"][0]
            self.assertEqual(point["status"], "dry_run")
            self.assertEqual(point["active_jobs"], 2)
            self.assertEqual([chunk["events"] for chunk in point["chunks"]], [2, 2])
            first_card = Path(point["chunks"][0]["stage1_card"]).read_text()
            self.assertIn("set ForceSplitVeto:MinB 2", first_card)
            self.assertIn("set ForceSplitVeto:MinSplitPairs 1", first_card)
            self.assertIn("Requested ProbeTrials 99999 capped", first_card)
            self.assertIn("set ForceSplitVeto:ProbeTrials 90000", first_card)
            stage2_card = Path(point["stage2_card"]).read_text()
            self.assertIn(str(Path(point["merged_weighted_lhe"])), stage2_card)

    def test_monitor_mg5_grid_summarizes_complete_incomplete_and_failed_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            mg5_dir = tmpdir / "MG5" / "gg_hhhg"
            complete = mg5_dir / "Events" / "run_gg_hhhg_4_0.0_0.0"
            incomplete = mg5_dir / "Events" / "run_gg_hhhg_4_1.0_100.0"
            complete.mkdir(parents=True)
            incomplete.mkdir(parents=True)
            (complete / "unweighted_events.lhe").write_text(_lhe_document([1.0, 2.0], xsec_pb=0.25))
            debug_log = mg5_dir / "run_gg_hhhg_4_1.0_100.0_tag_1_debug.log"
            debug_log.write_text("Traceback\ninternal.MadGraph5Error: failed initialization\n")
            deck_dir = mg5_dir / "ForcedSplittingDecks"
            deck_dir.mkdir(parents=True)
            (deck_dir / "mg5_grid.log").write_text(
                "line 1\n"
                "INFO:  Idle: 1,  Running: 56,  Completed: 0 [ current time: 16h28 ]\n"
                "Error: something went wrong\n"
                "line 3\n"
            )
            manifest = tmpdir / "reference_manifest.csv"
            _write_manifest(manifest, [("0.0", "0.0"), ("1.0", "100.0"), ("2.0", "200.0")])

            summary = monitor_mg5_grid(
                mg5_dir=mg5_dir,
                reference_grid_manifest=manifest,
                count_events=True,
                tail=4,
                show_points=3,
            )

        self.assertEqual(summary["grid_points"], 3)
        self.assertEqual(summary["complete_lhes"], 1)
        self.assertEqual(summary["incomplete_run_dirs"], 1)
        self.assertEqual(summary["pending_run_dirs"], 1)
        self.assertEqual(summary["debug_logs"], 1)
        self.assertEqual(summary["total_counted_events"], 2)
        self.assertEqual(summary["grid_log_tail"][-2:], ["Error: something went wrong", "line 3"])
        self.assertIn("Running: 56", summary["latest_progress_line"])
        self.assertIn("MadGraph5Error", "\n".join(summary["recent_debug_logs"][0]["errors"]))

    def test_c3d4_metadata_accepts_hhhg_campaign_names(self):
        metadata = _point_metadata_from_path(
            "/events/run_gg_hhhg_4_-2.5_150.0/hhhbb_var.smear.root"
        )
        self.assertEqual(metadata["run_group"], "4")
        self.assertEqual(metadata["process"], "gg_hhhg")
        self.assertTrue(math.isclose(metadata["c3"], -2.5))
        self.assertTrue(math.isclose(metadata["d4"], 150.0))

    def test_component_rows_add_hhhbb_only_to_final_signal_yield(self):
        hhhh = [
            {
                "file": "hhhh.root",
                "run_group": "4",
                "c3": 0.0,
                "d4": 0.0,
                "entries": 100,
                "selected_entries": 10,
                "expected_preselected_events": 20.0,
                "expected_selected_events": 4.0,
                "expected_selected_error": 0.4,
                "effective_sigma_eff_fb": 0.01,
                "raw_sigma_eff_fb": 0.02,
            }
        ]
        hhhbb = [
            {
                "file": "hhhbb.root",
                "run_group": "4",
                "c3": 0.0,
                "d4": 0.0,
                "entries": 80,
                "selected_entries": 5,
                "expected_preselected_events": 7.0,
                "expected_selected_events": 1.5,
                "expected_selected_error": 0.3,
                "effective_sigma_eff_fb": 0.004,
                "raw_sigma_eff_fb": 0.008,
            }
        ]

        combined = combine_signal_component_rows(hhhh, hhhbb)

        self.assertEqual(len(combined), 1)
        row = combined[0]
        self.assertEqual(row["signal_components"], "hhhh,hhhbb")
        self.assertTrue(math.isclose(row["expected_selected_events"], 5.5))
        self.assertTrue(math.isclose(row["expected_preselected_events"], 27.0))
        self.assertTrue(math.isclose(row["expected_selected_error"], 0.5))
        self.assertTrue(math.isclose(row["effective_sigma_eff_fb"], 0.014))
        self.assertTrue(math.isclose(row["hhhh_expected_selected_events"], 4.0))
        self.assertTrue(math.isclose(row["hhhbb_expected_selected_events"], 1.5))
        self.assertEqual(row["entries"], 180)
        self.assertEqual(row["selected_entries"], 15)


if __name__ == "__main__":
    unittest.main()
