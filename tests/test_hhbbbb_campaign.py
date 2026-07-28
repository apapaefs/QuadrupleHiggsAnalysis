import csv
import importlib.util
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "Code"))
if importlib.util.find_spec("xgboost") is None:
    sys.modules.setdefault("xgboost", types.SimpleNamespace(XGBClassifier=object))
if importlib.util.find_spec("sklearn") is None:
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
if importlib.util.find_spec("tqdm") is None:
    sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
    sys.modules.setdefault(
        "tqdm.auto",
        types.SimpleNamespace(tqdm=lambda iterable=None, *args, **kwargs: iterable),
    )

from ForcedSplitting.hhbbbb_campaign import HHBBBBCampaignConfig, monitor_mg5_grid, run_hhbbbb_campaign  # noqa: E402
from xgboost_root_varfiles_module import _point_metadata_from_path, combine_signal_component_rows  # noqa: E402


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
        for c3, d4 in points:
            writer.writerow({"status": "written", "run_group": "4", "c3": c3, "d4": d4})


class HHBBBBCampaignTests(unittest.TestCase):
    def test_campaign_dry_run_projects_to_unique_c3_and_uses_hhgg_split_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            mg5_dir = tmpdir / "MG5" / "gg_hhgg"
            run_dir = mg5_dir / "Events" / "run_gg_hhgg_4_0.0_0.0"
            run_dir.mkdir(parents=True)
            (run_dir / "unweighted_events.lhe").write_text(_lhe_document([1.0, 1.0, 1.0, 1.0], xsec_pb=0.25))
            manifest = tmpdir / "reference_manifest.csv"
            _write_manifest(manifest, [("0.0", "-100.0"), ("0.0", "0.0")])

            summary = run_hhbbbb_campaign(
                HHBBBBCampaignConfig(
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
            self.assertEqual(summary["points_requested"], 1)
            point = summary["points"][0]
            self.assertEqual(point["run_name"], "run_gg_hhgg_4_0.0_0.0")
            self.assertEqual([chunk["events"] for chunk in point["chunks"]], [2, 2])
            first_card = Path(point["chunks"][0]["stage1_card"]).read_text()
            self.assertIn("set ForceSplitVeto:MinB 4", first_card)
            self.assertIn("set ForceSplitVeto:MinSplitPairs 2", first_card)
            self.assertIn("set ForceSplitVeto:RequireDistinctHardGluons Yes", first_card)
            self.assertIn("set ShowerHandler:LimitEmissions NoLimit", first_card)
            self.assertTrue(point["stage2_root"].endswith("run_gg_hhgg_4_0.0_0.0_hhbbbb_stage2.root"))

    def test_monitor_mg5_grid_uses_c3_only_hhgg_run_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            mg5_dir = tmpdir / "MG5" / "gg_hhgg"
            complete = mg5_dir / "Events" / "run_gg_hhgg_4_0.0_0.0"
            complete.mkdir(parents=True)
            (complete / "unweighted_events.lhe").write_text(_lhe_document([1.0, 2.0], xsec_pb=0.25))
            manifest = tmpdir / "reference_manifest.csv"
            _write_manifest(manifest, [("0.0", "-100.0"), ("0.0", "100.0"), ("1.0", "0.0")])

            summary = monitor_mg5_grid(
                mg5_dir=mg5_dir,
                reference_grid_manifest=manifest,
                count_events=True,
            )

            self.assertEqual(summary["grid_points"], 2)
            self.assertTrue(summary["c3_only"])
            self.assertEqual(summary["complete_lhes"], 1)
            self.assertEqual(summary["pending_run_dirs"], 1)
            self.assertEqual(summary["total_counted_events"], 2)
            self.assertEqual(summary["points"][0]["run_name"], "run_gg_hhgg_4_0.0_0.0")

    def test_c3d4_metadata_accepts_hhgg_campaign_names(self):
        metadata = _point_metadata_from_path(
            "/events/run_gg_hhgg_4_-2.5_0.0_hhbbbb_stage2_var.smear.root"
        )
        self.assertEqual(metadata["run_group"], "4")
        self.assertEqual(metadata["process"], "gg_hhgg")
        self.assertTrue(math.isclose(metadata["c3"], -2.5))
        self.assertTrue(math.isclose(metadata["d4"], 0.0))

    def test_component_rows_add_hhbbbb_by_c3_to_each_d4_row(self):
        hhhh = [
            {
                "file": "hhhh_a.root",
                "run_group": "4",
                "c3": 0.0,
                "d4": -100.0,
                "entries": 100,
                "selected_entries": 10,
                "expected_preselected_events": 20.0,
                "expected_selected_events": 4.0,
                "expected_selected_error": 0.4,
                "effective_sigma_eff_fb": 0.01,
                "raw_sigma_eff_fb": 0.02,
            },
            {
                "file": "hhhh_b.root",
                "run_group": "4",
                "c3": 0.0,
                "d4": 100.0,
                "entries": 100,
                "selected_entries": 8,
                "expected_preselected_events": 10.0,
                "expected_selected_events": 2.0,
                "expected_selected_error": 0.2,
                "effective_sigma_eff_fb": 0.005,
                "raw_sigma_eff_fb": 0.01,
            },
        ]
        hhbbbb = [
            {
                "file": "hhbbbb.root",
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

        combined = combine_signal_component_rows(hhhh, hhbbbb_rows=hhbbbb)

        self.assertEqual(len(combined), 2)
        for row in combined:
            self.assertEqual(row["signal_components"], "hhhh,hhbbbb")
            self.assertTrue(math.isclose(row["hhbbbb_expected_selected_events"], 1.5))
            self.assertTrue(math.isclose(row["expected_selected_events"], row["hhhh_expected_selected_events"] + 1.5))
            self.assertEqual(row["hhbbbb_file"], "hhbbbb.root")


if __name__ == "__main__":
    unittest.main()
