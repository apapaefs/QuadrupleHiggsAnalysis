import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "Code"))

from ForcedSplitting.validation_hbb import (  # noqa: E402
    ValidationRunConfig,
    extract_lhe_4b_sample,
    prepare_mg5_decks,
    run_validation_chain,
    write_lhe_validation_report,
)


def validation_lhe_text(weight=2.0, xsec_pb=5.0, include_bad_event=False):
    good_event = """<event>
    6      1     {weight:.9e}        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  68.071  68.071  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -68.071  68.071  0.0  0.0  9.0
         5  1    0    0  0  0  20.0  0.0  30.0  36.05551275  0.0  0.0  9.0
        -5  1    0    0  0  0 -20.0  0.0 -30.0  36.05551275  0.0  0.0  9.0
         5  1    0    0  0  0  0.0  25.0  20.0  32.01562119  0.0  0.0  9.0
        -5  1    0    0  0  0  0.0 -25.0 -20.0  32.01562119  0.0  0.0  9.0
</event>
""".format(weight=weight)
    bad_event = """<event>
    3      1     1.000000000e+00        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  50.0  50.0  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -50.0  50.0  0.0  0.0  9.0
         5  1    0    0  0  0  10.0  0.0  0.0  10.0  0.0  0.0  9.0
</event>
"""
    events = good_event + (bad_event if include_bad_event else "")
    return """<LesHouchesEvents version="1.0">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   {xsec:.9e}   5.000000000e-01            100      1
</init>
{events}</LesHouchesEvents>
""".format(xsec=xsec_pb, events=events)


def validation_lhe_text_with_higgs_mothers(weight=2.0, xsec_pb=5.0):
    return """<LesHouchesEvents version="1.0">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   {xsec:.9e}   5.000000000e-01            100      1
</init>
<event>
    7      1     {weight:.9e}        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  68.071  68.071  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -68.071  68.071  0.0  0.0  9.0
        25  2    0    0  0  0  0.0  0.0  0.0  64.03124238  64.03124238  0.0  9.0
         5  1    0    0  0  0  20.0  0.0  30.0  36.05551275  0.0  0.0  9.0
        -5  1    0    0  0  0 -20.0  0.0 -30.0  36.05551275  0.0  0.0  9.0
         5  1    3    3  0  0  0.0  25.0  20.0  32.01562119  0.0  0.0  9.0
        -5  1    3    3  0  0  0.0 -25.0 -20.0  32.01562119  0.0  0.0  9.0
</event>
</LesHouchesEvents>
""".format(weight=weight, xsec=xsec_pb)


class HbbValidationTests(unittest.TestCase):
    def test_extract_lhe_4b_sample_builds_weighted_validation_observables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final4b.lhe"
            path.write_text(validation_lhe_text(weight=2.0, xsec_pb=5.0, include_bad_event=True))

            sample = extract_lhe_4b_sample(path, label="gg_hbb_direct")

        self.assertEqual(sample["label"], "gg_hbb_direct")
        self.assertEqual(sample["summary"]["event_count"], 2)
        self.assertEqual(sample["summary"]["accepted_4b_events"], 1)
        self.assertEqual(sample["summary"]["skipped_events"], 1)
        self.assertTrue(math.isclose(sample["summary"]["xsec_pb"], 5.0))
        self.assertTrue(math.isclose(sample["summary"]["weighted_event_sum"], 2.0))

        observables = sample["observables"]
        self.assertEqual(
            sorted(observables),
            [
                "b1_pt",
                "b2_pt",
                "b3_pt",
                "b4_pt",
                "b_pt_all",
                "dr_associated_bb",
                "dr_bb_all",
                "dr_cross_bb",
                "dr_higgs_bb",
                "dr_min_bb",
                "m_4b",
                "m_bb_all",
            ],
        )
        self.assertEqual(len(observables["b_pt_all"]["values"]), 4)
        self.assertEqual(len(observables["dr_bb_all"]["values"]), 6)
        self.assertEqual(len(observables["m_bb_all"]["values"]), 6)
        self.assertEqual(len(observables["m_4b"]["values"]), 1)
        self.assertEqual(observables["b1_pt"]["values"][0], 25.0)
        self.assertTrue(all(weight == 2.0 for weight in observables["b_pt_all"]["weights"]))

    def test_extract_lhe_4b_sample_splits_delta_r_by_higgs_ancestry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final4b_with_mothers.lhe"
            path.write_text(validation_lhe_text_with_higgs_mothers(weight=3.0, xsec_pb=5.0))

            sample = extract_lhe_4b_sample(path, label="gg_hg_forced_split")

        observables = sample["observables"]
        self.assertEqual(len(observables["dr_associated_bb"]["values"]), 1)
        self.assertEqual(len(observables["dr_higgs_bb"]["values"]), 1)
        self.assertEqual(len(observables["dr_cross_bb"]["values"]), 4)
        self.assertEqual(len(observables["dr_min_bb"]["values"]), 1)
        self.assertTrue(all(weight == 3.0 for weight in observables["dr_cross_bb"]["weights"]))
        self.assertTrue(
            math.isclose(
                observables["dr_min_bb"]["values"][0],
                min(observables["dr_bb_all"]["values"]),
            )
        )

    def test_write_lhe_validation_report_uses_sample_report_webpage_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            split_lhe = tmpdir / "gg_hg_forced_split.final4b.lhe"
            direct_lhe = tmpdir / "gg_hbb_direct.final4b.lhe"
            split_lhe.write_text(validation_lhe_text_with_higgs_mothers(weight=1.5, xsec_pb=3.0))
            direct_lhe.write_text(validation_lhe_text_with_higgs_mothers(weight=2.0, xsec_pb=5.0))

            metadata = write_lhe_validation_report(
                split_lhe=split_lhe,
                direct_lhe=direct_lhe,
                output_dir=tmpdir / "report",
            )

            index = Path(metadata["index"])
            table = Path(metadata["table"])
            metadata_path = Path(metadata["metadata"])

            self.assertTrue(index.exists())
            self.assertTrue(table.exists())
            self.assertTrue(metadata_path.exists())
            self.assertIn("4H LHE 4b Validation Observables", index.read_text())
            self.assertIn("class=\"grid\"", index.read_text())
            self.assertIn("gg_hg_forced_split", table.read_text())
            self.assertIn("gg_hbb_direct", table.read_text())
            self.assertEqual(len(metadata["plots"]), 12)
            self.assertIn("associated_pair_deltaR_bb", index.read_text())
            self.assertIn("higgs_decay_deltaR_bb", index.read_text())
            self.assertIn("cross_pair_deltaR_bb", index.read_text())
            self.assertIn("min_deltaR_bb", index.read_text())
            self.assertTrue(all(Path(row["path"]).exists() for row in metadata["plots"]))
            self.assertTrue(json.loads(metadata_path.read_text())["validation_only"])

    def test_prepare_mg5_decks_writes_hg_and_hbb_validation_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            decks = prepare_mg5_decks(
                output_dir=Path(tmp),
                mg5_root=Path("/mg5"),
                events=123,
                overwrite=True,
            )

            split_text = Path(decks["gg_hg"]).read_text()
            direct_text = Path(decks["gg_hbb"]).read_text()

        self.assertIn("generate g g > h g [noborn=QCD]", split_text)
        self.assertIn("output /mg5/gg_hg", split_text)
        self.assertIn("set ebeam1 7000", split_text)
        self.assertIn("set ebeam2 7000", split_text)
        self.assertIn("set nevents 123", split_text)
        self.assertIn("generate g g > h b b~", direct_text)
        self.assertIn("output /mg5/gg_hbb", direct_text)
        self.assertIn("set ebeam1 7000", direct_text)
        self.assertIn("set ebeam2 7000", direct_text)
        self.assertIn("set ptb 15", direct_text)
        self.assertIn("set etab 3.0", direct_text)
        self.assertIn("set drbb 0.3", direct_text)

    def test_validation_chain_dry_run_writes_cards_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            split_input = tmpdir / "gg_hg.lhe"
            direct_input = tmpdir / "gg_hbb.lhe"
            split_input.write_text(validation_lhe_text(weight=1.0, xsec_pb=1.0))
            direct_input.write_text(validation_lhe_text(weight=1.0, xsec_pb=2.0))

            summary = run_validation_chain(
                ValidationRunConfig(
                    split_input_lhe=split_input,
                    direct_input_lhe=direct_input,
                    workdir=tmpdir / "validation",
                    events=1,
                    probe_trials=7,
                    dry_run=True,
                )
            )

            stage1_card = Path(summary["split_stage1_card"])
            split_decay_card = Path(summary["split_decay_card"])
            direct_decay_card = Path(summary["direct_decay_card"])

            self.assertTrue(stage1_card.exists())
            self.assertTrue(split_decay_card.exists())
            self.assertTrue(direct_decay_card.exists())
            self.assertIn("set ForceSplitVeto:MinB 2", stage1_card.read_text())
            self.assertIn("set ForceSplitVeto:ProbeTrials 7", stage1_card.read_text())
            self.assertIn("create Herwig::LHEWriter", split_decay_card.read_text())
            self.assertIn("decaymode h0->b,bbar; 1.0 1 /Herwig/Decays/Hff", direct_decay_card.read_text())
            self.assertEqual(len(summary["commands"]), 6)
            self.assertTrue(Path(summary["summary"]).exists())


if __name__ == "__main__":
    unittest.main()
