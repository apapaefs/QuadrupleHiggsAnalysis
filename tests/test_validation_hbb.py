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


def validation_lhe_text_without_mothers_higgs_mass_pair(weight=2.0, xsec_pb=5.0):
    return """<LesHouchesEvents version="1.0">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   {xsec:.9e}   5.000000000e-01            100      1
</init>
<event>
    6      1     {weight:.9e}        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  200.0  200.0  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -200.0  200.0  0.0  0.0  9.0
         5  1    1    2  0  0  18.0  0.0  0.0  18.68154169  5.0  0.0  9.0
        -5  1    1    2  0  0  20.0  1.0  0.0  20.63976744  5.0  0.0  9.0
         5  1    1    2  0  0  62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
        -5  1    1    2  0  0 -62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
</event>
</LesHouchesEvents>
""".format(weight=weight, xsec=xsec_pb)


def validation_final_lhe_text_source_truth_overrides_mass_pair(weight=2.0, xsec_pb=5.0):
    return """<LesHouchesEvents version="1.0">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   {xsec:.9e}   5.000000000e-01            100      1
</init>
<event>
    6      1     {weight:.9e}        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  200.0  200.0  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -200.0  200.0  0.0  0.0  9.0
         5  1    1    2  0  0  62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
        -5  1    1    2  0  0 -62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
         5  1    1    2  0  0  18.0  0.0  0.0  18.68154169  5.0  0.0  9.0
        -5  1    1    2  0  0  20.0  1.0  0.0  20.63976744  5.0  0.0  9.0
</event>
</LesHouchesEvents>
""".format(weight=weight, xsec=xsec_pb)


def validation_source_lhe_text_associated_high_mass_pair(weight=2.0, xsec_pb=5.0):
    return """<LesHouchesEvents version="1.0">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   {xsec:.9e}   5.000000000e-01            100      1
</init>
<event>
    5      1     {weight:.9e}        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  200.0  200.0  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -200.0  200.0  0.0  0.0  9.0
        25  1    1    2  0  0  0.0  0.0  0.0  125.0  125.0  0.0  9.0
         5  1    1    2  0  0  62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
        -5  1    1    2  0  0 -62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
</event>
</LesHouchesEvents>
""".format(weight=weight, xsec=xsec_pb)


def validation_source_lhe_text_two_events_for_global_matching(weight=2.0, xsec_pb=5.0):
    return """<LesHouchesEvents version="1.0">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   {xsec:.9e}   5.000000000e-01            100      1
</init>
<event>
    5      1     {weight:.9e}        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  200.0  200.0  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -200.0  200.0  0.0  0.0  9.0
        25  1    1    2  0  0  0.0  0.0  0.0  125.0  125.0  0.0  9.0
         5  1    1    2  0  0  62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
        -5  1    1    2  0  0 -62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
</event>
<event>
    5      1     {weight:.9e}        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  200.0  200.0  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -200.0  200.0  0.0  0.0  9.0
        25  1    1    2  0  0  0.0  0.0  0.0  125.0  125.0  0.0  9.0
         5  1    1    2  0  0  30.0  5.0  10.0  32.01562119  5.0  0.0  9.0
        -5  1    1    2  0  0 -12.0  24.0 -8.0  28.46049894  5.0  0.0  9.0
</event>
</LesHouchesEvents>
""".format(weight=weight, xsec=xsec_pb)


def validation_final_lhe_text_reversed_source_events(weight=2.0, xsec_pb=5.0):
    return """<LesHouchesEvents version="1.0">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   {xsec:.9e}   5.000000000e-01            100      1
</init>
<event>
    6      1     {weight:.9e}        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  200.0  200.0  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -200.0  200.0  0.0  0.0  9.0
         5  1    1    2  0  0  30.0  5.0  10.0  32.01562119  5.0  0.0  9.0
        -5  1    1    2  0  0 -12.0  24.0 -8.0  28.46049894  5.0  0.0  9.0
         5  1    1    2  0  0  18.0  0.0  0.0  18.68154169  5.0  0.0  9.0
        -5  1    1    2  0  0  20.0  1.0  0.0  20.63976744  5.0  0.0  9.0
</event>
<event>
    6      1     {weight:.9e}        200.0    0.007546771     0.09944864
        21 -1    0    0  501  0  0.0  0.0  200.0  200.0  0.0  0.0  9.0
        21 -1    0    0  0  501  0.0  0.0 -200.0  200.0  0.0  0.0  9.0
         5  1    1    2  0  0  62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
        -5  1    1    2  0  0 -62.29979936  0.0  0.0  62.5  5.0  0.0  9.0
         5  1    1    2  0  0  18.0  0.0  0.0  18.68154169  5.0  0.0  9.0
        -5  1    1    2  0  0  20.0  1.0  0.0  20.63976744  5.0  0.0  9.0
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

    def test_extract_lhe_4b_sample_falls_back_to_higgs_mass_pair_without_mothers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final4b_no_higgs_mothers.lhe"
            path.write_text(validation_lhe_text_without_mothers_higgs_mass_pair(weight=4.0, xsec_pb=5.0))

            sample = extract_lhe_4b_sample(path, label="gg_hg_forced_split")

        observables = sample["observables"]
        self.assertEqual(len(observables["dr_associated_bb"]["values"]), 1)
        self.assertEqual(len(observables["dr_higgs_bb"]["values"]), 1)
        self.assertEqual(len(observables["dr_cross_bb"]["values"]), 4)
        self.assertTrue(all(weight == 4.0 for weight in observables["dr_higgs_bb"]["weights"]))

    def test_extract_lhe_4b_sample_uses_source_lhe_associated_pair_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            final_path = tmpdir / "final4b_no_higgs_mothers.lhe"
            source_path = tmpdir / "pre_decay_source.lhe"
            final_path.write_text(validation_final_lhe_text_source_truth_overrides_mass_pair(weight=4.0, xsec_pb=5.0))
            source_path.write_text(validation_source_lhe_text_associated_high_mass_pair(weight=4.0, xsec_pb=5.0))

            sample = extract_lhe_4b_sample(final_path, label="gg_hg_forced_split", source_lhe=source_path)

        observables = sample["observables"]
        self.assertEqual(len(observables["dr_associated_bb"]["values"]), 1)
        self.assertTrue(math.isclose(observables["dr_associated_bb"]["values"][0], math.pi))
        self.assertLess(observables["dr_higgs_bb"]["values"][0], 0.1)
        self.assertEqual(sample["summary"]["pair_classification"]["source_lhe_match"], 1)
        self.assertEqual(sample["summary"]["pair_classification"]["higgs_mass_fallback"], 0)

    def test_extract_lhe_4b_sample_globally_matches_source_events_when_order_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            final_path = tmpdir / "final4b_reordered.lhe"
            source_path = tmpdir / "pre_decay_source.lhe"
            final_path.write_text(validation_final_lhe_text_reversed_source_events(weight=4.0, xsec_pb=5.0))
            source_path.write_text(validation_source_lhe_text_two_events_for_global_matching(weight=4.0, xsec_pb=5.0))

            sample = extract_lhe_4b_sample(final_path, label="gg_hg_forced_split", source_lhe=source_path)

        self.assertEqual(sample["summary"]["pair_classification"]["source_lhe_match"], 2)
        self.assertEqual(sample["summary"]["pair_classification"]["source_lhe_unmatched"], 0)
        self.assertEqual(sample["summary"]["pair_classification"]["higgs_mass_fallback"], 0)

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

    def test_write_lhe_validation_report_passes_source_lhe_truth_to_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            split_lhe = tmpdir / "gg_hg_forced_split.final4b.lhe"
            split_source = tmpdir / "gg_hg_forced_split.pre_decay.lhe"
            direct_lhe = tmpdir / "gg_hbb_direct.final4b.lhe"
            direct_source = tmpdir / "gg_hbb_direct.pre_decay.lhe"
            split_lhe.write_text(validation_final_lhe_text_source_truth_overrides_mass_pair(weight=1.5, xsec_pb=3.0))
            split_source.write_text(validation_source_lhe_text_associated_high_mass_pair(weight=1.5, xsec_pb=3.0))
            direct_lhe.write_text(validation_final_lhe_text_source_truth_overrides_mass_pair(weight=2.0, xsec_pb=5.0))
            direct_source.write_text(validation_source_lhe_text_associated_high_mass_pair(weight=2.0, xsec_pb=5.0))

            metadata = write_lhe_validation_report(
                split_lhe=split_lhe,
                direct_lhe=direct_lhe,
                output_dir=tmpdir / "report",
                split_source_lhe=split_source,
                direct_source_lhe=direct_source,
            )
            table_text = Path(metadata["table"]).read_text()

        samples = {sample["label"]: sample for sample in metadata["samples"]}
        self.assertEqual(samples["gg_hg_forced_split"]["source_file"], str(split_source))
        self.assertEqual(samples["gg_hbb_direct"]["source_file"], str(direct_source))
        self.assertEqual(samples["gg_hg_forced_split"]["pair_classification"]["source_lhe_match"], 1)
        self.assertEqual(samples["gg_hbb_direct"]["pair_classification"]["source_lhe_match"], 1)
        self.assertEqual(samples["gg_hg_forced_split"]["pair_classification"]["higgs_mass_fallback"], 0)
        self.assertIn("source match", table_text)
        self.assertIn("mH fallback", table_text)

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
