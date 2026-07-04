import math
import csv
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from ForcedSplitting.herwig_cards import (  # noqa: E402
    PROCESS_CONFIGS,
    stage1_lhewriter_card,
    stage2_hwsim_card,
)
from ForcedSplitting.lhe_validation import (  # noqa: E402
    declared_process_ids,
    event_process_ids,
    normalize_single_process_lprup,
    parse_lhe_events,
)
from ForcedSplitting.signal_pipeline import prepare_forced_splitting_inputs  # noqa: E402


TOY_FIXTURE = REPO_DIR / "ForcedSplitting" / "fixtures" / "toy_hhgg.lhe"


class ForcedSplittingTests(unittest.TestCase):
    def test_toy_hhgg_fixture_is_marked_nonproduction_and_valid(self):
        text = TOY_FIXTURE.read_text()

        self.assertIn("toy_hhgg", text)
        self.assertIn("SYNTHETIC", text)
        self.assertIn("not for physics", text)

        events = parse_lhe_events(text)
        self.assertGreaterEqual(len(events), 2)

        for event in events:
            incoming = [p for p in event.particles if p.status == -1]
            final = [p for p in event.particles if p.status == 1]
            self.assertEqual(len(incoming), 2)
            self.assertEqual(len(final), 4)
            self.assertEqual(sorted(abs(p.pid) for p in final), [21, 21, 25, 25])

            total_px = sum(p.px for p in incoming) - sum(p.px for p in final)
            total_py = sum(p.py for p in incoming) - sum(p.py for p in final)
            total_pz = sum(p.pz for p in incoming) - sum(p.pz for p in final)
            total_e = sum(p.energy for p in incoming) - sum(p.energy for p in final)
            self.assertTrue(math.isclose(total_px, 0.0, abs_tol=1e-8))
            self.assertTrue(math.isclose(total_py, 0.0, abs_tol=1e-8))
            self.assertTrue(math.isclose(total_pz, 0.0, abs_tol=1e-8))
            self.assertTrue(math.isclose(total_e, 0.0, abs_tol=1e-8))

            for higgs in [p for p in final if abs(p.pid) == 25]:
                mass2 = higgs.energy**2 - higgs.px**2 - higgs.py**2 - higgs.pz**2
                self.assertTrue(math.isclose(math.sqrt(mass2), 125.0, rel_tol=0.0, abs_tol=1e-8))

            endpoints = Counter()
            for p in incoming:
                if p.color1:
                    endpoints[(p.color1, "anti")] += 1
                if p.color2:
                    endpoints[(p.color2, "colour")] += 1
            for p in final:
                if p.color1:
                    endpoints[(p.color1, "colour")] += 1
                if p.color2:
                    endpoints[(p.color2, "anti")] += 1

            tags = {tag for tag, _ in endpoints}
            for tag in tags:
                self.assertEqual(endpoints[(tag, "colour")], 1)
                self.assertEqual(endpoints[(tag, "anti")], 1)

    def test_hhhg_stage1_card_has_one_forced_split_pair(self):
        card = stage1_lhewriter_card(
            PROCESS_CONFIGS["gg_hhhg"],
            input_lhe="/data/gg_hhhg/unweighted_events.lhe.gz",
            output_prefix="gg_hhhg_split",
            events=100,
            probe_trials=0,
            correction_file="gg_hhhg_split.weights",
        )

        self.assertIn("set ForceSplitVeto:MinB 2", card)
        self.assertIn("set ForceSplitVeto:MinSplitPairs 1", card)
        self.assertIn("set ForceSplitVeto:RequireDistinctHardGluons No", card)
        self.assertIn("set ForceSplitVeto:SplitMaxBEta 3.0", card)
        self.assertIn("set ShowerHandler:LimitEmissions OneFinalStateEmission", card)
        self.assertIn("erase ShowerHandler:DecayInShower 3", card)
        self.assertIn("set LesHouchesHandler:HadronizationHandler NULL", card)
        self.assertIn("set LesHouchesHandler:DecayHandler NULL", card)
        self.assertIn("set ShowerHandler:MPIHandler NULL", card)
        self.assertIn("do SplittingGenerator:DeleteFinalSplitting g->u,ubar", card)
        self.assertIn("do SplittingGenerator:DeleteFinalSplitting g->c,cbar", card)
        self.assertNotIn("SelectDecayModes h0->b,bbar", card)

    def test_hhgg_stage1_card_requires_two_distinct_split_pairs(self):
        card = stage1_lhewriter_card(
            PROCESS_CONFIGS["gg_hhgg"],
            input_lhe="toy_hhgg.lhe",
            output_prefix="toy_hhgg_split",
            events=2,
            probe_trials=10,
            correction_file="toy_hhgg_split.weights",
        )

        self.assertIn("set ForceSplitVeto:MinB 4", card)
        self.assertIn("set ForceSplitVeto:MinSplitPairs 2", card)
        self.assertIn("set ForceSplitVeto:RequireDistinctHardGluons Yes", card)
        self.assertIn("set ForceSplitVeto:SplitMinBPt 15*GeV", card)
        self.assertIn("set ForceSplitVeto:SplitMaxBEta 3.0", card)
        self.assertIn("set ForceSplitVeto:SplitMinDeltaR 0.3", card)
        self.assertIn("set ForceSplitVeto:SplitMinDeltaRToOtherB 0.3", card)
        self.assertIn("set ShowerHandler:LimitEmissions NoLimit", card)
        self.assertIn("set ForceSplitVeto:ProbeTrials 10", card)

    def test_stage2_card_forces_higgs_decays_and_runs_hwsim(self):
        card = stage2_hwsim_card(
            input_lhe="gg_hhhg_split.lhe",
            output_location="events/gg_hhhg_split_hdecay",
            events=100,
            run_name="gg_hhhg_split_hdecay",
        )

        self.assertIn("decaymode h0->b,bbar; 1.0 1 /Herwig/Decays/Hff", card)
        self.assertIn("do /Herwig/Particles/h0:SelectDecayModes h0->b,bbar;", card)
        self.assertIn("library HwSim.so", card)
        self.assertIn("create Herwig::HwSim /Herwig/Analysis/HwSim", card)
        self.assertIn("set /Herwig/Analysis/HwSim:OutputLocation events/gg_hhhg_split_hdecay/", card)
        self.assertNotIn("ForceSplitVeto", card)

    def test_stage2_card_adds_hwsim_output_location_slash(self):
        card = stage2_hwsim_card(
            input_lhe="gg_hhhg_split.lhe",
            output_location="events",
            events=100,
            run_name="gg_hhhg_split_hdecay",
        )

        self.assertIn("set /Herwig/Analysis/HwSim:OutputLocation events/", card)

    def test_card_cli_writes_stage1_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            card_path = Path(tmp) / "gg_hhhg_stage1.in"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ForcedSplitting.herwig_cards",
                    "stage1",
                    "gg_hhhg",
                    "--input-lhe",
                    "unweighted_events.lhe.gz",
                    "--output-prefix",
                    "gg_hhhg_split",
                    "--events",
                    "100",
                    "--card-out",
                    str(card_path),
                ],
                cwd=str(REPO_DIR),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            self.assertTrue(card_path.exists())
            self.assertIn("set ForceSplitVeto:MinB 2", card_path.read_text())

    def test_forced_splitting_pipeline_prepares_stage_cards_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            run_dir = tmpdir / "mg5" / "Events" / "run_gg_hhhg_4_0.0_0.0"
            run_dir.mkdir(parents=True)
            (run_dir / "unweighted_events.lhe.gz").write_text("placeholder\n")

            manifest = prepare_forced_splitting_inputs(
                process="gg_hhhg",
                mg5_dir=tmpdir / "mg5",
                output_dir=tmpdir / "forced",
                events=1000,
                probe_trials=0,
            )

            with manifest.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "written")
            self.assertEqual(rows[0]["c3"], "0.0")
            self.assertEqual(rows[0]["d4"], "0.0")

            stage1_card = Path(rows[0]["stage1_input"])
            stage2_card = Path(rows[0]["stage2_input"])
            self.assertTrue(stage1_card.exists())
            self.assertTrue(stage2_card.exists())
            self.assertIn("set ForceSplitVeto:MinB 2", stage1_card.read_text())
            self.assertIn("set theLHReader:FileName %s" % (run_dir / "unweighted_events.lhe.gz"), stage1_card.read_text())
            self.assertIn("set theLHReader:FileName %s.lhe" % rows[0]["stage1_run_name"], stage2_card.read_text())
            self.assertTrue((tmpdir / "forced" / "stage1_inputs_to_run.txt").exists())
            self.assertTrue((tmpdir / "forced" / "stage1_outputs_to_normalize.txt").exists())
            self.assertTrue((tmpdir / "forced" / "stage2_inputs_to_run.txt").exists())
            self.assertEqual(
                (tmpdir / "forced" / "stage1_inputs_to_run.txt").read_text().strip(),
                stage1_card.name,
            )
            self.assertEqual(
                (tmpdir / "forced" / "stage1_outputs_to_normalize.txt").read_text().strip(),
                rows[0]["stage1_run_name"] + ".lhe",
            )
            self.assertEqual(
                (tmpdir / "forced" / "stage2_inputs_to_run.txt").read_text().strip(),
                stage2_card.name,
            )

    def test_forced_splitting_pipeline_can_filter_to_reference_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            events = tmpdir / "mg5" / "Events"
            good = events / "run_gg_hhgg_4_0.0_0.0"
            skipped = events / "run_gg_hhgg_4_1.0_100.0"
            good.mkdir(parents=True)
            skipped.mkdir(parents=True)
            (good / "unweighted_events.lhe.gz").write_text("placeholder\n")
            (skipped / "unweighted_events.lhe.gz").write_text("placeholder\n")

            reference_manifest = tmpdir / "hhhh_manifest.csv"
            reference_manifest.write_text(
                "status,c3,d4\n"
                "skipped_existing,0.0,0.0\n"
            )

            manifest = prepare_forced_splitting_inputs(
                process="gg_hhgg",
                mg5_dir=tmpdir / "mg5",
                output_dir=tmpdir / "forced",
                events=1000,
                reference_grid_manifest=reference_manifest,
            )

            with manifest.open() as handle:
                rows = list(csv.DictReader(handle))
            statuses = {row["run_dir"]: row["status"] for row in rows}
            self.assertEqual(statuses[str(good)], "written")
            self.assertEqual(statuses[str(skipped)], "skipped_not_in_reference_grid")

    def test_lhe_process_id_normalizer_repairs_herwig_lhewriter_mismatch(self):
        lhe_text = """<LesHouchesEvents version=\"1.0\">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   4.690073e-05   4.690073e-05            100      0
</init>
<event>
    2      1              1        581.354    0.007546771     0.09944864
        25  1    0    0  0  0  0.0  0.0  0.0  125.0  125.0  0.0  9.0
        21  1    0    0  501  502  0.0  0.0  0.0  10.0  0.0  0.0  9.0
</event>
</LesHouchesEvents>
"""

        self.assertEqual(declared_process_ids(lhe_text), [0])
        self.assertEqual(event_process_ids(lhe_text), [1])

        normalized, changed, message = normalize_single_process_lprup(lhe_text)

        self.assertTrue(changed)
        self.assertIn("declared process id 0 -> 1", message)
        self.assertEqual(declared_process_ids(normalized), [1])
        self.assertEqual(event_process_ids(normalized), [1])


if __name__ == "__main__":
    unittest.main()
