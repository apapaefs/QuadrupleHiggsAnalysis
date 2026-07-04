import math
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
from ForcedSplitting.lhe_validation import parse_lhe_events  # noqa: E402


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
        self.assertIn("set /Herwig/Analysis/HwSim:OutputLocation events/gg_hhhg_split_hdecay", card)
        self.assertNotIn("ForceSplitVeto", card)

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


if __name__ == "__main__":
    unittest.main()
