import math
import csv
import json
import gzip
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
from ForcedSplitting.lhe_weights import apply_weights  # noqa: E402
from ForcedSplitting.run_chain import ChainConfig, count_lhe_events, run_chain  # noqa: E402
from ForcedSplitting.signal_pipeline import prepare_forced_splitting_inputs  # noqa: E402


TOY_FIXTURE = REPO_DIR / "ForcedSplitting" / "fixtures" / "toy_hhgg.lhe"


def minimal_lhe_text(event_count=1):
    event = """<event>
    2      1              1        581.354    0.007546771     0.09944864
        25  1    0    0  0  0  0.0  0.0  0.0  125.0  125.0  0.0  9.0
        21  1    0    0  501  502  0.0  0.0  0.0  10.0  0.0  0.0  9.0
</event>
"""
    return """<LesHouchesEvents version=\"1.0\">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   1.000000e+01   1.000000e+00            100      0
</init>
%s</LesHouchesEvents>
""" % (event * event_count)


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

    def test_forced_splitting_pipeline_uses_weighted_lhe_when_probe_trials_are_enabled(self):
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
                probe_trials=25,
            )

            with manifest.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["stage1_weighted_lhe"].endswith(".weighted.lhe"))
            self.assertTrue(rows[0]["stage2_lhe_file"].endswith(".weighted.lhe"))

            reweight_line = (tmpdir / "forced" / "stage1_outputs_to_reweight.txt").read_text().strip()
            fields = reweight_line.split()
            self.assertEqual(len(fields), 3)
            self.assertTrue(fields[0].endswith(".lhe"))
            self.assertTrue(fields[1].endswith(".force_split.weights"))
            self.assertTrue(fields[2].endswith(".weighted.lhe"))

            stage2_card = Path(rows[0]["stage2_input"]).read_text()
            self.assertIn("set theLHReader:FileName %s" % fields[2], stage2_card)

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

    def test_lhe_prob_weight_application_updates_event_weights_and_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_lhe = tmpdir / "input.lhe"
            corrections = tmpdir / "split.weights"
            output_lhe = tmpdir / "weighted.lhe"
            input_lhe.write_text(
                """<LesHouchesEvents version=\"1.0\">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   1.000000e+01   1.000000e+00            100      0
</init>
<event>
    2      1              1        581.354    0.007546771     0.09944864
        25  1    0    0  0  0  0.0  0.0  0.0  125.0  125.0  0.0  9.0
        21  1    0    0  501  502  0.0  0.0  0.0  10.0  0.0  0.0  9.0
</event>
<event>
    2      1              3        581.354    0.007546771     0.09944864
        25  1    0    0  0  0  0.0  0.0  0.0  125.0  125.0  0.0  9.0
        21  1    0    0  501  502  0.0  0.0  0.0  10.0  0.0  0.0  9.0
</event>
</LesHouchesEvents>
"""
            )
            corrections.write_text(
                "# accepted_event probe_trials probe_successes p_hat total_attempts post_probe_attempts\n"
                "1 20 5 0.25 23 3\n"
                "2 20 10 0.5 21 1\n"
            )

            apply_weights(input_lhe, corrections, output_lhe)

            weighted_text = output_lhe.read_text()
            self.assertIn("4.375000000e+00", weighted_text)
            self.assertEqual(declared_process_ids(weighted_text), [1])
            event_headers = []
            lines = weighted_text.splitlines()
            for index, line in enumerate(lines):
                if line.strip() == "<event>":
                    event_headers.append(lines[index + 1].split())

            weights = [float(header[2]) for header in event_headers]
            self.assertEqual(len(weights), 2)
            self.assertTrue(math.isclose(weights[0], 1.25, rel_tol=0.0, abs_tol=1e-12))
            self.assertTrue(math.isclose(weights[1], 7.5, rel_tol=0.0, abs_tol=1e-12))
            self.assertTrue(math.isclose(sum(weights) / len(weights), 4.375, rel_tol=0.0, abs_tol=1e-12))

    def test_single_command_chain_applies_trial_weights_before_stage2(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_lhe = tmpdir / "unweighted_events.lhe"
            input_lhe.write_text(minimal_lhe_text(event_count=2))
            workdir = tmpdir / "work"
            commands = []

            def fake_runner(command, cwd):
                commands.append((tuple(command), Path(cwd)))
                run_name = Path(command[-1]).stem
                if command[:2] == ["Herwig", "read"]:
                    (Path(cwd) / ("%s.run" % run_name)).write_text("run\n")
                if command[:2] == ["Herwig", "run"] and run_name.endswith("_stage1"):
                    (Path(cwd) / ("%s.lhe" % run_name)).write_text(
                        """<LesHouchesEvents version=\"1.0\">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   1.000000e+01   1.000000e+00            100      0
</init>
<event>
    2      1              1        581.354    0.007546771     0.09944864
        25  1    0    0  0  0  0.0  0.0  0.0  125.0  125.0  0.0  9.0
        21  1    0    0  501  502  0.0  0.0  0.0  10.0  0.0  0.0  9.0
</event>
<event>
    2      1              3        581.354    0.007546771     0.09944864
        25  1    0    0  0  0  0.0  0.0  0.0  125.0  125.0  0.0  9.0
        21  1    0    0  501  502  0.0  0.0  0.0  10.0  0.0  0.0  9.0
</event>
</LesHouchesEvents>
"""
                    )
                    (Path(cwd) / ("%s.force_split.weights" % run_name)).write_text(
                        "# accepted_event probe_trials probe_successes p_hat total_attempts post_probe_attempts\n"
                        "1 20 5 0.25 23 3\n"
                        "2 20 10 0.5 21 1\n"
                    )
                if command[:2] == ["Herwig", "run"] and run_name.endswith("_stage2"):
                    events_dir = Path(cwd) / "events"
                    events_dir.mkdir(exist_ok=True)
                    (events_dir / ("%s.root" % run_name)).write_text("root\n")

            summary = run_chain(
                ChainConfig(
                    process="gg_hhhg",
                    input_lhe=input_lhe,
                    workdir=workdir,
                    events=2,
                    probe_trials=20,
                    run_name="pilot",
                ),
                runner=fake_runner,
            )

            self.assertEqual(
                [command for command, _ in commands],
                [
                    ("Herwig", "read", "pilot_stage1.in"),
                    ("Herwig", "run", "pilot_stage1.run"),
                    ("Herwig", "read", "pilot_stage2.in"),
                    ("Herwig", "run", "pilot_stage2.run"),
                ],
            )
            self.assertEqual(summary["stage2_lhe"], "pilot_stage1.weighted.lhe")
            self.assertEqual(summary["weight_check"]["correction_rows"], 2)
            self.assertEqual(summary["weight_check"]["zero_success_rows"], 0)
            self.assertTrue(math.isclose(summary["weight_check"]["mean_p_hat"], 0.375))
            self.assertTrue(math.isclose(summary["weight_check"]["weighted_mean_xwgtup"], 4.375))
            self.assertIn("set theLHReader:FileName pilot_stage1.weighted.lhe", (workdir / "pilot_stage2.in").read_text())
            self.assertEqual(json.loads((workdir / "pilot_summary.json").read_text())["stage2_lhe"], "pilot_stage1.weighted.lhe")

    def test_single_command_chain_refuses_more_events_than_input_lhe_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_lhe = tmpdir / "unweighted_events.lhe"
            input_lhe.write_text(minimal_lhe_text(event_count=1))
            commands = []

            with self.assertRaisesRegex(RuntimeError, "only contains 1 event"):
                run_chain(
                    ChainConfig(
                        process="gg_hhhg",
                        input_lhe=input_lhe,
                        workdir=tmpdir / "work",
                        events=2,
                        probe_trials=0,
                        run_name="oversample",
                    ),
                    runner=lambda command, cwd: commands.append((command, cwd)),
                )

            self.assertEqual(commands, [])

    def test_lhe_event_counter_supports_gzipped_lhe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unweighted_events.lhe.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(minimal_lhe_text(event_count=3))

            self.assertEqual(count_lhe_events(path), 3)


if __name__ == "__main__":
    unittest.main()
