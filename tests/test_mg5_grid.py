import csv
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


from ForcedSplitting.mg5_grid import MG5_PROCESS_CONFIGS, _madloop_library_env, _run_command, prepare_mg5_grid


def write_manifest(path):
    rows = [
        {
            "status": "skipped_nonmatching_events",
            "run_name": "HW-run_gg_4h_1_0.0_0.0",
            "run_group": "1",
            "c3": "0.0",
            "d4": "0.0",
        },
        {
            "status": "skipped_existing",
            "run_name": "HW-run_gg_4h_4_0.0_0.0",
            "run_group": "4",
            "c3": "0.0",
            "d4": "0.0",
        },
        {
            "status": "written",
            "run_name": "HW-run_gg_4h_4_1.0_100.0",
            "run_group": "4",
            "c3": "1.0",
            "d4": "100.0",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "run_name", "run_group", "c3", "d4"])
        writer.writeheader()
        writer.writerows(rows)


def write_signal_run_card(path):
    path.write_text(
        """  6500.0 = ebeam1 ! beam 1 total energy in GeV
  6500.0 = ebeam2 ! beam 2 total energy in GeV
  lhapdf = pdlabel ! PDF set
  260000 = lhaid ! if pdlabel=lhapdf
  False = fixed_ren_scale ! if .true. use fixed ren scale
  False = fixed_fac_scale ! if .true. use fixed fac scale
  91.188 = scale ! fixed ren scale
  -1 = dynamical_scale_choice ! Choose one of the preselected dynamical choices
  1.0 = scalefact ! scale factor for event-by-event scales
  average = event_norm ! average/sum
  1 = nhel ! helicities
  1 = sde_strategy ! integration strategy
  15.0 = bwcutoff ! BW cutoff
  4 = maxjetflavor ! Maximum jet pdg code
  False = use_syst ! Enable systematics studies
"""
    )


class MG5GridTests(unittest.TestCase):
    def test_mg5_runner_streams_subprocess_output_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            log_path = tmpdir / "mg5_grid.log"
            captured = io.StringIO()

            with contextlib.redirect_stdout(captured):
                exit_code = _run_command(
                    [sys.executable, "-c", "print('visible mg5 output')"],
                    tmpdir,
                    log_path,
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("visible mg5 output", captured.getvalue())
            self.assertIn("visible mg5 output", log_path.read_text())

    def test_mg5_runner_env_includes_mg5_root_heptools_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            process_dir = tmpdir / "gg_hhhg"
            process_lib = process_dir / "HEPTools" / "lib"
            process_collier = process_dir / "HEPTools" / "collier"
            root_lib = tmpdir / "HEPTools" / "lib"
            root_collier = tmpdir / "HEPTools" / "collier"
            for path in (process_lib, process_collier, root_lib, root_collier):
                path.mkdir(parents=True)

            env = _madloop_library_env(process_dir, {"LD_LIBRARY_PATH": "/existing/lib"})
            paths = env["LD_LIBRARY_PATH"].split(":")

            self.assertEqual(paths[0], str(process_lib))
            self.assertEqual(paths[1], str(process_collier))
            self.assertIn(str(root_lib), paths)
            self.assertIn(str(root_collier), paths)
            self.assertIn("/existing/lib", paths)

    def test_mg5_grid_uses_signal_manifest_and_run_card_for_launch_deck(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            process_dir = tmpdir / "gg_hhhg"
            process_dir.mkdir()
            manifest = tmpdir / "signal_manifest.csv"
            run_card = tmpdir / "run_card.dat"
            write_manifest(manifest)
            write_signal_run_card(run_card)

            summary = prepare_mg5_grid(
                process="gg_hhhg",
                process_dir=process_dir,
                reference_grid_manifest=manifest,
                events=1234,
                signal_run_card=run_card,
                dry_run=True,
            )

            deck_text = Path(summary["deck"]).read_text()
            self.assertNotIn("run_gg_hhhg_1_0.0_0.0", deck_text)
            self.assertTrue(deck_text.startswith("set run_mode 2\nset nb_core 324\n\nlaunch "))
            self.assertIn("launch run_gg_hhhg_4_0.0_0.0 --accuracy=0.02 --points=3000 --iterations=5", deck_text)
            self.assertIn("launch run_gg_hhhg_4_1.0_100.0 --accuracy=0.02 --points=3000 --iterations=5", deck_text)
            self.assertEqual(deck_text.count("set nevents 1234"), 2)
            self.assertEqual(deck_text.count("set run_mode 2"), 1)
            self.assertEqual(deck_text.count("set nb_core 324"), 1)
            self.assertIn("set ebeam1 6500.0", deck_text)
            self.assertIn("set pdlabel lhapdf", deck_text)
            self.assertIn("set lhaid 260000", deck_text)
            self.assertIn("set c3 1.0", deck_text)
            self.assertIn("set d4 100.0", deck_text)

            with Path(summary["manifest"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["status"] for row in rows], ["scheduled", "scheduled"])

    def test_mg5_grid_skips_existing_lhe_unless_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            process_dir = tmpdir / "gg_hhgg"
            existing = process_dir / "Events" / "run_gg_hhgg_4_0.0_0.0"
            existing.mkdir(parents=True)
            (existing / "unweighted_events.lhe.gz").write_text("placeholder\n")
            manifest = tmpdir / "signal_manifest.csv"
            run_card = tmpdir / "run_card.dat"
            write_manifest(manifest)
            write_signal_run_card(run_card)

            summary = prepare_mg5_grid(
                process="gg_hhgg",
                process_dir=process_dir,
                reference_grid_manifest=manifest,
                events=1000,
                signal_run_card=run_card,
                dry_run=True,
            )

            deck_text = Path(summary["deck"]).read_text()
            self.assertNotIn("launch run_gg_hhgg_4_0.0_0.0", deck_text)
            self.assertIn("launch run_gg_hhgg_4_1.0_100.0", deck_text)

            with Path(summary["manifest"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "skipped_existing_lhe")
            self.assertEqual(rows[1]["status"], "scheduled")

    def test_mg5_grid_runs_madevent_with_command_deck(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            process_dir = tmpdir / "gg_hhhg"
            (process_dir / "bin").mkdir(parents=True)
            madevent = process_dir / "bin" / "madevent"
            madevent.write_text("#!/bin/sh\n")
            manifest = tmpdir / "signal_manifest.csv"
            run_card = tmpdir / "run_card.dat"
            write_manifest(manifest)
            write_signal_run_card(run_card)
            calls = []

            def fake_runner(command, cwd, log_path):
                calls.append((command, cwd, log_path))
                return 0

            summary = prepare_mg5_grid(
                process="gg_hhhg",
                process_dir=process_dir,
                reference_grid_manifest=manifest,
                events=1000,
                signal_run_card=run_card,
                runner=fake_runner,
            )

            self.assertEqual(len(calls), 1)
            command, cwd, log_path = calls[0]
            self.assertEqual(command[0], str(madevent))
            self.assertEqual(command[1], summary["deck"])
            self.assertEqual(cwd, process_dir)
            self.assertTrue(str(log_path).endswith("mg5_grid.log"))
            self.assertEqual(summary["run_status"], "complete")

    def test_mg5_grid_allows_core_count_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            process_dir = tmpdir / "gg_hhhg"
            process_dir.mkdir()
            manifest = tmpdir / "signal_manifest.csv"
            run_card = tmpdir / "run_card.dat"
            write_manifest(manifest)
            write_signal_run_card(run_card)

            summary = prepare_mg5_grid(
                process="gg_hhhg",
                process_dir=process_dir,
                reference_grid_manifest=manifest,
                events=1000,
                signal_run_card=run_card,
                cores=8,
                dry_run=True,
            )

            deck_text = Path(summary["deck"]).read_text()
            self.assertTrue(deck_text.startswith("set run_mode 2\nset nb_core 8\n\nlaunch "))
            self.assertEqual(deck_text.count("set run_mode 2"), 1)
            self.assertEqual(deck_text.count("set nb_core 8"), 1)
            self.assertEqual(summary["cores"], 8)

    def test_mg5_grid_supports_hggg_process_directory_and_run_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            process_dir = tmpdir / "gg_hggg"
            process_dir.mkdir()
            manifest = tmpdir / "signal_manifest.csv"
            run_card = tmpdir / "run_card.dat"
            write_manifest(manifest)
            write_signal_run_card(run_card)

            summary = prepare_mg5_grid(
                process="gg_hggg",
                process_dir=process_dir,
                reference_grid_manifest=manifest,
                events=1000,
                signal_run_card=run_card,
                dry_run=True,
            )

            deck_text = Path(summary["deck"]).read_text()
            self.assertIn("generate g g > h g g g [noborn=QCD]", MG5_PROCESS_CONFIGS["gg_hggg"].process_card_line)
            self.assertIn("launch run_gg_hggg_4_0.0_0.0", deck_text)
            self.assertIn("launch run_gg_hggg_4_1.0_100.0", deck_text)
            self.assertEqual(MG5_PROCESS_CONFIGS["gg_hggg"].default_process_dir, "gg_hggg")

    def test_mg5_grid_can_project_reference_grid_to_unique_c3_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            process_dir = tmpdir / "gg_hhgg"
            process_dir.mkdir()
            manifest = tmpdir / "signal_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["status", "run_name", "run_group", "c3", "d4"])
                writer.writeheader()
                writer.writerow({"status": "written", "run_name": "a", "run_group": "4", "c3": "0.0", "d4": "-100.0"})
                writer.writerow({"status": "written", "run_name": "b", "run_group": "4", "c3": "0.0", "d4": "0.0"})
                writer.writerow({"status": "written", "run_name": "c", "run_group": "4", "c3": "1.0", "d4": "100.0"})

            summary = prepare_mg5_grid(
                process="gg_hhgg",
                process_dir=process_dir,
                reference_grid_manifest=manifest,
                events=1000,
                dry_run=True,
                c3_only=True,
            )

            deck_text = Path(summary["deck"]).read_text()
            self.assertIn("launch run_gg_hhgg_4_0.0_0.0", deck_text)
            self.assertIn("launch run_gg_hhgg_4_1.0_0.0", deck_text)
            self.assertNotIn("run_gg_hhgg_4_0.0_-100.0", deck_text)
            self.assertEqual(deck_text.count("set nevents 1000"), 2)
            self.assertEqual(summary["scheduled_points"], 2)
            self.assertTrue(summary["c3_only"])

    def test_mg5_grid_supports_hhbbbb_heft_c3_only_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            process_dir = tmpdir / "gg_hhbbbb_heft"
            process_dir.mkdir()
            manifest = tmpdir / "signal_manifest.csv"
            write_manifest(manifest)

            summary = prepare_mg5_grid(
                process="gg_hhbbbb_heft",
                process_dir=process_dir,
                reference_grid_manifest=manifest,
                events=1000,
                dry_run=True,
                c3_only=True,
            )

            deck_text = Path(summary["deck"]).read_text()
            self.assertIn("generate g g > h h b b~ b b~", MG5_PROCESS_CONFIGS["gg_hhbbbb_heft"].process_card_line)
            self.assertIn("launch run_gg_hhbbbb_heft_4_0.0_0.0", deck_text)
            self.assertIn("launch run_gg_hhbbbb_heft_4_1.0_0.0", deck_text)
            self.assertNotIn("run_gg_hhbbbb_heft_4_1.0_100.0", deck_text)
            self.assertEqual(summary["scheduled_points"], 2)
            self.assertTrue(summary["c3_only"])


if __name__ == "__main__":
    unittest.main()
