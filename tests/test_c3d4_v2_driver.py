from __future__ import annotations

import tempfile
import unittest
import os
import json
import subprocess
import sys
import types
from unittest import mock
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "4h_analyzer.py"
LOCAL_CLI_MARKER = '\n\nif __name__ == "__main__" and "--legacy" not in _sys.argv:'
SOURCE = MODULE_PATH.read_text().split(LOCAL_CLI_MARKER, 1)[0]
DRIVER = {"__file__": str(MODULE_PATH), "__name__": "fourhiggs_driver_v2_test"}
exec(compile(SOURCE, str(MODULE_PATH), "exec"), DRIVER)


class C3D4V2DriverTests(unittest.TestCase):
    def test_replot_is_mutually_exclusive_with_the_legacy_limit_scan(self):
        result = subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--replot-c3d4-study-contours",
                "--run-c3d4-limit-scan",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)
        self.assertIn("--run-c3d4-limit-scan", result.stderr)

    def test_v2_mode_defaults_use_distinct_output_directories_and_strategies(self):
        configured = {}
        for mode in (
            "smoke",
            "preview",
            "fast-sm",
            "fast-pooled",
            "fast-parameterized",
            "full",
        ):
            args = types.SimpleNamespace(
                study_mode=mode,
                study_outdir=None,
                training_strategy=None,
            )
            configured[mode] = DRIVER["_configure_v2_mode_defaults"](args)

        self.assertEqual(
            configured["smoke"].study_outdir,
            DRIVER["_REPO_DIR"]
            / "xgboost_c3d4_study_v2_uniform-smear-v1_smoke",
        )
        self.assertEqual(
            configured["preview"].study_outdir,
            DRIVER["_REPO_DIR"]
            / "xgboost_c3d4_study_v2_uniform-smear-v1_preview",
        )
        self.assertEqual(
            configured["fast-sm"].study_outdir,
            DRIVER["_REPO_DIR"]
            / "xgboost_c3d4_study_v2_uniform-smear-v1_fast-sm",
        )
        self.assertEqual(
            configured["fast-pooled"].study_outdir,
            DRIVER["_REPO_DIR"]
            / "xgboost_c3d4_study_v2_uniform-smear-v1_fast-pooled",
        )
        self.assertEqual(
            configured["fast-parameterized"].study_outdir,
            DRIVER["_REPO_DIR"]
            / "xgboost_c3d4_study_v2_uniform-smear-v1_fast-parameterized",
        )
        self.assertEqual(
            configured["full"].study_outdir,
            DRIVER["_REPO_DIR"] / "xgboost_c3d4_study_v2_uniform-smear-v1",
        )
        self.assertEqual(
            len({args.study_outdir for args in configured.values()}), 6
        )
        self.assertEqual(
            configured["smoke"].training_strategy, "sm-crossfit-v2"
        )
        self.assertEqual(
            configured["fast-sm"].training_strategy, "sm-crossfit-v2"
        )
        self.assertEqual(
            configured["preview"].training_strategy, "pooled-crossfit-v2"
        )
        self.assertEqual(
            configured["fast-pooled"].training_strategy,
            "pooled-crossfit-v2",
        )
        self.assertEqual(
            configured["fast-parameterized"].training_strategy,
            "parameterized-crossfit-v1",
        )
        self.assertEqual(
            configured["full"].training_strategy, "pooled-crossfit-v2"
        )

    def test_no_pyhf_uses_a_distinct_fast_sm_output_directory(self):
        args = types.SimpleNamespace(
            study_mode="fast-sm",
            study_outdir=None,
            training_strategy=None,
            no_pyhf=True,
        )

        configured = DRIVER["_configure_v2_mode_defaults"](args)

        self.assertEqual(
            configured.study_outdir,
            DRIVER["_REPO_DIR"]
            / "xgboost_c3d4_study_v2_uniform-smear-v1_fast-sm_cut-only",
        )
        self.assertEqual(configured.training_strategy, "sm-crossfit-v2")

    def test_no_pyhf_uses_a_distinct_fast_parameterized_output_directory(self):
        args = types.SimpleNamespace(
            study_mode="fast-parameterized",
            study_outdir=None,
            training_strategy=None,
            no_pyhf=True,
        )

        configured = DRIVER["_configure_v2_mode_defaults"](args)

        self.assertEqual(
            configured.study_outdir,
            DRIVER["_REPO_DIR"]
            / (
                "xgboost_c3d4_study_v2_uniform-smear-v1_"
                "fast-parameterized_cut-only"
            ),
        )
        self.assertEqual(
            configured.training_strategy,
            "parameterized-crossfit-v1",
        )

    def test_no_pyhf_is_forwarded_as_a_cut_only_mode_override(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args = types.SimpleNamespace(
                shape_jobs=1,
                progress_interval=30.0,
                analysis_max_events=None,
                reuse_sm_optuna_from=None,
                study_mode="fast-sm",
                observable_set="extended-91-v2",
                feature_profile="full91",
                training_strategy="sm-crossfit-v2",
                optuna_trials=0,
                max_events=None,
                smoke_max_events=2000,
                no_pyhf=True,
                study_outdir=directory / "cut-only",
                c3d4_scan_outdir=directory / "legacy",
            )
            policy = types.SimpleNamespace(name="fast-sm")
            with mock.patch(
                "c3d4_xgboost_runner._resolve_study_mode",
                return_value=policy,
            ) as resolve_mode, mock.patch(
                "c3d4_xgboost_runner._validate_study_output_mode",
                side_effect=ValueError("stop after mode resolution"),
            ):
                with self.assertRaisesRegex(SystemExit, "stop after mode resolution"):
                    DRIVER["_run_c3d4_xgboost_study_cli_impl"](args)

        self.assertIs(resolve_mode.call_args.kwargs["run_shape"], False)

    def test_cli_help_exposes_no_pyhf_cut_only_switch(self):
        result = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--no-pyhf", result.stdout)
        self.assertIn("--no-shape-limits", result.stdout)
        self.assertIn("fast-parameterized", result.stdout)

    def test_analysis_max_events_is_rejected_before_root_discovery(self):
        args = types.SimpleNamespace(
            shape_jobs=1,
            progress_interval=30.0,
            analysis_max_events=10,
        )
        discover = mock.Mock()
        original = DRIVER["_discover_analysis_inputs"]
        DRIVER["_discover_analysis_inputs"] = discover
        try:
            with self.assertRaisesRegex(
                SystemExit, "does not allow --analysis-max-events"
            ):
                DRIVER["_run_c3d4_xgboost_study_cli_impl"](args)
        finally:
            DRIVER["_discover_analysis_inputs"] = original
        discover.assert_not_called()

    def test_nonfinite_progress_interval_is_rejected_before_root_discovery(self):
        args = types.SimpleNamespace(shape_jobs=1, progress_interval=float("inf"))
        discover = mock.Mock()
        original = DRIVER["_discover_analysis_inputs"]
        DRIVER["_discover_analysis_inputs"] = discover
        try:
            with self.assertRaisesRegex(SystemExit, "finite and positive"):
                DRIVER["_run_c3d4_xgboost_study_cli_impl"](args)
        finally:
            DRIVER["_discover_analysis_inputs"] = original
        discover.assert_not_called()

    def test_cli_rejects_invalid_shape_jobs_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args = types.SimpleNamespace(
                shape_jobs=0,
                progress_interval=30.0,
                study_outdir=directory / "study",
                c3d4_scan_outdir=directory / "legacy",
            )
            with self.assertRaisesRegex(SystemExit, "shape-jobs"):
                DRIVER["_run_c3d4_xgboost_study_cli"](args)

            progress = json.loads(
                (args.study_outdir / "study_progress.json").read_text()
            )
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["last_error"]["error_type"], "SystemExit")

    def test_preview_can_retry_after_failure_before_input_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args = types.SimpleNamespace(
                shape_jobs=0,
                progress_interval=30.0,
                study_mode="preview",
                study_outdir=directory / "preview",
                c3d4_scan_outdir=directory / "legacy",
            )

            with self.assertRaisesRegex(SystemExit, "shape-jobs"):
                DRIVER["_run_c3d4_xgboost_study_cli"](args)

            manifest = json.loads(
                (args.study_outdir / "method_manifest.json").read_text()
            )
            self.assertEqual(manifest["study_mode"], "preview")

            code_dir = Path(__file__).resolve().parents[1] / "Code"
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))
            from c3d4_xgboost_runner import _validate_study_output_mode

            _validate_study_output_mode(args.study_outdir, "preview")

    def test_invalid_tagged_v2_is_not_reused_when_regeneration_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            DRIVER,
            {"_extended_v2_output_is_current": mock.Mock(return_value=False)},
        ):
            directory = Path(directory)
            raw = directory / "sample.root"
            tagged = directory / f"sample-{DRIVER['EXTENDED_V2_TAG']}_var.smearCMS.root"
            raw.touch()
            tagged.touch()

            with self.assertRaisesRegex(
                SystemExit, "--no-run-missing-analysis"
            ):
                DRIVER["_ensure_analysis_var_roots"](
                    [directory],
                    executable=Path("/tmp/analyzer"),
                    source_file=Path("/tmp/analyzer.cc"),
                    analysis_tag=DRIVER["EXTENDED_V2_TAG"],
                    run_missing=False,
                )

    def test_csv_background_v2_checks_completeness_and_mistag_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "background.root"
            tagged = directory / f"background-{DRIVER['EXTENDED_V2_TAG']}_var.smearCMS.root"
            raw.touch()
            tagged.touch()
            sample = {
                "process_id": "background",
                "raw_root": raw,
                "c_quarks": 1,
                "light_jets": 2,
            }
            args = types.SimpleNamespace(
                force_analysis=False,
                no_run_missing_analysis=True,
                analysis_source=Path("/tmp/analyzer.cc"),
                analysis_exe=Path("/tmp/analyzer"),
                analysis_max_events=None,
                analysis_jobs=1,
                include_auxiliary_samples=False,
            )
            validator = mock.Mock(return_value=True)
            with mock.patch.dict(
                DRIVER, {"_extended_v2_output_is_current": validator}
            ):
                roots = DRIVER["_ensure_background_csv_var_roots"](
                    [sample], args, analysis_tag=DRIVER["EXTENDED_V2_TAG"]
                )

            self.assertEqual(roots, [tagged])
            validator.assert_called_once_with(
                tagged,
                raw_root=raw,
                source_file=args.analysis_source,
                expected_c_mistags=1,
                expected_light_mistags=2,
            )

            with mock.patch.dict(
                DRIVER,
                {"_extended_v2_output_is_current": mock.Mock(return_value=False)},
            ):
                with self.assertRaisesRegex(
                    SystemExit, "--no-run-missing-analysis"
                ):
                    DRIVER["_ensure_background_csv_var_roots"](
                        [sample], args, analysis_tag=DRIVER["EXTENDED_V2_TAG"]
                    )

    def test_explicit_backgrounds_reject_heterogeneous_global_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first = directory / "HW-first.root"
            second = directory / "HW-second.root"
            first.touch()
            second.touch()
            csv_file = directory / "backgrounds.csv"
            csv_file.write_text(
                "process_id,events,cross_section_pb,local_lhe,b_quarks,c_quarks,light_jets\n"
                "first,100,1.0,first.lhe.gz,8,0,0\n"
                "second,100,1.0,second.lhe.gz,6,1,1\n"
            )

            with self.assertRaisesRegex(SystemExit, "heterogeneous mistag compositions"):
                DRIVER["_validate_explicit_background_composition"](
                    [first, second], csv_file, 0, 0
                )
            with self.assertRaisesRegex(SystemExit, "does not match"):
                DRIVER["_validate_explicit_background_composition"](
                    [second], csv_file, 0, 0
                )

            DRIVER["_validate_explicit_background_composition"](
                [second], csv_file, 1, 1
            )

    def test_legacy_csv_background_retains_existing_output_without_raw_source(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "background.root"
            legacy = directory / "background_var.smearCMS.root"
            legacy.touch()
            sample = {
                "process_id": "background",
                "raw_root": raw,
                "c_quarks": 0,
                "light_jets": 0,
            }
            args = types.SimpleNamespace(
                force_analysis=True,
                no_run_missing_analysis=False,
                analysis_source=Path("/tmp/analyzer.cc"),
                analysis_exe=Path("/tmp/analyzer"),
                analysis_max_events=None,
                analysis_jobs=1,
                include_auxiliary_samples=False,
            )

            roots = DRIVER["_ensure_background_csv_var_roots"](
                [sample], args, analysis_tag=None
            )

            self.assertEqual(roots, [legacy])

    def test_parallel_shape_threads_are_forced_to_one_before_runner_import(self):
        variables = DRIVER["_SHAPE_THREAD_ENVIRONMENT"]
        original = {name: os.environ.get(name) for name in variables}
        try:
            for name in variables:
                os.environ[name] = "8"
            configured = DRIVER["_configure_parallel_shape_threads"](4)
            self.assertEqual(configured, {name: "1" for name in variables})
            self.assertEqual({name: os.environ[name] for name in variables}, configured)
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_tagged_analysis_paths_do_not_replace_legacy_paths(self):
        raw = Path("/tmp/HW-point.root")
        tag = DRIVER["EXTENDED_V2_TAG"]
        self.assertEqual(
            DRIVER["_analysis_output_root"](raw),
            Path("/tmp/HW-point_var.smearCMS.root"),
        )
        self.assertEqual(
            DRIVER["_analysis_output_root"](raw, tag),
            Path(f"/tmp/HW-point-{tag}_var.smearCMS.root"),
        )
        self.assertEqual(
            DRIVER["_analysis_log_file"](raw, tag),
            Path(f"/tmp/HW-point-{tag}.analysis.log"),
        )

    def test_tagged_and_untagged_discovery_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "HW-point.root"
            legacy = directory / "HW-point_var.smearCMS.root"
            previous_v2 = directory / "HW-point-extended-v2_var.smearCMS.root"
            tagged = directory / f"HW-point-{DRIVER['EXTENDED_V2_TAG']}_var.smearCMS.root"
            for path in (raw, legacy, previous_v2, tagged):
                path.touch()

            legacy_roots, legacy_raw = DRIVER["_discover_analysis_inputs"]([directory])
            tagged_roots, tagged_raw = DRIVER["_discover_analysis_inputs"](
                [directory], analysis_tag=DRIVER["EXTENDED_V2_TAG"]
            )

            self.assertEqual(legacy_roots, [legacy])
            self.assertEqual(tagged_roots, [tagged])
            self.assertEqual(legacy_raw, [raw])
            self.assertEqual(tagged_raw, [raw])

    def test_tagged_variable_root_uses_canonical_herwig_out_file(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            events = directory / "events"
            events.mkdir()
            tagged = events / f"HW-point-{DRIVER['EXTENDED_V2_TAG']}_var.smearCMS.root"
            tagged.touch()
            out_file = directory / "HW-point.out"
            out_file.write_text("Total: 10000 10000 1.25e-09\n")

            xsec_fb, generated, source = DRIVER["_metadata_for_root_file"](tagged)

            self.assertEqual(source, out_file)
            self.assertEqual(generated, 10000)
            self.assertAlmostEqual(xsec_fb, 1.25e-3)

    def test_hhhbb_metadata_prefers_exact_merged_lhe_cross_section(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            events = directory / "events"
            point = directory / "run_gg_hhhg_4_-7.5_50.0"
            events.mkdir()
            point.mkdir()
            tagged = (
                events
                / (
                    "run_gg_hhhg_4_-7.5_50.0_hhhbb_stage2-"
                    f"{DRIVER['EXTENDED_V2_TAG']}_var.smearCMS.root"
                )
            )
            tagged.touch()
            summary = point / "merge_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "merged_xsec_pb": 0.00020191483113482072,
                        "total_events": 10000,
                    }
                )
            )
            (point / "run_gg_hhhg_4_-7.5_50.0_hhhbb_stage2.out").write_text(
                "Total: 10000 10002 0.202(2)e-06\n"
            )

            xsec_fb, generated, source = DRIVER[
                "_metadata_for_hhhbb_scored_signal_root"
            ](tagged, 10000)

            self.assertEqual(source, summary)
            self.assertEqual(generated, 10000)
            self.assertAlmostEqual(xsec_fb, 0.20191483113482072)

    def test_summary_lookup_keeps_extended_tag(self):
        tag = DRIVER["EXTENDED_V2_TAG"]
        tagged = Path(f"/tmp/HW-point-{tag}_var.smearCMS.root")
        self.assertEqual(
            DRIVER["_analysis_summary_file_for_var_root"](tagged),
            Path(f"/tmp/HW-point-{tag}.analysis_summary.json"),
        )
        self.assertEqual(DRIVER["_canonical_sample_name"](tagged), "HW-point")

    def test_extended_header_newer_than_binary_triggers_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory)
            source = code / "FourHiggs8bAnalysis_smear_CMS.cc"
            header = code / "Extended91Observables.h"
            executable = code / "FourHiggs8bAnalysis_smear_CMS"
            makefile = code / "Makefile"
            for path in (source, header, executable, makefile):
                path.touch()
            os.utime(source, (100.0, 100.0))
            os.utime(executable, (110.0, 110.0))
            os.utime(header, (120.0, 120.0))

            with mock.patch("subprocess.run") as run:
                result = DRIVER["_ensure_analysis_executable"](
                    executable, source, rebuild=True
                )

            self.assertEqual(result, executable)
            run.assert_called_once()

    def test_feature_tree_log_fallback_accepts_cpp_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tag = DRIVER["EXTENDED_V2_TAG"]
            var_root = directory / f"sample-{tag}_var.smearCMS.root"
            var_root.touch()
            log = directory / f"sample-{tag}.analysis.log"
            log.write_text(
                "feature-tree MC events = 99\n"
                "feature-tree weight out = 98.5\n"
                "feature-tree efficiency = 0.099\n"
            )

            summary = DRIVER["_read_analysis_summary_for_var_root"](var_root)

            self.assertEqual(summary["feature_tree_mc_events_out"], 99.0)
            self.assertEqual(summary["feature_tree_weight_out"], 98.5)
            self.assertEqual(summary["feature_tree_efficiency"], 0.099)

    def test_invalid_existing_tagged_output_is_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "sample.root"
            tagged = directory / f"sample-{DRIVER['EXTENDED_V2_TAG']}_var.smearCMS.root"
            raw.touch()
            tagged.touch()
            run_one = mock.Mock(return_value=tagged)
            progress = mock.Mock()
            original_validator = DRIVER["_extended_v2_output_is_current"]
            original_ensure = DRIVER["_ensure_analysis_executable"]
            original_run = DRIVER["_run_one_cpp_analysis"]
            try:
                DRIVER["_extended_v2_output_is_current"] = mock.Mock(
                    side_effect=(False, True)
                )
                DRIVER["_ensure_analysis_executable"] = mock.Mock(return_value=Path("/tmp/analyzer"))
                DRIVER["_run_one_cpp_analysis"] = run_one
                DRIVER["_ensure_analysis_var_roots"](
                    [directory],
                    executable=Path("/tmp/analyzer"),
                    source_file=Path("/tmp/analyzer.cc"),
                    analysis_tag=DRIVER["EXTENDED_V2_TAG"],
                    progress_callback=progress,
                )
            finally:
                DRIVER["_extended_v2_output_is_current"] = original_validator
                DRIVER["_ensure_analysis_executable"] = original_ensure
                DRIVER["_run_one_cpp_analysis"] = original_run

            run_one.assert_called_once()
            progress.assert_called_once_with(1, 1, "sample.root")
            self.assertTrue(run_one.call_args.kwargs["force"])

    def test_legacy_mixed_variable_root_discovery_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "a.root"
            paired = directory / "a_var.smearCMS.root"
            standalone = directory / "b_var.smearCMS.root"
            for path in (raw, paired, standalone):
                path.touch()

            roots = DRIVER["_ensure_analysis_var_roots"](
                [directory],
                executable=Path("/tmp/analyzer"),
                source_file=Path("/tmp/analyzer.cc"),
                analysis_tag=None,
            )

            self.assertEqual(roots, [paired, standalone])

    def test_current_tagged_output_requires_the_complete_data3_contract(self):
        from observable_schemas import (
            EXTENDED_FEATURE_NAMES,
            EXTENDED_FEATURE_UNITS,
            EXTENDED_SCHEMA_ID,
            PAIRING_COUNT,
        )

        class Named:
            def __init__(self, title):
                self.title = title

            def GetTitle(self):
                return self.title

        class Parameter:
            def __init__(self, value):
                self.value = value

            def GetVal(self):
                return self.value

        class Leaf:
            def __init__(self, length):
                self.length = length

            def GetLenStatic(self):
                return self.length

        class Tree:
            def __init__(self, feature_length=None):
                self.feature_length = feature_length

            def GetEntries(self):
                return 3

            def GetBranch(self, name):
                return object()

            def GetLeaf(self, name):
                return Leaf(self.feature_length) if name == "features" else None

        class RootFile:
            def __init__(self, feature_length=91, pairing_count=105):
                self.objects = {
                    "Data2": Tree(),
                    "Data3": Tree(feature_length),
                    "Data3_observable_schema": Named(EXTENDED_SCHEMA_ID),
                    "Data3_feature_count": Parameter(91),
                    "Data3_feature_names_json": Named(json.dumps(EXTENDED_FEATURE_NAMES)),
                    "Data3_feature_units_json": Named(json.dumps(EXTENDED_FEATURE_UNITS)),
                    "Data3_pairing_count": Parameter(pairing_count),
                    "analysis_output_tag": Named(DRIVER["EXTENDED_V2_TAG"]),
                    "jet_smearing_model_id": Named(DRIVER["JET_SMEARING_MODEL_ID"]),
                    "jet_smearing_acceptance_order": Named(
                        DRIVER["JET_SMEARING_ACCEPTANCE_ORDER"]
                    ),
                    "jet_smearing_fourvector_scaling": Named(
                        DRIVER["JET_SMEARING_FOURVECTOR_SCALING"]
                    ),
                    "jet_smearing_seed": Parameter(DRIVER["JET_SMEARING_SEED"]),
                    "jet_smearing_min_energy_gev": Parameter(
                        DRIVER["JET_SMEARING_MIN_ENERGY_GEV"]
                    ),
                    "jet_smearing_gaussian_draws_per_jet": Parameter(1),
                    "jet_smearing_correlated_mass_scaling": Parameter(1),
                    "max_smearing_mass_scaling_residual_gev": Parameter(1.0e-12),
                }

            def IsZombie(self):
                return False

            def Get(self, name):
                return self.objects.get(name)

            def Close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tag = DRIVER["EXTENDED_V2_TAG"]
            tagged = directory / f"sample-{tag}_var.smearCMS.root"
            raw = directory / "sample.root"
            tagged.touch()
            raw.touch()
            summary = directory / f"sample-{tag}.analysis_summary.json"
            summary.write_text(json.dumps({
                "observable_schema": EXTENDED_SCHEMA_ID,
                "analysis_output_tag": DRIVER["EXTENDED_V2_TAG"],
                "jet_smearing_model_id": DRIVER["JET_SMEARING_MODEL_ID"],
                "jet_smearing_acceptance_order": DRIVER[
                    "JET_SMEARING_ACCEPTANCE_ORDER"
                ],
                "jet_smearing_fourvector_scaling": DRIVER[
                    "JET_SMEARING_FOURVECTOR_SCALING"
                ],
                "jet_smearing_correlated_mass_scaling": True,
                "jet_smearing_preserves_jet_mass": False,
                "jet_smearing_gaussian_draws_per_jet": 1,
                "jet_smearing_seed": DRIVER["JET_SMEARING_SEED"],
                "jet_smearing_min_energy_gev": DRIVER[
                    "JET_SMEARING_MIN_ENERGY_GEV"
                ],
                "max_smearing_mass_scaling_residual_gev": 1.0e-12,
                "input_file": str(raw),
                "mc_events_in": 3,
                "c_mistags": 0,
                "light_mistags": 0,
                "required_true_bjets": 8,
                "true_b_upward_pt_migrations": 3,
                "true_b_downward_pt_migrations": 2,
                "non_b_upward_pt_migrations": 6,
                "non_b_downward_pt_migrations": 4,
                "true_b_upward_pt_migrations_raw_pt_10_12_gev": 1,
                "true_b_upward_pt_migrations_raw_pt_12_15_gev": 1,
                "true_b_upward_pt_migrations_raw_pt_15_20_gev": 1,
                "non_b_upward_pt_migrations_raw_pt_10_12_gev": 1,
                "non_b_upward_pt_migrations_raw_pt_12_15_gev": 2,
                "non_b_upward_pt_migrations_raw_pt_15_20_gev": 3,
                "total_weight_in": 10.0,
                "preselection_mc_events_out": 2,
                "preselection_weight_out": 2.0,
                "preselection_efficiency": 0.2,
                "feature_tree_mc_events_out": 3,
                "feature_tree_weight_out": 3.0,
                "feature_tree_efficiency": 0.3,
                "analysis_mc_events_out": 1,
                "analysis_weight_out": 1.0,
                "analysis_efficiency": 0.1,
            }))
            valid_summary = json.loads(summary.read_text())
            os.utime(raw, (99.0, 99.0))
            os.utime(tagged, (100.0, 100.0))
            os.utime(summary, (101.0, 101.0))

            current_file = RootFile()
            current_file.objects["Data"] = Tree()
            fake_root = types.SimpleNamespace(
                TFile=types.SimpleNamespace(Open=lambda *args: current_file)
            )
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertTrue(
                    DRIVER["_extended_v2_output_is_current"](
                        tagged,
                        raw_root=raw,
                        expected_c_mistags=0,
                        expected_light_mistags=0,
                    )
                )
                self.assertFalse(
                    DRIVER["_extended_v2_output_is_current"](
                        tagged,
                        raw_root=raw,
                        expected_c_mistags=1,
                        expected_light_mistags=0,
                    )
                )

            wrong_smearing = RootFile()
            wrong_smearing.objects["Data"] = Tree()
            wrong_smearing.objects["jet_smearing_model_id"] = Named(
                "cms-energy-massless-v0"
            )
            fake_root.TFile.Open = lambda *args: wrong_smearing
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertFalse(
                    DRIVER["_extended_v2_output_is_current"](
                        tagged, raw_root=raw
                    )
                )

            wrong_pairings = RootFile(pairing_count=104)
            wrong_pairings.objects["Data"] = Tree()
            fake_root.TFile.Open = lambda *args: wrong_pairings
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertFalse(
                    DRIVER["_extended_v2_output_is_current"](
                        tagged, raw_root=raw
                    )
                )

            wrong_seed = RootFile()
            wrong_seed.objects["Data"] = Tree()
            wrong_seed.objects["jet_smearing_seed"] = Parameter(
                DRIVER["JET_SMEARING_SEED"] + 1
            )
            fake_root.TFile.Open = lambda *args: wrong_seed
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertFalse(
                    DRIVER["_extended_v2_output_is_current"](
                        tagged, raw_root=raw
                    )
                )

            fake_root.TFile.Open = lambda *args: current_file
            invalid_summary_fields = (
                ("input_file", str(directory / "different.root")),
                ("jet_smearing_seed", DRIVER["JET_SMEARING_SEED"] + 1),
                ("jet_smearing_min_energy_gev", 1.0e-5),
                ("true_b_upward_pt_migrations", 4),
                ("max_smearing_mass_scaling_residual_gev", 1.0e-4),
                ("feature_tree_efficiency", 0.31),
            )
            for index, (field, value) in enumerate(invalid_summary_fields, start=1):
                invalid_summary = dict(valid_summary)
                invalid_summary[field] = value
                summary.write_text(json.dumps(invalid_summary))
                os.utime(summary, (101.0 + index, 101.0 + index))
                with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                    self.assertFalse(
                        DRIVER["_extended_v2_output_is_current"](
                            tagged, raw_root=raw
                        ),
                        field,
                    )

            summary.write_text(json.dumps(valid_summary))
            os.utime(summary, (109.0, 109.0))
            stale_summary = json.loads(summary.read_text())
            stale_summary["mc_events_in"] = 2
            summary.write_text(json.dumps(stale_summary))
            os.utime(summary, (102.0, 102.0))
            fake_root.TFile.Open = lambda *args: current_file
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertFalse(
                    DRIVER["_extended_v2_output_is_current"](
                        tagged, raw_root=raw
                    )
                )

            stale_summary["mc_events_in"] = 3
            stale_summary["feature_tree_mc_events_out"] = 2
            summary.write_text(json.dumps(stale_summary))
            os.utime(summary, (103.0, 103.0))
            fake_root.TFile.Open = lambda *args: current_file
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertFalse(
                    DRIVER["_extended_v2_output_is_current"](
                        tagged, raw_root=raw
                    )
                )

            stale_summary["feature_tree_mc_events_out"] = 3
            summary.write_text(json.dumps(stale_summary))
            os.utime(summary, (104.0, 104.0))
            wrong_length = RootFile(feature_length=90)
            wrong_length.objects["Data"] = Tree()
            fake_root.TFile.Open = lambda *args: wrong_length
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertFalse(
                    DRIVER["_extended_v2_output_is_current"](
                        tagged, raw_root=raw
                    )
                )


if __name__ == "__main__":
    unittest.main()
