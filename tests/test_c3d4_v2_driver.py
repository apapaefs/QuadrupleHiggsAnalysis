from __future__ import annotations

import tempfile
import unittest
import os
import json
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
    def test_tagged_analysis_paths_do_not_replace_legacy_paths(self):
        raw = Path("/tmp/HW-point.root")
        self.assertEqual(
            DRIVER["_analysis_output_root"](raw),
            Path("/tmp/HW-point_var.smearCMS.root"),
        )
        self.assertEqual(
            DRIVER["_analysis_output_root"](raw, DRIVER["EXTENDED_V2_TAG"]),
            Path("/tmp/HW-point-extended-v2_var.smearCMS.root"),
        )
        self.assertEqual(
            DRIVER["_analysis_log_file"](raw, DRIVER["EXTENDED_V2_TAG"]),
            Path("/tmp/HW-point-extended-v2.analysis.log"),
        )

    def test_tagged_and_untagged_discovery_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "HW-point.root"
            legacy = directory / "HW-point_var.smearCMS.root"
            tagged = directory / "HW-point-extended-v2_var.smearCMS.root"
            for path in (raw, legacy, tagged):
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
            tagged = events / "HW-point-extended-v2_var.smearCMS.root"
            tagged.touch()
            out_file = directory / "HW-point.out"
            out_file.write_text("Total: 10000 10000 1.25e-09\n")

            xsec_fb, generated, source = DRIVER["_metadata_for_root_file"](tagged)

            self.assertEqual(source, out_file)
            self.assertEqual(generated, 10000)
            self.assertAlmostEqual(xsec_fb, 1.25e-3)

    def test_summary_lookup_keeps_extended_tag(self):
        tagged = Path("/tmp/HW-point-extended-v2_var.smearCMS.root")
        self.assertEqual(
            DRIVER["_analysis_summary_file_for_var_root"](tagged),
            Path("/tmp/HW-point-extended-v2.analysis_summary.json"),
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
            var_root = directory / "sample-extended-v2_var.smearCMS.root"
            var_root.touch()
            log = directory / "sample-extended-v2.analysis.log"
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
            tagged = directory / "sample-extended-v2_var.smearCMS.root"
            raw.touch()
            tagged.touch()
            run_one = mock.Mock(return_value=tagged)
            original_validator = DRIVER["_extended_v2_output_is_current"]
            original_ensure = DRIVER["_ensure_analysis_executable"]
            original_run = DRIVER["_run_one_cpp_analysis"]
            try:
                DRIVER["_extended_v2_output_is_current"] = mock.Mock(return_value=False)
                DRIVER["_ensure_analysis_executable"] = mock.Mock(return_value=Path("/tmp/analyzer"))
                DRIVER["_run_one_cpp_analysis"] = run_one
                DRIVER["_ensure_analysis_var_roots"](
                    [directory],
                    executable=Path("/tmp/analyzer"),
                    source_file=Path("/tmp/analyzer.cc"),
                    analysis_tag=DRIVER["EXTENDED_V2_TAG"],
                )
            finally:
                DRIVER["_extended_v2_output_is_current"] = original_validator
                DRIVER["_ensure_analysis_executable"] = original_ensure
                DRIVER["_run_one_cpp_analysis"] = original_run

            run_one.assert_called_once()
            self.assertTrue(run_one.call_args.kwargs["force"])

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
                }

            def IsZombie(self):
                return False

            def Get(self, name):
                return self.objects.get(name)

            def Close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tagged = directory / "sample-extended-v2_var.smearCMS.root"
            tagged.touch()
            summary = directory / "sample-extended-v2.analysis_summary.json"
            summary.write_text(json.dumps({
                "observable_schema": EXTENDED_SCHEMA_ID,
                "total_weight_in": 10.0,
                "feature_tree_mc_events_out": 3,
                "feature_tree_weight_out": 3.0,
                "feature_tree_efficiency": 0.3,
            }))
            os.utime(tagged, (100.0, 100.0))
            os.utime(summary, (101.0, 101.0))

            current_file = RootFile()
            fake_root = types.SimpleNamespace(
                TFile=types.SimpleNamespace(Open=lambda *args: current_file)
            )
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertTrue(DRIVER["_extended_v2_output_is_current"](tagged))

            wrong_pairings = RootFile(pairing_count=104)
            fake_root.TFile.Open = lambda *args: wrong_pairings
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertFalse(DRIVER["_extended_v2_output_is_current"](tagged))

            stale_summary = json.loads(summary.read_text())
            stale_summary["feature_tree_mc_events_out"] = 2
            summary.write_text(json.dumps(stale_summary))
            os.utime(summary, (102.0, 102.0))
            fake_root.TFile.Open = lambda *args: current_file
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertFalse(DRIVER["_extended_v2_output_is_current"](tagged))

            stale_summary["feature_tree_mc_events_out"] = 3
            summary.write_text(json.dumps(stale_summary))
            os.utime(summary, (103.0, 103.0))
            wrong_length = RootFile(feature_length=90)
            fake_root.TFile.Open = lambda *args: wrong_length
            with mock.patch.dict(sys.modules, {"ROOT": fake_root}):
                self.assertFalse(DRIVER["_extended_v2_output_is_current"](tagged))


if __name__ == "__main__":
    unittest.main()
