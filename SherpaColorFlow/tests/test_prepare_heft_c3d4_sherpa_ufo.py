#!/usr/bin/env python3
"""Tests for the Sherpa-specific heft_c3d4 UFO adapter."""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_heft_c3d4_sherpa_ufo.py"
SOURCE_UFO = ROOT.parent / "MadGraphModels" / "heft_c3d4"
COMMITTED_UFO = ROOT.parent / "MadGraphModels" / "heft_c3d4_sherpa"

SPEC = importlib.util.spec_from_file_location("prepare_heft_c3d4_sherpa_ufo", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load {}".format(SCRIPT))
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class PrepareHeftC3D4SherpaUFOTests(unittest.TestCase):
    def copy_source(self, parent: Path) -> Path:
        source = parent / "heft_c3d4"
        shutil.copytree(
            str(SOURCE_UFO),
            str(source),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        return source

    def test_adds_two_qcd_powers_to_every_reviewed_effective_coupling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            source = self.copy_source(parent)
            output = parent / ADAPTER.OUTPUT_MODEL_NAME

            ADAPTER.adapt_ufo(source, output)

            source_couplings = ADAPTER.discover_effective_couplings(
                (source / "couplings.py").read_text(encoding="utf-8")
            )
            adapted_couplings = ADAPTER.discover_effective_couplings(
                (output / "couplings.py").read_text(encoding="utf-8")
            )
            self.assertEqual(set(source_couplings), set(ADAPTER.EXPECTED_ORIGINAL_ORDERS))
            self.assertEqual(set(adapted_couplings), set(source_couplings))
            for name, source_coupling in source_couplings.items():
                expected = dict(source_coupling.order)
                expected["QCD"] = expected.get("QCD", 0) + 2
                self.assertEqual(adapted_couplings[name].order, expected, name)

            self.assertEqual(adapted_couplings["GC_13"].order, {"HIG": 1, "QCD": 2})
            self.assertEqual(adapted_couplings["GC_GGGGHH"].order, {"HIG": 1, "QCD": 4})
            self.assertEqual(adapted_couplings["GC_GGGGHHH"].order, {"HIG": 1, "QCD": 4})
            self.assertEqual(adapted_couplings["GC_16"].order, {"HIG": 1, "QCD": 2})
            self.assertEqual(adapted_couplings["GC_17"].order, {"HIG": 1, "QCD": 3})

    def test_preserves_source_and_records_matching_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            source = self.copy_source(parent)
            output = parent / ADAPTER.OUTPUT_MODEL_NAME
            original_tree_hash = ADAPTER.sha256_tree(source)
            original_couplings = (source / "couplings.py").read_bytes()

            ADAPTER.adapt_ufo(source, output)

            self.assertEqual(ADAPTER.sha256_tree(source), original_tree_hash)
            self.assertEqual((source / "couplings.py").read_bytes(), original_couplings)
            provenance = json.loads((output / ADAPTER.PROVENANCE_FILENAME).read_text())
            self.assertEqual(provenance["source"]["tree_sha256"], original_tree_hash)
            self.assertEqual(
                provenance["source"]["couplings_py_sha256"],
                ADAPTER.sha256_file(source / "couplings.py"),
            )
            self.assertEqual(
                provenance["adapted"]["tree_sha256_excluding_provenance"],
                ADAPTER.sha256_tree(output),
            )
            self.assertEqual(
                provenance["adapted"]["couplings_py_sha256"],
                ADAPTER.sha256_file(output / "couplings.py"),
            )
            self.assertEqual(provenance["output_model_name"], ADAPTER.OUTPUT_MODEL_NAME)
            self.assertEqual(provenance["transformation"]["qcd_order_increment"], 2)
            self.assertEqual(len(provenance["transformation"]["couplings"]), 11)

    def test_source_root_records_portable_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            source = self.copy_source(parent)
            output = parent / ADAPTER.OUTPUT_MODEL_NAME

            ADAPTER.adapt_ufo(source, output, source_root=parent)

            provenance = json.loads((output / ADAPTER.PROVENANCE_FILENAME).read_text())
            self.assertEqual(provenance["source"]["path"], "heft_c3d4")

    def test_source_root_must_contain_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            parent = Path(tmp)
            source = self.copy_source(parent)
            output = parent / ADAPTER.OUTPUT_MODEL_NAME

            with self.assertRaisesRegex(ADAPTER.AdapterError, "not inside"):
                ADAPTER.adapt_ufo(source, output, source_root=Path(other))
            self.assertFalse(output.exists())

    def test_committed_model_matches_exact_regeneration(self) -> None:
        self.assertTrue(COMMITTED_UFO.is_dir())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / ADAPTER.OUTPUT_MODEL_NAME
            ADAPTER.adapt_ufo(SOURCE_UFO, output, source_root=ROOT.parent)

            expected_files = {
                path.relative_to(COMMITTED_UFO): path.read_bytes()
                for path in COMMITTED_UFO.rglob("*")
                if path.is_file() and not ADAPTER._is_transient(path.relative_to(COMMITTED_UFO))
            }
            regenerated_files = {
                path.relative_to(output): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file() and not ADAPTER._is_transient(path.relative_to(output))
            }
            self.assertEqual(regenerated_files, expected_files)

    def test_metadata_drift_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            source = self.copy_source(parent)
            output = parent / ADAPTER.OUTPUT_MODEL_NAME
            couplings_path = source / "couplings.py"
            text = couplings_path.read_text(encoding="utf-8")
            text = text.replace(
                "order = {'HIG':1})",
                "order = {'HIG':1,'QCD':99})",
                1,
            )
            couplings_path.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ADAPTER.AdapterError, "metadata drifted"):
                ADAPTER.adapt_ufo(source, output)
            self.assertFalse(output.exists())

    def test_cli_refuses_nonempty_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            source = self.copy_source(parent)
            output = parent / ADAPTER.OUTPUT_MODEL_NAME
            output.mkdir()
            marker = output / "do_not_replace.txt"
            marker.write_text("keep me\n")

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), str(output)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not empty", result.stderr)
            self.assertEqual(marker.read_text(), "keep me\n")
            self.assertFalse((output / ADAPTER.PROVENANCE_FILENAME).exists())

    def test_cli_force_replaces_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            source = self.copy_source(parent)
            output = parent / ADAPTER.OUTPUT_MODEL_NAME
            output.mkdir()
            marker = output / "old.txt"
            marker.write_text("old\n")

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), str(output), "--force"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            self.assertFalse(marker.exists())
            self.assertTrue((output / "vertices.py").is_file())
            self.assertTrue((output / ADAPTER.PROVENANCE_FILENAME).is_file())
            self.assertIn("Prepared Sherpa UFO", result.stdout)

    def test_rejects_output_inside_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.copy_source(Path(tmp))
            output = source / ADAPTER.OUTPUT_MODEL_NAME
            with self.assertRaisesRegex(ADAPTER.AdapterError, "inside"):
                ADAPTER.adapt_ufo(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
