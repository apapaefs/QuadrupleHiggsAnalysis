from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "prepare_resonance_background_normalization_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_resonance_background_normalization_manifest", MODULE_PATH
)
assert SPEC and SPEC.loader
normalization = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = normalization
SPEC.loader.exec_module(normalization)


class BackgroundNormalizationManifestTests(unittest.TestCase):
    def test_selected_lhe_header_cross_section_is_adopted_with_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lhe = root / "events.lhe"
            lhe.write_text(
                "<LesHouchesEvents>\n"
                "<init>\n"
                " 2212 2212 7000 7000 0 0 0 0 3 2\n"
                " 2.5 0.2 1 1\n"
                " 0.25 0.05 1 2\n"
                "</init>\n"
                "</LesHouchesEvents>\n",
                encoding="utf-8",
            )
            source = root / "input.csv"
            source.write_text(
                "sample_id,source_lhe,cross_section_fb,k_factor\n"
                "target,events.lhe,1500,2\n"
                "other,missing.lhe,3,1\n",
                encoding="utf-8",
            )
            output = root / "output.csv"
            payload = normalization.prepare_manifest(
                root, source, output, ["target"]
            )
            with output.open(newline="", encoding="utf-8") as handle:
                rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(float(rows["target"]["cross_section_fb"]), 2750.0)
            self.assertEqual(rows["target"]["normalization_source"], "source_lhe_init")
            self.assertEqual(float(rows["target"]["normalization_previous_cross_section_fb"]), 1500.0)
            self.assertEqual(float(rows["other"]["cross_section_fb"]), 3.0)
            self.assertEqual(payload["adopted_samples"][0]["sample_id"], "target")
            audit = json.loads(
                output.with_suffix(".normalization_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertAlmostEqual(
                audit["adopted_samples"][0]["adopted_uncertainty_fb"],
                (200.0**2 + 50.0**2) ** 0.5,
            )

    def test_existing_different_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "events.lhe").write_text(
                "<init>\n2212 2212 1 1 0 0 0 0 3 1\n1 0.1 1 1\n</init>\n",
                encoding="utf-8",
            )
            source = root / "input.csv"
            source.write_text(
                "sample_id,source_lhe,cross_section_fb\n"
                "target,events.lhe,500\n",
                encoding="utf-8",
            )
            output = root / "output.csv"
            output.write_text("user data\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "existing manifest differs"):
                normalization.prepare_manifest(root, source, output, ["target"])
            self.assertEqual(output.read_text(encoding="utf-8"), "user data\n")


if __name__ == "__main__":
    unittest.main()
