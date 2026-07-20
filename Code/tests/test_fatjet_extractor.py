from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest


try:
    import ROOT  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent integration test
    ROOT = None


class FatJetExtractorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.code_dir = Path(__file__).resolve().parents[1]
        cls.executable = cls.code_dir / "FourHiggsFatJetAnalysis"
        if ROOT is None or not cls.executable.is_file():
            raise unittest.SkipTest("PyROOT and a built FourHiggsFatJetAnalysis are required")
        ROOT.gROOT.SetBatch(True)

    def _make_input(self, path: Path) -> None:
        root_file = ROOT.TFile(str(path), "RECREATE")
        tree = ROOT.TTree("Data", "synthetic AK8 reconstruction input")
        weight = array("d", [1.0])
        numb = array("i", [8])
        bjets = array("d", [0.0] * 500)
        bmult = array("i", [1] * 100)
        nfat = array("i", [0])
        fat = array("d", [0.0] * 400)
        softdrop = array("d", [0.0] * 400)
        tau21 = array("d", [0.0] * 100)
        fat_b = array("i", [0] * 100)
        fat_c = array("i", [0] * 100)
        tree.Branch("evweight", weight, "evweight/D")
        tree.Branch("numbJets", numb, "numbJets/I")
        tree.Branch("thebJets", bjets, "thebJets[5][100]/D")
        tree.Branch("bHadronMultiplicity", bmult, "bHadronMultiplicity[100]/I")
        tree.Branch("numFatJets", nfat, "numFatJets/I")
        tree.Branch("theFatJets", fat, "theFatJets[4][100]/D")
        tree.Branch("theSoftDropFatJets", softdrop, "theSoftDropFatJets[4][100]/D")
        tree.Branch("tau21FatJets", tau21, "tau21FatJets[100]/D")
        tree.Branch("bHadronMultiplicityFatJets", fat_b, "bHadronMultiplicityFatJets[100]/I")
        tree.Branch("cHadronMultiplicityFatJets", fat_c, "cHadronMultiplicityFatJets[100]/I")

        for index in range(8):
            pt = 150.0 - 8.0 * index
            phi = 2.0 * math.pi * index / 8.0
            px, py, pz = pt * math.cos(phi), pt * math.sin(phi), 0.0
            energy = math.sqrt(px * px + py * py + 10.0**2)
            for axis, value in enumerate((energy, px, py, pz, 0.0)):
                bjets[axis * 100 + index] = value
        for count in (0, 1, 2, 3, 4):
            nfat[0] = count
            for index in range(100):
                fat_b[index] = fat_c[index] = 0
                tau21[index] = 0.0
                for axis in range(4):
                    fat[axis * 100 + index] = 0.0
                    softdrop[axis * 100 + index] = 0.0
            for index in range(count):
                pt = 450.0 - 25.0 * index
                eta = 1.5 + 0.1 * index
                phi = -2.5 + 0.7 * index
                px = pt * math.cos(phi)
                py = pt * math.sin(phi)
                pz = pt * math.sinh(eta)
                energy = math.sqrt(px * px + py * py + pz * pz + 125.0**2)
                for axis, value in enumerate((energy, px, py, pz)):
                    fat[axis * 100 + index] = value
                    softdrop[axis * 100 + index] = value
                tau21[index] = 0.25 + 0.05 * index
                fat_b[index] = 2 if index % 2 == 0 else 1
                fat_c[index] = 0 if index % 2 == 0 else 1
            tree.Fill()
        # A >=4B AK8 jet is diagnostic-only and therefore leaves one resolved
        # hypothesis, rather than becoming a fifth single-Higgs candidate.
        nfat[0] = 1
        fat_b[0] = 4
        tree.Fill()
        tree.Write()
        root_file.Close()

    def _run(self, input_path: Path, output_path: Path, seed: int, no_smear: bool) -> None:
        command = [
            str(self.executable),
            str(input_path),
            "--output",
            str(output_path),
            "--seed",
            str(seed),
        ]
        if no_smear:
            command.append("--no-smear")
        subprocess.run(command, check=True, capture_output=True, text=True)

    def test_zero_to_four_candidates_and_deterministic_smearing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "input.root"
            output = base / "features.root"
            repeated = base / "features-repeat.root"
            self._make_input(source)
            self._run(source, output, seed=77, no_smear=False)
            self._run(source, repeated, seed=77, no_smear=False)
            first = ROOT.TFile.Open(str(output))
            second = ROOT.TFile.Open(str(repeated))
            first_tree = first.Get("ResonanceFeatures")
            second_tree = second.Get("ResonanceFeatures")
            self.assertEqual(int(first_tree.GetEntries()), 32)
            self.assertEqual(int(second_tree.GetEntries()), 32)
            for entry in range(32):
                first_tree.GetEntry(entry)
                second_tree.GetEntry(entry)
                self.assertEqual(int(first_tree.event_index), int(second_tree.event_index))
                self.assertEqual(int(first_tree.hypothesis_index), int(second_tree.hypothesis_index))
                self.assertEqual(float(first_tree.m4h), float(second_tree.m4h))
                self.assertEqual(int(first_tree.n_merged), int(second_tree.n_merged))
                self.assertEqual(
                    int(first_tree.n_ak8_retained),
                    int(first_tree.n_true_fat_pass)
                    + int(first_tree.n_true_fat_fail)
                    + int(first_tree.n_fake_fat_pass)
                    + int(first_tree.n_fake_fat_fail),
                )
                self.assertEqual(
                    int(first_tree.n_true_single)
                    + int(first_tree.n_c_mistag)
                    + int(first_tree.n_light_mistag),
                    2 * (4 - int(first_tree.n_merged)),
                )
            first.Close()
            second.Close()
            summary = json.loads(output.with_suffix(".analysis_summary.json").read_text())
            self.assertEqual(summary["reconstructable_counter"]["events"], 6)
            self.assertEqual(summary["hypothesis_row_counter"]["events"], 32)
            self.assertEqual(summary["diagnostics"]["hh_diagnostic_events"], 1)
            self.assertLessEqual(
                summary["diagnostics"]["max_pattern_probability_residual_nominal"],
                1.0e-12,
            )
            self.assertLessEqual(
                summary["diagnostics"]["max_pattern_probability_residual_conservative"],
                1.0e-12,
            )


if __name__ == "__main__":
    unittest.main()
