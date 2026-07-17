#!/usr/bin/env python3
"""Tests for the UFO ``gg -> hh`` cards and three-point fit workflow."""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIT_SCRIPT = ROOT / "scripts" / "run_gg_hh_c3_fit.py"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_sherpa_run.py"
HH_EXAMPLE = ROOT / "Examples" / "GluonFusion_UFO_HEFT_GG_HH_LHE"
HH4B_EXAMPLE = ROOT / "Examples" / "GluonFusion_UFO_HEFT_GG_HH_2bbbar_LHE"
VENDORED_EXAMPLES = ROOT / "sherpa" / "Examples" / "QuadrupleHiggs"

SPEC = importlib.util.spec_from_file_location("run_gg_hh_c3_fit", FIT_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load %s" % FIT_SCRIPT)
FIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIT)


class GGHHUfoCardTests(unittest.TestCase):
    def test_common_hh_settings_and_orders(self) -> None:
        card = (HH_EXAMPLE / "Sherpa.yaml").read_text()
        self.assertIn("BEAM_ENERGIES: 7000", card)
        self.assertIn("PDF_SET: NNPDF23_nlo_as_0119", card)
        self.assertIn("ALPHAS: {USE_PDF: 1}", card)
        self.assertIn("MODEL: heft_c3d4_sherpa", card)
        self.assertIn("ME_GENERATORS: Comix", card)
        self.assertIn("SHOWER_GENERATOR: None", card)
        self.assertIn("FRAGMENTATION: None", card)
        self.assertIn("MI_HANDLER: None", card)
        self.assertIn("BEAM_REMNANTS: false", card)
        self.assertIn("HARD_DECAYS: {Enabled: false}", card)
        self.assertIn("COLOR_SCHEME: SAMPLE", card)
        self.assertIn(
            "SCALES: VAR{0.25*(H_T2+sqr(2*125.0))}{0.25*(H_T2+sqr(2*125.0))}",
            card,
        )
        self.assertIn("- 21 21 -> 25 25:", card)
        self.assertIn("Min_Amplitude_Order: {QCD: 2, QED: 0, HIG: 1, HIW: 0}", card)
        self.assertIn("Max_Amplitude_Order: {QCD: 2, QED: 1, HIG: 1, HIW: 0}", card)
        self.assertIn("Integrator: SChannel", card)
        self.assertIn("Integration_Error: 0.005", card)
        self.assertIn("SELECTORS: []", card)
        self.assertNotIn("LHEF_ASSIGN_MISSING_QQBAR_SINGLET", card)

    def test_hh4b_inclusive_orders_and_direct_bottom_cuts(self) -> None:
        card = (HH4B_EXAMPLE / "Sherpa.yaml").read_text()
        self.assertIn("- 21 21 -> 25 25 5 -5 5 -5:", card)
        self.assertIn("Min_Amplitude_Order: {QCD: 4, QED: 0, HIG: 0, HIW: 0}", card)
        self.assertIn("Max_Amplitude_Order: {QCD: 6, QED: 2, HIG: 1, HIW: 1}", card)
        self.assertIn("Integration_Error: 0.05", card)
        expected_selectors = (
            "- [PT, 5, 15.0, E_CMS]",
            "- [PT, -5, 15.0, E_CMS]",
            "- [Eta, 5, -3.0, 3.0]",
            "- [Eta, -5, -3.0, 3.0]",
            "- [DR, 5, 5, 0.3, E_CMS]",
            "- [DR, -5, -5, 0.3, E_CMS]",
            "- [DR, 5, -5, 0.3, E_CMS]",
        )
        for selector in expected_selectors:
            self.assertIn(selector, card)
        self.assertEqual(card.count("- [PT,"), 2)
        self.assertEqual(card.count("- [Eta,"), 2)
        self.assertEqual(card.count("- [DR,"), 3)

    def test_parameter_card_has_requested_tagged_inputs(self) -> None:
        param = (HH_EXAMPLE / "param_heft_c3d4_sherpa.dat").read_text()
        self.assertIn("3 1.190000e-01 # aS", param)
        self.assertIn("5 4.750000e+00 # MB", param)
        self.assertIn("6 1.730000e+02 # MT", param)
        self.assertIn("25 1.250000e+02 # MH", param)
        self.assertIn("4 $(C3) # c3", param)
        self.assertIn("6 0.000000e+00 # d4", param)
        self.assertIn("5 4.200000e+00 # ymb", param)
        self.assertEqual(
            (HH4B_EXAMPLE / "param_heft_c3d4_sherpa.dat").read_bytes(),
            (HH_EXAMPLE / "param_heft_c3d4_sherpa.dat").read_bytes(),
        )

    def test_vendored_examples_are_exact_mirrors(self) -> None:
        pairs = (
            (HH_EXAMPLE, VENDORED_EXAMPLES / HH_EXAMPLE.name),
            (HH4B_EXAMPLE, VENDORED_EXAMPLES / HH4B_EXAMPLE.name),
        )
        for source, mirror in pairs:
            self.assertEqual((source / "Sherpa.yaml").read_bytes(), (mirror / "Sherpa.yaml").read_bytes())
            self.assertEqual(
                (source / "param_heft_c3d4_sherpa.dat").read_bytes(),
                (mirror / "param_heft_c3d4_sherpa.dat").read_bytes(),
            )

    def test_prepare_script_copies_both_new_example_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for alias in ("gg_hh_ufo", "gg_hh4b_ufo"):
                run_dir = Path(tmp) / alias
                subprocess.run(
                    [sys.executable, str(PREPARE_SCRIPT), alias, str(run_dir), "--np", "32"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                self.assertTrue((run_dir / "Sherpa.yaml").is_file())
                self.assertTrue((run_dir / "param_heft_c3d4_sherpa.dat").is_file())


class GGHHFitDriverTests(unittest.TestCase):
    def test_log_parser_uses_last_sherpa_cross_section(self) -> None:
        text = (
            "warmup : 1.0 pb +- ( 0.2 pb = 20 % )\n"
            "\x1b[32m21 21 -> 25 25 : 1.08292e-2 pb +- ( 2.83736e-5 pb = 0.26 % )\x1b[0m\n"
        )
        self.assertEqual(FIT.parse_sherpa_cross_sections(text), [(1.0, 0.2), (0.0108292, 2.83736e-5)])

    def test_log_result_above_precision_target_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "integrate.log"
            log.write_text("21 21 -> 25 25 : 1.0 pb +- ( 0.006 pb = 0.6 % )\n")
            with self.assertRaisesRegex(FIT.FitDriverError, "above the 0.005 target"):
                FIT.parse_sherpa_cross_section_log(log)

    def test_symmetric_basis_extracts_exact_quadratic(self) -> None:
        coefficients = FIT.quadratic_coefficients(15.0, 10.0, 9.0)
        self.assertEqual(coefficients, {"A": 2.0, "B": -3.0, "C": 10.0})

    def test_coefficient_covariance_is_exact_for_independent_points(self) -> None:
        covariance = FIT.coefficient_covariance(1.0, 2.0, 3.0)
        expected = (
            (6.5, 2.0, -4.0),
            (2.0, 2.5, 0.0),
            (-4.0, 0.0, 4.0),
        )
        for actual_row, expected_row in zip(covariance, expected):
            for actual, wanted in zip(actual_row, expected_row):
                self.assertTrue(math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-15))

    def test_rendered_fit_cards_use_c3_basis_and_preserve_template(self) -> None:
        source = (HH_EXAMPLE / "Sherpa.yaml").read_text()
        seeds = []
        for c3, _kappa, _label, random_seed in FIT.POINTS:
            rendered = FIT.render_card(source, c3, random_seed)
            self.assertIn("TAGS: {C3: %s}" % FIT.c3_text(c3), rendered)
            self.assertIn("RANDOM_SEED: %d" % random_seed, rendered)
            self.assertIn("Integration_Error: 0.005", rendered)
            self.assertIn("SELECTORS: []", rendered)
            seeds.append(random_seed)
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(seed > 0 for seed in seeds))

    def test_completed_result_hash_mismatch_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / FIT.RESULT_FILENAME
            payload = {
                "c3": -1.0,
                "kappa_lambda": 0.0,
                "random_seed": 2000003,
                "mpi_ranks": 32,
                "sherpa_executable": "Sherpa",
                "mpirun_executable": "mpirun",
                "cross_section_pb": 1.0,
                "uncertainty_pb": 0.01,
                "hashes": {
                    "sherpa_card_sha256": "old-card",
                    "parameter_card_sha256": "param",
                    "model_library_sha256": "model",
                },
            }
            result_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(FIT.FitDriverError, "refusing to overwrite completed grids"):
                FIT.validate_completed_result(
                    result_path,
                    -1.0,
                    0.0,
                    2000003,
                    {
                        "sherpa_card_sha256": "new-card",
                        "parameter_card_sha256": "param",
                        "model_library_sha256": "model",
                    },
                    32,
                    "Sherpa",
                    "mpirun",
                )


if __name__ == "__main__":
    unittest.main()
