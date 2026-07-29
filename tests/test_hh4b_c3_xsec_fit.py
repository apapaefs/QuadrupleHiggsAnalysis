from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "Code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from hh4b_c3_xsec import (  # noqa: E402
    evaluate_hh4b_c3_fit,
    fit_hh4b_c3_cross_section,
    load_hh4b_c3_fit,
)


def exact_points():
    return [
        {
            "label": f"p{c3:g}",
            "c3": c3,
            "cross_section_pb": 1.0e-5 + 2.0e-6 * c3 + 3.0e-7 * c3 * c3,
            "integration_error_pb": 1.0e-8,
            "relative_error_percent": 0.1,
        }
        for c3 in (-20.0, -2.0, -1.0, 0.0, 20.0)
    ]


class Hh4bC3CrossSectionFitTests(unittest.TestCase):
    def test_weighted_quadratic_fit_recovers_generator_polynomial(self):
        fit = fit_hh4b_c3_cross_section(exact_points())

        self.assertAlmostEqual(fit["coefficients_pb"][0], 1.0e-5)
        self.assertAlmostEqual(fit["coefficients_pb"][1], 2.0e-6)
        self.assertAlmostEqual(fit["coefficients_pb"][2], 3.0e-7)
        self.assertEqual(fit["ndof"], 2)
        self.assertAlmostEqual(fit["chi2"], 0.0, places=12)

        evaluated = evaluate_hh4b_c3_fit(fit, 3.0)
        self.assertAlmostEqual(evaluated["cross_section_pb"], 1.87e-5)
        self.assertAlmostEqual(evaluated["cross_section_fb"], 0.0187)
        self.assertGreater(evaluated["cross_section_uncertainty_pb"], 0.0)

    def test_fit_round_trip_is_validated(self):
        fit = fit_hh4b_c3_cross_section(exact_points())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit.json"
            path.write_text(json.dumps(fit))
            loaded = load_hh4b_c3_fit(path)

        self.assertEqual(loaded["fit_file"], str(path))
        self.assertEqual(loaded["coefficient_order"], [
            "constant",
            "linear",
            "quadratic",
        ])

    def test_campaign_script_refuses_an_incomplete_five_point_set(self):
        script = (
            REPO
            / "SherpaColorFlow"
            / "scripts"
            / "fit_hh4b_c3_cross_section.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--campaign-dir",
                    str(directory),
                    "--output",
                    str(directory / "fit.json"),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing completed results", result.stderr)

    def test_campaign_script_writes_fit_from_complete_audited_archives(self):
        script = (
            REPO
            / "SherpaColorFlow"
            / "scripts"
            / "fit_hh4b_c3_cross_section.py"
        )
        labels = {
            "c3_m20": -20.0,
            "c3_m2": -2.0,
            "c3_m1": -1.0,
            "c3_0": 0.0,
            "c3_p20": 20.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for label, c3 in labels.items():
                point_dir = directory / label
                point_dir.mkdir()
                xsec = 1.0e-5 + 2.0e-6 * c3 + 3.0e-7 * c3 * c3
                error = xsec * 0.005
                (point_dir / "integrate.hiacc.np64.log").write_text(
                    "2_6__G__G__H__H__b__b~__b__b~ : "
                    f"{xsec:.16g} pb +- ( {error:.16g} pb = 0.5 % ) "
                    "exp. eff: 1 %\n"
                )
                with zipfile.ZipFile(
                    point_dir / "Results_PartiallyUnweighted.zip",
                    "w",
                ) as archive:
                    archive.writestr("result.txt", f"{label}\n")
            output = directory / "fit.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--campaign-dir",
                    str(directory),
                    "--output",
                    str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            fit = load_hh4b_c3_fit(output)

        self.assertEqual(len(fit["points"]), 5)
        self.assertAlmostEqual(fit["coefficients_pb"][0], 1.0e-5)
        self.assertAlmostEqual(fit["coefficients_pb"][1], 2.0e-6)
        self.assertAlmostEqual(fit["coefficients_pb"][2], 3.0e-7)


if __name__ == "__main__":
    unittest.main()
