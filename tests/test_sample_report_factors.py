import math
import sys
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_DIR / "Code"
sys.path.insert(0, str(CODE_DIR))

from sample_report import (  # noqa: E402
    background_generation_rate_factor,
    background_tag_rate_factor,
    cutflow_rates,
    signal_generation_rate_factor,
    signal_tag_rate_factor,
    terminal_cutflow_table,
)


class SampleReportFactorTests(unittest.TestCase):
    def test_signal_generation_factor_excludes_btagging(self):
        self.assertTrue(
            math.isclose(
                signal_generation_rate_factor(
                    hbb_branching_ratio=0.5824,
                    hbb_power=4,
                    k_factor=2.0,
                ),
                2.0 * 0.5824**4,
            )
        )

    def test_signal_tag_factor_includes_eight_btags(self):
        self.assertTrue(math.isclose(signal_tag_rate_factor(btagging_rate=0.85, btag_power=8), 0.85**8))

    def test_decayed_z6b_generation_factor_does_not_multiply_zbb_again(self):
        metadata = {
            "process_id": "pp_to_z_6b_z_to_bb",
            "description": "pp -> Z + 6b, Z -> b bbar",
            "local_lhe": "merged_z6b_events_10k_1_to_11_unique_20260703-121854.lhe",
        }
        self.assertTrue(
            math.isclose(
                background_generation_rate_factor(
                    metadata,
                    k_factor=2.0,
                    zbb_branching_ratio=0.150998,
                ),
                2.0,
            )
        )

    def test_undecayed_z6b_generation_factor_multiplies_zbb(self):
        metadata = {
            "process_id": "pp_to_z_6b",
            "description": "pp -> Z + 6b",
            "local_lhe": "pp_to_z_6b.lhe",
        }
        self.assertTrue(
            math.isclose(
                background_generation_rate_factor(
                    metadata,
                    k_factor=2.0,
                    zbb_branching_ratio=0.150998,
                ),
                2.0 * 0.150998,
            )
        )

    def test_background_tag_factor_uses_flavor_specific_tags(self):
        metadata = {"b_quarks": 4, "c_quarks": 2, "light_jets": 2}
        self.assertTrue(
            math.isclose(
                background_tag_rate_factor(
                    metadata,
                    btagging_rate=0.85,
                    c_mistag_rate=0.1,
                    light_mistag_rate=0.01,
                ),
                0.85**4 * 0.1**2 * 0.01**2,
            )
        )

    def test_input_rate_includes_tag_factor(self):
        rates = cutflow_rates(
            raw_xsec_fb=10.0,
            generation_rate_factor=2.0,
            tag_rate_factor=0.25,
            normalisation_weight=100.0,
            input_weight_sum=50.0,
            selected_weight_sum=10.0,
        )

        self.assertTrue(math.isclose(rates["generation_xsec_fb"], 20.0))
        self.assertTrue(math.isclose(rates["input_xsec_fb"], 2.5))
        self.assertTrue(math.isclose(rates["xgboost_xsec_fb"], 0.5))

    def test_terminal_cutflow_table_renders_plain_aligned_columns(self):
        rows = [
            {
                "label": r"SM $gg\to hhhh\to 8b$",
                "generation_xsec_fb": 0.023,
                "generation_events": 69.0,
                "input_xsec_fb": 0.0012,
                "input_events": 3.6,
                "xgboost_xsec_fb": 4.2e-5,
                "xgboost_events": 0.126,
            },
            {
                "label": r"$pp\to Z+6b,\ Z\to b\bar{b}$",
                "generation_xsec_fb": 0.0675,
                "generation_events": 202.5,
                "input_xsec_fb": 0.0098,
                "input_events": 29.4,
                "xgboost_xsec_fb": 1.1e-4,
                "xgboost_events": 0.33,
            },
        ]

        table = terminal_cutflow_table(rows, luminosity=3000.0, threshold=0.904)

        self.assertIn("4H sample cutflow / rates", table)
        self.assertIn("sigma_gen [fb]", table)
        self.assertIn("N_XGB", table)
        self.assertIn("SM gg->hhhh->8b", table)
        self.assertIn("pp->Z+6b, Z->b bbar", table)
        self.assertNotIn("$", table)
        self.assertIn("+", table)


if __name__ == "__main__":
    unittest.main()
