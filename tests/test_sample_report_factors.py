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
    signal_generation_rate_factor,
    signal_tag_rate_factor,
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


if __name__ == "__main__":
    unittest.main()
