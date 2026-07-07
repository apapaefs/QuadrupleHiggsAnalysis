import ast
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_DIR / "Code"
sys.path.insert(0, str(CODE_DIR))

from sample_report import (  # noqa: E402
    attach_poisson_event_interval,
    background_variation_scale_factors,
    best_significance_threshold,
    background_generation_rate_factor,
    background_tag_rate_factor,
    cutflow_rates,
    event_interval_text,
    group_stacked_input_samples,
    poisson_event_interval,
    signal_generation_rate_factor,
    signal_tag_rate_factor,
    stacked_input_bin_edges,
    stacked_input_cross_section_histogram,
    stacked_sample_order,
    terminal_cutflow_table,
    terminal_xgboost_mc_table,
    write_stacked_input_cross_section_plot,
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

    def test_heft_h6b_generation_factor_multiplies_hbb_once(self):
        metadata = {
            "process_id": "gg_h6b_heft",
            "description": "HEFT gg -> h + 6b, h -> b bbar forced with BR=1 in Sherpa",
            "local_lhe": "gg_heft_h_3bbbar_hbb_1k.lhe",
        }
        self.assertTrue(
            math.isclose(
                background_generation_rate_factor(
                    metadata,
                    k_factor=2.0,
                    zbb_branching_ratio=0.150998,
                    hbb_branching_ratio=0.5824,
                ),
                2.0 * 0.5824,
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
                "is_signal": True,
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

    def test_terminal_cutflow_table_keeps_sm_first_then_orders_by_xgboost_yield(self):
        rows = [
            {
                "label": "low background",
                "generation_xsec_fb": 10.0,
                "generation_events": 30000.0,
                "input_xsec_fb": 1.0,
                "input_events": 3000.0,
                "xgboost_xsec_fb": 0.01,
                "xgboost_events": 30.0,
            },
            {
                "label": r"SM $gg\to hhhh\to 8b$",
                "is_signal": True,
                "generation_xsec_fb": 0.023,
                "generation_events": 69.0,
                "input_xsec_fb": 0.0012,
                "input_events": 3.6,
                "xgboost_xsec_fb": 4.2e-5,
                "xgboost_events": 0.126,
            },
            {
                "label": "high background",
                "generation_xsec_fb": 10.0,
                "generation_events": 30000.0,
                "input_xsec_fb": 1.0,
                "input_events": 3000.0,
                "xgboost_xsec_fb": 0.1,
                "xgboost_events": 300.0,
            },
        ]

        table = terminal_cutflow_table(rows, luminosity=3000.0, threshold=0.5)

        self.assertLess(table.index("SM gg->hhhh->8b"), table.index("high background"))
        self.assertLess(table.index("high background"), table.index("low background"))

    def test_terminal_xgboost_mc_table_renders_per_sample_counts(self):
        rows = [
            {
                "process_id": "sm_4h",
                "is_signal": True,
                "description": r"SM $gg\to hhhh\to 8b$",
                "entries": 120,
                "selected_entries": 7,
                "expected_selected_events": 0.42,
                "expected_selected_error": 0.08,
            },
            {
                "process_id": "gg_to_6b_2c",
                "description": r"$gg\to 6b+c\bar{c}$",
                "entries": 50000,
                "selected_entries": 31,
                "expected_selected_events": 1.25,
                "expected_selected_error": 0.2,
            },
        ]

        table = terminal_xgboost_mc_table(
            rows,
            title="Per-sample XGBoost MC event counts",
            threshold=0.904,
        )

        self.assertIn("Per-sample XGBoost MC event counts", table)
        self.assertIn("MC selected", table)
        self.assertIn("7 / 120", table)
        self.assertIn("31 / 50000", table)
        self.assertIn("N_XGB", table)
        self.assertIn("SM gg->hhhh->8b", table)
        self.assertIn("gg->6b+c cbar", table)
        self.assertNotIn("$", table)

    def test_terminal_xgboost_mc_table_orders_by_expected_selected_events(self):
        rows = [
            {
                "description": "low background",
                "entries": 50000,
                "selected_entries": 1,
                "expected_selected_events": 0.1,
            },
            {
                "description": "high background",
                "entries": 50000,
                "selected_entries": 20,
                "expected_selected_events": 2.0,
            },
        ]

        table = terminal_xgboost_mc_table(rows, threshold=0.5)

        self.assertLess(table.index("high background"), table.index("low background"))

    def test_latex_cutflow_writer_sorts_rows_by_importance(self):
        module_path = CODE_DIR / "xgboost_root_varfiles_module.py"
        tree = ast.parse(module_path.read_text())
        writer = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_write_cutflow_latex_table"
        )
        calls_sorter = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sort_rows_by_importance"
            for node in ast.walk(writer)
        )

        self.assertTrue(calls_sorter)

    def test_poisson_event_interval_reports_zero_count_upper_limit(self):
        interval = poisson_event_interval(
            selected_entries=0,
            expected_events=0.0,
            input_entries=100,
            expected_input_events=10.0,
            confidence_level=0.95,
        )

        self.assertTrue(interval["is_upper_limit"])
        self.assertTrue(math.isclose(interval["count_lower"], 0.0))
        self.assertTrue(math.isclose(interval["count_upper"], -math.log(0.05), rel_tol=1.0e-12))
        self.assertTrue(math.isclose(interval["count_upper"] * 0.1, interval["event_upper"], rel_tol=1.0e-12))

    def test_attach_poisson_event_interval_adds_prefixed_fields(self):
        row = {
            "selected_entries": 0,
            "expected_selected_events": 0.0,
            "entries": 100,
            "expected_input_events": 10.0,
        }

        attach_poisson_event_interval(
            row,
            "selected_entries",
            "expected_selected_events",
            "entries",
            "expected_input_events",
            "selected_events",
        )

        self.assertTrue(row["selected_events_is_upper_limit"])
        self.assertTrue(math.isclose(row["selected_events_upper_limit_95cl"], -math.log(0.05) * 0.1))

    def test_event_interval_text_formats_poisson_upper_limit(self):
        row = {
            "expected_selected_events": 0.0,
            "expected_selected_events_is_upper_limit": True,
            "expected_selected_events_upper_limit_95cl": 0.299573,
        }

        self.assertEqual(event_interval_text(row, "expected_selected_events"), "< 0.2996")

    def test_terminal_xgboost_mc_table_uses_poisson_95cl_upper_limit_for_zero_rows(self):
        rows = [
            {
                "process_id": "gg_to_8b",
                "description": r"$gg\to8b$",
                "entries": 100,
                "selected_entries": 0,
                "expected_selected_events": 0.0,
                "expected_selected_events_upper_limit_95cl": 0.2995732273553991,
                "expected_selected_events_is_upper_limit": True,
            }
        ]

        table = terminal_xgboost_mc_table(rows, threshold=0.9)

        self.assertIn("95% CL", table)
        self.assertIn("< 0.2996", table)
        self.assertNotIn("+/- 0", table)

    def test_best_significance_threshold_obeys_min_background_mc_entries(self):
        scores = [0.95, 0.93, 0.91, 0.90, 0.89, 0.80] + [0.20] * 30
        labels = [1, 1, 1, 1, 1, 0] + [0] * 30
        physical_weights = [1.0] * 5 + [0.01] + [0.01] * 30

        unconstrained = best_significance_threshold(scores, labels, physical_weights, min_background_mc_entries=0)
        constrained = best_significance_threshold(scores, labels, physical_weights, min_background_mc_entries=25)

        self.assertGreater(unconstrained["threshold"], 0.2)
        self.assertLessEqual(constrained["threshold"], 0.2)
        self.assertGreaterEqual(constrained["selected_background_entries"], 25)
        self.assertTrue(constrained["mc_stat_requirement_satisfied"])

    def test_background_variation_scale_factors_default_to_factor_four(self):
        factors = background_variation_scale_factors()

        self.assertTrue(math.isclose(factors["factor"], 4.0))
        self.assertTrue(math.isclose(factors["down_factor"], 0.25))
        self.assertTrue(math.isclose(factors["up_factor"], 4.0))

    def test_background_variation_scale_factors_can_be_disabled(self):
        self.assertIsNone(background_variation_scale_factors(enabled=False))

    def test_background_variation_scale_factors_reject_subunit_factor(self):
        with self.assertRaises(ValueError):
            background_variation_scale_factors(0.9)

    def test_stacked_input_histogram_integrates_to_input_cross_section(self):
        y, yerr = stacked_input_cross_section_histogram(
            values=[0.2, 0.8, 1.2],
            weights=[1.0, 3.0, 2.0],
            edges=[0.0, 1.0, 2.0],
            input_xsec_fb=12.0,
        )

        self.assertEqual(list(y), [8.0, 4.0])
        self.assertTrue(math.isclose(sum(y), 12.0))
        self.assertTrue(yerr[0] > 0.0)

    def test_stacked_input_histogram_can_keep_overflow_out_of_visible_normalisation(self):
        edges = stacked_input_bin_edges("m8b", values=[1000.0, 3500.0], sample_weights=[[1.0, 1.0]])

        y, _ = stacked_input_cross_section_histogram(
            values=[1000.0, 3500.0],
            weights=[1.0, 1.0],
            edges=edges,
            input_xsec_fb=10.0,
            normalisation_weight_sum=2.0,
        )

        self.assertTrue(math.isclose(float(sum(y)), 5.0))

    def test_stacked_input_m8b_uses_sixty_bins_to_three_tev(self):
        edges = stacked_input_bin_edges("m8b", values=[0.0, 4000.0], sample_weights=[[1.0, 1.0]])

        self.assertEqual(len(edges), 61)
        self.assertTrue(math.isclose(float(edges[0]), 0.0))
        self.assertTrue(math.isclose(float(edges[-1]), 3000.0))

    def test_stacked_input_non_m8b_keeps_adaptive_binning(self):
        edges = stacked_input_bin_edges("bjet1_pt", values=[10.0, 20.0, 30.0], sample_weights=[[1.0, 1.0, 1.0]])

        self.assertNotEqual(len(edges), 61)
        self.assertTrue(math.isclose(float(edges[0]), 10.0))
        self.assertTrue(math.isclose(float(edges[-1]), 30.0))

    def test_stacked_input_grouping_merges_requested_backgrounds_with_latex_labels(self):
        grouped = group_stacked_input_samples(
            [
                {
                    "label": "SM",
                    "process_id": "sm_4h",
                    "values": [100.0],
                    "weights": [1.0],
                    "input_xsec_fb": 0.01,
                    "is_signal": True,
                },
                {
                    "label": "6b 2j",
                    "process_id": "gg_to_6b_2j",
                    "values": [1.0],
                    "weights": [2.0],
                    "input_xsec_fb": 10.0,
                    "is_signal": False,
                },
                {
                    "label": "6b 2c",
                    "metadata": {"process_id": "gg_to_6b_2c"},
                    "values": [2.0],
                    "weights": [3.0],
                    "input_xsec_fb": 20.0,
                    "is_signal": False,
                },
                {
                    "label": "4b 2c 2j",
                    "process_id": "gg_to_4b_2c_2j",
                    "values": [3.0],
                    "weights": [4.0],
                    "input_xsec_fb": 30.0,
                    "is_signal": False,
                },
                {
                    "label": "4b 4j",
                    "process_id": "gg_to_4b_4j",
                    "values": [4.0],
                    "weights": [5.0],
                    "input_xsec_fb": 40.0,
                    "is_signal": False,
                },
                {
                    "label": "4b 4c",
                    "process_id": "gg_to_4b_4c",
                    "values": [5.0],
                    "weights": [6.0],
                    "input_xsec_fb": 50.0,
                    "is_signal": False,
                },
                {
                    "label": "ttbar 1",
                    "process_id": "ttbar4b_1c3j",
                    "values": [6.0],
                    "weights": [7.0],
                    "input_xsec_fb": 60.0,
                    "is_signal": False,
                },
                {
                    "label": "ttbar 2",
                    "process_id": "ttbar4b_2c2j",
                    "values": [7.0],
                    "weights": [8.0],
                    "input_xsec_fb": 70.0,
                    "is_signal": False,
                },
                {
                    "label": "ttbar 3",
                    "process_id": "ttbar4b_0c4j",
                    "values": [8.0],
                    "weights": [9.0],
                    "input_xsec_fb": 80.0,
                    "is_signal": False,
                },
                {
                    "label": "8b",
                    "process_id": "gg_to_8b",
                    "values": [9.0],
                    "weights": [10.0],
                    "input_xsec_fb": 90.0,
                    "is_signal": False,
                },
                {
                    "label": "Z6b",
                    "process_id": "pp_to_z_6b_z_to_bb",
                    "values": [10.0],
                    "weights": [11.0],
                    "input_xsec_fb": 100.0,
                    "is_signal": False,
                },
                {
                    "label": "h6b",
                    "process_id": "gg_h6b_heft",
                    "values": [11.0],
                    "weights": [12.0],
                    "input_xsec_fb": 110.0,
                    "is_signal": False,
                },
            ]
        )

        by_label = {sample["label"]: sample for sample in grouped}
        expected_labels = [
            r"$gg \rightarrow 8b$",
            r"$gg \rightarrow 6b + 2\slash{b}$",
            r"$gg \rightarrow 4b + 4\slash{b}$",
            r"$gg \rightarrow t\bar{t} + 4b$",
            r"$pp \rightarrow Z+6b$",
            r"$gg \rightarrow h + 6b\ (m_t \rightarrow \infty)$",
            r"$\mathrm{SM}\ gg \rightarrow hhhh \rightarrow 8b$",
        ]

        for label in expected_labels:
            self.assertIn(label, by_label)
            self.assertNotIn("->", label)
        self.assertEqual(list(by_label[r"$gg \rightarrow 6b + 2\slash{b}$"]["values"]), [1.0, 2.0])
        self.assertEqual(list(by_label[r"$gg \rightarrow 6b + 2\slash{b}$"]["weights"]), [2.0, 3.0])
        self.assertTrue(math.isclose(by_label[r"$gg \rightarrow 6b + 2\slash{b}$"]["input_xsec_fb"], 30.0))
        self.assertEqual(list(by_label[r"$gg \rightarrow 4b + 4\slash{b}$"]["values"]), [3.0, 4.0, 5.0])
        self.assertTrue(math.isclose(by_label[r"$gg \rightarrow 4b + 4\slash{b}$"]["input_xsec_fb"], 120.0))
        self.assertEqual(list(by_label[r"$gg \rightarrow t\bar{t} + 4b$"]["values"]), [6.0, 7.0, 8.0])
        self.assertTrue(math.isclose(by_label[r"$gg \rightarrow t\bar{t} + 4b$"]["input_xsec_fb"], 210.0))
        self.assertTrue(grouped[-1]["is_signal"])

    def test_stacked_sample_order_keeps_signal_last(self):
        ordered = stacked_sample_order(
            [
                {"label": "SM", "is_signal": True},
                {"label": "gg->8b", "is_signal": False},
                {"label": "gg->6b+2c", "is_signal": False},
            ]
        )

        self.assertEqual([sample["label"] for sample in ordered], ["gg->8b", "gg->6b+2c", "SM"])

    def test_sample_report_wires_stacked_input_plots_into_index_metadata(self):
        module_text = (CODE_DIR / "xgboost_root_varfiles_module.py").read_text()

        self.assertIn("write_stacked_input_cross_section_plot", module_text)
        self.assertIn("_stacked_input_xsec.png", module_text)
        self.assertIn("stacked_input_xsec", module_text)

    def test_stacked_input_plot_writer_creates_png_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            plot_path = Path(tmp) / "bjet1_pt_stacked_input_xsec.png"

            metadata = write_stacked_input_cross_section_plot(
                plot_path,
                "bjet1_pt",
                [
                    {
                        "label": "background",
                        "values": [10.0, 20.0],
                        "weights": [1.0, 1.0],
                        "input_xsec_fb": 2.0,
                        "is_signal": False,
                    },
                    {
                        "label": "SM",
                        "values": [15.0, 25.0],
                        "weights": [1.0, 1.0],
                        "input_xsec_fb": 0.01,
                        "is_signal": True,
                    },
                ],
            )

            self.assertTrue(plot_path.exists())
            self.assertEqual(metadata["kind"], "stacked_input_xsec")
            self.assertEqual(metadata["signal_scale"], 1000.0)

    def test_stacked_input_plot_writer_reports_group_labels_and_bold_signal_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            plot_path = Path(tmp) / "m8b_stacked_input_xsec.png"

            metadata = write_stacked_input_cross_section_plot(
                plot_path,
                "m8b",
                [
                    {
                        "label": "6b 2j",
                        "process_id": "gg_to_6b_2j",
                        "values": [500.0, 3500.0],
                        "weights": [1.0, 1.0],
                        "input_xsec_fb": 2.0,
                        "is_signal": False,
                    },
                    {
                        "label": "6b 2c",
                        "process_id": "gg_to_6b_2c",
                        "values": [600.0],
                        "weights": [1.0],
                        "input_xsec_fb": 3.0,
                        "is_signal": False,
                    },
                    {
                        "label": "SM",
                        "process_id": "sm_4h",
                        "values": [700.0],
                        "weights": [1.0],
                        "input_xsec_fb": 0.01,
                        "is_signal": True,
                    },
                ],
            )

            self.assertTrue(plot_path.exists())
            self.assertEqual(len(metadata["bin_edges"]), 61)
            self.assertTrue(math.isclose(metadata["bin_edges"][-1], 3000.0))
            labels = [group["label"] for group in metadata["stacked_groups"]]
            display_labels = [group["display_label"] for group in metadata["stacked_groups"]]
            self.assertIn(r"$gg \rightarrow 6b + 2\slash{b}$", labels)
            self.assertIn(r"$\mathrm{SM}\ gg \rightarrow hhhh \rightarrow 8b$", labels)
            self.assertTrue(any(r"\mathbf{\times 1000}" in label for label in display_labels))


if __name__ == "__main__":
    unittest.main()
