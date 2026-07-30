from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_DIR = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_DIR / "Signals" / "hhh_c3d4_10k"
sys.path.insert(0, str(CAMPAIGN_DIR))
sys.path.insert(0, str(REPO_DIR))

import campaign  # noqa: E402
import c3d4_bjet_ratio_scan as scan  # noqa: E402
from ForcedSplitting.mg5_grid import prepare_mg5_grid  # noqa: E402


def component_row(
    index: int,
    c3: float,
    d4: float,
    process: str,
    sigma: float,
    error: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "index": index,
        "c3": c3,
        "d4": d4,
        "process": process,
        "audit_status": "ok",
        "audit_issues": "",
        "probe_trial_weight_correction_applied": process == "hhhbb",
    }
    fractions = {
        "exact6": 0.5,
        "exact7": 0.3,
        "ge8": 0.2,
        "ge6": 1.0,
    }
    for category, fraction in fractions.items():
        row[f"acceptance_{category}"] = fraction
        row[f"acceptance_error_{category}"] = 0.01 * fraction
        row[f"sigma_{category}_pb"] = sigma * fraction
        row[f"sigma_{category}_error_pb"] = error * fraction
    return row


class HHHGe6BScanTests(unittest.TestCase):
    def test_authoritative_manifest_is_exactly_153_unique_points(self) -> None:
        points = campaign.load_points(
            REPO_DIR
            / "Signals"
            / "c3d4_40k"
            / "metadata"
            / "points_153.csv"
        )
        self.assertEqual(len(points), 153)
        self.assertEqual(len({point.coordinate for point in points}), 153)
        self.assertEqual(len({point.seed for point in points}), 153)
        self.assertTrue(all(point.run_name.startswith("run_gg_hhh_5_") for point in points))

    def test_statusless_manifest_and_per_point_seeds_feed_hhh_deck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            process_dir = base / "gg_hhh"
            process_dir.mkdir()
            manifest = base / "points.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("c3", "d4", "run_group", "seed"),
                )
                writer.writeheader()
                writer.writerow(
                    {"c3": "0.0", "d4": "0.0", "run_group": "5", "seed": "101"}
                )
                writer.writerow(
                    {"c3": "1.0", "d4": "2.0", "run_group": "5", "seed": "202"}
                )
            summary = prepare_mg5_grid(
                process="gg_hhh",
                process_dir=process_dir,
                reference_grid_manifest=manifest,
                events=10_000,
                cores=64,
                dry_run=True,
            )
            deck = Path(summary["deck"]).read_text()
            self.assertIn("launch run_gg_hhh_5_0.0_0.0", deck)
            self.assertIn("launch run_gg_hhh_5_1.0_2.0", deck)
            self.assertEqual(deck.count("set nevents 10000"), 2)
            self.assertIn("set iseed 101", deck)
            self.assertIn("set iseed 202", deck)
            self.assertTrue(deck.startswith("set run_mode 2\nset nb_core 64"))

    def test_lhe_validator_reads_indexed_coupling_block_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lhe = Path(temporary) / "events.lhe"
            lhe.write_text(
                """<LesHouchesEvents>
#  Integrated weight (pb)  : 4.4032e-05
BLOCK QUARTCOUP
      6 0.000000e+00 # d4
BLOCK TRIPCOUP
      4 0.000000e+00 # c3
<event>
</event>
</LesHouchesEvents>
"""
            )
            audit = campaign.inspect_lhe(lhe, 1, 0.0, 0.0)
            self.assertEqual(audit["status"], "ok")
            self.assertEqual(audit["embedded_c3"], 0.0)
            self.assertEqual(audit["embedded_d4"], 0.0)

    def test_binomial_tag_components_and_boundaries(self) -> None:
        self.assertEqual(scan.binomial_tag_probabilities(5)["ge6"], 0.0)
        six = scan.binomial_tag_probabilities(6)
        self.assertTrue(math.isclose(six["exact6"], 0.85**6))
        self.assertEqual(six["exact7"], 0.0)
        self.assertEqual(six["ge8"], 0.0)
        eight = scan.binomial_tag_probabilities(8)
        self.assertTrue(
            math.isclose(
                eight["ge6"],
                eight["exact6"] + eight["exact7"] + eight["ge8"],
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        )
        perfect = scan.binomial_tag_probabilities(8, efficiency=1.0)
        self.assertEqual(perfect["exact6"], 0.0)
        self.assertEqual(perfect["exact7"], 0.0)
        self.assertEqual(perfect["ge8"], 1.0)
        self.assertEqual(perfect["ge6"], 1.0)

    def test_herwig_parenthetical_cross_section_parser(self) -> None:
        central, error = scan.parse_parenthetical_number("0.12450(1)e-09")
        self.assertTrue(math.isclose(central, 0.12450e-9))
        self.assertTrue(math.isclose(error, 0.00001e-9))
        central, error = scan.parse_parenthetical_number("0.431(4)e-09")
        self.assertTrue(math.isclose(central, 0.431e-9))
        self.assertTrue(math.isclose(error, 0.004e-9))

    def test_cross_section_applies_branching_power_and_uncertainties(self) -> None:
        value3, error3 = scan.cross_section_with_error(
            inclusive_pb=2.0,
            inclusive_error_pb=0.1,
            branching_factor=scan.HBB_BRANCHING_RATIO**3,
            acceptance=0.25,
            acceptance_error=0.02,
        )
        expected3 = 2.0 * scan.HBB_BRANCHING_RATIO**3 * 0.25
        self.assertTrue(math.isclose(value3, expected3))
        self.assertGreater(error3, 0.0)
        value4, _ = scan.cross_section_with_error(
            inclusive_pb=2.0,
            inclusive_error_pb=0.0,
            branching_factor=scan.HBB_BRANCHING_RATIO**4,
            acceptance=0.25,
            acceptance_error=0.0,
        )
        self.assertTrue(
            math.isclose(value4 / value3, scan.HBB_BRANCHING_RATIO)
        )

    def test_hhhbb_normalization_uses_probe_corrected_merge_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            merge_summary = base / "merge_summary.json"
            merge_summary.write_text(
                """{
  "merged_xsec_pb": 4.3137165723808403e-7,
  "merged_xsec_error_pb": 3.6261064868847914e-9,
  "total_events": 10000,
  "zero_weight_events": 2
}
"""
            )
            herwig_out = base / "stage2.out"
            herwig_out.write_text(
                "Total: 10000 10000 0.431(4)e-09\n"
            )
            sample = scan.SampleInput(
                process="hhhbb",
                point=campaign.Point(1, "0.0", "0.0", "fixture", 1),
                run_name="fixture",
                root_file=base / "unused.root",
                hbb_power=3,
                expected_events=10_000,
                herwig_out=herwig_out,
                merge_summary=merge_summary,
                consolidated_xsec_fb=0.000431371657238,
            )
            normalization = scan.normalization(sample)
            self.assertEqual(normalization["status"], "ok")
            self.assertTrue(
                math.isclose(
                    float(normalization["inclusive_xsec_pb"]),
                    4.3137165723808403e-7,
                )
            )
            self.assertNotEqual(
                normalization["inclusive_xsec_pb"],
                normalization["herwig_reference_xsec_pb"],
            )
            self.assertIn(
                "probe-trial-corrected",
                str(normalization["normalization_source"]),
            )
            self.assertGreater(
                float(normalization["inclusive_xsec_error_pb"]), 0.0
            )
            self.assertEqual(normalization["zero_weight_events"], 2)
            self.assertTrue(
                normalization["probe_trial_weight_correction_applied"]
            )

    def test_hhhbb_consolidated_rounding_is_not_an_audit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            merge_summary = base / "merge_summary.json"
            merge_summary.write_text(
                """{
  "merged_xsec_pb": 1.1266428831753292e-4,
  "merged_xsec_error_pb": 1.0e-7,
  "total_events": 10000
}
"""
            )
            herwig_out = base / "stage2.out"
            herwig_out.write_text(
                "Total: 10000 10000 0.11266(1)e-06\n"
            )
            sample = scan.SampleInput(
                process="hhhbb",
                point=campaign.Point(1, "0.0", "0.0", "fixture", 1),
                run_name="fixture",
                root_file=base / "unused.root",
                hbb_power=3,
                expected_events=10_000,
                herwig_out=herwig_out,
                merge_summary=merge_summary,
                consolidated_xsec_fb=0.112664288318,
            )
            normalization = scan.normalization(sample)
            relative_difference = float(
                normalization["relative_difference"]
            )
            self.assertGreater(relative_difference, 1.0e-12)
            self.assertLess(
                relative_difference,
                scan.HHHBB_CONSOLIDATED_RELATIVE_TOLERANCE,
            )
            self.assertEqual(normalization["status"], "ok")

    def test_additive_unmatched_denominator_and_ratios(self) -> None:
        hhh = [component_row(1, 0.0, 0.0, "hhh", 10.0, 1.0)]
        hhhbb = [component_row(1, 0.0, 0.0, "hhhbb", 2.0, 0.5)]
        hhhh = [component_row(1, 0.0, 0.0, "hhhh", 3.0, 0.3)]
        for row, exact6, exact7, ge8 in (
            (hhh[0], 4.0, 3.0, 3.0),
            (hhhbb[0], 0.5, 0.6, 0.9),
            (hhhh[0], 0.9, 0.9, 1.2),
        ):
            row["sigma_exact6_pb"] = exact6
            row["sigma_exact7_pb"] = exact7
            row["sigma_ge8_pb"] = ge8
        combined = scan.combine_component_rows(hhh, hhhbb)
        self.assertEqual(combined[0]["combination_scheme"], "additive_unmatched")
        self.assertTrue(math.isclose(float(combined[0]["sigma_ge6_pb"]), 12.0))
        self.assertTrue(
            math.isclose(float(combined[0]["sigma_exact6_pb"]), 4.5)
        )
        self.assertTrue(
            math.isclose(float(combined[0]["hhhbb_fraction_ge6"]), 1.0 / 6.0)
        )
        self.assertTrue(
            math.isclose(
                float(combined[0]["hhhbb_fraction_exact6"]), 1.0 / 9.0
            )
        )
        ratios = scan.make_ratio_rows(hhhh, hhh, hhhbb, combined)
        self.assertTrue(
            math.isclose(
                float(ratios[0]["ratio_hhhh_over_hhh_plus_hhhbb"]), 0.25
            )
        )
        self.assertTrue(
            math.isclose(float(ratios[0]["ratio_hhhh_over_hhh"]), 0.3)
        )
        self.assertTrue(
            math.isclose(
                float(
                    ratios[0][
                        "ratio_hhhh_exact6_over_hhh_plus_hhhbb_exact6"
                    ]
                ),
                0.2,
            )
        )
        self.assertGreater(
            float(ratios[0]["ratio_hhhh_over_hhh_plus_hhhbb_error"]), 0.0
        )
        self.assertGreater(
            float(
                ratios[0][
                    "ratio_hhhh_exact6_over_hhh_plus_hhhbb_exact6_error"
                ]
            ),
            0.0,
        )
        self.assertGreater(
            float(combined[0]["hhhbb_fraction_ge6_error"]), 0.0
        )
        self.assertGreater(
            float(combined[0]["hhhbb_fraction_exact6_error"]), 0.0
        )

    def test_exact_join_retains_all_153_authoritative_points(self) -> None:
        points = campaign.load_points(
            REPO_DIR
            / "Signals"
            / "c3d4_40k"
            / "metadata"
            / "points_153.csv"
        )
        hhh = [
            component_row(
                point.index,
                float(point.c3),
                float(point.d4),
                "hhh",
                1.0,
                0.1,
            )
            for point in points
        ]
        hhhbb = [
            component_row(
                point.index,
                float(point.c3),
                float(point.d4),
                "hhhbb",
                0.2,
                0.02,
            )
            for point in points
        ]
        combined = scan.combine_component_rows(hhh, hhhbb)
        self.assertEqual(len(combined), 153)
        self.assertEqual(
            {(row["c3"], row["d4"]) for row in combined},
            {point.coordinate for point in points},
        )

    def test_exact6_pointwise_table_uses_exact6_components(self) -> None:
        hhh = [component_row(1, 0.0, 0.0, "hhh", 10.0, 1.0)]
        hhhbb = [component_row(1, 0.0, 0.0, "hhhbb", 2.0, 0.5)]
        hhhh = [component_row(1, 0.0, 0.0, "hhhh", 3.0, 0.3)]
        combined = scan.combine_component_rows(hhh, hhhbb)
        ratios = scan.make_ratio_rows(hhhh, hhh, hhhbb, combined)
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary)
            scan.write_pointwise_ratio_tables(results_dir, ratios)
            payload = json.loads(
                (
                    results_dir
                    / f"{scan.EXACT6_PRIMARY_RATIO_STEM}.json"
                ).read_text()
            )
            row = payload["rows"][0]
            self.assertEqual(
                payload["metadata"]["ratio"],
                "hhhh_exact6/(hhh_exact6+hhhbb_exact6)",
            )
            self.assertEqual(
                payload["metadata"]["tag_requirement"], "exactly 6"
            )
            self.assertIn("hhhh_sigma_exact6_pb", row)
            self.assertNotIn("hhhh_sigma_ge6_pb", row)
            self.assertTrue(
                math.isclose(
                    row[
                        "ratio_hhhh_exact6_over_hhh_plus_hhhbb_exact6"
                    ],
                    0.25,
                )
            )

    def test_zero_denominator_is_masked(self) -> None:
        value, error = scan.ratio_with_error(1.0, 0.1, 0.0, 0.0)
        self.assertTrue(math.isnan(value))
        self.assertTrue(math.isnan(error))

    def test_exact_coordinate_join_rejects_missing_component(self) -> None:
        hhh = [component_row(1, 0.0, 0.0, "hhh", 1.0, 0.1)]
        hhhbb = [component_row(1, 1.0, 0.0, "hhhbb", 1.0, 0.1)]
        with self.assertRaisesRegex(ValueError, "coordinates"):
            scan.combine_component_rows(hhh, hhhbb)

    def test_herwig_card_has_hhh_input_seed_and_no_analysis_delta_r(self) -> None:
        template = (REPO_DIR / "Signals" / "HW-gg_hhhh_SM.in").read_text()
        rendered = campaign.render_herwig_card(
            template,
            Path("/tmp/run_gg_hhh.lhe.gz"),
            "HW-run_gg_hhh_5_0.0_0.0",
            events=10_000,
            seed=12345,
        )
        self.assertIn(
            "set theLHReader:FileName /tmp/run_gg_hhh.lhe.gz", rendered
        )
        self.assertIn("set theGenerator:NumberOfEvents 10000", rendered)
        self.assertIn("set theGenerator:RandomNumberGenerator:Seed 12345", rendered)
        self.assertIn("decaymode h0->b,bbar; 1.0", rendered)
        self.assertIn("set /Herwig/Analysis/HwSim:PTCutJets 10.0", rendered)
        self.assertNotIn("DeltaR", rendered)

    def test_ratio_and_atlas_overlay_plots_are_created_from_fixture(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is unavailable")
        rows = [
            {
                "c3": c3,
                "d4": d4,
                "primary": primary,
                "diagnostic": diagnostic,
                "exact6_primary": exact6_primary,
            }
            for c3, d4, primary, diagnostic, exact6_primary in (
                (-10.0, -200.0, 0.008, 0.012, 0.01),
                (0.0, -200.0, 0.04, 0.06, 0.05),
                (10.0, -200.0, 0.2, 0.3, 0.25),
                (-10.0, 200.0, 0.08, 0.12, 0.1),
                (0.0, 200.0, 0.8, 1.2, 1.0),
                (10.0, 200.0, 12.0, 15.0, 14.0),
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = scan.plot_ratio_contours(
                rows, "primary", base / "primary.pdf", "Primary fixture"
            )
            second = scan.plot_ratio_contours(
                rows,
                "diagnostic",
                base / "diagnostic.pdf",
                "Diagnostic fixture",
            )
            third = scan.plot_ratio_contours(
                rows,
                "exact6_primary",
                base / "exact6_primary.pdf",
                "Exactly-six fixture",
            )
            atlas = scan.plot_ratio_contours(
                rows,
                "exact6_primary",
                base / "exact6_primary_atlas.pdf",
                "Exactly-six fixture",
                include_atlas=True,
            )
            for metadata in (first, second, third, atlas):
                self.assertEqual(metadata["status"], "ok")
                self.assertTrue(Path(metadata["output_pdf"]).is_file())
                self.assertTrue(Path(metadata["output_png"]).is_file())
                self.assertNotIn(0.05, metadata["visible_levels"])
                self.assertIn(0.5, metadata["visible_levels"])
                self.assertEqual(
                    metadata["contour_level_styles"]["0.5"],
                    {"color": "purple", "linestyle": "dashed"},
                )
                self.assertEqual(
                    metadata["interpolation"],
                    "C1 cubic triangular interpolation of log10(pointwise ratio)",
                )
                self.assertLess(
                    metadata[
                        "point_interpolation_max_abs_log10_residual"
                    ],
                    1.0e-8,
                )
                self.assertIn(
                    "outside the Delaunay convex hull",
                    metadata["extrapolation"],
                )
                self.assertEqual(
                    metadata["figure_size_inches"],
                    [8.2, 6.2],
                )
                self.assertEqual(
                    metadata["layout"],
                    "constrained; matches the Fig. 3 plotting canvas",
                )
                self.assertGreater(
                    metadata["axes_box_aspect_ratio"],
                    1.2,
                )
                self.assertLess(
                    metadata["axes_box_aspect_ratio"],
                    1.5,
                )
            self.assertFalse(first["atlas_overlay"])
            self.assertIsNone(first["atlas_reference_curve"])
            self.assertTrue(atlas["atlas_overlay"])
            self.assertIn(
                "ATL-PHYS-PUB-2025-003.pdf",
                atlas["atlas_reference_curve"]["source"],
            )
            self.assertEqual(
                atlas["atlas_reference_curve"]["coordinate_system"],
                (
                    "digitized in kappa3,kappa4 and plotted as "
                    "c3=kappa3-1, d4=kappa4-1"
                ),
            )

    def test_make_plots_writes_three_plain_and_three_atlas_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            results_dir = base / "results"
            results_dir.mkdir()
            with (results_dir / "ratio_points.csv").open(
                "w", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "c3",
                        "d4",
                        "ratio_hhhh_over_hhh_plus_hhhbb",
                        "ratio_hhhh_over_hhh",
                        "ratio_hhhh_exact6_over_hhh_plus_hhhbb_exact6",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "c3": 0.0,
                        "d4": 0.0,
                        "ratio_hhhh_over_hhh_plus_hhhbb": 0.1,
                        "ratio_hhhh_over_hhh": 0.2,
                        "ratio_hhhh_exact6_over_hhh_plus_hhhbb_exact6": 0.3,
                    }
                )
            paths = scan.AnalysisPaths(
                source_repo=base,
                mg5_process=base,
                hhh_herwig_dir=base,
                hhhh_herwig_dir=base,
                hhhbb_workdir=base,
                results_dir=results_dir,
                analyzer=base / "analyzer",
                points_file=base / "points.csv",
            )
            with mock.patch.object(
                scan,
                "plot_ratio_contours",
                side_effect=lambda rows, value_field, output_pdf, title, **kwargs: {
                    "output_pdf": str(output_pdf),
                    "value_field": value_field,
                    "title": title,
                    "atlas_overlay": kwargs.get("include_atlas", False),
                },
            ) as plot:
                payload = scan.make_plots(paths)

            self.assertEqual(plot.call_count, 6)
            output_stems = {
                Path(call.args[2]).stem for call in plot.call_args_list
            }
            self.assertEqual(output_stems, set(scan.PLOT_STEMS))
            self.assertEqual(
                sum(
                    call.kwargs.get("include_atlas", False)
                    for call in plot.call_args_list
                ),
                3,
            )
            self.assertEqual(
                {
                    key
                    for key, value in payload.items()
                    if isinstance(value, dict)
                    and "atlas_overlay" in value
                },
                {
                    "primary",
                    "diagnostic",
                    "exact6_primary",
                    "primary_atlas",
                    "diagnostic_atlas",
                    "exact6_primary_atlas",
                },
            )

    def test_ratio_contour_levels_and_title(self) -> None:
        self.assertEqual(
            scan.RATIO_LEVELS,
            (0.01, 0.1, 0.5, 1.0, 10.0),
        )
        self.assertEqual(
            scan.RATIO_LEVEL_STYLES[0.5],
            {"color": "purple", "linestyle": "dashed"},
        )
        self.assertEqual(
            scan.FIDUCIAL_PLOT_TITLE,
            (
                r"Fiducial $\sigma(gg\rightarrow hhhh\geq 6b)"
                r"/\sigma(gg\rightarrow hhh\geq 6b)$"
            ),
        )
        self.assertEqual(
            scan.FIDUCIAL_EXACT6_PLOT_TITLE,
            (
                r"Fiducial $\sigma(gg\rightarrow hhhh,\,N_{b\mathrm{-tag}}=6)"
                r"/\sigma(gg\rightarrow hhh,\,N_{b\mathrm{-tag}}=6)$"
            ),
        )
        self.assertEqual(len(scan.PLOT_STEMS), 6)
        self.assertTrue(
            all(
                f"{stem}{scan.ATLAS_PLOT_SUFFIX}" in scan.PLOT_STEMS
                for stem in (
                    scan.PRIMARY_PLOT_STEM,
                    scan.DIAGNOSTIC_PLOT_STEM,
                    scan.EXACT6_PRIMARY_PLOT_STEM,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
