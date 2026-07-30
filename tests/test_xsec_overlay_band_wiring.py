import ast
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_DIR / "Code" / "xgboost_root_varfiles_module.py"


def _module_tree():
    return ast.parse(MODULE_PATH.read_text())


def _function_def(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


class HHHHXsecOverlayBandWiringTests(unittest.TestCase):
    def test_hhhh_xsec_overlay_accepts_and_draws_background_variation_band(self):
        tree = _module_tree()
        writer = _function_def(tree, "_write_hhhh_xsec_limit_overlay_plot")
        argument_names = [arg.arg for arg in writer.args.args]
        self.assertIn("background_variation_band", argument_names)

        calls = [
            node
            for node in ast.walk(writer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_draw_background_variation_band"
        ]
        self.assertTrue(calls)

        metadata_keys = [
            node.value
            for node in ast.walk(writer)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertIn("background_variation_band_drawn", metadata_keys)

    def test_limit_scan_passes_background_variation_band_to_hhhh_xsec_overlay(self):
        tree = _module_tree()
        scan_writer = _function_def(tree, "write_c3d4_limit_scan")
        overlay_calls = [
            node
            for node in ast.walk(scan_writer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_write_hhhh_xsec_limit_overlay_plot"
        ]
        self.assertTrue(overlay_calls)
        keyword_names = {keyword.arg for call in overlay_calls for keyword in call.keywords}
        self.assertIn("background_variation_band", keyword_names)

    def test_limit_scan_accepts_hbb_branching_ratio_keyword_from_cli(self):
        tree = _module_tree()
        scan_writer = _function_def(tree, "write_c3d4_limit_scan")
        argument_names = [arg.arg for arg in scan_writer.args.args]

        self.assertIn("hbb_branching_ratio", argument_names)

    def test_hhhh_xsec_overlay_uses_requested_perturbative_unitarity_label(self):
        module_text = MODULE_PATH.read_text()
        self.assertIn(r"Perturbative unitarity, $hh \rightarrow hh$", module_text)
        self.assertNotIn(r"Perturbativity $|\mathrm{Re}\,a_0| = 0.5$", module_text)

    def test_hhhh_xsec_overlay_uses_requested_atlas_and_our_limit_labels(self):
        module_text = MODULE_PATH.read_text()
        self.assertIn(r"$gg\rightarrow hhh \rightarrow 6 b$, ATL-PHYS-PUB-2025-003 (no syst.)", module_text)
        self.assertIn(r"$gg \rightarrow hhhh \rightarrow 8b$, Poisson", module_text)
        self.assertNotIn(r"Our limit", module_text)
        self.assertNotIn(r"\%", module_text)
        self.assertIn("poisson_confidence_percent_label", module_text)

    def test_hhhh_xsec_overlay_title_carries_luminosity_without_ratio_text(self):
        module_text = MODULE_PATH.read_text()
        self.assertIn(r"$gg \to hhhh$ at 14 TeV, ", module_text)
        self.assertIn("_luminosity_legend_label(luminosity)", module_text)
        self.assertNotIn(r"$gg \to hhhh$ at 14 TeV: $\sigma(c_3,d_4)/\sigma(0,0)$", module_text)

    def test_c3d4_plot_defaults_match_api_and_cli_viewports(self):
        module_text = MODULE_PATH.read_text()
        self.assertIn("plot_c3_range=(-20.0, 20.0)", module_text)
        self.assertIn("plot_d4_range=(-300.0, 300.0)", module_text)
        self.assertIn('default=-20.0, help="Minimum c3', (REPO_DIR / "4h_analyzer.py").read_text())
        self.assertIn('default=20.0, help="Maximum c3', (REPO_DIR / "4h_analyzer.py").read_text())
        self.assertIn('default=-300.0, help="Minimum d4', (REPO_DIR / "4h_analyzer.py").read_text())
        self.assertIn('default=300.0, help="Maximum d4', (REPO_DIR / "4h_analyzer.py").read_text())

    def test_hhhh_xsec_overlay_writes_atlas_variant_without_ratio_contours(self):
        module_text = MODULE_PATH.read_text()
        self.assertIn("c3d4_hhhh_xsec_with_95cl_atl_phys_pub_2025_003_no_ratio_contours.png", module_text)
        self.assertIn("_write_c3d4_atlas_limit_overlay_no_xsec_plot", module_text)
        self.assertIn("hhhh_xsec_atlas_overlay_no_ratio_contours_plot", module_text)

    def test_no_ratio_contours_variant_does_not_use_hhhh_cross_section_calculation(self):
        tree = _module_tree()
        helper = _function_def(tree, "_write_c3d4_atlas_limit_overlay_no_xsec_plot")
        called_names = {
            node.func.id
            for node in ast.walk(helper)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        self.assertNotIn("_read_hhhh_xsec_points", called_names)
        self.assertNotIn("_fit_c3d4_chebyshev", called_names)
        self.assertNotIn("_evaluate_c3d4_chebyshev_grid", called_names)
        self.assertNotIn("_make_hhhh_xsec_log_levels", called_names)
        self.assertNotIn("_make_hhhh_xsec_line_levels", called_names)

    def test_background_variation_band_style_uses_stronger_fill_and_thinner_boundaries(self):
        module_text = MODULE_PATH.read_text()
        self.assertIn('"alpha": 0.32', module_text)
        self.assertIn('"boundary_linewidth": 0.75', module_text)
        self.assertIn('band.get("boundary_linewidth", 0.75)', module_text)

    def test_c3d4_limit_json_writes_sanitize_path_objects(self):
        tree = _module_tree()
        helper_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("_json_safe_value", helper_names)

        scan_writer = _function_def(tree, "write_c3d4_limit_scan")
        dump_calls = [
            node
            for node in ast.walk(scan_writer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "dump"
        ]
        self.assertTrue(dump_calls)
        unsafe_dump_calls = [
            node
            for node in dump_calls
            if not (
                node.args
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == "_json_safe_value"
            )
        ]
        self.assertFalse(unsafe_dump_calls)

    def test_overlay_marks_sm_with_red_star_label_and_larger_axes(self):
        module_text = MODULE_PATH.read_text()
        self.assertIn("def _plot_sm_marker", module_text)
        self.assertIn('marker="*"', module_text)
        self.assertIn('color="red"', module_text)
        self.assertIn('"SM"', module_text)
        self.assertIn("DEFAULT_C3D4_OVERLAY_AXIS_LABEL_FONTSIZE = 20", module_text)
        self.assertIn("fontsize=DEFAULT_C3D4_OVERLAY_AXIS_LABEL_FONTSIZE", module_text)
        self.assertNotIn('marker="o", color="white", markeredgecolor="black", markersize=5', module_text)

    def test_hhhh_over_hhh_ratio_contour_plot_is_wired(self):
        module_text = MODULE_PATH.read_text()
        analyzer_text = (REPO_DIR / "4h_analyzer.py").read_text()
        self.assertIn("DEFAULT_HHH_XSEC_SOURCE_DIR", module_text)
        self.assertIn("/mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_hhh_c3d4", module_text)
        self.assertIn("c3d4_hhhh_over_hhh_ratio_contours.png", module_text)
        self.assertIn("hhhh_over_hhh_ratio_contours_plot", module_text)
        self.assertIn("--hhh-xsec-source-dir", analyzer_text)

        tree = _module_tree()
        writer = _function_def(tree, "_write_hhhh_over_hhh_ratio_contour_plot")
        called_names = {
            node.func.id
            for node in ast.walk(writer)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_read_hhhh_xsec_points", called_names)
        self.assertIn("_read_hhh_xsec_points", called_names)

        called_attrs = {
            node.func.attr
            for node in ast.walk(writer)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("contour", called_attrs)
        self.assertNotIn("contourf", called_attrs)

        string_constants = [
            node.value
            for node in ast.walk(writer)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertIn("black", string_constants)
        self.assertIn(r"$\sigma(hhhh)/\sigma(hhh)$", string_constants)

    def test_limit_scan_passes_hhh_source_to_ratio_contour_plot(self):
        tree = _module_tree()
        scan_writer = _function_def(tree, "write_c3d4_limit_scan")
        argument_names = [arg.arg for arg in scan_writer.args.args]
        self.assertIn("hhh_xsec_source_dir", argument_names)

        ratio_calls = [
            node
            for node in ast.walk(scan_writer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_write_hhhh_over_hhh_ratio_contour_plot"
        ]
        self.assertTrue(ratio_calls)
        keyword_names = {keyword.arg for call in ratio_calls for keyword in call.keywords}
        self.assertIn("hhh_source_dir", keyword_names)


if __name__ == "__main__":
    unittest.main()
