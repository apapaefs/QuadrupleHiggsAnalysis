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

    def test_c3d4_plot_defaults_use_requested_viewport(self):
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


if __name__ == "__main__":
    unittest.main()
