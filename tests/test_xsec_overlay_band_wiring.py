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


if __name__ == "__main__":
    unittest.main()
