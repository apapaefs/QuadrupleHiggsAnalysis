import tempfile
import unittest
import json
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_DIR / "4h_analyzer.py"
LOCAL_CLI_MARKER = '\n\nif __name__ == "__main__" and "--legacy" not in _sys.argv:'


def _load_local_cli_namespace():
    source = MODULE_PATH.read_text().split(LOCAL_CLI_MARKER, 1)[0]
    namespace = {
        "__file__": str(MODULE_PATH),
        "__name__": "test_4h_analyzer_local_cli",
    }
    exec(compile(source, str(MODULE_PATH), "exec"), namespace)
    return namespace


class SignalXsecMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_cli = _load_local_cli_namespace()

    def _signal_root_and_output(self, directory):
        root_file = (
            directory
            / "HerwigSignalPoints"
            / "c3d4_10k"
            / "events"
            / "HW-run_gg_4h_4_0.0_0.0_var.smearCMS.root"
        )
        root_file.parent.mkdir(parents=True)
        root_file.touch()
        return root_file, root_file.parent.parent / "HW-run_gg_4h_4_0.0_0.0.out"

    def test_missing_signal_xsec_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_file, expected_output = self._signal_root_and_output(Path(temporary_directory))

            with self.assertRaisesRegex(SystemExit, "Could not infer c3/d4 signal cross section") as raised:
                self.local_cli["_infer_scored_signal_metadata"](
                    [root_file],
                    xsec_values=None,
                    generated_values=None,
                    default_generated_events=10000,
                    label="c3/d4 signal",
                    xsec_option="--c3d4-signal-xsec-fb",
                )

            self.assertIn(str(expected_output), str(raised.exception))
            self.assertIn("--c3d4-signal-xsec-fb", str(raised.exception))

    def test_herwig_output_metadata_is_used_for_signal_xsec(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_file, output_file = self._signal_root_and_output(Path(temporary_directory))
            output_file.write_text("Total: 10000 0 1.0e-08(1)\n")

            xsecs, generated_events, normalisation_weights = self.local_cli[
                "_infer_scored_signal_metadata"
            ](
                [root_file],
                xsec_values=None,
                generated_values=None,
                default_generated_events=10000,
                label="c3/d4 signal",
                xsec_option="--c3d4-signal-xsec-fb",
            )

            self.assertEqual(xsecs, [0.01])
            self.assertEqual(generated_events, [10000])
            self.assertEqual(normalisation_weights, [None])

    def test_explicit_signal_xsec_does_not_require_output_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_file, _ = self._signal_root_and_output(Path(temporary_directory))

            xsecs, generated_events, _ = self.local_cli["_infer_scored_signal_metadata"](
                [root_file],
                xsec_values=[2.5],
                generated_values=None,
                default_generated_events=10000,
                label="c3/d4 signal",
                xsec_option="--c3d4-signal-xsec-fb",
            )

            self.assertEqual(xsecs, [2.5])
            self.assertEqual(generated_events, [10000])

    def test_normalisation_weight_prefers_analysis_summary_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_file, _ = self._signal_root_and_output(Path(temporary_directory))
            summary_file = root_file.with_name("HW-run_gg_4h_4_0.0_0.0.analysis_summary.json")
            summary_file.write_text(json.dumps({"total_weight_in": 123.5}))

            normalisation = self.local_cli["_normalisation_weight_for_var_root"](root_file)

            self.assertEqual(normalisation, 123.5)

    def test_standalone_signal_scoring_uses_full_rate_factor(self):
        analyzer_text = MODULE_PATH.read_text()

        self.assertIn("signal_rate_factor = _signal_final_rate_factor_for_cli(args)", analyzer_text)
        self.assertIn("signal_rate_factors=signal_rate_factor", analyzer_text)

    def test_c3d4_limit_uses_held_out_background_yield(self):
        analyzer_text = MODULE_PATH.read_text()

        self.assertIn('background_events = float(best_threshold.get("background_events", 0.0))', analyzer_text)


if __name__ == "__main__":
    unittest.main()
