import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "Code"))

from ForcedSplitting.validation_hhhbb import (  # noqa: E402
    HHHBB_DIRECT_LABEL,
    HHHBB_OBSERVABLE_ORDER,
    HHHBB_REPORT_TITLE,
    HHHBB_SHAPE_MAX_BINS,
    HHHBB_SPLIT_LABEL,
    HHHBBValidationRunConfig,
    extract_lhe_8b_sample,
    run_hhhbb_validation_chain,
    weight_check_report_lines,
    write_lhe_8b_validation_report,
)
from sample_report import observable_axis_label  # noqa: E402


def _row(pid, status, mother1, mother2, px, py, pz, energy, mass=0.0):
    return (
        "%10d %2d %4d %4d %4d %4d "
        "% .10e % .10e % .10e % .10e % .10e  0.0  9.0"
        % (pid, status, mother1, mother2, 0, 0, px, py, pz, energy, mass)
    )


ASSOCIATED_PAIR = [
    (5, 25.0, 0.0, 0.0, 25.0),
    (-5, -25.0, 0.0, 0.0, 25.0),
]

HIGGS_DECAYS = [
    [
        (5, 70.0, 0.0, 0.0, 70.0),
        (-5, -50.0, 0.0, 0.0, 50.0),
    ],
    [
        (5, 0.0, 80.0, 0.0, 80.0),
        (-5, 0.0, -50.0, 0.0, 50.0),
    ],
    [
        (5, 60.0, 60.0, 0.0, math.hypot(60.0, 60.0)),
        (-5, -40.0, -20.0, 0.0, math.hypot(40.0, 20.0)),
    ],
]


def _combined(rows):
    px = sum(row[1] for row in rows)
    py = sum(row[2] for row in rows)
    pz = sum(row[3] for row in rows)
    energy = sum(row[4] for row in rows)
    mass2 = energy * energy - px * px - py * py - pz * pz
    return px, py, pz, energy, math.sqrt(max(0.0, mass2))


def _lhe_document(events, xsec_pb=7.0):
    return """<LesHouchesEvents version="1.0">
<init>
     2212     2212           7000           7000  999  999  999  999    0    1
   {xsec:.9e}   5.000000000e-01            100      1
</init>
{events}</LesHouchesEvents>
""".format(xsec=xsec_pb, events="".join(events))


def source_hhhbb_lhe_text(weight=2.0, xsec_pb=7.0):
    rows = [
        _row(21, -1, 0, 0, 0.0, 0.0, 500.0, 500.0),
        _row(21, -1, 0, 0, 0.0, 0.0, -500.0, 500.0),
    ]
    for decay in HIGGS_DECAYS:
        px, py, pz, energy, mass = _combined(decay)
        rows.append(_row(25, 1, 1, 2, px, py, pz, energy, mass))
    for pid, px, py, pz, energy in ASSOCIATED_PAIR:
        rows.append(_row(pid, 1, 1, 2, px, py, pz, energy))
    return _lhe_document(
        [
            """<event>
    {nup:d}      1     {weight:.9e}        200.0    0.007546771     0.09944864
{rows}
</event>
""".format(nup=len(rows), weight=weight, rows="\n".join(rows))
        ],
        xsec_pb=xsec_pb,
    )


def final_8b_lhe_text(weight=2.0, xsec_pb=7.0, include_higgs_mothers=True, include_bad_event=False):
    rows = [
        _row(21, -1, 0, 0, 0.0, 0.0, 500.0, 500.0),
        _row(21, -1, 0, 0, 0.0, 0.0, -500.0, 500.0),
    ]
    higgs_indices = []
    if include_higgs_mothers:
        for decay in HIGGS_DECAYS:
            px, py, pz, energy, mass = _combined(decay)
            rows.append(_row(25, 2, 1, 2, px, py, pz, energy, mass))
            higgs_indices.append(len(rows))
    for pid, px, py, pz, energy in ASSOCIATED_PAIR:
        rows.append(_row(pid, 1, 1, 2, px, py, pz, energy))
    for decay_index, decay in enumerate(HIGGS_DECAYS):
        mother = higgs_indices[decay_index] if include_higgs_mothers else 1
        for pid, px, py, pz, energy in decay:
            rows.append(_row(pid, 1, mother, mother, px, py, pz, energy))
    good_event = """<event>
    {nup:d}      1     {weight:.9e}        200.0    0.007546771     0.09944864
{rows}
</event>
""".format(nup=len(rows), weight=weight, rows="\n".join(rows))
    bad_event = """<event>
    3      1     1.000000000e+00        200.0    0.007546771     0.09944864
%s
%s
%s
</event>
""" % (
        _row(21, -1, 0, 0, 0.0, 0.0, 100.0, 100.0),
        _row(21, -1, 0, 0, 0.0, 0.0, -100.0, 100.0),
        _row(5, 1, 1, 2, 10.0, 0.0, 0.0, 10.0),
    )
    return _lhe_document([good_event] + ([bad_event] if include_bad_event else []), xsec_pb=xsec_pb)


def split_input_lhe_text(weight=1.0, xsec_pb=3.0):
    rows = [
        _row(21, -1, 0, 0, 0.0, 0.0, 500.0, 500.0),
        _row(21, -1, 0, 0, 0.0, 0.0, -500.0, 500.0),
        _row(25, 1, 1, 2, 20.0, 0.0, 0.0, 120.0, 118.3215957),
        _row(25, 1, 1, 2, 0.0, 30.0, 0.0, 130.0, 126.4911064),
        _row(25, 1, 1, 2, 20.0, 40.0, 0.0, 129.5739561, 121.6223260),
        _row(21, 1, 1, 2, -40.0, -70.0, 0.0, 80.6225775),
    ]
    return _lhe_document(
        [
            """<event>
    {nup:d}      1     {weight:.9e}        200.0    0.007546771     0.09944864
{rows}
</event>
""".format(nup=len(rows), weight=weight, rows="\n".join(rows))
        ],
        xsec_pb=xsec_pb,
    )


class HHHBBValidationTests(unittest.TestCase):
    def test_extract_lhe_8b_sample_builds_weighted_pair_observables(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            final_path = tmpdir / "final8b.lhe"
            source_path = tmpdir / "source.lhe"
            final_path.write_text(final_8b_lhe_text(weight=2.5, xsec_pb=9.0, include_bad_event=True))
            source_path.write_text(source_hhhbb_lhe_text(weight=2.5, xsec_pb=9.0))

            sample = extract_lhe_8b_sample(final_path, label="gg_hhhbb_direct", source_lhe=source_path)

        self.assertEqual(sample["summary"]["event_count"], 2)
        self.assertEqual(sample["summary"]["accepted_8b_events"], 1)
        self.assertEqual(sample["summary"]["skipped_events"], 1)
        self.assertTrue(math.isclose(sample["summary"]["weighted_event_sum"], 2.5))
        self.assertEqual(sorted(sample["observables"]), sorted(HHHBB_OBSERVABLE_ORDER))
        self.assertEqual(len(sample["observables"]["b_pt_all"]["values"]), 8)
        self.assertEqual(len(sample["observables"]["b8_pt"]["values"]), 1)
        self.assertEqual(len(sample["observables"]["dr_bb_all"]["values"]), 28)
        self.assertEqual(len(sample["observables"]["m_bb_all"]["values"]), 28)
        self.assertEqual(len(sample["observables"]["dr_associated_bb"]["values"]), 1)
        self.assertEqual(len(sample["observables"]["dr_higgs_bb"]["values"]), 3)
        self.assertEqual(len(sample["observables"]["dr_associated_higgs_cross_bb"]["values"]), 12)
        self.assertEqual(len(sample["observables"]["dr_inter_higgs_cross_bb"]["values"]), 12)
        self.assertEqual(len(sample["observables"]["m_8b"]["values"]), 1)
        self.assertEqual(len(sample["observables"]["ht_b"]["values"]), 1)
        self.assertEqual(sample["summary"]["pair_classification"]["associated_source_match"], 1)
        self.assertEqual(sample["summary"]["pair_classification"]["higgs_ancestry_pairs"], 3)

    def test_extract_lhe_8b_sample_uses_source_higgs_fallback_without_mothers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            final_path = tmpdir / "final8b_no_mothers.lhe"
            source_path = tmpdir / "source.lhe"
            final_path.write_text(final_8b_lhe_text(weight=3.0, include_higgs_mothers=False))
            source_path.write_text(source_hhhbb_lhe_text(weight=3.0))

            sample = extract_lhe_8b_sample(final_path, label="gg_hhhg_forced_split", source_lhe=source_path)

        self.assertEqual(len(sample["observables"]["dr_higgs_bb"]["values"]), 3)
        self.assertEqual(len(sample["observables"]["m_higgs_bb"]["values"]), 3)
        self.assertEqual(sample["summary"]["pair_classification"]["source_higgs_match_pairs"], 3)
        self.assertEqual(sample["summary"]["pair_classification"]["higgs_mass_fallback_pairs"], 0)

    def test_write_lhe_8b_validation_report_uses_sample_report_webpage_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            split_lhe = tmpdir / "split.final8b.lhe"
            direct_lhe = tmpdir / "direct.final8b.lhe"
            split_source = tmpdir / "split.source.lhe"
            direct_source = tmpdir / "direct.source.lhe"
            split_lhe.write_text(final_8b_lhe_text(weight=1.5, xsec_pb=3.0))
            direct_lhe.write_text(final_8b_lhe_text(weight=2.0, xsec_pb=5.0))
            split_source.write_text(source_hhhbb_lhe_text(weight=1.5, xsec_pb=3.0))
            direct_source.write_text(source_hhhbb_lhe_text(weight=2.0, xsec_pb=5.0))

            metadata = write_lhe_8b_validation_report(
                split_lhe=split_lhe,
                direct_lhe=direct_lhe,
                output_dir=tmpdir / "report",
                split_source_lhe=split_source,
                direct_source_lhe=direct_source,
            )

            index = Path(metadata["index"])
            table = Path(metadata["table"])
            metadata_path = Path(metadata["metadata"])

            self.assertTrue(index.exists())
            self.assertTrue(table.exists())
            self.assertTrue(metadata_path.exists())
            self.assertIn(HHHBB_REPORT_TITLE, index.read_text())
            self.assertIn("class=\"grid\"", index.read_text())
            self.assertIn(HHHBB_SPLIT_LABEL, table.read_text())
            self.assertIn(HHHBB_DIRECT_LABEL, table.read_text())
            self.assertIn("associated_pair_deltaR_bb", index.read_text())
            self.assertIn("higgs_decay_deltaR_bb", index.read_text())
            self.assertIn("m_8b", index.read_text())
            self.assertIn("ht_b", index.read_text())
            self.assertTrue(all(Path(row["path"]).exists() for row in metadata["plots"]))
            self.assertTrue(json.loads(metadata_path.read_text())["validation_only"])
            self.assertEqual(
                json.loads(metadata_path.read_text())["normalisation"]["shape_plot_binning"]["max_bins"],
                HHHBB_SHAPE_MAX_BINS,
            )

    def test_hhhbb_validation_axis_labels_are_latex(self):
        self.assertEqual(observable_axis_label("b8_pt"), r"$p_T(b_8)$ [GeV]")
        self.assertEqual(observable_axis_label("dr_associated_bb"), r"$\Delta R(b,b)_{\mathrm{assoc}}$")
        self.assertEqual(observable_axis_label("m_higgs_bb"), r"$m(b,b)_{\mathrm{same}\ h}$ [GeV]")
        self.assertEqual(observable_axis_label("m_8b"), r"$m(8b)$ [GeV]")
        self.assertEqual(observable_axis_label("ht_b"), r"$H_T(b)$ [GeV]")

    def test_hhhbb_validation_chain_dry_run_writes_baseline_cards_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            split_input = tmpdir / "gg_hhhg.lhe"
            direct_input = tmpdir / "gg_hhhbb.lhe"
            split_input.write_text(split_input_lhe_text(weight=1.0, xsec_pb=1.0))
            direct_input.write_text(source_hhhbb_lhe_text(weight=1.0, xsec_pb=2.0))

            summary = run_hhhbb_validation_chain(
                HHHBBValidationRunConfig(
                    split_input_lhe=split_input,
                    direct_input_lhe=direct_input,
                    workdir=tmpdir / "validation",
                    events=1,
                    probe_trials=11,
                    pdf_name="CT10nlo_as_0119",
                    dry_run=True,
                )
            )

            stage1_card = Path(summary["split_stage1_card"])
            split_decay_card = Path(summary["split_decay_card"])
            direct_decay_card = Path(summary["direct_decay_card"])

            self.assertTrue(stage1_card.exists())
            self.assertTrue(split_decay_card.exists())
            self.assertTrue(direct_decay_card.exists())
            self.assertIn("gg -> hhh + g with one forced final-state g -> b bbar split", stage1_card.read_text())
            self.assertIn("set ForceSplitVeto:MinB 2", stage1_card.read_text())
            self.assertIn("set ForceSplitVeto:ProbeTrials 11", stage1_card.read_text())
            self.assertIn("set /Herwig/Partons/thePDFset:PDFName CT10nlo_as_0119", stage1_card.read_text())
            self.assertIn("set /Herwig/Partons/thePDFset:PDFName CT10nlo_as_0119", split_decay_card.read_text())
            self.assertIn("set /Herwig/Partons/thePDFset:PDFName CT10nlo_as_0119", direct_decay_card.read_text())
            self.assertIn("decaymode h0->b,bbar; 1.0 1 /Herwig/Decays/Hff", split_decay_card.read_text())
            self.assertIn("decaymode h0->b,bbar; 1.0 1 /Herwig/Decays/Hff", direct_decay_card.read_text())
            self.assertEqual(len(summary["commands"]), 6)
            self.assertTrue(Path(summary["summary"]).exists())

    def test_weight_check_report_lines_print_unsuccessful_rows(self):
        lines = weight_check_report_lines(
            {
                "weight_check": {
                    "correction_rows": 100,
                    "zero_success_rows": 2,
                    "nonzero_weight_rows": 98,
                    "mean_p_hat": 0.025,
                }
            }
        )

        self.assertIn("total unsuccessful rows: 2", lines)
        self.assertIn(
            "  run_1: unsuccessful rows=2/100 nonzero_weight_rows=98 mean_p_hat=0.025",
            lines,
        )


if __name__ == "__main__":
    unittest.main()
