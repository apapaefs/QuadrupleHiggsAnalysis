from __future__ import annotations

import importlib.util
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "resonance_xgboost_analysis.py"
SPEC = importlib.util.spec_from_file_location("resonance_xgboost_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)

PREPARE_MODULE_PATH = Path(__file__).resolve().parents[1] / "prepare_resonance_features.py"
PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_resonance_features", PREPARE_MODULE_PATH
)
assert PREPARE_SPEC and PREPARE_SPEC.loader
prepare = importlib.util.module_from_spec(PREPARE_SPEC)
sys.modules[PREPARE_SPEC.name] = prepare
PREPARE_SPEC.loader.exec_module(prepare)


def synthetic_table(nmerged: np.ndarray | None = None) -> analysis.EventTable:
    nmerged = np.asarray([0, 1, 2, 3, 4] if nmerged is None else nmerged, dtype=int)
    count = len(nmerged)
    category = np.where(nmerged == 0, 0, np.where(nmerged <= 2, 1, 2))
    arrays: dict[str, np.ndarray] = {}
    for name in analysis.SCALAR_BRANCHES:
        arrays[name] = np.ones(count, dtype=float)
    arrays.update(
        event_index=np.arange(count, dtype=np.int64),
        weight=np.ones(count),
        n_merged=nmerged,
        n_double_b=nmerged.copy(),
        n_true_single=8 - 2 * nmerged,
        n_c_mistag=np.zeros(count, dtype=int),
        n_light_mistag=np.zeros(count, dtype=int),
        category=category,
        second_score=np.full(count, -1.0),
        score_gap=np.full(count, -1.0),
        m4h=np.full(count, 1000.0),
    )
    for name, width in analysis.ARRAY_BRANCH_WIDTHS.items():
        arrays[name] = np.ones((count, width), dtype=float)
    arrays["pair_mass"] = np.tile(np.asarray([300.0, 420.0, 510.0, 490.0, 380.0, 310.0]), (count, 1))
    arrays["jet_pt"] = np.tile(np.arange(8, 0, -1, dtype=float), (count, 1))
    return analysis.EventTable(
        arrays=arrays,
        input_events=count,
        input_sumw=float(count),
        input_sumw2=float(count),
        reconstructable_events=count,
        reconstructable_sumw=float(count),
        summary={"tag_efficiencies_applied": False},
    )


def synthetic_loaded_sample(
    sample_id: str,
    role: str,
    *,
    point: analysis.MassPoint | None = None,
    nmerged: np.ndarray | None = None,
    luminosity: float = analysis.LUMINOSITY_FB,
    hbb_branching_ratio: float = analysis.HBB_BRANCHING_RATIO,
    hbb_power: int = 4,
    eps_b: float = analysis.EPS_B,
    eps_c: float = analysis.EPS_C,
    eps_light: float = analysis.EPS_LIGHT,
) -> analysis.LoadedSample:
    table = synthetic_table(nmerged)
    spec = analysis.SampleSpec(
        sample_id=sample_id,
        role=role,
        root_file=Path(f"{sample_id}.root"),
        summary_file=Path(f"{sample_id}.analysis_summary.json"),
        generated_events_expected=table.input_events,
        cross_section_fb=1.0 if role == "signal" else 2.0,
        generated_cross_section_fb=1.0 if role == "signal" else 2.0,
        cross_section_source="synthetic",
        k_factor=1.0 if role == "signal" else 2.0,
        hbb_power=hbb_power,
        rate_factor=1.0,
        c_mistags=0,
        light_mistags=0,
        lhe_event_count=table.input_events,
        hard_event_policy="unique",
        point=point,
    )
    base, names = analysis.base_feature_matrix(table)
    folds = np.arange(table.entries, dtype=int) % analysis.N_FOLDS
    scenarios = {
        "nominal": eps_b**2,
        "conservative": analysis.EPS_BB_CONSERVATIVE,
    }
    weights = {
        scenario: analysis.physical_event_weights(
            spec,
            table,
            luminosity,
            hbb_branching_ratio,
            eps_b,
            eps_bb,
            eps_c,
            eps_light,
        )
        for scenario, eps_bb in scenarios.items()
    }
    return analysis.LoadedSample(spec, table, folds, base, names, weights)


class MassPointTests(unittest.TestCase):
    def test_manifest_and_filename_parsing(self) -> None:
        direct = analysis.parse_mass_point("direct", {"miota_GeV": "1250"})
        cascade = analysis.parse_mass_point(
            "cascade", filename="HW-miota_1600-meta_0550_resonance.root"
        )
        self.assertEqual(direct.point_id, "MS_1250")
        self.assertEqual(cascade.point_id, "M2_0550_M3_1600")

    def test_hierarchy_is_enforced(self) -> None:
        with self.assertRaises(analysis.AnalysisInputError):
            analysis.MassPoint("cascade", m2=400.0, m3=800.0)

    def test_signal_manifest_rejects_hard_event_recycling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text(
                "scenario,run_name,miota_GeV,events,lhe_event_count,hard_event_policy\n"
                "direct,signal_1000,1000,11,10,unique\n",
                encoding="utf-8",
            )
            with self.assertRaises(analysis.AnalysisInputError):
                analysis.load_signal_specs(
                    manifest,
                    "direct",
                    Path(directory),
                    "{run_name}.root",
                )


class BackgroundManifestTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(
        directory: Path,
        hhhbb_power: int = 3,
    ) -> Path:
        rows = [
            {
                "sample_id": "sm_hhhh",
                "role": "sm_hhhh",
                "root_file": "sm_hhhh.root",
                "cross_section_fb": "0.001",
                "generated_events": "10",
                "hbb_power": "4",
            },
            {
                "sample_id": "sm_hhhbb",
                "role": "sm_hhhbb",
                "root_file": "sm_hhhbb.root",
                "cross_section_fb": "0.002",
                "generated_events": "10",
                "hbb_power": str(hhhbb_power),
            },
            {
                "sample_id": "sm_hh4b",
                "role": "sm_hh4b",
                "root_file": "sm_hh4b.root",
                "cross_section_fb": "0.003",
                "generated_events": "10",
                "hbb_power": "2",
            },
        ]
        for row in rows:
            (directory / row["root_file"]).touch()
        manifest = directory / "backgrounds.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_sm_multihiggs_roles_use_their_physical_hbb_powers(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            specs, missing = analysis.load_background_specs(
                self._write_manifest(directory),
                directory,
                default_k_factor=2.0,
            )
            self.assertEqual(missing, [])
            self.assertEqual(
                {spec.role: spec.hbb_power for spec in specs},
                analysis.SM_BACKGROUND_HBB_POWERS,
            )
            analysis.require_full_sm_background_roles(specs)
            with self.assertRaises(analysis.AnalysisInputError):
                analysis.require_full_sm_background_roles(specs[:-1])

    def test_sm_hhhbb_rejects_four_higgs_branching_power(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with self.assertRaisesRegex(
                analysis.AnalysisInputError,
                "sm_hhhbb must use hbb_power=3",
            ):
                analysis.load_background_specs(
                    self._write_manifest(directory, hhhbb_power=4),
                    directory,
                    default_k_factor=2.0,
                )

class FeatureCampaignContractTests(unittest.TestCase):
    def _summary(self, sample_id: str, input_path: Path, method_version: str) -> dict:
        seed = prepare._seed(sample_id)
        counter = {"events": 1, "sumw": 1.0, "sumw2": 1.0}
        return {
            "schema": "resonance-hybrid-v1",
            "method_version": method_version,
            "preprocessing_version": prepare.PREPROCESSING_VERSION,
            "higgs_mass_targets_gev": list(
                prepare.DEFAULT_HIGGS_MASS_TARGETS_GEV
            ),
            "higgs_mass_target_assignment": "candidate_pt_rank_descending",
            "input": str(input_path.resolve()),
            "events_available": 1,
            "events_requested": 1,
            "c_mistags": 0,
            "light_mistags": 0,
            "max_reco_true_bjets": 10,
            "tag_efficiencies_applied": False,
            "smearing": {
                "enabled": True,
                "seed": seed,
                "preprocessing_version": prepare.PREPROCESSING_VERSION,
                "model_id": prepare.SMEARING_MODEL_ID,
                "fourvector_scaling": "uniform_correlated",
                "correlated_mass_scaling": True,
                "preserves_jet_mass": False,
                "gaussian_draws_per_jet": 1,
                "energy_floor_gev": 1.0e-6,
                "eta_preselection": "finite |eta|<2.5 before smearing",
                "pt_threshold": "smeared pT>20 GeV",
                "smear_before_pt_threshold": True,
                "acceptance_order": (
                    "raw_abs_eta_then_smear_then_smeared_pt"
                ),
            },
            "input_counter": counter,
            "reconstructable_counter": counter,
            "categories": {"resolved": counter},
            "n_merged": {"0": counter},
            "diagnostics": {
                "true_b_upward_pt_migrations": 0,
                "true_b_downward_pt_migrations": 0,
                "non_b_upward_pt_migrations": 0,
                "non_b_downward_pt_migrations": 0,
                "true_b_upward_pt_migrations_by_raw_pt_gev": {
                    "[10,12)": 0,
                    "[12,15)": 0,
                    "[15,20]": 0,
                },
                "non_b_upward_pt_migrations_by_raw_pt_gev": {
                    "[10,12)": 0,
                    "[12,15)": 0,
                    "[15,20]": 0,
                },
                "max_smearing_mass_scaling_residual_gev": 0.0,
            },
        }

    def test_feature_validator_accepts_only_uniform_fourvector_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_id = "sample"
            input_path = root / "input.root"
            output = root / "features.root"
            summary_path = root / "features.analysis_summary.json"
            input_path.write_bytes(b"input")
            output.write_bytes(b"features")

            legacy = self._summary(
                sample_id, input_path, "resonance-hybrid-v1.1-leading-composition"
            )
            legacy["preprocessing_version"] = "resonance-preprocessing-v1"
            legacy["smearing"] = {
                "enabled": True,
                "seed": prepare._seed(sample_id),
                "preserves_jet_mass": True,
            }
            summary_path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match this campaign"):
                prepare._validate_feature_pair(
                    sample_id,
                    input_path,
                    output,
                    summary_path,
                    0,
                    0,
                    None,
                    10,
                    False,
                )

            current = self._summary(sample_id, input_path, prepare.METHOD_VERSION)
            summary_path.write_text(json.dumps(current), encoding="utf-8")
            accepted = prepare._validate_feature_pair(
                sample_id,
                input_path,
                output,
                summary_path,
                0,
                0,
                None,
                10,
                False,
            )
            self.assertEqual(accepted["method_version"], prepare.METHOD_VERSION)

    def test_feature_validator_rejects_incomplete_upward_migration_bins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_id = "sample"
            input_path = root / "input.root"
            output = root / "features.root"
            summary_path = root / "features.analysis_summary.json"
            input_path.write_bytes(b"input")
            output.write_bytes(b"features")
            summary = self._summary(sample_id, input_path, prepare.METHOD_VERSION)
            summary["diagnostics"]["true_b_upward_pt_migrations"] = 2
            summary["diagnostics"][
                "true_b_upward_pt_migrations_by_raw_pt_gev"
            ]["[15,20]"] = 1
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "upward_migration_pt_bin_sums"
            ):
                prepare._validate_feature_pair(
                    sample_id,
                    input_path,
                    output,
                    summary_path,
                    0,
                    0,
                    None,
                    10,
                    False,
                )

    def test_feature_job_uses_shared_baseline_in_separate_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.root"
            input_path.touch()
            record = prepare._run_job(
                {
                    "id": "sample",
                    "kind": "signal",
                    "input": input_path,
                    "output": root / "features" / "sample.root",
                    "c_mistags": 0,
                    "light_mistags": 0,
                },
                root / "extractor",
                root / "logs",
                None,
                10,
                False,
                True,
            )
            option = record["command"].index("--higgs-mass-targets")
            self.assertEqual(record["command"][option + 1], "120,115,110,105")
            self.assertEqual(
                prepare.DEFAULT_FEATURE_BASE.name,
                prepare.MASS_TARGET_PROFILE_ID,
            )

    def test_xgboost_summary_accepts_current_extractor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.analysis_summary.json"
            summary = self._summary(
                "sample", Path(directory) / "input.root", prepare.METHOD_VERSION
            )
            path.write_text(json.dumps(summary), encoding="utf-8")

            metadata = analysis._summary_metadata(path)
            self.assertEqual(metadata[-1]["method_version"], prepare.METHOD_VERSION)
            self.assertEqual(
                analysis.EXTRACTOR_PREPROCESSING_VERSION,
                prepare.PREPROCESSING_VERSION,
            )
            self.assertEqual(
                analysis.EXTRACTOR_SMEARING_MODEL_ID, prepare.SMEARING_MODEL_ID
            )

    def test_xgboost_summary_rejects_incompatible_extractor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.analysis_summary.json"
            current = self._summary(
                "sample", Path(directory) / "input.root", prepare.METHOD_VERSION
            )
            cases = {
                "method_version": (
                    lambda summary: summary.__setitem__(
                        "method_version", "resonance-hybrid-v1.1-leading-composition"
                    ),
                    "expected extractor method_version",
                ),
                "preprocessing_version": (
                    lambda summary: summary.__setitem__(
                        "preprocessing_version", "resonance-preprocessing-v1"
                    ),
                    "expected preprocessing_version",
                ),
                "higgs_mass_targets": (
                    lambda summary: summary.__setitem__(
                        "higgs_mass_targets_gev",
                        [125.0, 125.0, 125.0, 125.0],
                    ),
                    "expected Higgs mass targets",
                ),
                "smearing_preprocessing_version": (
                    lambda summary: summary["smearing"].__setitem__(
                        "preprocessing_version", "resonance-preprocessing-v1"
                    ),
                    "incompatible smearing metadata: preprocessing_version",
                ),
                "smearing_model_id": (
                    lambda summary: summary["smearing"].__setitem__(
                        "model_id", "legacy-fixed-mass-smearing"
                    ),
                    "incompatible smearing metadata: model_id",
                ),
                "acceptance_order": (
                    lambda summary: summary["smearing"].__setitem__(
                        "acceptance_order", "raw_pt_then_smear"
                    ),
                    "incompatible smearing metadata: acceptance_order",
                ),
            }
            for label, (mutate, message) in cases.items():
                with self.subTest(label=label):
                    summary = json.loads(json.dumps(current))
                    mutate(summary)
                    path.write_text(json.dumps(summary), encoding="utf-8")
                    with self.assertRaisesRegex(
                        analysis.AnalysisInputError, message
                    ):
                        analysis._summary_metadata(path)

    def test_cpp_smearing_contract_has_one_draw_and_uniform_scaling(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "FourHiggsResonanceAnalysis.cc"
        ).read_text()
        function = source.split("TLorentzVector smearJetCMSUniformFourVector", 1)[1].split(
            "void combinationsRecursive", 1
        )[0]
        self.assertEqual(function.count("random.Gaus("), 1)
        self.assertIn("std::max(1.0e-6, energy + delta_energy)", function)
        self.assertIn("const double scale = smeared_energy / energy", function)
        self.assertIn("!std::isfinite(energy) || energy <= 0.0", function)
        self.assertIn("output.M() - scale * input.M()", function)

        source_body = source.split("for (event_index = 0;", 1)[1]
        true_b_body = source_body.split("for (int index = 0; index < safe_number_bjets;", 1)[
            1
        ].split("std::sort(true_bjets.begin()", 1)[0]
        self.assertLess(
            true_b_body.index("std::fabs(raw.Eta()) >= kBJetEtaCut"),
            true_b_body.index("smearJetCMSUniformFourVector"),
        )
        self.assertLess(
            true_b_body.index("smearJetCMSUniformFourVector"),
            true_b_body.index("const bool smeared_passes_pt"),
        )

        non_b_body = source_body.split("for (int index = 0; index < safe_number_jets;", 1)[
            1
        ].split("const auto by_pt", 1)[0]
        self.assertLess(
            non_b_body.index("std::fabs(raw.Eta()) >= kBJetEtaCut"),
            non_b_body.index("smearJetCMSUniformFourVector"),
        )
        self.assertLess(
            non_b_body.index("smearJetCMSUniformFourVector"),
            non_b_body.index("const bool smeared_passes_pt"),
        )

    def test_explicit_common_target_preserves_historical_scoring_order(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "FourHiggsResonanceAnalysis.cc"
        ).read_text()
        self.assertIn("{{120.0, 115.0, 110.0, 105.0}}", source)
        consider = source.split("  void consider(", 1)[1].split(
            "\n};\n\nusing IndexCallback", 1
        )[0]
        self.assertIn(
            "const bool common_target = massTargetsAreCommon(mass_targets)",
            consider,
        )
        self.assertIn("if (!common_target)", consider)
        self.assertIn("common_target ? proposal : ranked", consider)
        self.assertIn("candidates = scoring_candidates", consider)

        event_loop = source.split("for (event_index = 0;", 1)[1]
        self.assertIn(
            "if (massTargetsAreCommon(options.higgs_mass_targets))",
            event_loop,
        )
        historical_sort = event_loop.split(
            "if (massTargetsAreCommon(options.higgs_mass_targets))", 1
        )[1].split("n_merged = 0;", 1)[0]
        self.assertIn("return first.p4.Pt() > second.p4.Pt();", historical_sort)
        self.assertNotIn("first.type", historical_sort)

    def test_versioned_background_manifest_redirects_feature_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "background_manifest.csv"
            source.write_text(
                "sample_id,root_file,raw_root,c_mistags,light_mistags\n"
                "bkg,legacy/bkg.root,raw/bkg.root,1,2\n",
                encoding="utf-8",
            )
            output_base = root / "features" / prepare.SMEARING_MODEL_ID / "backgrounds"
            text = prepare._versioned_background_manifest_text(source, root, output_base)
            rows = list(csv.DictReader(text.splitlines()))
            self.assertEqual(rows[0]["raw_root"], "raw/bkg.root")
            self.assertEqual(
                rows[0]["root_file"],
                f"features/{prepare.SMEARING_MODEL_ID}/backgrounds/bkg_resonance.root",
            )

class FeatureTests(unittest.TestCase):
    def test_truth_audit_branches_are_not_classifier_features(self) -> None:
        table = synthetic_table()
        _, names = analysis.base_feature_matrix(table)
        forbidden = {
            "raw_bjets",
            "accepted_bjets",
            "accepted_single_bjets",
            "accepted_merged_bjets",
            "accepted_cjet_candidates",
            "accepted_lightjet_candidates",
            "n_true_single",
            "n_double_b",
            "n_c_mistag",
            "n_light_mistag",
        }
        self.assertTrue(forbidden.isdisjoint(names))
        self.assertIn("n_merged", names)

    def test_direct_features_contain_mass_residual_and_six_pair_ratios(self) -> None:
        table = synthetic_table()
        base, names = analysis.base_feature_matrix(table)
        matrix, feature_names = analysis.engineer_features(
            base, names, table, "direct", ms=1000.0
        )
        self.assertEqual(matrix.shape[1], len(names) + 8)
        self.assertTrue(np.allclose(matrix[:, feature_names.index("m4h_minus_MS_over_MS")], 0.0))
        self.assertTrue(all(f"pair_mass_{label}_over_MS" in feature_names for label in analysis.PAIR_LABELS))

    def test_cascade_selects_best_of_three_pairings(self) -> None:
        table = synthetic_table(np.asarray([0]))
        base, names = analysis.base_feature_matrix(table)
        matrix, feature_names = analysis.engineer_features(
            base, names, table, "cascade", m2=300.0, m3=1000.0
        )
        # Pairing (12,34) has masses 300 and 310 and is the best pairing.
        self.assertAlmostEqual(matrix[0, feature_names.index("cascade_pairing_index")], 0.0)
        self.assertAlmostEqual(matrix[0, feature_names.index("cascade_eta1_residual")], 0.0)
        self.assertGreaterEqual(
            matrix[0, feature_names.index("cascade_second_score")],
            matrix[0, feature_names.index("cascade_best_score")],
        )


class FoldAndNormalizationTests(unittest.TestCase):
    def test_source_local_folds_are_balanced_and_order_independent(self) -> None:
        entries = np.arange(23)
        sources = np.full(23, "sample", dtype=object)
        folds = analysis.deterministic_folds(sources, entries)
        counts = np.bincount(folds, minlength=5)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        order = np.arange(22, -1, -1)
        reordered = analysis.deterministic_folds(sources[order], entries[order])
        restored = np.empty_like(reordered)
        restored[order] = reordered
        self.assertTrue(np.array_equal(folds, restored))

    def test_nominal_double_tag_factor_closes_to_eight_single_tags(self) -> None:
        table = synthetic_table()
        factors = analysis.tag_efficiency(
            table,
            analysis.EPS_B,
            analysis.EPS_B**2,
            analysis.EPS_C,
            analysis.EPS_LIGHT,
        )
        self.assertTrue(np.allclose(factors, analysis.EPS_B**8, rtol=0.0, atol=1e-14))

    def test_exact_one_fb_normalization_constants(self) -> None:
        produced = analysis.LUMINOSITY_FB * analysis.SIGNAL_HYPOTHESIS_FB
        eight_b = produced * analysis.HBB_BRANCHING_RATIO**4
        nominal = eight_b * analysis.EPS_B**8
        self.assertEqual(produced, 3000.0)
        self.assertTrue(math.isclose(eight_b, 345.1490798665728, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(nominal, 94.0498539896, rel_tol=0.0, abs_tol=1e-10))

    def test_sm_multihiggs_backgrounds_apply_only_physical_higgs_decays(self) -> None:
        hh4b = synthetic_loaded_sample("sm_hh4b", "sm_hh4b", hbb_power=2)
        hhhbb = synthetic_loaded_sample("sm_hhhbb", "sm_hhhbb", hbb_power=3)
        hhhh = synthetic_loaded_sample("sm_hhhh", "sm_hhhh", hbb_power=4)
        yields = [
            float(np.sum(sample.scenario_weights["nominal"]))
            for sample in (hh4b, hhhbb, hhhh)
        ]
        self.assertAlmostEqual(yields[1] / yields[0], analysis.HBB_BRANCHING_RATIO)
        self.assertAlmostEqual(yields[2] / yields[1], analysis.HBB_BRANCHING_RATIO)

    def test_charm_and_light_factors_are_applied_once(self) -> None:
        table = synthetic_table(np.asarray([0]))
        table.arrays["n_true_single"][:] = 4
        table.arrays["n_c_mistag"][:] = 2
        table.arrays["n_light_mistag"][:] = 2
        factor = analysis.tag_efficiency(table, 0.85, 0.85**2, 0.10, 0.01)[0]
        self.assertAlmostEqual(factor, 0.85**4 * 0.10**2 * 0.01**2)

    def test_per_event_mistag_composition_must_match_manifest(self) -> None:
        sample = synthetic_loaded_sample(
            "background", "sm_hhhh", point=None, nmerged=np.asarray([0])
        )
        sample.table.arrays["n_c_mistag"][:] = 1
        sample.table.arrays["n_true_single"][:] = 7
        with self.assertRaises(analysis.AnalysisInputError):
            analysis._validate_event_arrays(sample.spec, sample.table.arrays)

    def test_runtime_overrides_do_not_mutate_benchmark_closure(self) -> None:
        point = analysis.MassPoint("direct", ms=1000.0)
        signal = synthetic_loaded_sample(
            "signal",
            "signal",
            point=point,
            luminosity=100.0,
            hbb_branching_ratio=0.5,
            eps_b=0.8,
        )
        background = synthetic_loaded_sample(
            "sm_hhhh",
            "sm_hhhh",
            luminosity=100.0,
            hbb_branching_ratio=0.5,
            eps_b=0.8,
        )
        audit = analysis.normalization_audit(
            [signal],
            [background],
            100.0,
            0.5,
            0.8,
            analysis.EPS_C,
            analysis.EPS_LIGHT,
            {"nominal": 0.8**2, "conservative": analysis.EPS_BB_CONSERVATIVE},
        )
        self.assertTrue(audit["all_checks_pass"])
        self.assertEqual(audit["runtime_parameters"]["produced_events"], 100.0)
        self.assertEqual(
            audit["benchmark_constants"]["produced_events"],
            analysis.PRODUCED_SIGNAL_EVENTS,
        )


class CrossfitAndBinningTests(unittest.TestCase):
    class DummyModel:
        def __init__(self, rotation: int) -> None:
            self.rotation = rotation
            self.calls = 0

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            self.calls += 1
            probability = np.full(len(features), 0.1 * (self.rotation + 1))
            return np.column_stack([1.0 - probability, probability])

    def test_point_scores_are_cached_once_with_disjoint_fold_roles(self) -> None:
        point = analysis.MassPoint("direct", ms=1000.0)
        sample = synthetic_loaded_sample(
            "signal", "signal", point=point, nmerged=np.zeros(10, dtype=int)
        )
        models = [self.DummyModel(rotation) for rotation in range(analysis.N_FOLDS)]
        scores = analysis._predict_point_crossfit(sample, point, models)
        for event, fold in enumerate(sample.folds):
            self.assertAlmostEqual(scores.test[event], 0.1 * (fold + 1))
            validation_model = (int(fold) - 1) % analysis.N_FOLDS
            self.assertAlmostEqual(
                scores.validation[event], 0.1 * (validation_model + 1)
            )
        self.assertTrue(all(model.calls == 1 for model in models))
        for rotation in range(analysis.N_FOLDS):
            validation = sample.folds == (rotation + 1) % analysis.N_FOLDS
            test = sample.folds == rotation
            self.assertFalse(np.any(validation & test))

    def test_binning_checks_both_scenarios_and_has_inclusive_fallback(self) -> None:
        scores = np.linspace(0.1, 0.9, 5)
        selection = analysis._select_category_edges(
            scores,
            np.ones(5),
            scores,
            {
                "nominal": np.ones(5),
                "conservative": np.asarray([-1.0, -1.0, -1.0, -1.0, 10.0]),
            },
            min_raw=1,
            min_neff=0.0,
        )
        self.assertEqual(selection["status"], "ok")
        self.assertEqual(selection["fallback_level"], "inclusive_1_bin")
        self.assertIn("nominal", selection["inclusive_scenario_validation"])
        self.assertIn("conservative", selection["inclusive_scenario_validation"])


class YieldOutputTests(unittest.TestCase):
    def test_staged_category_yields_and_background_total_close(self) -> None:
        point = analysis.MassPoint("direct", ms=1000.0)
        signal = synthetic_loaded_sample("signal", "signal", point=point)
        background = synthetic_loaded_sample("sm_hhhh", "sm_hhhh")
        rows = analysis._category_yield_rows(
            "direct",
            point,
            [signal, background],
            {0, 1, 2},
            analysis.TAGGING_SCENARIOS,
            analysis.LUMINOSITY_FB,
            analysis.HBB_BRANCHING_RATIO,
            analysis.EPS_B,
            analysis.EPS_C,
            analysis.EPS_LIGHT,
        )
        signal_all = next(
            row
            for row in rows
            if row["tagging_scenario"] == "nominal"
            and row["sample_id"] == "signal"
            and row["category"] == "all"
        )
        self.assertAlmostEqual(signal_all["generated_yield"], 3000.0)
        self.assertAlmostEqual(
            signal_all["after_hbb_yield"], analysis.EIGHT_B_SIGNAL_EVENTS
        )
        self.assertAlmostEqual(
            signal_all["reconstructed_before_tag_yield"],
            analysis.EIGHT_B_SIGNAL_EVENTS,
        )
        self.assertAlmostEqual(
            signal_all["tagged_yield"], analysis.NOMINAL_TAG_SIGNAL_EVENTS
        )
        self.assertAlmostEqual(
            signal_all["used_in_limit_yield"], analysis.NOMINAL_TAG_SIGNAL_EVENTS
        )
        for stage in (
            "reconstructed_before_tag_yield",
            "tagged_yield",
            "used_in_limit_yield",
        ):
            self.assertTrue(signal_all[f"category_partition_{stage}_closure_pass"])
        total_background = next(
            row
            for row in rows
            if row["tagging_scenario"] == "nominal"
            and row["sample_id"] == "TOTAL_BACKGROUND"
            and row["category"] == "all"
        )
        self.assertAlmostEqual(
            total_background["tagged_yield"],
            sum(
                row["tagged_yield"]
                for row in rows
                if row["tagging_scenario"] == "nominal"
                and row["role"] != "signal"
                and row["role"] != "total_background"
                and row["category"] == "all"
            ),
        )


@unittest.skipUnless(
    os.environ.get("RUN_PYHF_INTEGRATION") == "1"
    and importlib.util.find_spec("pyhf") is not None,
    "set RUN_PYHF_INTEGRATION=1 with pyhf installed",
)
class PyhfIntegrationTests(unittest.TestCase):
    def test_reference_scaled_poi_avoids_wide_bound_minimizer_failure(self) -> None:
        fit = analysis._pyhf_limit(
            [
                {
                    "name": "resolved_fold0",
                    "signal": np.asarray([1.0, 2.0]),
                    "background": np.asarray([10.0, 5.0]),
                    "signal_staterror": np.asarray([0.1, 0.2]),
                    "background_staterror": np.asarray([1.0, 0.7]),
                }
            ]
        )
        self.assertEqual(fit["status"], "ok", fit)
        self.assertAlmostEqual(fit["expected_median"], 2.7769926, places=4)
        self.assertLessEqual(
            fit["maximum_returned_mu"],
            fit["boundary_fraction_threshold"] * fit["mu_fit_bounds"][1],
        )
        self.assertAlmostEqual(
            fit["expected_plus2sigma"],
            6.1742253,
            places=4,
        )


class CheckpointTests(unittest.TestCase):
    @staticmethod
    def point_shard(statuses: tuple[str, str] = ("ok", "ok")) -> dict[str, object]:
        point_id = "MS_1000"
        return {
            "run_fingerprint": "fingerprint",
            "point_id": point_id,
            "point_category_yields": [
                {"point_id": point_id, "tagging_scenario": scenario, "yield": 1.0}
                for scenario in ("nominal", "conservative")
            ],
            "score_bin_yields": [
                {"point_id": point_id, "tagging_scenario": scenario, "yield": 1.0}
                for scenario in ("nominal", "conservative")
            ],
            "point_limits": [
                {
                    "point_id": point_id,
                    "tagging_scenario": scenario,
                    "status": status,
                    "n_channels": 1,
                    "observed_asimov": 1.0,
                    "expected_minus2sigma": 0.7,
                    "expected_minus1sigma": 0.8,
                    "expected_median": 1.0,
                    "expected_plus1sigma": 1.2,
                    "expected_plus2sigma": 1.4,
                }
                for scenario, status in zip(
                    ("nominal", "conservative"), statuses, strict=True
                )
            ],
            "binning_audit": {
                category: {"status": "ok"} for category in analysis.CATEGORY_NAMES
            },
        }

    def test_run_config_refuses_mismatched_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            output.mkdir()
            first = analysis._initialize_run_config(output, {"mass": 1000})
            second = analysis._initialize_run_config(output, {"mass": 1000})
            self.assertEqual(first, second)
            with self.assertRaises(analysis.AnalysisInputError):
                analysis._initialize_run_config(output, {"mass": 1200})

    def test_full_retries_failed_or_malformed_shards_but_smoke_is_explicit(self) -> None:
        scenarios = ("nominal", "conservative")
        reusable, _ = analysis._point_shard_reuse_decision(
            self.point_shard(),
            mode="full",
            point_id="MS_1000",
            run_fingerprint="fingerprint",
            tagging_scenarios=scenarios,
        )
        self.assertTrue(reusable)
        failed = self.point_shard(("ok", "failed"))
        reusable_full, reason = analysis._point_shard_reuse_decision(
            failed,
            mode="full",
            point_id="MS_1000",
            run_fingerprint="fingerprint",
            tagging_scenarios=scenarios,
        )
        self.assertFalse(reusable_full)
        self.assertIn("non-successful", reason)
        reusable_smoke, _ = analysis._point_shard_reuse_decision(
            failed,
            mode="smoke",
            point_id="MS_1000",
            run_fingerprint="fingerprint",
            tagging_scenarios=scenarios,
        )
        self.assertTrue(reusable_smoke)
        malformed = self.point_shard()
        malformed["point_limits"] = malformed["point_limits"][:1]  # type: ignore[index]
        reusable_malformed, _ = analysis._point_shard_reuse_decision(
            malformed,
            mode="full",
            point_id="MS_1000",
            run_fingerprint="fingerprint",
            tagging_scenarios=scenarios,
        )
        self.assertFalse(reusable_malformed)

    def test_streamed_shard_outputs_are_valid_and_manifest_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_paths: list[Path] = []
            for index in range(3):
                path = root / f"point_{index}.json"
                analysis._write_json(
                    path,
                    {
                        "point_id": f"point_{index}",
                        "rows": [
                            {
                                "point_id": f"point_{index}",
                                "order": index,
                                f"field_{index}": index,
                            }
                        ],
                        "binning_audit": {"order": index},
                    },
                )
                shard_paths.append(path)
            iterator = analysis._iter_point_shard_rows(shard_paths, "rows")
            self.assertFalse(isinstance(iterator, list))
            self.assertEqual([row["order"] for row in iterator], [0, 1, 2])
            csv_path = root / "rows.csv"
            json_path = root / "rows.json"
            binning_path = root / "binning.json"
            analysis._write_sharded_csv(csv_path, shard_paths, "rows")
            analysis._write_sharded_json_array(json_path, shard_paths, "rows")
            analysis._write_sharded_binning_json(binning_path, shard_paths)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["order"]) for row in csv_rows], [0, 1, 2])
            self.assertTrue(all(f"field_{index}" in csv_rows[0] for index in range(3)))
            with json_path.open(encoding="utf-8") as handle:
                json_rows = json.load(handle)
            self.assertEqual([row["order"] for row in json_rows], [0, 1, 2])
            with binning_path.open(encoding="utf-8") as handle:
                binning = json.load(handle)
            self.assertEqual(list(binning), ["point_0", "point_1", "point_2"])


if __name__ == "__main__":
    unittest.main()
