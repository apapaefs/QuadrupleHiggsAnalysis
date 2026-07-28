from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
ANALYZER_SOURCE = REPOSITORY / "Code" / "FourHiggs8bAnalysis_smear_CMS.cc"
SCHEMA_SOURCE = REPOSITORY / "Code" / "observable_schemas.py"


def _load_schema_module():
    specification = importlib.util.spec_from_file_location("observable_schemas_cpp_test", SCHEMA_SOURCE)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _cpp_string_vector(source: str, function_name: str) -> tuple[str, ...]:
    match = re.search(
        rf"const std::vector<std::string>& {function_name}\(\) \{{.*?= \{{(.*?)\n  \}};",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"could not find {function_name} initializer")
    return tuple(re.findall(r'\"([^\"]+)\"', match.group(1)))


class ExtendedObservablesCppTests(unittest.TestCase):
    def test_cms_smearing_uniformly_scales_the_massive_four_vector(self) -> None:
        source = ANALYZER_SOURCE.read_text()
        energy_smearing = source.split(
            "double smearedJetEnergyCMS(const PseudoJet& jet) {", 1
        )[1].split("\n}\n\nPseudoJet smearJetCMSLegacyMassless", 1)[0]
        legacy_mapping = source.split(
            "PseudoJet smearJetCMSLegacyMassless(const PseudoJet& jet) {", 1
        )[1].split("\n}\n\nPseudoJet smearJetCMSUniformFourVector", 1)[0]
        uniform_mapping = source.split(
            "PseudoJet smearJetCMSUniformFourVector", 2
        )[-1].split("\n}\n\nstd::string makeOutputName", 1)[0]

        self.assertEqual(energy_smearing.count("rnd.Gaus("), 1)
        self.assertIn("!std::isfinite(energy) || energy <= 0.0", energy_smearing)
        self.assertIn("std::max(kMinimumSmearedEnergy", energy_smearing)
        self.assertIn("energy + rnd.Gaus(0.0, sigma_energy)", energy_smearing)
        self.assertIn("SetPtEtaPhiE", legacy_mapping)
        self.assertIn("smeared_energy / std::cosh(jet.eta())", legacy_mapping)
        self.assertIn("const double scale = smeared_energy / energy;", uniform_mapping)
        self.assertIn("scale * jet.px()", uniform_mapping)
        self.assertIn("scale * jet.py()", uniform_mapping)
        self.assertIn("scale * jet.pz()", uniform_mapping)
        self.assertIn("output.m() - scale * jet.m()", uniform_mapping)
        self.assertNotIn("SetPtEtaPhiE", uniform_mapping)

    def test_smearing_model_and_output_tag_are_versioned(self) -> None:
        source = ANALYZER_SOURCE.read_text()
        self.assertIn(
            'kExtendedOutputTag = "extended-v2-uniform-smear-v1"', source
        )
        self.assertIn(
            'kJetSmearingModelId = "cms-energy-uniform-fourvector-v1"', source
        )
        self.assertIn(
            'kJetSmearingFourVectorScaling = "uniform_correlated"', source
        )
        self.assertIn(
            'tag == std::string("-") + kExtendedOutputTag', source
        )
        self.assertIn(
            'tag == std::string("-") + kLegacyExtendedOutputTag', source
        )
        self.assertIn('TNamed analysis_output_tag("analysis_output_tag"', source)
        self.assertIn('TNamed jet_smearing_model_id("jet_smearing_model_id"', source)
        self.assertIn(
            'TNamed jet_smearing_fourvector_scaling("jet_smearing_fourvector_scaling"',
            source,
        )
        self.assertIn('TParameter<Long64_t>("jet_smearing_seed"', source)
        self.assertIn(
            'TParameter<double>("max_smearing_mass_scaling_residual_gev"', source
        )

    def test_v2_pt_cut_is_applied_after_smearing_with_finite_guards(self) -> None:
        source = ANALYZER_SOURCE.read_text()
        candidate_block = source.split(
            "std::vector<PseudoJet> true_bjets_unsorted;", 1
        )[1].split("std::vector<PseudoJet> true_bjets =", 1)[0]
        tagged_block = candidate_block.split("if (write_extended_v2) {", 1)[1].split(
            "    } else {", 1
        )[0]
        true_b_block = tagged_block.split(
            "for (int jj = 0; jj < numbJets; ++jj)", 1
        )[1].split("for (int jj = 0; jj < numJets; ++jj)", 1)[0]
        non_b_block = tagged_block.split(
            "for (int jj = 0; jj < numJets; ++jj)", 1
        )[1]

        for block, candidate in (
            (true_b_block, "bjet_candidate"),
            (non_b_block, "jet_candidate"),
        ):
            raw_eta_position = block.index(f"const double raw_eta = {candidate}.eta()")
            raw_pt_position = block.index(f"const double raw_pt = {candidate}.perp()")
            finite_position = block.index("!std::isfinite(raw_eta)")
            smear_position = block.index("smearJetCMSUniformFourVector")
            smeared_pt_position = block.index("const double smeared_pt = smeared.perp()")
            finite_smeared_position = block.index("std::isfinite(smeared_pt)")
            self.assertLess(raw_eta_position, finite_position)
            self.assertLess(raw_pt_position, finite_position)
            self.assertLess(finite_position, smear_position)
            self.assertLess(smear_position, smeared_pt_position)
            self.assertLess(smeared_pt_position, finite_smeared_position)

        non_b_position = tagged_block.index("for (int jj = 0; jj < numJets; ++jj)")
        true_b_selection = source.index("std::vector<PseudoJet> true_bjets =")
        self.assertLess(source.index("std::vector<NonBJetCandidate> tagged_non_b_candidates"), true_b_selection)
        self.assertLess(
            source.index("for (int jj = 0; jj < numJets; ++jj)", source.index(tagged_block)),
            true_b_selection,
        )
        self.assertGreater(non_b_position, 0)

    def test_legacy_preprocessing_keeps_raw_pt_then_massless_smearing(self) -> None:
        source = ANALYZER_SOURCE.read_text()
        legacy_true_b = source.split(
            "// Preserve the historical untagged preprocessing", 1
        )[1].split("std::vector<PseudoJet> true_bjets =", 1)[0]
        self.assertLess(
            legacy_true_b.index("raw_pt <= kBJetPtCut"),
            legacy_true_b.index("smearJetCMSLegacyMassless"),
        )
        self.assertIn("!std::isfinite(raw_eta)", legacy_true_b)
        self.assertIn("!std::isfinite(raw_pt)", legacy_true_b)

    def test_dormant_jet_efficiency_interpolation_starts_at_20_gev(self) -> None:
        source = ANALYZER_SOURCE.read_text()
        efficiency = source.split("bool jetEfficiencyAccept", 2)[-1].split(
            "\n}\n\ndouble btagWeight", 1
        )[0]
        self.assertIn("(jet.perp() - 20.0) / (50.0 - 20.0)", efficiency)

    def test_object_level_pt_migrations_are_recorded_in_the_summary(self) -> None:
        source = ANALYZER_SOURCE.read_text()
        for population in ("true_b", "non_b"):
            for direction in ("upward", "downward"):
                counter = f"{population}_{direction}_pt_migrations"
                self.assertIn(f"long long {counter} = 0;", source)
                self.assertIn(f'\\"{counter}\\": ', source)
            for raw_pt_bin in ("10_12", "12_15", "15_20"):
                counter = (
                    f"{population}_upward_pt_migrations_"
                    f"raw_pt_{raw_pt_bin}_gev"
                )
                self.assertIn(f"long long {counter} = 0;", source)
                self.assertIn(f'\\"{counter}\\": ', source)
            self.assertEqual(
                source.count(f"++{population}_upward_pt_migrations;"), 1
            )
        self.assertIn(
            '"max_smearing_mass_scaling_residual_gev"', source
        )

    def test_root_metadata_matches_authoritative_schema(self) -> None:
        schemas = _load_schema_module()
        source = ANALYZER_SOURCE.read_text()
        self.assertEqual(
            _cpp_string_vector(source, "extendedFeatureNames"),
            schemas.EXTENDED_FEATURE_NAMES,
        )
        self.assertEqual(
            _cpp_string_vector(source, "extendedFeatureUnits"),
            schemas.EXTENDED_FEATURE_UNITS,
        )

    def test_corrected_residual_slots_preserve_candidate_order(self) -> None:
        schemas = _load_schema_module()
        source = ANALYZER_SOURCE.read_text()
        self.assertEqual(
            schemas.EXTENDED_FEATURE_NAMES[10:14],
            ("delta_m_h1", "delta_m_h2", "delta_m_h3", "delta_m_h4"),
        )
        self.assertIn("features[10 + h] = reconstruction.delta_m[h];", source)
        fill_block = source.split("void fillExtendedFeatures", 2)[-1].split(
            "const std::vector<std::string>& extendedFeatureNames", 1
        )[0]
        self.assertNotIn("sorted_delta_m", fill_block)

    def test_extended_observables_cpp_contract(self) -> None:
        compiler = shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest(
                "a C++ compiler is required for the extended-observable contract test"
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "extended-observables-test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-Wall",
                    "-Wextra",
                    "-pedantic",
                    "-I",
                    str(REPOSITORY / "Code"),
                    str(REPOSITORY / "tests" / "test_extended_observables_driver.cc"),
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=REPOSITORY,
            )
            subprocess.run([str(executable)], check=True, cwd=REPOSITORY)


if __name__ == "__main__":
    unittest.main()
