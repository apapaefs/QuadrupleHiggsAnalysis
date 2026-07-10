import builtins
import importlib.util
import json
import re
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch


REPO_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_DIR / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import observable_schemas as schemas
import read_root_varfiles as reader


class _FakeNamed:
    def __init__(self, title):
        self.title = title

    def GetTitle(self):
        return self.title


class _FakeParameter:
    def __init__(self, value):
        self.value = value

    def GetVal(self):
        return self.value


class _FakeLeaf:
    def __init__(self, length):
        self.length = length

    def GetLenStatic(self):
        return self.length


class _FakeBranch:
    def __init__(self, name, length):
        self.name = name
        self.length = length

    def GetLeaf(self, name):
        return _FakeLeaf(self.length) if name == self.name else None

    def GetTitle(self):
        suffix = f"[{self.length}]" if self.length != 1 else ""
        return f"{self.name}{suffix}/D"


class _FakeTree:
    def __init__(self, rows, branch_lengths):
        self.rows = rows
        self.branches = {
            name: _FakeBranch(name, length) for name, length in branch_lengths.items()
        }

    def GetBranch(self, name):
        return self.branches.get(name)

    def GetLeaf(self, name):
        branch = self.GetBranch(name)
        return branch.GetLeaf(name) if branch else None

    def GetEntries(self):
        return len(self.rows)

    def GetEntry(self, index):
        for name, value in self.rows[index].items():
            setattr(self, name, value)
        return 1


class _FakeFile:
    def __init__(self, objects):
        self.objects = objects
        self.closed = False

    def Get(self, name):
        return self.objects.get(name)

    def IsZombie(self):
        return False

    def Close(self):
        self.closed = True


class _FakeRoot:
    def __init__(self, root_file):
        self.root_file = root_file
        owner = self

        class TFile:
            @staticmethod
            def Open(_path):
                return owner.root_file

        self.TFile = TFile


class _FakeBooster:
    def __init__(self, feature_count, feature_names=None):
        self.feature_count = feature_count
        self.feature_names = feature_names
        self.attributes = {}

    def set_attr(self, **attributes):
        self.attributes.update(attributes)

    def attr(self, name):
        return self.attributes.get(name)

    def num_features(self):
        return self.feature_count


def _extended_root_file(rows=None, feature_names=None, feature_units=None):
    rows = rows or [
        {
            "features": [float(index) for index in range(91)],
            "weight": -2.0,
            "event_index": 17,
            "cut_mask": 31,
            "passes_legacy_full_selection": True,
        }
    ]
    tree = _FakeTree(
        rows,
        {
            "features": 91,
            "weight": 1,
            "event_index": 1,
            "cut_mask": 1,
            "passes_legacy_full_selection": 1,
        },
    )
    return _FakeFile(
        {
            "Data3": tree,
            "Data3_observable_schema": _FakeNamed(schemas.EXTENDED_SCHEMA_ID),
            "Data3_feature_names_json": _FakeNamed(
                json.dumps(list(feature_names or schemas.EXTENDED_FEATURE_NAMES))
            ),
            "Data3_feature_units_json": _FakeNamed(
                json.dumps(list(feature_units or schemas.EXTENDED_FEATURE_UNITS))
            ),
            "Data3_feature_count": _FakeParameter(91),
            "Data3_pairing_count": _FakeParameter(105),
        }
    )


class ObservableSchemaTests(unittest.TestCase):
    def test_exact_schema_and_profile_boundaries(self):
        legacy = schemas.get_schema("legacy-28-v1")
        extended = schemas.get_schema("extended-91-v2")

        self.assertEqual(legacy.tree_name, "Data2")
        self.assertEqual(legacy.stored_value_count, 29)
        self.assertEqual(extended.tree_name, "Data3")
        self.assertEqual(len(extended.feature_names), 91)
        self.assertEqual(extended.feature_names[28], "m_bb_h1")
        self.assertEqual(extended.feature_names[34], "n_pairings_chi8_lt60")
        self.assertEqual(extended.feature_names[90], "zness")
        self.assertEqual(extended.feature_units[34], "count")
        self.assertEqual(extended.feature_units[63], "rad")

        self.assertEqual(schemas.get_feature_profile("corrected28").feature_count, 28)
        self.assertEqual(schemas.get_feature_profile("core52").feature_count, 52)
        self.assertEqual(schemas.get_feature_profile("full91").feature_count, 91)
        with self.assertRaisesRegex(schemas.ObservableSchemaError, "not available"):
            schemas.get_feature_contract("legacy-28-v1", "core52")

    def test_schema_objects_are_immutable(self):
        schema = schemas.get_schema("extended-91-v2")
        with self.assertRaises(TypeError):
            schemas.SCHEMA_REGISTRY["new"] = schema
        with self.assertRaises(Exception):
            schema.tree_name = "Other"

    def test_cpp_root_metadata_matches_python_contract_exactly(self):
        source = (CODE_DIR / "FourHiggs8bAnalysis_smear_CMS.cc").read_text()

        def extract(function_name, variable_name):
            match = re.search(
                rf"{function_name}\(\)\s*\{{.*?{variable_name}\s*=\s*\{{(.*?)\}};",
                source,
                re.DOTALL,
            )
            self.assertIsNotNone(match, f"could not find {function_name} metadata")
            return tuple(re.findall(r'"([^"]+)"', match.group(1)))

        self.assertEqual(
            extract("extendedFeatureNames", "names"),
            schemas.EXTENDED_FEATURE_NAMES,
        )
        self.assertEqual(
            extract("extendedFeatureUnits", "units"),
            schemas.EXTENDED_FEATURE_UNITS,
        )

    def test_canonical_tagged_basename(self):
        tagged = "/analysis/HW-run_gg_4h_4_0.0_0.0-extended-v2_var.smearCMS.root"
        untagged = "/analysis/HW-run_gg_4h_4_0.0_0.0_var.smearCMS.root"
        self.assertEqual(
            schemas.strip_extended_v2_tag(tagged),
            untagged,
        )
        self.assertEqual(
            schemas.canonical_sample_basename(tagged),
            "HW-run_gg_4h_4_0.0_0.0",
        )
        self.assertEqual(
            schemas.canonical_sample_basename(untagged),
            "HW-run_gg_4h_4_0.0_0.0",
        )

    def test_model_metadata_round_trip_and_semantic_mismatch(self):
        booster = _FakeBooster(52)
        metadata = schemas.attach_model_metadata(
            booster,
            observable_set="extended-91-v2",
            feature_profile="core52",
            training_strategy="pooled-crossfit-v2",
        )
        self.assertEqual(metadata, schemas.read_model_metadata(booster))
        validated = schemas.validate_model_contract(
            booster, "extended-91-v2", "core52"
        )
        self.assertEqual(validated["training_strategy"], "pooled-crossfit-v2")

        with self.assertRaisesRegex(schemas.ModelContractError, "feature profile"):
            schemas.validate_model_contract(booster, "extended-91-v2", "full91")
        with self.assertRaisesRegex(schemas.ModelContractError, "observable schema"):
            schemas.validate_model_contract(booster, "legacy-28-v1", "corrected28")

        with self.assertRaisesRegex(schemas.ModelContractError, "has 28 inputs"):
            schemas.attach_model_metadata(
                _FakeBooster(28),
                observable_set="extended-91-v2",
                feature_profile="core52",
            )

        wrong_names = list(schemas.EXTENDED_FEATURE_NAMES[:52])
        wrong_names[0], wrong_names[1] = wrong_names[1], wrong_names[0]
        with self.assertRaisesRegex(schemas.ModelContractError, "feature 0"):
            schemas.attach_model_metadata(
                _FakeBooster(52, wrong_names),
                observable_set="extended-91-v2",
                feature_profile="core52",
            )

    def test_metadata_free_models_only_use_explicit_legacy_path(self):
        legacy_booster = _FakeBooster(28)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metadata = schemas.validate_model_contract(
                legacy_booster, "legacy-28-v1", "corrected28"
            )
        self.assertTrue(metadata["metadata_inferred"])
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0].message, schemas.LegacyModelWarning)

        with self.assertRaisesRegex(schemas.ModelContractError, "no observable metadata"):
            schemas.validate_model_contract(
                legacy_booster, "extended-91-v2", "corrected28"
            )


class RootContractTests(unittest.TestCase):
    def test_reader_module_can_import_when_pyroot_is_unavailable(self):
        module_path = CODE_DIR / "read_root_varfiles.py"
        original_import = builtins.__import__

        def without_root(name, *args, **kwargs):
            if name == "ROOT":
                raise ImportError("ROOT deliberately unavailable")
            return original_import(name, *args, **kwargs)

        module_name = "read_root_varfiles_without_root_test"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            with patch("builtins.__import__", side_effect=without_root):
                spec.loader.exec_module(module)
            self.assertIsNone(module.ROOT)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "empty.root"
                path.touch()
                with self.assertRaisesRegex(RuntimeError, "PyROOT is required"):
                    module.inspect_ROOT_varfile(path)
        finally:
            sys.modules.pop(module_name, None)

    def test_extended_root_metadata_and_rows_are_validated(self):
        root_file = _extended_root_file()
        fake_root = _FakeRoot(root_file)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample-extended-v2_var.smearCMS.root"
            path.touch()

            info = reader.inspect_ROOT_varfile(
                path,
                observable_set="extended-91-v2",
                feature_profile="core52",
                root_module=fake_root,
            )
            self.assertTrue(info.root_metadata_verified)
            self.assertEqual(info.feature_count, 52)
            self.assertEqual(info.pairing_count, 105)
            self.assertEqual(info.canonical_sample_id, "sample")

            features, labels, weights, metadata = reader.read_ROOT_varfile(
                path,
                sample_id=7,
                xsec=5.0,
                observable_set="extended-91-v2",
                feature_profile="core52",
                return_metadata=True,
                root_module=fake_root,
            )
            self.assertEqual(len(features), 1)
            self.assertEqual(len(features[0]), 52)
            self.assertEqual(labels, [7])
            self.assertEqual(weights, [-10.0])
            self.assertEqual(metadata["event_indices"], [17])
            self.assertEqual(metadata["source_entry_indices"], [0])
            self.assertEqual(metadata["cut_masks"], [31])
            self.assertEqual(metadata["passes_legacy_full_selection"], [True])

    def test_extended_metadata_name_order_mismatch_fails_closed(self):
        names = list(schemas.EXTENDED_FEATURE_NAMES)
        names[0], names[1] = names[1], names[0]
        fake_root = _FakeRoot(_extended_root_file(feature_names=names))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample-extended-v2_var.smearCMS.root"
            path.touch()
            with self.assertRaisesRegex(reader.RootVariableFileError, "feature 0"):
                reader.inspect_ROOT_varfile(
                    path,
                    observable_set="extended-91-v2",
                    root_module=fake_root,
                )

    def test_legacy_default_api_and_weight_fallback_are_unchanged(self):
        values = [2.0] + [float(index) for index in range(1, 29)]
        tree = _FakeTree([{"variables": values}], {"variables": 29})
        fake_root = _FakeRoot(_FakeFile({"Data2": tree}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_var.smearCMS.root"
            path.touch()
            features, labels, weights = reader.read_ROOT_varfile(
                path, sample_id=3, xsec=4.0, root_module=fake_root
            )
        self.assertEqual(features, [[float(index) for index in range(1, 29)]])
        self.assertEqual(labels, [3])
        self.assertEqual(weights, [8.0])


if __name__ == "__main__":
    unittest.main()
