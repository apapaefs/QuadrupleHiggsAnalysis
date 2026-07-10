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
