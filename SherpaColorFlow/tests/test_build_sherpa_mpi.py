#!/usr/bin/env python3
"""Focused tests for the Sherpa MPI/UFO build wrapper."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_sherpa_mpi.sh"


class BuildSherpaMpiTests(unittest.TestCase):
    def _run_wrapper(self, **overrides):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        tmp = Path(temporary_directory.name)
        fake_bin = tmp / "bin"
        fake_bin.mkdir()
        cmake_log = tmp / "cmake.log"
        fake_cmake = fake_bin / "cmake"
        fake_cmake.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            "printf '%s\\n' \"$*\" >> \"$FAKE_CMAKE_LOG\"\n"
            "if [ \"${1:-}\" = '-S' ]; then\n"
            "  mkdir -p \"$BUILD_DIR\"\n"
            "  printf '%s\\n' 'CMAKE_INSTALL_LIBDIR:PATH=lib64' > \"$BUILD_DIR/CMakeCache.txt\"\n"
            "elif [ \"${1:-}\" = '--install' ]; then\n"
            "  mkdir -p \"$PREFIX/lib64/SHERPA-MC\"\n"
            "fi\n"
        )
        fake_cmake.chmod(0o755)

        mpi_home = tmp / "mpi"
        (mpi_home / "bin").mkdir(parents=True)
        for executable_name in ("mpicc", "mpicxx", "mpifort", "mpirun"):
            executable = mpi_home / "bin" / executable_name
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)

        env = os.environ.copy()
        env.pop("SHERPA_ENABLE_UFO", None)
        env.update(
            {
                "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
                "SRC_DIR": str(tmp / "source"),
                "BUILD_DIR": str(tmp / "build"),
                "PREFIX": str(tmp / "install"),
                "JOBS": "3",
                "MPI_HOME": str(mpi_home),
                "PYTHON_EXECUTABLE": sys.executable,
                "FAKE_CMAKE_LOG": str(cmake_log),
            }
        )
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        cmake_calls = cmake_log.read_text().splitlines() if cmake_log.exists() else []
        return result, cmake_calls, tmp

    def test_ufo_defaults_on_and_selects_requested_python(self):
        result, cmake_calls, tmp = self._run_wrapper()

        self.assertEqual(result.returncode, 0, result.stderr)
        configure_call = cmake_calls[0]
        self.assertIn("-DSHERPA_ENABLE_UFO=ON", configure_call)
        self.assertIn("-DSHERPA_ENABLE_PYTHON=OFF", configure_call)
        self.assertIn("-DPython_EXECUTABLE=" + sys.executable, configure_call)
        self.assertIn("-DCMAKE_C_COMPILER=" + str(tmp / "mpi/bin/mpicc"), configure_call)
        self.assertIn("--build " + str(tmp / "build") + " --parallel 3", cmake_calls[1])
        self.assertIn("export SHERPA_LIBDIR=" + str(tmp / "install/lib64"), result.stdout)

    def test_ufo_can_be_disabled_explicitly(self):
        result, cmake_calls, _ = self._run_wrapper(
            SHERPA_ENABLE_UFO="OFF", PYTHON_EXECUTABLE=""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        configure_call = cmake_calls[0]
        self.assertIn("-DSHERPA_ENABLE_UFO=OFF", configure_call)
        self.assertNotIn("-DPython_EXECUTABLE=", configure_call)

    def test_invalid_python_is_rejected_before_cmake(self):
        result, cmake_calls, _ = self._run_wrapper(
            PYTHON_EXECUTABLE="/definitely/not/a/python"
        )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(cmake_calls, [])
        self.assertIn("requires an executable Python 3 interpreter", result.stderr)

    def test_system_mpi_fallback_is_declared(self):
        script_text = SCRIPT.read_text()
        self.assertIn("system_mpi_home=/usr/lib64/openmpi", script_text)
        self.assertIn('elif has_mpi_toolchain "$system_mpi_home"', script_text)


if __name__ == "__main__":
    unittest.main()
