#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
root_dir=$(cd "$script_dir/.." && pwd -P)
src_dir=${SRC_DIR:-"$root_dir/sherpa"}
build_dir=${BUILD_DIR:-"$root_dir/build/sherpa-mpi"}
prefix=${PREFIX:-"$root_dir/install/sherpa-mpi"}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
default_mpi_home=/home/apapaefs/Projects/4H/sherpa-deps/openmpi-4.1.6
system_mpi_home=/usr/lib64/openmpi
sherpa_enable_ufo=${SHERPA_ENABLE_UFO:-ON}
python_executable=${PYTHON_EXECUTABLE:-}

case "$sherpa_enable_ufo" in
  1|ON|On|on|TRUE|True|true|YES|Yes|yes)
    sherpa_enable_ufo=ON
    ;;
  0|OFF|Off|off|FALSE|False|false|NO|No|no)
    sherpa_enable_ufo=OFF
    ;;
  *)
    echo "SHERPA_ENABLE_UFO must be ON or OFF (got '$sherpa_enable_ufo')." >&2
    exit 2
    ;;
esac

has_mpi_toolchain() {
  local candidate=$1
  [[ -x "$candidate/bin/mpicc" && \
     -x "$candidate/bin/mpicxx" && \
     -x "$candidate/bin/mpifort" ]]
}

if [[ -n "${MPI_HOME:-}" ]]; then
  mpi_home=$MPI_HOME
elif has_mpi_toolchain "$default_mpi_home"; then
  mpi_home=$default_mpi_home
elif has_mpi_toolchain "$system_mpi_home"; then
  mpi_home=$system_mpi_home
  echo "Default MPI installation not found; using $mpi_home." >&2
else
  mpi_home=$default_mpi_home
fi

if [[ "$sherpa_enable_ufo" == ON ]]; then
  if [[ -z "$python_executable" ]]; then
    python_executable=$(command -v python3 || true)
  fi
  if [[ -z "$python_executable" || ! -x "$python_executable" ]]; then
    echo "UFO support requires an executable Python 3 interpreter; set PYTHON_EXECUTABLE." >&2
    exit 2
  fi
  if ! "$python_executable" -c 'import sys; raise SystemExit(sys.version_info < (3, 5))'; then
    echo "PYTHON_EXECUTABLE=$python_executable must provide Python 3.5 or newer." >&2
    exit 2
  fi
elif [[ -n "$python_executable" && ! -x "$python_executable" ]]; then
  echo "PYTHON_EXECUTABLE=$python_executable is not executable." >&2
  exit 2
fi

cmake_args=(
  -S "$src_dir"
  -B "$build_dir"
  -DCMAKE_INSTALL_PREFIX="$prefix"
  -DSHERPA_ENABLE_MPI=ON
  -DSHERPA_ENABLE_UFO="$sherpa_enable_ufo"
  -DSHERPA_ENABLE_LHAPDF=ON
  -DSHERPA_ENABLE_INSTALL_LHAPDF=ON
  -DSHERPA_ENABLE_INTERNAL_PDFS=ON
  -DSHERPA_ENABLE_INSTALL_LIBZIP=ON
  -DSHERPA_ENABLE_GZIP=OFF
  -DSHERPA_ENABLE_HEPMC3=OFF
  -DSHERPA_ENABLE_HEPMC3_ROOT=OFF
  -DSHERPA_ENABLE_PYTHON=OFF
  -DSHERPA_ENABLE_RIVET=OFF
  -DSHERPA_ENABLE_ROOT=OFF
  -DSHERPA_ENABLE_OPENLOOPS=OFF
  -DSHERPA_ENABLE_RECOLA=OFF
  -DSHERPA_ENABLE_GOSAM=OFF
  -DSHERPA_ENABLE_ANALYSIS=OFF
  -DSHERPA_ENABLE_TESTING=OFF
)

if [[ "$sherpa_enable_ufo" == ON ]]; then
  cmake_args+=(
    -DPython_EXECUTABLE="$python_executable"
  )
fi

if has_mpi_toolchain "$mpi_home"; then
  cmake_args+=(
    -DCMAKE_C_COMPILER="$mpi_home/bin/mpicc"
    -DCMAKE_CXX_COMPILER="$mpi_home/bin/mpicxx"
    -DCMAKE_Fortran_COMPILER="$mpi_home/bin/mpifort"
    -DMPIEXEC_EXECUTABLE="$mpi_home/bin/mpirun"
  )
else
  echo "MPI_HOME=$mpi_home does not contain mpicc/mpicxx/mpifort; using compilers from PATH." >&2
fi

cmake "${cmake_args[@]}"
cmake --build "$build_dir" --parallel "$jobs"
cmake --install "$build_dir"

install_libdir_rel=
if [[ -f "$build_dir/CMakeCache.txt" ]]; then
  while IFS='=' read -r key value; do
    if [[ "$key" == CMAKE_INSTALL_LIBDIR:* ]]; then
      install_libdir_rel=$value
    fi
  done < "$build_dir/CMakeCache.txt"
fi

if [[ -z "$install_libdir_rel" ]]; then
  if [[ -d "$prefix/lib64/SHERPA-MC" ]]; then
    install_libdir_rel=lib64
  else
    install_libdir_rel=lib
  fi
fi

if [[ "$install_libdir_rel" == /* ]]; then
  install_libdir=$install_libdir_rel
else
  install_libdir=$prefix/$install_libdir_rel
fi

cat <<EOF
Installed patched Sherpa to:
  $prefix

UFO support:
  $sherpa_enable_ufo

To use it:
  export PATH=$prefix/bin:$mpi_home/bin:\$PATH
  export SHERPA_LIBDIR=$install_libdir
  export LD_LIBRARY_PATH=\$SHERPA_LIBDIR/SHERPA-MC:\$SHERPA_LIBDIR:$mpi_home/lib:$mpi_home/lib64:\${LD_LIBRARY_PATH:-}
EOF
