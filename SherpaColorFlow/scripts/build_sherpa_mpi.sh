#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
root_dir=$(cd "$script_dir/.." && pwd -P)
src_dir=${SRC_DIR:-"$root_dir/sherpa"}
build_dir=${BUILD_DIR:-"$root_dir/build/sherpa-mpi"}
prefix=${PREFIX:-"$root_dir/install/sherpa-mpi"}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
mpi_home=${MPI_HOME:-/home/apapaefs/Projects/4H/sherpa-deps/openmpi-4.1.6}

cmake_args=(
  -S "$src_dir"
  -B "$build_dir"
  -DCMAKE_INSTALL_PREFIX="$prefix"
  -DSHERPA_ENABLE_MPI=ON
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

if [[ -x "$mpi_home/bin/mpicc" && -x "$mpi_home/bin/mpicxx" && -x "$mpi_home/bin/mpifort" ]]; then
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

cat <<EOF
Installed patched Sherpa to:
  $prefix

To use it:
  export PATH=$prefix/bin:\$PATH
  export LD_LIBRARY_PATH=$prefix/lib:$prefix/lib64:\${LD_LIBRARY_PATH:-}
EOF
