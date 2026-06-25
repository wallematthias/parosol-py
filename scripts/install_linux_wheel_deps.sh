#!/usr/bin/env bash
set -euo pipefail

if command -v dnf >/dev/null 2>&1; then
  dnf config-manager --set-enabled crb >/dev/null 2>&1 || true
  dnf install -y \
    eigen3-devel \
    hdf5-devel \
    openmpi-devel
elif command -v yum >/dev/null 2>&1; then
  yum config-manager --enable crb >/dev/null 2>&1 || true
  yum install -y \
    eigen3-devel \
    hdf5-devel \
    openmpi-devel
else
  echo "Neither dnf nor yum is available; cannot install manylinux build dependencies." >&2
  exit 1
fi

openmpi_prefix="${PAROSOL_OPENMPI_PREFIX:-/usr/lib64/openmpi}"
if [[ ! -x "${openmpi_prefix}/bin/mpirun" ]]; then
  echo "OpenMPI launcher is missing or not executable: ${openmpi_prefix}/bin/mpirun" >&2
  exit 1
fi
if [[ ! -x "${openmpi_prefix}/bin/ompi_info" ]]; then
  echo "OpenMPI ompi_info is missing or not executable: ${openmpi_prefix}/bin/ompi_info" >&2
  exit 1
fi

"${openmpi_prefix}/bin/mpirun" --version
"${openmpi_prefix}/bin/ompi_info" --parsable --path prefix
