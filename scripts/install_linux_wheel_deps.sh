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

/usr/lib64/openmpi/bin/mpirun --version
