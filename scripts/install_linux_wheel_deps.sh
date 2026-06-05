#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${PAROSOL_LINUX_CONDA_ROOT:-/opt/parosol-conda}"
CONDA_BIN="${CONDA_ROOT}/bin/conda"
MAMBA_BIN="${CONDA_ROOT}/bin/mamba"

if [[ ! -x "${CONDA_BIN}" ]]; then
  installer="/tmp/parosol-miniforge.sh"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "${installer}" "${url}"
  else
    python - "${url}" "${installer}" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  fi
  bash "${installer}" -b -p "${CONDA_ROOT}"
fi

"${CONDA_BIN}" config --system --set always_yes yes
"${CONDA_BIN}" config --system --set channel_priority strict
"${CONDA_BIN}" config --system --add channels conda-forge

PKG="${CONDA_BIN}"
if [[ -x "${MAMBA_BIN}" ]]; then
  PKG="${MAMBA_BIN}"
fi

"${PKG}" install -y -p "${CONDA_ROOT}" \
  cmake \
  ninja \
  eigen \
  'hdf5=1.14.*' \
  openmpi=4.1.6

"${CONDA_ROOT}/bin/mpirun" --version
