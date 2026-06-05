from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .api import solve
from .runner import packaged_executable, packaged_mpi_launcher


def main() -> int:
    executable = packaged_executable()
    launcher = packaged_mpi_launcher()
    print(f"parosol executable: {executable}")
    print(f"MPI launcher: {launcher}")
    if not executable.exists():
        raise SystemExit(f"Packaged ParOSol executable not found: {executable}")
    if launcher is None or not launcher.exists():
        raise SystemExit(f"Packaged MPI launcher not found: {launcher}")

    with tempfile.TemporaryDirectory(prefix="parosol_wheel_smoke_") as tmp:
        result = solve(
            material=np.ones((3, 3, 3), dtype=np.float32) * 1000.0,
            spacing=(1.0, 1.0, 1.0),
            outputs=("sed",),
            tolerance=1e-4,
            level=2,
            mpi_processes=2,
            mpi_launcher="packaged",
            work_dir=Path(tmp),
            stream_output=True,
        )

    if result.summary.run is None:
        raise SystemExit("Smoke solve did not return solver summary metrics.")
    print(f"smoke MPI command: {' '.join(result.command)}")
    print(f"smoke iterations: {result.summary.run.iterations}")
    print(f"smoke residual: {result.summary.run.relative_residual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
