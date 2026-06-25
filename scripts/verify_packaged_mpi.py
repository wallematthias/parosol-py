from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parosol_py.runner import (  # noqa: E402
    mpi_runtime_environment,
    packaged_mpi_launcher,
    resolve_mpi_launcher,
)


def main() -> int:
    launcher = packaged_mpi_launcher()
    if launcher is None:
        raise SystemExit("Packaged MPI launcher was not found.")
    resolved = Path(resolve_mpi_launcher("packaged"))
    if resolved != launcher:
        raise SystemExit(f"Packaged MPI launcher resolved incorrectly: {resolved}")

    command = [
        str(launcher),
        "-np",
        "2",
        sys.executable,
        "-c",
        "print('parosol-mpi-ok')",
    ]
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=mpi_runtime_environment(command),
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "Packaged MPI launcher failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Return code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if proc.stdout.count("parosol-mpi-ok") != 2:
        raise SystemExit(
            "Packaged MPI launcher did not start two ranks.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    print(f"Packaged MPI launcher OK: {launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
