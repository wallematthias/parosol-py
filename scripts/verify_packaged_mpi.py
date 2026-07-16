from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_runner_module():
    runner_path = ROOT / "src" / "parosol_py" / "runner.py"
    spec = importlib.util.spec_from_file_location(
        "_parosol_py_runner_for_mpi_check",
        runner_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner module from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_runner = _load_runner_module()
_runner._package_bin_dir = lambda: ROOT / "src" / "parosol_py" / "bin"
mpi_runtime_environment = _runner.mpi_runtime_environment
packaged_mpi_launcher = _runner.packaged_mpi_launcher
resolve_mpi_launcher = _runner.resolve_mpi_launcher


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
