from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "dist-local-check"
REQUIRED_WHEEL_MEMBERS = (
    "parosol_py/bin/parosol",
    "parosol_py/config_templates/default.yaml",
    "parosol_py/config_templates/profiles/xtremectii.yaml",
    "parosol_py/config_templates/profiles/vertebra.yaml",
    "parosol_py/licenses/parosol_native_LICENSE.txt",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local parosol-py verification gate before GitHub Actions."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for the locally built wheel.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Build and inspect the wheel without running pytest first.",
    )
    parser.add_argument(
        "--smoke-install",
        action="store_true",
        help="Install the built wheel into a temporary venv and import it.",
    )
    args = parser.parse_args(argv)

    env = _build_env()
    _require_tools(env, ("cmake", "ninja"))

    if not args.skip_tests:
        _run([sys.executable, "-m", "pytest", "-q"], env=env)

    outdir = args.outdir.expanduser().resolve()
    if outdir.exists():
        shutil.rmtree(outdir)
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(outdir),
            "--no-isolation",
        ],
        env=env,
    )

    wheel = _single_wheel(outdir)
    _inspect_wheel(wheel)
    if args.smoke_install:
        _smoke_install(wheel)
    print(f"local check OK: {wheel}")
    return 0


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    python_bin = Path(sys.executable).resolve().parent
    env["PATH"] = f"{python_bin}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "1")
    if sys.platform == "darwin" and not env.get("SDKROOT"):
        sdkroot = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
            capture_output=True,
            text=True,
            check=False,
        )
        if sdkroot.returncode == 0 and sdkroot.stdout.strip():
            env["SDKROOT"] = sdkroot.stdout.strip()
    return env


def _require_tools(env: dict[str, str], names: tuple[str, ...]) -> None:
    missing = [name for name in names if shutil.which(name, path=env["PATH"]) is None]
    if missing:
        raise SystemExit(
            "missing build tool(s): "
            + ", ".join(missing)
            + ". Activate the conda/build environment first."
        )


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _single_wheel(outdir: Path) -> Path:
    wheels = sorted(outdir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in {outdir}, found {len(wheels)}")
    return wheels[0]


def _inspect_wheel(wheel: Path) -> None:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = [name for name in REQUIRED_WHEEL_MEMBERS if name not in names]
    if missing:
        raise SystemExit("wheel is missing required member(s): " + ", ".join(missing))
    template_count = sum(
        name.startswith("parosol_py/config_templates/") and name.endswith(".yaml")
        for name in names
    )
    if template_count < 3:
        raise SystemExit(f"wheel has too few config templates: {template_count}")


def _smoke_install(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="parosol_wheel_smoke_") as tmp:
        venv = Path(tmp) / "venv"
        _run([sys.executable, "-m", "venv", str(venv)])
        python = venv / "bin" / "python"
        if sys.platform.startswith("win"):
            python = venv / "Scripts" / "python.exe"
        _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(python), "-m", "pip", "install", str(wheel)])
        _run(
            [
                str(python),
                "-c",
                (
                    "from parosol_py.runner import packaged_executable, packaged_mpi_launcher; "
                    "from parosol_py.config_templates import read_config_template; "
                    "p = packaged_executable(); "
                    "m = packaged_mpi_launcher(); "
                    "assert p.exists(), p; "
                    "assert m is not None and m.exists(), m; "
                    "assert 'materials:' in read_config_template('default')"
                ),
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())
