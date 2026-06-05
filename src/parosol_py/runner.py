from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from importlib import metadata
from importlib import resources
from pathlib import Path

OUTPUT_FLAGS = {
    "sed": "--SED",
    "strain": "--strain",
    "stress": "--stress",
    "von_mises": "--VonMises",
    "effective_strain": "--EFF",
    "deviatoric_strain": "--e_dev",
    "volumetric_strain": "--e_vol",
}


@dataclass(frozen=True)
class RunSummary:
    iterations: int | None = None
    relative_residual: float | None = None
    absolute_residual: float | None = None
    overall_time_seconds: float | None = None


@dataclass(frozen=True)
class RunResult:
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    summary: RunSummary


def packaged_executable() -> Path:
    for name in _platform_executable_names("parosol"):
        executable = _package_bin_dir() / name
        if executable.exists():
            return executable
    try:
        distribution = metadata.distribution("parosol-py")
    except metadata.PackageNotFoundError:
        return Path(resources.files("parosol_py").joinpath("bin/parosol"))
    for name in _platform_executable_names("parosol"):
        installed = Path(distribution.locate_file(f"parosol_py/bin/{name}"))
        if installed.exists():
            return installed
    return Path(distribution.locate_file("parosol_py/bin/parosol"))


def packaged_mpi_launcher() -> Path | None:
    bin_dir = _package_bin_dir()
    candidates = [
        bin_dir / "msmpi" / "mpiexec.exe",
        bin_dir / "openmpi" / "bin" / "mpirun",
        bin_dir / "openmpi" / "bin" / "mpiexec",
        bin_dir / "openmpi" / "mpirun",
        bin_dir / "openmpi" / "mpiexec",
        bin_dir / "mpirun",
        bin_dir / "mpiexec",
        bin_dir / "mpiexec.exe",
    ]
    for launcher in candidates:
        if launcher.exists():
            return launcher

    try:
        distribution = metadata.distribution("parosol-py")
    except metadata.PackageNotFoundError:
        return None
    for relative in (
        "parosol_py/bin/msmpi/mpiexec.exe",
        "parosol_py/bin/openmpi/bin/mpirun",
        "parosol_py/bin/openmpi/bin/mpiexec",
        "parosol_py/bin/openmpi/mpirun",
        "parosol_py/bin/openmpi/mpiexec",
        "parosol_py/bin/mpirun",
        "parosol_py/bin/mpiexec",
        "parosol_py/bin/mpiexec.exe",
    ):
        launcher = Path(distribution.locate_file(relative))
        if launcher.exists():
            return launcher
    return None


def resolve_mpi_launcher(mpi_launcher: str | Path = "mpirun") -> str:
    token = str(mpi_launcher).strip()
    packaged = packaged_mpi_launcher()
    if token.lower() in {"", "auto", "packaged", "mpirun", "mpiexec", "mpiexec.exe"}:
        if packaged is not None:
            return str(packaged)
        found = shutil.which(token or "mpirun")
        return found if found is not None else (token or "mpirun")
    return str(Path(mpi_launcher))


def _package_bin_dir() -> Path:
    return Path(resources.files("parosol_py").joinpath("bin"))


def _platform_executable_names(base: str) -> tuple[str, ...]:
    return (base, f"{base}.exe")


def build_parosol_command(
    *,
    executable: str | Path,
    input_file: str | Path,
    outputs: tuple[str, ...],
    tolerance: float = 1e-6,
    level: int = 6,
    mpi_processes: int = 1,
    mpi_launcher: str | Path = "mpirun",
) -> list[str]:
    if int(mpi_processes) < 1:
        raise ValueError("mpi_processes must be >= 1")
    cmd = [str(Path(executable))]
    for output in outputs:
        token = output.strip().lower()
        if token in {"forces", "force", "displacements", "disp"}:
            continue
        if token not in OUTPUT_FLAGS:
            raise ValueError(f"Unsupported ParOSol output '{output}'")
        flag = OUTPUT_FLAGS[token]
        if flag not in cmd:
            cmd.append(flag)

    cmd.extend(
        [
            "--tol",
            f"{float(tolerance):g}",
            "--level",
            str(int(level)),
            str(Path(input_file)),
        ]
    )
    if int(mpi_processes) > 1:
        cmd = [resolve_mpi_launcher(mpi_launcher), "-np", str(int(mpi_processes)), *cmd]
    return cmd


def parse_run_summary(stdout: str) -> RunSummary:
    patterns = {
        "iterations": (r"#\s+Nr of It:\s+([0-9]+)", int),
        "relative_residual": (r"#\s+Relative residuum:\s+([-+0-9.eE]+)", float),
        "absolute_residual": (r"#\s+Absolute residuum:\s+([-+0-9.eE]+)", float),
        "overall_time_seconds": (r"#\s+Overall:\s+([-+0-9.eE]+)", float),
    }
    values = {}
    for name, (pattern, cast) in patterns.items():
        match = re.search(pattern, stdout)
        values[name] = cast(match.group(1)) if match else None
    return RunSummary(**values)


def run_parosol(
    command: list[str], *, cwd: str | Path | None = None, stream: bool = False
) -> RunResult:
    if stream:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_thread = threading.Thread(
            target=_tee_pipe,
            args=(proc.stdout, sys.stdout, stdout_chunks),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_tee_pipe,
            args=(proc.stderr, sys.stderr, stderr_chunks),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
    else:
        proc = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=False
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    summary = parse_run_summary(stdout)
    return RunResult(
        command=command,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        summary=summary,
    )


def _tee_pipe(pipe, sink, chunks: list[str]) -> None:
    if pipe is None:
        return
    try:
        for line in pipe:
            chunks.append(line)
            sink.write(line)
            sink.flush()
    finally:
        pipe.close()
