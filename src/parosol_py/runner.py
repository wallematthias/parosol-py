from __future__ import annotations

import re
import os
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
        if _is_compatible_executable(launcher):
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
        if _is_compatible_executable(launcher):
            return launcher
    return None


def resolve_mpi_launcher(mpi_launcher: str | Path = "mpirun") -> str:
    token = str(mpi_launcher).strip()
    if not token:
        token = "auto"
    explicit_env = os.environ.get("PAROSOL_MPI_LAUNCHER")
    automatic_tokens = {"auto", "packaged", "mpirun", "mpiexec", "mpiexec.exe"}
    if explicit_env and token.lower() in automatic_tokens:
        return _resolve_explicit_mpi_launcher(explicit_env)
    if token.lower() in automatic_tokens:
        packaged = packaged_mpi_launcher()
        if packaged is not None:
            return str(packaged)
        for name in _mpi_launcher_names(token):
            found = shutil.which(name)
            if found and _is_compatible_executable(Path(found)):
                return found
        common = _common_mpi_launcher()
        if common is not None:
            return str(common)
        raise RuntimeError(_missing_mpi_launcher_message(token))
    return _resolve_explicit_mpi_launcher(token)


def _resolve_explicit_mpi_launcher(value: str | Path) -> str:
    token = str(value).strip()
    path_like = os.sep in token or (os.altsep is not None and os.altsep in token)
    if path_like:
        path = Path(token).expanduser()
        if not _is_compatible_executable(path):
            raise RuntimeError(
                f"MPI launcher is not executable for this platform: {path}. "
                "Set PAROSOL_MPI_LAUNCHER to a compatible mpirun/mpiexec, "
                "load the cluster MPI module, or install/run a parosol-py wheel "
                "with a bundled MPI runtime."
            )
        return str(path)
    found = shutil.which(token)
    if found and _is_compatible_executable(Path(found)):
        return found
    raise RuntimeError(_missing_mpi_launcher_message(token))


def _mpi_launcher_names(token: str) -> tuple[str, ...]:
    lower = token.lower()
    if lower in {"mpirun", "mpiexec", "mpiexec.exe"}:
        return (token,)
    if sys.platform.startswith("win"):
        return ("mpiexec.exe", "mpiexec")
    return ("mpirun", "mpiexec")


def _common_mpi_launcher() -> Path | None:
    candidates = [
        Path("/usr/lib64/openmpi/bin/mpirun"),
        Path("/usr/lib64/openmpi/bin/mpiexec"),
        Path("/usr/lib/x86_64-linux-gnu/openmpi/bin/mpirun"),
        Path("/usr/lib/x86_64-linux-gnu/openmpi/bin/mpiexec"),
        Path("/opt/homebrew/bin/mpirun"),
        Path("/opt/homebrew/bin/mpiexec"),
    ]
    for candidate in candidates:
        if _is_compatible_executable(candidate):
            return candidate
    return None


def _is_compatible_executable(path: Path) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        header = path.read_bytes()[:4]
    except OSError:
        return False
    if header.startswith(b"#!"):
        return True
    if sys.platform.startswith("linux"):
        return header == b"\x7fELF"
    if sys.platform == "darwin":
        return header in {
            b"\xcf\xfa\xed\xfe",
            b"\xca\xfe\xba\xbe",
            b"\xca\xfe\xba\xbf",
            b"\xfe\xed\xfa\xcf",
        }
    if sys.platform.startswith("win"):
        return header[:2] == b"MZ"
    return True


def _missing_mpi_launcher_message(requested: str) -> str:
    return (
        f"Could not find a compatible MPI launcher for {requested!r}. "
        "Use mpi_processes: 1 for serial runs, load an MPI module so mpirun/mpiexec "
        "is on PATH, set PAROSOL_MPI_LAUNCHER=/path/to/mpirun, or install/run a "
        "parosol-py wheel that bundles a compatible MPI runtime."
    )


def mpi_runtime_environment(
    command: list[str], base_env: dict[str, str] | None = None
) -> dict[str, str] | None:
    """Return subprocess environment for bundled MPI launchers.

    OpenMPI embeds its installation prefix in help/config lookup paths. The
    bundled runtime is relocated into site-packages, so launcher subprocesses
    need an explicit prefix when using the packaged OpenMPI tree.
    """

    if not command:
        return base_env
    launcher = Path(command[0])
    msmpi_dir = _packaged_msmpi_dir_for_launcher(launcher)
    if msmpi_dir is not None:
        env = dict(os.environ if base_env is None else base_env)
        old_path = env.get("PATH")
        env["PATH"] = (
            str(msmpi_dir)
            if not old_path
            else f"{msmpi_dir}{os.pathsep}{old_path}"
        )
        return env
    openmpi_prefix = _packaged_openmpi_prefix_for_launcher(launcher)
    if openmpi_prefix is None:
        return base_env
    env = dict(os.environ if base_env is None else base_env)
    prefix_text = str(openmpi_prefix)
    bin_path = str(openmpi_prefix / "bin")
    lib_paths = [openmpi_prefix / "lib"]
    # Conda/pip installs may provide OpenMPI's secondary runtime libraries
    # (for example libhwloc) in the active environment rather than inside the
    # relocated OpenMPI tree. Keep this after the bundled lib dir so packaged
    # libraries remain preferred when present.
    env_lib = Path(sys.prefix) / "lib"
    if env_lib.exists() and env_lib not in lib_paths:
        lib_paths.append(env_lib)
    old_path = env.get("PATH")
    env["PATH"] = bin_path if not old_path else f"{bin_path}{os.pathsep}{old_path}"
    library_path = os.pathsep.join(str(path) for path in lib_paths)
    old_library_path = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        library_path
        if not old_library_path
        else f"{library_path}{os.pathsep}{old_library_path}"
    )
    env.setdefault("OPAL_PREFIX", prefix_text)
    env.setdefault("PRTE_PREFIX", prefix_text)
    env.setdefault("PMIX_PREFIX", prefix_text)
    env.setdefault("PMIX_MCA_pcompress_base_silence_warning", "1")
    _set_component_path_if_exists(
        env, "OPAL_MCA_mca_base_component_path", openmpi_prefix / "lib" / "openmpi"
    )
    _set_component_path_if_exists(
        env, "PMIX_MCA_mca_base_component_path", openmpi_prefix / "lib" / "pmix"
    )
    _set_component_path_if_exists(
        env, "PRTE_MCA_mca_base_component_path", openmpi_prefix / "lib" / "prte"
    )
    if sys.platform.startswith("linux"):
        env.setdefault("OMPI_MCA_pml", "ob1")
        env.setdefault("OMPI_MCA_btl", "self,vader,tcp")
        if (openmpi_prefix / "lib" / "openmpi" / "mca_osc_pt2pt.so").exists():
            env.setdefault("OMPI_MCA_osc", "pt2pt")
        env.setdefault("OMPI_MCA_mpi_warn_on_fork", "0")
        env.setdefault("UCX_TLS", "sm,self,tcp")
        env.setdefault("UCX_NET_DEVICES", "none")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    return env


def _set_component_path_if_exists(
    env: dict[str, str], key: str, path: Path
) -> None:
    if path.exists():
        env.setdefault(key, str(path))


def _packaged_openmpi_prefix_for_launcher(launcher: Path) -> Path | None:
    prefix = _package_bin_dir() / "openmpi"
    try:
        resolved_launcher = launcher.resolve()
    except OSError:
        return None
    try:
        resolved_launcher.relative_to(prefix.resolve())
    except ValueError:
        pass
    else:
        return prefix
    candidate = resolved_launcher.parent.parent
    if (
        candidate.name == "openmpi"
        and candidate.parent.name == "bin"
        and candidate.parent.parent.name == "parosol_py"
    ):
        return candidate
    return None


def _packaged_msmpi_dir_for_launcher(launcher: Path) -> Path | None:
    msmpi_dir = _package_bin_dir() / "msmpi"
    try:
        resolved_launcher = launcher.resolve()
    except OSError:
        return None
    try:
        resolved_launcher.relative_to(msmpi_dir.resolve())
    except ValueError:
        pass
    else:
        return msmpi_dir
    candidate = resolved_launcher.parent
    if (
        candidate.name == "msmpi"
        and candidate.parent.name == "bin"
        and candidate.parent.parent.name == "parosol_py"
    ):
        return candidate
    return None


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
    env = mpi_runtime_environment(command)
    if stream:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
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
            command, cwd=cwd, env=env, text=True, capture_output=True, check=False
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
