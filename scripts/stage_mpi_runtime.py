from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_BIN = ROOT / "src" / "parosol_py" / "bin"
MSMPI_URL = (
    "https://download.microsoft.com/download/7/2/7/"
    "72731ebb-b63c-4170-ade7-836966263a8f/msmpisetup.exe"
)


def main() -> int:
    for name in ("msmpi", "openmpi"):
        shutil.rmtree(PACKAGE_BIN / name, ignore_errors=True)
    if sys.platform.startswith("win"):
        _stage_msmpi()
    elif sys.platform in {"darwin", "linux"}:
        _stage_openmpi()
    else:
        print(f"Skipping bundled MPI runtime for unsupported platform: {sys.platform}")
    return 0


def _stage_msmpi() -> None:
    dest = PACKAGE_BIN / "msmpi"
    dest.mkdir(parents=True, exist_ok=True)
    for filename in (
        "mpiexec.exe",
        "smpd.exe",
        "msmpi.dll",
        "msmpires.dll",
        "msmpilaunchsvc.exe",
    ):
        source = _find_msmpi_runtime_file(filename)
        if source is None:
            raise SystemExit(
                f"Required MS-MPI runtime file was not found: {filename}. "
                "Install Microsoft MPI before building Windows wheels."
            )
        shutil.copy2(source, dest / filename)

    _write_msmpi_notice(dest / "NOTICE.txt")
    _copy_msmpi_license_files(dest)


def _msmpi_search_roots() -> list[Path]:
    roots = []
    if os.environ.get("MSMPI_BIN"):
        bin_root = Path(os.environ["MSMPI_BIN"])
        roots.extend([bin_root, bin_root.parent, bin_root.parent / "License"])
    for root_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(root_name)
        if root:
            install_root = Path(root) / "Microsoft MPI"
            roots.extend([install_root / "Bin", install_root, install_root / "License"])
    system_root = os.environ.get("SystemRoot") or os.environ.get("windir")
    if system_root:
        roots.extend([Path(system_root) / "System32", Path(system_root) / "SysWOW64"])
    return _unique_existing_paths(roots)


def _find_msmpi_runtime_file(filename: str) -> Path | None:
    for root in _msmpi_search_roots():
        candidate = root / filename
        if candidate.exists():
            return candidate
    found = shutil.which(filename)
    if found:
        return Path(found).resolve()
    return None


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            continue
        unique.append(resolved)
        seen.add(resolved)
    return unique


def _copy_msmpi_license_files(dest: Path) -> None:
    install_roots = [dest.parent]
    install_roots.extend(_msmpi_search_roots())
    copied = False
    for root in install_roots:
        for filename in ("MicrosoftMPI_Redistributable_EULA.rtf", "MPI_Redistributables_TPN.txt"):
            source = root / filename
            if source.exists():
                shutil.copy2(source, dest / filename)
                copied = True
    if copied:
        return

    installer = dest / "msmpisetup.exe"
    _download(MSMPI_URL, installer)
    seven_zip = shutil.which("7z") or shutil.which("7zz")
    if not seven_zip:
        raise SystemExit(
            "Could not copy MS-MPI license files and 7z is unavailable to extract "
            "them from msmpisetup.exe."
        )
    subprocess.run(
        [
            seven_zip,
            "e",
            "-y",
            f"-o{dest}",
            str(installer),
            "MicrosoftMPI_Redistributable_EULA.rtf",
            "MPI_Redistributables_TPN.txt",
        ],
        check=True,
    )
    installer.unlink(missing_ok=True)


def _write_msmpi_notice(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Bundled MPI runtime notice",
                "",
                "This wheel includes Microsoft MPI runtime files for Windows so",
                "ParOSol can launch multi-process solves after a single pip install.",
                "Microsoft MPI remains licensed under its own Microsoft/MIT terms",
                "and third-party notices; it is not relicensed as GPL by parosol-py.",
                "",
                "Included runtime files are staged from a Microsoft MPI installation",
                "or the official Microsoft MPI redistributable installer.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _stage_openmpi() -> None:
    prefix = _openmpi_prefix()
    dest = PACKAGE_BIN / "openmpi"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("mpirun", "mpiexec", "prterun", "prte", "orted", "ompi_info"):
        source = _find_in_prefix_or_path(prefix, "bin", name)
        if source is not None:
            target = dest / "bin" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            target.chmod(target.stat().st_mode | 0o755)
    if not any((dest / "bin" / name).exists() for name in ("mpirun", "mpiexec")):
        raise SystemExit("OpenMPI launcher was not found; cannot stage MPI runtime.")

    _copy_openmpi_libraries(prefix, dest / "lib")
    _copy_openmpi_runtime_dependencies(prefix, dest)
    _copy_openmpi_config(prefix / "etc", dest / "etc")
    _copy_optional_tree(prefix / "share" / "openmpi", dest / "share" / "openmpi")
    _copy_openmpi_licenses(prefix, dest)
    _write_openmpi_notice(dest / "NOTICE.txt")


def _openmpi_prefix() -> Path:
    override = os.environ.get("PAROSOL_OPENMPI_PREFIX")
    if override:
        return Path(override).resolve()
    ompi_info = shutil.which("ompi_info")
    if ompi_info:
        proc = subprocess.run(
            [ompi_info, "--parsable", "--path", "prefix"],
            text=True,
            capture_output=True,
            check=False,
        )
        for line in proc.stdout.splitlines():
            if "path:prefix:" in line:
                return Path(line.rsplit(":", 1)[-1]).resolve()
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher:
        return Path(launcher).resolve().parent.parent
    raise SystemExit("OpenMPI prefix could not be detected.")


def _find_in_prefix_or_path(prefix: Path, subdir: str, name: str) -> Path | None:
    candidate = prefix / subdir / name
    if candidate.exists():
        return candidate
    found = shutil.which(name)
    return Path(found).resolve() if found else None


def _copy_openmpi_libraries(prefix: Path, dest: Path) -> None:
    lib = prefix / "lib"
    if not lib.exists():
        raise SystemExit(f"OpenMPI lib directory not found: {lib}")
    dest.mkdir(parents=True, exist_ok=True)
    patterns = (
        "libmpi*",
        "libopen-*",
        "libpmix*",
        "libprrte*",
        "libevent*",
        "libhwloc*",
    )
    copied = 0
    for pattern in patterns:
        for source in lib.glob(pattern):
            if source.is_file() or source.is_symlink():
                shutil.copy2(source, dest / source.name, follow_symlinks=False)
                copied += 1
    for child in ("openmpi", "pmix", "prte"):
        _copy_optional_tree(lib / child, dest / child)
    if copied == 0:
        raise SystemExit(f"No OpenMPI libraries found in {lib}")


def _copy_openmpi_runtime_dependencies(prefix: Path, dest: Path) -> None:
    """Copy MPI runtime dependencies that live inside the detected prefix."""

    lib_dest = dest / "lib"
    seen = {path.name for path in _openmpi_runtime_files(dest)}
    changed = True
    while changed:
        changed = False
        for path in list(_openmpi_runtime_files(dest)):
            for dependency in _runtime_dependencies(path):
                source = _dependency_source(prefix, dependency)
                if source is None or source.name in seen:
                    continue
                shutil.copy2(source, lib_dest / source.name)
                seen.add(source.name)
                changed = True


def _openmpi_runtime_files(dest: Path) -> list[Path]:
    files: list[Path] = []
    for root in (dest / "bin", dest / "lib"):
        if not root.exists():
            continue
        files.extend(path for path in root.iterdir() if path.is_file() or path.is_symlink())
    return files


def _runtime_dependencies(path: Path) -> list[str]:
    if sys.platform == "darwin":
        proc = subprocess.run(
            ["otool", "-L", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return []
        dependencies: list[str] = []
        for line in proc.stdout.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            dependencies.append(line.split(" ", 1)[0])
        return dependencies

    if sys.platform.startswith("linux"):
        proc = subprocess.run(
            ["ldd", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return []
        dependencies = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if "=>" in line:
                candidate = line.split("=>", 1)[1].strip().split(" ", 1)[0]
            else:
                candidate = line.split(" ", 1)[0]
            if candidate.startswith("/"):
                dependencies.append(candidate)
        return dependencies

    return []


def _dependency_source(prefix: Path, dependency: str) -> Path | None:
    if dependency.startswith(("/usr/lib/", "/System/Library/")):
        return None
    if dependency.startswith("@"):
        candidate = prefix / "lib" / Path(dependency).name
    else:
        candidate = Path(dependency)
        try:
            candidate.relative_to(prefix)
        except ValueError:
            return None
    if candidate.exists():
        return candidate.resolve()
    return None


def _copy_openmpi_config(source: Path, dest: Path) -> None:
    if not source.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "openmpi*",
        "pmix*",
        "prte*",
    ):
        for path in source.glob(pattern):
            if path.is_file() or path.is_symlink():
                shutil.copy2(path, dest / path.name, follow_symlinks=False)


def _copy_openmpi_licenses(prefix: Path, dest: Path) -> None:
    license_dir = dest / "licenses"
    license_dir.mkdir(parents=True, exist_ok=True)
    metadata = _conda_runtime_license_metadata(prefix)
    if metadata:
        lines = ["Conda package license metadata for bundled MPI runtime files:", ""]
        for name, license_text in metadata:
            lines.append(f"- {name}: {license_text}")
        lines.extend(
            [
                "",
                "OpenMPI is distributed under the 3-clause BSD license; see",
                "https://www.open-mpi.org/community/license.php.",
            ]
        )
        (license_dir / "OpenMPI_LICENSE_NOTE.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        return

    (license_dir / "OpenMPI_LICENSE_NOTE.txt").write_text(
        "OpenMPI runtime files are bundled from the wheel build environment. "
        "OpenMPI is distributed under the 3-clause BSD license; see "
        "https://www.open-mpi.org/community/license.php.\n",
        encoding="utf-8",
    )


def _conda_runtime_license_metadata(prefix: Path) -> list[tuple[str, str]]:
    conda_meta = prefix / "conda-meta"
    if not conda_meta.exists():
        return []
    package_prefixes = (
        "openmpi-",
        "libevent-",
        "libhwloc-",
        "hwloc-",
        "libpmix-",
        "pmix-",
        "libprrte-",
        "prrte-",
    )
    metadata: list[tuple[str, str]] = []
    for path in sorted(conda_meta.glob("*.json")):
        if not path.name.startswith(package_prefixes):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        license_text = str(data.get("license", "unknown"))
        metadata.append((str(data.get("name", path.stem)), license_text))
    return metadata


def _write_openmpi_notice(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Bundled MPI runtime notice",
                "",
                "This wheel includes OpenMPI runtime files for macOS/Linux so",
                "ParOSol can launch multi-process solves after a single pip install.",
                "OpenMPI remains licensed under its own BSD-style and third-party",
                "notices; it is not relicensed as GPL by parosol-py.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_optional_tree(source: Path, dest: Path) -> None:
    if source.exists():
        shutil.copytree(source, dest, dirs_exist_ok=True, symlinks=True)


def _copy_required(source: Path, dest: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Required MPI runtime file not found: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _download(url: str, path: Path) -> None:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


if __name__ == "__main__":
    raise SystemExit(main())
