from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_stage_module():
    spec = importlib.util.spec_from_file_location(
        "stage_mpi_runtime", ROOT / "scripts" / "stage_mpi_runtime.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_msmpi_copies_runtime_and_notice(monkeypatch, tmp_path):
    stage = _load_stage_module()
    source_bin = tmp_path / "Microsoft MPI" / "Bin"
    source_bin.mkdir(parents=True)
    for filename in (
        "mpiexec.exe",
        "smpd.exe",
        "msmpi.dll",
        "msmpires.dll",
        "msmpilaunchsvc.exe",
    ):
        (source_bin / filename).write_text(filename, encoding="utf-8")
    for filename in (
        "MicrosoftMPI_Redistributable_EULA.rtf",
        "MPI_Redistributables_TPN.txt",
    ):
        (source_bin.parent / filename).write_text(filename, encoding="utf-8")

    monkeypatch.setattr(stage, "PACKAGE_BIN", tmp_path / "package" / "bin")
    monkeypatch.setenv("MSMPI_BIN", str(source_bin))

    stage._stage_msmpi()

    dest = tmp_path / "package" / "bin" / "msmpi"
    assert (dest / "mpiexec.exe").is_file()
    assert (dest / "msmpi.dll").is_file()
    assert (dest / "MicrosoftMPI_Redistributable_EULA.rtf").is_file()
    notice = (dest / "NOTICE.txt").read_text(encoding="utf-8")
    assert "Microsoft MPI remains licensed under its own Microsoft/MIT terms" in notice
    assert "not relicensed as GPL" in notice


def test_stage_msmpi_allows_dlls_in_windows_system_directory(monkeypatch, tmp_path):
    stage = _load_stage_module()
    program_files = tmp_path / "Program Files"
    source_bin = program_files / "Microsoft MPI" / "Bin"
    system32 = tmp_path / "Windows" / "System32"
    source_bin.mkdir(parents=True)
    system32.mkdir(parents=True)
    for filename in ("mpiexec.exe", "smpd.exe", "msmpilaunchsvc.exe"):
        (source_bin / filename).write_text(filename, encoding="utf-8")
    for filename in ("msmpi.dll", "msmpires.dll"):
        (system32 / filename).write_text(filename, encoding="utf-8")
    for filename in (
        "MicrosoftMPI_Redistributable_EULA.rtf",
        "MPI_Redistributables_TPN.txt",
    ):
        (source_bin.parent / filename).write_text(filename, encoding="utf-8")

    monkeypatch.setattr(stage, "PACKAGE_BIN", tmp_path / "package" / "bin")
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))

    stage._stage_msmpi()

    dest = tmp_path / "package" / "bin" / "msmpi"
    assert (dest / "mpiexec.exe").read_text(encoding="utf-8") == "mpiexec.exe"
    assert (dest / "msmpi.dll").read_text(encoding="utf-8") == "msmpi.dll"
    assert (dest / "msmpires.dll").read_text(encoding="utf-8") == "msmpires.dll"


def test_stage_windows_native_dlls_copies_vcpkg_runtime(monkeypatch, tmp_path):
    stage = _load_stage_module()
    installed = tmp_path / "vcpkg" / "installed"
    vcpkg_bin = installed / "x64-windows" / "bin"
    vcpkg_bin.mkdir(parents=True)
    (vcpkg_bin / "hdf5.dll").write_text("hdf5", encoding="utf-8")
    (vcpkg_bin / "zlib1.dll").write_text("zlib", encoding="utf-8")
    (vcpkg_bin / "ignore.txt").write_text("not a dll", encoding="utf-8")

    monkeypatch.setattr(stage, "PACKAGE_BIN", tmp_path / "package" / "bin")
    monkeypatch.setenv("VCPKG_INSTALLED_DIR", str(installed))
    monkeypatch.setenv("VCPKG_TARGET_TRIPLET", "x64-windows")

    stage._stage_windows_native_dlls()

    assert (tmp_path / "package" / "bin" / "hdf5.dll").is_file()
    assert (tmp_path / "package" / "bin" / "zlib1.dll").is_file()
    assert not (tmp_path / "package" / "bin" / "ignore.txt").exists()


def test_stage_openmpi_copies_launcher_libraries_and_notice(monkeypatch, tmp_path):
    stage = _load_stage_module()
    prefix = tmp_path / "openmpi"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    (prefix / "lib" / "openmpi").mkdir()
    (prefix.parent / "pmix").mkdir()
    (prefix.parent / "prte").mkdir()
    (prefix / "share" / "openmpi").mkdir(parents=True)
    (prefix / "share" / "pmix").mkdir(parents=True)
    (prefix / "share" / "prte").mkdir(parents=True)
    (prefix / "share" / "doc" / "openmpi").mkdir(parents=True)
    for filename in ("mpirun", "mpiexec", "prterun"):
        path = prefix / "bin" / filename
        path.write_text(f"#!/bin/sh\nexec echo {filename}\n", encoding="utf-8")
        path.chmod(0o755)
    (prefix / "lib" / "libmpi.so").write_text("mpi", encoding="utf-8")
    (prefix / "lib" / "libpmix.so").write_text("pmix", encoding="utf-8")
    (prefix.parent / "libevent_core.so.2").write_text("event", encoding="utf-8")
    (prefix.parent / "libhwloc.so.15").write_text("hwloc", encoding="utf-8")
    (prefix.parent / "libpmix.so.2").write_text("rpm pmix", encoding="utf-8")
    (prefix.parent / "libprrte.so.3").write_text("rpm prrte", encoding="utf-8")
    (prefix / "lib" / "openmpi" / "mca_pml_ob1.so").write_text(
        "openmpi plugin", encoding="utf-8"
    )
    (prefix.parent / "pmix" / "mca_bfrops_v12.so").write_text(
        "pmix plugin", encoding="utf-8"
    )
    (prefix.parent / "prte" / "mca_odls_default.so").write_text(
        "prte plugin", encoding="utf-8"
    )
    (prefix / "share" / "openmpi" / "help.txt").write_text("help", encoding="utf-8")
    (prefix / "share" / "pmix" / "help-pmix.txt").write_text("pmix", encoding="utf-8")
    (prefix / "share" / "prte" / "help-prte.txt").write_text("prte", encoding="utf-8")
    (prefix / "share" / "doc" / "openmpi" / "LICENSE").write_text(
        "OpenMPI license", encoding="utf-8"
    )

    monkeypatch.setattr(stage, "PACKAGE_BIN", tmp_path / "package" / "bin")
    monkeypatch.setattr(stage, "_openmpi_prefix", lambda: prefix)
    monkeypatch.setattr(stage.shutil, "which", lambda name: None)

    stage._stage_openmpi()

    dest = tmp_path / "package" / "bin" / "openmpi"
    assert (dest / "bin" / "mpirun").is_file()
    assert (dest / "lib" / "libmpi.so").is_file()
    assert (dest / "lib" / "libevent_core.so.2").is_file()
    assert (dest / "lib" / "libhwloc.so.15").is_file()
    assert (dest / "lib" / "libpmix.so.2").is_file()
    assert (dest / "lib" / "libprrte.so.3").is_file()
    assert (dest / "lib" / "openmpi" / "mca_pml_ob1.so").is_file()
    assert (dest / "lib" / "pmix" / "mca_bfrops_v12.so").is_file()
    assert (dest / "lib" / "prte" / "mca_odls_default.so").is_file()
    assert (dest / "share" / "openmpi" / "help.txt").is_file()
    assert (dest / "share" / "pmix" / "help-pmix.txt").is_file()
    assert (dest / "share" / "prte" / "help-prte.txt").is_file()
    assert any((dest / "licenses").iterdir())
    notice = (dest / "NOTICE.txt").read_text(encoding="utf-8")
    assert "OpenMPI remains licensed under its own BSD-style" in notice
    assert "not relicensed as GPL" in notice


def test_stage_openmpi_rejects_wrong_format_launcher(monkeypatch, tmp_path):
    stage = _load_stage_module()
    prefix = tmp_path / "openmpi"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    launcher = prefix / "bin" / "mpirun"
    launcher.write_text("wrong-platform-launcher", encoding="utf-8")
    launcher.chmod(0o755)
    (prefix / "lib" / "libmpi.so").write_text("mpi", encoding="utf-8")

    monkeypatch.setattr(stage, "PACKAGE_BIN", tmp_path / "package" / "bin")
    monkeypatch.setattr(stage, "_openmpi_prefix", lambda: prefix)
    monkeypatch.setattr(stage.shutil, "which", lambda name: None)

    try:
        stage._stage_openmpi()
    except SystemExit as exc:
        assert "compatible OpenMPI launcher" in str(exc)
    else:
        raise AssertionError("wrong-format OpenMPI launcher was staged")


def test_stage_openmpi_copies_prefixed_runtime_dependencies(monkeypatch, tmp_path):
    stage = _load_stage_module()
    prefix = tmp_path / "openmpi"
    dest = tmp_path / "package" / "bin" / "openmpi"
    (prefix / "lib").mkdir(parents=True)
    (dest / "bin").mkdir(parents=True)
    (dest / "lib").mkdir(parents=True)
    (prefix / "lib" / "libxml2.16.dylib").write_text("xml", encoding="utf-8")
    (dest / "bin" / "mpirun").write_text("mpirun", encoding="utf-8")
    (dest / "lib" / "libhwloc.15.dylib").write_text("hwloc", encoding="utf-8")

    def fake_run(args, **kwargs):
        assert args[0] == "otool"
        stdout = "\n".join(
            [
                f"{args[-1]}:",
                "\t@rpath/libxml2.16.dylib (compatibility version 17.0.0, current version 17.6.0)",
                "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1292.0.0)",
            ]
        )
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(stage.sys, "platform", "darwin")
    monkeypatch.setattr(stage.subprocess, "run", fake_run)

    stage._copy_openmpi_runtime_dependencies(prefix, dest)

    assert (dest / "lib" / "libxml2.16.dylib").read_text(encoding="utf-8") == "xml"


def test_runtime_dependency_copy_preserves_soname_symlinks(tmp_path):
    stage = _load_stage_module()
    source_dir = tmp_path / "prefix" / "lib"
    dest = tmp_path / "package" / "lib"
    source_dir.mkdir(parents=True)
    dest.mkdir(parents=True)
    real = source_dir / "libxml2.so.16.1.3"
    real.write_text("xml", encoding="utf-8")
    (source_dir / "libxml2.so.16").symlink_to(real.name)
    (source_dir / "libxml2.so").symlink_to(real.name)

    copied = stage._copy_runtime_dependency_with_links(real, dest, set())

    assert {"libxml2.so", "libxml2.so.16", "libxml2.so.16.1.3"} <= copied
    assert (dest / "libxml2.so.16").is_symlink()
    assert (dest / "libxml2.so.16").resolve() == (dest / "libxml2.so.16.1.3")
