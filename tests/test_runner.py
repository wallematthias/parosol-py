from pathlib import Path
import os
import sys

import pytest

from parosol_py.runner import (
    _platform_executable_names,
    build_parosol_command,
    mpi_runtime_environment,
    packaged_executable,
    packaged_mpi_launcher,
    parse_run_summary,
    resolve_mpi_launcher,
    run_parosol,
)


def test_build_parosol_command_maps_outputs():
    cmd = build_parosol_command(
        executable=Path("/opt/parosol"),
        input_file=Path("/tmp/case.h5"),
        outputs=("sed", "strain", "stress"),
        tolerance=1e-7,
        level=4,
    )
    assert cmd == [
        "/opt/parosol",
        "--SED",
        "--strain",
        "--stress",
        "--tol",
        "1e-07",
        "--level",
        "4",
        "/tmp/case.h5",
    ]


def test_build_parosol_command_can_launch_with_mpi(monkeypatch):
    monkeypatch.setattr("parosol_py.runner.packaged_mpi_launcher", lambda: None)
    monkeypatch.setattr("parosol_py.runner.shutil.which", lambda name: None)
    cmd = build_parosol_command(
        executable=Path("/opt/parosol"),
        input_file=Path("/tmp/case.h5"),
        outputs=("sed",),
        tolerance=1e-6,
        level=6,
        mpi_processes=4,
        mpi_launcher="mpirun",
    )

    assert cmd == [
        "mpirun",
        "-np",
        "4",
        "/opt/parosol",
        "--SED",
        "--tol",
        "1e-06",
        "--level",
        "6",
        "/tmp/case.h5",
    ]


def test_build_parosol_command_prefers_packaged_mpi_launcher(monkeypatch, tmp_path):
    launcher = tmp_path / "bin" / "msmpi" / "mpiexec.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("fake launcher", encoding="utf-8")
    monkeypatch.setattr(
        "parosol_py.runner.packaged_mpi_launcher", lambda: launcher
    )

    cmd = build_parosol_command(
        executable=Path("/opt/parosol"),
        input_file=Path("/tmp/case.h5"),
        outputs=("sed",),
        mpi_processes=2,
    )

    assert cmd[:3] == [str(launcher), "-np", "2"]


def test_build_parosol_command_respects_explicit_mpi_launcher(monkeypatch):
    monkeypatch.setattr(
        "parosol_py.runner.packaged_mpi_launcher",
        lambda: Path("/package/mpiexec.exe"),
    )

    cmd = build_parosol_command(
        executable=Path("/opt/parosol"),
        input_file=Path("/tmp/case.h5"),
        outputs=("sed",),
        mpi_processes=2,
        mpi_launcher=Path("/cluster/mpiexec"),
    )

    assert cmd[:3] == ["/cluster/mpiexec", "-np", "2"]


def test_resolve_mpi_launcher_can_use_packaged_alias(monkeypatch):
    launcher = Path("/package/msmpi/mpiexec.exe")
    monkeypatch.setattr("parosol_py.runner.packaged_mpi_launcher", lambda: launcher)

    assert resolve_mpi_launcher("packaged") == str(launcher)


def test_mpi_runtime_environment_sets_packaged_openmpi_prefix(monkeypatch, tmp_path):
    package_bin = tmp_path / "bin"
    launcher = package_bin / "openmpi" / "bin" / "mpirun"
    launcher.parent.mkdir(parents=True)
    (package_bin / "openmpi" / "lib" / "openmpi").mkdir(parents=True)
    (package_bin / "openmpi" / "lib" / "pmix").mkdir()
    (package_bin / "openmpi" / "lib" / "prte").mkdir()
    launcher.write_text("fake launcher", encoding="utf-8")
    monkeypatch.setattr("parosol_py.runner._package_bin_dir", lambda: package_bin)
    monkeypatch.setattr("parosol_py.runner.os.geteuid", lambda: 1000)

    env = mpi_runtime_environment([str(launcher), "-np", "2"], base_env={"KEEP": "1"})

    assert env is not None
    assert env["KEEP"] == "1"
    assert env["PATH"] == str(package_bin / "openmpi" / "bin")
    assert env["LD_LIBRARY_PATH"] == str(package_bin / "openmpi" / "lib")
    assert env["OPAL_PREFIX"] == str(package_bin / "openmpi")
    assert env["PRTE_PREFIX"] == str(package_bin / "openmpi")
    assert env["PMIX_PREFIX"] == str(package_bin / "openmpi")
    assert env["OPAL_MCA_mca_base_component_path"] == str(
        package_bin / "openmpi" / "lib" / "openmpi"
    )
    assert env["PMIX_MCA_mca_base_component_path"] == str(
        package_bin / "openmpi" / "lib" / "pmix"
    )
    assert env["PRTE_MCA_mca_base_component_path"] == str(
        package_bin / "openmpi" / "lib" / "prte"
    )
    assert "OMPI_ALLOW_RUN_AS_ROOT" not in env


def test_mpi_runtime_environment_skips_missing_component_paths(
    monkeypatch, tmp_path
):
    package_bin = tmp_path / "bin"
    launcher = package_bin / "openmpi" / "bin" / "mpirun"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("fake launcher", encoding="utf-8")
    monkeypatch.setattr("parosol_py.runner._package_bin_dir", lambda: package_bin)

    env = mpi_runtime_environment([str(launcher), "-np", "2"], base_env={})

    assert env is not None
    assert "OPAL_MCA_mca_base_component_path" not in env
    assert "PMIX_MCA_mca_base_component_path" not in env
    assert "PRTE_MCA_mca_base_component_path" not in env


def test_mpi_runtime_environment_prepends_packaged_openmpi_lib(monkeypatch, tmp_path):
    package_bin = tmp_path / "bin"
    launcher = package_bin / "openmpi" / "bin" / "mpirun"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("fake launcher", encoding="utf-8")
    monkeypatch.setattr("parosol_py.runner._package_bin_dir", lambda: package_bin)

    env = mpi_runtime_environment(
        [str(launcher), "-np", "2"],
        base_env={"PATH": "/cluster/bin", "LD_LIBRARY_PATH": "/cluster/lib"},
    )

    assert env is not None
    assert env["PATH"] == f"{package_bin / 'openmpi' / 'bin'}:/cluster/bin"
    assert env["LD_LIBRARY_PATH"] == (
        f"{package_bin / 'openmpi' / 'lib'}:/cluster/lib"
    )


def test_mpi_runtime_environment_allows_packaged_openmpi_as_root(monkeypatch, tmp_path):
    package_bin = tmp_path / "bin"
    launcher = package_bin / "openmpi" / "bin" / "mpirun"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("fake launcher", encoding="utf-8")
    monkeypatch.setattr("parosol_py.runner._package_bin_dir", lambda: package_bin)
    monkeypatch.setattr("parosol_py.runner.os.geteuid", lambda: 0)

    env = mpi_runtime_environment([str(launcher), "-np", "2"], base_env={})

    assert env is not None
    assert env["OMPI_ALLOW_RUN_AS_ROOT"] == "1"
    assert env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] == "1"


def test_mpi_runtime_environment_prepends_packaged_msmpi_path(monkeypatch, tmp_path):
    package_bin = tmp_path / "bin"
    launcher = package_bin / "msmpi" / "mpiexec.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("fake launcher", encoding="utf-8")
    monkeypatch.setattr("parosol_py.runner._package_bin_dir", lambda: package_bin)

    env = mpi_runtime_environment(
        [str(launcher), "-np", "2"],
        base_env={"PATH": "C:\\System32"},
    )

    assert env is not None
    assert env["PATH"] == f"{package_bin / 'msmpi'}{os.pathsep}C:\\System32"


def test_mpi_runtime_environment_leaves_explicit_system_mpi_alone(monkeypatch, tmp_path):
    monkeypatch.setattr("parosol_py.runner._package_bin_dir", lambda: tmp_path / "bin")

    assert mpi_runtime_environment(["/cluster/mpiexec", "-np", "2"]) is None


def test_packaged_mpi_launcher_returns_none_when_not_bundled(monkeypatch, tmp_path):
    monkeypatch.setattr("parosol_py.runner._package_bin_dir", lambda: tmp_path)
    assert packaged_mpi_launcher() is None


def test_parse_run_summary_extracts_solver_metrics():
    text = """#  Nr of It: 123
#  Relative residuum: 4.5e-08
#  Absolute residuum: 2.3e-04
#  Overall:  1.25
"""
    summary = parse_run_summary(text)
    assert summary.iterations == 123
    assert summary.relative_residual == pytest.approx(4.5e-8)
    assert summary.absolute_residual == pytest.approx(2.3e-4)
    assert summary.overall_time_seconds == pytest.approx(1.25)


def test_run_parosol_can_stream_and_still_parse_summary(capsys):
    command = [
        sys.executable,
        "-c",
        "import sys; print('#  Nr of It: 7'); print('warn', file=sys.stderr)",
    ]

    result = run_parosol(command, stream=True)

    captured = capsys.readouterr()
    assert "#  Nr of It: 7" in captured.out
    assert "warn" in captured.err
    assert result.summary.iterations == 7
    assert "#  Nr of It: 7" in result.stdout
    assert "warn" in result.stderr


def test_packaged_executable_returns_path():
    assert packaged_executable().name in {"parosol", "parosol.exe"}


def test_platform_executable_names_include_windows_suffix():
    assert _platform_executable_names("parosol") == ("parosol", "parosol.exe")
