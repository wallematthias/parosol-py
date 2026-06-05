from pathlib import Path
import sys

import pytest

from parosol_py.runner import (
    _platform_executable_names,
    build_parosol_command,
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
