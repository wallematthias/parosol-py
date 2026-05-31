from pathlib import Path

import pytest

from parosol_py.runner import (
    build_parosol_command,
    packaged_executable,
    parse_run_summary,
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


def test_build_parosol_command_can_launch_with_mpi():
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


def test_packaged_executable_returns_path():
    assert packaged_executable().name == "parosol"
