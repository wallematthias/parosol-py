import os
from pathlib import Path

import numpy as np
import pytest

from parosol_py import solve
from parosol_py.runner import packaged_executable, run_parosol


@pytest.mark.skipif(
    os.name == "nt", reason="Windows filesystem long-path support is separately configured"
)
@pytest.mark.parametrize(
    "restart_flags",
    [(), ("--startvector",), ("--startvector", "--extrapolation")],
    ids=["no-startvector", "startvector", "extrapolation"],
)
def test_native_solver_handles_long_input_and_startvector_paths(tmp_path, restart_flags):
    executable = packaged_executable()
    if not executable.exists():
        pytest.skip(f"packaged executable not found: {executable}")

    work_dir = tmp_path
    while len(str(work_dir / "parosol_input.h5").encode()) < 320:
        work_dir /= "nested-" + "x" * 40
    prepared = solve(
        material=np.full((3, 3, 3), 1000.0),
        spacing=(1.0, 1.0, 1.0),
        work_dir=work_dir,
        outputs=("sed",),
        tolerance=1e-4,
        level=2,
        dry_run=True,
    )
    command = prepared.command + list(restart_flags)
    first = run_parosol(command, cwd=work_dir)
    assert first.returncode == 0, first.stderr
    assert first.summary.relative_residual is not None
    assert first.summary.relative_residual <= 1e-4

    if restart_flags:
        startvector = Path(str(prepared.input_file) + ".sv_0")
        assert startvector.is_file()
        assert startvector.stat().st_size > 0
        second = run_parosol(command, cwd=work_dir)
        assert second.returncode == 0, second.stderr
        assert f"Startvector: Startvector read ({startvector})" in second.stdout
        if "--extrapolation" in restart_flags:
            assert "Startvector: Linear extrapolation" in second.stdout
