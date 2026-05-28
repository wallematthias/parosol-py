from pathlib import Path

import numpy as np
import pytest

from parosol_py import solve
from parosol_py.runner import packaged_executable


def test_real_solver_smoke_sed(tmp_path: Path):
    executable = packaged_executable()
    if not executable.exists():
        pytest.skip(f"packaged executable not found: {executable}")

    material_zyx = np.ones((3, 3, 3)) * 1000.0

    result = solve(
        material=material_zyx,
        spacing=(1, 1, 1),
        test="axial",
        test_axis="z",
        strain=-0.01,
        outputs=("sed",),
        work_dir=tmp_path,
        tolerance=1e-4,
        level=2,
    )

    assert "sed" in result.fields
    assert result.summary.run is not None
    assert result.summary.run.iterations is None or result.summary.run.iterations > 0
