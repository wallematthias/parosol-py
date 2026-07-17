from __future__ import annotations

from pathlib import Path

import numpy as np

from parosol_py import solve
from parosol_py.nonlinear import NonlinearSolverOptions, VonMisesMaterial
from parosol_py.runner import packaged_executable


def test_native_nonlinear_cube_writes_plastic_state_and_diagnostics(tmp_path: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"

    result = solve(
        material=np.ones((3, 3, 3), dtype=np.float32) * 6829.0,
        spacing=(1.0, 1.0, 1.0),
        strain=-0.05,
        test="axial",
        load_case_type="constrained_axial",
        outputs=("von_mises", "stress", "strain", "plastic_strain"),
        nonlinear_material=VonMisesMaterial(6829.0, 0.3, 50.0),
        nonlinear_solver=NonlinearSolverOptions(
            convergence_tolerance=1.0e-6,
            maximum_plastic_iterations=50,
            plastic_convergence_window=2,
        ),
        work_dir=tmp_path / "parosol",
        tolerance=1.0e-4,
        level=2,
    )

    plastic_strain = result.fields["plastic_strain"]
    nonlinear = result.diagnostics["nonlinear"]

    assert plastic_strain.shape == (27, 6)
    assert nonlinear["plastic_iterations"] >= 1
    assert nonlinear["yielded_last"] == 27
    assert np.isfinite(nonlinear["plastic_convergence_last"])
    assert nonlinear["plastic_convergence_last"] <= 1.0e-5
