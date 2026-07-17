from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from parosol_py.nonlinear import NonlinearSolverOptions, VonMisesMaterial
from parosol_py.nonlinear_workflow import run_nonlinear_load_history


def test_run_nonlinear_load_history_records_force_displacement(tmp_path):
    result = run_nonlinear_load_history(
        material=np.ones((3, 3, 3), dtype=np.float32) * 6829.0,
        spacing=(1.0, 1.0, 1.0),
        final_strain=-0.05,
        steps=2,
        nonlinear_material=VonMisesMaterial(6829.0, 0.3, 50.0),
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=20),
        work_dir=tmp_path,
    )

    assert len(result.steps) == 2
    assert result.steps[0]["strain"] == -0.025
    assert result.steps[1]["strain"] == -0.05
    assert "generalized_load" in result.steps[1]
    assert result.steps[1]["generalized_load"]["units"] == "N"


def test_run_nonlinear_load_history_rejects_nonpositive_steps(tmp_path):
    with pytest.raises(ValueError, match="steps must be positive"):
        run_nonlinear_load_history(
            material=np.ones((3, 3, 3), dtype=np.float32) * 6829.0,
            spacing=(1.0, 1.0, 1.0),
            final_strain=-0.05,
            steps=0,
            nonlinear_material=VonMisesMaterial(6829.0, 0.3, 50.0),
            nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=20),
            work_dir=tmp_path,
        )


def test_run_nonlinear_load_history_creates_per_step_work_directories(tmp_path):
    run_nonlinear_load_history(
        material=np.ones((3, 3, 3), dtype=np.float32) * 6829.0,
        spacing=(1.0, 1.0, 1.0),
        final_strain=-0.05,
        steps=2,
        nonlinear_material=VonMisesMaterial(6829.0, 0.3, 50.0),
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=20),
        work_dir=tmp_path,
    )

    assert (tmp_path / "step_001").is_dir()
    assert (tmp_path / "step_002").is_dir()


def test_run_nonlinear_load_history_merges_outputs_and_forwards_solve_kwargs(
    monkeypatch, tmp_path
):
    calls = []

    def fake_solve(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            diagnostics={
                "mechanics": {
                    "generalized_load": {"name": "force", "value": -12.0, "units": "N"},
                    "reaction_force": {"x": 0.0, "y": 0.0, "z": -12.0},
                },
                "nonlinear": {"plastic_iterations": 4},
            }
        )

    monkeypatch.setattr("parosol_py.nonlinear_workflow.solve", fake_solve)

    result = run_nonlinear_load_history(
        material="material",
        spacing=(1.0, 1.0, 1.0),
        final_strain=-0.01,
        steps=1,
        nonlinear_material="nonlinear-material",
        nonlinear_solver="nonlinear-solver",
        work_dir=tmp_path,
        outputs=("sed", "forces"),
        tolerance=1.0e-4,
    )

    assert calls[0]["outputs"] == (
        "forces",
        "displacements",
        "von_mises",
        "plastic_strain",
        "sed",
    )
    assert calls[0]["tolerance"] == 1.0e-4
    assert result.steps[0]["reaction_force"] == {"x": 0.0, "y": 0.0, "z": -12.0}
    assert result.steps[0]["plastic_iterations"] == 4


@pytest.mark.parametrize("steps", [True, 1.5])
def test_run_nonlinear_load_history_rejects_noninteger_steps(tmp_path, steps):
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        run_nonlinear_load_history(
            material="material",
            spacing=(1.0, 1.0, 1.0),
            final_strain=-0.05,
            steps=steps,
            nonlinear_material="nonlinear-material",
            nonlinear_solver="nonlinear-solver",
            work_dir=tmp_path,
        )
