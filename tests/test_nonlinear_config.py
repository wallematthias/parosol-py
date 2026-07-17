from __future__ import annotations

import h5py
import numpy as np
import pytest

from parosol_py.api import solve
from parosol_py.hdf5_io import write_parosol_input
from parosol_py.nonlinear import NonlinearSolverOptions, VonMisesMaterial
from parosol_py.runner import packaged_executable


def test_von_mises_material_validates_positive_values():
    material = VonMisesMaterial(
        youngs_modulus_mpa=6829.0,
        poisson_ratio=0.3,
        yield_strength_mpa=50.0,
    )

    assert material.to_hdf5_attrs() == {
        "type": "VonMisesIsotropic",
        "youngs_modulus_mpa": 6829.0,
        "poisson_ratio": 0.3,
        "yield_strength_mpa": 50.0,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "youngs_modulus_mpa": 0.0,
            "poisson_ratio": 0.3,
            "yield_strength_mpa": 50.0,
        },
        {
            "youngs_modulus_mpa": 6829.0,
            "poisson_ratio": 0.5,
            "yield_strength_mpa": 50.0,
        },
        {
            "youngs_modulus_mpa": 6829.0,
            "poisson_ratio": 0.3,
            "yield_strength_mpa": -1.0,
        },
    ],
)
def test_von_mises_material_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        VonMisesMaterial(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in (
            "youngs_modulus_mpa",
            "poisson_ratio",
            "yield_strength_mpa",
        )
        for value in (float("-inf"), float("nan"), float("inf"))
    ],
)
def test_von_mises_material_rejects_non_finite_values(field, value):
    kwargs = {
        "youngs_modulus_mpa": 6829.0,
        "poisson_ratio": 0.3,
        "yield_strength_mpa": 50.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        VonMisesMaterial(**kwargs)


@pytest.mark.parametrize("value", [float("-inf"), float("nan"), float("inf")])
def test_nonlinear_solver_options_rejects_non_finite_tolerance(value):
    with pytest.raises(ValueError):
        NonlinearSolverOptions(convergence_tolerance=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in ("maximum_plastic_iterations", "plastic_convergence_window")
        for value in (float("-inf"), float("nan"), float("inf"), 1.0, 1.5, True)
    ],
)
def test_nonlinear_solver_options_rejects_non_integer_iteration_values(field, value):
    kwargs = {field: value}

    with pytest.raises(ValueError):
        NonlinearSolverOptions(**kwargs)


def test_write_parosol_input_writes_optional_nonlinear_group(tmp_path):
    path = tmp_path / "case.h5"
    stiffness = np.ones((2, 2, 2), dtype=np.float32)
    fixed_coords = np.array([[0, 0, 0, 0]], dtype=np.uint16)
    fixed_values = np.array([0.0], dtype=np.float32)

    write_parosol_input(
        path,
        stiffness_gpa_xyz=stiffness,
        fixed_displacement_coordinates=fixed_coords,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        nonlinear_material=VonMisesMaterial(
            youngs_modulus_mpa=1000.0,
            poisson_ratio=0.3,
            yield_strength_mpa=25.0,
        ),
        nonlinear_solver=NonlinearSolverOptions(
            convergence_tolerance=1.0e-6,
            maximum_plastic_iterations=20,
            plastic_convergence_window=2,
        ),
    )

    with h5py.File(path, "r") as h5:
        group = h5["Nonlinear"]
        assert group.attrs["enabled"] == 1
        assert group.attrs["material_type"] == "VonMisesIsotropic"
        assert group.attrs["youngs_modulus_mpa"] == pytest.approx(1000.0)
        assert group.attrs["poisson_ratio"] == pytest.approx(0.3)
        assert group.attrs["yield_strength_mpa"] == pytest.approx(25.0)
        assert group.attrs["convergence_tolerance"] == pytest.approx(1.0e-6)
        assert group.attrs["maximum_plastic_iterations"] == 20
        assert group.attrs["plastic_convergence_window"] == 2


def test_write_parosol_input_rejects_solver_without_material(tmp_path):
    with pytest.raises(ValueError, match="nonlinear_solver requires nonlinear_material"):
        write_parosol_input(
            tmp_path / "case.h5",
            stiffness_gpa_xyz=np.ones((2, 2, 2), dtype=np.float32),
            fixed_displacement_coordinates=np.array([[0, 0, 0, 0]], dtype=np.uint16),
            fixed_displacement_values=np.array([0.0], dtype=np.float32),
            voxel_size_mm=1.0,
            poisson_ratio=0.3,
            nonlinear_solver=NonlinearSolverOptions(),
        )


def test_solve_dry_run_writes_nonlinear_configuration(tmp_path):
    material = VonMisesMaterial(
        youngs_modulus_mpa=1000.0,
        poisson_ratio=0.3,
        yield_strength_mpa=25.0,
    )
    solver = NonlinearSolverOptions(maximum_plastic_iterations=20)

    result = solve(
        material=np.ones((2, 2, 2), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        nonlinear_material=material,
        nonlinear_solver=solver,
        executable="parosol",
        work_dir=tmp_path,
        dry_run=True,
    )

    with h5py.File(result.input_file, "r") as h5:
        group = h5["Nonlinear"]
        assert group.attrs["material_type"] == "VonMisesIsotropic"
        assert group.attrs["maximum_plastic_iterations"] == 20


def test_nonlinear_dry_run_builds_command_without_running_solver(tmp_path):
    result = solve(
        material=np.ones((2, 2, 2), dtype=np.float32) * 1000.0,
        spacing=(1.0, 1.0, 1.0),
        nonlinear_material=VonMisesMaterial(1000.0, 0.3, 25.0),
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=3),
        work_dir=tmp_path,
        dry_run=True,
        executable=packaged_executable(),
    )

    assert result.input_file.name == "parosol_input.h5"
    with h5py.File(result.input_file, "r") as h5:
        assert h5["Nonlinear"].attrs["maximum_plastic_iterations"] == 3


def test_material_only_nonlinear_input_runs_native_reader(tmp_path):
    result = solve(
        material=np.ones((3, 3, 3), dtype=np.float32) * 1000.0,
        spacing=(1.0, 1.0, 1.0),
        test="axial",
        test_axis="z",
        strain=-0.01,
        outputs=("sed",),
        nonlinear_material=VonMisesMaterial(1000.0, 0.3, 25.0),
        executable=packaged_executable(),
        work_dir=tmp_path,
        tolerance=1.0e-4,
        level=2,
    )

    with h5py.File(result.input_file, "r") as h5:
        group = h5["Nonlinear"]
        assert "convergence_tolerance" not in group.attrs
        assert "maximum_plastic_iterations" not in group.attrs
        assert "plastic_convergence_window" not in group.attrs
    assert "sed" in result.fields
    assert result.summary.run is not None
