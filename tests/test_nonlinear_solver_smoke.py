from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from parosol_py import solve
from parosol_py.boundary_conditions import axial_compression
from parosol_py.hdf5_io import write_parosol_input
from parosol_py.nonlinear import NonlinearSolverOptions, VonMisesMaterial
from parosol_py.runner import (
    build_parosol_command,
    packaged_executable,
    run_parosol,
)


def test_native_nonlinear_cube_writes_plastic_state_and_diagnostics(tmp_path: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"
    plastic_tolerance = 1.0e-6
    maximum_plastic_iterations = 50

    result = solve(
        material=np.ones((3, 3, 3), dtype=np.float32) * 6829.0,
        spacing=(1.0, 1.0, 1.0),
        strain=-0.05,
        test="axial",
        load_case_type="constrained_axial",
        outputs=("forces", "von_mises", "stress", "strain", "plastic_strain"),
        nonlinear_material=VonMisesMaterial(6829.0, 0.3, 50.0),
        nonlinear_solver=NonlinearSolverOptions(
            convergence_tolerance=plastic_tolerance,
            maximum_plastic_iterations=maximum_plastic_iterations,
            plastic_convergence_window=2,
        ),
        work_dir=tmp_path / "parosol",
        tolerance=1.0e-4,
        level=2,
    )
    linear_result = solve(
        material=np.ones((3, 3, 3), dtype=np.float32) * 6829.0,
        spacing=(1.0, 1.0, 1.0),
        strain=-0.05,
        test="axial",
        load_case_type="constrained_axial",
        outputs=("forces", "von_mises"),
        work_dir=tmp_path / "linear",
        tolerance=1.0e-4,
        level=2,
    )

    plastic_strain = result.fields["plastic_strain"]
    nonlinear = result.diagnostics["nonlinear"]
    nonlinear_load = result.diagnostics["mechanics"]["generalized_load"]["value"]
    linear_load = linear_result.diagnostics["mechanics"]["generalized_load"]["value"]
    top_indices = [
        index
        for index, coord in enumerate(_active_node_coordinates(np.ones((3, 3, 3))))
        if coord[2] == 3
    ]
    exported_nonlinear_load = float(np.sum(result.fields["forces"][top_indices, 2]))

    assert plastic_strain.shape == (27, 6)
    assert np.all(np.linalg.norm(plastic_strain, axis=1) > 0.0)
    with h5py.File(result.input_file, "r") as h5:
        gauss_plastic_strain = h5["Solution/GaussPoint8Values/PlasticStrain"]
        assert gauss_plastic_strain.shape == (27, 48)
        assert np.all(np.linalg.norm(gauss_plastic_strain[...], axis=1) > 0.0)
    assert nonlinear["plastic_iterations"] >= 1
    assert nonlinear["plastic_iterations"] < maximum_plastic_iterations
    assert 0 < nonlinear["yielded_last"] <= 27
    assert np.isfinite(nonlinear["plastic_convergence_last"])
    assert nonlinear["plastic_convergence_last"] <= plastic_tolerance
    assert nonlinear_load < 0.0
    assert abs(nonlinear_load) < abs(linear_load)
    assert not np.isclose(nonlinear_load, linear_load)
    assert exported_nonlinear_load == np.float64(nonlinear_load)
    assert result.summary.run is not None
    assert result.summary.run.iterations is not None
    assert result.summary.run.iterations > 0
    assert result.summary.run.relative_residual is not None
    assert result.summary.run.relative_residual > 0.0
    assert result.summary.run.absolute_residual is not None
    assert result.summary.run.absolute_residual > 0.0


def test_native_disabled_nonlinear_group_uses_linear_path(tmp_path: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"

    stiffness_gpa_xyz = np.ones((3, 3, 3), dtype=np.float32)
    fixed_coordinates, fixed_values = axial_compression(
        stiffness_gpa_xyz,
        axis="z",
        strain=-0.01,
        voxel_size_mm=1.0,
    )
    input_file = write_parosol_input(
        tmp_path / "disabled_nonlinear.h5",
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coordinates,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        nonlinear_material=VonMisesMaterial(1000.0, 0.3, 25.0),
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=3),
    )
    with h5py.File(input_file, "r+") as h5:
        h5["Nonlinear"].attrs["enabled"] = 0

    run = run_parosol(
        build_parosol_command(
            executable=executable,
            input_file=input_file,
            outputs=("sed",),
            tolerance=1.0e-4,
            level=2,
        ),
        cwd=tmp_path,
    )

    assert run.returncode == 0, run.stderr
    assert run.summary.iterations is not None
    assert run.summary.iterations > 0
    with h5py.File(input_file, "r") as h5:
        assert "SED" in h5["Solution"]
        assert "PlasticStrain" not in h5["Solution"]
        assert "NonlinearResults" not in h5


def test_native_nonlinear_group_without_enabled_uses_linear_path(tmp_path: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"

    stiffness_gpa_xyz = np.ones((3, 3, 3), dtype=np.float32)
    fixed_coordinates, fixed_values = axial_compression(
        stiffness_gpa_xyz,
        axis="z",
        strain=-0.01,
        voxel_size_mm=1.0,
    )
    input_file = write_parosol_input(
        tmp_path / "missing_enabled_nonlinear.h5",
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coordinates,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        nonlinear_material=VonMisesMaterial(1000.0, 0.3, 25.0),
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=3),
    )
    with h5py.File(input_file, "r+") as h5:
        del h5["Nonlinear"].attrs["enabled"]

    run = run_parosol(
        build_parosol_command(
            executable=executable,
            input_file=input_file,
            outputs=("sed",),
            tolerance=1.0e-4,
            level=2,
        ),
        cwd=tmp_path,
    )

    assert run.returncode == 0, run.stderr
    with h5py.File(input_file, "r") as h5:
        assert "SED" in h5["Solution"]
        assert "PlasticStrain" not in h5["Solution"]
        assert "NonlinearResults" not in h5


def test_native_rejects_invalid_nonlinear_hdf5_config(tmp_path: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"

    stiffness_gpa_xyz = np.ones((3, 3, 3), dtype=np.float32)
    fixed_coordinates, fixed_values = axial_compression(
        stiffness_gpa_xyz,
        axis="z",
        strain=-0.01,
        voxel_size_mm=1.0,
    )
    input_file = write_parosol_input(
        tmp_path / "invalid_nonlinear.h5",
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coordinates,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        nonlinear_material=VonMisesMaterial(1000.0, 0.3, 25.0),
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=3),
    )
    with h5py.File(input_file, "r+") as h5:
        h5["Nonlinear"].attrs["yield_strength_mpa"] = 0.0

    run = run_parosol(
        build_parosol_command(
            executable=executable,
            input_file=input_file,
            outputs=("sed",),
            tolerance=1.0e-4,
            level=2,
        ),
        cwd=tmp_path,
    )

    assert run.returncode != 0
    assert "invalid nonlinear configuration" in run.stdout + run.stderr


def _active_node_coordinates(stiffness_gpa_xyz) -> list[tuple[int, int, int]]:
    elements = np.argwhere(np.asarray(stiffness_gpa_xyz) > 0.0).astype(
        np.int64, copy=False
    )
    offsets = np.asarray(
        [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
        dtype=np.int64,
    )
    nodes = (elements[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    nodes = np.unique(nodes, axis=0)
    return [tuple(int(v) for v in row) for row in sorted(nodes, key=_morton_key)]


def _morton_key(coord) -> int:
    x, y, z = (int(v) for v in coord)
    key = 0
    bit_index = 0
    limit = max(x, y, z)
    while (1 << bit_index) <= limit:
        key |= ((x >> bit_index) & 1) << (3 * bit_index)
        key |= ((y >> bit_index) & 1) << (3 * bit_index + 1)
        key |= ((z >> bit_index) & 1) << (3 * bit_index + 2)
        bit_index += 1
    return key
