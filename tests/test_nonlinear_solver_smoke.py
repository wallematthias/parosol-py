from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from parosol_py import solve
from parosol_py.boundary_conditions import axial_compression
from parosol_py.hdf5_io import write_parosol_input
from parosol_py.nonlinear import (
    KeavenyNonlinearMaterialMap,
    NonlinearSolverOptions,
    VonMisesMaterial,
    spine_keaveny_nonlinear,
)
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


def test_asymmetric_density_map_requires_all_datasets(tmp_path: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"

    rho_qct = np.ones((3, 3, 3), dtype=np.float64)
    nonlinear_map = spine_keaveny_nonlinear(rho_qct)
    stiffness_gpa_xyz = (nonlinear_map.youngs_modulus_mpa / 1000.0).astype(
        np.float32
    )
    fixed_coordinates, fixed_values = axial_compression(
        stiffness_gpa_xyz,
        axis="z",
        strain=-0.01,
        voxel_size_mm=1.0,
    )
    input_file = write_parosol_input(
        tmp_path / "missing_tensile_yield.h5",
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coordinates,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        nonlinear_material=nonlinear_map,
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=3),
    )
    with h5py.File(input_file, "r+") as h5:
        del h5["Nonlinear"]["TensileYieldStressMPa"]

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

    output = run.stdout + run.stderr
    assert run.returncode != 0
    assert "missing TensileYieldStressMPa" in output


def test_asymmetric_density_map_rejects_rank_two_dataset(tmp_path: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"

    rho_qct = np.ones((3, 3, 3), dtype=np.float64)
    nonlinear_map = spine_keaveny_nonlinear(rho_qct)
    stiffness_gpa_xyz = (nonlinear_map.youngs_modulus_mpa / 1000.0).astype(
        np.float32
    )
    fixed_coordinates, fixed_values = axial_compression(
        stiffness_gpa_xyz,
        axis="z",
        strain=-0.01,
        voxel_size_mm=1.0,
    )
    input_file = write_parosol_input(
        tmp_path / "rank_two_tensile_yield.h5",
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coordinates,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        nonlinear_material=nonlinear_map,
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=3),
    )
    with h5py.File(input_file, "r+") as h5:
        del h5["Nonlinear"]["TensileYieldStressMPa"]
        h5["Nonlinear"].create_dataset(
            "TensileYieldStressMPa",
            data=np.ones((3, 3), dtype=np.float64),
        )

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

    output = run.stdout + run.stderr
    assert run.returncode != 0
    assert "invalid nonlinear configuration" in output
    assert "TensileYieldStressMPa rank must be 3" in output
    assert "only VonMisesIsotropic nonlinear material" not in output


def test_asymmetric_density_map_rejects_rank_four_dataset(tmp_path: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"

    rho_qct = np.ones((3, 3, 3), dtype=np.float64)
    nonlinear_map = spine_keaveny_nonlinear(rho_qct)
    stiffness_gpa_xyz = (nonlinear_map.youngs_modulus_mpa / 1000.0).astype(
        np.float32
    )
    fixed_coordinates, fixed_values = axial_compression(
        stiffness_gpa_xyz,
        axis="z",
        strain=-0.01,
        voxel_size_mm=1.0,
    )
    input_file = write_parosol_input(
        tmp_path / "rank_four_tensile_yield.h5",
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coordinates,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        nonlinear_material=nonlinear_map,
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=3),
    )
    with h5py.File(input_file, "r+") as h5:
        del h5["Nonlinear"]["TensileYieldStressMPa"]
        h5["Nonlinear"].create_dataset(
            "TensileYieldStressMPa",
            data=np.ones((3, 3, 3, 1), dtype=np.float64),
        )

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

    output = run.stdout + run.stderr
    assert run.returncode != 0
    assert "invalid nonlinear configuration" in output
    assert "TensileYieldStressMPa rank must be 3" in output
    assert "only VonMisesIsotropic nonlinear material" not in output


def test_asymmetric_density_map_reports_dataset_read_failure(tmp_path: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"

    rho_qct = np.ones((3, 3, 3), dtype=np.float64)
    nonlinear_map = spine_keaveny_nonlinear(rho_qct)
    stiffness_gpa_xyz = (nonlinear_map.youngs_modulus_mpa / 1000.0).astype(
        np.float32
    )
    fixed_coordinates, fixed_values = axial_compression(
        stiffness_gpa_xyz,
        axis="z",
        strain=-0.01,
        voxel_size_mm=1.0,
    )
    input_file = write_parosol_input(
        tmp_path / "unreadable_material_id.h5",
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coordinates,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        nonlinear_material=nonlinear_map,
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=3),
    )
    with h5py.File(input_file, "r+") as h5:
        del h5["Nonlinear"]["MaterialID"]
        h5["Nonlinear"].create_dataset(
            "MaterialID",
            data=np.full((3, 3, 3), b"bad", dtype="S3"),
        )

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

    output = run.stdout + run.stderr
    assert run.returncode != 0
    assert "invalid nonlinear configuration" in output
    assert "failed to read MaterialID" in output
    assert "only VonMisesIsotropic nonlinear material" not in output


def test_asymmetric_density_map_rejects_E_image_stiffness_mismatch(tmp_path: Path):
    input_file = _write_valid_asymmetric_input(tmp_path, "mismatched_stiffness.h5")
    with h5py.File(input_file, "r+") as h5:
        h5["Nonlinear"]["YoungsModulusMPa"][1, 1, 1] = 2000.0

    run = _run_native_sed(input_file, tmp_path)

    output = run.stdout + run.stderr
    assert run.returncode != 0
    assert "invalid nonlinear configuration" in output
    assert "YoungsModulusMPa must match Image stiffness" in output


def test_asymmetric_density_map_rejects_invalid_active_poisson_ratio(tmp_path: Path):
    input_file = _write_valid_asymmetric_input(tmp_path, "invalid_nu.h5")
    with h5py.File(input_file, "r+") as h5:
        h5["Nonlinear"]["PoissonRatio"][1, 1, 1] = 0.5

    run = _run_native_sed(input_file, tmp_path)

    output = run.stdout + run.stderr
    assert run.returncode != 0
    assert "invalid nonlinear configuration" in output
    assert "PoissonRatio values must satisfy -1 < nu < 0.5" in output


def test_asymmetric_density_map_rejects_nonpositive_active_yield_and_plateau(
    tmp_path: Path,
):
    invalid_fields = [
        (
            "CompressiveYieldStressMPa",
            "CompressiveYieldStressMPa values must be finite and positive",
        ),
        (
            "TensileYieldStressMPa",
            "TensileYieldStressMPa values must be finite and positive",
        ),
        (
            "PlateauStressMPa",
            "PlateauStressMPa values must be finite and positive",
        ),
    ]
    for dataset_name, expected_error in invalid_fields:
        input_file = _write_valid_asymmetric_input(
            tmp_path,
            f"invalid_{dataset_name}.h5",
        )
        with h5py.File(input_file, "r+") as h5:
            h5["Nonlinear"][dataset_name][1, 1, 1] = 0.0

        run = _run_native_sed(input_file, tmp_path)

        output = run.stdout + run.stderr
        assert run.returncode != 0
        assert "invalid nonlinear configuration" in output
        assert expected_error in output


def test_asymmetric_density_map_accepts_plateau_distinct_from_compressive_yield(
    tmp_path: Path,
):
    material_map = _constant_asymmetric_map(
        shape=(3, 3, 3),
        youngs_mpa=1000.0,
        poisson_ratio=0.3,
        tensile_yield_mpa=50.0,
        compressive_yield_mpa=5.0,
        plateau_mpa=20.0,
    )
    stiffness_gpa = (material_map.youngs_modulus_mpa / 1000.0).astype(np.float32)

    result = solve(
        material=stiffness_gpa,
        material_unit="GPa",
        spacing=(1.0, 1.0, 1.0),
        array_order="xyz",
        strain=-0.008,
        test="axial",
        load_case_type="constrained_axial",
        outputs=("plastic_strain",),
        nonlinear_material=material_map,
        nonlinear_solver=NonlinearSolverOptions(
            convergence_tolerance=1.0e-6,
            maximum_plastic_iterations=20,
        ),
        work_dir=tmp_path / "plateau_distinct_from_sigma_c",
        tolerance=1.0e-4,
        level=2,
    )

    nonlinear = result.diagnostics["nonlinear"]
    assert nonlinear["yielded_last"] > 0
    assert np.linalg.norm(result.fields["plastic_strain"]) > 0.0


def test_asymmetric_density_map_yields_in_tension_before_compression(tmp_path: Path):
    material_map = _constant_asymmetric_map(
        shape=(3, 3, 3),
        youngs_mpa=1000.0,
        poisson_ratio=0.3,
        tensile_yield_mpa=5.0,
        compressive_yield_mpa=20.0,
        plateau_mpa=20.0,
    )
    stiffness_gpa = (material_map.youngs_modulus_mpa / 1000.0).astype(np.float32)
    solver_options = NonlinearSolverOptions(
        convergence_tolerance=1.0e-6,
        maximum_plastic_iterations=20,
    )

    tensile_result = solve(
        material=stiffness_gpa,
        material_unit="GPa",
        spacing=(1.0, 1.0, 1.0),
        array_order="xyz",
        strain=0.008,
        test="axial",
        load_case_type="constrained_axial",
        outputs=("forces", "plastic_strain"),
        nonlinear_material=material_map,
        nonlinear_solver=solver_options,
        work_dir=tmp_path / "tension",
        tolerance=1.0e-4,
        level=2,
    )
    compressive_result = solve(
        material=stiffness_gpa,
        material_unit="GPa",
        spacing=(1.0, 1.0, 1.0),
        array_order="xyz",
        strain=-0.008,
        test="axial",
        load_case_type="constrained_axial",
        outputs=("forces", "plastic_strain"),
        nonlinear_material=material_map,
        nonlinear_solver=solver_options,
        work_dir=tmp_path / "compression",
        tolerance=1.0e-4,
        level=2,
    )

    tensile_nonlinear = tensile_result.diagnostics["nonlinear"]
    compressive_nonlinear = compressive_result.diagnostics["nonlinear"]

    assert tensile_nonlinear["yielded_last"] > 0
    assert compressive_nonlinear["yielded_last"] == 0
    assert np.linalg.norm(tensile_result.fields["plastic_strain"]) > 0.0
    assert np.linalg.norm(compressive_result.fields["plastic_strain"]) == 0.0


def test_asymmetric_density_map_low_strength_voxels_yield_first(tmp_path: Path):
    shape = (4, 4, 4)
    youngs_mpa = np.full(shape, 1000.0, dtype=np.float64)
    tensile_yield_mpa = np.full(shape, 50.0, dtype=np.float64)
    compressive_yield_mpa = np.full(shape, 50.0, dtype=np.float64)
    plateau_mpa = np.full(shape, 50.0, dtype=np.float64)
    material_id = np.full(shape, 2, dtype=np.uint16)
    low_strength = np.zeros(shape, dtype=bool)
    low_strength[:2, :, :] = True
    tensile_yield_mpa[low_strength] = 5.0
    compressive_yield_mpa[low_strength] = 5.0
    plateau_mpa[low_strength] = 5.0
    material_id[low_strength] = 1
    material_map = KeavenyNonlinearMaterialMap(
        youngs_modulus_mpa=youngs_mpa,
        poisson_ratio=np.full(shape, 0.3, dtype=np.float64),
        compressive_yield_mpa=compressive_yield_mpa,
        tensile_yield_mpa=tensile_yield_mpa,
        plateau_mpa=plateau_mpa,
        material_id=material_id,
        metadata={"preset": "test_two_material"},
    )

    result = solve(
        material=(youngs_mpa / 1000.0).astype(np.float32),
        material_unit="GPa",
        spacing=(1.0, 1.0, 1.0),
        array_order="xyz",
        strain=0.008,
        test="axial",
        load_case_type="constrained_axial",
        outputs=("plastic_strain",),
        nonlinear_material=material_map,
        nonlinear_solver=NonlinearSolverOptions(
            convergence_tolerance=1.0e-6,
            maximum_plastic_iterations=20,
        ),
        work_dir=tmp_path / "two_material",
        tolerance=1.0e-4,
        level=2,
    )

    plastic_norm = np.linalg.norm(result.fields["plastic_strain"], axis=1)
    element_coords = _active_element_coordinates(np.ones(shape, dtype=np.float32))
    low_indices = [i for i, coord in enumerate(element_coords) if coord[0] < 2]
    high_indices = [i for i, coord in enumerate(element_coords) if coord[0] >= 2]

    yielded_low = plastic_norm[low_indices] > 0.0
    yielded_high = plastic_norm[high_indices] > 0.0

    assert 0 < result.diagnostics["nonlinear"]["yielded_last"] <= int(
        np.count_nonzero(yielded_low)
    )
    assert np.any(yielded_low)
    assert not np.any(yielded_high)


def test_asymmetric_density_map_pmma_fixture_ids_are_elastic_with_zero_yield_fields(
    tmp_path: Path,
):
    shape = (4, 4, 4)
    youngs_mpa = np.full(shape, 1000.0, dtype=np.float64)
    poisson_ratio = np.full(shape, 0.3, dtype=np.float64)
    tensile_yield_mpa = np.full(shape, 5.0, dtype=np.float64)
    compressive_yield_mpa = np.full(shape, 5.0, dtype=np.float64)
    plateau_mpa = np.full(shape, 5.0, dtype=np.float64)
    material_id = np.ones(shape, dtype=np.uint16)

    pmma_fixture = np.zeros(shape, dtype=bool)
    pmma_fixture[:2, :, :] = True
    youngs_mpa[pmma_fixture] = 2500.0
    poisson_ratio[pmma_fixture] = 0.31
    tensile_yield_mpa[pmma_fixture] = 0.0
    compressive_yield_mpa[pmma_fixture] = 0.0
    plateau_mpa[pmma_fixture] = 0.0
    material_id[pmma_fixture] = 2

    material_map = KeavenyNonlinearMaterialMap(
        youngs_modulus_mpa=youngs_mpa,
        poisson_ratio=poisson_ratio,
        compressive_yield_mpa=compressive_yield_mpa,
        tensile_yield_mpa=tensile_yield_mpa,
        plateau_mpa=plateau_mpa,
        material_id=material_id,
        metadata={"preset": "test_pmma_fixture"},
    )

    result = solve(
        material=(youngs_mpa / 1000.0).astype(np.float32),
        material_unit="GPa",
        spacing=(1.0, 1.0, 1.0),
        array_order="xyz",
        strain=0.008,
        test="axial",
        load_case_type="constrained_axial",
        outputs=("plastic_strain",),
        nonlinear_material=material_map,
        nonlinear_solver=NonlinearSolverOptions(
            convergence_tolerance=1.0e-6,
            maximum_plastic_iterations=20,
        ),
        work_dir=tmp_path / "pmma_fixture",
        tolerance=1.0e-4,
        level=2,
    )

    plastic_norm = np.linalg.norm(result.fields["plastic_strain"], axis=1)
    element_coords = _active_element_coordinates(np.ones(shape, dtype=np.float32))
    pmma_indices = [i for i, coord in enumerate(element_coords) if coord[0] < 2]
    bone_indices = [i for i, coord in enumerate(element_coords) if coord[0] >= 2]

    assert result.diagnostics["nonlinear"]["yielded_last"] > 0
    assert not np.any(plastic_norm[pmma_indices] > 0.0)
    assert np.any(plastic_norm[bone_indices] > 0.0)


def _constant_asymmetric_map(
    *,
    shape: tuple[int, int, int],
    youngs_mpa: float,
    poisson_ratio: float,
    tensile_yield_mpa: float,
    compressive_yield_mpa: float,
    plateau_mpa: float,
) -> KeavenyNonlinearMaterialMap:
    return KeavenyNonlinearMaterialMap(
        youngs_modulus_mpa=np.full(shape, youngs_mpa, dtype=np.float64),
        poisson_ratio=np.full(shape, poisson_ratio, dtype=np.float64),
        compressive_yield_mpa=np.full(shape, compressive_yield_mpa, dtype=np.float64),
        tensile_yield_mpa=np.full(shape, tensile_yield_mpa, dtype=np.float64),
        plateau_mpa=np.full(shape, plateau_mpa, dtype=np.float64),
        material_id=np.ones(shape, dtype=np.uint16),
        metadata={"preset": "test_constant_asymmetric"},
    )


def _write_valid_asymmetric_input(tmp_path: Path, filename: str) -> Path:
    material_map = _constant_asymmetric_map(
        shape=(3, 3, 3),
        youngs_mpa=1000.0,
        poisson_ratio=0.3,
        tensile_yield_mpa=20.0,
        compressive_yield_mpa=20.0,
        plateau_mpa=20.0,
    )
    stiffness_gpa_xyz = (material_map.youngs_modulus_mpa / 1000.0).astype(
        np.float32
    )
    fixed_coordinates, fixed_values = axial_compression(
        stiffness_gpa_xyz,
        axis="z",
        strain=-0.01,
        voxel_size_mm=1.0,
    )
    return write_parosol_input(
        tmp_path / filename,
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coordinates,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        nonlinear_material=material_map,
        nonlinear_solver=NonlinearSolverOptions(maximum_plastic_iterations=3),
    )


def _run_native_sed(input_file: Path, cwd: Path):
    executable = packaged_executable()
    assert executable.exists(), f"packaged executable not found: {executable}"
    return run_parosol(
        build_parosol_command(
            executable=executable,
            input_file=input_file,
            outputs=("sed",),
            tolerance=1.0e-4,
            level=2,
        ),
        cwd=cwd,
    )


def _active_element_coordinates(stiffness_gpa_xyz) -> list[tuple[int, int, int]]:
    elements = np.argwhere(np.asarray(stiffness_gpa_xyz) > 0.0).astype(
        np.int64, copy=False
    )
    return [tuple(int(v) for v in row) for row in sorted(elements, key=_morton_key)]


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
