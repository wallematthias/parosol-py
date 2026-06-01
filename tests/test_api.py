import numpy as np
import pytest

import parosol_py
from parosol_py import solve
from parosol_py.api import solve_aim
from parosol_py.reports import solve_summary_dict
from parosol_py.runner import RunResult, RunSummary


def test_package_imports():
    assert parosol_py.__version__ == "0.1.0"


def test_solve_dry_run_writes_input_and_reports_summary(tmp_path):
    material_zyx = np.zeros((4, 3, 2))
    material_zyx[:, 1, 1] = 1000.0

    result = solve(
        material=material_zyx,
        spacing=(0.061, 0.061, 0.061),
        origin=(1.0, 2.0, 3.0),
        material_unit="MPa",
        test="axial",
        test_axis="z",
        strain=-0.01,
        outputs=("sed",),
        work_dir=tmp_path,
        dry_run=True,
    )

    assert result.input_file.exists()
    assert result.command[-1] == str(result.input_file)
    assert "--SED" in result.command
    assert result.fields == {}
    assert result.summary.dimensions_xyz == (2, 3, 4)
    assert result.summary.spacing == (0.061, 0.061, 0.061)


def test_solve_dry_run_accepts_export_dir(tmp_path):
    material_zyx = np.ones((4, 3, 2)) * 1000.0

    result = solve(
        material=material_zyx,
        spacing=(1, 1, 1),
        outputs=("sed",),
        work_dir=tmp_path,
        export_dir=tmp_path / "exports",
        dry_run=True,
    )

    assert result.exported == {}


def test_solve_accepts_scanner_rounding_in_isotropic_spacing(tmp_path):
    material_zyx = np.ones((4, 3, 2)) * 1000.0

    result = solve(
        material=material_zyx,
        spacing=(0.06069965288043022, 0.06069965288043022, 0.06069643050432205),
        outputs=("sed",),
        work_dir=tmp_path,
        dry_run=True,
    )

    assert result.summary.spacing == (
        0.06069965288043022,
        0.06069965288043022,
        0.06069643050432205,
    )


def test_solve_exports_native_scalar_field_in_dense_xyz_order(monkeypatch, tmp_path):
    material_zyx = np.ones((2, 2, 3)) * 1000.0
    dimensions_xyz = (3, 2, 2)
    dense = np.empty(dimensions_xyz, dtype=np.float32)
    coords = []
    for x in range(dimensions_xyz[0]):
        for y in range(dimensions_xyz[1]):
            for z in range(dimensions_xyz[2]):
                dense[x, y, z] = 100 * x + 10 * y + z
                coords.append((x, y, z))

    def morton_key(x, y, z):
        key = 0
        bit = 1
        while bit <= max(x, y, z):
            key += (x & bit) * bit * bit
            key += (y & bit) * bit * bit * 2
            key += (z & bit) * bit * bit * 4
            bit <<= 1
        return key

    native_values = np.asarray(
        [dense[x, y, z] for x, y, z in sorted(coords, key=lambda c: morton_key(*c))],
        dtype=np.float32,
    ).reshape((-1, 1))
    captured = {}

    def fake_run_parosol(command, *, cwd=None, stream=False):
        return RunResult(
            command=command,
            stdout="",
            stderr="",
            returncode=0,
            summary=RunSummary(),
        )

    def fake_export_scalar_image(grid, output_path):
        captured["grid"] = grid
        captured["path"] = output_path
        return output_path

    monkeypatch.setattr("parosol_py.api.run_parosol", fake_run_parosol)
    monkeypatch.setattr(
        "parosol_py.api.read_solution_fields",
        lambda input_file, *, outputs: {"sed": native_values},
    )
    monkeypatch.setattr("parosol_py.api.export_scalar_image", fake_export_scalar_image)

    result = solve(
        material=material_zyx,
        spacing=(1, 1, 1),
        outputs=("sed",),
        work_dir=tmp_path,
        export_dir=tmp_path / "exports",
    )

    assert result.exported["sed"].name == "sed.nii.gz"
    assert result.exported["command_log"].name == "parosol_command.txt"
    assert result.exported["stdout_log"].name == "parosol_stdout.log"
    assert result.exported["stderr_log"].name == "parosol_stderr.log"
    np.testing.assert_array_equal(captured["grid"].array_xyz, dense)
    assert result.fields == {"sed": native_values}


def test_solve_exports_sparse_native_scalar_field_to_dense_xyz(monkeypatch, tmp_path):
    material_zyx = np.zeros((1, 2, 3), dtype=np.float32)
    active_coords_xyz = [(2, 0, 0), (0, 1, 0), (1, 1, 0)]
    for x, y, z in active_coords_xyz:
        material_zyx[z, y, x] = 1000.0

    def morton_key(x, y, z):
        key = 0
        bit = 1
        while bit <= max(x, y, z):
            key += (x & bit) * bit * bit
            key += (y & bit) * bit * bit * 2
            key += (z & bit) * bit * bit * 4
            bit <<= 1
        return key

    expected_dense = np.zeros((3, 2, 1), dtype=np.float32)
    values_by_coord = {
        (0, 1, 0): 21.0,
        (1, 1, 0): 31.0,
        (2, 0, 0): 40.0,
    }
    for coord, value in values_by_coord.items():
        expected_dense[coord] = value
    native_values = np.asarray(
        [
            values_by_coord[coord]
            for coord in sorted(active_coords_xyz, key=lambda c: morton_key(*c))
        ],
        dtype=np.float32,
    ).reshape((-1, 1))
    captured = {}

    def fake_run_parosol(command, *, cwd=None, stream=False):
        return RunResult(
            command=command,
            stdout="",
            stderr="",
            returncode=0,
            summary=RunSummary(),
        )

    def fake_export_scalar_image(grid, output_path):
        captured["grid"] = grid
        captured["path"] = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("exported")
        return output_path

    monkeypatch.setattr("parosol_py.api.run_parosol", fake_run_parosol)
    monkeypatch.setattr(
        "parosol_py.api.read_solution_fields",
        lambda input_file, *, outputs: {"sed": native_values},
    )
    monkeypatch.setattr("parosol_py.api.export_scalar_image", fake_export_scalar_image)

    result = solve(
        material=material_zyx,
        spacing=(1, 1, 1),
        outputs=("sed",),
        work_dir=tmp_path,
        export_dir=tmp_path / "exports",
    )

    assert result.exported["sed"].exists()
    np.testing.assert_array_equal(captured["grid"].array_xyz, expected_dense)
    assert result.fields == {"sed": native_values}


def test_solve_derives_summary_diagnostics_from_fea_fields(monkeypatch, tmp_path):
    material_zyx = np.ones((2, 2, 2), dtype=np.float32) * 1000.0
    dimensions_xyz = (2, 2, 2)
    node_coords = [
        (x, y, z)
        for x in range(dimensions_xyz[0] + 1)
        for y in range(dimensions_xyz[1] + 1)
        for z in range(dimensions_xyz[2] + 1)
    ]

    def morton_key(x, y, z):
        key = 0
        bit = 1
        while bit <= max(x, y, z):
            key += (x & bit) * bit * bit
            key += (y & bit) * bit * bit * 2
            key += (z & bit) * bit * bit * 4
            bit <<= 1
        return key

    forces = np.zeros((len(node_coords), 3), dtype=np.float64)
    displacements = np.zeros((len(node_coords), 3), dtype=np.float64)
    for index, (_x, _y, z) in enumerate(
        sorted(node_coords, key=lambda c: morton_key(*c))
    ):
        if z == dimensions_xyz[2]:
            forces[index, 2] = -2.0
            displacements[index, 2] = -0.02

    read_calls = {}

    def fake_run_parosol(command, *, cwd=None, stream=False):
        return RunResult(
            command=command,
            stdout="",
            stderr="",
            returncode=0,
            summary=RunSummary(iterations=12, relative_residual=1e-7),
        )

    def fake_read_solution_fields(input_file, *, outputs):
        read_calls["outputs"] = outputs
        return {
            "sed": np.full((8, 1), 1e-4, dtype=np.float64),
            "forces": forces,
            "displacements": displacements,
        }

    monkeypatch.setattr("parosol_py.api.run_parosol", fake_run_parosol)
    monkeypatch.setattr(
        "parosol_py.api.read_solution_fields", fake_read_solution_fields
    )

    result = solve(
        material=material_zyx,
        spacing=(1, 1, 1),
        outputs=("sed",),
        test_axis="z",
        strain=-0.01,
        critical_strain=0.007,
        critical_volume_percent=12.5,
        work_dir=tmp_path,
    )

    assert read_calls["outputs"] == ("sed", "forces", "displacements")
    summary = solve_summary_dict(result)

    assert summary["mechanics"]["reaction_force"]["z"] == pytest.approx(-18.0)
    assert summary["mechanics"]["applied_displacement"]["z"] == pytest.approx(-0.02)
    assert summary["mechanics"]["stiffness"]["z"] == pytest.approx(900.0)
    assert summary["failure"]["criterion"] == "pistoia"
    assert summary["failure"]["critical_strain"] == pytest.approx(0.007)
    assert summary["failure"]["critical_volume_percent"] == pytest.approx(12.5)
    assert summary["failure"]["ees_at_critical_volume"] == pytest.approx(
        np.sqrt(2e-4 / 1000.0)
    )
    assert summary["failure"]["factor"] == pytest.approx(0.007 / np.sqrt(2e-4 / 1000.0))
    assert summary["failure"]["failure_load"]["z"] == pytest.approx(
        -18.0 * 0.007 / np.sqrt(2e-4 / 1000.0)
    )


def test_solve_accepts_explicit_boundary_condition_set(monkeypatch, tmp_path):
    from parosol_py import BoundaryConditionSet

    captured = {}
    bc = BoundaryConditionSet(
        fixed_coordinates=np.array([[0, 0, 0, 0]], dtype=np.uint16),
        fixed_values=np.array([1e-16], dtype=np.float32),
        loaded_coordinates=np.array([[2, 2, 2, 2]], dtype=np.uint16),
        loaded_values=np.array([-10.0], dtype=np.float32),
    )

    def fake_write_parosol_input(**kwargs):
        captured.update(kwargs)
        path = kwargs["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("h5")
        return path

    monkeypatch.setattr("parosol_py.api.write_parosol_input", fake_write_parosol_input)

    result = solve(
        material=np.ones((2, 2, 2)),
        spacing=(1, 1, 1),
        boundary_conditions=bc,
        work_dir=tmp_path,
        dry_run=True,
    )

    assert result.input_file.exists()
    np.testing.assert_array_equal(
        captured["loaded_node_coordinates"], bc.loaded_coordinates
    )
    np.testing.assert_array_equal(captured["loaded_node_values"], bc.loaded_values)


def test_solve_rejects_anisotropic_spacing_in_dry_run(tmp_path):
    material_zyx = np.zeros((4, 3, 2))
    material_zyx[:, 1, 1] = 1000.0

    with pytest.raises(ValueError, match="isotropic|anisotropic"):
        solve(
            material=material_zyx,
            spacing=(0.061, 0.08, 0.061),
            work_dir=tmp_path,
            dry_run=True,
        )


def test_solve_aim_reads_aim_and_preserves_metadata(monkeypatch, tmp_path):
    calls = {}

    def fake_read_aim(path):
        calls["path"] = path
        arr = np.zeros((4, 3, 2))
        arr[:, 1, 1] = 1000.0
        return arr, {
            "element_size": (0.061, 0.061, 0.061),
            "position": (1.0, 2.0, 3.0),
        }

    monkeypatch.setattr("parosol_py.api.read_aim", fake_read_aim)

    result = solve_aim("case.aim", work_dir=tmp_path, dry_run=True)

    assert calls["path"] == "case.aim"
    assert result.input_file.exists()
    assert result.summary.spacing == (0.061, 0.061, 0.061)
