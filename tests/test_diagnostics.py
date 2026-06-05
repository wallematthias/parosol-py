import numpy as np
import pytest

from parosol_py.core import BoundaryConditionSet
from parosol_py.diagnostics import build_fea_diagnostics


def test_shear_failure_uses_same_factor_and_reports_generalized_force():
    stiffness = np.ones((1, 1, 1), dtype=np.float32)
    forces = np.zeros((8, 3), dtype=np.float32)
    for index, coord in enumerate(_unit_cube_nodes()):
        if coord[2] == 1:
            forces[index, 0] = 2.0
    fields = {
        "forces": forces,
        "sed": np.full((1,), 0.05, dtype=np.float32),
    }

    diagnostics = build_fea_diagnostics(
        fields=fields,
        stiffness_gpa_xyz=stiffness,
        axis="z",
        strain=0.01,
        load_case_type="shear",
        load_direction="x",
        critical_strain=0.02,
        critical_volume_percent=100,
    )

    assert diagnostics["mechanics"]["generalized_load"]["name"] == "force"
    assert diagnostics["mechanics"]["generalized_load"]["value"] == pytest.approx(8.0)
    assert diagnostics["failure"]["factor"] == pytest.approx(2.0)
    assert diagnostics["failure"]["failure_generalized_load"]["value"] == pytest.approx(
        16.0
    )


def test_torsion_reports_moment_and_rotational_failure_load():
    stiffness = np.ones((1, 1, 1), dtype=np.float32)
    forces = np.zeros((8, 3), dtype=np.float32)
    for index, coord in enumerate(_unit_cube_nodes()):
        if coord == (1, 0, 1):
            forces[index, 1] = 1.0
        if coord == (0, 1, 1):
            forces[index, 0] = -1.0
    fields = {
        "forces": forces,
        "sed": np.full((1,), 0.05, dtype=np.float32),
    }

    diagnostics = build_fea_diagnostics(
        fields=fields,
        stiffness_gpa_xyz=stiffness,
        axis="z",
        strain=0.0,
        load_case_type="torsion",
        rotation_degrees=1.0,
        critical_strain=0.02,
        critical_volume_percent=100,
    )

    assert diagnostics["mechanics"]["generalized_load"]["name"] == "moment"
    assert diagnostics["mechanics"]["generalized_load"]["value"] == pytest.approx(1.0)
    assert (
        diagnostics["mechanics"]["generalized_stiffness"]["name"]
        == "rotational_stiffness"
    )
    assert diagnostics["failure"]["failure_generalized_load"]["value"] == pytest.approx(
        2.0
    )


def test_projected_boundary_conditions_drive_reaction_summary():
    stiffness = np.ones((1, 1, 2), dtype=np.float32)
    node_coords = _two_voxel_nodes()
    forces = np.zeros((len(node_coords), 3), dtype=np.float32)
    loaded_nodes = [coord for coord in node_coords if coord[2] == 1]
    fixed_nodes = [coord for coord in node_coords if coord[2] == 0]
    for index, coord in enumerate(node_coords):
        if coord in loaded_nodes:
            forces[index, 2] = -3.0
    fixed_coordinates = []
    fixed_values = []
    for node in fixed_nodes:
        fixed_coordinates.append((*node, 2))
        fixed_values.append(0.0)
    for node in loaded_nodes:
        fixed_coordinates.append((*node, 2))
        fixed_values.append(-0.2)
    boundary_conditions = BoundaryConditionSet(
        fixed_coordinates=np.asarray(fixed_coordinates, dtype=np.uint16),
        fixed_values=np.asarray(fixed_values, dtype=np.float32),
    )

    diagnostics = build_fea_diagnostics(
        fields={"forces": forces, "sed": np.full((2,), 0.001, dtype=np.float32)},
        stiffness_gpa_xyz=stiffness,
        axis="z",
        strain=-0.1,
        load_case_type="spine_compression",
        critical_strain=0.02,
        critical_volume_percent=100,
        boundary_conditions=boundary_conditions,
    )

    mechanics = diagnostics["mechanics"]
    assert mechanics["reaction_node_source"] == "boundary_conditions"
    assert mechanics["top_node_count"] == len(loaded_nodes)
    assert mechanics["bottom_node_count"] == len(fixed_nodes)
    assert mechanics["reaction_force"]["z"] == pytest.approx(-12.0)
    assert mechanics["generalized_stiffness"]["value"] == pytest.approx(60.0)


def test_nodeset_diagnostics_report_applied_displacement_on_load_direction():
    stiffness = np.ones((1, 2, 1), dtype=np.float32)
    boundary_conditions = BoundaryConditionSet(
        fixed_coordinates=np.asarray([[0, 2, 0, 1]], dtype=np.uint16),
        fixed_values=np.asarray([1.0], dtype=np.float32),
    )

    diagnostics = build_fea_diagnostics(
        fields={},
        stiffness_gpa_xyz=stiffness,
        axis="y",
        strain=0.0,
        load_case_type="nodeset",
        load_direction="y",
        boundary_conditions=boundary_conditions,
    )

    mechanics = diagnostics["mechanics"]
    assert mechanics["applied_displacement"]["x"] is None
    assert mechanics["applied_displacement"]["y"] == pytest.approx(1.0)
    assert mechanics["applied_displacement"]["z"] is None


def test_explicit_analysis_dimensions_control_linear_strength_height():
    stiffness = np.ones((1, 1, 4), dtype=np.float32)
    node_coords = sorted(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in range(5)],
        key=lambda coord: _morton_key(*coord),
    )
    forces = np.zeros((len(node_coords), 3), dtype=np.float32)
    loaded_nodes = [coord for coord in node_coords if coord[2] == 4]
    fixed_nodes = [coord for coord in node_coords if coord[2] == 0]
    for index, coord in enumerate(node_coords):
        if coord in loaded_nodes:
            forces[index, 2] = -2.0
    fixed_coordinates = []
    fixed_values = []
    for node in fixed_nodes:
        fixed_coordinates.append((*node, 2))
        fixed_values.append(0.0)
    for node in loaded_nodes:
        fixed_coordinates.append((*node, 2))
        fixed_values.append(-0.2)
    analysis_mask = np.zeros(stiffness.shape, dtype=bool)
    analysis_mask[:, :, 1:3] = True

    diagnostics = build_fea_diagnostics(
        fields={"forces": forces, "sed": np.full((4,), 0.001, dtype=np.float32)},
        stiffness_gpa_xyz=stiffness,
        axis="z",
        strain=-0.1,
        load_case_type="spine_compression",
        critical_strain=0.02,
        critical_volume_percent=100,
        boundary_conditions=BoundaryConditionSet(
            fixed_coordinates=np.asarray(fixed_coordinates, dtype=np.uint16),
            fixed_values=np.asarray(fixed_values, dtype=np.float32),
        ),
        evaluation_mask_xyz=analysis_mask,
        analysis_dimensions_xyz=(1, 1, 2),
        linear_failure_estimates=True,
    )

    assert diagnostics["mechanics"]["analysis_dimensions"]["z"] == 2
    assert diagnostics["mechanics"]["generalized_stiffness"]["value"] == pytest.approx(
        40.0
    )
    assert diagnostics["failure"]["ees_distribution"]["count"] == 2
    assert diagnostics["failure"]["linear_reaction_at_deformation"][
        "failure_load"
    ]["z"] == pytest.approx(-0.16)
    assert diagnostics["failure"]["crawford_stiffness_height"]["failure_load"][
        "z"
    ] == pytest.approx(-0.544)
    assert diagnostics["failure"]["crawford_stiffness_height"][
        "equivalent_deformation"
    ] == pytest.approx(0.0068)
    assert diagnostics["failure"]["crawford_stiffness_height"][
        "relative_to_linear_deformation"
    ] == pytest.approx(3.4)


def test_interface_stiffness_uses_displacement_jump_across_analysis_mask():
    stiffness = np.ones((1, 1, 4), dtype=np.float32)
    node_coords = sorted(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in range(5)],
        key=lambda coord: _morton_key(*coord),
    )
    forces = np.zeros((len(node_coords), 3), dtype=np.float32)
    displacements = np.zeros((len(node_coords), 3), dtype=np.float32)
    loaded_nodes = [coord for coord in node_coords if coord[2] == 4]
    fixed_nodes = [coord for coord in node_coords if coord[2] == 0]
    for index, coord in enumerate(node_coords):
        if coord in loaded_nodes:
            forces[index, 2] = -2.0
        if coord[2] == 1:
            displacements[index, 2] = -0.03
        elif coord[2] == 3:
            displacements[index, 2] = -0.13
        elif coord[2] == 4:
            displacements[index, 2] = -0.20
    fixed_coordinates = []
    fixed_values = []
    for node in fixed_nodes:
        fixed_coordinates.append((*node, 2))
        fixed_values.append(0.0)
    for node in loaded_nodes:
        fixed_coordinates.append((*node, 2))
        fixed_values.append(-0.2)
    analysis_mask = np.zeros(stiffness.shape, dtype=bool)
    analysis_mask[:, :, 1:3] = True

    diagnostics = build_fea_diagnostics(
        fields={
            "forces": forces,
            "displacements": displacements,
            "sed": np.full((4,), 0.001, dtype=np.float32),
        },
        stiffness_gpa_xyz=stiffness,
        axis="z",
        strain=-0.1,
        load_case_type="spine_compression",
        critical_strain=0.02,
        critical_volume_percent=100,
        boundary_conditions=BoundaryConditionSet(
            fixed_coordinates=np.asarray(fixed_coordinates, dtype=np.uint16),
            fixed_values=np.asarray(fixed_values, dtype=np.float32),
        ),
        evaluation_mask_xyz=analysis_mask,
        analysis_dimensions_xyz=(1, 1, 2),
        linear_failure_estimates=True,
    )

    mechanics = diagnostics["mechanics"]
    assert mechanics["generalized_stiffness"]["value"] == pytest.approx(40.0)
    assert mechanics["interface_stiffness"]["value"] == pytest.approx(80.0)
    assert mechanics["interface_stiffness"]["displacement_difference"]["z"] == pytest.approx(
        -0.10
    )
    assert diagnostics["failure"]["linear_reaction_at_deformation"][
        "stiffness_n_per_mm"
    ] == pytest.approx(40.0)
    assert diagnostics["failure"]["linear_reaction_at_deformation"][
        "failure_load"
    ]["z"] == pytest.approx(-0.16)


def test_linear_strength_estimates_are_opt_in():
    forces = np.zeros((8, 3), dtype=np.float64)
    stiffness = np.ones((1, 1, 1), dtype=np.float64)

    diagnostics = build_fea_diagnostics(
        fields={"forces": forces, "sed": np.full((1,), 0.001, dtype=np.float32)},
        stiffness_gpa_xyz=stiffness,
        axis="z",
        strain=-0.01,
    )

    assert "linear_reaction_at_deformation" not in diagnostics["failure"]
    assert "crawford_stiffness_height" not in diagnostics["failure"]


def _unit_cube_nodes():
    return sorted(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        key=lambda coord: coord[0] + 2 * coord[1] + 4 * coord[2],
    )


def _two_voxel_nodes():
    return sorted(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1, 2)],
        key=lambda coord: _morton_key(*coord),
    )


def _morton_key(x: int, y: int, z: int) -> int:
    key = 0
    bit_index = 0
    limit = max(x, y, z)
    while (1 << bit_index) <= limit:
        key |= ((x >> bit_index) & 1) << (3 * bit_index)
        key |= ((y >> bit_index) & 1) << (3 * bit_index + 1)
        key |= ((z >> bit_index) & 1) << (3 * bit_index + 2)
        bit_index += 1
    return key
