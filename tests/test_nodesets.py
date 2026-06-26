import numpy as np
import pytest

from parosol_py.nodesets import (
    boundary_conditions_from_nodesets,
    nodes_from_labeled_voxels,
    nodes_from_mask_face,
)


def test_all_corner_nodes_selects_every_corner_touched_by_label():
    labels = np.zeros((2, 2, 2), dtype=np.uint8)
    labels[0, 0, 0] = 1

    nodes = nodes_from_labeled_voxels(labels, label=1, selection="all_corner_nodes")

    assert nodes == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]


def test_surface_nodes_omits_internal_interface_nodes_between_labeled_voxels():
    labels = np.ones((2, 2, 2), dtype=np.uint8)

    nodes = nodes_from_labeled_voxels(labels, label=1, selection="surface_nodes")

    assert (1, 1, 1) not in nodes
    assert len(nodes) == 26


def test_surface_nodes_can_select_interface_with_material_object():
    labels = np.zeros((2, 1, 1), dtype=np.uint8)
    labels[0, 0, 0] = 1
    material = np.ones((2, 1, 1), dtype=np.float32)

    nodes = nodes_from_labeled_voxels(
        labels,
        label=1,
        selection="interface_nodes",
        material=material,
    )

    assert nodes == [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]


def test_mask_face_nodes_selects_only_requested_outer_face():
    mask = np.ones((2, 2, 2), dtype=bool)

    nodes = nodes_from_mask_face(mask, axis="z", side=1)

    assert all(node[2] == 2 for node in nodes)
    assert len(nodes) == 9


def test_nodeset_percent_displacement_uses_fixed_to_prescribed_centroid_span():
    node_sets = {
        "bottom": [(0, 0, 1)],
        "top": [(0, 0, 4)],
    }

    bc = boundary_conditions_from_nodesets(
        node_sets,
        fixed=[{"nodeset": "bottom", "dofs": ["x", "y", "z"], "value": 0.0}],
        prescribed=[{"nodeset": "top", "dof": "z", "value": "-1%"}],
        dimensions_xyz=(2, 2, 6),
        spacing=(1.0, 1.0, 1.0),
        percent_reference_lengths_mm={"x": 2.0, "y": 2.0, "z": 4.0},
    )

    prescribed_z = bc.fixed_values[
        (bc.fixed_coordinates[:, 3] == 2) & (~np.isclose(bc.fixed_values, 0.0))
    ]

    assert prescribed_z.tolist() == pytest.approx([-0.03])


def test_nodeset_percent_displacement_honors_explicit_reference_nodeset():
    node_sets = {
        "near_support": [(0, 0, 8)],
        "far_support": [(0, 0, 2)],
        "top": [(0, 0, 10)],
    }

    bc = boundary_conditions_from_nodesets(
        node_sets,
        fixed=[
            {"nodeset": "near_support", "dofs": ["z"], "value": 0.0},
            {"nodeset": "far_support", "dofs": ["z"], "value": 0.0},
        ],
        prescribed=[
            {
                "nodeset": "top",
                "dof": "z",
                "value": "-10%",
                "reference_nodeset": "near_support",
            }
        ],
        dimensions_xyz=(2, 2, 20),
        spacing=(1.0, 1.0, 1.0),
        percent_reference_lengths_mm={"z": 20.0},
    )

    prescribed_z = bc.fixed_values[
        (bc.fixed_coordinates[:, 3] == 2) & (~np.isclose(bc.fixed_values, 0.0))
    ]

    assert prescribed_z.tolist() == pytest.approx([-0.2])


def test_nodeset_percent_displacement_honors_explicit_reference_length():
    node_sets = {
        "bottom": [(0, 0, 16), (0, 0, 17)],
        "top": [(0, 0, 50), (0, 0, 51)],
    }

    bc = boundary_conditions_from_nodesets(
        node_sets,
        fixed=[{"nodeset": "bottom", "dofs": ["z"], "value": 0.0}],
        prescribed=[
            {
                "nodeset": "top",
                "dof": "z",
                "value": "-0.68%",
                "reference_nodeset": "bottom",
                "reference_length_mm": 38.0,
            }
        ],
        dimensions_xyz=(2, 2, 60),
        spacing=(1.0, 1.0, 1.0),
        percent_reference_lengths_mm={"z": 60.0},
    )

    prescribed_z = bc.fixed_values[
        (bc.fixed_coordinates[:, 3] == 2) & (~np.isclose(bc.fixed_values, 0.0))
    ]

    assert prescribed_z.tolist() == pytest.approx([-0.2584, -0.2584])


def test_nodeset_percent_displacement_infers_reference_along_load_direction():
    node_sets = {
        "above_support": [(0, 0, 20)],
        "below_support": [(0, 0, 6)],
        "top": [(0, 0, 10)],
    }

    bc = boundary_conditions_from_nodesets(
        node_sets,
        fixed=[
            {"nodeset": "above_support", "dofs": ["z"], "value": 0.0},
            {"nodeset": "below_support", "dofs": ["z"], "value": 0.0},
        ],
        prescribed=[{"nodeset": "top", "dof": "z", "value": "-10%"}],
        dimensions_xyz=(2, 2, 24),
        spacing=(1.0, 1.0, 1.0),
        percent_reference_lengths_mm={"z": 24.0},
    )

    prescribed_z = bc.fixed_values[
        (bc.fixed_coordinates[:, 3] == 2) & (~np.isclose(bc.fixed_values, 0.0))
    ]

    assert prescribed_z.tolist() == pytest.approx([-0.4])
