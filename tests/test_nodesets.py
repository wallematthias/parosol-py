import numpy as np

from parosol_py.nodesets import nodes_from_labeled_voxels, nodes_from_mask_face


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
