import numpy as np

from parosol_py.images import largest_connected_component
from parosol_py.surfaces import top_bottom_surface_nodes


def test_visible_surface_nodes_follow_uneven_top_and_bottom():
    material = np.zeros((2, 2, 4), dtype=np.float32)
    material[0, 0, 1:3] = 1.0
    material[1, 0, 0:4] = 1.0

    nodes = top_bottom_surface_nodes(material, axis="z", selection="visible")

    assert (0, 0, 1) in nodes["bottom"]
    assert (1, 0, 0) in nodes["bottom"]
    assert (0, 0, 3) in nodes["top"]
    assert (1, 0, 4) in nodes["top"]
    assert all(node[2] != 0 for node in nodes["bottom"] if node[0] == 0)


def test_smart_surface_auto_depth_samples_more_than_one_layer_for_thick_bone():
    material = np.ones((3, 3, 30), dtype=np.float32)

    nodes = top_bottom_surface_nodes(
        material,
        axis="z",
        selection={"mode": "smart", "depth": "auto"},
    )

    assert any(node[2] == 1 for node in nodes["bottom"])
    assert any(node[2] == 29 for node in nodes["top"])


def test_largest_connected_component_preserves_original_values():
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[0, 0, 0] = 7
    labels[2:4, 2:4, 2:4] = 9

    filtered = largest_connected_component(labels)

    assert np.count_nonzero(filtered == 9) == 8
    assert filtered[0, 0, 0] == 0
