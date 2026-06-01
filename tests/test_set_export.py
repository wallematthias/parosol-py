import json

import numpy as np

from parosol_py.images import coarsen_array
from parosol_py.set_export import write_element_sets, write_node_sets


def test_write_node_sets_json_and_vtk(tmp_path):
    written = write_node_sets(
        {"top": [(0, 0, 1), (1, 0, 1)]},
        directory=tmp_path,
        spacing=(0.5, 0.5, 0.5),
        formats=("json", "vtk"),
    )

    data = json.loads(written["node_sets_json"].read_text(encoding="utf-8"))
    assert data["top"] == [[0, 0, 1], [1, 0, 1]]
    assert "POINTS 2 float" in written["top_nodes_vtk"].read_text(encoding="utf-8")


def test_write_element_sets_groups_nonzero_materials(tmp_path):
    material = np.zeros((2, 2, 1), dtype=np.uint8)
    material[0, 0, 0] = 126
    material[1, 0, 0] = 127

    written = write_element_sets(
        material,
        directory=tmp_path,
        spacing=(1, 1, 1),
        formats=("json",),
    )

    data = json.loads(written["element_sets_json"].read_text(encoding="utf-8"))
    assert data == {"126": [[0, 0, 0]], "127": [[1, 0, 0]]}


def test_coarsen_array_uses_block_reducer():
    array = np.arange(8, dtype=np.float32).reshape((2, 2, 2))

    coarsened = coarsen_array(array, factor=2, reducer="mean")

    assert coarsened.shape == (1, 1, 1)
    assert coarsened[0, 0, 0] == np.mean(array)
