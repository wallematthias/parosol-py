import numpy as np
import pytest

from parosol_py.boundary_conditions import axial_compression


def test_axial_compression_z_generates_bottom_and_top_constraints():
    stiffness = np.ones((3, 2, 4), dtype=np.float32)
    coords, values = axial_compression(stiffness, axis="z", strain=-0.01)

    assert coords.shape[1] == 4
    assert values.shape == (coords.shape[0],)
    assert coords.dtype == np.uint16
    assert values.dtype == np.float32

    bottom_z = coords[:, 2] == 0
    top_z = coords[:, 2] == 4
    assert np.any(bottom_z)
    assert np.any(top_z)
    assert np.all(values[bottom_z] == 1e-16)
    assert np.allclose(values[top_z & (coords[:, 3] == 2)], -0.04)
    assert np.all(values[top_z & (coords[:, 3] != 2)] == 1e-16)
    assert set(coords[bottom_z, 3]) == {0, 1, 2}
    assert set(coords[top_z, 3]) == {0, 1, 2}
    assert bottom_z.sum() == 3 * (3 + 1) * (2 + 1)
    assert top_z.sum() == 3 * (3 + 1) * (2 + 1)


def test_axial_compression_ignores_empty_columns():
    stiffness = np.zeros((2, 2, 2), dtype=np.float32)
    stiffness[0, 0, :] = 1.0

    coords, _values = axial_compression(stiffness, axis="z", strain=-0.01)

    unique_xy = set(map(tuple, coords[:, :2]))
    assert unique_xy == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_axial_compression_rejects_dimensions_outside_native_coordinate_range():
    stiffness = np.ones((32768, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="native|int16|range"):
        axial_compression(stiffness, axis="x", strain=-0.01)
