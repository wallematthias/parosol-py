import numpy as np

from parosol_py.field_export import NativeFieldMapper


def test_native_field_mapper_maps_active_values_to_dense_xyz():
    stiffness = np.zeros((3, 2, 1), dtype=np.float32)
    stiffness[0, 1, 0] = 1
    stiffness[1, 1, 0] = 1
    stiffness[2, 0, 0] = 1
    mapper = NativeFieldMapper(stiffness)

    dense = mapper.scalar_to_dense(np.array([21.0, 31.0, 40.0], dtype=np.float32))

    assert dense.shape == (3, 2, 1)
    assert dense[0, 1, 0] == 21.0
    assert dense[1, 1, 0] == 31.0
    assert dense[2, 0, 0] == 40.0


def test_native_field_mapper_reuses_coordinate_arrays():
    stiffness = np.ones((2, 2, 2), dtype=np.float32)
    mapper = NativeFieldMapper(stiffness)

    first = mapper.active_coordinates
    second = mapper.active_coordinates

    assert first is second
