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


def test_native_field_mapper_maps_h5_mesh_vectors_to_dense_elements():
    stiffness = np.zeros((3, 3, 3), dtype=np.float32)
    stiffness[1, 1, 1] = 1
    mapper = NativeFieldMapper(stiffness)
    coordinates = np.asarray(
        [
            [1, 1, 1],
            [2, 1, 1],
            [1, 2, 1],
            [2, 2, 1],
            [1, 1, 2],
            [2, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
        ],
        dtype=np.float32,
    )
    elements = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    displacements = np.asarray(
        [[index, index + 1, index + 2] for index in range(8)],
        dtype=np.float32,
    )

    dense = mapper.mesh_vector_to_dense_element(coordinates, elements, displacements)

    assert dense.shape == (3, 3, 3, 3)
    np.testing.assert_allclose(dense[1, 1, 1], np.mean(displacements, axis=0))
    assert np.count_nonzero(dense[..., 0]) == 1
