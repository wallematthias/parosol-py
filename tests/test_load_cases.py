import numpy as np

from parosol_py import AxialCompression, BodyWeightCompression, Model, SimpleShear


def test_axial_compression_generates_named_top_and_bottom_sets():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(0.5, 0.5, 0.5))

    bc = AxialCompression(axis="z", strain=-0.01).generate(model)

    assert "top" in bc.node_sets
    assert "bottom" in bc.node_sets
    top_z_values = bc.fixed_coordinates[bc.fixed_coordinates[:, 2] == 2]
    assert np.any(top_z_values[:, 3] == 2)
    assert np.min(bc.fixed_values) == np.float32(-0.01)


def test_body_weight_compression_distributes_total_force_over_top_nodes():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = BodyWeightCompression(axis="z", force_n=-90.0).generate(model)

    assert bc.loaded_coordinates.shape[0] == 9
    assert np.all(bc.loaded_coordinates[:, 3] == 2)
    assert np.sum(bc.loaded_values) == np.float32(-90.0)


def test_simple_shear_moves_top_in_lateral_direction():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = SimpleShear(axis="z", direction="x", strain=0.02).generate(model)

    top_x_values = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 0)
    ]
    assert np.any(np.isclose(top_x_values, 0.04))
