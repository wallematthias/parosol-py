import pytest
import numpy as np

from parosol_py import (
    Bending,
    BodyWeightCompression,
    ConfinedCompression,
    ConstrainedAxialCompression,
    Model,
    SimpleShear,
    Torsion,
    UniaxialCompression,
)


def test_constrained_axial_compression_generates_named_top_and_bottom_sets():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(0.5, 0.5, 0.5))

    bc = ConstrainedAxialCompression(axis="z", strain=-0.01).generate(model)

    assert "top" in bc.node_sets
    assert "bottom" in bc.node_sets
    top_z_values = bc.fixed_coordinates[bc.fixed_coordinates[:, 2] == 2]
    assert np.any(top_z_values[:, 3] == 2)
    assert np.min(bc.fixed_values) == np.float32(-0.01)


def test_constrained_axial_compression_accepts_absolute_displacement():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = ConstrainedAxialCompression(axis="z", displacement=-0.25).generate(model)

    top_z_values = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 2)
    ]
    assert np.any(np.isclose(top_z_values, -0.25))


def test_body_weight_compression_distributes_total_force_over_top_nodes():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = BodyWeightCompression(axis="z", force_n=-90.0).generate(model)

    assert bc.loaded_coordinates.shape[0] == 9
    assert np.all(bc.loaded_coordinates[:, 3] == 2)
    assert np.sum(bc.loaded_values) == np.float32(-90.0)


def test_uniaxial_compression_leaves_top_and_bottom_laterally_free():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = UniaxialCompression(axis="z", strain=-0.01).generate(model)

    assert np.all(bc.fixed_coordinates[:, 3] == 2)
    assert np.any(
        (bc.fixed_coordinates[:, 2] == 2) & np.isclose(bc.fixed_values, -0.02)
    )
    assert np.any((bc.fixed_coordinates[:, 2] == 0) & np.isclose(bc.fixed_values, 0.0))


def test_simple_shear_moves_top_in_lateral_direction():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = SimpleShear(axis="z", direction="x", strain=0.02).generate(model)

    top_x_values = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 0)
    ]
    assert np.any(np.isclose(top_x_values, 0.04))


def test_simple_shear_supports_y_direction_on_z_axis():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = SimpleShear(axis="z", direction="y", strain=0.02).generate(model)

    top_y_values = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 1)
    ]
    top_x_values = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 0)
    ]
    assert np.any(np.isclose(top_y_values, 0.04))
    assert np.allclose(top_x_values, 0.0)


def test_simple_shear_accepts_absolute_displacement():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = SimpleShear(axis="z", direction="x", displacement=0.25).generate(model)

    top_x_values = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 0)
    ]
    assert np.any(np.isclose(top_x_values, 0.25))


def test_simple_shear_accepts_xy_vector_for_z_axis():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = SimpleShear(axis="z", vector=(0.02, 0.03)).generate(model)

    top_x_values = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 0)
    ]
    top_y_values = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 1)
    ]
    assert np.any(np.isclose(top_x_values, 0.04))
    assert np.any(np.isclose(top_y_values, 0.06))


def test_simple_shear_rejects_axis_parallel_direction():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    with pytest.raises(ValueError, match="must differ"):
        SimpleShear(axis="z", direction="z", strain=0.02).generate(model)


def test_confined_compression_fixes_top_and_bottom_lateral_motion():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = ConfinedCompression(axis="z", strain=-0.01).generate(model)

    top_lateral = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] != 2)
    ]
    top_axial = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 2)
    ]
    bottom_values = bc.fixed_values[bc.fixed_coordinates[:, 2] == 0]

    assert top_lateral.size > 0
    assert np.allclose(top_lateral, 0.0)
    assert np.any(np.isclose(top_axial, -0.02))
    assert np.allclose(bottom_values, 0.0)


def test_constrained_axial_can_use_visible_uneven_surfaces():
    material = np.zeros((2, 2, 4), dtype=np.float32)
    material[0, 0, 1:3] = 1.0
    material[1, 0, 0:4] = 1.0
    model = Model.from_array(material, spacing=(1, 1, 1), array_order="xyz")

    bc = ConstrainedAxialCompression(
        axis="z",
        displacement=-0.2,
        surface={"mode": "visible"},
    ).generate(model)

    assert (0, 0, 3) in bc.node_sets["top"]
    assert (1, 0, 4) in bc.node_sets["top"]
    assert np.any(
        (bc.fixed_coordinates[:, 0] == 0)
        & (bc.fixed_coordinates[:, 2] == 3)
        & (bc.fixed_coordinates[:, 3] == 2)
        & np.isclose(bc.fixed_values, -0.2)
    )


def test_torsion_rotates_top_surface_around_center():
    model = Model.from_array(np.ones((3, 3, 3)), spacing=(1, 1, 1))

    bc = Torsion(axis="z", twist_angle_degrees=1.0).generate(model)

    node = (0, 0, 3)
    x_value = bc.fixed_values[np.all(bc.fixed_coordinates == (*node, 0), axis=1)][0]
    y_value = bc.fixed_values[np.all(bc.fixed_coordinates == (*node, 1), axis=1)][0]
    z_value = bc.fixed_values[np.all(bc.fixed_coordinates == (*node, 2), axis=1)][0]
    assert x_value == pytest.approx(0.0264070667)
    assert y_value == pytest.approx(-0.0259501524)
    assert z_value == pytest.approx(0.0)


def test_bending_tilts_top_and_bottom_surfaces_in_opposing_directions():
    model = Model.from_array(np.ones((3, 3, 3)), spacing=(1, 1, 1))

    bc = Bending(
        axis="z",
        bending_angle_degrees=1.0,
        neutral_axis_angle_degrees=90.0,
    ).generate(model)

    top_left = bc.fixed_values[np.all(bc.fixed_coordinates == (0, 0, 3, 2), axis=1)][0]
    bottom_left = bc.fixed_values[np.all(bc.fixed_coordinates == (0, 0, 0, 2), axis=1)][
        0
    ]
    assert top_left == pytest.approx(-0.0130903013)
    assert bottom_left == pytest.approx(0.0130903013)
