import numpy as np

from parosol_py import BoundaryConditionSet, Model, OutputProfile, SolverProfile


def test_model_normalizes_material_to_xyz_and_tracks_spacing():
    material_zyx = np.arange(24, dtype=np.float32).reshape((2, 3, 4))

    model = Model.from_array(
        material_zyx,
        spacing=(0.061, 0.061, 0.061),
        origin=(1.0, 2.0, 3.0),
        array_order="zyx",
        material_unit="MPa",
    )

    assert model.material_xyz.shape == (4, 3, 2)
    assert model.spacing == (0.061, 0.061, 0.061)
    assert model.origin == (1.0, 2.0, 3.0)
    assert model.material_unit == "MPa"
    np.testing.assert_array_equal(
        model.material_xyz, np.transpose(material_zyx, (2, 1, 0))
    )


def test_boundary_condition_set_serializes_fixed_and_loaded_nodes():
    bc = BoundaryConditionSet(
        fixed_coordinates=np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint16),
        fixed_values=np.array([1e-16, 1e-16], dtype=np.float32),
        loaded_coordinates=np.array([[1, 1, 1, 2]], dtype=np.uint16),
        loaded_values=np.array([-12.5], dtype=np.float32),
        node_sets={"top": [(1, 1, 1)]},
    )

    data = bc.to_dict()
    restored = BoundaryConditionSet.from_dict(data)

    np.testing.assert_array_equal(restored.fixed_coordinates, bc.fixed_coordinates)
    np.testing.assert_array_equal(restored.loaded_coordinates, bc.loaded_coordinates)
    assert restored.node_sets == {"top": [(1, 1, 1)]}


def test_profiles_have_stable_defaults():
    solver = SolverProfile()
    output = OutputProfile()

    assert solver.tolerance == 1e-6
    assert solver.level == 6
    assert solver.mpi_processes == 1
    assert output.export_fields is True
    assert output.image_fields == ("sed",)
