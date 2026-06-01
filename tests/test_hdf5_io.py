from pathlib import Path

import h5py
import numpy as np
import pytest

from parosol_py.boundary_conditions import axial_compression
from parosol_py.hdf5_io import write_parosol_input


def test_write_parosol_input_schema(tmp_path: Path):
    stiffness_xyz = np.ones((3, 2, 4), dtype=np.float32)
    coords, values = axial_compression(stiffness_xyz, axis="z", strain=-0.01)
    out = tmp_path / "case.h5"

    write_parosol_input(
        out,
        stiffness_gpa_xyz=stiffness_xyz,
        fixed_displacement_coordinates=coords,
        fixed_displacement_values=values,
        voxel_size_mm=0.061,
        poisson_ratio=0.3,
    )

    with h5py.File(out, "r") as h5:
        group = h5["Image_Data"]
        assert set(group.keys()) == {
            "Fixed_Displacement_Coordinates",
            "Fixed_Displacement_Values",
            "Image",
            "Loaded_Nodes_Coordinates",
            "Loaded_Nodes_Values",
            "Poisons_ratio",
            "Voxelsize",
        }
        assert group["Image"].shape == (4, 2, 3)
        assert np.array_equal(group["Image"][...], np.swapaxes(stiffness_xyz, 0, 2))
        assert float(group["Voxelsize"][()]) == 0.061
        assert float(group["Poisons_ratio"][()]) == 0.3


def test_write_parosol_input_reorders_bc_coordinates_for_native_reader(tmp_path: Path):
    stiffness_xyz = np.ones((3, 2, 4), dtype=np.float32)
    coords_xyz = np.array([[2, 1, 4, 2], [0, 0, 0, 2]], dtype=np.uint16)
    values = np.array([-0.04, 1e-16], dtype=np.float32)
    out = tmp_path / "coords.h5"

    write_parosol_input(
        out,
        stiffness_gpa_xyz=stiffness_xyz,
        fixed_displacement_coordinates=coords_xyz,
        fixed_displacement_values=values,
        voxel_size_mm=0.061,
        poisson_ratio=0.3,
    )

    with h5py.File(out, "r") as h5:
        assert np.array_equal(
            h5["Image_Data/Fixed_Displacement_Coordinates"][...],
            np.array([[4, 1, 2, 2], [0, 0, 0, 2]], dtype=np.uint16),
        )


def test_write_parosol_input_writes_loaded_nodes(tmp_path: Path):
    path = write_parosol_input(
        tmp_path / "case.h5",
        stiffness_gpa_xyz=np.ones((2, 2, 2), dtype=np.float32),
        fixed_displacement_coordinates=np.array([[0, 0, 0, 0]], dtype=np.uint16),
        fixed_displacement_values=np.array([1e-16], dtype=np.float32),
        loaded_node_coordinates=np.array([[2, 2, 2, 2]], dtype=np.uint16),
        loaded_node_values=np.array([-10.0], dtype=np.float32),
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
    )

    with h5py.File(path, "r") as h5:
        group = h5["Image_Data"]
        np.testing.assert_array_equal(
            group["Loaded_Nodes_Coordinates"][...],
            np.array([[2, 2, 2, 2]], dtype=np.uint16),
        )
        np.testing.assert_allclose(group["Loaded_Nodes_Values"][...], [-10.0])


def test_write_parosol_input_rejects_coordinate_outside_node_bounds(tmp_path: Path):
    stiffness_xyz = np.ones((3, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="bounds"):
        write_parosol_input(
            tmp_path / "bad_coords.h5",
            stiffness_gpa_xyz=stiffness_xyz,
            fixed_displacement_coordinates=np.array([[4, 0, 0, 0]]),
            fixed_displacement_values=np.array([0.0], dtype=np.float32),
            voxel_size_mm=0.061,
            poisson_ratio=0.3,
        )


def test_write_parosol_input_rejects_coordinates_outside_native_coordinate_range(
    tmp_path: Path,
):
    stiffness_xyz = np.ones((32768, 1, 1), dtype=np.float32)
    coords = np.array([[32768, 0, 0, 0]], dtype=np.int64)
    values = np.array([1e-16], dtype=np.float32)

    with pytest.raises(ValueError, match="native|int16|range"):
        write_parosol_input(
            tmp_path / "bad_native_coords.h5",
            stiffness_gpa_xyz=stiffness_xyz,
            fixed_displacement_coordinates=coords,
            fixed_displacement_values=values,
            voxel_size_mm=0.061,
            poisson_ratio=0.3,
        )


def test_write_parosol_input_rejects_non_finite_stiffness(tmp_path: Path):
    stiffness_xyz = np.ones((3, 2, 4), dtype=np.float32)
    stiffness_xyz[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        write_parosol_input(
            tmp_path / "bad_stiffness.h5",
            stiffness_gpa_xyz=stiffness_xyz,
            fixed_displacement_coordinates=np.array([[0, 0, 0, 0]]),
            fixed_displacement_values=np.array([0.0], dtype=np.float32),
            voxel_size_mm=0.061,
            poisson_ratio=0.3,
        )


def test_write_parosol_input_rejects_non_positive_voxel_size(tmp_path: Path):
    stiffness_xyz = np.ones((3, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="voxel_size"):
        write_parosol_input(
            tmp_path / "bad_voxel_size.h5",
            stiffness_gpa_xyz=stiffness_xyz,
            fixed_displacement_coordinates=np.array([[0, 0, 0, 0]]),
            fixed_displacement_values=np.array([0.0], dtype=np.float32),
            voxel_size_mm=0.0,
            poisson_ratio=0.3,
        )
