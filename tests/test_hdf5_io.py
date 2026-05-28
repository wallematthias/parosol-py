from pathlib import Path

import h5py
import numpy as np

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
            "Poisons_ratio",
            "Voxelsize",
        }
        assert group["Image"].shape == (4, 2, 3)
        assert np.array_equal(group["Image"][...], np.swapaxes(stiffness_xyz, 0, 2))
        assert float(group["Voxelsize"][()]) == 0.061
        assert float(group["Poisons_ratio"][()]) == 0.3
