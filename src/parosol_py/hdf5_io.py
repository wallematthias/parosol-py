from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_parosol_input(
    path: str | Path,
    *,
    stiffness_gpa_xyz,
    fixed_displacement_coordinates,
    fixed_displacement_values,
    voxel_size_mm: float,
    poisson_ratio: float,
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    stiffness = np.asarray(stiffness_gpa_xyz, dtype=np.float32)
    coords = np.asarray(fixed_displacement_coordinates, dtype=np.uint16)
    values = np.asarray(fixed_displacement_values, dtype=np.float32)
    if stiffness.ndim != 3:
        raise ValueError(f"stiffness_gpa_xyz must be 3D, got shape {stiffness.shape}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("fixed_displacement_coordinates must have shape (n, 4)")
    if values.shape != (coords.shape[0],):
        raise ValueError("fixed_displacement_values must have shape (n,)")

    with h5py.File(out, "w") as h5:
        group = h5.create_group("Image_Data")
        group.create_dataset("Fixed_Displacement_Coordinates", data=coords)
        group.create_dataset("Fixed_Displacement_Values", data=values)
        group.create_dataset("Poisons_ratio", data=float(poisson_ratio))
        group.create_dataset("Voxelsize", data=float(voxel_size_mm))
        group.create_dataset("Image", data=np.swapaxes(stiffness, 0, 2))
    return out
