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
    coords = np.asarray(fixed_displacement_coordinates)
    values = np.asarray(fixed_displacement_values, dtype=np.float32)
    if stiffness.ndim != 3:
        raise ValueError(f"stiffness_gpa_xyz must be 3D, got shape {stiffness.shape}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("fixed_displacement_coordinates must have shape (n, 4)")
    if values.shape != (coords.shape[0],):
        raise ValueError("fixed_displacement_values must have shape (n,)")
    if not np.all(np.isfinite(stiffness)):
        raise ValueError("stiffness_gpa_xyz must contain only finite values")
    if np.any(stiffness < 0):
        raise ValueError("stiffness_gpa_xyz must contain only non-negative values")
    if not np.all(np.isfinite(coords)):
        raise ValueError("fixed_displacement_coordinates must contain only finite values")
    if np.any(coords < 0):
        raise ValueError("fixed_displacement_coordinates must contain only non-negative values")
    if not np.all(coords == np.floor(coords)):
        raise ValueError("fixed_displacement_coordinates must contain integer values")
    uint16_max = np.iinfo(np.uint16).max
    if np.any(coords > uint16_max):
        raise ValueError("fixed_displacement_coordinates exceed uint16 storage range")
    if not np.all(np.isin(coords[:, 3], [0, 1, 2])):
        raise ValueError("fixed_displacement_coordinates direction must be one of {0, 1, 2}")

    node_max_xyz = np.array(stiffness.shape, dtype=np.float64)
    if np.any(coords[:, :3] > node_max_xyz):
        raise ValueError("fixed_displacement_coordinates exceed node bounds")
    if not np.all(np.isfinite(values)):
        raise ValueError("fixed_displacement_values must contain only finite values")
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be positive")
    if not np.isfinite(poisson_ratio) or not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must satisfy -1.0 < nu < 0.5")

    coords = coords.astype(np.uint16, copy=False)
    coords_zyx = coords[:, [2, 1, 0, 3]]

    with h5py.File(out, "w") as h5:
        group = h5.create_group("Image_Data")
        group.create_dataset("Fixed_Displacement_Coordinates", data=coords_zyx)
        group.create_dataset("Fixed_Displacement_Values", data=values)
        group.create_dataset("Poisons_ratio", data=float(poisson_ratio))
        group.create_dataset("Voxelsize", data=float(voxel_size_mm))
        group.create_dataset("Image", data=np.swapaxes(stiffness, 0, 2))
    return out
