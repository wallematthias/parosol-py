from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

MAX_NATIVE_COORDINATE = np.iinfo(np.int16).max


def write_parosol_input(
    path: str | Path,
    *,
    stiffness_gpa_xyz,
    fixed_displacement_coordinates,
    fixed_displacement_values,
    voxel_size_mm: float,
    poisson_ratio: float,
    loaded_node_coordinates=None,
    loaded_node_values=None,
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    stiffness = np.asarray(stiffness_gpa_xyz, dtype=np.float32)
    coords = np.asarray(fixed_displacement_coordinates)
    values = np.asarray(fixed_displacement_values, dtype=np.float32)
    loaded_coords = _optional_coordinates(loaded_node_coordinates)
    loaded_values = _optional_values(loaded_node_values)
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
    if np.any(coords[:, :3] > MAX_NATIVE_COORDINATE):
        raise ValueError(
            "fixed_displacement_coordinates exceed native int16 coordinate range"
        )
    if not np.all(np.isin(coords[:, 3], [0, 1, 2])):
        raise ValueError("fixed_displacement_coordinates direction must be one of {0, 1, 2}")
    if loaded_values.shape != (loaded_coords.shape[0],):
        raise ValueError("loaded_node_values must have shape (n,)")
    if not np.all(np.isfinite(loaded_coords)):
        raise ValueError("loaded_node_coordinates must contain only finite values")
    if np.any(loaded_coords < 0):
        raise ValueError("loaded_node_coordinates must contain only non-negative values")
    if not np.all(loaded_coords == np.floor(loaded_coords)):
        raise ValueError("loaded_node_coordinates must contain integer values")
    if np.any(loaded_coords[:, :3] > MAX_NATIVE_COORDINATE):
        raise ValueError("loaded_node_coordinates exceed native int16 coordinate range")
    if not np.all(np.isin(loaded_coords[:, 3], [0, 1, 2])):
        raise ValueError("loaded_node_coordinates direction must be one of {0, 1, 2}")
    if not np.all(np.isfinite(loaded_values)):
        raise ValueError("loaded_node_values must contain only finite values")

    node_max_xyz = np.array(stiffness.shape, dtype=np.float64)
    if np.any(coords[:, :3] > node_max_xyz):
        raise ValueError("fixed_displacement_coordinates exceed node bounds")
    if np.any(loaded_coords[:, :3] > node_max_xyz):
        raise ValueError("loaded_node_coordinates exceed node bounds")
    if not np.all(np.isfinite(values)):
        raise ValueError("fixed_displacement_values must contain only finite values")
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be positive")
    if not np.isfinite(poisson_ratio) or not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must satisfy -1.0 < nu < 0.5")

    coords = coords.astype(np.uint16, copy=False)
    coords_zyx = coords[:, [2, 1, 0, 3]]
    loaded_coords = loaded_coords.astype(np.uint16, copy=False)
    loaded_coords_zyx = loaded_coords[:, [2, 1, 0, 3]]

    with h5py.File(out, "w") as h5:
        group = h5.create_group("Image_Data")
        group.create_dataset("Fixed_Displacement_Coordinates", data=coords_zyx)
        group.create_dataset("Fixed_Displacement_Values", data=values)
        group.create_dataset("Loaded_Nodes_Coordinates", data=loaded_coords_zyx)
        group.create_dataset("Loaded_Nodes_Values", data=loaded_values)
        group.create_dataset("Poisons_ratio", data=float(poisson_ratio))
        group.create_dataset("Voxelsize", data=float(voxel_size_mm))
        group.create_dataset("Image", data=np.swapaxes(stiffness, 0, 2))
    return out


def _optional_coordinates(values) -> np.ndarray:
    if values is None:
        return np.zeros((0, 4), dtype=np.uint16)
    array = np.asarray(values)
    if array.size == 0:
        return array.reshape((0, 4))
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("loaded_node_coordinates must have shape (n, 4)")
    return array


def _optional_values(values) -> np.ndarray:
    if values is None:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)
