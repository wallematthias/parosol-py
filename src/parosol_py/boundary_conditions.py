from __future__ import annotations

import numpy as np

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def axial_compression(
    stiffness_gpa_xyz,
    *,
    axis: str = "z",
    strain: float = -0.01,
) -> tuple[np.ndarray, np.ndarray]:
    stiffness = np.asarray(stiffness_gpa_xyz)
    if stiffness.ndim != 3:
        raise ValueError(f"stiffness_gpa_xyz must be 3D, got shape {stiffness.shape}")
    token = axis.strip().lower()
    if token not in AXIS_TO_INDEX:
        raise ValueError("axis must be one of: x, y, z")

    axis_index = AXIS_TO_INDEX[token]
    dims = np.asarray(stiffness.shape, dtype=np.int64)
    node_max = int(dims[axis_index])
    displacement = float(strain) * float(node_max)

    coords: list[list[int]] = []
    values: list[float] = []

    occupied = stiffness > 0.0
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    projected = np.any(occupied, axis=axis_index)
    for lateral_index in np.argwhere(projected):
        base = [0, 0, 0]
        base[lateral_axes[0]] = int(lateral_index[0])
        base[lateral_axes[1]] = int(lateral_index[1])
        for node_coord, value in ((0, 1e-16), (node_max, displacement)):
            coord = base.copy()
            coord[axis_index] = int(node_coord)
            coord.append(axis_index)
            coords.append(coord)
            values.append(float(value))

    if not coords:
        raise ValueError("No non-zero stiffness voxels found for boundary conditions")
    return np.asarray(coords, dtype=np.uint16), np.asarray(values, dtype=np.float32)
