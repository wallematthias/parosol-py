from __future__ import annotations

import numpy as np

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
MAX_NATIVE_COORDINATE = np.iinfo(np.int16).max


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
    if np.any(dims > MAX_NATIVE_COORDINATE):
        raise ValueError(
            f"stiffness dimensions must be within native int16 coordinate range "
            f"(<= {MAX_NATIVE_COORDINATE}), got shape {stiffness.shape}"
        )
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
        bottom = base.copy()
        bottom[axis_index] = 0
        for direction in range(3):
            coords.append([*bottom, direction])
            values.append(1e-16)

        top = base.copy()
        top[axis_index] = int(node_max)
        coords.append([*top, axis_index])
        values.append(float(displacement))

    if not coords:
        raise ValueError("No non-zero stiffness voxels found for boundary conditions")
    return np.asarray(coords, dtype=np.uint16), np.asarray(values, dtype=np.float32)
