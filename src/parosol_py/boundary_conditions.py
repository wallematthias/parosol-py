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

    occupied = stiffness > 0.0
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    bottom_slice = np.take(occupied, indices=0, axis=axis_index)
    top_slice = np.take(occupied, indices=node_max - 1, axis=axis_index)
    constraints: dict[tuple[int, int, int, int], float] = {}

    def add_face_constraints(surface, *, node_axis_value: int, top: bool) -> None:
        for lateral_index in np.argwhere(surface):
            base = [0, 0, 0]
            base[axis_index] = int(node_axis_value)
            base[lateral_axes[0]] = int(lateral_index[0])
            base[lateral_axes[1]] = int(lateral_index[1])
            for du in (0, 1):
                for dv in (0, 1):
                    node = base.copy()
                    node[lateral_axes[0]] += du
                    node[lateral_axes[1]] += dv
                    if top:
                        for direction in lateral_axes:
                            constraints[(*node, direction)] = 1e-16
                        constraints[(*node, axis_index)] = float(displacement)
                    else:
                        for direction in range(3):
                            constraints[(*node, direction)] = 1e-16

    add_face_constraints(bottom_slice, node_axis_value=0, top=False)
    add_face_constraints(top_slice, node_axis_value=node_max, top=True)

    if not constraints:
        raise ValueError("No non-zero stiffness voxels found for boundary conditions")
    coords = [list(coord) for coord in sorted(constraints)]
    values = [constraints[tuple(coord)] for coord in coords]
    return np.asarray(coords, dtype=np.uint16), np.asarray(values, dtype=np.float32)
