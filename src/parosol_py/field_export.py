from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NativeFieldMapper:
    stiffness_gpa_xyz: np.ndarray
    _active_coordinates: list[tuple[int, int, int]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _dense_coordinates: list[tuple[int, int, int]] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def active_coordinates(self) -> list[tuple[int, int, int]]:
        if self._active_coordinates is None:
            coords = [
                tuple(int(v) for v in coord)
                for coord in np.argwhere(np.asarray(self.stiffness_gpa_xyz) > 0)
            ]
            self._active_coordinates = sorted(coords, key=lambda coord: morton_key(*coord))
        return self._active_coordinates

    @property
    def dense_coordinates(self) -> list[tuple[int, int, int]]:
        if self._dense_coordinates is None:
            x_dim, y_dim, z_dim = (int(v) for v in np.asarray(self.stiffness_gpa_xyz).shape)
            coords = [
                (x, y, z)
                for x in range(x_dim)
                for y in range(y_dim)
                for z in range(z_dim)
            ]
            self._dense_coordinates = sorted(coords, key=lambda coord: morton_key(*coord))
        return self._dense_coordinates

    def scalar_to_dense(self, values) -> np.ndarray:
        array = np.asarray(values).reshape(-1)
        stiffness = np.asarray(self.stiffness_gpa_xyz)
        if array.size == stiffness.size:
            coords = self.dense_coordinates
        elif array.size == len(self.active_coordinates):
            coords = self.active_coordinates
        else:
            raise ValueError(
                f"field has {array.size} values, expected dense size {stiffness.size} "
                f"or active size {len(self.active_coordinates)}"
            )

        dense = np.zeros(stiffness.shape, dtype=array.dtype)
        for index, coord in enumerate(coords):
            dense[coord] = array[index]
        return dense


def morton_key(x: int, y: int, z: int) -> int:
    key = 0
    bit_index = 0
    limit = max(x, y, z)
    while (1 << bit_index) <= limit:
        key |= ((x >> bit_index) & 1) << (3 * bit_index)
        key |= ((y >> bit_index) & 1) << (3 * bit_index + 1)
        key |= ((z >> bit_index) & 1) << (3 * bit_index + 2)
        bit_index += 1
    return key
