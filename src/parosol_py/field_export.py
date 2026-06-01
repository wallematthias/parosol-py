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
            coords = np.argwhere(np.asarray(self.stiffness_gpa_xyz) > 0)
            self._active_coordinates = _coords_to_tuples(_morton_sorted(coords))
        return self._active_coordinates

    @property
    def dense_coordinates(self) -> list[tuple[int, int, int]]:
        if self._dense_coordinates is None:
            x_dim, y_dim, z_dim = (
                int(v) for v in np.asarray(self.stiffness_gpa_xyz).shape
            )
            coords = np.stack(
                np.meshgrid(
                    np.arange(x_dim, dtype=np.int64),
                    np.arange(y_dim, dtype=np.int64),
                    np.arange(z_dim, dtype=np.int64),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 3)
            self._dense_coordinates = _coords_to_tuples(
                _morton_sorted(coords)
            )
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
        coords_array = np.asarray(coords, dtype=np.int64)
        dense[coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]] = array
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


def morton_keys(coords: np.ndarray) -> np.ndarray:
    values = np.asarray(coords, dtype=np.uint64)
    if values.size == 0:
        return np.zeros((0,), dtype=np.uint64)
    keys = np.zeros(values.shape[0], dtype=np.uint64)
    limit = int(values.max())
    bit_index = 0
    while (1 << bit_index) <= limit:
        keys |= ((values[:, 0] >> bit_index) & 1) << (3 * bit_index)
        keys |= ((values[:, 1] >> bit_index) & 1) << (3 * bit_index + 1)
        keys |= ((values[:, 2] >> bit_index) & 1) << (3 * bit_index + 2)
        bit_index += 1
    return keys


def _morton_sorted(coords: np.ndarray) -> np.ndarray:
    values = np.asarray(coords, dtype=np.int64)
    if values.size == 0:
        return values.reshape((0, 3))
    order = np.argsort(morton_keys(values), kind="stable")
    return values[order]


def _coords_to_tuples(coords: np.ndarray) -> list[tuple[int, int, int]]:
    return [tuple(int(v) for v in coord) for coord in np.asarray(coords)]
