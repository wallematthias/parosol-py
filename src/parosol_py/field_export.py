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

    @property
    def active_node_coordinates(self) -> list[tuple[int, int, int]]:
        elements = np.argwhere(np.asarray(self.stiffness_gpa_xyz) > 0)
        if elements.size == 0:
            return []
        offsets = np.asarray(
            [
                (dx, dy, dz)
                for dx in (0, 1)
                for dy in (0, 1)
                for dz in (0, 1)
            ],
            dtype=np.int64,
        )
        nodes = (elements[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        nodes = np.unique(nodes, axis=0)
        return _coords_to_tuples(_morton_sorted(nodes))

    def nodal_vector_to_dense_element(self, values) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("nodal vector field must have shape (n, 3)")
        node_coords = self.active_node_coordinates
        if array.shape[0] != len(node_coords):
            raise ValueError(
                f"nodal vector field has {array.shape[0]} nodes, expected {len(node_coords)}"
            )
        node_values = {
            coord: array[index]
            for index, coord in enumerate(node_coords)
        }
        stiffness = np.asarray(self.stiffness_gpa_xyz)
        dense = np.zeros((*stiffness.shape, 3), dtype=array.dtype)
        offsets = [
            (dx, dy, dz)
            for dx in (0, 1)
            for dy in (0, 1)
            for dz in (0, 1)
        ]
        for element in np.argwhere(stiffness > 0):
            coord = tuple(int(value) for value in element)
            corners = [
                node_values[(coord[0] + dx, coord[1] + dy, coord[2] + dz)]
                for dx, dy, dz in offsets
            ]
            dense[coord] = np.mean(corners, axis=0)
        return dense

    def mesh_vector_to_dense_element(self, coordinates, elements, values) -> np.ndarray:
        array = np.asarray(values)
        node_coords = np.asarray(coordinates)
        element_nodes = np.asarray(elements)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("mesh vector field must have shape (n_nodes, 3)")
        if node_coords.ndim != 2 or node_coords.shape[1] != 3:
            raise ValueError("mesh coordinates must have shape (n_nodes, 3)")
        if element_nodes.ndim != 2 or element_nodes.shape[1] != 8:
            raise ValueError("mesh elements must have shape (n_elements, 8)")
        if array.shape[0] != node_coords.shape[0]:
            raise ValueError(
                f"mesh vector field has {array.shape[0]} nodes, expected {node_coords.shape[0]}"
            )
        if element_nodes.size and (element_nodes.min() < 0 or element_nodes.max() >= node_coords.shape[0]):
            raise ValueError("mesh elements reference node indices outside coordinates")

        stiffness = np.asarray(self.stiffness_gpa_xyz)
        dense = np.zeros((*stiffness.shape, 3), dtype=array.dtype)
        if element_nodes.size == 0:
            return dense

        corner_coords = node_coords[element_nodes]
        element_coords = np.floor(corner_coords.min(axis=1)).astype(np.int64)
        in_bounds = (
            (element_coords[:, 0] >= 0)
            & (element_coords[:, 0] < stiffness.shape[0])
            & (element_coords[:, 1] >= 0)
            & (element_coords[:, 1] < stiffness.shape[1])
            & (element_coords[:, 2] >= 0)
            & (element_coords[:, 2] < stiffness.shape[2])
        )
        if not np.all(in_bounds):
            raise ValueError("mesh element coordinates fall outside dense image bounds")

        element_values = np.mean(array[element_nodes], axis=1)
        dense[
            element_coords[:, 0],
            element_coords[:, 1],
            element_coords[:, 2],
        ] = element_values
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
