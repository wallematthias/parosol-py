from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary_conditions import AXIS_TO_INDEX, axial_compression
from .core import BoundaryConditionSet, Model


@dataclass(frozen=True)
class AxialCompression:
    axis: str = "z"
    strain: float = -0.01

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        axis_index = AXIS_TO_INDEX[axis]
        coords, values = axial_compression(
            model.material_xyz,
            axis=axis,
            strain=self.strain,
            voxel_size_mm=model.spacing[axis_index],
        )
        return BoundaryConditionSet(
            fixed_coordinates=coords,
            fixed_values=values,
            node_sets=_top_bottom_sets(model.material_xyz, axis),
        )


@dataclass(frozen=True)
class UniaxialCompression:
    axis: str = "z"
    strain: float = -0.01

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        axis_index = AXIS_TO_INDEX[axis]
        dimensions = tuple(int(v) for v in model.material_xyz.shape)
        height = dimensions[axis_index] * model.spacing[axis_index]
        displacement = float(self.strain) * float(height)
        node_sets = _top_bottom_sets(model.material_xyz, axis)
        constraints: dict[tuple[int, int, int, int], float] = {}

        for node in node_sets["bottom"]:
            constraints[(*node, axis_index)] = 0.0
        for node in node_sets["top"]:
            constraints[(*node, axis_index)] = displacement

        coords = np.asarray([coord for coord in sorted(constraints)], dtype=np.uint16)
        values = np.asarray([constraints[tuple(coord)] for coord in coords], dtype=np.float32)
        return BoundaryConditionSet(
            fixed_coordinates=coords,
            fixed_values=values,
            node_sets=node_sets,
        )


@dataclass(frozen=True)
class BodyWeightCompression:
    axis: str = "z"
    force_n: float = -1.0

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        axis_index = AXIS_TO_INDEX[axis]
        base = AxialCompression(axis=axis, strain=0.0).generate(model)
        top = base.node_sets["top"]
        values = np.full((len(top),), float(self.force_n) / len(top), dtype=np.float32)
        coords = np.asarray([(*coord, axis_index) for coord in top], dtype=np.uint16)
        return BoundaryConditionSet(
            fixed_coordinates=base.fixed_coordinates,
            fixed_values=base.fixed_values,
            loaded_coordinates=coords,
            loaded_values=values,
            node_sets=base.node_sets,
        )


@dataclass(frozen=True)
class ConfinedCompression:
    axis: str = "z"
    strain: float = -0.01

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        axis_index = AXIS_TO_INDEX[axis]
        base = AxialCompression(axis=axis, strain=self.strain).generate(model)
        constraints = {
            tuple(int(v) for v in coord): float(value)
            for coord, value in zip(base.fixed_coordinates, base.fixed_values)
        }
        dimensions = tuple(int(v) for v in model.material_xyz.shape)
        lateral_axes = [idx for idx in range(3) if idx != axis_index]

        for node in _active_nodes(model.material_xyz):
            for direction in lateral_axes:
                if node[direction] in {0, dimensions[direction]}:
                    constraints[(*node, direction)] = 0.0

        coords = np.asarray([coord for coord in sorted(constraints)], dtype=np.uint16)
        values = np.asarray([constraints[tuple(coord)] for coord in coords], dtype=np.float32)
        return BoundaryConditionSet(
            fixed_coordinates=coords,
            fixed_values=values,
            loaded_coordinates=base.loaded_coordinates,
            loaded_values=base.loaded_values,
            node_sets=base.node_sets,
        )


@dataclass(frozen=True)
class SimpleShear:
    axis: str = "z"
    direction: str = "x"
    strain: float = 0.01

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        direction = _axis_token(self.direction)
        axis_index = AXIS_TO_INDEX[axis]
        direction_index = AXIS_TO_INDEX[direction]
        if axis_index == direction_index:
            raise ValueError("shear axis and direction must differ")

        bc = AxialCompression(axis=axis, strain=0.0).generate(model)
        height = model.material_xyz.shape[axis_index] * model.spacing[axis_index]
        displacement = float(self.strain) * float(height)
        fixed = bc.fixed_coordinates.copy()
        values = bc.fixed_values.copy()
        top_axis_value = model.material_xyz.shape[axis_index]
        mask = (fixed[:, axis_index] == top_axis_value) & (fixed[:, 3] == direction_index)
        values[mask] = displacement
        return BoundaryConditionSet(
            fixed_coordinates=fixed,
            fixed_values=values,
            loaded_coordinates=bc.loaded_coordinates,
            loaded_values=bc.loaded_values,
            node_sets=bc.node_sets,
        )


def _top_bottom_sets(
    stiffness_xyz: np.ndarray,
    axis: str,
) -> dict[str, list[tuple[int, int, int]]]:
    axis_index = AXIS_TO_INDEX[axis]
    dims = tuple(int(v) for v in stiffness_xyz.shape)
    occupied = np.asarray(stiffness_xyz) > 0
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    out = {"bottom": set(), "top": set()}
    for label, element_axis_value, node_axis_value in (
        ("bottom", 0, 0),
        ("top", dims[axis_index] - 1, dims[axis_index]),
    ):
        surface = np.take(occupied, indices=element_axis_value, axis=axis_index)
        for lateral_index in np.argwhere(surface):
            base = [0, 0, 0]
            base[axis_index] = node_axis_value
            base[lateral_axes[0]] = int(lateral_index[0])
            base[lateral_axes[1]] = int(lateral_index[1])
            for du in (0, 1):
                for dv in (0, 1):
                    node = base.copy()
                    node[lateral_axes[0]] += du
                    node[lateral_axes[1]] += dv
                    out[label].add(tuple(node))
    return {name: sorted(coords) for name, coords in out.items()}


def _active_nodes(stiffness_xyz: np.ndarray) -> set[tuple[int, int, int]]:
    nodes: set[tuple[int, int, int]] = set()
    for element in np.argwhere(np.asarray(stiffness_xyz) > 0):
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    nodes.add(
                        (
                            int(element[0]) + dx,
                            int(element[1]) + dy,
                            int(element[2]) + dz,
                        )
                    )
    return nodes


def _axis_token(axis: str) -> str:
    token = axis.strip().lower()
    if token not in AXIS_TO_INDEX:
        raise ValueError("axis must be one of: x, y, z")
    return token
