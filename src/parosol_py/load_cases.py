from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary_conditions import AXIS_TO_INDEX, axial_compression
from .core import BoundaryConditionSet, Model
from .surfaces import SurfaceSelection, top_bottom_surface_nodes


@dataclass(frozen=True)
class ConstrainedAxialCompression:
    axis: str = "z"
    strain: float = -0.01
    displacement: float | None = None
    surface: SurfaceSelection | str | dict | None = None

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        if self.surface is not None:
            return _constrained_axial_from_surfaces(
                model,
                axis=axis,
                displacement=_displacement_from_strain(
                    self.displacement,
                    strain=self.strain,
                    height=model.material_xyz.shape[AXIS_TO_INDEX[axis]]
                    * model.spacing[AXIS_TO_INDEX[axis]],
                ),
                surface=self.surface,
            )
        axis_index = AXIS_TO_INDEX[axis]
        strain = _strain_from_displacement(
            self.displacement,
            strain=self.strain,
            height=model.material_xyz.shape[axis_index] * model.spacing[axis_index],
        )
        coords, values = axial_compression(
            model.material_xyz,
            axis=axis,
            strain=strain,
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
    displacement: float | None = None
    surface: SurfaceSelection | str | dict | None = None

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        axis_index = AXIS_TO_INDEX[axis]
        dimensions = tuple(int(v) for v in model.material_xyz.shape)
        height = dimensions[axis_index] * model.spacing[axis_index]
        displacement = _displacement_from_strain(
            self.displacement,
            strain=self.strain,
            height=height,
        )
        node_sets = _top_bottom_sets(model.material_xyz, axis, surface=self.surface)
        constraints: dict[tuple[int, int, int, int], float] = {}

        for node in node_sets["bottom"]:
            constraints[(*node, axis_index)] = 0.0
        for node in node_sets["top"]:
            constraints[(*node, axis_index)] = displacement

        coords = np.asarray([coord for coord in sorted(constraints)], dtype=np.uint16)
        values = np.asarray(
            [constraints[tuple(coord)] for coord in coords], dtype=np.float32
        )
        return BoundaryConditionSet(
            fixed_coordinates=coords,
            fixed_values=values,
            node_sets=node_sets,
        )


@dataclass(frozen=True)
class BodyWeightCompression:
    axis: str = "z"
    force_n: float = -1.0
    surface: SurfaceSelection | str | dict | None = None

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        axis_index = AXIS_TO_INDEX[axis]
        base = ConstrainedAxialCompression(
            axis=axis,
            strain=0.0,
            surface=self.surface,
        ).generate(model)
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
    displacement: float | None = None
    surface: SurfaceSelection | str | dict | None = None

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        axis_index = AXIS_TO_INDEX[axis]
        base = ConstrainedAxialCompression(
            axis=axis,
            strain=self.strain,
            displacement=self.displacement,
            surface=self.surface,
        ).generate(model)
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
        values = np.asarray(
            [constraints[tuple(coord)] for coord in coords], dtype=np.float32
        )
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
    displacement: float | None = None
    vector: tuple[float, float] | None = None
    surface: SurfaceSelection | str | dict | None = None

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        direction = _axis_token(self.direction)
        axis_index = AXIS_TO_INDEX[axis]
        direction_index = AXIS_TO_INDEX[direction]
        if axis_index == direction_index:
            raise ValueError("shear axis and direction must differ")

        bc = ConstrainedAxialCompression(
            axis=axis,
            strain=0.0,
            surface=self.surface,
        ).generate(model)
        height = model.material_xyz.shape[axis_index] * model.spacing[axis_index]
        fixed = bc.fixed_coordinates.copy()
        values = bc.fixed_values.copy()
        top_axis_value = model.material_xyz.shape[axis_index]
        if self.vector is None:
            displacement = _displacement_from_strain(
                self.displacement,
                strain=self.strain,
                height=height,
            )
            mask = (fixed[:, axis_index] == top_axis_value) & (
                fixed[:, 3] == direction_index
            )
            values[mask] = displacement
        else:
            lateral_axes = [idx for idx in range(3) if idx != axis_index]
            vector = tuple(float(v) for v in self.vector)
            if len(vector) != 2:
                raise ValueError("shear vector must contain exactly two values")
            for lateral_axis, component in zip(lateral_axes, vector, strict=True):
                mask = (fixed[:, axis_index] == top_axis_value) & (
                    fixed[:, 3] == lateral_axis
                )
                values[mask] = component * float(height)
        return BoundaryConditionSet(
            fixed_coordinates=fixed,
            fixed_values=values,
            loaded_coordinates=bc.loaded_coordinates,
            loaded_values=bc.loaded_values,
            node_sets=bc.node_sets,
        )


@dataclass(frozen=True)
class Torsion:
    axis: str = "z"
    twist_angle_degrees: float = 1.0
    center: tuple[float, float] | None = None

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        axis_index = AXIS_TO_INDEX[axis]
        lateral_axes = [idx for idx in range(3) if idx != axis_index]
        node_sets = _top_bottom_sets(model.material_xyz, axis)
        center = _center_on_lateral_plane(
            model.material_xyz,
            model.spacing,
            lateral_axes,
            self.center,
        )
        angle = np.deg2rad(float(self.twist_angle_degrees))
        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))
        constraints: dict[tuple[int, int, int, int], float] = {}

        for node in node_sets["bottom"]:
            for direction in range(3):
                constraints[(*node, direction)] = 0.0

        for node in node_sets["top"]:
            position = _physical_node_position(node, model.spacing)
            u = position[lateral_axes[0]] - center[0]
            v = position[lateral_axes[1]] - center[1]
            rotated_u = cos_angle * u - sin_angle * v
            rotated_v = sin_angle * u + cos_angle * v
            constraints[(*node, lateral_axes[0])] = rotated_u - u
            constraints[(*node, lateral_axes[1])] = rotated_v - v
            constraints[(*node, axis_index)] = 0.0

        coords = np.asarray([coord for coord in sorted(constraints)], dtype=np.uint16)
        values = np.asarray(
            [constraints[tuple(coord)] for coord in coords], dtype=np.float32
        )
        return BoundaryConditionSet(
            fixed_coordinates=coords,
            fixed_values=values,
            node_sets=node_sets,
        )


@dataclass(frozen=True)
class Bending:
    axis: str = "z"
    bending_angle_degrees: float = 1.0
    neutral_axis_angle_degrees: float = 90.0
    center: tuple[float, float] | None = None

    def generate(self, model: Model) -> BoundaryConditionSet:
        axis = _axis_token(self.axis)
        axis_index = AXIS_TO_INDEX[axis]
        lateral_axes = [idx for idx in range(3) if idx != axis_index]
        node_sets = _top_bottom_sets(model.material_xyz, axis)
        center = _center_on_lateral_plane(
            model.material_xyz,
            model.spacing,
            lateral_axes,
            self.center,
        )
        neutral_angle = np.deg2rad(float(self.neutral_axis_angle_degrees))
        tilt = np.tan(np.deg2rad(float(self.bending_angle_degrees)) / 2.0)
        constraints: dict[tuple[int, int, int, int], float] = {}

        for surface, sign in (("bottom", -1.0), ("top", 1.0)):
            for node in node_sets[surface]:
                position = _physical_node_position(node, model.spacing)
                u = position[lateral_axes[0]] - center[0]
                v = position[lateral_axes[1]] - center[1]
                distance = np.sin(neutral_angle) * u + np.cos(neutral_angle) * v
                for direction in lateral_axes:
                    constraints[(*node, direction)] = 0.0
                constraints[(*node, axis_index)] = sign * float(distance) * float(tilt)

        coords = np.asarray([coord for coord in sorted(constraints)], dtype=np.uint16)
        values = np.asarray(
            [constraints[tuple(coord)] for coord in coords], dtype=np.float32
        )
        return BoundaryConditionSet(
            fixed_coordinates=coords,
            fixed_values=values,
            node_sets=node_sets,
        )


def _top_bottom_sets(
    stiffness_xyz: np.ndarray,
    axis: str,
    surface: SurfaceSelection | str | dict | None = None,
) -> dict[str, list[tuple[int, int, int]]]:
    return top_bottom_surface_nodes(stiffness_xyz, axis=axis, selection=surface)


def _constrained_axial_from_surfaces(
    model: Model,
    *,
    axis: str,
    displacement: float,
    surface: SurfaceSelection | str | dict,
) -> BoundaryConditionSet:
    axis_index = AXIS_TO_INDEX[axis]
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    node_sets = _top_bottom_sets(model.material_xyz, axis, surface=surface)
    constraints: dict[tuple[int, int, int, int], float] = {}
    for node in node_sets["bottom"]:
        for direction in range(3):
            constraints[(*node, direction)] = 0.0
    for node in node_sets["top"]:
        for direction in lateral_axes:
            constraints[(*node, direction)] = 0.0
        constraints[(*node, axis_index)] = float(displacement)
    coords = np.asarray([coord for coord in sorted(constraints)], dtype=np.uint16)
    values = np.asarray(
        [constraints[tuple(coord)] for coord in coords], dtype=np.float32
    )
    return BoundaryConditionSet(
        fixed_coordinates=coords,
        fixed_values=values,
        node_sets=node_sets,
    )


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


def _physical_node_position(
    node: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        (float(index) - 0.5) * float(step) for index, step in zip(node, spacing)
    )


def _center_on_lateral_plane(
    stiffness_xyz: np.ndarray,
    spacing: tuple[float, float, float],
    lateral_axes: list[int],
    center: tuple[float, float] | None,
) -> tuple[float, float]:
    if center is not None:
        if len(center) != 2:
            raise ValueError("center must contain exactly two coordinates")
        return tuple(float(value) for value in center)
    dimensions = tuple(int(value) for value in stiffness_xyz.shape)
    return tuple(
        ((float(dimensions[axis]) - 1.0) / 2.0) * float(spacing[axis])
        for axis in lateral_axes
    )


def _displacement_from_strain(
    displacement: float | None,
    *,
    strain: float,
    height: float,
) -> float:
    if displacement is not None:
        return float(displacement)
    return float(strain) * float(height)


def _strain_from_displacement(
    displacement: float | None,
    *,
    strain: float,
    height: float,
) -> float:
    if displacement is None:
        return float(strain)
    if np.isclose(height, 0.0):
        raise ValueError("cannot convert displacement to strain for zero height")
    return float(displacement) / float(height)


def _axis_token(axis: str) -> str:
    token = axis.strip().lower()
    if token not in AXIS_TO_INDEX:
        raise ValueError("axis must be one of: x, y, z")
    return token
