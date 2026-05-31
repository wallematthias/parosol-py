from __future__ import annotations

import itertools

import numpy as np

from .boundary_conditions import AXIS_TO_INDEX, MAX_NATIVE_COORDINATE
from .core import BoundaryConditionSet

Node = tuple[int, int, int]


def nodes_from_labeled_voxels(
    labels,
    *,
    label: int,
    selection: str = "surface_nodes",
    material=None,
) -> list[Node]:
    mask = np.asarray(labels) == int(label)
    if mask.ndim != 3:
        raise ValueError(f"labels must be 3D, got shape {mask.shape}")
    if np.any(np.asarray(mask.shape, dtype=np.int64) + 1 > MAX_NATIVE_COORDINATE):
        raise ValueError(
            f"label image dimensions must be within native int16 coordinate range "
            f"(<= {MAX_NATIVE_COORDINATE - 1}), got shape {mask.shape}"
        )

    token = selection.strip().lower()
    if token == "all_corner_nodes":
        nodes = _all_corner_nodes(mask)
    elif token == "surface_nodes":
        nodes = _surface_nodes(mask, neighbor_mask=mask)
    elif token == "interface_nodes":
        if material is None:
            raise ValueError("material is required for selection='interface_nodes'")
        material_mask = np.asarray(material) > 0
        if material_mask.shape != mask.shape:
            raise ValueError("material and labels must have the same shape")
        nodes = _interface_nodes(mask, material_mask=material_mask)
    else:
        raise ValueError(
            "selection must be one of: all_corner_nodes, surface_nodes, interface_nodes"
        )

    return sorted(nodes)


def boundary_conditions_from_nodesets(
    node_sets: dict[str, list[Node]],
    *,
    fixed: list[dict] | tuple[dict, ...] = (),
    prescribed: list[dict] | tuple[dict, ...] = (),
    loaded: list[dict] | tuple[dict, ...] = (),
    dimensions_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> BoundaryConditionSet:
    fixed_constraints: dict[tuple[int, int, int, int], float] = {}
    loaded_constraints: dict[tuple[int, int, int, int], float] = {}

    for spec in fixed:
        _add_displacement_spec(
            fixed_constraints,
            node_sets,
            spec,
            dimensions_xyz=dimensions_xyz,
            spacing=spacing,
            default_value=0.0,
        )
    for spec in prescribed:
        _add_displacement_spec(
            fixed_constraints,
            node_sets,
            spec,
            dimensions_xyz=dimensions_xyz,
            spacing=spacing,
            default_value=None,
        )
    for spec in loaded:
        _add_load_spec(loaded_constraints, node_sets, spec)

    fixed_coords = np.asarray(
        [coord for coord in sorted(fixed_constraints)], dtype=np.uint16
    ).reshape((-1, 4))
    fixed_values = np.asarray(
        [fixed_constraints[tuple(coord)] for coord in fixed_coords], dtype=np.float32
    )
    loaded_coords = np.asarray(
        [coord for coord in sorted(loaded_constraints)], dtype=np.uint16
    ).reshape((-1, 4))
    loaded_values = np.asarray(
        [loaded_constraints[tuple(coord)] for coord in loaded_coords], dtype=np.float32
    )
    return BoundaryConditionSet(
        fixed_coordinates=fixed_coords,
        fixed_values=fixed_values,
        loaded_coordinates=loaded_coords,
        loaded_values=loaded_values,
        node_sets=node_sets,
    )


def _all_corner_nodes(mask: np.ndarray) -> set[Node]:
    nodes: set[Node] = set()
    for voxel in np.argwhere(mask):
        for offset in itertools.product((0, 1), repeat=3):
            nodes.add(tuple(int(v) + int(o) for v, o in zip(voxel, offset, strict=True)))
    return nodes


def _surface_nodes(mask: np.ndarray, *, neighbor_mask: np.ndarray) -> set[Node]:
    nodes: set[Node] = set()
    dims = mask.shape
    for voxel_array in np.argwhere(mask):
        voxel = tuple(int(v) for v in voxel_array)
        for axis in range(3):
            for side in (-1, 1):
                neighbor = list(voxel)
                neighbor[axis] += side
                outside = any(neighbor[idx] < 0 or neighbor[idx] >= dims[idx] for idx in range(3))
                exposed = outside or not bool(neighbor_mask[tuple(neighbor)])
                if exposed:
                    node_axis_value = voxel[axis] + (1 if side > 0 else 0)
                    for du, dv in itertools.product((0, 1), repeat=2):
                        node = list(voxel)
                        node[axis] = node_axis_value
                        lateral_axes = [idx for idx in range(3) if idx != axis]
                        node[lateral_axes[0]] += du
                        node[lateral_axes[1]] += dv
                        nodes.add(tuple(node))
    return nodes


def _interface_nodes(mask: np.ndarray, *, material_mask: np.ndarray) -> set[Node]:
    nodes: set[Node] = set()
    dims = mask.shape
    for voxel_array in np.argwhere(mask):
        voxel = tuple(int(v) for v in voxel_array)
        for axis in range(3):
            for side in (-1, 1):
                neighbor = list(voxel)
                neighbor[axis] += side
                outside = any(neighbor[idx] < 0 or neighbor[idx] >= dims[idx] for idx in range(3))
                if outside:
                    continue
                neighbor_tuple = tuple(neighbor)
                touches_material = bool(material_mask[neighbor_tuple])
                touches_other_label = not bool(mask[neighbor_tuple])
                if touches_material and touches_other_label:
                    node_axis_value = voxel[axis] + (1 if side > 0 else 0)
                    for du, dv in itertools.product((0, 1), repeat=2):
                        node = list(voxel)
                        node[axis] = node_axis_value
                        lateral_axes = [idx for idx in range(3) if idx != axis]
                        node[lateral_axes[0]] += du
                        node[lateral_axes[1]] += dv
                        nodes.add(tuple(node))
    return nodes


def _add_displacement_spec(
    constraints: dict[tuple[int, int, int, int], float],
    node_sets: dict[str, list[Node]],
    spec: dict,
    *,
    dimensions_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
    default_value: float | None,
) -> None:
    value = spec.get("value", default_value)
    if value is None:
        raise ValueError("displacement specs require a value")
    for node in _spec_nodes(node_sets, spec):
        for dof in _spec_dofs(spec):
            direction = AXIS_TO_INDEX[dof]
            constraints[(*node, direction)] = _displacement_value(
                value,
                dof=dof,
                dimensions_xyz=dimensions_xyz,
                spacing=spacing,
            )


def _add_load_spec(
    constraints: dict[tuple[int, int, int, int], float],
    node_sets: dict[str, list[Node]],
    spec: dict,
) -> None:
    value = float(spec["value"])
    nodes = _spec_nodes(node_sets, spec)
    distribute = bool(spec.get("distribute", False))
    if distribute and nodes:
        value = value / len(nodes)
    for node in nodes:
        for dof in _spec_dofs(spec):
            constraints[(*node, AXIS_TO_INDEX[dof])] = value


def _spec_nodes(node_sets: dict[str, list[Node]], spec: dict) -> list[Node]:
    name = str(spec["nodeset"])
    if name not in node_sets:
        raise ValueError(f"Unknown nodeset '{name}'")
    return node_sets[name]


def _spec_dofs(spec: dict) -> list[str]:
    value = spec.get("dofs", spec.get("dof"))
    if isinstance(value, str):
        raw = [value]
    else:
        raw = list(value)
    dofs = [str(dof).strip().lower() for dof in raw]
    invalid = [dof for dof in dofs if dof not in AXIS_TO_INDEX]
    if invalid:
        raise ValueError(f"Invalid dof(s): {invalid}")
    return dofs


def _displacement_value(
    value,
    *,
    dof: str,
    dimensions_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> float:
    if isinstance(value, str) and value.strip().endswith("%"):
        fraction = float(value.strip()[:-1]) / 100.0
        axis_index = AXIS_TO_INDEX[dof]
        return fraction * dimensions_xyz[axis_index] * spacing[axis_index]
    return float(value)
