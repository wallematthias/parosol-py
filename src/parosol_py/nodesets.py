from __future__ import annotations

import itertools
import math

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


def nodes_from_mask_face(
    mask,
    *,
    axis: str,
    side: int,
) -> list[Node]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {values.shape}")
    axis_index = AXIS_TO_INDEX[axis.strip().lower()]
    direction = 1 if int(side) > 0 else -1
    nodes: set[Node] = set()
    dims = values.shape
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    for voxel_array in np.argwhere(values):
        voxel = tuple(int(v) for v in voxel_array)
        neighbor = list(voxel)
        neighbor[axis_index] += direction
        outside = (
            neighbor[axis_index] < 0 or neighbor[axis_index] >= dims[axis_index]
        )
        if not outside and bool(values[tuple(neighbor)]):
            continue
        node_axis_value = voxel[axis_index] + (1 if direction > 0 else 0)
        for du, dv in itertools.product((0, 1), repeat=2):
            node = list(voxel)
            node[axis_index] = node_axis_value
            node[lateral_axes[0]] += du
            node[lateral_axes[1]] += dv
            nodes.add(tuple(node))
    return sorted(nodes)


def nodes_from_mask_directional_faces(mask, direction) -> list[Node]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {values.shape}")
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or not np.any(vector):
        raise ValueError("direction must be a finite non-zero 3-vector")

    nodes: set[Node] = set()
    dims = values.shape
    selected_faces = [
        (axis, side)
        for axis in range(3)
        for side in (-1, 1)
        if side * float(vector[axis]) > 0.0
    ]
    for voxel_array in np.argwhere(values):
        voxel = tuple(int(v) for v in voxel_array)
        for axis, side in selected_faces:
            neighbor = list(voxel)
            neighbor[axis] += side
            outside = (
                neighbor[axis] < 0 or neighbor[axis] >= dims[axis]
            )
            if not outside and bool(values[tuple(neighbor)]):
                continue
            node_axis_value = voxel[axis] + (1 if side > 0 else 0)
            lateral_axes = [idx for idx in range(3) if idx != axis]
            for du, dv in itertools.product((0, 1), repeat=2):
                node = list(voxel)
                node[axis] = node_axis_value
                node[lateral_axes[0]] += du
                node[lateral_axes[1]] += dv
                nodes.add(tuple(node))
    return sorted(nodes)


def boundary_conditions_from_nodesets(
    node_sets: dict[str, list[Node]],
    *,
    fixed: list[dict] | tuple[dict, ...] = (),
    prescribed: list[dict] | tuple[dict, ...] = (),
    loaded: list[dict] | tuple[dict, ...] = (),
    dimensions_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
    percent_reference_lengths_mm: dict[str, float] | None = None,
    percent_reference_node_sets: dict[str, list[Node]] | None = None,
) -> BoundaryConditionSet:
    fixed_constraints: dict[tuple[int, int, int, int], float] = {}
    loaded_constraints: dict[tuple[int, int, int, int], float] = {}
    nodeset_percent_reference_lengths = _nodeset_percent_reference_lengths(
        node_sets,
        fixed=fixed,
        prescribed=prescribed,
        spacing=spacing,
        reference_node_sets=percent_reference_node_sets,
    )

    for spec in fixed:
        _add_displacement_spec(
            fixed_constraints,
            node_sets,
            spec,
            dimensions_xyz=dimensions_xyz,
            spacing=spacing,
            default_value=0.0,
            percent_reference_lengths_mm=percent_reference_lengths_mm,
            nodeset_percent_reference_lengths=nodeset_percent_reference_lengths,
        )
    for spec in prescribed:
        _add_prescribed_spec(
            fixed_constraints,
            node_sets,
            spec,
            dimensions_xyz=dimensions_xyz,
            spacing=spacing,
            percent_reference_lengths_mm=percent_reference_lengths_mm,
            nodeset_percent_reference_lengths=nodeset_percent_reference_lengths,
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
        reference_lengths_mm=_axis_reference_lengths(
            nodeset_percent_reference_lengths
        ),
    )


def _all_corner_nodes(mask: np.ndarray) -> set[Node]:
    nodes: set[Node] = set()
    for voxel in np.argwhere(mask):
        for offset in itertools.product((0, 1), repeat=3):
            nodes.add(
                tuple(int(v) + int(o) for v, o in zip(voxel, offset, strict=True))
            )
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
                outside = any(
                    neighbor[idx] < 0 or neighbor[idx] >= dims[idx] for idx in range(3)
                )
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
                outside = any(
                    neighbor[idx] < 0 or neighbor[idx] >= dims[idx] for idx in range(3)
                )
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
    percent_reference_lengths_mm: dict[str, float] | None,
    nodeset_percent_reference_lengths: dict[tuple[str, str], float] | None,
) -> None:
    value = spec.get("value", default_value)
    if value is None:
        raise ValueError("displacement specs require a value")
    nodeset_name = str(spec["nodeset"])
    for node in _spec_nodes(node_sets, spec):
        for dof in _spec_dofs(spec):
            direction = AXIS_TO_INDEX[dof]
            _accumulate_constraint(
                constraints,
                (*node, direction),
                _displacement_value(
                    value,
                    nodeset=nodeset_name,
                    dof=dof,
                    dimensions_xyz=dimensions_xyz,
                    spacing=spacing,
                    percent_reference_lengths_mm=percent_reference_lengths_mm,
                    nodeset_percent_reference_lengths=nodeset_percent_reference_lengths,
                ),
            )


def _add_prescribed_spec(
    constraints: dict[tuple[int, int, int, int], float],
    node_sets: dict[str, list[Node]],
    spec: dict,
    *,
    dimensions_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
    percent_reference_lengths_mm: dict[str, float] | None,
    nodeset_percent_reference_lengths: dict[tuple[str, str], float] | None,
) -> None:
    kind = str(spec.get("kind", "uniform")).strip().lower()
    if kind in {"uniform", "constant"}:
        _add_displacement_spec(
            constraints,
            node_sets,
            spec,
            dimensions_xyz=dimensions_xyz,
            spacing=spacing,
            default_value=None,
            percent_reference_lengths_mm=percent_reference_lengths_mm,
            nodeset_percent_reference_lengths=nodeset_percent_reference_lengths,
        )
        return
    if kind in {"bending", "bend"}:
        _add_bending_spec(constraints, node_sets, spec, spacing=spacing)
        return
    if kind in {"torsion", "twist"}:
        _add_torsion_spec(constraints, node_sets, spec, spacing=spacing)
        return
    raise ValueError(f"Unknown prescribed nodeset kind: {kind!r}")


def _add_bending_spec(
    constraints: dict[tuple[int, int, int, int], float],
    node_sets: dict[str, list[Node]],
    spec: dict,
    *,
    spacing: tuple[float, float, float],
) -> None:
    nodes = _spec_nodes(node_sets, spec)
    if not nodes:
        return
    dof = str(spec.get("dof", spec.get("axis", "z"))).strip().lower()
    dof_index = AXIS_TO_INDEX[dof]
    gradient_axis = str(spec.get("gradient_axis", "x")).strip().lower()
    gradient_index = AXIS_TO_INDEX[gradient_axis]
    positions = _node_positions(nodes, spacing)
    center = _spec_center(spec, positions)
    distances = positions[:, gradient_index] - center[gradient_index]
    half_width = float(np.max(np.abs(distances))) if distances.size else 0.0
    if half_width <= 0.0:
        return
    amplitude = _angle_or_length_amplitude(spec, reference_length=half_width)
    mode = str(spec.get("mode", "linear")).strip().lower()
    neutral_fraction = float(spec.get("neutral_fraction", 0.5))
    for node, distance in zip(nodes, distances, strict=True):
        relative = float(np.clip(distance / half_width, -1.0, 1.0))
        if mode in {"symmetric", "quadratic"}:
            value = amplitude * (relative * relative - neutral_fraction)
        else:
            value = amplitude * relative
        _accumulate_constraint(constraints, (*node, dof_index), float(value))


def _add_torsion_spec(
    constraints: dict[tuple[int, int, int, int], float],
    node_sets: dict[str, list[Node]],
    spec: dict,
    *,
    spacing: tuple[float, float, float],
) -> None:
    nodes = _spec_nodes(node_sets, spec)
    if not nodes:
        return
    axis = str(spec.get("axis", "z")).strip().lower()
    axis_index = AXIS_TO_INDEX[axis]
    lateral = [index for index in range(3) if index != axis_index]
    positions = _node_positions(nodes, spacing)
    center = _spec_center(spec, positions)
    rel = positions - center
    radial = np.sqrt(np.sum(rel[:, lateral] * rel[:, lateral], axis=1))
    radius = float(np.max(radial)) if radial.size else 0.0
    if radius <= 0.0:
        return
    amplitude = _angle_or_length_amplitude(spec, reference_length=radius, torsion=True)
    for node, vector in zip(nodes, rel, strict=True):
        tangent = np.zeros(3, dtype=np.float64)
        tangent[lateral[0]] = -vector[lateral[1]]
        tangent[lateral[1]] = vector[lateral[0]]
        norm = float(np.linalg.norm(tangent))
        if norm <= 0.0:
            continue
        scale = amplitude * min(float(np.linalg.norm(vector[lateral])) / radius, 1.0)
        tangent = tangent / norm * scale
        for dof_index, value in enumerate(tangent):
            if dof_index == axis_index or abs(float(value)) <= 0.0:
                continue
            _accumulate_constraint(constraints, (*node, dof_index), float(value))


def _node_positions(nodes: list[Node], spacing: tuple[float, float, float]) -> np.ndarray:
    coordinates = np.asarray(nodes, dtype=np.float64)
    return coordinates * np.asarray(spacing, dtype=np.float64)


def _spec_center(spec: dict, positions: np.ndarray) -> np.ndarray:
    center = spec.get("center", "centroid")
    if isinstance(center, str):
        if center.strip().lower() in {"centroid", "center", "centre"}:
            return np.mean(positions, axis=0)
        raise ValueError(f"Unknown center value: {center!r}")
    values = np.asarray(center, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("center must be 'centroid' or a 3-value coordinate")
    return values


def _angle_or_length_amplitude(
    spec: dict,
    *,
    reference_length: float,
    torsion: bool = False,
) -> float:
    value, units = _spec_value_and_units(spec)
    if units in {"deg", "degree", "degrees"}:
        radians = math.radians(value)
        if torsion:
            return float(reference_length) * radians
        return float(reference_length) * math.tan(radians / 2.0)
    if units in {"rad", "radian", "radians"}:
        if torsion:
            return float(reference_length) * value
        return float(reference_length) * math.tan(value / 2.0)
    return float(value)


def _spec_value_and_units(spec: dict) -> tuple[float, str]:
    raw = spec["value"]
    units = str(spec.get("units", "")).strip().lower()
    if isinstance(raw, str):
        text = raw.strip().lower()
        for suffix in ("degrees", "degree", "deg", "radians", "radian", "rad", "mm"):
            if text.endswith(suffix):
                return float(text[: -len(suffix)].strip()), suffix
        return float(text), units
    return float(raw), units


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
            _accumulate_constraint(constraints, (*node, AXIS_TO_INDEX[dof]), value)


def _accumulate_constraint(
    constraints: dict[tuple[int, int, int, int], float],
    key: tuple[int, int, int, int],
    value: float,
) -> None:
    constraints[key] = float(constraints.get(key, 0.0)) + float(value)


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


def _nodeset_percent_reference_lengths(
    node_sets: dict[str, list[Node]],
    *,
    fixed: list[dict] | tuple[dict, ...],
    prescribed: list[dict] | tuple[dict, ...],
    spacing: tuple[float, float, float],
    reference_node_sets: dict[str, list[Node]] | None = None,
) -> dict[tuple[str, str], float]:
    fixed_specs = list(fixed)
    if not fixed_specs:
        return {}

    centroid_node_sets = dict(node_sets)
    if reference_node_sets:
        centroid_node_sets.update(
            {name: nodes for name, nodes in reference_node_sets.items() if nodes}
        )
    centroids = {
        name: _nodeset_centroid(nodes, spacing)
        for name, nodes in centroid_node_sets.items()
        if nodes
    }
    references: dict[tuple[str, str], float] = {}
    for spec in prescribed:
        if not _is_percent_value(spec.get("value")):
            continue
        nodeset_name = str(spec["nodeset"])
        if nodeset_name not in centroids:
            continue
        explicit_length = spec.get("reference_length_mm")
        value = _percent_value(spec.get("value"))
        for dof in _spec_dofs(spec):
            if explicit_length is not None:
                length = float(explicit_length)
                if length > 0.0:
                    references[(nodeset_name, dof)] = length
                    continue
            axis_index = AXIS_TO_INDEX[dof]
            explicit_reference = spec.get("reference_nodeset")
            candidates = []
            if explicit_reference is not None:
                candidates = [str(explicit_reference)]
            else:
                candidates = [
                    str(candidate["nodeset"])
                    for candidate in fixed_specs
                    if dof in _spec_dofs(candidate)
                ]
            distances = []
            load_sign = -1.0 if value < 0.0 else 1.0
            for candidate in candidates:
                if candidate not in centroids or candidate == nodeset_name:
                    continue
                delta = float(centroids[candidate][axis_index] - centroids[nodeset_name][axis_index])
                projected = delta * load_sign
                if explicit_reference is not None:
                    distances.append(abs(delta))
                elif projected > 0.0:
                    distances.append(projected)
            if distances:
                references[(nodeset_name, dof)] = max(distances)
    return references


def _axis_reference_lengths(
    references: dict[tuple[str, str], float],
) -> dict[str, float]:
    axis_lengths: dict[str, float] = {}
    for (_nodeset, dof), value in references.items():
        length = float(value)
        if dof in axis_lengths and not np.isclose(axis_lengths[dof], length):
            axis_lengths[dof] = max(axis_lengths[dof], length)
        else:
            axis_lengths[dof] = length
    return axis_lengths


def _nodeset_centroid(
    nodes: list[Node],
    spacing: tuple[float, float, float],
) -> np.ndarray:
    return _node_positions(nodes, spacing).mean(axis=0)


def _is_percent_value(value) -> bool:
    return isinstance(value, str) and value.strip().endswith("%")


def _percent_value(value) -> float:
    if isinstance(value, str) and value.strip().endswith("%"):
        return float(value.strip()[:-1])
    return float(value)


def _displacement_value(
    value,
    *,
    nodeset: str,
    dof: str,
    dimensions_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
    percent_reference_lengths_mm: dict[str, float] | None,
    nodeset_percent_reference_lengths: dict[tuple[str, str], float] | None,
) -> float:
    if _is_percent_value(value):
        fraction = float(value.strip()[:-1]) / 100.0
        if (
            nodeset_percent_reference_lengths is not None
            and (nodeset, dof) in nodeset_percent_reference_lengths
        ):
            return fraction * float(nodeset_percent_reference_lengths[(nodeset, dof)])
        if percent_reference_lengths_mm is not None and dof in percent_reference_lengths_mm:
            return fraction * float(percent_reference_lengths_mm[dof])
        axis_index = AXIS_TO_INDEX[dof]
        return fraction * dimensions_xyz[axis_index] * spacing[axis_index]
    return float(value)
