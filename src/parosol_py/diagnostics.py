from __future__ import annotations

from typing import Any

import numpy as np

from .boundary_conditions import AXIS_TO_INDEX
from .field_export import morton_keys

AXIS_NAMES = ("x", "y", "z")


def build_fea_diagnostics(
    *,
    fields: dict[str, Any],
    stiffness_gpa_xyz,
    axis: str,
    strain: float,
    voxel_size_mm: float = 1.0,
    load_case_type: str = "constrained_axial",
    load_direction: str | None = None,
    rotation_degrees: float | None = None,
    load_case_center: tuple[float, float] | None = None,
    critical_strain: float | None = 0.007,
    critical_volume_percent: float | None = 2.0,
    failure_criterion: str = "pistoia",
    boundary_conditions=None,
    evaluation_mask_xyz=None,
    analysis_dimensions_xyz=None,
    linear_failure_deformation: float = 0.002,
    crawford_coefficient: float = 0.0068,
    linear_failure_estimates: bool = False,
) -> dict[str, Any]:
    """Derive compact mechanics summaries from ParOSol solution fields."""
    stiffness = np.asarray(stiffness_gpa_xyz, dtype=np.float64)
    axis_token = axis.strip().lower()
    if axis_token not in AXIS_TO_INDEX:
        raise ValueError("axis must be one of: x, y, z")

    mechanics = _mechanics_from_node_fields(
        fields=fields,
        stiffness_gpa_xyz=stiffness,
        axis=axis_token,
        strain=strain,
        voxel_size_mm=voxel_size_mm,
        load_case_type=load_case_type,
        load_direction=load_direction,
        rotation_degrees=rotation_degrees,
        load_case_center=load_case_center,
        boundary_conditions=boundary_conditions,
        evaluation_mask_xyz=evaluation_mask_xyz,
        analysis_dimensions_xyz=analysis_dimensions_xyz,
    )
    failure = _pistoia_failure(
        fields=fields,
        stiffness_gpa_xyz=stiffness,
        mechanics=mechanics,
        axis=axis_token,
        critical_strain=critical_strain,
        critical_volume_percent=critical_volume_percent,
        failure_criterion=failure_criterion,
        evaluation_mask_xyz=evaluation_mask_xyz,
    )
    if linear_failure_estimates:
        failure.update(
            _linear_failure_estimates(
                mechanics=mechanics,
                axis=axis_token,
                voxel_size_mm=voxel_size_mm,
                linear_failure_deformation=linear_failure_deformation,
                crawford_coefficient=crawford_coefficient,
            )
        )
    return {"mechanics": mechanics, "failure": failure}


def _mechanics_from_node_fields(
    *,
    fields: dict[str, Any],
    stiffness_gpa_xyz: np.ndarray,
    axis: str,
    strain: float,
    voxel_size_mm: float,
    load_case_type: str,
    load_direction: str | None,
    rotation_degrees: float | None,
    load_case_center: tuple[float, float] | None,
    boundary_conditions,
    evaluation_mask_xyz,
    analysis_dimensions_xyz,
) -> dict[str, Any]:
    axis_index = AXIS_TO_INDEX[axis]
    load_type = load_case_type.strip().lower()
    direction = _load_direction(load_type, axis, load_direction)
    direction_index = AXIS_TO_INDEX[direction]
    dimensions = tuple(int(v) for v in stiffness_gpa_xyz.shape)
    analysis_dimensions = (
        dimensions
        if analysis_dimensions_xyz is None
        else tuple(int(v) for v in analysis_dimensions_xyz)
    )
    applied_from_strain = (
        float(strain) * float(analysis_dimensions[axis_index]) * float(voxel_size_mm)
    )
    applied = _prescribed_displacement_from_boundary_conditions(
        boundary_conditions,
        direction_index=direction_index,
    )
    if applied is None:
        applied = applied_from_strain
    applied_strain = float(applied) / (
        float(analysis_dimensions[axis_index]) * float(voxel_size_mm)
    )
    applied_rotation = (
        None if rotation_degrees is None else np.deg2rad(float(rotation_degrees))
    )

    result: dict[str, Any] = {
        "axis": axis,
        "load_case_type": load_type,
        "load_direction": direction,
        "applied_strain": applied_strain,
        "analysis_dimensions": {
            name: int(analysis_dimensions[index])
            for index, name in enumerate(AXIS_NAMES)
        },
        "applied_displacement": _xyz(direction_index, applied),
        "applied_rotation_degrees": rotation_degrees,
        "reaction_force": _xyz(direction_index, None),
        "stiffness": _xyz(direction_index, None),
        "generalized_load": {"name": "force", "value": None, "units": "N"},
        "generalized_stiffness": {"name": "stiffness", "value": None, "units": "N/mm"},
        "top_node_count": 0,
        "bottom_node_count": 0,
        "status": "not_computed",
    }

    forces = _as_vector_field(fields.get("forces"))
    displacements = _as_vector_field(fields.get("displacements"))
    if forces is None:
        result["reason"] = "forces field is missing"
        return result

    node_coords = _active_node_coordinates(stiffness_gpa_xyz)
    if len(node_coords) != forces.shape[0]:
        result["reason"] = (
            f"forces field has {forces.shape[0]} nodes, expected {len(node_coords)}"
        )
        return result

    selected = _reaction_node_indices_from_boundary_conditions(
        boundary_conditions,
        node_coords=node_coords,
        direction_index=direction_index,
    )
    if selected is None:
        top_value = dimensions[axis_index]
        top_indices = [
            index
            for index, coord in enumerate(node_coords)
            if coord[axis_index] == top_value
        ]
        bottom_indices = [
            index for index, coord in enumerate(node_coords) if coord[axis_index] == 0
        ]
        selection_name = "top_face"
    else:
        top_indices, bottom_indices = selected
        selection_name = "boundary_conditions"
    reaction_vector = np.sum(forces[top_indices, :], axis=0)
    reaction = float(reaction_vector[direction_index])
    generalized = _generalized_load(
        forces=forces,
        node_coords=node_coords,
        node_indices=top_indices,
        axis=axis,
        direction=direction,
        load_type=load_type,
        dimensions=dimensions,
        voxel_size_mm=voxel_size_mm,
        center=load_case_center,
    )
    generalized_stiffness = _generalized_stiffness(
        generalized,
        applied=applied,
        applied_rotation=applied_rotation,
        load_type=load_type,
    )
    result.update(
        {
            "reaction_force": {
                name: float(reaction_vector[index])
                for index, name in enumerate(AXIS_NAMES)
            },
            "stiffness": _xyz(
                direction_index,
                None if np.isclose(applied, 0.0) else reaction / applied,
            ),
            "generalized_load": generalized,
            "generalized_stiffness": generalized_stiffness,
            "top_node_count": len(top_indices),
            "bottom_node_count": len(bottom_indices),
            "reaction_node_source": selection_name,
            "status": "computed",
        }
    )
    if displacements is not None and displacements.shape[0] == len(node_coords):
        result["mean_top_displacement"] = _xyz(
            direction_index,
            float(np.mean(displacements[top_indices, direction_index]))
            if top_indices
            else None,
        )
        interface = _interface_stiffness(
            displacements=displacements,
            node_coords=node_coords,
            reaction=reaction,
            axis=axis,
            direction=direction,
            load_type=load_type,
            evaluation_mask_xyz=evaluation_mask_xyz,
            voxel_size_mm=voxel_size_mm,
        )
        if interface is not None:
            result["interface_stiffness"] = interface
    return result


def _reaction_node_indices_from_boundary_conditions(
    boundary_conditions,
    *,
    node_coords: list[tuple[int, int, int]],
    direction_index: int,
) -> tuple[list[int], list[int]] | None:
    if boundary_conditions is None:
        return None
    fixed_coordinates = np.asarray(
        getattr(boundary_conditions, "fixed_coordinates", []), dtype=np.int64
    )
    fixed_values = np.asarray(
        getattr(boundary_conditions, "fixed_values", []), dtype=np.float64
    ).reshape(-1)
    if fixed_coordinates.size == 0 or fixed_coordinates.shape[0] != fixed_values.size:
        return None
    coord_to_index = {coord: index for index, coord in enumerate(node_coords)}
    prescribed_nodes: set[tuple[int, int, int]] = set()
    fixed_nodes: set[tuple[int, int, int]] = set()
    for coord, value in zip(fixed_coordinates, fixed_values, strict=True):
        node = tuple(int(v) for v in coord[:3])
        dof = int(coord[3])
        if dof != direction_index:
            continue
        if np.isclose(value, 0.0) or np.isclose(value, 1e-16):
            fixed_nodes.add(node)
        else:
            prescribed_nodes.add(node)
    if not prescribed_nodes:
        return None
    top_indices = sorted(
        coord_to_index[node] for node in prescribed_nodes if node in coord_to_index
    )
    bottom_indices = sorted(
        coord_to_index[node] for node in fixed_nodes if node in coord_to_index
    )
    if not top_indices:
        return None
    return top_indices, bottom_indices


def _prescribed_displacement_from_boundary_conditions(
    boundary_conditions,
    *,
    direction_index: int,
) -> float | None:
    if boundary_conditions is None:
        return None
    fixed_coordinates = np.asarray(
        getattr(boundary_conditions, "fixed_coordinates", []), dtype=np.int64
    )
    fixed_values = np.asarray(
        getattr(boundary_conditions, "fixed_values", []), dtype=np.float64
    ).reshape(-1)
    if fixed_coordinates.size == 0 or fixed_coordinates.shape[0] != fixed_values.size:
        return None
    prescribed = [
        float(value)
        for coord, value in zip(fixed_coordinates, fixed_values, strict=True)
        if int(coord[3]) == direction_index
        and not np.isclose(value, 0.0)
        and not np.isclose(value, 1e-16)
    ]
    if not prescribed:
        return None
    return float(np.mean(prescribed))


def _pistoia_failure(
    *,
    fields: dict[str, Any],
    stiffness_gpa_xyz: np.ndarray,
    mechanics: dict[str, Any],
    axis: str,
    critical_strain: float | None,
    critical_volume_percent: float | None,
    failure_criterion: str,
    evaluation_mask_xyz,
) -> dict[str, Any]:
    axis_index = AXIS_TO_INDEX[axis]
    criterion = failure_criterion.strip().lower()
    result: dict[str, Any] = {
        "criterion": criterion,
        "critical_strain": None if critical_strain is None else float(critical_strain),
        "critical_volume_percent": (
            None if critical_volume_percent is None else float(critical_volume_percent)
        ),
        "ees_at_critical_volume": None,
        "factor": None,
        "failure_load": _xyz(axis_index, None),
        "status": "not_computed",
    }
    if criterion not in {"pistoia", "none"}:
        result["reason"] = f"unsupported failure criterion: {failure_criterion}"
        return result
    if criterion == "none":
        result["status"] = "disabled"
        return result
    if critical_strain is None or critical_volume_percent is None:
        result["reason"] = "critical_strain and critical_volume_percent are required"
        return result
    if "sed" not in fields:
        result["reason"] = "sed field is missing"
        return result

    active_values = _active_element_values(
        fields["sed"], stiffness_gpa_xyz, evaluation_mask_xyz=evaluation_mask_xyz
    )
    active_modulus_mpa = _active_element_values(
        stiffness_gpa_xyz * 1000.0,
        stiffness_gpa_xyz,
        evaluation_mask_xyz=evaluation_mask_xyz,
    )
    valid = (active_modulus_mpa > 0.0) & np.isfinite(active_values)
    if not np.any(valid):
        result["reason"] = "no active finite SED/modulus values"
        return result

    ees = np.sqrt(
        np.maximum(0.0, 2.0 * active_values[valid] / active_modulus_mpa[valid])
    )
    percentile = max(0.0, min(100.0, 100.0 - float(critical_volume_percent)))
    ees_at_critical_volume = float(np.percentile(ees, percentile))
    factor = (
        None
        if np.isclose(ees_at_critical_volume, 0.0)
        else float(critical_strain) / ees_at_critical_volume
    )
    reaction = mechanics.get("reaction_force", {}).get(axis)
    generalized = mechanics.get("generalized_load", {})
    generalized_value = generalized.get("value")
    failure_load = (
        None if factor is None or reaction is None else float(reaction) * factor
    )
    failure_generalized = (
        None
        if factor is None or generalized_value is None
        else float(generalized_value) * factor
    )

    result.update(
        {
            "ees_at_critical_volume": ees_at_critical_volume,
            "factor": factor,
            "failure_load": _xyz(axis_index, failure_load),
            "failure_generalized_load": {
                "name": generalized.get("name", "load"),
                "value": failure_generalized,
                "units": generalized.get("units"),
            },
            "ees_distribution": _array_statistics(ees),
            "status": "computed" if factor is not None else "not_computed",
        }
    )
    if factor is None:
        result["reason"] = "ees_at_critical_volume is zero"
    return result


def _linear_failure_estimates(
    *,
    mechanics: dict[str, Any],
    axis: str,
    voxel_size_mm: float,
    linear_failure_deformation: float,
    crawford_coefficient: float,
) -> dict[str, Any]:
    stiffness = mechanics.get("generalized_stiffness", {})
    stiffness_source = "generalized_stiffness"
    k_fe = stiffness.get("value")
    units = stiffness.get("units")
    analysis_dimensions = mechanics.get("analysis_dimensions", {})
    height_voxels = analysis_dimensions.get(axis)
    if k_fe is None or height_voxels is None or units != "N/mm":
        reason = "linear strength estimates require translational stiffness in N/mm"
        return {
            "linear_reaction_at_deformation": {"status": "not_computed", "reason": reason},
            "crawford_stiffness_height": {"status": "not_computed", "reason": reason},
        }

    height_mm = float(height_voxels) * float(voxel_size_mm)
    applied = mechanics.get("applied_displacement", {}).get(axis)
    reaction = mechanics.get("reaction_force", {}).get(axis)
    sign_source = applied if applied is not None and not np.isclose(applied, 0.0) else reaction
    sign = -1.0 if sign_source is not None and float(sign_source) < 0.0 else 1.0
    k_abs = abs(float(k_fe))
    threshold = float(linear_failure_deformation)
    coefficient = float(crawford_coefficient)
    threshold_displacement_mm = threshold * height_mm
    linear_load = sign * k_abs * threshold_displacement_mm
    crawford_load = sign * coefficient * k_abs * height_mm
    return {
        "linear_reaction_at_deformation": {
            "status": "computed",
            "method": "linear_reaction_at_deformation",
            "deformation": threshold,
            "height_mm": height_mm,
            "stiffness_n_per_mm": k_abs,
            "stiffness_source": stiffness_source,
            "displacement_mm": threshold_displacement_mm,
            "failure_load": _xyz(AXIS_TO_INDEX[axis], linear_load),
            "failure_generalized_load": {
                "name": "force",
                "value": linear_load,
                "units": "N",
            },
        },
        "crawford_stiffness_height": {
            "status": "computed",
            "method": "crawford_stiffness_height",
            "coefficient": coefficient,
            "equivalent_deformation": coefficient,
            "relative_to_linear_deformation": (
                None if np.isclose(threshold, 0.0) else coefficient / threshold
            ),
            "height_mm": height_mm,
            "stiffness_n_per_mm": k_abs,
            "stiffness_source": stiffness_source,
            "failure_load": _xyz(AXIS_TO_INDEX[axis], crawford_load),
            "failure_generalized_load": {
                "name": "force",
                "value": crawford_load,
                "units": "N",
            },
        },
    }


def _active_element_values(
    values,
    stiffness_gpa_xyz: np.ndarray,
    *,
    evaluation_mask_xyz=None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    active_count = int(np.count_nonzero(stiffness_gpa_xyz > 0.0))
    if evaluation_mask_xyz is None and array.ndim == 1 and array.size == active_count:
        return array.reshape(-1)

    source_active_coords = _active_element_coordinates(stiffness_gpa_xyz)
    active_coords = source_active_coords
    if evaluation_mask_xyz is not None:
        mask = np.asarray(evaluation_mask_xyz, dtype=bool)
        if mask.shape != stiffness_gpa_xyz.shape:
            raise ValueError(
                f"evaluation_mask_xyz shape {mask.shape} does not match stiffness shape {stiffness_gpa_xyz.shape}"
            )
        active_coords = [coord for coord in active_coords if bool(mask[coord])]
    if array.ndim == 1 and array.size == len(active_coords):
        return array.reshape(-1)
    if array.ndim == 1 and array.size == len(source_active_coords):
        dense = np.zeros(stiffness_gpa_xyz.shape, dtype=array.dtype)
        for index, coord in enumerate(source_active_coords):
            dense[coord] = array[index]
        return np.asarray([dense[coord] for coord in active_coords], dtype=np.float64)
    if array.ndim == 1 and array.size == stiffness_gpa_xyz.size:
        dense = _native_scalar_to_dense_xyz(array, stiffness_gpa_xyz, dense=True)
        return np.asarray([dense[coord] for coord in active_coords], dtype=np.float64)
    if array.shape == stiffness_gpa_xyz.shape:
        return np.asarray([array[coord] for coord in active_coords], dtype=np.float64)
    raise ValueError(
        f"field has shape {array.shape}, expected active element count "
        f"{len(active_coords)} or dense shape {stiffness_gpa_xyz.shape}"
    )


def _native_scalar_to_dense_xyz(
    values: np.ndarray,
    stiffness_gpa_xyz: np.ndarray,
    *,
    dense: bool,
) -> np.ndarray:
    coords = (
        _dense_element_coordinates(stiffness_gpa_xyz.shape)
        if dense
        else _active_element_coordinates(stiffness_gpa_xyz)
    )
    out = np.zeros(stiffness_gpa_xyz.shape, dtype=values.dtype)
    for index, coord in enumerate(coords):
        out[coord] = values[index]
    return out


def _as_vector_field(value) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        return None
    return array


def _load_direction(load_type: str, axis: str, load_direction: str | None) -> str:
    if load_direction is not None:
        token = load_direction.strip().lower()
        if token not in AXIS_TO_INDEX:
            raise ValueError("load_direction must be one of: x, y, z")
        return token
    if load_type in {"shear", "simple_shear", "directional_shear"}:
        return next(name for name in AXIS_NAMES if name != axis)
    return axis


def _generalized_load(
    *,
    forces: np.ndarray,
    node_coords: list[tuple[int, int, int]],
    node_indices: list[int],
    axis: str,
    direction: str,
    load_type: str,
    dimensions: tuple[int, int, int],
    voxel_size_mm: float,
    center: tuple[float, float] | None,
) -> dict[str, Any]:
    axis_index = AXIS_TO_INDEX[axis]
    direction_index = AXIS_TO_INDEX[direction]
    if load_type in {"bending", "bend", "torsion", "twist"}:
        moment = _reaction_moment(
            forces=forces,
            node_coords=node_coords,
            node_indices=node_indices,
            dimensions=dimensions,
            voxel_size_mm=voxel_size_mm,
            center=center,
        )
        return {
            "name": "moment",
            "component": axis,
            "value": float(moment[axis_index]),
            "units": "N*mm",
            "vector": {
                name: float(moment[index]) for index, name in enumerate(AXIS_NAMES)
            },
        }
    reaction = float(np.sum(forces[node_indices, direction_index]))
    return {
        "name": "force",
        "component": direction,
        "value": reaction,
        "units": "N",
    }


def _reaction_moment(
    *,
    forces: np.ndarray,
    node_coords: list[tuple[int, int, int]],
    node_indices: list[int],
    dimensions: tuple[int, int, int],
    voxel_size_mm: float,
    center: tuple[float, float] | None,
) -> np.ndarray:
    origin = np.asarray(
        [
            ((float(dimensions[index]) - 1.0) / 2.0) * float(voxel_size_mm)
            for index in range(3)
        ],
        dtype=np.float64,
    )
    if center is not None:
        origin[:2] = np.asarray(center, dtype=np.float64)
    moment = np.zeros(3, dtype=np.float64)
    for index in node_indices:
        position = (np.asarray(node_coords[index], dtype=np.float64) - 0.5) * float(
            voxel_size_mm
        )
        moment += np.cross(position - origin, forces[index])
    return moment


def _generalized_stiffness(
    generalized_load: dict[str, Any],
    *,
    applied: float,
    applied_rotation: float | None,
    load_type: str,
) -> dict[str, Any]:
    value = generalized_load.get("value")
    if value is None:
        return {"name": "stiffness", "value": None, "units": None}
    if load_type in {"bending", "bend", "torsion", "twist"}:
        stiffness = (
            None
            if applied_rotation is None or np.isclose(applied_rotation, 0.0)
            else float(value) / float(applied_rotation)
        )
        return {"name": "rotational_stiffness", "value": stiffness, "units": "N*mm/rad"}
    stiffness = None if np.isclose(applied, 0.0) else float(value) / float(applied)
    return {"name": "stiffness", "value": stiffness, "units": "N/mm"}


def _interface_stiffness(
    *,
    displacements: np.ndarray,
    node_coords: list[tuple[int, int, int]],
    reaction: float,
    axis: str,
    direction: str,
    load_type: str,
    evaluation_mask_xyz,
    voxel_size_mm: float,
) -> dict[str, Any] | None:
    if evaluation_mask_xyz is None:
        return None
    if load_type in {"bending", "bend", "torsion", "twist"}:
        return None
    axis_index = AXIS_TO_INDEX[axis]
    direction_index = AXIS_TO_INDEX[direction]
    mask = np.asarray(evaluation_mask_xyz, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        return None
    active = np.argwhere(mask)
    low = int(active[:, axis_index].min())
    high = int(active[:, axis_index].max())
    low_nodes = _mask_face_nodes(mask, axis_index=axis_index, element_index=low, side=-1)
    high_nodes = _mask_face_nodes(mask, axis_index=axis_index, element_index=high, side=1)
    coord_to_index = {coord: index for index, coord in enumerate(node_coords)}
    low_indices = [coord_to_index[node] for node in low_nodes if node in coord_to_index]
    high_indices = [coord_to_index[node] for node in high_nodes if node in coord_to_index]
    if not low_indices or not high_indices:
        return None
    low_mean = float(np.mean(displacements[low_indices, direction_index]))
    high_mean = float(np.mean(displacements[high_indices, direction_index]))
    displacement = high_mean - low_mean
    if np.isclose(displacement, 0.0):
        stiffness = None
    else:
        stiffness = float(reaction) / float(displacement)
    return {
        "name": "interface_stiffness",
        "value": stiffness,
        "units": "N/mm",
        "reference": "evaluation_mask_surface_displacement",
        "displacement_difference": _xyz(direction_index, displacement),
        "mean_top_displacement": _xyz(direction_index, high_mean),
        "mean_bottom_displacement": _xyz(direction_index, low_mean),
        "top_node_count": len(high_indices),
        "bottom_node_count": len(low_indices),
        "height_mm": float(high - low + 1) * float(voxel_size_mm),
    }


def _mask_face_nodes(
    mask: np.ndarray,
    *,
    axis_index: int,
    element_index: int,
    side: int,
) -> set[tuple[int, int, int]]:
    nodes: set[tuple[int, int, int]] = set()
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    face_elements = np.argwhere(mask & (_axis_index_grid(mask.shape, axis_index) == element_index))
    node_axis_value = int(element_index) + (1 if int(side) > 0 else 0)
    for element in face_elements:
        base = [int(v) for v in element]
        for du in (0, 1):
            for dv in (0, 1):
                node = list(base)
                node[axis_index] = node_axis_value
                node[lateral_axes[0]] += du
                node[lateral_axes[1]] += dv
                nodes.add(tuple(node))
    return nodes


def _axis_index_grid(shape: tuple[int, int, int], axis_index: int) -> np.ndarray:
    values = np.arange(shape[axis_index], dtype=np.int64)
    view_shape = [1, 1, 1]
    view_shape[axis_index] = shape[axis_index]
    return np.broadcast_to(values.reshape(view_shape), shape)


def _active_element_coordinates(
    stiffness_gpa_xyz: np.ndarray,
) -> list[tuple[int, int, int]]:
    return _coords_to_tuples(_morton_sorted(np.argwhere(stiffness_gpa_xyz > 0.0)))


def _dense_element_coordinates(
    shape: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    x_dim, y_dim, z_dim = (int(v) for v in shape)
    coords = np.stack(
        np.meshgrid(
            np.arange(x_dim, dtype=np.int64),
            np.arange(y_dim, dtype=np.int64),
            np.arange(z_dim, dtype=np.int64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    return _coords_to_tuples(_morton_sorted(coords))


def _active_node_coordinates(
    stiffness_gpa_xyz: np.ndarray,
) -> list[tuple[int, int, int]]:
    elements = np.argwhere(stiffness_gpa_xyz > 0.0).astype(np.int64, copy=False)
    if elements.size == 0:
        return []
    offsets = np.asarray(
        [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
        dtype=np.int64,
    )
    nodes = (elements[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    nodes = np.unique(nodes, axis=0)
    return _coords_to_tuples(_morton_sorted(nodes))


def _morton_key(x: int, y: int, z: int) -> int:
    key = 0
    bit_index = 0
    limit = max(x, y, z)
    while (1 << bit_index) <= limit:
        key |= ((x >> bit_index) & 1) << (3 * bit_index)
        key |= ((y >> bit_index) & 1) << (3 * bit_index + 1)
        key |= ((z >> bit_index) & 1) << (3 * bit_index + 2)
        bit_index += 1
    return key


def _morton_sorted(coords: np.ndarray) -> np.ndarray:
    values = np.asarray(coords, dtype=np.int64)
    if values.size == 0:
        return values.reshape((0, 3))
    order = np.argsort(morton_keys(values), kind="stable")
    return values[order]


def _coords_to_tuples(coords: np.ndarray) -> list[tuple[int, int, int]]:
    return [tuple(int(v) for v in coord) for coord in np.asarray(coords)]


def _xyz(axis_index: int, value) -> dict[str, Any]:
    return {
        name: (value if index == axis_index else None)
        for index, name in enumerate(AXIS_NAMES)
    }


def _array_statistics(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": int(values.size),
            "finite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
        }
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
    }
