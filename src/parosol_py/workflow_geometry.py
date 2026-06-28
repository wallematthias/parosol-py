from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .nodesets import nodes_from_labeled_voxels


@dataclass(slots=True)
class WorkflowGeometryResult:
    disk_labels_xyz: np.ndarray
    nodeset_labels_xyz: np.ndarray
    node_sets: dict[str, list[tuple[int, int, int]]]


def read_reference_points(
    path: str | Path,
    *,
    max_points: int | None = None,
    coordinate_system: str = "raw",
) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npz"):
        with np.load(path) as data:
            key = "points" if "points" in data else data.files[0]
            points = np.asarray(data[key], dtype=float)
    elif path.suffix.lower() == ".vtk":
        points = _read_vtk_points(path)
    else:
        points = np.loadtxt(path, dtype=float)
    points = _points_for_coordinate_system(points, coordinate_system, path=path)
    return sample_points(points_array(points, "reference points"), max_points=max_points)


def resolve_reference_space_editor(
    editor: dict[str, Any],
    *,
    reference_points: np.ndarray,
    sample_points: np.ndarray,
    iterations: int = 50,
    tolerance: float = 1.0e-4,
    allow_scale: bool = False,
    snap_planes: bool = False,
    prealign_reference: bool = False,
    transform_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(editor)
    planes = resolved.get("planes", [])
    if not isinstance(planes, list):
        return resolved
    reference = points_array(reference_points, "reference_points")
    sample = points_array(sample_points, "sample_points")
    prealign_metadata = {}
    if prealign_reference:
        reference, prealign_metadata = prealign_reference_points_to_sample(
            reference, sample
        )
    if transform_override is None:
        transform = estimate_reference_to_sample_transform(
            reference,
            sample,
            iterations=iterations,
            tolerance=tolerance,
            allow_scale=allow_scale,
        )
    else:
        transform = {
            "rotation": np.asarray(transform_override["rotation"], dtype=float),
            "translation": np.asarray(transform_override["translation"], dtype=float),
            "scale": float(transform_override.get("scale", 1.0)),
            "iterations": int(transform_override.get("iterations", 0)),
            "mean_distance": float(transform_override.get("mean_distance", 0.0)),
        }
    for index, plane in enumerate(planes):
        if not isinstance(plane, dict):
            continue
        plane_spec = plane
        if plane_spec.get("derive_from") and not _has_plane_pose(plane_spec):
            plane_spec = derive_reference_plane(plane_spec, reference)
        if _is_reference_space_plane(plane_spec):
            plane_spec = transform_plane_spec(plane_spec, transform)
        if snap_planes or bool(plane_spec.get("snap_to_sample_surface", False)):
            plane_spec = snap_plane_to_sample_surface(plane_spec, sample)
        planes[index] = plane_spec
    resolved["planes"] = planes
    resolved["registration"] = {
        "reference_space_replayed": True,
        "iterations": int(transform["iterations"]),
        "mean_distance": float(transform["mean_distance"]),
        "scale": float(transform.get("scale", 1.0)),
        "rotation": np.asarray(transform["rotation"], dtype=float).tolist(),
        "translation": np.asarray(transform["translation"], dtype=float).tolist(),
        **prealign_metadata,
    }
    return resolved


def generate_disk_and_nodeset_geometry(
    editor: dict[str, Any],
    *,
    mask_xyz: np.ndarray,
    material_xyz: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    nodeset_specs: dict[str, dict[str, Any]] | None = None,
    nodeset_labels: dict[str, int] | None = None,
    nodeset_names: dict[str, str] | None = None,
    disk_labels: dict[str, int] | None = None,
) -> WorkflowGeometryResult:
    planes = editor.get("planes", []) if isinstance(editor, dict) else []
    disk_label_array = np.zeros(mask_xyz.shape, dtype=np.uint16)
    nodeset_label_array = np.zeros(mask_xyz.shape, dtype=np.uint16)
    node_sets: dict[str, list[tuple[int, int, int]]] = {}
    nodeset_specs = {} if nodeset_specs is None else nodeset_specs
    nodeset_labels = {} if nodeset_labels is None else nodeset_labels
    nodeset_names = {} if nodeset_names is None else nodeset_names
    disk_labels = {} if disk_labels is None else disk_labels
    active = np.asarray(mask_xyz, dtype=bool)
    material = np.asarray(material_xyz)

    for plane in planes:
        if not isinstance(plane, dict):
            continue
        plane_name = str(plane.get("name", "plane")).strip()
        nodeset_name = nodeset_names.get(plane_name, _slug(plane_name))
        spec = nodeset_specs.get(nodeset_name, {})
        selection = str(spec.get("selection", "surface_nodes")).strip().lower()
        nodeset_label = int(
            spec.get("label", nodeset_labels.get(nodeset_name, 0) or 0)
        )
        disk_label = int(disk_labels.get(plane_name, disk_labels.get(nodeset_name, 0) or 0))
        contact = str(plane.get("contact", "Material disks")).strip().lower()
        surface_mode = _projection_mode(plane.get("surface_mode", "project_bounded"))

        if contact in {"material disks", "pmma caps", "connective disk"} and disk_label > 0:
            disk_mask = _generate_projected_disk_mask(
                active,
                spacing=spacing,
                origin=origin,
                plane_spec=plane,
                material_xyz=material,
                cap_mode="connective_disk"
                if contact == "connective disk"
                else "projected_cap",
                surface_mode=surface_mode,
            )
            disk_label_array[disk_mask] = np.uint16(disk_label)
            if nodeset_label > 0 and contact != "connective disk":
                face_mask = _outer_disk_face_mask(
                    disk_mask,
                    spacing=spacing,
                    origin=origin,
                    plane_spec=plane,
                )
                nodeset_label_array[face_mask] = np.uint16(nodeset_label)
                node_sets[nodeset_name] = _node_set_from_mask(
                    face_mask, selection=selection, material_xyz=material
                )
        elif contact == "bone surface" and nodeset_label > 0:
            if surface_mode == "intersect":
                surface_mask = _intersect_surface_mask(
                    active,
                    spacing=spacing,
                    origin=origin,
                    plane_spec=plane,
                )
            else:
                surface_mask = _projected_surface_mask(
                    active,
                    spacing=spacing,
                    origin=origin,
                    plane_spec=plane,
                )
            nodeset_label_array[surface_mask] = np.uint16(nodeset_label)
            node_sets[nodeset_name] = _node_set_from_mask(
                surface_mask, selection=selection, material_xyz=material
            )
    return WorkflowGeometryResult(
        disk_labels_xyz=disk_label_array,
        nodeset_labels_xyz=nodeset_label_array,
        node_sets=node_sets,
    )


def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    return float(scale) * (
        np.asarray(points, dtype=float) @ np.asarray(rotation, dtype=float).T
    ) + np.asarray(translation, dtype=float)


def estimate_reference_to_sample_transform(
    reference_points: np.ndarray,
    sample_points: np.ndarray,
    *,
    iterations: int = 50,
    tolerance: float = 1.0e-4,
    allow_scale: bool = False,
) -> dict[str, Any]:
    moving = points_array(reference_points, "reference_points")
    fixed = points_array(sample_points, "sample_points")
    rotation, translation, scale = _best_initial_transform(
        moving, fixed, allow_scale=allow_scale
    )
    previous_error = np.inf
    used_iterations = 0
    for used_iterations in range(1, max(1, int(iterations)) + 1):
        transformed = transform_points(moving, rotation, translation, scale=scale)
        matched = fixed[_nearest_indices(transformed, fixed)]
        step_rotation, step_translation, step_scale = _kabsch_similarity(
            transformed, matched, allow_scale=allow_scale
        )
        rotation = step_rotation @ rotation
        scale *= step_scale
        translation = step_scale * (step_rotation @ translation) + step_translation
        error = float(
            np.mean(
                np.linalg.norm(
                    transform_points(moving, rotation, translation, scale=scale)
                    - matched,
                    axis=1,
                )
            )
        )
        if abs(previous_error - error) <= float(tolerance):
            previous_error = error
            break
        previous_error = error
    return {
        "rotation": rotation,
        "translation": translation,
        "scale": float(scale),
        "iterations": used_iterations,
        "mean_distance": previous_error,
    }


def prealign_reference_points_to_sample(
    reference_points: np.ndarray,
    sample_points: np.ndarray,
    *,
    min_scale: tuple[float, float, float] = (0.75, 0.75, 0.65),
    max_scale: tuple[float, float, float] = (1.35, 1.35, 1.45),
) -> tuple[np.ndarray, dict[str, Any]]:
    reference = points_array(reference_points, "reference_points")
    sample = points_array(sample_points, "sample_points")
    ref_axes, ref_lengths, ref_center = _pca_axes_and_lengths(reference)
    sample_axes, sample_lengths, sample_center = _pca_axes_and_lengths(sample)
    scale = sample_lengths / np.maximum(ref_lengths, 1.0e-6)
    scale = np.clip(
        scale, np.asarray(min_scale, dtype=float), np.asarray(max_scale, dtype=float)
    )
    reference_coordinates = (reference - ref_center) @ ref_axes
    aligned = sample_center + (reference_coordinates * scale) @ sample_axes.T
    return aligned, {
        "prealigned_reference_to_sample": True,
        "reference_center": ref_center.tolist(),
        "sample_center": sample_center.tolist(),
        "reference_lengths": ref_lengths.tolist(),
        "sample_lengths": sample_lengths.tolist(),
        "prealign_scale": scale.tolist(),
    }


def transform_plane_spec(plane_spec: dict[str, Any], transform: dict[str, Any]) -> dict[str, Any]:
    rotation = np.asarray(transform["rotation"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    scale = float(transform.get("scale", 1.0))
    resolved = copy.deepcopy(plane_spec)
    center = _vector_or_none(resolved.get("center_ras"))
    normal = _vector_or_none(resolved.get("normal_ras"))
    u_axis = _vector_or_none(resolved.get("u_axis_ras"))
    v_axis = _vector_or_none(resolved.get("v_axis_ras"))
    if center is not None:
        resolved["center_ras"] = transform_points(
            center.reshape(1, 3), rotation, translation, scale=scale
        )[0].tolist()
    if normal is not None:
        resolved["normal_ras"] = _safe_unit(rotation @ normal).tolist()
    if u_axis is not None:
        resolved["u_axis_ras"] = _safe_unit(rotation @ u_axis).tolist()
    if v_axis is not None:
        resolved["v_axis_ras"] = _safe_unit(rotation @ v_axis).tolist()
    if "size_mm" in resolved and isinstance(resolved["size_mm"], (list, tuple)):
        resolved["size_mm"] = [float(v) * scale for v in resolved["size_mm"]]
    resolved["reference_space"] = False
    resolved["resolved_from_reference_space"] = True
    return resolved


def snap_plane_to_sample_surface(
    plane_spec: dict[str, Any], sample_points: np.ndarray
) -> dict[str, Any]:
    center = _vector_or_none(plane_spec.get("center_ras"))
    normal = _vector_or_none(plane_spec.get("normal_ras"))
    if center is None or normal is None:
        return plane_spec
    points = points_array(sample_points, "sample_points")
    normal = _safe_unit(normal)
    rel = points - center
    distance = rel @ normal
    max_distance = float(plane_spec.get("snap_max_distance_mm", 50.0))
    candidates = np.abs(distance) <= max_distance
    if not np.any(candidates):
        return plane_spec
    resolved = copy.deepcopy(plane_spec)
    local_distance = distance[candidates]
    clearance = max(
        0.0,
        float(
            plane_spec.get(
                "outside_clearance_mm", plane_spec.get("snap_clearance_mm", 1.0)
            )
        ),
    )
    snapped_distance = float(np.min(local_distance) - clearance)
    resolved["center_ras"] = (center + snapped_distance * normal).tolist()
    resolved["snapped_outside_sample"] = True
    resolved["snap_side"] = "opposite_normal"
    resolved["outside_clearance_mm"] = clearance
    resolved["snap_candidate_distance_range_mm"] = [
        float(np.min(local_distance)),
        float(np.max(local_distance)),
    ]
    resolved["snap_distance_mm"] = snapped_distance
    return resolved


def derive_reference_plane(
    plane_spec: dict[str, Any], reference_points: np.ndarray
) -> dict[str, Any]:
    points = points_array(reference_points, "reference_points")
    token = str(plane_spec.get("derive_from", "")).strip().lower()
    if not token:
        return copy.deepcopy(plane_spec)
    axis = _derive_axis(plane_spec, token)
    side = _derive_side(plane_spec, token)
    cap_fraction = float(np.clip(float(plane_spec.get("derive_cap_fraction", 0.12)), 0.02, 0.40))
    projection = points @ axis
    threshold = float(
        np.quantile(projection, 1.0 - cap_fraction if side > 0 else cap_fraction)
    )
    cap = points[projection >= threshold] if side > 0 else points[projection <= threshold]
    if cap.shape[0] < 3:
        cap = points
    center, normal = _fit_plane(cap, preferred=axis * side)
    u_axis, v_axis = _plane_axes(normal)
    rel = cap - center
    u = rel @ u_axis
    v = rel @ v_axis
    size_scale = float(plane_spec.get("derive_size_scale", 1.1))
    size_u = max(
        1.0,
        (float(np.percentile(u, 95)) - float(np.percentile(u, 5))) * size_scale,
    )
    size_v = max(
        1.0,
        (float(np.percentile(v, 95)) - float(np.percentile(v, 5))) * size_scale,
    )
    resolved = copy.deepcopy(plane_spec)
    resolved.setdefault("reference_space", True)
    resolved["center_ras"] = center.tolist()
    resolved["normal_ras"] = normal.tolist()
    resolved["u_axis_ras"] = u_axis.tolist()
    resolved["v_axis_ras"] = v_axis.tolist()
    resolved["size_mm"] = [float(size_u), float(size_v)]
    resolved["derived_from_reference"] = token
    return resolved


def sample_points(points: np.ndarray, *, max_points: int | None) -> np.ndarray:
    array = points_array(points, "points")
    if max_points is None or max_points <= 0 or array.shape[0] <= int(max_points):
        return array
    indices = np.linspace(0, array.shape[0] - 1, int(max_points), dtype=int)
    return array[indices]


def points_array(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    array = array[np.all(np.isfinite(array), axis=1)]
    if array.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three finite points")
    return array


def _generate_projected_disk_mask(
    active_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    plane_spec: dict[str, Any],
    material_xyz: np.ndarray,
    cap_mode: str,
    surface_mode: str,
) -> np.ndarray:
    center, normal, u_axis, v_axis, half_u, half_v = _plane_geometry(plane_spec)
    active_idx = np.argwhere(active_xyz)
    active_points = _indices_to_ras_xyz(active_idx, spacing=spacing, origin=origin)
    rel = active_points - center
    distance = rel @ normal
    u = rel @ u_axis
    v = rel @ v_axis
    tol = max(min(spacing) * 0.75, 1.0e-6)
    inside = _inside_shape_vectorized(
        str(plane_spec.get("shape", "anatomy")).strip().lower(),
        u,
        v,
        half_u,
        half_v,
        tolerance=tol,
    )
    forward = distance >= -tol
    candidate_idx = active_idx[inside & forward]
    candidate_points = active_points[inside & forward]
    candidate_distance = distance[inside & forward]
    candidate_u = u[inside & forward]
    candidate_v = v[inside & forward]
    if candidate_idx.size == 0:
        return np.zeros_like(active_xyz, dtype=bool)
    surface_points, surface_keys, first_distance, distance_by_key = _first_surface_points_by_bucket(
        candidate_idx,
        candidate_points,
        candidate_distance,
        candidate_u,
        candidate_v,
        spacing=spacing,
    )
    if len(surface_keys) == 0:
        return np.zeros_like(active_xyz, dtype=bool)
    thickness = float(plane_spec.get("thickness_mm", 3.0))
    intrusion = _disk_intrusion_depth_mm(plane_spec, default=2.0)
    if str(plane_spec.get("shape", "anatomy")).strip().lower() == "anatomy":
        max_surface_distance = first_distance + max(thickness, 0.0) + max(intrusion, 0.0) + tol
        distance_by_key = {
            key: value
            for key, value in distance_by_key.items()
            if value <= max_surface_distance
        }
        surface_keys = set(distance_by_key)
        if not surface_keys:
            return np.zeros_like(active_xyz, dtype=bool)
    cap_inner_distance = first_distance + intrusion
    cap_outer_distance = cap_inner_distance - thickness
    full_idx = np.argwhere(np.ones(active_xyz.shape, dtype=bool))
    full_points = _indices_to_ras_xyz(full_idx, spacing=spacing, origin=origin)
    rel_all = full_points - center
    d_all = rel_all @ normal
    u_all = rel_all @ u_axis
    v_all = rel_all @ v_axis
    inside_all = _inside_shape_vectorized(
        str(plane_spec.get("shape", "anatomy")).strip().lower(),
        u_all,
        v_all,
        half_u,
        half_v,
        tolerance=tol,
    )
    d_ok = (d_all >= cap_outer_distance - tol) & (d_all <= cap_inner_distance + tol)
    empty = ~np.asarray(material_xyz > 0)
    empty_mask = empty[tuple(full_idx.T)]
    if str(plane_spec.get("shape", "anatomy")).strip().lower() == "anatomy":
        keys = [_bucket_key(u_val, v_val, spacing=spacing) for u_val, v_val in zip(u_all, v_all, strict=True)]
        local_surface = np.asarray(
            [distance_by_key.get(key, np.nan) for key in keys],
            dtype=float,
        )
        bucket_mask = np.isfinite(local_surface)
        flat_outer = cap_outer_distance
        if flat_outer >= first_distance:
            flat_outer = first_distance - thickness
        local_inner = local_surface
        local_min = np.minimum(flat_outer, local_inner)
        local_max = np.maximum(flat_outer, local_inner)
        d_ok = (d_all >= local_min - tol) & (d_all <= local_max + tol)
    else:
        bucket_mask = np.ones(full_idx.shape[0], dtype=bool)
    final = inside_all & d_ok & empty_mask & bucket_mask
    out = np.zeros(active_xyz.shape, dtype=bool)
    if np.any(final):
        out[tuple(full_idx[final].T)] = True
    return out


def _outer_disk_face_mask(
    disk_mask_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    plane_spec: dict[str, Any],
) -> np.ndarray:
    if not np.any(disk_mask_xyz):
        return np.zeros_like(disk_mask_xyz, dtype=bool)
    center, normal, _u_axis, _v_axis, _half_u, _half_v = _plane_geometry(plane_spec)
    idx = np.argwhere(disk_mask_xyz)
    points = _indices_to_ras_xyz(idx, spacing=spacing, origin=origin)
    rel = points - center
    distance = rel @ normal
    u = rel @ _u_axis
    v = rel @ _v_axis
    tol = max(min(spacing) * 0.75, 1.0e-6)
    outer_by_key: dict[tuple[int, int], float] = {}
    for dist, uu, vv in zip(distance, u, v, strict=True):
        key = _bucket_key(float(uu), float(vv), spacing=spacing)
        current = outer_by_key.get(key)
        if current is None or float(dist) < current:
            outer_by_key[key] = float(dist)
    local_outer = np.asarray(
        [
            outer_by_key[_bucket_key(float(uu), float(vv), spacing=spacing)]
            for uu, vv in zip(u, v, strict=True)
        ],
        dtype=float,
    )
    face = np.abs(distance - local_outer) <= tol
    out = np.zeros_like(disk_mask_xyz, dtype=bool)
    out[tuple(idx[face].T)] = True
    return out


def _intersect_surface_mask(
    active_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    plane_spec: dict[str, Any],
) -> np.ndarray:
    center, normal, u_axis, v_axis, half_u, half_v = _plane_geometry(plane_spec)
    idx = np.argwhere(active_xyz)
    points = _indices_to_ras_xyz(idx, spacing=spacing, origin=origin)
    rel = points - center
    distance = rel @ normal
    u = rel @ u_axis
    v = rel @ v_axis
    tol = max(min(spacing) * 0.75, 1.0e-6)
    inside = _inside_shape_vectorized(
        str(plane_spec.get("shape", "anatomy")).strip().lower(),
        u,
        v,
        half_u,
        half_v,
        tolerance=tol,
    )
    near = np.abs(distance) <= tol
    out = np.zeros_like(active_xyz, dtype=bool)
    chosen = idx[inside & near]
    if chosen.size:
        out[tuple(chosen.T)] = True
    return out


def _projected_surface_mask(
    active_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    plane_spec: dict[str, Any],
) -> np.ndarray:
    center, normal, u_axis, v_axis, half_u, half_v = _plane_geometry(plane_spec)
    idx = np.argwhere(active_xyz)
    points = _indices_to_ras_xyz(idx, spacing=spacing, origin=origin)
    rel = points - center
    distance = rel @ normal
    u = rel @ u_axis
    v = rel @ v_axis
    tol = max(min(spacing) * 0.75, 1.0e-6)
    inside = _inside_shape_vectorized(
        str(plane_spec.get("shape", "anatomy")).strip().lower(),
        u,
        v,
        half_u,
        half_v,
        tolerance=tol,
    )
    forward = distance >= -tol
    candidate_idx = idx[inside & forward]
    candidate_points = points[inside & forward]
    candidate_distance = distance[inside & forward]
    candidate_u = u[inside & forward]
    candidate_v = v[inside & forward]
    out = np.zeros_like(active_xyz, dtype=bool)
    if candidate_idx.size == 0:
        return out
    surface_points, _surface_keys, _first_distance, _distance_by_key = _first_surface_points_by_bucket(
        candidate_idx,
        candidate_points,
        candidate_distance,
        candidate_u,
        candidate_v,
        spacing=spacing,
    )
    if surface_points.size:
        out[tuple(surface_points.T)] = True
    return out


def _first_surface_points_by_bucket(
    idx: np.ndarray,
    points: np.ndarray,
    distance: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, set[tuple[int, int]], float, dict[tuple[int, int], float]]:
    best: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for voxel, dist, uu, vv in zip(idx, distance, u, v, strict=True):
        if dist < 0:
            continue
        key = _bucket_key(float(uu), float(vv), spacing=spacing)
        current = best.get(key)
        if current is None or float(dist) < current[0]:
            best[key] = (float(dist), np.asarray(voxel, dtype=np.int64))
    if not best:
        return np.zeros((0, 3), dtype=np.int64), set(), 0.0
    points_idx = np.stack([value[1] for value in best.values()], axis=0)
    first_distance = min(value[0] for value in best.values())
    distances = {key: float(value[0]) for key, value in best.items()}
    return points_idx, set(best.keys()), float(first_distance), distances


def _bucket_key(u: float, v: float, *, spacing: tuple[float, float, float]) -> tuple[int, int]:
    resolution = max(min(float(x) for x in spacing), 1.0e-6)
    return (int(round(u / resolution)), int(round(v / resolution)))


def _node_set_from_mask(
    mask_xyz: np.ndarray, *, selection: str, material_xyz: np.ndarray
) -> list[tuple[int, int, int]]:
    labels = np.zeros(mask_xyz.shape, dtype=np.uint16)
    labels[np.asarray(mask_xyz, dtype=bool)] = 1
    return nodes_from_labeled_voxels(
        labels, label=1, selection=selection, material=material_xyz
    )


def _plane_geometry(
    plane_spec: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    center = _vector_or_none(plane_spec.get("center_ras"))
    normal = _safe_unit(_vector_or_none(plane_spec.get("normal_ras")))
    u_axis = _vector_or_none(plane_spec.get("u_axis_ras"))
    v_axis = _vector_or_none(plane_spec.get("v_axis_ras"))
    if center is None or normal is None or not np.any(normal):
        raise ValueError("plane spec must define center_ras and normal_ras")
    if u_axis is None or v_axis is None:
        u_axis, v_axis = _plane_axes(normal)
    else:
        u_axis = _safe_unit(u_axis)
        v_axis = _safe_unit(v_axis)
    size = plane_spec.get("size_mm", [24.0, 24.0])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        size = [24.0, 24.0]
    half_u = max(float(size[0]) / 2.0, 0.5)
    half_v = max(float(size[1]) / 2.0, 0.5)
    shape = str(plane_spec.get("shape", "anatomy")).strip().lower()
    if shape in {"round", "circle", "circular", "oval"}:
        half_u = max(float(size[0]) / 2.0, 0.5)
        half_v = max(float(size[1]) / 2.0, 0.5)
    elif shape == "square":
        half = min(half_u, half_v)
        half_u = half_v = half
    return (
        np.asarray(center, dtype=float),
        np.asarray(normal, dtype=float),
        np.asarray(u_axis, dtype=float),
        np.asarray(v_axis, dtype=float),
        half_u,
        half_v,
    )


def _inside_shape_vectorized(
    shape: str,
    u: np.ndarray,
    v: np.ndarray,
    half_u: float,
    half_v: float,
    *,
    tolerance: float,
) -> np.ndarray:
    if shape in {"anatomy", "rectangle", "rectangular"}:
        return (np.abs(u) <= half_u + tolerance) & (np.abs(v) <= half_v + tolerance)
    if shape == "square":
        half = min(half_u, half_v)
        return (np.abs(u) <= half + tolerance) & (np.abs(v) <= half + tolerance)
    if shape == "hex":
        half = max(min(half_u, half_v), 1.0e-9)
        uu = u / half
        vv = v / half
        hex_tol = tolerance / half
        return (
            (np.abs(uu) <= 1.0 + hex_tol)
            & (np.abs(0.5 * uu + 0.8660254 * vv) <= 1.0 + hex_tol)
            & (np.abs(0.5 * uu - 0.8660254 * vv) <= 1.0 + hex_tol)
        )
    if shape in {"oval", "round", "circle", "circular"}:
        half_u = max(half_u + tolerance, 1.0e-9)
        half_v = max(half_v + tolerance, 1.0e-9)
        return ((u / half_u) ** 2 + (v / half_v) ** 2) <= 1.0
    return (np.abs(u) <= half_u + tolerance) & (np.abs(v) <= half_v + tolerance)


def _indices_to_ras_xyz(
    idx_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> np.ndarray:
    idx = np.asarray(idx_xyz, dtype=float)
    return np.asarray(origin, dtype=float) + idx * np.asarray(spacing, dtype=float)


def _points_for_coordinate_system(
    points: np.ndarray, coordinate_system: str, *, path: Path
) -> np.ndarray:
    token = str(coordinate_system or "raw").strip().lower()
    if token == "auto":
        token = "lps" if path.suffix.lower() == ".vtk" else "ras"
    if token in {"raw", "ras", "slicer_ras"}:
        return np.asarray(points, dtype=float)
    if token in {"lps", "slicer_lps", "lps_to_ras"}:
        out = np.asarray(points, dtype=float).copy()
        out[:, 0] *= -1.0
        out[:, 1] *= -1.0
        return out
    raise ValueError("coordinate_system must be raw, ras, lps, lps_to_ras, or auto")


def _read_vtk_points(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    header = b"".join(raw.splitlines(keepends=True)[:6]).decode(
        "ascii", errors="ignore"
    )
    if "BINARY" in header.upper():
        return _read_binary_vtk_points(raw, path)
    tokens = raw.decode("utf-8", errors="ignore").split()
    try:
        idx = tokens.index("POINTS")
    except ValueError as exc:
        raise ValueError(f"{path} does not contain a VTK POINTS block") from exc
    count = int(tokens[idx + 1])
    start = idx + 3
    values = np.asarray(tokens[start : start + count * 3], dtype=float)
    if values.size != count * 3:
        raise ValueError(f"{path} POINTS block is incomplete")
    return values.reshape((count, 3))


def _read_binary_vtk_points(raw: bytes, path: Path) -> np.ndarray:
    offset = 0
    for line in raw.splitlines(keepends=True):
        decoded = line.decode("ascii", errors="ignore").strip().split()
        offset += len(line)
        if len(decoded) >= 3 and decoded[0].upper() == "POINTS":
            count = int(decoded[1])
            dtype = ">f4" if decoded[2].lower() == "float" else ">f8"
            values = np.frombuffer(raw, dtype=dtype, count=count * 3, offset=offset)
            if values.size != count * 3:
                raise ValueError(f"{path} binary POINTS block is incomplete")
            return values.astype(float).reshape((count, 3))
    raise ValueError(f"{path} does not contain a VTK POINTS block")


def _has_plane_pose(plane_spec: dict[str, Any]) -> bool:
    return _vector_or_none(plane_spec.get("center_ras")) is not None and _vector_or_none(
        plane_spec.get("normal_ras")
    ) is not None


def _is_reference_space_plane(plane_spec: dict[str, Any]) -> bool:
    return bool(plane_spec.get("reference_space", False)) and _has_plane_pose(
        plane_spec
    )


def _derive_axis(plane_spec: dict[str, Any], token: str) -> np.ndarray:
    vector = _vector_or_none(plane_spec.get("reference_axis_ras"))
    if vector is not None:
        return _safe_unit(vector)
    if "femur" in token or "sideways" in token or "impact" in token or "support" in token:
        return _axis_vector(str(plane_spec.get("axis", "y")))
    return _axis_vector(str(plane_spec.get("axis", "z")))


def _derive_side(plane_spec: dict[str, Any], token: str) -> int:
    if any(word in token for word in ("inferior", "lower", "bottom", "impact", "distal")):
        return -1
    if "support" in token and str(plane_spec.get("normal", "+")) == "+":
        return 1
    return -1 if str(plane_spec.get("normal", "")).strip() == "-" else 1


def _axis_vector(axis: str) -> np.ndarray:
    token = axis.lower().strip()
    if token == "x":
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    if token == "y":
        return np.asarray([0.0, 1.0, 0.0], dtype=float)
    return np.asarray([0.0, 0.0, 1.0], dtype=float)


def _vector_or_none(value: Any) -> np.ndarray | None:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        arr = np.asarray(value, dtype=float)
        if np.all(np.isfinite(arr)):
            return arr
    return None


def _safe_unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        return np.zeros_like(arr, dtype=float)
    return arr / norm


def _fit_plane(points: np.ndarray, *, preferred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(points, dtype=float).mean(axis=0)
    centered = np.asarray(points, dtype=float) - center
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _safe_unit(vh[-1])
    preferred = _safe_unit(preferred)
    if float(np.dot(normal, preferred)) < 0.0:
        normal = -normal
    return center, normal


def _plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _safe_unit(normal)
    helper = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(normal, helper))) > 0.85:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=float)
    u_axis = _safe_unit(np.cross(normal, helper))
    v_axis = _safe_unit(np.cross(normal, u_axis))
    return u_axis, v_axis


def _best_initial_transform(
    moving: np.ndarray, fixed: np.ndarray, *, allow_scale: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    candidates = [
        (
            np.eye(3),
            fixed_center - moving_center,
            _initial_scale(moving, fixed, allow_scale=allow_scale),
        ),
        *_initial_pca_transform_candidates(moving, fixed, allow_scale=allow_scale),
    ]
    return min(candidates, key=lambda item: _nearest_mean_distance(moving, fixed, *item))


def _initial_pca_transform_candidates(
    moving: np.ndarray, fixed: np.ndarray, *, allow_scale: bool
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    moving_axes = _principal_axes(moving - moving_center)
    fixed_axes = _principal_axes(fixed - fixed_center)
    candidates = []
    for signs in (
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    ):
        signed_fixed_axes = fixed_axes @ np.diag(signs)
        rotation = signed_fixed_axes @ moving_axes.T
        if np.linalg.det(rotation) <= 0.0:
            continue
        scale = _initial_scale(moving, fixed, allow_scale=allow_scale)
        translation = fixed_center - scale * (rotation @ moving_center)
        candidates.append((rotation, translation, scale))
    return candidates


def _initial_scale(moving: np.ndarray, fixed: np.ndarray, *, allow_scale: bool) -> float:
    if not allow_scale:
        return 1.0
    moving_radius = float(
        np.sqrt(np.mean(np.sum((moving - moving.mean(axis=0)) ** 2, axis=1)))
    )
    fixed_radius = float(
        np.sqrt(np.mean(np.sum((fixed - fixed.mean(axis=0)) ** 2, axis=1)))
    )
    if moving_radius <= 1.0e-12:
        return 1.0
    return max(0.25, min(4.0, fixed_radius / moving_radius))


def _principal_axes(points: np.ndarray) -> np.ndarray:
    _u, _s, vh = np.linalg.svd(np.asarray(points, dtype=float), full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return axes


def _nearest_indices(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    distances = np.sum((moving[:, None, :] - fixed[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def _kabsch_similarity(
    moving: np.ndarray, fixed: np.ndarray, *, allow_scale: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    x = moving - moving_center
    y = fixed - fixed_center
    covariance = x.T @ y
    u, s, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    if allow_scale:
        denom = float(np.sum(x * x))
        scale = float(np.sum(s) / denom) if denom > 1.0e-12 else 1.0
    else:
        scale = 1.0
    translation = fixed_center - scale * (rotation @ moving_center)
    return rotation, translation, scale


def _nearest_mean_distance(
    moving: np.ndarray,
    fixed: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
) -> float:
    transformed = transform_points(moving, rotation, translation, scale=scale)
    matched = fixed[_nearest_indices(transformed, fixed)]
    return float(np.mean(np.linalg.norm(transformed - matched, axis=1)))


def _pca_axes_and_lengths(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(points, dtype=float)
    center = array.mean(axis=0)
    centered = array - center
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    coordinates = centered @ axes
    lengths = np.percentile(coordinates, 95, axis=0) - np.percentile(
        coordinates, 5, axis=0
    )
    lengths = np.maximum(lengths, 1.0e-6)
    return axes, lengths, center


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _projection_mode(value: Any) -> str:
    mode = str(value or "project").strip().lower().replace("-", "_")
    if mode in {"project", "bounded", "project_bounded", "bounded_project"}:
        return "project_bounded"
    if mode in {"project_global", "global", "legacy_project", "legacy_global"}:
        return "project_global"
    if mode == "intersect":
        return "intersect"
    return "project_bounded"


def _disk_intrusion_depth_mm(plane_spec: dict[str, Any], *, default: float) -> float:
    value = plane_spec.get(
        "intrusion_depth_mm", plane_spec.get("protrusion_depth_mm", default)
    )
    try:
        depth = float(value)
    except (TypeError, ValueError):
        depth = float(default)
    if not np.isfinite(depth) or depth < 0.0:
        return max(float(default), 0.0)
    return depth
