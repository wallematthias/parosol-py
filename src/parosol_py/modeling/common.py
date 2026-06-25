from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from parosol_py.core import BoundaryConditionSet
from parosol_py.images import ImageGrid, export_scalar_image, to_output_order
from parosol_py.images import largest_connected_component
from parosol_py.materials import density_to_material_map
from parosol_py.nodesets import (
    boundary_conditions_from_nodesets,
    nodes_from_labeled_voxels,
)
from parosol_py.visualization import write_case_overview

from .io import read_image_zyx, resolve_path

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def load_density_and_mask(
    model_config: dict[str, Any],
    *,
    base_dir: Path,
    preprocessing_config: dict[str, Any] | None = None,
) -> tuple[
    np.ndarray, np.ndarray, tuple[float, float, float], tuple[float, float, float]
]:
    density_path = resolve_path(model_config["density_image"], base_dir=base_dir)
    mask_path = resolve_path(model_config["mask_image"], base_dir=base_dir)
    density_zyx, spacing, origin = read_image_zyx(density_path)
    mask_zyx, mask_spacing, _mask_origin = read_image_zyx(mask_path)
    if density_zyx.shape != mask_zyx.shape:
        raise ValueError(
            f"density image shape {density_zyx.shape} does not match mask shape {mask_zyx.shape}"
        )
    if not np.allclose(spacing, mask_spacing):
        raise ValueError("density image and mask spacing differ")
    preprocessing = {} if preprocessing_config is None else preprocessing_config
    if _enabled(preprocessing.get("largest_cc", False)):
        mask_zyx = _largest_connected_label_component(mask_zyx)
    geometry = model_config.get("geometry", {})
    crop_spec = preprocessing.get(
        "crop_to_bb",
        geometry.get("crop_to_mask", model_config.get("crop_to_mask", False)),
    )
    if _enabled(crop_spec):
        margin = (
            crop_spec.get("margin_voxels")
            if isinstance(crop_spec, dict)
            else preprocessing.get(
                "crop_margin_voxels", geometry.get("crop_margin_voxels", 4)
            )
        )
        crop_labels = _crop_labels(model_config, crop_spec)
        density_zyx, mask_zyx, origin = _crop_to_mask_bbox(
            density_zyx,
            mask_zyx,
            spacing=spacing,
            origin=origin,
            margin_voxels=int(margin),
            labels=crop_labels,
        )
    if "spacing" in model_config:
        spacing = _triple(model_config["spacing"], "model.spacing")
    if "spacing" in geometry:
        spacing = _triple(geometry["spacing"], "model.geometry.spacing")
    target_spacing = _target_resample_spacing(geometry, spacing)
    if target_spacing is not None:
        density_zyx = _resample_array_zyx(
            density_zyx,
            spacing=spacing,
            target_spacing=target_spacing,
            interpolation="linear",
        )
        mask_zyx = _resample_array_zyx(
            mask_zyx,
            spacing=spacing,
            target_spacing=target_spacing,
            interpolation="nearest",
        )
        spacing = target_spacing
    smooth_spec = preprocessing.get("smooth", geometry.get("smooth", False))
    if _enabled(smooth_spec):
        density_zyx, mask_zyx = _smooth_density_and_labels(
            density_zyx,
            mask_zyx,
            spacing=spacing,
            smooth_spec=smooth_spec,
        )
    if "origin" in model_config:
        origin = _triple(model_config["origin"], "model.origin")
    return (
        np.asarray(density_zyx, dtype=np.float64),
        np.asarray(mask_zyx),
        spacing,
        origin,
    )


def _largest_connected_label_component(mask_zyx: np.ndarray) -> np.ndarray:
    labels = np.asarray(mask_zyx)
    out = np.zeros(labels.shape, dtype=labels.dtype)
    for label in np.unique(labels):
        if int(label) == 0:
            continue
        component = largest_connected_component(labels == label, background=False)
        out[component > 0] = label
    return out


def _crop_to_mask_bbox(
    density_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    margin_voxels: int,
    labels: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    labels_array = np.asarray(mask_zyx)
    active = (
        np.isin(labels_array, sorted(labels))
        if labels
        else labels_array > 0
    )
    if not np.any(active):
        raise ValueError("model mask has no foreground voxels")
    coords = np.argwhere(active)
    margin = max(0, int(margin_voxels))
    lo_zyx = np.maximum(coords.min(axis=0) - margin, 0)
    hi_zyx = np.minimum(coords.max(axis=0) + margin + 1, active.shape)
    slices = tuple(slice(int(lo_zyx[idx]), int(hi_zyx[idx])) for idx in range(3))
    lo_xyz = lo_zyx[[2, 1, 0]]
    cropped_origin = tuple(
        float(origin[idx]) + float(lo_xyz[idx]) * float(spacing[idx])
        for idx in range(3)
    )
    return density_zyx[slices], mask_zyx[slices], cropped_origin


def _crop_labels(
    model_config: dict[str, Any],
    crop_spec: Any,
) -> set[int] | None:
    if isinstance(crop_spec, dict):
        explicit = crop_spec.get(
            "target_labels",
            crop_spec.get("labels", crop_spec.get("target_label")),
        )
        parsed = _parse_label_values(explicit)
        if parsed:
            return parsed
    parsed = _parse_label_values(model_config.get("labels"))
    return parsed or None


def _parse_label_values(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    labels: set[int] = set()
    for raw in values:
        try:
            labels.add(int(raw))
        except (TypeError, ValueError):
            continue
    return labels


def _enabled(value) -> bool:
    if isinstance(value, dict):
        return _enabled(value.get("enabled", True))
    if isinstance(value, str):
        return value.strip().lower() in {"on", "true", "yes", "1", "largest"}
    return bool(value)


def _target_resample_spacing(
    geometry: dict[str, Any],
    spacing: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    tolerance = float(
        geometry.get(
            "spacing_tolerance_mm",
            geometry.get("resample_tolerance_mm", 1.0e-3),
        )
    )
    rtol = float(geometry.get("spacing_tolerance_relative", 1.0e-5))
    if "resample_spacing" in geometry:
        target = _triple(geometry["resample_spacing"], "model.geometry.resample_spacing")
    elif "voxel_size" in geometry:
        target = _triple(geometry["voxel_size"], "model.geometry.voxel_size")
    else:
        isotropic = geometry.get("isotropic_spacing", "auto")
        if not isotropic:
            return None
        if str(isotropic).strip().lower() == "auto":
            if np.allclose(spacing, spacing[0], rtol=rtol, atol=tolerance):
                return None
            target_value = min(float(v) for v in spacing)
        else:
            target_value = float(isotropic)
        target = (target_value, target_value, target_value)
    if np.allclose(spacing, target, rtol=rtol, atol=tolerance):
        return None
    return target


def _resample_array_zyx(
    array_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    interpolation: str,
) -> np.ndarray:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(spacing)
    original_size = np.asarray(image.GetSize(), dtype=np.int64)
    original_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    new_spacing = np.asarray(target_spacing, dtype=np.float64)
    new_size = np.maximum(
        1, np.round(original_size * original_spacing / new_spacing)
    ).astype(int)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(v) for v in new_spacing))
    resampler.SetSize([int(v) for v in new_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if interpolation == "nearest" else sitk.sitkLinear
    )
    return sitk.GetArrayFromImage(resampler.Execute(image))


def _smooth_density_and_labels(
    density_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    smooth_spec: Any,
) -> tuple[np.ndarray, np.ndarray]:
    spec = smooth_spec if isinstance(smooth_spec, dict) else {}
    sigma_mm = float(spec.get("sigma_mm", spec.get("sigma", 0.5)))
    if sigma_mm <= 0:
        return density_zyx, mask_zyx
    minimum = int(spec.get("minimum_size_voxels", 4))
    if min(int(v) for v in density_zyx.shape) < minimum:
        return density_zyx, mask_zyx
    smooth_density = _enabled(spec.get("density", True))
    smooth_labels = _enabled(spec.get("labels", spec.get("segmentation", True)))
    out_density = (
        _smooth_scalar_image(density_zyx, spacing=spacing, sigma_mm=sigma_mm)
        if smooth_density
        else np.asarray(density_zyx)
    )
    out_mask = (
        _smooth_label_image(
            mask_zyx,
            spacing=spacing,
            sigma_mm=sigma_mm,
            threshold=float(spec.get("label_threshold", spec.get("threshold", 0.5))),
        )
        if smooth_labels
        else np.asarray(mask_zyx)
    )
    return out_density, out_mask.astype(np.asarray(mask_zyx).dtype, copy=False)


def _smooth_scalar_image(
    array_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    sigma_mm: float,
) -> np.ndarray:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(array_zyx, dtype=np.float32))
    image.SetSpacing(tuple(float(v) for v in spacing))
    smoothed = sitk.SmoothingRecursiveGaussian(image, float(sigma_mm))
    return sitk.GetArrayFromImage(smoothed)


def _smooth_label_image(
    labels_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    sigma_mm: float,
    threshold: float,
) -> np.ndarray:
    labels = np.asarray(labels_zyx)
    unique_labels = [int(v) for v in np.unique(labels) if int(v) != 0]
    if not unique_labels:
        return labels.copy()
    scores = []
    for label in unique_labels:
        scores.append(
            _smooth_scalar_image(
                (labels == label).astype(np.float32),
                spacing=spacing,
                sigma_mm=sigma_mm,
            )
        )
    stacked = np.stack(scores, axis=0)
    best_index = np.argmax(stacked, axis=0)
    best_score = np.take_along_axis(
        stacked,
        best_index[np.newaxis, ...],
        axis=0,
    )[0]
    out = np.zeros(labels.shape, dtype=labels.dtype)
    label_values = np.asarray(unique_labels, dtype=labels.dtype)
    active = best_score >= float(threshold)
    out[active] = label_values[best_index[active]]
    return out


def material_from_density(
    density_zyx: np.ndarray,
    active_mask_zyx: np.ndarray,
    *,
    material_config: dict[str, Any],
) -> tuple[np.ndarray, float]:
    density_cfg = dict(material_config.get("density", {}))
    e_cfg = density_cfg.get("E", density_cfg.get("youngs_modulus", density_cfg))
    if not isinstance(e_cfg, dict):
        raise ValueError("materials.density.E must be an object")
    equation = str(e_cfg.get("equation", density_cfg.get("equation", "linear")))
    poisson_spec = density_cfg.get(
        "nu",
        density_cfg.get(
            "poisson_ratio",
            material_config.get("poisson_ratio", material_config.get("nu", 0.3)),
        ),
    )
    mapped = density_to_material_map(
        density_zyx,
        equation=equation,
        poisson_ratio=poisson_spec,
        mask_threshold=float(
            density_cfg.get("active_threshold", density_cfg.get("mask_threshold", 0.0))
        ),
        active_mask=active_mask_zyx,
        minimum_e_mpa=_density_floor_config_value(e_cfg, density_cfg),
        maximum_e_mpa=_optional_float(
            e_cfg.get("maximum_e_mpa", density_cfg.get("maximum_e_mpa"))
        ),
        **{
            key: value
            for key, value in e_cfg.items()
            if key
            not in {
                "equation",
                "active_threshold",
                "mask_threshold",
                "minimum_e_mpa",
                "floor_e_mpa",
                "floor_mpa",
                "floor",
                "maximum_e_mpa",
            }
        },
    )
    return mapped.youngs_modulus_mpa, mapped.poisson_ratio


def pmma_spec(material_config: dict[str, Any]) -> dict[str, float]:
    spec = material_config.get("pmma", {})
    return {
        "E": float(spec.get("E", spec.get("youngs_modulus_mpa", 2500.0))),
        "nu": float(spec.get("nu", spec.get("poisson_ratio", 0.3))),
    }


def pad_along_axis(
    array_xyz: np.ndarray, *, axis: str, before: int, after: int, value=0
):
    axis_index = AXIS_TO_INDEX[axis]
    pad_width = [(0, 0), (0, 0), (0, 0)]
    pad_width[axis_index] = (int(before), int(after))
    return np.pad(array_xyz, pad_width, mode="constant", constant_values=value)


def projected_caps_from_mask(
    mask_xyz: np.ndarray,
    *,
    axis: str,
    thickness_voxels: int,
    intrusion_depth_voxels: int | None = None,
    shape: str = "anatomy",
) -> tuple[np.ndarray, np.ndarray]:
    axis_index = AXIS_TO_INDEX[axis]
    thickness = max(1, int(thickness_voxels))
    intrusion = max(1, int(intrusion_depth_voxels or round(thickness * 2.5)))
    inferior = np.zeros(mask_xyz.shape, dtype=bool)
    superior = np.zeros(mask_xyz.shape, dtype=bool)
    lateral_shape = tuple(mask_xyz.shape[idx] for idx in range(3) if idx != axis_index)
    lows = np.full(lateral_shape, -1, dtype=np.int32)
    highs = np.full(lateral_shape, -1, dtype=np.int32)
    for lateral in np.ndindex(lateral_shape):
        selector = _column_selector(lateral, axis_index)
        line = mask_xyz[tuple(selector)]
        occupied = np.flatnonzero(line)
        if occupied.size == 0:
            continue
        lows[lateral] = int(occupied.min())
        highs[lateral] = int(occupied.max())
    valid = lows >= 0
    if not np.any(valid):
        return inferior, superior
    global_lo = int(np.min(lows[valid]))
    global_hi = int(np.max(highs[valid]))
    lower_limit = min(global_lo + intrusion, int(np.max(lows[valid])))
    upper_limit = max(global_hi - intrusion, int(np.min(highs[valid])))
    inferior_footprint = _shape_footprint(
        _clean_largest_2d_component(valid & (lows <= lower_limit)),
        shape=shape,
    )
    superior_footprint = _shape_footprint(
        _clean_largest_2d_component(valid & (highs >= upper_limit)),
        shape=shape,
    )
    for lateral in np.ndindex(lateral_shape):
        if inferior_footprint[lateral]:
            lo = int(lows[lateral])
            selector = _column_selector(lateral, axis_index)
            selector[axis_index] = slice(max(0, global_lo - thickness), lo)
            inferior[tuple(selector)] = True
        if superior_footprint[lateral]:
            hi = int(highs[lateral])
            selector = _column_selector(lateral, axis_index)
            selector[axis_index] = slice(
                hi + 1,
                min(mask_xyz.shape[axis_index], global_hi + 1 + thickness),
            )
            superior[tuple(selector)] = True
    return inferior, superior


def _column_selector(lateral: tuple[int, ...], axis_index: int) -> list:
    selector: list = [slice(None), slice(None), slice(None)]
    lateral_iter = iter(lateral)
    for idx in range(3):
        if idx != axis_index:
            selector[idx] = next(lateral_iter)
    return selector


def _largest_2d_component(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    visited = np.zeros(values.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    for start_array in np.argwhere(values):
        start = tuple(int(v) for v in start_array)
        if visited[start]:
            continue
        component: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([start])
        visited[start] = True
        while queue:
            coord = queue.popleft()
            component.append(coord)
            for axis in range(2):
                for offset in (-1, 1):
                    neighbor = [coord[0], coord[1]]
                    neighbor[axis] += offset
                    if neighbor[axis] < 0 or neighbor[axis] >= values.shape[axis]:
                        continue
                    token = tuple(neighbor)
                    if visited[token] or not values[token]:
                        continue
                    visited[token] = True
                    queue.append(token)
        if len(component) > len(best):
            best = component
    out = np.zeros(values.shape, dtype=bool)
    if best:
        coords = tuple(np.asarray(best, dtype=np.int64).T)
        out[coords] = True
    return out


def _clean_largest_2d_component(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    opened = _dilate_2d(_erode_2d(values))
    if int(opened.sum()) >= max(4, int(values.sum() * 0.25)):
        values = opened
    return _largest_2d_component(values)


def _shape_footprint(mask: np.ndarray, *, shape: str) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if not np.any(values):
        return values
    mode = str(shape).strip().lower()
    if mode in {"", "anatomy", "projected", "mask"}:
        return values
    coords = np.argwhere(values)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    yy, xx = np.indices(values.shape, dtype=np.float64)
    center = (lo + hi) / 2.0
    half = np.maximum((hi - lo + 1) / 2.0, 0.5)
    if mode == "square":
        shaped = np.ones(values.shape, dtype=bool)
    elif mode in {"round", "circle", "circular"}:
        norm_y = (yy - center[0]) / half[0]
        norm_x = (xx - center[1]) / half[1]
        shaped = (norm_y * norm_y + norm_x * norm_x) <= 1.0
    elif mode in {"hex", "hexagon", "hexagonal"}:
        norm_y = np.abs((yy - center[0]) / half[0])
        norm_x = np.abs((xx - center[1]) / half[1])
        shaped = (norm_x <= 1.0) & (norm_y <= 1.0) & (norm_x + norm_y / 2.0 <= 1.0)
    else:
        raise ValueError(
            "disk shape must be one of anatomy, square, round, or hex"
        )
    out = np.zeros(values.shape, dtype=bool)
    slices = tuple(slice(int(lo[idx]), int(hi[idx]) + 1) for idx in range(2))
    out[slices] = shaped[slices]
    return out


def _erode_2d(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _dilate_2d(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def nodes_for_labels(
    labels_xyz: np.ndarray,
    label_map: dict[str, int],
    *,
    material_xyz: np.ndarray,
) -> dict[str, list[tuple[int, int, int]]]:
    return {
        name: nodes_from_labeled_voxels(
            labels_xyz,
            label=label,
            selection="surface_nodes",
            material=material_xyz,
        )
        for name, label in label_map.items()
    }


def displacement_from_load_case(
    load_case_config: dict[str, Any] | None,
    *,
    axis: str,
    dimensions_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
    default: float,
    length_mm: float | None = None,
) -> float:
    cfg = {} if load_case_config is None else load_case_config
    if "displacement" in cfg:
        return float(cfg["displacement"])
    if "normal_displacement" in cfg:
        return float(cfg["normal_displacement"])
    if "target_displacement_percent" in cfg:
        strain = float(cfg["target_displacement_percent"]) / 100.0
    else:
        strain = float(cfg.get("strain", cfg.get("normal_strain", default)))
    axis_index = AXIS_TO_INDEX[axis]
    if length_mm is None:
        length = float(dimensions_xyz[axis_index]) * float(spacing[axis_index])
    else:
        length = float(length_mm)
    return strain * length


def occupied_length_mm(
    material_xyz: np.ndarray,
    *,
    axis: str,
    spacing: tuple[float, float, float],
) -> float:
    axis_index = AXIS_TO_INDEX[axis]
    lateral_axes = tuple(idx for idx in range(3) if idx != axis_index)
    occupied = np.any(np.asarray(material_xyz) > 0, axis=lateral_axes)
    indices = np.where(occupied)[0]
    if indices.size == 0:
        raise ValueError(f"Could not infer occupied model length along {axis}")
    return float(indices[-1] - indices[0] + 1) * float(spacing[axis_index])


def constrained_contact_bcs(
    node_sets: dict[str, list[tuple[int, int, int]]],
    *,
    inferior_name: str,
    superior_name: str,
    axis: str,
    displacement: float,
    dimensions_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> BoundaryConditionSet:
    axis_index = AXIS_TO_INDEX[axis]
    dofs = ["x", "y", "z"]
    lateral = [name for name, idx in AXIS_TO_INDEX.items() if idx != axis_index]
    return boundary_conditions_from_nodesets(
        node_sets,
        fixed=[{"nodeset": inferior_name, "dofs": dofs, "value": 0.0}],
        prescribed=[
            {"nodeset": superior_name, "dofs": lateral, "value": 0.0},
            {"nodeset": superior_name, "dof": axis, "value": displacement},
        ],
        dimensions_xyz=dimensions_xyz,
        spacing=spacing,
    )


def sideways_fall_bcs(
    node_sets: dict[str, list[tuple[int, int, int]]],
    *,
    displacement: float,
    dimensions_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> BoundaryConditionSet:
    return boundary_conditions_from_nodesets(
        node_sets,
        fixed=[
            {
                "nodeset": "greater_trochanter_pmma",
                "dofs": ["x", "y", "z"],
                "value": 0.0,
            },
            {"nodeset": "distal_femur", "dofs": ["x", "y", "z"], "value": 0.0},
        ],
        prescribed=[
            {"nodeset": "femoral_head_pmma", "dof": "y", "value": displacement}
        ],
        dimensions_xyz=dimensions_xyz,
        spacing=spacing,
    )


def export_model_artifacts(
    *,
    material_xyz: np.ndarray,
    labels_xyz: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    node_sets: dict[str, list[tuple[int, int, int]]],
    element_sets: dict[str, int],
    boundary_conditions: BoundaryConditionSet,
    model_config: dict[str, Any],
    base_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    output_cfg = model_config.get("outputs", {})
    exported: dict[str, Path] = {}
    if "material_image" in output_cfg:
        exported["material_image"] = export_scalar_image(
            ImageGrid(material_xyz.astype(np.float32), spacing, origin),
            resolve_path(output_cfg["material_image"], base_dir=base_dir),
        )
    if "nodeset_image" in output_cfg:
        exported["nodeset_image"] = export_scalar_image(
            ImageGrid(labels_xyz.astype(np.float32), spacing, origin),
            resolve_path(output_cfg["nodeset_image"], base_dir=base_dir),
        )
    if "qc_image" in output_cfg:
        qc_path = resolve_path(output_cfg["qc_image"], base_dir=base_dir)
        qc_material, qc_labels, qc_origin = _crop_for_qc(
            material_xyz,
            labels_xyz,
            spacing=spacing,
            origin=origin,
        )
        exported["qc_image"] = write_case_overview(
            qc_material,
            output_path=qc_path,
            spacing=spacing,
            origin=qc_origin,
            field_xyz=qc_labels.astype(np.float32),
            field_name="MODEL LABELS",
            material_labels_xyz=qc_labels,
            boundary_conditions=boundary_conditions,
            title=str(metadata["model"]["type"]),
        )
    if "manifest" in output_cfg:
        manifest_path = resolve_path(output_cfg["manifest"], base_dir=base_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            **metadata,
            "shape_zyx": list(to_output_order(material_xyz, array_order="zyx").shape),
            "spacing": list(spacing),
            "origin": list(origin),
            "node_sets": {name: len(nodes) for name, nodes in node_sets.items()},
            "element_sets": element_sets,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        exported["manifest"] = manifest_path
    return exported


def to_zyx(array_xyz: np.ndarray) -> np.ndarray:
    return to_output_order(array_xyz, array_order="zyx")


def require_non_empty(nodes: dict[str, list[tuple[int, int, int]]]) -> None:
    empty = [name for name, values in nodes.items() if not values]
    if empty:
        raise ValueError(f"generated node set(s) are empty: {', '.join(empty)}")


def _crop_for_qc(
    material_xyz: np.ndarray,
    labels_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    margin: int = 8,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    active = np.asarray(material_xyz) > 0
    if not np.any(active):
        return material_xyz, labels_xyz, origin
    coords = np.argwhere(active)
    lo = np.maximum(coords.min(axis=0) - int(margin), 0)
    hi = np.minimum(coords.max(axis=0) + int(margin) + 1, material_xyz.shape)
    slices = tuple(slice(int(lo[idx]), int(hi[idx])) for idx in range(3))
    cropped_origin = tuple(
        float(origin[idx]) + float(lo[idx]) * float(spacing[idx]) for idx in range(3)
    )
    return material_xyz[slices], labels_xyz[slices], cropped_origin


def _triple(value, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(float(v) for v in value)


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _density_floor_config_value(
    e_cfg: dict[str, Any], density_cfg: dict[str, Any]
) -> float | None:
    for cfg in (e_cfg, density_cfg):
        for key in ("minimum_e_mpa", "floor_e_mpa", "floor_mpa", "floor"):
            value = cfg.get(key)
            if value is not None:
                return float(value)
    return None
