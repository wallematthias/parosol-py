from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from parosol_py.nodesets import nodes_from_mask_face

from .alignment import align_mask_to_reference
from .common import (
    AXIS_TO_INDEX,
    displacement_from_load_case,
    export_model_artifacts,
    pad_arrays_to_foreground_margin,
    load_density_and_mask,
    material_from_density,
    nodes_for_labels,
    occupied_length_mm,
    pad_along_axis,
    pmma_spec,
    require_non_empty,
    sideways_fall_bcs,
    to_zyx,
)
from .types import BuiltModel


def build_proximal_femur_model(
    model_config: dict[str, Any],
    *,
    base_dir: Path,
    material_config: dict[str, Any],
    load_case_config: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
) -> BuiltModel:
    density_zyx, mask_zyx, spacing, origin = load_density_and_mask(
        model_config, base_dir=base_dir, preprocessing_config=preprocessing_config
    )
    labels = model_config.get("labels", {})
    femur_label = int(
        labels.get(
            "femur",
            labels.get("left", labels.get("right", _cap_target_label(model_config, 2))),
        )
    )
    femur_zyx = mask_zyx == femur_label
    if not np.any(femur_zyx):
        raise ValueError(f"proximal femur label {femur_label} is absent")
    registration_meta = {"enabled": False}
    registration = model_config.get("registration", model_config.get("pose", {}))
    if isinstance(registration, dict) and registration.get("enabled", False):
        aligned = align_mask_to_reference(
            density_zyx=density_zyx,
            mask_zyx=femur_zyx,
            spacing=spacing,
            origin=origin,
            registration_config=registration,
            base_dir=base_dir,
        )
        density_zyx = aligned.density_zyx
        femur_zyx = aligned.mask_zyx
        spacing = aligned.spacing
        origin = aligned.origin
        registration_meta = aligned.metadata
    geometry = model_config.get("geometry", {})
    axis = str(geometry.get("cap_axis", "y")).strip().lower()
    if axis not in AXIS_TO_INDEX:
        raise ValueError("model.geometry.cap_axis must be one of x, y, z")
    thickness = _thickness_voxels(model_config, spacing=spacing, axis=axis)
    intrusion_depth = _intrusion_depth_voxels(
        model_config, spacing=spacing, axis=axis, default=thickness
    )
    padded, origin = pad_arrays_to_foreground_margin(
        anchor_mask_zyx=femur_zyx,
        spacing=spacing,
        origin=origin,
        margin_voxels=thickness + intrusion_depth,
        arrays={"density": density_zyx, "mask": femur_zyx},
        constant_values={"density": 0.0, "mask": False},
    )
    density_zyx = padded["density"]
    femur_zyx = padded["mask"]
    bone_mpa_zyx, poisson_ratio = material_from_density(
        density_zyx,
        femur_zyx,
        material_config=material_config,
    )
    material_xyz = np.transpose(bone_mpa_zyx, (2, 1, 0))
    femur_xyz = np.transpose(femur_zyx, (2, 1, 0))
    shaft_meta: dict[str, Any] = {"enabled": False}
    shaft_cfg = _shaft_standardization_config(model_config)
    if shaft_cfg.get("enabled", False):
        material_xyz, femur_xyz, shaft_meta = standardize_femur_shaft_length(
            density_xyz=material_xyz,
            mask_xyz=femur_xyz,
            spacing=spacing,
            origin=origin,
            cut_mode=str(shaft_cfg.get("cut_mode", "proportional_length")),
            retained_length_mm=_optional_float(shaft_cfg.get("fixed_length_mm")),
            lesser_trochanter_distal_offset_mm=float(
                shaft_cfg.get("lesser_trochanter_distal_offset_mm", 50.0)
            ),
            lesser_trochanter_distal_offset_percent=_optional_float(
                shaft_cfg.get("lesser_trochanter_distal_offset_percent")
            ),
            cut_axis=str(shaft_cfg.get("cut_axis", "z")),
            cut_side=str(shaft_cfg.get("cut_side", "low")),
            reference_extent_axis=str(shaft_cfg.get("reference_extent_axis", "y")),
            retain_multiplier=float(shaft_cfg.get("retain_multiplier", 1.35)),
        )
    cap_shape = _cap_shape(model_config)
    pmma = pmma_spec(material_config)

    material_xyz = pad_along_axis(
        material_xyz,
        axis=axis,
        before=thickness,
        after=thickness,
        value=0.0,
    )
    femur_xyz = pad_along_axis(
        femur_xyz,
        axis=axis,
        before=thickness,
        after=thickness,
        value=False,
    )
    fh_cap, gt_cap, distal = _femur_caps_and_distal(
        femur_xyz,
        axis=axis,
        thickness=thickness,
        intrusion_depth=intrusion_depth,
        shape=cap_shape,
        distal_support_mode=str(shaft_meta.get("support_mode", "flat_cut_face")),
    )
    material_xyz[fh_cap | gt_cap] = pmma["E"]

    labels_xyz = np.zeros(material_xyz.shape, dtype=np.uint8)
    labels_xyz[femur_xyz] = 1
    labels_xyz[fh_cap] = 20
    labels_xyz[gt_cap] = 21
    labels_xyz[distal] = 22
    node_sets = nodes_for_labels(
        labels_xyz,
        {
            "distal_femur": 22,
        },
        material_xyz=material_xyz,
    )
    node_sets.update(
        {
            "femoral_head_pmma": nodes_from_mask_face(fh_cap, axis=axis, side=-1),
            "greater_trochanter_pmma": nodes_from_mask_face(
                gt_cap, axis=axis, side=1
            ),
        }
    )
    require_non_empty(node_sets)
    displacement = displacement_from_load_case(
        load_case_config,
        axis="y",
        dimensions_xyz=tuple(int(v) for v in material_xyz.shape),
        spacing=spacing,
        default=1.0 / max(float(material_xyz.shape[1]) * spacing[1], 1.0),
        length_mm=occupied_length_mm(material_xyz, axis="y", spacing=spacing),
    )
    boundary_conditions = sideways_fall_bcs(
        node_sets,
        displacement=displacement,
        dimensions_xyz=tuple(int(v) for v in material_xyz.shape),
        spacing=spacing,
    )
    element_sets = {
        "bone": int(np.count_nonzero(femur_xyz)),
        "femoral_head_cap": int(np.count_nonzero(fh_cap)),
        "greater_trochanter_cap": int(np.count_nonzero(gt_cap)),
        "distal_femur": int(np.count_nonzero(distal)),
    }
    metadata = {
        "model": {
            "type": "proximal_femur",
            "side": str(model_config.get("side", "left")),
            "cap_axis": axis,
            "load_axis": "y",
            "load_direction": "y",
            "pmma_thickness_voxels": thickness,
            "caps": {
                "target_label": str(_cap_target_label(model_config, femur_label)),
                "shape": cap_shape,
                "thickness_voxels": thickness,
                "intrusion_depth_voxels": intrusion_depth,
                "method": "projected_cap",
            },
            "labels": {"femur": femur_label},
            "displacement": displacement,
            "registration": registration_meta,
            "shaft_standardization": shaft_meta,
        },
        "materials": {"pmma": pmma, "poisson_ratio": poisson_ratio},
    }
    padded_origin = _padded_origin(origin, spacing, AXIS_TO_INDEX[axis], thickness)
    exported = export_model_artifacts(
        material_xyz=material_xyz,
        labels_xyz=labels_xyz,
        spacing=spacing,
        origin=padded_origin,
        node_sets=node_sets,
        element_sets=element_sets,
        boundary_conditions=boundary_conditions,
        model_config=model_config,
        base_dir=base_dir,
        metadata=metadata,
    )
    return BuiltModel(
        material=to_zyx(material_xyz),
        spacing=spacing,
        origin=padded_origin,
        poisson_ratio=poisson_ratio,
        boundary_conditions=boundary_conditions,
        node_sets=node_sets,
        element_sets=element_sets,
        postprocess_mask=to_zyx(femur_xyz),
        exported=exported,
        metadata=metadata,
    )

def _femur_caps_and_distal(
    femur_xyz: np.ndarray,
    *,
    axis: str,
    thickness: int,
    intrusion_depth: int,
    shape: str,
    distal_support_mode: str = "flat_cut_face",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_index = AXIS_TO_INDEX[axis]
    coords = np.argwhere(femur_xyz)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    fh = np.zeros(femur_xyz.shape, dtype=bool)
    gt = np.zeros(femur_xyz.shape, dtype=bool)
    distal = np.zeros(femur_xyz.shape, dtype=bool)
    axis_span = max(1, int(hi[axis_index] - lo[axis_index] + 1))
    z_mid = int(lo[2] + (hi[2] - lo[2]) / 2)
    fh_roi = femur_xyz.copy()
    fh_roi &= _axis_band(
        femur_xyz.shape,
        axis=axis_index,
        start=int(lo[axis_index]),
        stop=int(lo[axis_index]) + max(1, axis_span // 5),
    )
    fh_roi &= _axis_band(
        femur_xyz.shape,
        axis=2,
        start=z_mid,
        stop=int(hi[2]) + 1,
    )
    gt_roi = femur_xyz.copy()
    gt_roi &= _axis_band(
        femur_xyz.shape,
        axis=axis_index,
        start=int(hi[axis_index]) - max(1, axis_span // 20),
        stop=int(hi[axis_index]) + 1,
    )
    if not np.any(fh_roi):
        fh_roi = femur_xyz
    if not np.any(gt_roi):
        gt_roi = femur_xyz

    fh_contact = _femoral_head_contact_roi(
        femur_xyz,
        fh_roi,
        width_extension_voxels=10,
        long_axis_extension_voxels=80,
    )
    gt_contact = gt_roi.copy()

    fh = _fixture_cap_from_contact(
        fh_contact,
        axis=axis,
        direction="down",
        thickness=thickness,
        intrusion=intrusion_depth,
        shape="box",
        crop_to_contact=True,
    )
    gt = _fixture_cap_from_contact(
        gt_contact,
        axis=axis,
        direction="up",
        thickness=thickness,
        intrusion=intrusion_depth,
        shape="round" if shape in {"anatomy", "round", "circle", "circular"} else shape,
        crop_to_contact=False,
    )

    if str(distal_support_mode).strip().lower() == "flat_cut_face":
        distal = _contact_surface(femur_xyz, axis_index=2, direction="down")
    else:
        distal_axis = 2
        distal_slice = [slice(int(lo[idx]), int(hi[idx]) + 1) for idx in range(3)]
        distal_slice[distal_axis] = slice(
            int(lo[distal_axis]),
            min(int(lo[distal_axis]) + max(1, thickness), femur_xyz.shape[distal_axis]),
        )
        distal[tuple(distal_slice)] = femur_xyz[tuple(distal_slice)]
    return fh, gt, distal


def standardize_femur_shaft_length(
    density_xyz: np.ndarray,
    mask_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    cut_mode: str = "proportional_length",
    retained_length_mm: float | None = None,
    lesser_trochanter_distal_offset_mm: float = 50.0,
    lesser_trochanter_distal_offset_percent: float | None = None,
    cut_axis: str = "z",
    cut_side: str = "low",
    reference_extent_axis: str = "y",
    retain_multiplier: float = 1.35,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    occupied = np.argwhere(mask_xyz)
    if occupied.size == 0:
        raise ValueError("Cannot standardize femur shaft length from an empty mask.")
    mode = str(cut_mode).strip().lower()
    if mode == "proportional_length":
        meta = _proportional_length_crop_meta(
            mask_xyz,
            spacing=spacing,
            origin=origin,
            cut_axis=cut_axis,
            cut_side=cut_side,
            reference_extent_axis=reference_extent_axis,
            retain_multiplier=retain_multiplier,
        )
        cut_axis_name = str(meta["cut_axis"])
        cut_axis_index = AXIS_TO_INDEX[cut_axis_name]
        cut_coordinate = float(meta["cut_coordinate_mm"])
        keep_coords = origin[cut_axis_index] + np.arange(
            mask_xyz.shape[cut_axis_index], dtype=np.float64
        ) * spacing[cut_axis_index]
        keep = keep_coords >= cut_coordinate if str(meta["cut_side"]) == "low" else keep_coords <= cut_coordinate
    elif mode == "lesser_trochanter":
        meta = detect_lesser_trochanter_cut_z(
            mask_xyz,
            spacing=spacing,
            origin=origin,
            distal_offset_mm=lesser_trochanter_distal_offset_mm,
            distal_offset_percent=lesser_trochanter_distal_offset_percent,
        )
        cut_z = float(meta["cut_z"])
        keep_coords = origin[2] + np.arange(mask_xyz.shape[2], dtype=np.float64) * spacing[2]
        keep = keep_coords >= cut_z
    elif mode == "fixed_length":
        if retained_length_mm is None:
            raise ValueError("retained_length_mm is required for fixed_length femur cuts.")
        z_min = float(origin[2] + int(occupied[:, 2].min()) * spacing[2])
        z_max = float(origin[2] + int(occupied[:, 2].max()) * spacing[2])
        cut_z = float(z_max - float(retained_length_mm))
        meta = {"retained_length_mm": float(retained_length_mm), "cut_z": cut_z}
        keep_coords = origin[2] + np.arange(mask_xyz.shape[2], dtype=np.float64) * spacing[2]
        keep = keep_coords >= cut_z
    else:
        raise ValueError("cut_mode must be 'proportional_length', 'lesser_trochanter' or 'fixed_length'.")

    cropped_density = np.array(density_xyz, copy=True)
    cropped_mask = np.array(mask_xyz, copy=True)
    if mode == "proportional_length":
        axis_index = AXIS_TO_INDEX[str(meta["cut_axis"])]
        if axis_index == 0:
            cropped_density[~keep, :, :] = 0.0
            cropped_mask[~keep, :, :] = False
        elif axis_index == 1:
            cropped_density[:, ~keep, :] = 0.0
            cropped_mask[:, ~keep, :] = False
        else:
            cropped_density[:, :, ~keep] = 0.0
            cropped_mask[:, :, ~keep] = False
    else:
        cropped_density[:, :, ~keep] = 0.0
        cropped_mask[:, :, ~keep] = False
    cropped_occupied = np.argwhere(cropped_mask)
    if cropped_occupied.size == 0:
        raise ValueError("shaft standardization removed the entire femur mask.")
    cropped_lo = cropped_occupied.min(axis=0)
    cropped_hi = cropped_occupied.max(axis=0)
    meta.update(
        {
            "enabled": True,
            "cut_mode": mode,
            "support_mode": "flat_cut_face",
            "cropped_bbox_min_xyz": [int(v) for v in cropped_lo],
            "cropped_bbox_max_xyz": [int(v) for v in cropped_hi],
        }
    )
    if mode == "proportional_length":
        axis_name = str(meta["cut_axis"])
        axis_index = AXIS_TO_INDEX[axis_name]
        meta["retained_length_mm"] = float(
            (int(cropped_hi[axis_index]) - int(cropped_lo[axis_index]) + 1) * spacing[axis_index]
        )
        if axis_name == "z":
            meta["cut_z"] = float(meta["cut_coordinate_mm"])
    else:
        z_min = float(origin[2] + int(occupied[:, 2].min()) * spacing[2])
        z_max = float(origin[2] + int(occupied[:, 2].max()) * spacing[2])
        meta.update(
            {
                "mask_z_min": z_min,
                "mask_z_max": z_max,
                "retained_length_mm": float(z_max - float(meta["cut_z"])),
            }
        )
    if lesser_trochanter_distal_offset_percent is not None:
        meta["lesser_trochanter_distal_offset_percent"] = float(
            lesser_trochanter_distal_offset_percent
        )
    if mode == "lesser_trochanter":
        meta["lesser_trochanter_distal_offset_mm"] = float(
            meta.get("distal_offset_mm", lesser_trochanter_distal_offset_mm)
        )
    return cropped_density, cropped_mask, meta


def _proportional_length_crop_meta(
    mask_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    cut_axis: str,
    cut_side: str,
    reference_extent_axis: str,
    retain_multiplier: float,
) -> dict[str, Any]:
    cut_axis_name = str(cut_axis).strip().lower()
    reference_axis_name = str(reference_extent_axis).strip().lower()
    side = str(cut_side).strip().lower()
    if cut_axis_name not in AXIS_TO_INDEX:
        raise ValueError("shaft_standardization.cut_axis must be one of x, y, z")
    if reference_axis_name not in AXIS_TO_INDEX:
        raise ValueError("shaft_standardization.reference_extent_axis must be one of x, y, z")
    if side not in {"low", "high"}:
        raise ValueError("shaft_standardization.cut_side must be 'low' or 'high'")
    if retain_multiplier <= 0:
        raise ValueError("shaft_standardization.retain_multiplier must be positive")
    occupied = np.argwhere(mask_xyz)
    lo = occupied.min(axis=0)
    hi = occupied.max(axis=0)
    reference_axis_index = AXIS_TO_INDEX[reference_axis_name]
    cut_axis_index = AXIS_TO_INDEX[cut_axis_name]
    reference_extent_mm = float((int(hi[reference_axis_index]) - int(lo[reference_axis_index]) + 1) * spacing[reference_axis_index])
    retained_length_mm = float(reference_extent_mm * float(retain_multiplier))
    cut_axis_min = float(origin[cut_axis_index] + int(lo[cut_axis_index]) * spacing[cut_axis_index])
    cut_axis_max = float(origin[cut_axis_index] + int(hi[cut_axis_index]) * spacing[cut_axis_index])
    occupied_length_mm = float((int(hi[cut_axis_index]) - int(lo[cut_axis_index]) + 1) * spacing[cut_axis_index])
    warnings: list[str] = []
    if retained_length_mm >= occupied_length_mm:
        warnings.append(
            "requested retained length exceeds occupied length along the cut axis; crop collapses to full extent"
        )
    if side == "low":
        cut_coordinate = max(cut_axis_min, cut_axis_max - retained_length_mm)
    else:
        cut_coordinate = min(cut_axis_max, cut_axis_min + retained_length_mm)
    return {
        "cut_axis": cut_axis_name,
        "cut_side": side,
        "reference_extent_axis": reference_axis_name,
        "reference_extent_mm": reference_extent_mm,
        "retain_multiplier": float(retain_multiplier),
        "occupied_length_mm": occupied_length_mm,
        "cut_coordinate_mm": float(cut_coordinate),
        "warnings": warnings,
    }


def detect_lesser_trochanter_cut_z(
    mask_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    distal_offset_mm: float = 0.0,
    distal_offset_percent: float | None = None,
    min_distal_to_greater_mm: float = 8.0,
    max_distal_to_greater_mm: float = 45.0,
) -> dict[str, float | None]:
    profile = _femur_z_profile(mask_xyz, spacing=spacing, origin=origin)
    z = profile["z"]
    area = _smooth_profile(profile["area"], window=7)
    y_max = _smooth_profile(profile["y_max"], window=7)
    z_min = float(z.min())
    z_max = float(z.max())
    if z_max - z_min < max_distal_to_greater_mm:
        raise ValueError(
            "Femur scan does not include enough proximal-distal coverage to identify "
            "the lesser trochanter."
        )
    proximal_mask = z >= (z_min + 0.55 * (z_max - z_min))
    if not np.any(proximal_mask):
        raise ValueError("Cannot identify greater trochanter from femur profile.")
    proximal_indices = np.where(proximal_mask)[0]
    greater_index = int(proximal_indices[np.argmax(y_max[proximal_indices])])
    greater_z = float(z[greater_index])
    distal_mask = (
        (z <= greater_z - float(min_distal_to_greater_mm))
        & (z >= greater_z - float(max_distal_to_greater_mm))
    )
    distal_indices = np.where(distal_mask)[0]
    if distal_indices.size < 5:
        raise ValueError(
            "Femur scan does not include the distal profile needed to identify "
            "the lesser trochanter."
        )
    lesser_index, lesser_z = _peak_center_z(z, area, distal_indices)
    offset_mm = float(distal_offset_mm)
    if distal_offset_percent is not None:
        offset_mm = abs(greater_z - lesser_z) * float(distal_offset_percent) / 100.0
    cut_z = lesser_z - offset_mm
    shaft_area = (
        float(np.median(area[z < lesser_z - 10.0])) if np.any(z < lesser_z - 10.0) else 0.0
    )
    if shaft_area > 0.0 and float(area[lesser_index]) < 1.08 * shaft_area:
        raise ValueError(
            "Could not identify a clear lesser-trochanter cross-section peak; "
            "the femur field of view is likely incomplete or the alignment failed."
        )
    return {
        "cut_z": float(cut_z),
        "lesser_trochanter_z": float(lesser_z),
        "greater_trochanter_z": float(greater_z),
        "mask_z_min": z_min,
        "mask_z_max": z_max,
        "retained_length_mm": float(z_max - cut_z),
        "distal_offset_mm": float(offset_mm),
        "distal_offset_percent": None if distal_offset_percent is None else float(distal_offset_percent),
        "lesser_area": float(area[lesser_index]),
        "shaft_area_median": float(shaft_area),
    }


def _femur_z_profile(
    mask_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    data = np.asarray(mask_xyz, dtype=bool)
    rows: list[tuple[float, float, float, float]] = []
    y_coords = origin[1] + np.arange(data.shape[1], dtype=np.float64) * spacing[1]
    z_coords = origin[2] + np.arange(data.shape[2], dtype=np.float64) * spacing[2]
    for z_index, z_value in enumerate(z_coords):
        plane = data[:, :, z_index]
        if not plane.any():
            continue
        y_indices = np.where(plane)[1]
        rows.append(
            (
                float(z_value),
                float(plane.sum()),
                float(y_coords[y_indices].min()),
                float(y_coords[y_indices].max()),
            )
        )
    if not rows:
        raise ValueError("Cannot compute femur profile from an empty mask.")
    profile = np.asarray(rows, dtype=np.float64)
    return {
        "z": profile[:, 0],
        "area": profile[:, 1],
        "y_min": profile[:, 2],
        "y_max": profile[:, 3],
    }


def _smooth_profile(values: np.ndarray, *, window: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if values.size < window or window == 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / float(window), mode="valid")


def _peak_center_z(
    z: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
    *,
    relative_height: float = 0.95,
) -> tuple[int, float]:
    peak_index = int(indices[np.argmax(values[indices])])
    threshold = float(values[peak_index]) * float(relative_height)
    left = peak_index
    right = peak_index
    valid = set(int(i) for i in indices)
    while left - 1 in valid and values[left - 1] >= threshold:
        left -= 1
    while right + 1 in valid and values[right + 1] >= threshold:
        right += 1
    plateau = np.arange(left, right + 1, dtype=int)
    weights = np.maximum(values[plateau], 0.0)
    if float(weights.sum()) <= 0.0:
        return peak_index, float(z[peak_index])
    return peak_index, float(np.average(z[plateau], weights=weights))


def _femoral_head_contact_roi(
    femur_xyz: np.ndarray,
    fh_roi: np.ndarray,
    *,
    width_extension_voxels: int,
    long_axis_extension_voxels: int,
) -> np.ndarray:
    coords = np.argwhere(fh_roi)
    if coords.size == 0:
        return fh_roi.copy()
    lo = coords.min(axis=0).astype(np.float64)
    hi = coords.max(axis=0).astype(np.float64)
    x_center = (lo[0] + hi[0]) / 2.0
    z_center = (lo[2] + hi[2]) / 2.0
    x_len = (hi[0] - lo[0] + 1.0)
    z_len = (hi[2] - lo[2] + 1.0)
    expanded_x = z_len + float(width_extension_voxels)
    expanded_z = x_len + float(long_axis_extension_voxels)
    x_lo = max(0, int(np.floor(x_center - expanded_x / 2.0)))
    x_hi = min(femur_xyz.shape[0], int(np.ceil(x_center + expanded_x / 2.0)))
    y_lo = max(0, int(lo[1]))
    y_hi = min(femur_xyz.shape[1], int(hi[1] + 1.0))
    z_lo = max(0, int(np.floor(z_center - expanded_z / 2.0)))
    z_hi = min(femur_xyz.shape[2], int(np.ceil(z_center + expanded_z / 2.0)))
    roi = np.zeros_like(femur_xyz, dtype=bool)
    roi[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi] = femur_xyz[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi]
    return roi


def _fixture_cap_from_contact(
    contact_xyz: np.ndarray,
    *,
    axis: str,
    direction: str,
    thickness: int,
    intrusion: int,
    shape: str,
    crop_to_contact: bool,
) -> np.ndarray:
    values = np.asarray(contact_xyz, dtype=bool)
    if not np.any(values):
        return np.zeros_like(values, dtype=bool)
    axis_index = AXIS_TO_INDEX[axis]
    surface = _contact_surface(values, axis_index=axis_index, direction=direction)
    footprint = _projected_footprint(surface, axis_index=axis_index, shape=shape)
    if crop_to_contact:
        footprint &= _projected_contact_within_depth(
            values,
            axis_index=axis_index,
            direction=direction,
            depth=intrusion,
        )
    cap = np.zeros_like(values, dtype=bool)
    hit = np.where(values)[axis_index]
    contact = int(hit.min() if direction == "down" else hit.max())
    if direction == "up":
        slab = slice(
            max(contact - int(intrusion) + 1, 0),
            min(contact + int(thickness) + 1, values.shape[axis_index]),
        )
    else:
        slab = slice(
            max(contact - int(thickness), 0),
            min(contact + int(intrusion), values.shape[axis_index]),
        )
    if axis_index == 0:
        cap[slab, :, :] = footprint
    elif axis_index == 1:
        cap[:, slab, :] = footprint[:, None, :]
    else:
        cap[:, :, slab] = footprint[:, :, None]
    return _largest_connected_component_3d(cap)


def _contact_surface(
    mask_xyz: np.ndarray,
    *,
    axis_index: int,
    direction: str,
) -> np.ndarray:
    surface = np.zeros_like(mask_xyz, dtype=bool)
    dims = mask_xyz.shape
    if axis_index == 0:
        for j in range(dims[1]):
            for k in range(dims[2]):
                hit = np.flatnonzero(mask_xyz[:, j, k])
                if hit.size:
                    surface[int(hit.max() if direction == "up" else hit.min()), j, k] = True
    elif axis_index == 1:
        for i in range(dims[0]):
            for k in range(dims[2]):
                hit = np.flatnonzero(mask_xyz[i, :, k])
                if hit.size:
                    surface[i, int(hit.max() if direction == "up" else hit.min()), k] = True
    else:
        for i in range(dims[0]):
            for j in range(dims[1]):
                hit = np.flatnonzero(mask_xyz[i, j, :])
                if hit.size:
                    surface[i, j, int(hit.max() if direction == "up" else hit.min())] = True
    return surface


def _projected_footprint(
    surface_xyz: np.ndarray,
    *,
    axis_index: int,
    shape: str,
) -> np.ndarray:
    footprint = surface_xyz.any(axis=axis_index)
    if not np.any(footprint):
        return footprint
    mode = str(shape).strip().lower()
    coords = np.array(np.where(footprint))
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1)
    if mode in {"box", "square"}:
        out = np.zeros_like(footprint, dtype=bool)
        out[tuple(slice(int(mins[i]), int(maxs[i]) + 1) for i in range(2))] = True
        return out
    if mode in {"round", "circle", "circular"}:
        grids = np.ogrid[tuple(slice(0, size) for size in footprint.shape)]
        center = (mins + maxs) / 2.0
        radii = np.maximum((maxs - mins + 1) / 2.0, 0.5)
        distance = sum(((grids[i] - center[i]) / radii[i]) ** 2 for i in range(2))
        return distance <= 1.0
    return footprint


def _projected_contact_within_depth(
    mask_xyz: np.ndarray,
    *,
    axis_index: int,
    direction: str,
    depth: int,
) -> np.ndarray:
    hit = np.where(mask_xyz)[axis_index]
    contact = int(hit.max() if direction == "up" else hit.min())
    depth = max(int(depth), 1)
    if direction == "up":
        support_slice = slice(max(contact - depth + 1, 0), contact + 1)
    else:
        support_slice = slice(contact, min(contact + depth, mask_xyz.shape[axis_index]))
    if axis_index == 0:
        return mask_xyz[support_slice, :, :].any(axis=0)
    if axis_index == 1:
        return mask_xyz[:, support_slice, :].any(axis=1)
    return mask_xyz[:, :, support_slice].any(axis=2)


def _largest_connected_component_3d(mask_xyz: np.ndarray) -> np.ndarray:
    values = np.asarray(mask_xyz, dtype=bool)
    if not np.any(values):
        return values
    visited = np.zeros(values.shape, dtype=bool)
    best: list[tuple[int, int, int]] = []
    neighbors = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )
    for start_array in np.argwhere(values):
        start = tuple(int(v) for v in start_array)
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[tuple[int, int, int]] = []
        while stack:
            coord = stack.pop()
            component.append(coord)
            for dx, dy, dz in neighbors:
                nx, ny, nz = coord[0] + dx, coord[1] + dy, coord[2] + dz
                if (
                    nx < 0
                    or ny < 0
                    or nz < 0
                    or nx >= values.shape[0]
                    or ny >= values.shape[1]
                    or nz >= values.shape[2]
                ):
                    continue
                token = (nx, ny, nz)
                if visited[token] or not values[token]:
                    continue
                visited[token] = True
                stack.append(token)
        if len(component) > len(best):
            best = component
    out = np.zeros_like(values, dtype=bool)
    if best:
        out[tuple(np.asarray(best, dtype=np.int64).T)] = True
    return out


def _axis_band(
    shape: tuple[int, int, int],
    *,
    axis: int,
    start: int,
    stop: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    slices = [slice(None), slice(None), slice(None)]
    slices[axis] = slice(max(0, start), min(shape[axis], stop))
    mask[tuple(slices)] = True
    return mask


def _thickness_voxels(
    model_config: dict[str, Any],
    *,
    spacing: tuple[float, float, float],
    axis: str,
) -> int:
    geometry = model_config.get("geometry", {})
    cap = _cap_config(model_config)
    if "thickness_voxels" in cap:
        value = int(cap["thickness_voxels"])
    elif "thickness_mm" in cap:
        value = int(round(float(cap["thickness_mm"]) / spacing[AXIS_TO_INDEX[axis]]))
    elif "pmma_thickness_voxels" in geometry:
        value = int(geometry["pmma_thickness_voxels"])
    else:
        value = int(
            round(
                float(geometry.get("pmma_thickness_mm", 3.0))
                / spacing[AXIS_TO_INDEX[axis]]
            )
        )
    if value < 1:
        raise ValueError("PMMA cap thickness rounds to zero voxels")
    return value


def _intrusion_depth_voxels(
    model_config: dict[str, Any],
    *,
    spacing: tuple[float, float, float],
    axis: str,
    default: int,
) -> int:
    geometry = model_config.get("geometry", {})
    cap = _cap_config(model_config)
    if "intrusion_depth_voxels" in cap:
        value = int(cap["intrusion_depth_voxels"])
    elif "intrusion_depth_mm" in cap:
        value = int(
            round(float(cap["intrusion_depth_mm"]) / spacing[AXIS_TO_INDEX[axis]])
        )
    elif "intrusion_depth_voxels" in geometry:
        value = int(geometry["intrusion_depth_voxels"])
    elif "intrusion_depth_mm" in geometry:
        value = int(
            round(float(geometry["intrusion_depth_mm"]) / spacing[AXIS_TO_INDEX[axis]])
        )
    else:
        value = int(round(float(default) * 2.0))
    if value < 1:
        raise ValueError("PMMA cap intrusion depth rounds to zero voxels")
    return value


def _cap_shape(model_config: dict[str, Any]) -> str:
    geometry = model_config.get("geometry", {})
    cap = _cap_config(model_config)
    return str(cap.get("shape", geometry.get("cap_shape", "anatomy"))).strip().lower()


def _cap_target_label(model_config: dict[str, Any], default: int) -> int:
    cap = _cap_config(model_config)
    raw = cap.get("target_label", cap.get("target", default))
    if str(raw).strip().lower() in {"femur", "bone", "proximal_femur"}:
        return int(default)
    return int(raw)


def _cap_config(model_config: dict[str, Any]) -> dict[str, Any]:
    geometry = model_config.get("geometry", {})
    cap = geometry.get("cap")
    if isinstance(cap, dict):
        return cap
    disk = geometry.get("disk")
    return disk if isinstance(disk, dict) else {}


def _shaft_standardization_config(model_config: dict[str, Any]) -> dict[str, Any]:
    geometry = model_config.get("geometry", {})
    config = geometry.get("shaft_standardization")
    return config if isinstance(config, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _padded_origin(
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    axis_index: int,
    thickness: int,
) -> tuple[float, float, float]:
    out = list(origin)
    out[axis_index] -= float(thickness) * float(spacing[axis_index])
    return tuple(out)
