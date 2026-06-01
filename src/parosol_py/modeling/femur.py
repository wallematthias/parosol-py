from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    AXIS_TO_INDEX,
    displacement_from_load_case,
    export_model_artifacts,
    load_density_and_mask,
    material_from_density,
    nodes_for_labels,
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
) -> BuiltModel:
    density_zyx, mask_zyx, spacing, origin = load_density_and_mask(
        model_config, base_dir=base_dir
    )
    labels = model_config.get("labels", {})
    femur_label = int(labels.get("femur", labels.get("left", labels.get("right", 2))))
    femur_zyx = mask_zyx == femur_label
    if not np.any(femur_zyx):
        raise ValueError(f"proximal femur label {femur_label} is absent")
    bone_mpa_zyx, poisson_ratio = material_from_density(
        density_zyx,
        femur_zyx,
        material_config=material_config,
    )
    material_xyz = np.transpose(bone_mpa_zyx, (2, 1, 0))
    femur_xyz = np.transpose(femur_zyx, (2, 1, 0))
    geometry = model_config.get("geometry", {})
    axis = str(geometry.get("cap_axis", "y")).strip().lower()
    if axis not in AXIS_TO_INDEX:
        raise ValueError("model.geometry.cap_axis must be one of x, y, z")
    thickness = _thickness_voxels(model_config, spacing=spacing, axis=axis)
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
            "femoral_head_pmma": 20,
            "greater_trochanter_pmma": 21,
            "distal_femur": 22,
        },
        material_xyz=material_xyz,
    )
    require_non_empty(node_sets)
    displacement = displacement_from_load_case(
        load_case_config,
        axis="y",
        dimensions_xyz=tuple(int(v) for v in material_xyz.shape),
        spacing=spacing,
        default=-1.0 / max(float(material_xyz.shape[1]) * spacing[1], 1.0),
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
            "pmma_thickness_voxels": thickness,
            "labels": {"femur": femur_label},
            "displacement": displacement,
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
        exported=exported,
        metadata=metadata,
    )


def _femur_caps_and_distal(
    femur_xyz: np.ndarray,
    *,
    axis: str,
    thickness: int,
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

    _project_contact_cap(
        fh,
        fh_roi,
        axis_index=axis_index,
        side=-1,
        thickness=thickness,
    )
    _project_contact_cap(
        gt,
        gt_roi,
        axis_index=axis_index,
        side=1,
        thickness=thickness,
    )

    distal_axis = 2
    distal_slice = [slice(int(lo[idx]), int(hi[idx]) + 1) for idx in range(3)]
    distal_slice[distal_axis] = slice(
        int(lo[distal_axis]),
        min(int(lo[distal_axis]) + max(1, thickness), femur_xyz.shape[distal_axis]),
    )
    distal[tuple(distal_slice)] = femur_xyz[tuple(distal_slice)]
    return fh, gt, distal


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


def _project_contact_cap(
    out: np.ndarray,
    roi: np.ndarray,
    *,
    axis_index: int,
    side: int,
    thickness: int,
) -> None:
    lateral_shape = tuple(roi.shape[idx] for idx in range(3) if idx != axis_index)
    for lateral in np.ndindex(lateral_shape):
        selector = [slice(None), slice(None), slice(None)]
        lateral_iter = iter(lateral)
        for idx in range(3):
            if idx != axis_index:
                selector[idx] = next(lateral_iter)
        line = roi[tuple(selector)]
        occupied = np.flatnonzero(line)
        if occupied.size == 0:
            continue
        if side < 0:
            contact = int(occupied.min())
            selector[axis_index] = slice(max(0, contact - thickness), contact)
        else:
            contact = int(occupied.max())
            selector[axis_index] = slice(
                contact + 1, min(roi.shape[axis_index], contact + 1 + thickness)
            )
        out[tuple(selector)] = True


def _thickness_voxels(
    model_config: dict[str, Any],
    *,
    spacing: tuple[float, float, float],
    axis: str,
) -> int:
    geometry = model_config.get("geometry", {})
    if "pmma_thickness_voxels" in geometry:
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


def _padded_origin(
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    axis_index: int,
    thickness: int,
) -> tuple[float, float, float]:
    out = list(origin)
    out[axis_index] -= float(thickness) * float(spacing[axis_index])
    return tuple(out)
