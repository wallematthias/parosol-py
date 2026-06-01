from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    AXIS_TO_INDEX,
    constrained_contact_bcs,
    displacement_from_load_case,
    export_model_artifacts,
    load_density_and_mask,
    material_from_density,
    nodes_for_labels,
    pad_along_axis,
    pmma_spec,
    projected_caps_from_mask,
    require_non_empty,
    to_zyx,
)
from .types import BuiltModel


def build_spine_compression_model(
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
    body_label = int(labels.get("body", labels.get("vertebral_body", 20)))
    process_label = int(labels.get("process", labels.get("vertebral_process", 48)))
    body_mask_zyx = mask_zyx == body_label
    process_mask_zyx = mask_zyx == process_label
    if not np.any(body_mask_zyx):
        raise ValueError(f"spine model body label {body_label} is absent")
    if not np.any(process_mask_zyx):
        raise ValueError(f"spine model process label {process_label} is absent")

    active_zyx = body_mask_zyx | process_mask_zyx
    bone_mpa_zyx, poisson_ratio = material_from_density(
        density_zyx,
        active_zyx,
        material_config=material_config,
    )
    grid_bone = np.transpose(bone_mpa_zyx, (2, 1, 0))
    grid_body = np.transpose(body_mask_zyx, (2, 1, 0))
    grid_process = np.transpose(process_mask_zyx, (2, 1, 0))
    axis = str(model_config.get("geometry", {}).get("axis", "z")).strip().lower()
    if axis not in AXIS_TO_INDEX:
        raise ValueError("model.geometry.axis must be one of x, y, z")
    axis_index = AXIS_TO_INDEX[axis]
    thickness = _thickness_voxels(model_config, spacing=spacing, axis=axis)
    pmma = pmma_spec(material_config)

    material_xyz = pad_along_axis(
        grid_bone,
        axis=axis,
        before=thickness,
        after=thickness,
        value=0.0,
    )
    body_padded = pad_along_axis(
        grid_body,
        axis=axis,
        before=thickness,
        after=thickness,
        value=False,
    )
    process_padded = pad_along_axis(
        grid_process,
        axis=axis,
        before=thickness,
        after=thickness,
        value=False,
    )
    inferior, superior = projected_caps_from_mask(
        body_padded,
        axis=axis,
        thickness_voxels=thickness,
    )
    material_xyz[inferior | superior] = pmma["E"]

    labels_xyz = np.zeros(material_xyz.shape, dtype=np.uint8)
    labels_xyz[body_padded] = 1
    labels_xyz[process_padded] = 2
    labels_xyz[inferior] = 10
    labels_xyz[superior] = 11

    node_sets = nodes_for_labels(
        labels_xyz,
        {"inferior": 10, "superior": 11},
        material_xyz=material_xyz,
    )
    require_non_empty(node_sets)
    displacement = displacement_from_load_case(
        load_case_config,
        axis=axis,
        dimensions_xyz=tuple(int(v) for v in material_xyz.shape),
        spacing=spacing,
        default=-0.01,
    )
    boundary_conditions = constrained_contact_bcs(
        node_sets,
        inferior_name="inferior",
        superior_name="superior",
        axis=axis,
        displacement=displacement,
        dimensions_xyz=tuple(int(v) for v in material_xyz.shape),
        spacing=spacing,
    )
    element_sets = {
        "body": int(np.count_nonzero(body_padded)),
        "process": int(np.count_nonzero(process_padded)),
        "inferior_disk": int(np.count_nonzero(inferior)),
        "superior_disk": int(np.count_nonzero(superior)),
    }
    metadata = {
        "model": {
            "type": "spine_compression",
            "axis": axis,
            "pmma_thickness_voxels": thickness,
            "labels": {"body": body_label, "process": process_label},
            "displacement": displacement,
        },
        "materials": {"pmma": pmma, "poisson_ratio": poisson_ratio},
    }
    exported = export_model_artifacts(
        material_xyz=material_xyz,
        labels_xyz=labels_xyz,
        spacing=spacing,
        origin=_padded_origin(origin, spacing, axis_index, thickness),
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
        origin=_padded_origin(origin, spacing, axis_index, thickness),
        poisson_ratio=poisson_ratio,
        boundary_conditions=boundary_conditions,
        node_sets=node_sets,
        element_sets=element_sets,
        exported=exported,
        metadata=metadata,
    )


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
        mm = float(geometry.get("pmma_thickness_mm", 3.0))
        value = int(round(mm / spacing[AXIS_TO_INDEX[axis]]))
    if value < 1:
        raise ValueError("PMMA disk thickness rounds to zero voxels")
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
