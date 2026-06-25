from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from parosol_py.nodesets import nodes_from_mask_face

from .alignment import align_spine_body_to_reference
from .common import (
    AXIS_TO_INDEX,
    constrained_contact_bcs,
    displacement_from_load_case,
    export_model_artifacts,
    load_density_and_mask,
    material_from_density,
    nodes_for_labels,
    occupied_length_mm,
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
    preprocessing_config: dict[str, Any] | None = None,
) -> BuiltModel:
    density_zyx, mask_zyx, spacing, origin = load_density_and_mask(
        model_config, base_dir=base_dir, preprocessing_config=preprocessing_config
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

    registration_metadata: dict[str, Any] = {"enabled": False}
    registration_cfg = model_config.get("registration", {})
    if registration_cfg:
        aligned = align_spine_body_to_reference(
            density_zyx=density_zyx,
            body_mask_zyx=body_mask_zyx,
            process_mask_zyx=process_mask_zyx,
            spacing=spacing,
            origin=origin,
            registration_config=registration_cfg,
            base_dir=base_dir,
        )
        density_zyx = aligned.density_zyx
        body_mask_zyx = aligned.body_mask_zyx
        process_mask_zyx = aligned.process_mask_zyx
        spacing = aligned.spacing
        origin = aligned.origin
        registration_metadata = aligned.metadata

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
    label_masks = {
        body_label: body_padded,
        process_label: process_padded,
    }
    disk_target_name, disk_target_mask = _disk_target_mask(
        model_config,
        body_padded=body_padded,
        process_padded=process_padded,
        label_masks=label_masks,
    )
    intrusion_depth = _intrusion_depth_voxels(
        model_config, spacing=spacing, axis=axis, default=thickness
    )
    inferior, superior = projected_caps_from_mask(
        disk_target_mask,
        axis=axis,
        thickness_voxels=thickness,
        intrusion_depth_voxels=intrusion_depth,
        shape=_disk_shape(model_config),
    )
    material_xyz[inferior | superior] = pmma["E"]

    labels_xyz = np.zeros(material_xyz.shape, dtype=np.uint8)
    labels_xyz[body_padded] = 1
    labels_xyz[process_padded] = 2
    labels_xyz[inferior] = 10
    labels_xyz[superior] = 11

    node_sets = {
        "inferior": nodes_from_mask_face(inferior, axis=axis, side=-1),
        "superior": nodes_from_mask_face(superior, axis=axis, side=1),
    }
    require_non_empty(node_sets)
    displacement = displacement_from_load_case(
        load_case_config,
        axis=axis,
        dimensions_xyz=tuple(int(v) for v in material_xyz.shape),
        spacing=spacing,
        default=-0.01,
        length_mm=occupied_length_mm(material_xyz, axis=axis, spacing=spacing),
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
            "load_axis": axis,
            "load_direction": axis,
            "pmma_thickness_voxels": thickness,
            "disk": {
                "target_label": disk_target_name,
                "shape": _disk_shape(model_config),
                "thickness_voxels": thickness,
                "intrusion_depth_voxels": intrusion_depth,
                "method": "projected_cap",
            },
            "labels": {"body": body_label, "process": process_label},
            "displacement": displacement,
            "registration": registration_metadata,
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
        postprocess_mask=to_zyx(body_padded | process_padded),
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
    disk = geometry.get("disk", {})
    if "thickness_voxels" in disk:
        value = int(disk["thickness_voxels"])
    elif "thickness_mm" in disk:
        mm = float(disk["thickness_mm"])
        value = int(round(mm / spacing[AXIS_TO_INDEX[axis]]))
    elif "pmma_thickness_voxels" in geometry:
        value = int(geometry["pmma_thickness_voxels"])
    else:
        mm = float(geometry.get("pmma_thickness_mm", 3.0))
        value = int(round(mm / spacing[AXIS_TO_INDEX[axis]]))
    if value < 1:
        raise ValueError("PMMA disk thickness rounds to zero voxels")
    return value


def _intrusion_depth_voxels(
    model_config: dict[str, Any],
    *,
    spacing: tuple[float, float, float],
    axis: str,
    default: int,
) -> int:
    geometry = model_config.get("geometry", {})
    disk = geometry.get("disk", {})
    if "intrusion_depth_voxels" in disk:
        value = int(disk["intrusion_depth_voxels"])
    elif "intrusion_depth_mm" in disk:
        value = int(
            round(float(disk["intrusion_depth_mm"]) / spacing[AXIS_TO_INDEX[axis]])
        )
    elif "intrusion_depth_voxels" in geometry:
        value = int(geometry["intrusion_depth_voxels"])
    elif "intrusion_depth_mm" in geometry:
        value = int(
            round(float(geometry["intrusion_depth_mm"]) / spacing[AXIS_TO_INDEX[axis]])
        )
    elif "endplate_depth_voxels" in geometry:
        value = int(geometry["endplate_depth_voxels"])
    elif "endplate_depth_mm" in geometry:
        value = int(
            round(float(geometry["endplate_depth_mm"]) / spacing[AXIS_TO_INDEX[axis]])
        )
    elif "surface_depth_voxels" in geometry:
        value = int(geometry["surface_depth_voxels"])
    elif "surface_depth_mm" in geometry:
        value = int(round(float(geometry["surface_depth_mm"]) / spacing[AXIS_TO_INDEX[axis]]))
    else:
        value = int(round(float(default) * 2.5))
    if value < 1:
        raise ValueError("disk intrusion depth rounds to zero voxels")
    return value


def _disk_target_mask(
    model_config: dict[str, Any],
    *,
    body_padded: np.ndarray,
    process_padded: np.ndarray,
    label_masks: dict[int, np.ndarray],
) -> tuple[str, np.ndarray]:
    geometry = model_config.get("geometry", {})
    disk = geometry.get("disk", {})
    raw_target = disk.get("target_label", disk.get("target", geometry.get("disk_target", "body")))
    if isinstance(raw_target, int | float) or str(raw_target).strip().lstrip("-").isdigit():
        label = int(raw_target)
        if label not in label_masks:
            raise ValueError(f"model.geometry.disk.target_label {label} is absent")
        return str(label), label_masks[label]
    target = str(raw_target).strip().lower()
    if target in {"body", "vertebral_body", "label_body"}:
        return "body", body_padded
    if target in {"process", "posterior_elements", "vertebral_process"}:
        return "process", process_padded
    if target in {"bone", "all_bone", "body_process", "active"}:
        return "bone", body_padded | process_padded
    raise ValueError(
        "model.geometry.disk.target_label must be one of body, process, or bone"
    )


def _disk_shape(model_config: dict[str, Any]) -> str:
    geometry = model_config.get("geometry", {})
    disk = geometry.get("disk", {})
    return str(disk.get("shape", geometry.get("disk_shape", "anatomy"))).strip().lower()


def _padded_origin(
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    axis_index: int,
    thickness: int,
) -> tuple[float, float, float]:
    out = list(origin)
    out[axis_index] -= float(thickness) * float(spacing[axis_index])
    return tuple(out)
