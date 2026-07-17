from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from parosol_py.nodesets import boundary_conditions_from_nodesets, nodes_from_labeled_voxels
from parosol_py.workflow_geometry import (
    generate_disk_and_nodeset_geometry,
    resolve_reference_space_editor,
    scale_reference_points_preserving_pose as _scale_reference_points_preserving_pose,
    scale_reference_space_editor as _scale_reference_space_editor,
)

from .alignment import (
    _output_grid_for_transform,
    _reference_points_from_config,
    _resample_with_transform,
    estimate_rigid_icp,
    surface_points_from_mask,
)
from .common import (
    AXIS_TO_INDEX,
    export_model_artifacts,
    fixture_margin_voxels,
    load_density_and_mask,
    material_from_density,
    nonlinear_material_from_density,
    occupied_length_mm,
    pad_arrays_to_foreground_margin,
    require_non_empty,
    target_mask_from_labels,
    to_zyx,
)
from .io import read_image_zyx, resolve_path
from .types import BuiltModel


GENERATED_BOUNDARY_LABEL_BASE = 10001


@dataclass(slots=True)
class _CroppedWorkflowModel:
    material_xyz: np.ndarray
    labels_xyz: np.ndarray
    node_label_xyz: np.ndarray
    postprocess_mask_xyz: np.ndarray
    origin: tuple[float, float, float]
    node_sets: dict[str, list[tuple[int, int, int]]]
    percent_reference_node_sets: dict[str, list[tuple[int, int, int]]] | None
    crop: dict[str, Any]


@dataclass(slots=True)
class WorkflowReplayPreview:
    density_zyx: np.ndarray
    mask_zyx: np.ndarray
    registration_mask_zyx: np.ndarray
    projection_mask_zyx: np.ndarray
    model_mask_zyx: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    registration_config: dict[str, Any]
    transform: dict[str, Any]
    metadata: dict[str, Any]
    reference_points: np.ndarray | None = None


def build_workflow_replay_preview(
    model_config: dict[str, Any],
    *,
    base_dir: Path,
    preprocessing_config: dict[str, Any] | None = None,
) -> WorkflowReplayPreview:
    """Build the shared pre-boundary-condition replay grid."""

    replay_cfg = model_config.get("workflow_replay", {})
    if not isinstance(replay_cfg, dict) or not replay_cfg.get("enabled", False):
        raise ValueError("workflow replay requires model.workflow_replay.enabled=true")

    density_zyx, mask_zyx, spacing, origin = load_density_and_mask(
        model_config,
        base_dir=base_dir,
        preprocessing_config=preprocessing_config,
    )
    registration_mask_zyx = _workflow_active_mask(mask_zyx, model_config, replay_cfg)
    projection_mask_zyx = _workflow_projection_mask(
        mask_zyx,
        model_config,
        replay_cfg,
        default_mask=registration_mask_zyx,
    )
    model_mask_zyx = _workflow_model_mask(mask_zyx, model_config)
    padded, origin = pad_arrays_to_foreground_margin(
        anchor_mask_zyx=model_mask_zyx,
        spacing=spacing,
        origin=origin,
        margin_voxels=fixture_margin_voxels(
            model_config,
            spacing=spacing,
            default_axis=str(model_config.get("geometry", {}).get("cap_axis", "z")),
        ),
        arrays={
            "density": density_zyx,
            "mask": mask_zyx,
            "registration": registration_mask_zyx,
            "projection": projection_mask_zyx,
            "model": model_mask_zyx,
        },
        constant_values={
            "density": 0.0,
            "mask": 0,
            "registration": False,
            "projection": False,
            "model": False,
        },
    )
    density_zyx = padded["density"]
    mask_zyx = padded["mask"]
    registration_mask_zyx = np.asarray(padded["registration"], dtype=bool)
    projection_mask_zyx = np.asarray(padded["projection"], dtype=bool)
    model_mask_zyx = np.asarray(padded["model"], dtype=bool)

    registration_cfg = _workflow_registration_config(model_config, replay_cfg)
    if _workflow_registration_enabled(registration_cfg):
        registration_meta, transform = _estimate_reference_to_sample_transform(
            registration_mask_zyx,
            spacing=spacing,
            origin=origin,
            registration_config=registration_cfg,
            base_dir=base_dir,
        )
    else:
        registration_meta = {"enabled": False, "applied_to_model_grid": False}
        transform = {
            "rotation": np.eye(3),
            "translation": np.zeros(3),
            "iterations": 0,
            "mean_distance": 0.0,
        }

    model_space = _workflow_model_space(replay_cfg, registration_cfg)
    reference_points = None
    if model_space == "reference":
        aligned = _align_workflow_arrays_to_reference(
            density_zyx=density_zyx,
            mask_zyx=mask_zyx,
            registration_mask_zyx=registration_mask_zyx,
            projection_mask_zyx=projection_mask_zyx,
            model_mask_zyx=model_mask_zyx,
            spacing=spacing,
            origin=origin,
            registration_config=registration_cfg,
            base_dir=base_dir,
        )
        density_zyx = aligned["density"]
        mask_zyx = aligned["mask"]
        registration_mask_zyx = aligned["registration"]
        projection_mask_zyx = aligned["projection"]
        model_mask_zyx = aligned["model"]
        origin = aligned["origin"]
        registration_meta = aligned["metadata"]
        reference_points = aligned.get("reference_points")
        transform = {
            "rotation": np.eye(3),
            "translation": np.zeros(3),
            "iterations": 0,
            "mean_distance": 0.0,
        }

    editor = model_config.get("slicer_editor")
    if isinstance(editor, dict) and isinstance(editor.get("planes"), list) and editor.get("planes"):
        padded_for_planes, origin = _pad_workflow_arrays_for_editor_planes(
            editor=editor,
            density_zyx=density_zyx,
            mask_zyx=mask_zyx,
            registration_mask_zyx=registration_mask_zyx,
            projection_mask_zyx=projection_mask_zyx,
            model_mask_zyx=model_mask_zyx,
            spacing=spacing,
            origin=origin,
        )
        density_zyx = padded_for_planes["density"]
        mask_zyx = padded_for_planes["mask"]
        registration_mask_zyx = np.asarray(padded_for_planes["registration"], dtype=bool)
        projection_mask_zyx = np.asarray(padded_for_planes["projection"], dtype=bool)
        model_mask_zyx = np.asarray(padded_for_planes["model"], dtype=bool)

    return WorkflowReplayPreview(
        density_zyx=np.asarray(density_zyx, dtype=np.float64),
        mask_zyx=np.asarray(mask_zyx),
        registration_mask_zyx=np.asarray(registration_mask_zyx, dtype=bool),
        projection_mask_zyx=np.asarray(projection_mask_zyx, dtype=bool),
        model_mask_zyx=np.asarray(model_mask_zyx, dtype=bool),
        spacing=tuple(float(value) for value in spacing),
        origin=tuple(float(value) for value in origin),
        registration_config=registration_cfg,
        transform=transform,
        metadata={
            "model_space": model_space,
            "registration": registration_meta,
        },
        reference_points=reference_points,
    )


def build_workflow_replay_model(
    model_config: dict[str, Any],
    *,
    base_dir: Path,
    material_config: dict[str, Any],
    load_case_config: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    nodeset_config: dict[str, Any] | None = None,
) -> BuiltModel:
    preview = build_workflow_replay_preview(
        model_config,
        base_dir=base_dir,
        preprocessing_config=preprocessing_config,
    )
    replay_cfg = model_config.get("workflow_replay", {})
    density_zyx = preview.density_zyx
    projection_mask_zyx = preview.projection_mask_zyx
    model_mask_zyx = preview.model_mask_zyx
    spacing = preview.spacing
    origin = preview.origin
    registration_cfg = preview.registration_config
    registration_meta = dict(preview.metadata.get("registration", {}))
    model_space = str(preview.metadata.get("model_space", "sample"))
    transform = preview.transform
    editor = model_config.get("slicer_editor")

    bone_mpa_zyx, poisson_ratio = material_from_density(
        density_zyx,
        model_mask_zyx,
        material_config=material_config,
    )
    nonlinear_material = nonlinear_material_from_density(
        density_zyx,
        model_mask_zyx,
        material_config=material_config,
        poisson_ratio=poisson_ratio,
    )
    if nonlinear_material is not None:
        bone_mpa_zyx = nonlinear_material.youngs_modulus_mpa
    material_xyz = np.transpose(bone_mpa_zyx, (2, 1, 0))
    labels_xyz = np.zeros(material_xyz.shape, dtype=np.uint16)

    mask_xyz = np.transpose(model_mask_zyx, (2, 1, 0))
    labels_xyz[mask_xyz] = 1
    base_labels_xyz = labels_xyz.copy()

    geometry_mode = "cached_labelmaps"
    resolved_editor: dict[str, Any] | None = None
    generated_node_sets: dict[str, list[tuple[int, int, int]]] | None = None
    generated_reference_node_sets: dict[str, list[tuple[int, int, int]]] | None = None
    if isinstance(editor, dict) and isinstance(editor.get("planes"), list) and editor.get("planes"):
        disk_xyz, node_label_xyz, node_sets, resolved_editor = _workflow_geometry_from_editor(
            editor=editor,
            active_mask_zyx=projection_mask_zyx,
            material_xyz=material_xyz,
            spacing=spacing,
            origin=origin,
            registration_config=registration_cfg,
            transform=transform,
            base_dir=base_dir,
            nodeset_config=nodeset_config or {},
            replay_cfg=replay_cfg,
        )
        disk_labels_zyx = np.transpose(np.asarray(disk_xyz, dtype=np.uint16), (2, 1, 0))
        nodeset_labels_zyx = np.transpose(np.asarray(node_label_xyz, dtype=np.uint16), (2, 1, 0))
        geometry_mode = "plane_driven"
        generated_node_sets = node_sets
        generated_reference_node_sets = _workflow_percent_reference_node_sets(
            node_label_xyz,
            material_xyz=material_xyz,
            nodeset_config=nodeset_config or {},
        )
        registration_meta["resolved_editor_plane_count"] = int(len(resolved_editor.get("planes", [])))
    else:
        disk_labels_zyx, nodeset_labels_zyx = _load_resampled_replay_labels(
            replay_cfg=replay_cfg,
            density_zyx=density_zyx,
            spacing=spacing,
            origin=origin,
            base_dir=base_dir,
            transform=transform,
        )
        node_label_xyz = np.transpose(np.asarray(nodeset_labels_zyx, dtype=np.uint16), (2, 1, 0))
    disk_xyz = np.transpose(np.asarray(disk_labels_zyx, dtype=np.uint16), (2, 1, 0))
    if np.any(disk_xyz > 0):
        pmma = material_config.get("pmma", {})
        pmma_e = float(pmma.get("E", 2500.0))
        pmma_nu = float(pmma.get("nu", 0.3))
        material_xyz[disk_xyz > 0] = pmma_e
        nonlinear_material = _assign_pmma_disks_to_nonlinear_material(
            nonlinear_material,
            disk_mask_xyz=disk_xyz > 0,
            pmma_e_mpa=pmma_e,
            pmma_nu=pmma_nu,
        )
        labels_xyz[disk_xyz > 0] = np.maximum(labels_xyz[disk_xyz > 0], disk_xyz[disk_xyz > 0])

    labels_xyz[node_label_xyz > 0] = np.maximum(
        labels_xyz[node_label_xyz > 0],
        node_label_xyz[node_label_xyz > 0],
    )
    artifact_labels_xyz = labels_xyz.copy()
    for label in np.unique(node_label_xyz):
        label_int = int(label)
        if label_int == 0:
            continue
        nodeset_mask = node_label_xyz == label_int
        colliding_disk_mask = (artifact_labels_xyz == label_int) & ~nodeset_mask
        artifact_labels_xyz[colliding_disk_mask] = base_labels_xyz[colliding_disk_mask]
    artifact_labels_xyz[node_label_xyz > 0] = node_label_xyz[node_label_xyz > 0]

    node_sets = (
        generated_node_sets
        if generated_node_sets is not None
        else _workflow_node_sets(
            node_label_xyz,
            material_xyz=material_xyz,
            nodeset_config=nodeset_config or {},
        )
    )
    cropped = _crop_workflow_model_to_material_bbox(
        material_xyz=material_xyz,
        labels_xyz=artifact_labels_xyz,
        node_label_xyz=node_label_xyz,
        postprocess_mask_xyz=mask_xyz,
        spacing=spacing,
        origin=origin,
        node_sets=node_sets,
        percent_reference_node_sets=generated_reference_node_sets,
    )
    material_xyz = cropped.material_xyz
    artifact_labels_xyz = cropped.labels_xyz
    node_label_xyz = cropped.node_label_xyz
    postprocess_mask_xyz = cropped.postprocess_mask_xyz
    origin = cropped.origin
    node_sets = cropped.node_sets
    generated_reference_node_sets = cropped.percent_reference_node_sets
    disk_xyz = _crop_array_to_workflow_crop(disk_xyz, cropped.crop)
    nonlinear_material = _crop_nonlinear_material_to_workflow_crop(
        nonlinear_material,
        cropped.crop,
        material_xyz=material_xyz,
    )

    require_non_empty(node_sets)
    effective_load_case_config = _workflow_effective_load_case_config(
        load_case_config,
        resolved_editor=resolved_editor,
    )
    boundary_conditions = boundary_conditions_from_nodesets(
        node_sets,
        fixed=list(effective_load_case_config.get("fixed", ())),
        prescribed=list(effective_load_case_config.get("prescribed", ())),
        loaded=list(effective_load_case_config.get("loaded", ())),
        dimensions_xyz=tuple(int(v) for v in material_xyz.shape),
        spacing=spacing,
        percent_reference_lengths_mm={
            axis: occupied_length_mm(material_xyz, axis=axis, spacing=spacing)
            for axis in ("x", "y", "z")
        },
        percent_reference_node_sets=generated_reference_node_sets,
    )
    element_sets = {
        "bone": int(np.count_nonzero(model_mask_zyx)),
        "workflow_disks": int(np.count_nonzero(disk_labels_zyx)),
    }
    for label in np.unique(disk_xyz):
        if int(label) == 0:
            continue
        element_sets[f"disk_label_{int(label)}"] = int(np.count_nonzero(disk_xyz == int(label)))

    metadata = {
        "model": {
            "type": str(model_config.get("type", "workflow_replay")),
            "load_axis": _workflow_load_axis(effective_load_case_config),
            "load_direction": _workflow_load_axis(effective_load_case_config),
            "effective_load_case": effective_load_case_config,
            "workflow_replay": {
                "enabled": True,
                "geometry_mode": geometry_mode,
                "model_space": model_space,
                "resolved_planes": _summarize_resolved_planes(resolved_editor),
                "resolved_editor": _json_safe_editor(resolved_editor),
                "disk_labels": str(resolve_path(replay_cfg["disk_labels"], base_dir=base_dir))
                if replay_cfg.get("disk_labels")
                else None,
                "nodesets": str(resolve_path(replay_cfg["nodesets"], base_dir=base_dir))
                if replay_cfg.get("nodesets")
                else None,
                "final_material_crop": cropped.crop,
            },
            "registration": registration_meta,
        },
        "materials": {
            "poisson_ratio": poisson_ratio,
            "pmma": material_config.get("pmma", {"E": 2500.0, "nu": 0.3}),
        },
    }
    exported = export_model_artifacts(
        material_xyz=material_xyz,
        labels_xyz=artifact_labels_xyz,
        nodeset_labels_xyz=node_label_xyz,
        disk_labels_xyz=disk_xyz,
        spacing=spacing,
        origin=origin,
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
        origin=origin,
        poisson_ratio=poisson_ratio,
        boundary_conditions=boundary_conditions,
        node_sets=node_sets,
        element_sets=element_sets,
        postprocess_mask=np.asarray(to_zyx(postprocess_mask_xyz), dtype=bool),
        nonlinear_material=nonlinear_material,
        exported=exported,
        metadata=metadata,
    )


def _workflow_geometry_from_editor(
    *,
    editor: dict[str, Any],
    active_mask_zyx: np.ndarray,
    material_xyz: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    registration_config: dict[str, Any],
    transform: dict[str, Any],
    base_dir: Path,
    nodeset_config: dict[str, Any],
    replay_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[int, int, int]]], dict[str, Any]]:
    max_points = int(registration_config.get("max_points", 8000))
    use_editor_reference = bool(replay_cfg.get("editor_reference_points"))
    if not _workflow_registration_enabled(registration_config):
        resolved_editor = _reference_space_editor_without_sample_transform(editor)
        resolved_editor = _resolve_bbox_relative_editor(
            resolved_editor,
            model_mask_zyx=active_mask_zyx,
            spacing=spacing,
            origin=origin,
        )
        nodeset_specs, nodeset_labels, nodeset_names = _editor_nodeset_maps(
            resolved_editor=resolved_editor,
            nodeset_config=nodeset_config,
        )
        disk_labels = _editor_disk_labels(
            resolved_editor=resolved_editor,
            replay_cfg=replay_cfg,
            base_dir=base_dir,
        )
        geometry = generate_disk_and_nodeset_geometry(
            resolved_editor,
            mask_xyz=np.transpose(np.asarray(active_mask_zyx, dtype=bool), (2, 1, 0)),
            material_xyz=material_xyz,
            spacing=spacing,
            origin=origin,
            nodeset_specs=nodeset_specs,
            nodeset_labels=nodeset_labels,
            nodeset_names=nodeset_names,
            disk_labels=disk_labels,
        )
        return (
            geometry.disk_labels_xyz,
            geometry.nodeset_labels_xyz,
            geometry.node_sets,
            resolved_editor,
        )
    reference_path = resolve_path(
        replay_cfg.get("editor_reference_points", registration_config["reference_points"]),
        base_dir=base_dir,
    )
    reference_points = _reference_points_from_config(
        reference_path,
        registration_config=registration_config,
        max_points=max_points,
    )
    sample_points = surface_points_from_mask(
        active_mask_zyx,
        spacing=spacing,
        origin=origin,
        max_points=max_points,
        sample_mode=str(
            registration_config.get(
                "source_landmark_mode",
                registration_config.get("landmark_mode", "linspace"),
            )
        ),
        sample_offset=int(registration_config.get("source_landmark_offset", 0)),
    )
    if _workflow_model_space(replay_cfg, registration_config) == "reference":
        reference_points, scaling_meta = _scale_reference_points_preserving_pose(
            reference_points=reference_points,
            sample_points=sample_points,
            registration_config=registration_config,
        )
    else:
        if not use_editor_reference:
            reference_points, scaling_meta = _maybe_scale_reference_points(
                reference_points=reference_points,
                sample_points=sample_points,
                registration_config=registration_config,
            )
        else:
            scaling_meta = {"enabled": False}
    if _workflow_model_space(replay_cfg, registration_config) == "reference":
        resolved_editor = _reference_space_editor_without_sample_transform(editor)
        resolved_editor = _scale_reference_space_editor(
            resolved_editor,
            scaling_meta=scaling_meta,
        )
        resolved_editor = _resolve_bbox_relative_editor(
            resolved_editor,
            model_mask_zyx=active_mask_zyx,
            spacing=spacing,
            origin=origin,
        )
        nodeset_specs, nodeset_labels, nodeset_names = _editor_nodeset_maps(
            resolved_editor=resolved_editor,
            nodeset_config=nodeset_config,
        )
        disk_labels = _editor_disk_labels(
            resolved_editor=resolved_editor,
            replay_cfg=replay_cfg,
            base_dir=base_dir,
        )
        geometry = generate_disk_and_nodeset_geometry(
            resolved_editor,
            mask_xyz=np.transpose(np.asarray(active_mask_zyx, dtype=bool), (2, 1, 0)),
            material_xyz=material_xyz,
            spacing=spacing,
            origin=origin,
            nodeset_specs=nodeset_specs,
            nodeset_labels=nodeset_labels,
            nodeset_names=nodeset_names,
            disk_labels=disk_labels,
        )
        return (
            geometry.disk_labels_xyz,
            geometry.nodeset_labels_xyz,
            geometry.node_sets,
            resolved_editor,
        )
    resolved_editor = resolve_reference_space_editor(
        editor,
        reference_points=reference_points,
        sample_points=sample_points,
        iterations=int(registration_config.get("iterations", 50)),
        tolerance=float(registration_config.get("tolerance", 1.0e-4)),
        allow_scale=False,
        snap_planes=False,
        prealign_reference=False
        if use_editor_reference
        else bool(registration_config.get("prealign_reference_to_sample", False)),
        transform_override=None if use_editor_reference else transform,
    )
    resolved_editor = _resolve_bbox_relative_editor(
        resolved_editor,
        model_mask_zyx=active_mask_zyx,
        spacing=spacing,
        origin=origin,
    )
    nodeset_specs, nodeset_labels, nodeset_names = _editor_nodeset_maps(
        resolved_editor=resolved_editor,
        nodeset_config=nodeset_config,
    )
    disk_labels = _editor_disk_labels(
        resolved_editor=resolved_editor,
        replay_cfg=replay_cfg,
        base_dir=base_dir,
    )
    geometry = generate_disk_and_nodeset_geometry(
        resolved_editor,
        mask_xyz=np.transpose(np.asarray(active_mask_zyx, dtype=bool), (2, 1, 0)),
        material_xyz=material_xyz,
        spacing=spacing,
        origin=origin,
        nodeset_specs=nodeset_specs,
        nodeset_labels=nodeset_labels,
        nodeset_names=nodeset_names,
        disk_labels=disk_labels,
    )
    return (
        geometry.disk_labels_xyz,
        geometry.nodeset_labels_xyz,
        geometry.node_sets,
        resolved_editor,
    )


def _load_resampled_replay_labels(
    *,
    replay_cfg: dict[str, Any],
    density_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    base_dir: Path,
    transform: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    disk_labels_zyx, disk_spacing, disk_origin = read_image_zyx(
        resolve_path(replay_cfg["disk_labels"], base_dir=base_dir)
    )
    nodeset_labels_zyx, nodeset_spacing, nodeset_origin = read_image_zyx(
        resolve_path(replay_cfg["nodesets"], base_dir=base_dir)
    )
    output_size_xyz = tuple(int(v) for v in density_zyx.shape[::-1])
    disk_labels_zyx = _resample_with_transform(
        np.asarray(disk_labels_zyx, dtype=np.uint16),
        spacing=disk_spacing,
        origin=disk_origin,
        output_spacing=spacing,
        output_origin=origin,
        output_size=output_size_xyz,
        rotation=transform["rotation"],
        translation=transform["translation"],
        interpolation="nearest",
    )
    nodeset_labels_zyx = _resample_with_transform(
        np.asarray(nodeset_labels_zyx, dtype=np.uint16),
        spacing=nodeset_spacing,
        origin=nodeset_origin,
        output_spacing=spacing,
        output_origin=origin,
        output_size=output_size_xyz,
        rotation=transform["rotation"],
        translation=transform["translation"],
        interpolation="nearest",
    )
    return disk_labels_zyx, nodeset_labels_zyx


def _editor_nodeset_maps(
    *,
    resolved_editor: dict[str, Any],
    nodeset_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, str]]:
    specs: dict[str, dict[str, Any]] = {}
    labels: dict[str, int] = {}
    plane_to_nodeset: dict[str, str] = {}
    for plane in resolved_editor.get("planes", []):
        if not isinstance(plane, dict):
            continue
        plane_name = str(plane.get("name", "plane")).strip()
        nodeset_name = _matching_nodeset_name(plane_name, nodeset_config)
        if nodeset_name is None:
            nodeset_name = _plane_to_nodeset_name(plane_name)
        plane_to_nodeset[plane_name] = nodeset_name
        spec = nodeset_config.get(nodeset_name, {})
        if isinstance(spec, dict):
            specs[nodeset_name] = spec
            if "label" in spec:
                labels[nodeset_name] = int(spec["label"])
        elif nodeset_name not in specs:
            specs[nodeset_name] = {"selection": "surface_nodes"}
    return specs, labels, plane_to_nodeset


def _editor_disk_labels(
    *,
    resolved_editor: dict[str, Any],
    replay_cfg: dict[str, Any],
    base_dir: Path,
) -> dict[str, int]:
    material_planes = [
        plane
        for plane in resolved_editor.get("planes", [])
        if isinstance(plane, dict)
        and str(plane.get("contact", "Material disks")).strip().lower()
        in {"material disks", "pmma caps", "connective disk"}
    ]
    preserved_labels: list[int] = []
    disk_path = replay_cfg.get("disk_labels")
    if isinstance(disk_path, str) and disk_path:
        try:
            reference_disk_labels, _spacing, _origin = read_image_zyx(
                resolve_path(disk_path, base_dir=base_dir)
            )
            preserved_labels = [
                int(value)
                for value in np.unique(np.asarray(reference_disk_labels, dtype=np.uint16))
                if int(value) > 0
            ]
        except Exception:
            preserved_labels = []
    labels: dict[str, int] = {}
    for index, plane in enumerate(material_planes):
        name = str(plane.get("name", f"disk_{index + 1}")).strip()
        explicit_label = _positive_int_or_none(plane.get("disk_label"))
        if explicit_label is not None:
            label = explicit_label
        elif index < len(preserved_labels):
            label = preserved_labels[index]
        else:
            label = GENERATED_BOUNDARY_LABEL_BASE + index
        labels[name] = int(label)
    return labels


def _positive_int_or_none(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _matching_nodeset_name(plane_name: str, nodeset_config: dict[str, Any]) -> str | None:
    slug = _plane_to_nodeset_name(plane_name)
    if slug in nodeset_config:
        return slug
    lowered = plane_name.strip().lower()
    for candidate in nodeset_config:
        if str(candidate).strip().lower().replace("_", " ") == lowered:
            return str(candidate)
    return None


def _plane_to_nodeset_name(name: str) -> str:
    text = str(name).strip().lower()
    return "_".join(part for part in text.replace("-", " ").split() if part)


def _workflow_node_sets(
    labels_xyz: np.ndarray,
    *,
    material_xyz: np.ndarray,
    nodeset_config: dict[str, Any],
) -> dict[str, list[tuple[int, int, int]]]:
    node_sets: dict[str, list[tuple[int, int, int]]] = {}
    for name, spec in nodeset_config.items():
        if not isinstance(spec, dict):
            continue
        label = int(spec["label"])
        selection = str(spec.get("selection", "surface_nodes")).strip().lower()
        node_sets[str(name)] = nodes_from_labeled_voxels(
            labels_xyz,
            label=label,
            selection=selection,
            material=material_xyz,
        )
    return node_sets


def _workflow_percent_reference_node_sets(
    labels_xyz: np.ndarray,
    *,
    material_xyz: np.ndarray,
    nodeset_config: dict[str, Any],
) -> dict[str, list[tuple[int, int, int]]] | None:
    reference_sets: dict[str, list[tuple[int, int, int]]] = {}
    for name, spec in nodeset_config.items():
        if not isinstance(spec, dict):
            continue
        reference_selection = spec.get("percent_reference_selection")
        if reference_selection is None:
            continue
        reference_sets[str(name)] = nodes_from_labeled_voxels(
            labels_xyz,
            label=int(spec["label"]),
            selection=str(reference_selection).strip().lower(),
            material=material_xyz,
        )
    return reference_sets or None


def _workflow_active_mask(
    mask_zyx: np.ndarray,
    model_config: dict[str, Any],
    replay_cfg: dict[str, Any],
) -> np.ndarray:
    targets = model_config.get("targets", {})
    registration = model_config.get("registration", {})
    selector = _first_present(
        replay_cfg,
        ("registration_labels", "registration_target"),
        targets if isinstance(targets, dict) else {},
        ("registration", "registration_labels"),
        registration if isinstance(registration, dict) else {},
        ("target_labels", "target"),
    )
    return _workflow_target_mask(
        mask_zyx,
        model_config,
        selector,
        default="first_declared_label",
    )


def _workflow_projection_mask(
    mask_zyx: np.ndarray,
    model_config: dict[str, Any],
    replay_cfg: dict[str, Any],
    *,
    default_mask: np.ndarray,
) -> np.ndarray:
    targets = model_config.get("targets", {})
    selector = _first_present(
        replay_cfg,
        ("projection_labels", "disk_projection_labels", "disk_projection"),
        targets if isinstance(targets, dict) else {},
        ("disk_projection", "projection", "projection_labels"),
    )
    if selector is None:
        return np.asarray(default_mask, dtype=bool)
    return _workflow_target_mask(mask_zyx, model_config, selector, default="nonzero")


def _workflow_model_mask(mask_zyx: np.ndarray, model_config: dict[str, Any]) -> np.ndarray:
    targets = model_config.get("targets", {})
    selector = _first_present(
        targets if isinstance(targets, dict) else {},
        ("model", "model_labels"),
    )
    return _workflow_target_mask(
        mask_zyx,
        model_config,
        selector,
        default="all_declared_labels",
    )


def _first_present(*pairs: Any) -> Any:
    for mapping, keys in zip(pairs[0::2], pairs[1::2], strict=False):
        if not isinstance(mapping, dict):
            continue
        for key in keys:
            if key in mapping and mapping[key] not in (None, ""):
                return mapping[key]
    return None


def _workflow_target_mask(
    mask_zyx: np.ndarray,
    model_config: dict[str, Any],
    selector: Any,
    *,
    default: str,
) -> np.ndarray:
    mask = np.asarray(mask_zyx)
    labels = _workflow_label_values(model_config)
    if selector is None:
        if default == "all_declared_labels" and labels:
            return target_mask_from_labels(
                mask,
                list(labels.values()),
                context="workflow target labels",
            )
        if default == "first_declared_label" and labels:
            return target_mask_from_labels(
                mask,
                [next(iter(labels.values()))],
                context="workflow target labels",
            )
        return mask > 0

    values = _workflow_selector_values(selector, labels)
    if values is None:
        return mask > 0
    if not values:
        raise ValueError("workflow target selector did not resolve to any labels")
    return target_mask_from_labels(
        mask,
        values,
        context="workflow target labels",
    )


def _workflow_label_values(model_config: dict[str, Any]) -> dict[str, int]:
    raw = model_config.get("labels", {})
    if not isinstance(raw, dict):
        return {}
    return {str(key): int(value) for key, value in raw.items()}


def _workflow_selector_values(selector: Any, labels: dict[str, int]) -> list[int] | None:
    if isinstance(selector, dict):
        mode = str(selector.get("mode", "")).strip().lower()
        if mode in {"all", "mask", "full_mask", "nonzero", "*"}:
            return None
        for key in ("labels", "label_keys", "keys", "values", "label_values"):
            if key in selector:
                return _workflow_selector_values(selector[key], labels)
        raise ValueError(
            "workflow target selector objects must define labels, label_keys, values, or mode"
        )
    if isinstance(selector, (list, tuple, set)):
        values: list[int] = []
        for item in selector:
            item_values = _workflow_selector_values(item, labels)
            if item_values is None:
                return None
            values.extend(item_values)
        return sorted(set(values))
    if isinstance(selector, (int, np.integer)):
        return [int(selector)]
    token = str(selector).strip()
    lowered = token.lower()
    if lowered in {"all", "mask", "full_mask", "nonzero", "*"}:
        return None
    if "," in token:
        return _workflow_selector_values(
            [part.strip() for part in token.split(",") if part.strip()],
            labels,
        )
    if token in labels:
        return [int(labels[token])]
    try:
        return [int(token)]
    except ValueError as exc:
        expected = ", ".join(sorted(labels)) or "numeric labels"
        raise ValueError(
            f"workflow target selector {selector!r} is not a declared label key; "
            f"expected one of: {expected}"
        ) from exc


def _workflow_model_space(
    replay_cfg: dict[str, Any], registration_config: dict[str, Any]
) -> str:
    default = "reference" if _workflow_registration_enabled(registration_config) else "sample"
    value = str(replay_cfg.get("model_space", default)).strip().lower()
    if value in {"reference", "reference_frame", "workflow", "workflow_reference"}:
        return "reference"
    if value in {"sample", "sample_space", "native"}:
        return "sample"
    raise ValueError("model.workflow_replay.model_space must be 'reference' or 'sample'")


def _reference_space_editor_without_sample_transform(editor: dict[str, Any]) -> dict[str, Any]:
    import copy

    resolved = copy.deepcopy(editor)
    planes = resolved.get("planes", [])
    if not isinstance(planes, list):
        return resolved
    for plane in planes:
        if isinstance(plane, dict) and plane.get("reference_space", False):
            plane["reference_space"] = False
            plane["resolved_from_reference_space"] = True
    resolved["registration"] = {"reference_space_replayed": True, "model_space": "reference"}
    return resolved


def _resolve_bbox_relative_editor(
    editor: dict[str, Any],
    *,
    model_mask_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> dict[str, Any]:
    resolved = {
        key: value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value
        for key, value in editor.items()
    }
    planes = resolved.get("planes", [])
    if not isinstance(planes, list):
        return resolved
    resolved_planes = []
    for plane in planes:
        if not isinstance(plane, dict):
            resolved_planes.append(plane)
            continue
        relative_to = str(plane.get("relative_to", "")).strip().lower()
        if relative_to not in {"model_bbox", "active_bbox", "image_bbox"}:
            resolved_planes.append(dict(plane))
            continue
        resolved_planes.append(
            _resolve_bbox_relative_plane(
                plane,
                model_mask_zyx=model_mask_zyx,
                spacing=spacing,
                origin=origin,
                relative_to=relative_to,
            )
        )
    resolved["planes"] = resolved_planes
    return resolved


def _pad_workflow_arrays_for_editor_planes(
    *,
    editor: dict[str, Any],
    density_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    registration_mask_zyx: np.ndarray,
    projection_mask_zyx: np.ndarray,
    model_mask_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> tuple[dict[str, np.ndarray], tuple[float, float, float]]:
    resolved_editor = _resolve_bbox_relative_editor(
        editor,
        model_mask_zyx=model_mask_zyx,
        spacing=spacing,
        origin=origin,
    )
    desired_bounds = _editor_plane_required_bounds(
        resolved_editor,
        model_mask_zyx=model_mask_zyx,
        spacing=spacing,
        origin=origin,
    )
    if desired_bounds is None:
        return {
            "density": np.asarray(density_zyx),
            "mask": np.asarray(mask_zyx),
            "registration": np.asarray(registration_mask_zyx),
            "projection": np.asarray(projection_mask_zyx),
            "model": np.asarray(model_mask_zyx),
        }, origin
    desired_min, desired_max = desired_bounds
    return _pad_arrays_to_xyz_bounds(
        arrays={
            "density": density_zyx,
            "mask": mask_zyx,
            "registration": registration_mask_zyx,
            "projection": projection_mask_zyx,
            "model": model_mask_zyx,
        },
        spacing=spacing,
        origin=origin,
        desired_min_xyz=desired_min,
        desired_max_xyz=desired_max,
        constant_values={
            "density": 0.0,
            "mask": 0,
            "registration": False,
            "projection": False,
            "model": False,
        },
    )


def _editor_plane_required_bounds(
    editor: dict[str, Any],
    *,
    model_mask_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    planes = editor.get("planes", []) if isinstance(editor, dict) else []
    if not isinstance(planes, list):
        return None
    active_idx_zyx = np.argwhere(np.asarray(model_mask_zyx, dtype=bool))
    if active_idx_zyx.size == 0:
        return None
    active_points = _indices_zyx_to_xyz_points(
        active_idx_zyx,
        spacing=spacing,
        origin=origin,
    )
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for plane in planes:
        if not isinstance(plane, dict):
            continue
        if str(plane.get("contact", "Material disks")).strip().lower() in {
            "bone surface",
            "nodeset",
            "nodesets",
        }:
            continue
        bounds = _plane_disk_required_bounds(
            plane,
            active_points=active_points,
            spacing=spacing,
        )
        if bounds is None:
            continue
        mins.append(bounds[0])
        maxs.append(bounds[1])
    if not mins:
        return None
    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def _plane_disk_required_bounds(
    plane: dict[str, Any],
    *,
    active_points: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    center = np.asarray(plane.get("center_ras"), dtype=float)
    if center.shape != (3,):
        return None
    normal = _unit_vector(plane.get("normal_ras", [0.0, 0.0, 1.0]))
    u_axis = _unit_vector(plane.get("u_axis_ras", _default_u_axis(normal)))
    v_axis = _unit_vector(plane.get("v_axis_ras", np.cross(normal, u_axis)))
    size = plane.get("size_mm", [24.0, 24.0])
    if not isinstance(size, (list, tuple)) or len(size) < 2:
        return None
    half_u = max(float(size[0]) / 2.0, 0.0)
    half_v = max(float(size[1]) / 2.0, 0.0)
    rel = active_points - center
    distances = rel @ normal
    u = rel @ u_axis
    v = rel @ v_axis
    tol = max(min(spacing) * 0.75, 1.0e-6)
    inside = (np.abs(u) <= half_u + tol) & (np.abs(v) <= half_v + tol) & (distances >= -tol)
    if not np.any(inside):
        return None
    surface_distance = float(np.min(distances[inside]))
    thickness = max(float(plane.get("thickness_mm", 3.0)), 0.0)
    intrusion = max(_disk_intrusion_depth_mm(plane, default=2.0), 0.0)
    cap_inner = surface_distance + intrusion
    cap_outer = cap_inner - thickness
    d_min = min(cap_outer, cap_inner)
    d_max = max(cap_outer, cap_inner)
    corners = []
    for u_val in (-half_u, half_u):
        for v_val in (-half_v, half_v):
            for d_val in (d_min, d_max):
                corners.append(center + u_val * u_axis + v_val * v_axis + d_val * normal)
    points = np.vstack(corners)
    margin = np.asarray(spacing, dtype=float)
    return points.min(axis=0) - margin, points.max(axis=0) + margin


def _indices_zyx_to_xyz_points(
    indices_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> np.ndarray:
    spacing_arr = np.asarray(spacing, dtype=float)
    origin_arr = np.asarray(origin, dtype=float)
    return origin_arr + indices_zyx[:, ::-1].astype(float) * spacing_arr


def _pad_arrays_to_xyz_bounds(
    *,
    arrays: dict[str, np.ndarray],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    desired_min_xyz: np.ndarray,
    desired_max_xyz: np.ndarray,
    constant_values: dict[str, Any],
) -> tuple[dict[str, np.ndarray], tuple[float, float, float]]:
    first = np.asarray(next(iter(arrays.values())))
    shape_xyz = np.asarray(first.shape[::-1], dtype=float)
    spacing_arr = np.asarray(spacing, dtype=float)
    origin_arr = np.asarray(origin, dtype=float)
    current_min = origin_arr
    current_max = origin_arr + np.maximum(shape_xyz - 1.0, 0.0) * spacing_arr
    lower_xyz = np.maximum(
        0,
        np.ceil((current_min - np.asarray(desired_min_xyz, dtype=float)) / spacing_arr).astype(int),
    )
    upper_xyz = np.maximum(
        0,
        np.ceil((np.asarray(desired_max_xyz, dtype=float) - current_max) / spacing_arr).astype(int),
    )
    if not np.any(lower_xyz) and not np.any(upper_xyz):
        return {name: np.asarray(value) for name, value in arrays.items()}, origin
    lower_zyx = lower_xyz[::-1]
    upper_zyx = upper_xyz[::-1]
    pad_width = tuple((int(lower_zyx[i]), int(upper_zyx[i])) for i in range(3))
    padded = {
        name: np.pad(
            np.asarray(value),
            pad_width,
            mode="constant",
            constant_values=constant_values.get(name, 0),
        )
        for name, value in arrays.items()
    }
    padded_origin = tuple((origin_arr - lower_xyz * spacing_arr).tolist())
    return padded, padded_origin


def _crop_workflow_model_to_material_bbox(
    *,
    material_xyz: np.ndarray,
    labels_xyz: np.ndarray,
    node_label_xyz: np.ndarray,
    postprocess_mask_xyz: np.ndarray | None = None,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    node_sets: dict[str, list[tuple[int, int, int]]],
    percent_reference_node_sets: dict[str, list[tuple[int, int, int]]] | None,
) -> _CroppedWorkflowModel:
    """Crop finished workflow arrays to active material while preserving physics.

    Interactive workflow replay may temporarily pad the image so authored
    projected disks have enough space to generate. Once material and node sets
    exist, any all-zero canvas can be removed. Node coordinates are shifted by
    the cropped element offset; high-face nodes may remain equal to the cropped
    element dimensions, which is valid for hexahedral node coordinates.
    """
    material = np.asarray(material_xyz)
    if material.ndim != 3:
        raise ValueError("material_xyz must be a 3D x/y/z array")
    if postprocess_mask_xyz is None:
        postprocess_mask = material > 0
    else:
        postprocess_mask = np.asarray(postprocess_mask_xyz, dtype=bool)
        if postprocess_mask.shape != material.shape:
            raise ValueError("postprocess_mask_xyz shape must match material_xyz")
    active = material > 0
    original_shape = np.asarray(material.shape, dtype=np.int64)
    origin_arr = np.asarray(origin, dtype=float)
    spacing_arr = np.asarray(spacing, dtype=float)
    if not np.any(active):
        crop = _workflow_crop_metadata(
            enabled=False,
            lower_xyz=np.zeros(3, dtype=np.int64),
            upper_xyz=original_shape,
            original_shape=original_shape,
            origin_before=origin_arr,
            origin_after=origin_arr,
        )
        return _CroppedWorkflowModel(
            material_xyz=material,
            labels_xyz=np.asarray(labels_xyz),
            node_label_xyz=np.asarray(node_label_xyz),
            postprocess_mask_xyz=postprocess_mask,
            origin=origin,
            node_sets={name: list(nodes) for name, nodes in node_sets.items()},
            percent_reference_node_sets=(
                None
                if percent_reference_node_sets is None
                else {
                    name: list(nodes)
                    for name, nodes in percent_reference_node_sets.items()
                }
            ),
            crop=crop,
        )

    coords = np.argwhere(active)
    lower_xyz = coords.min(axis=0).astype(np.int64)
    upper_xyz = (coords.max(axis=0) + 1).astype(np.int64)
    crop_slices = tuple(slice(int(lower_xyz[axis]), int(upper_xyz[axis])) for axis in range(3))
    cropped_origin = origin_arr + lower_xyz.astype(float) * spacing_arr
    enabled = bool(np.any(lower_xyz != 0) or np.any(upper_xyz != original_shape))

    crop = _workflow_crop_metadata(
        enabled=enabled,
        lower_xyz=lower_xyz,
        upper_xyz=upper_xyz,
        original_shape=original_shape,
        origin_before=origin_arr,
        origin_after=cropped_origin,
    )
    return _CroppedWorkflowModel(
        material_xyz=material[crop_slices],
        labels_xyz=np.asarray(labels_xyz)[crop_slices],
        node_label_xyz=np.asarray(node_label_xyz)[crop_slices],
        postprocess_mask_xyz=postprocess_mask[crop_slices],
        origin=tuple(float(value) for value in cropped_origin),
        node_sets=_shift_node_sets(node_sets, lower_xyz=lower_xyz),
        percent_reference_node_sets=(
            None
            if percent_reference_node_sets is None
            else _shift_node_sets(percent_reference_node_sets, lower_xyz=lower_xyz)
        ),
        crop=crop,
    )


def _crop_array_to_workflow_crop(array_xyz: np.ndarray, crop: dict[str, Any]) -> np.ndarray:
    array = np.asarray(array_xyz)
    lower = np.asarray(crop.get("lower_index_xyz", [0, 0, 0]), dtype=np.int64)
    upper = np.asarray(crop.get("upper_index_xyz", array.shape), dtype=np.int64)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("workflow crop metadata must contain three-dimensional bounds")
    slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    return array[slices]


def _crop_nonlinear_material_to_workflow_crop(
    nonlinear_material,
    crop: dict[str, Any],
    *,
    material_xyz: np.ndarray,
):
    if nonlinear_material is None:
        return None

    active_xyz = np.asarray(material_xyz, dtype=np.float64) > 0.0

    def crop_zyx(array_zyx):
        array_xyz = np.transpose(np.asarray(array_zyx), (2, 1, 0))
        return _crop_array_to_workflow_crop(array_xyz, crop)

    poisson = nonlinear_material.poisson_ratio
    if isinstance(poisson, np.ndarray):
        poisson = to_zyx(np.where(active_xyz, crop_zyx(poisson), 0.0))
    return replace(
        nonlinear_material,
        youngs_modulus_mpa=to_zyx(
            np.where(active_xyz, np.asarray(material_xyz, dtype=np.float64), 0.0)
        ),
        poisson_ratio=poisson,
        compressive_yield_mpa=to_zyx(
            np.where(active_xyz, crop_zyx(nonlinear_material.compressive_yield_mpa), 0.0)
        ),
        tensile_yield_mpa=to_zyx(
            np.where(active_xyz, crop_zyx(nonlinear_material.tensile_yield_mpa), 0.0)
        ),
        plateau_mpa=to_zyx(
            np.where(active_xyz, crop_zyx(nonlinear_material.plateau_mpa), 0.0)
        ),
        material_id=to_zyx(
            np.where(active_xyz, crop_zyx(nonlinear_material.material_id), 0)
        ).astype(np.uint16, copy=False),
    )


def _assign_pmma_disks_to_nonlinear_material(
    nonlinear_material,
    *,
    disk_mask_xyz: np.ndarray,
    pmma_e_mpa: float,
    pmma_nu: float,
):
    if nonlinear_material is None:
        return None

    disk_mask = np.asarray(disk_mask_xyz, dtype=bool)

    def to_xyz(array_zyx):
        return np.transpose(np.asarray(array_zyx), (2, 1, 0)).copy()

    def to_zyx_array(array_xyz):
        return to_zyx(array_xyz)

    youngs_xyz = to_xyz(nonlinear_material.youngs_modulus_mpa)
    compressive_xyz = to_xyz(nonlinear_material.compressive_yield_mpa)
    tensile_xyz = to_xyz(nonlinear_material.tensile_yield_mpa)
    plateau_xyz = to_xyz(nonlinear_material.plateau_mpa)
    material_id_xyz = to_xyz(nonlinear_material.material_id).astype(
        np.uint16,
        copy=False,
    )
    poisson = nonlinear_material.poisson_ratio
    if isinstance(poisson, np.ndarray):
        poisson_xyz = to_xyz(poisson)
    else:
        poisson_xyz = np.full(youngs_xyz.shape, float(poisson), dtype=np.float64)

    youngs_xyz[disk_mask] = pmma_e_mpa
    poisson_xyz[disk_mask] = pmma_nu
    compressive_xyz[disk_mask] = 0.0
    tensile_xyz[disk_mask] = 0.0
    plateau_xyz[disk_mask] = 0.0
    material_id_xyz[disk_mask] = 2

    metadata = dict(getattr(nonlinear_material, "metadata", {}))
    metadata["pmma_fixture_material_id"] = 2
    return replace(
        nonlinear_material,
        youngs_modulus_mpa=to_zyx_array(youngs_xyz),
        poisson_ratio=to_zyx_array(poisson_xyz),
        compressive_yield_mpa=to_zyx_array(compressive_xyz),
        tensile_yield_mpa=to_zyx_array(tensile_xyz),
        plateau_mpa=to_zyx_array(plateau_xyz),
        material_id=to_zyx_array(material_id_xyz).astype(np.uint16, copy=False),
        metadata=metadata,
    )


def _workflow_crop_metadata(
    *,
    enabled: bool,
    lower_xyz: np.ndarray,
    upper_xyz: np.ndarray,
    original_shape: np.ndarray,
    origin_before: np.ndarray,
    origin_after: np.ndarray,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "lower_index_xyz": [int(value) for value in lower_xyz],
        "upper_index_xyz": [int(value) for value in upper_xyz],
        "original_shape_xyz": [int(value) for value in original_shape],
        "cropped_shape_xyz": [int(value) for value in (upper_xyz - lower_xyz)],
        "origin_before": [float(value) for value in origin_before],
        "origin_after": [float(value) for value in origin_after],
    }


def _shift_node_sets(
    node_sets: dict[str, list[tuple[int, int, int]]],
    *,
    lower_xyz: np.ndarray,
) -> dict[str, list[tuple[int, int, int]]]:
    offset = np.asarray(lower_xyz, dtype=np.int64)
    shifted: dict[str, list[tuple[int, int, int]]] = {}
    for name, nodes in node_sets.items():
        shifted[name] = [
            tuple(int(value) for value in (np.asarray(node, dtype=np.int64) - offset))
            for node in nodes
        ]
    return shifted


def _summarize_resolved_planes(editor: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(editor, dict):
        return []
    planes = editor.get("planes", [])
    if not isinstance(planes, list):
        return []
    summary = []
    for plane in planes:
        if not isinstance(plane, dict):
            continue
        summary.append(
            {
                "name": plane.get("name"),
                "relative_to": plane.get("relative_to"),
                "center_ras": plane.get("center_ras"),
                "normal_ras": plane.get("normal_ras"),
                "size_mm": plane.get("size_mm"),
                "relative_definition": plane.get("relative_definition"),
            }
        )
    return summary


def _json_safe_editor(editor: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(editor, dict):
        return None
    return _json_safe_value(editor)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _resolve_bbox_relative_plane(
    plane: dict[str, Any],
    *,
    model_mask_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    relative_to: str,
) -> dict[str, Any]:
    bounds_min, bounds_max = _bbox_bounds_xyz(
        np.asarray(model_mask_zyx, dtype=bool),
        spacing=spacing,
        origin=origin,
        use_image_bounds=relative_to == "image_bbox",
    )
    extent = np.maximum(bounds_max - bounds_min, np.asarray(spacing, dtype=float))
    normal = _unit_vector(plane.get("normal_ras", [0.0, 0.0, 1.0]))
    u_axis = _unit_vector(plane.get("u_axis_ras", _default_u_axis(normal)))
    v_axis = _unit_vector(plane.get("v_axis_ras", np.cross(normal, u_axis)))
    fraction_bounds = _bbox_fraction_bounds(plane.get("bbox_fraction_bounds"))
    if fraction_bounds is not None:
        fraction_min = fraction_bounds[:, 0]
        fraction_max = fraction_bounds[:, 1]
        center_fraction = 0.5 * (fraction_min + fraction_max)
        span_extent = np.abs(fraction_max - fraction_min) * extent
        size_u = _axis_extent(span_extent, u_axis)
        size_v = _axis_extent(span_extent, v_axis)
        relative_definition = {
            "relative_to": relative_to,
            "bbox_fraction_bounds": _bbox_fraction_bounds_metadata(plane.get("bbox_fraction_bounds")),
        }
    else:
        center_fraction = np.asarray(
            plane.get("center_fraction", plane.get("center_normalized", [0.5, 0.5, 0.5])),
            dtype=float,
        )
        if center_fraction.shape != (3,):
            raise ValueError("bbox-relative plane center_fraction must have three values")
        size_fraction = np.asarray(
            plane.get("size_fraction", plane.get("size_normalized", [1.0, 1.0])),
            dtype=float,
        )
        if size_fraction.shape != (2,):
            raise ValueError("bbox-relative plane size_fraction must have two values")
        size_u = _axis_extent(extent, u_axis) * float(size_fraction[0])
        size_v = _axis_extent(extent, v_axis) * float(size_fraction[1])
        relative_definition = {
            "relative_to": relative_to,
            "center_fraction": center_fraction.tolist(),
            "size_fraction": size_fraction.tolist(),
        }

    resolved = dict(plane)
    resolved["relative_definition"] = relative_definition
    resolved["relative_to"] = f"resolved_{relative_to}"
    resolved["center_ras"] = (bounds_min + center_fraction * extent).tolist()
    resolved["normal_ras"] = normal.tolist()
    resolved["u_axis_ras"] = u_axis.tolist()
    resolved["v_axis_ras"] = v_axis.tolist()
    resolved["size_mm"] = [float(size_u), float(size_v)]
    return resolved


def _bbox_fraction_bounds(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = [value.get(axis) for axis in ("x", "y", "z")]
    else:
        raw = list(value) if isinstance(value, (list, tuple)) else []
    if len(raw) != 3 or any(item is None for item in raw):
        raise ValueError("bbox_fraction_bounds must define x, y, and z bounds")
    bounds = []
    for item in raw:
        values = list(item) if isinstance(item, (list, tuple)) else [item, item]
        if len(values) != 2:
            raise ValueError("each bbox_fraction_bounds axis must contain min and max")
        bounds.append([float(values[0]), float(values[1])])
    return np.asarray(bounds, dtype=float)


def _bbox_fraction_bounds_metadata(value: Any) -> dict[str, list[float]]:
    bounds = _bbox_fraction_bounds(value)
    if bounds is None:
        return {}
    return {
        axis: [float(bounds[index, 0]), float(bounds[index, 1])]
        for index, axis in enumerate(("x", "y", "z"))
    }


def _bbox_bounds_xyz(
    mask_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    use_image_bounds: bool,
) -> tuple[np.ndarray, np.ndarray]:
    spacing_arr = np.asarray(spacing, dtype=float)
    origin_arr = np.asarray(origin, dtype=float)
    if use_image_bounds:
        shape_xyz = np.asarray(mask_zyx.shape[::-1], dtype=float)
        return origin_arr, origin_arr + np.maximum(shape_xyz - 1.0, 0.0) * spacing_arr
    indices_zyx = np.argwhere(mask_zyx)
    if indices_zyx.size == 0:
        raise ValueError("bbox-relative workflow planes require a non-empty model mask")
    indices_xyz = indices_zyx[:, ::-1].astype(float)
    bounds_min = origin_arr + indices_xyz.min(axis=0) * spacing_arr
    bounds_max = origin_arr + indices_xyz.max(axis=0) * spacing_arr
    return bounds_min, bounds_max


def _axis_extent(extent_xyz: np.ndarray, axis: np.ndarray) -> float:
    return float(np.sum(np.abs(axis) * extent_xyz))


def _unit_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError("plane vectors must have three values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("plane vector cannot be zero")
    return vector / norm


def _default_u_axis(normal: np.ndarray) -> np.ndarray:
    x_axis = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(normal, x_axis))) < 0.95:
        return x_axis
    return np.asarray([0.0, 1.0, 0.0])


def _align_workflow_arrays_to_reference(
    *,
    density_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    registration_mask_zyx: np.ndarray,
    projection_mask_zyx: np.ndarray,
    model_mask_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    registration_config: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    if not registration_config.get("enabled", True):
        return {
            "density": density_zyx,
            "mask": mask_zyx,
            "registration": registration_mask_zyx,
            "projection": projection_mask_zyx,
            "model": model_mask_zyx,
            "origin": origin,
            "metadata": {"enabled": False, "applied_to_model_grid": False},
        }
    reference_path = resolve_path(
        registration_config["reference_points"], base_dir=base_dir
    )
    max_points = int(registration_config.get("max_points", 8000))
    reference_points = _reference_points_from_config(
        reference_path,
        registration_config=registration_config,
        max_points=max_points,
    )
    sample_points = surface_points_from_mask(
        registration_mask_zyx,
        spacing=spacing,
        origin=origin,
        max_points=max_points,
        sample_mode=str(
            registration_config.get(
                "source_landmark_mode",
                registration_config.get("landmark_mode", "linspace"),
            )
        ),
        sample_offset=int(registration_config.get("source_landmark_offset", 0)),
    )
    reference_points, scaling_meta = _scale_reference_points_preserving_pose(
        reference_points=reference_points,
        sample_points=sample_points,
        registration_config=registration_config,
    )
    icp_direction = _reference_model_space_icp_direction(registration_config)
    source_to_target = estimate_rigid_icp(
        moving_points=reference_points,
        fixed_points=sample_points,
        iterations=int(registration_config.get("iterations", 50)),
        tolerance=float(registration_config.get("tolerance", 1.0e-4)),
        start_by_matching_centroids_only=_centroid_start(registration_config),
        convergence=str(registration_config.get("convergence", "delta")),
        distance_mode=str(registration_config.get("distance_mode", "mean")),
    )
    rotation, translation = _invert_rigid_transform(
        source_to_target["rotation"],
        source_to_target["translation"],
    )
    transform = {
        **source_to_target,
        "rotation": rotation,
        "translation": translation,
        "icp_direction": icp_direction,
    }
    active_points = surface_points_from_mask(
        model_mask_zyx,
        spacing=spacing,
        origin=origin,
        max_points=None,
    )
    output_origin, output_size = _output_grid_for_transform(
        active_points,
        rotation=transform["rotation"],
        translation=transform["translation"],
        spacing=spacing,
        margin_voxels=int(registration_config.get("margin_voxels", 4)),
    )
    return {
        "density": np.asarray(
            _resample_with_transform(
                density_zyx,
                spacing=spacing,
                origin=origin,
                output_spacing=spacing,
                output_origin=output_origin,
                output_size=output_size,
                rotation=transform["rotation"],
                translation=transform["translation"],
                interpolation="bspline",
            ),
            dtype=np.float64,
        ),
        "mask": _resample_with_transform(
            np.asarray(mask_zyx, dtype=np.uint16),
            spacing=spacing,
            origin=origin,
            output_spacing=spacing,
            output_origin=output_origin,
            output_size=output_size,
            rotation=transform["rotation"],
            translation=transform["translation"],
            interpolation="nearest",
        ).astype(mask_zyx.dtype, copy=False),
        "registration": _resample_with_transform(
            np.asarray(registration_mask_zyx, dtype=np.uint8),
            spacing=spacing,
            origin=origin,
            output_spacing=spacing,
            output_origin=output_origin,
            output_size=output_size,
            rotation=transform["rotation"],
            translation=transform["translation"],
            interpolation="nearest",
        )
        > 0,
        "projection": _resample_with_transform(
            np.asarray(projection_mask_zyx, dtype=np.uint8),
            spacing=spacing,
            origin=origin,
            output_spacing=spacing,
            output_origin=output_origin,
            output_size=output_size,
            rotation=transform["rotation"],
            translation=transform["translation"],
            interpolation="nearest",
        )
        > 0,
        "model": _resample_with_transform(
            np.asarray(model_mask_zyx, dtype=np.uint8),
            spacing=spacing,
            origin=origin,
            output_spacing=spacing,
            output_origin=output_origin,
            output_size=output_size,
            rotation=transform["rotation"],
            translation=transform["translation"],
            interpolation="nearest",
        )
        > 0,
        "origin": output_origin,
        "metadata": {
            "enabled": True,
            "method": str(registration_config.get("method", "vtk_icp")),
            "reference_points": str(reference_path),
            "iterations": transform["iterations"],
            "mean_distance": transform["mean_distance"],
            "rotation": transform["rotation"].tolist(),
            "translation": transform["translation"].tolist(),
            "reference_scaling": scaling_meta,
            "icp_direction": icp_direction,
            "applied_to_model_grid": True,
        },
        "reference_points": reference_points,
    }


def _workflow_registration_config(
    model_config: dict[str, Any], replay_cfg: dict[str, Any]
) -> dict[str, Any]:
    registration = dict(model_config.get("registration", {}))
    reference_points = replay_cfg.get("reference_points")
    if reference_points:
        registration["reference_points"] = reference_points
    editor_reference_points = replay_cfg.get("editor_reference_points")
    replay_space = str(replay_cfg.get("model_space", "reference")).strip().lower()
    if editor_reference_points and replay_space not in {"sample", "sample_space", "native"}:
        registration["reference_points"] = editor_reference_points
        registration["model_reference_points"] = reference_points
    registration.setdefault("enabled", bool(registration.get("reference_points")))
    registration.setdefault("method", "vtk_icp")
    registration.setdefault("reference_axis_order", "xyz")
    registration.setdefault("initialization", "centroid")
    registration.setdefault("convergence", "delta")
    registration.setdefault("distance_mode", "mean")
    registration.setdefault("max_points", 8000)
    registration.setdefault("iterations", 50)
    registration.setdefault("source_landmark_mode", "stride")
    registration.setdefault("source_landmark_offset", 0)
    return registration


def _workflow_registration_enabled(registration_config: dict[str, Any]) -> bool:
    return bool(registration_config.get("enabled", False)) and bool(
        registration_config.get("reference_points")
    )


def _estimate_reference_to_sample_transform(
    sample_mask_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    registration_config: dict[str, Any],
    base_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_path = resolve_path(
        registration_config["reference_points"], base_dir=base_dir
    )
    max_points = int(registration_config.get("max_points", 8000))
    reference_points = _reference_points_from_config(
        reference_path,
        registration_config=registration_config,
        max_points=max_points,
    )
    sample_points = surface_points_from_mask(
        sample_mask_zyx,
        spacing=spacing,
        origin=origin,
        max_points=max_points,
        sample_mode=str(
            registration_config.get(
                "source_landmark_mode",
                registration_config.get("landmark_mode", "linspace"),
            )
        ),
        sample_offset=int(registration_config.get("source_landmark_offset", 0)),
    )
    reference_points, scaling_meta = _maybe_scale_reference_points(
        reference_points=reference_points,
        sample_points=sample_points,
        registration_config=registration_config,
    )
    transform = estimate_rigid_icp(
        moving_points=reference_points,
        fixed_points=sample_points,
        iterations=int(registration_config.get("iterations", 50)),
        tolerance=float(registration_config.get("tolerance", 1.0e-4)),
        start_by_matching_centroids_only=_centroid_start(registration_config),
        convergence=str(registration_config.get("convergence", "delta")),
        distance_mode=str(registration_config.get("distance_mode", "mean")),
    )
    metadata = {
        "enabled": True,
        "method": str(registration_config.get("method", "vtk_icp")),
        "reference_points": str(reference_path),
        "iterations": transform["iterations"],
        "mean_distance": transform["mean_distance"],
        "rotation": transform["rotation"].tolist(),
        "translation": transform["translation"].tolist(),
        "reference_scaling": scaling_meta,
    }
    return metadata, transform


def _maybe_scale_reference_points(
    *,
    reference_points: np.ndarray,
    sample_points: np.ndarray,
    registration_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    scaling_cfg = registration_config.get("reference_scaling", {})
    if scaling_cfg is False:
        return reference_points, {"enabled": False}
    if scaling_cfg is True:
        scaling_cfg = {}
    if not isinstance(scaling_cfg, dict):
        return reference_points, {"enabled": False}
    if not scaling_cfg.get("enabled", False):
        return reference_points, {"enabled": False}
    ref_axes, ref_lengths, ref_center = _pca_axes_and_lengths(reference_points)
    sample_axes, sample_lengths, sample_center = _pca_axes_and_lengths(sample_points)
    min_factors = np.asarray(scaling_cfg.get("min_factors", [0.8, 0.8, 0.75]), dtype=float)
    max_factors = np.asarray(scaling_cfg.get("max_factors", [1.2, 1.2, 1.3]), dtype=float)
    scale = np.clip(sample_lengths / np.maximum(ref_lengths, 1.0e-6), min_factors, max_factors)
    reference_coordinates = (reference_points - ref_center) @ ref_axes
    scaled = sample_center + (reference_coordinates * scale) @ sample_axes.T
    return scaled, {
        "enabled": True,
        "source": "pca_axis_lengths",
        "reference_axis_lengths": ref_lengths.tolist(),
        "sample_axis_lengths": sample_lengths.tolist(),
        "scale_factors": scale.tolist(),
        "min_factors": min_factors.tolist(),
        "max_factors": max_factors.tolist(),
    }


def _reference_model_space_icp_direction(registration_config: dict[str, Any]) -> str:
    if "icp_direction" not in registration_config:
        return "reference_to_sample"
    value = registration_config["icp_direction"]
    token = str(value).strip().lower().replace("-", "_")
    if token == "reference_to_sample":
        return "reference_to_sample"
    raise ValueError(
        "registration.icp_direction is no longer selectable; "
        "use reference_to_sample or omit it"
    )


def _invert_rigid_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the point transform convention ``p2 = p1 @ R.T + t``."""
    rotation_arr = np.asarray(rotation, dtype=float)
    translation_arr = np.asarray(translation, dtype=float)
    inverse_rotation = rotation_arr.T
    inverse_translation = -translation_arr @ rotation_arr
    return inverse_rotation, inverse_translation


def _pca_axes_and_lengths(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(points, dtype=float)
    center = array.mean(axis=0)
    centered = array - center
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    coordinates = centered @ axes
    lengths = np.percentile(coordinates, 95, axis=0) - np.percentile(coordinates, 5, axis=0)
    lengths = np.maximum(lengths, 1.0e-6)
    return axes, lengths, center


def _centroid_start(registration_config: dict[str, Any]) -> bool:
    mode = str(registration_config.get("initialization", "")).strip().lower()
    return mode in {"centroid", "centroids", "center", "centre"}


def _workflow_load_axis(load_case_config: dict[str, Any] | None) -> str:
    cfg = {} if load_case_config is None else load_case_config
    if "axis" in cfg:
        axis = str(cfg["axis"]).strip().lower()
        if axis in AXIS_TO_INDEX:
            return axis
    for section in ("prescribed", "loaded"):
        for spec in cfg.get(section, ()):
            dof = str(spec.get("dof", "")).strip().lower()
            if dof in AXIS_TO_INDEX:
                return dof
    return "z"


def _workflow_effective_load_case_config(
    load_case_config: dict[str, Any] | None,
    *,
    resolved_editor: dict[str, Any] | None,
) -> dict[str, Any]:
    editor_case = _workflow_load_case_from_editor(resolved_editor)
    if editor_case is not None:
        return editor_case
    cfg = {} if load_case_config is None else load_case_config
    return {
        key: list(value) if key in {"fixed", "prescribed", "loaded"} and isinstance(value, list)
        else value
        for key, value in cfg.items()
    }


def _workflow_load_case_from_editor(
    resolved_editor: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(resolved_editor, dict):
        return None
    loads = resolved_editor.get("loads", [])
    planes = resolved_editor.get("planes", [])
    if not isinstance(loads, list) or not loads:
        return None
    if not isinstance(planes, list):
        planes = []

    planes_by_token = _workflow_planes_by_load_token(planes)
    fixed: list[dict[str, Any]] = []
    prescribed: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    unsupported = False

    for load in loads:
        if not isinstance(load, dict):
            continue
        plane_token = str(load.get("nodeset", load.get("plane", ""))).strip()
        if not plane_token:
            continue
        plane = planes_by_token.get(plane_token) or planes_by_token.get(
            _plane_to_nodeset_name(plane_token)
        )
        nodeset = _plane_to_nodeset_name(plane_token)
        if isinstance(plane, dict):
            plane_name = str(plane.get("name", plane_token)).strip()
            nodeset = _plane_to_nodeset_name(plane_name)

        mode = str(load.get("mode", load.get("bc_mode", ""))).strip().lower()
        if mode in {"fixed", "fix", "support"}:
            fixed.append(
                {
                    "nodeset": nodeset,
                    "dofs": _workflow_fixed_dofs(load, plane),
                    "value": 0.0,
                }
            )
            continue

        if mode in {"displacement", "dirichlet", "prescribed"}:
            vector = _workflow_load_direction_vector(load, plane)
            if vector is None:
                unsupported = True
                continue
            signed_components, units = _workflow_signed_components(load, vector)
            prescribed.extend(
                {
                    "nodeset": nodeset,
                    "dof": axis,
                    "value": _workflow_component_value(component, units),
                    "units": units,
                }
                for axis, component in signed_components
            )
            continue

        if mode in {"force", "load", "neumann"}:
            vector = _workflow_load_direction_vector(load, plane)
            if vector is None:
                unsupported = True
                continue
            signed_components, units = _workflow_signed_components(load, vector)
            loaded.extend(
                {
                    "nodeset": nodeset,
                    "dof": axis,
                    "value": float(component),
                    "units": units,
                    "distribute": bool(load.get("distribute", True)),
                }
                for axis, component in signed_components
            )
            continue

        unsupported = True

    if unsupported:
        return None
    return {
        "type": "nodeset",
        "fixed": fixed,
        "prescribed": prescribed,
        "loaded": loaded,
    }


def _workflow_planes_by_load_token(
    planes: list[Any],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for plane in planes:
        if not isinstance(plane, dict):
            continue
        name = str(plane.get("name", "")).strip()
        if not name:
            continue
        slug = _plane_to_nodeset_name(name)
        indexed[name] = plane
        indexed[slug] = plane
    return indexed


def _workflow_fixed_dofs(
    load: dict[str, Any],
    plane: dict[str, Any] | None,
) -> list[str]:
    for key in ("fixed_dofs", "dofs", "dof"):
        if key in load:
            return _workflow_axis_tokens(load[key])
    if isinstance(plane, dict):
        for key in ("fixed_dofs", "dofs", "dof"):
            if key in plane:
                return _workflow_axis_tokens(plane[key])
    return ["x", "y", "z"]


def _workflow_axis_tokens(value: Any) -> list[str]:
    raw_values = value if isinstance(value, (list, tuple, set)) else [value]
    axes: list[str] = []
    for raw in raw_values:
        token = str(raw).strip().lower()
        if token in AXIS_TO_INDEX and token not in axes:
            axes.append(token)
    return axes or ["x", "y", "z"]


def _workflow_load_direction_vector(
    load: dict[str, Any],
    plane: dict[str, Any] | None,
) -> np.ndarray | None:
    direction = str(load.get("direction", load.get("axis", "Plane normal"))).strip().lower()
    if direction in {"plane normal", "normal", "plane_normal"}:
        if not isinstance(plane, dict):
            return None
        return _unit_vector(plane.get("normal_ras", [0.0, 0.0, 1.0]))
    if direction in AXIS_TO_INDEX:
        vector = np.zeros(3, dtype=float)
        vector[AXIS_TO_INDEX[direction]] = 1.0
        return vector
    for key in ("vector_ras", "vector", "direction_vector"):
        if key in load:
            return _unit_vector(load[key])
    return None


def _workflow_signed_components(
    load: dict[str, Any],
    vector: np.ndarray,
) -> tuple[list[tuple[str, float]], str]:
    magnitude, units = _workflow_load_value_and_units(load)
    direction = _unit_vector(vector)
    components: list[tuple[str, float]] = []
    for axis, index in AXIS_TO_INDEX.items():
        component = float(direction[index]) * magnitude
        if abs(component) > 1.0e-12:
            components.append((axis, component))
    return components, units


def _workflow_load_value_and_units(load: dict[str, Any]) -> tuple[float, str]:
    raw = load.get("value", 0.0)
    units = str(load.get("units", "")).strip()
    if isinstance(raw, str):
        text = raw.strip()
        if text.endswith("%"):
            return float(text[:-1].strip()), "%"
        return float(text), units
    return float(raw), units


def _workflow_component_value(component: float, units: str) -> str | float:
    if units.strip() == "%":
        return f"{component:g}%"
    return float(component)


def _disk_intrusion_depth_mm(plane: dict[str, Any], *, default: float) -> float:
    if "intrusion_depth_mm" in plane:
        return float(plane["intrusion_depth_mm"])
    return float(default)
