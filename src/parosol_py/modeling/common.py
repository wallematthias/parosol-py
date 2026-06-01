from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from parosol_py.core import BoundaryConditionSet
from parosol_py.images import ImageGrid, export_scalar_image, to_output_order
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
    geometry = model_config.get("geometry", {})
    if "spacing" in model_config:
        spacing = _triple(model_config["spacing"], "model.spacing")
    if "spacing" in geometry:
        spacing = _triple(geometry["spacing"], "model.geometry.spacing")
    isotropic = geometry.get("isotropic_spacing", "auto")
    if isotropic and not np.allclose(spacing, spacing[0], rtol=1e-6, atol=1e-9):
        target = (
            min(float(v) for v in spacing)
            if str(isotropic).lower() == "auto"
            else float(isotropic)
        )
        density_zyx = _resample_array_zyx(
            density_zyx,
            spacing=spacing,
            target_spacing=target,
            interpolation="linear",
        )
        mask_zyx = _resample_array_zyx(
            mask_zyx,
            spacing=spacing,
            target_spacing=target,
            interpolation="nearest",
        )
        spacing = (target, target, target)
    if "origin" in model_config:
        origin = _triple(model_config["origin"], "model.origin")
    return (
        np.asarray(density_zyx, dtype=np.float64),
        np.asarray(mask_zyx),
        spacing,
        origin,
    )


def _resample_array_zyx(
    array_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    target_spacing: float,
    interpolation: str,
) -> np.ndarray:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(spacing)
    original_size = np.asarray(image.GetSize(), dtype=np.int64)
    original_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    new_spacing = np.asarray(
        [target_spacing, target_spacing, target_spacing], dtype=np.float64
    )
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


def material_from_density(
    density_zyx: np.ndarray,
    active_mask_zyx: np.ndarray,
    *,
    material_config: dict[str, Any],
) -> tuple[np.ndarray, float]:
    density_cfg = dict(material_config.get("density", {}))
    density_cfg.setdefault("equation", "linear")
    masked_density = np.where(active_mask_zyx, density_zyx, 0.0)
    mapped = density_to_material_map(
        masked_density,
        equation=str(density_cfg.get("equation", "linear")),
        poisson_ratio=material_config.get(
            "poisson_ratio", material_config.get("nu", 0.3)
        ),
        mask_threshold=float(density_cfg.get("mask_threshold", 0.0)),
        minimum_e_mpa=float(density_cfg.get("minimum_e_mpa", 0.0)),
        maximum_e_mpa=_optional_float(density_cfg.get("maximum_e_mpa")),
        **{
            key: value
            for key, value in density_cfg.items()
            if key
            not in {
                "equation",
                "mask_threshold",
                "minimum_e_mpa",
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
) -> tuple[np.ndarray, np.ndarray]:
    axis_index = AXIS_TO_INDEX[axis]
    thickness = max(1, int(thickness_voxels))
    inferior = np.zeros(mask_xyz.shape, dtype=bool)
    superior = np.zeros(mask_xyz.shape, dtype=bool)
    lateral_shape = tuple(mask_xyz.shape[idx] for idx in range(3) if idx != axis_index)
    for lateral in np.ndindex(lateral_shape):
        selector = [slice(None), slice(None), slice(None)]
        lateral_iter = iter(lateral)
        for idx in range(3):
            if idx != axis_index:
                selector[idx] = next(lateral_iter)
        line = mask_xyz[tuple(selector)]
        occupied = np.flatnonzero(line)
        if occupied.size == 0:
            continue
        lo = int(occupied.min())
        hi = int(occupied.max())
        selector[axis_index] = slice(max(0, lo - thickness), lo)
        inferior[tuple(selector)] = True
        selector[axis_index] = slice(
            hi + 1, min(mask_xyz.shape[axis_index], hi + 1 + thickness)
        )
        superior[tuple(selector)] = True
    return inferior, superior


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
) -> float:
    cfg = {} if load_case_config is None else load_case_config
    if "displacement" in cfg:
        return float(cfg["displacement"])
    if "normal_displacement" in cfg:
        return float(cfg["normal_displacement"])
    strain = float(cfg.get("strain", cfg.get("normal_strain", default)))
    axis_index = AXIS_TO_INDEX[axis]
    return strain * float(dimensions_xyz[axis_index]) * float(spacing[axis_index])


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
