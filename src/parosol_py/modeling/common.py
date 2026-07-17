from __future__ import annotations

import json
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from parosol_py.core import BoundaryConditionSet
from parosol_py.images import ImageGrid, export_scalar_image, to_output_order
from parosol_py.images import largest_connected_component
from parosol_py.materials import apply_density_input_transform, density_to_material_map
from parosol_py.nonlinear import hip_nonlinear, manual_nonlinear, spine_nonlinear
from parosol_py.nodesets import (
    boundary_conditions_from_nodesets,
    nodes_from_labeled_voxels,
)
from parosol_py.paths import suffix_text
from parosol_py.visualization import write_case_overview

from .io import read_image_zyx, resolve_path

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
NONLINEAR_PRESET_ERROR = (
    "materials.nonlinear.preset must be 'spine_nonlinear', 'hip_nonlinear', or 'manual'"
)


@dataclass(frozen=True)
class PreprocessedInputsPreview:
    density_zyx: np.ndarray
    mask_zyx: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    metadata: dict[str, Any]


def build_preprocessed_inputs_preview(
    model_config: dict[str, Any],
    *,
    base_dir: Path,
    preprocessing_config: dict[str, Any] | None = None,
) -> PreprocessedInputsPreview:
    """Return the shared pre-boundary-condition input grid."""

    density_zyx, mask_zyx, spacing, origin = load_density_and_mask(
        model_config,
        base_dir=base_dir,
        preprocessing_config=preprocessing_config,
        allow_foreground_mask=True,
    )
    return PreprocessedInputsPreview(
        density_zyx=np.asarray(density_zyx, dtype=np.float64),
        mask_zyx=np.asarray(mask_zyx),
        spacing=tuple(float(value) for value in spacing),
        origin=tuple(float(value) for value in origin),
        metadata={
            "model_space": "sample",
            "preprocessing": dict(preprocessing_config or {}),
        },
    )


def load_density_and_mask(
    model_config: dict[str, Any],
    *,
    base_dir: Path,
    preprocessing_config: dict[str, Any] | None = None,
    allow_foreground_mask: bool = False,
) -> tuple[
    np.ndarray, np.ndarray, tuple[float, float, float], tuple[float, float, float]
]:
    density_path = resolve_path(model_config["density_image"], base_dir=base_dir)
    density_zyx, spacing, origin = read_image_zyx(density_path)
    mask_path_value = model_config.get("mask_image")
    if mask_path_value:
        mask_path = resolve_path(mask_path_value, base_dir=base_dir)
        mask_zyx, mask_spacing, _mask_origin = read_image_zyx(mask_path)
    elif allow_foreground_mask:
        mask_zyx = np.asarray(density_zyx != 0, dtype=np.uint8)
        mask_spacing = spacing
    else:
        raise KeyError("mask_image")
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
        default_margin_voxels = preprocessing.get(
            "crop_margin_voxels", geometry.get("crop_margin_voxels", 4)
        )
        default_margin_mm = preprocessing.get(
            "crop_margin_mm", geometry.get("crop_margin_mm")
        )
        margin = _crop_margin_voxels_zyx(
            crop_spec,
            spacing=spacing,
            default_margin_voxels=default_margin_voxels,
            default_margin_mm=default_margin_mm,
        )
        crop_labels = _crop_labels(model_config, crop_spec)
        density_zyx, mask_zyx, origin = _crop_to_mask_bbox(
            density_zyx,
            mask_zyx,
            spacing=spacing,
            origin=origin,
            margin_voxels=margin,
            labels=crop_labels,
        )
    bbox_ratio_spec = preprocessing.get("bbox_ratio")
    aspect_spec = (
        bbox_ratio_spec
        if bbox_ratio_spec is not None
        else preprocessing.get(
            "normalize_aspect_ratio",
            preprocessing.get("aspect_ratio", preprocessing.get("aspect-ratio", {})),
        )
    )
    if _enabled(aspect_spec):
        crop_labels = _crop_labels(model_config, aspect_spec)
        crop_from_zyx = (
            _workflow_replay_slicer_crop_from_to_ras_zyx(
                _bbox_crop_from_to_zyx(
                    preprocessing.get(
                        "bbox_crop_from",
                        preprocessing.get("bbox_crop-from", {}),
                    )
                ),
                model_config,
                density_path,
            )
            if bbox_ratio_spec is not None
            else None
        )
        ratio_zyx = (
            _bbox_ratio_to_zyx(aspect_spec)
            if bbox_ratio_spec is not None
            else _aspect_ratio_zyx(aspect_spec)
        )
        density_zyx, mask_zyx, origin = _crop_to_mask_aspect_ratio(
            density_zyx,
            mask_zyx,
            spacing=spacing,
            origin=origin,
            ratio=ratio_zyx,
            crop_from=crop_from_zyx,
            labels=crop_labels,
        )
    if "spacing" in model_config:
        spacing = _triple(model_config["spacing"], "model.spacing")
    if "spacing" in geometry:
        spacing = _triple(geometry["spacing"], "model.geometry.spacing")
    input_spacing = spacing
    target_spacing = _target_preprocessing_resample_spacing(
        preprocessing.get("resample_isotropic"),
        geometry=geometry,
        spacing=spacing,
    )
    if target_spacing is None:
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
    if _enabled(smooth_spec) and _smooth_spacing_guard_allows(
        smooth_spec,
        input_spacing=input_spacing,
    ):
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


def _workflow_replay_slicer_crop_from_to_ras_zyx(
    crop_from_zyx: tuple[str | None, str | None, str | None],
    model_config: dict[str, Any],
    image_path: Path,
) -> tuple[str | None, str | None, str | None]:
    """Convert workflow crop ends from Slicer IJK to canonical RAS arrays.

    Workflow replay recipes are authored in Slicer. Plane geometry is stored in
    RAS, but crop-from min/max choices refer to the Slicer IJK grid. Medical
    image inputs are reoriented into a canonical RAS array for headless model
    building, where x/z index ends can be reversed relative to Slicer's IJK.
    """
    is_replay_type = str(model_config.get("type", "")).strip().lower() == "workflow_replay"
    replay_spec = model_config.get("workflow_replay")
    is_replay_enabled = isinstance(replay_spec, dict) and _enabled(
        replay_spec.get("enabled", False)
    )
    if not is_replay_type and not is_replay_enabled:
        return crop_from_zyx
    if not _uses_slicer_ijk_crop_convention(image_path):
        return crop_from_zyx
    return (
        _opposite_crop_end(crop_from_zyx[0]),
        crop_from_zyx[1],
        _opposite_crop_end(crop_from_zyx[2]),
    )


def _uses_slicer_ijk_crop_convention(path: Path) -> bool:
    return suffix_text(path).endswith((".nii", ".nii.gz", ".mha", ".mhd"))


def _opposite_crop_end(value: str | None) -> str | None:
    if value == "min":
        return "max"
    if value == "max":
        return "min"
    return value


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
    margin_voxels: int | tuple[int, int, int],
    labels: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    labels_array = np.asarray(mask_zyx)
    active = target_mask_from_labels(
        labels_array,
        labels,
        context="crop target labels",
    )
    if not np.any(active):
        raise ValueError("model mask has no foreground voxels")
    coords = np.argwhere(active)
    margin_zyx = np.asarray(_margin_voxels_as_zyx(margin_voxels), dtype=np.int64)
    lo_zyx = np.maximum(coords.min(axis=0) - margin_zyx, 0)
    hi_zyx = np.minimum(coords.max(axis=0) + margin_zyx + 1, active.shape)
    slices = tuple(slice(int(lo_zyx[idx]), int(hi_zyx[idx])) for idx in range(3))
    lo_xyz = lo_zyx[[2, 1, 0]]
    cropped_origin = tuple(
        float(origin[idx]) + float(lo_xyz[idx]) * float(spacing[idx])
        for idx in range(3)
    )
    return density_zyx[slices], mask_zyx[slices], cropped_origin


def _crop_margin_voxels_zyx(
    crop_spec: Any,
    *,
    spacing: tuple[float, float, float],
    default_margin_voxels: Any,
    default_margin_mm: Any,
) -> tuple[int, int, int]:
    if isinstance(crop_spec, dict):
        if "margin_mm" in crop_spec:
            return _margin_mm_to_voxels_zyx(crop_spec["margin_mm"], spacing=spacing)
        if "margin_voxels" in crop_spec:
            return _margin_voxels_to_zyx(crop_spec["margin_voxels"])
    if default_margin_mm is not None:
        return _margin_mm_to_voxels_zyx(default_margin_mm, spacing=spacing)
    if default_margin_voxels is None:
        default_margin_voxels = 4
    return _margin_voxels_to_zyx(default_margin_voxels)


def _margin_mm_to_voxels_zyx(
    margin_mm: Any,
    *,
    spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    values_xyz = _margin_values_xyz(margin_mm, name="margin_mm")
    spacing_xyz = np.asarray(spacing, dtype=np.float64)
    voxels_xyz = np.ceil(values_xyz / spacing_xyz).astype(np.int64)
    return tuple(int(max(0, value)) for value in voxels_xyz[[2, 1, 0]])


def _margin_voxels_to_zyx(margin_voxels: Any) -> tuple[int, int, int]:
    values_xyz = _margin_values_xyz(margin_voxels, name="margin_voxels")
    return tuple(int(max(0, round(value))) for value in values_xyz[[2, 1, 0]])


def _margin_voxels_as_zyx(margin_voxels: Any) -> tuple[int, int, int]:
    if isinstance(margin_voxels, (list, tuple, np.ndarray)):
        values = np.asarray(margin_voxels, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError("margin_voxels must be a scalar or three z/y/x values")
    else:
        values = np.asarray([float(margin_voxels)] * 3, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("margin_voxels must contain finite values")
    return tuple(int(max(0, round(value))) for value in values)


def _margin_values_xyz(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(f"{name} must be a scalar or three x/y/z values")
    else:
        values = np.asarray([float(value)] * 3, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return np.maximum(values, 0.0)


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


def target_mask_from_labels(
    mask: np.ndarray,
    labels: set[int] | list[int] | tuple[int, ...] | None,
    *,
    context: str = "target labels",
) -> np.ndarray:
    array = np.asarray(mask)
    if not labels:
        return array > 0
    requested = sorted({int(label) for label in labels})
    selected = np.isin(array, requested)
    if np.any(selected):
        return selected
    foreground = sorted(int(value) for value in np.unique(array) if int(value) != 0)
    if len(requested) == 1 and len(foreground) == 1:
        warnings.warn(
            f"{context} {requested} not present in mask; "
            f"using single foreground label {foreground[0]}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return array == foreground[0]
    return selected


def _crop_to_mask_aspect_ratio(
    density_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    ratio: tuple[float | None, float | None, float | None],
    crop_from: tuple[str | None, str | None, str | None] | None = None,
    labels: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    mask = np.asarray(mask_zyx)
    active = target_mask_from_labels(
        mask,
        labels,
        context="aspect-ratio crop target labels",
    )
    if not np.any(active):
        raise ValueError("model mask has no foreground voxels")

    numeric_axes = [axis for axis, value in enumerate(ratio) if value is not None]
    if not numeric_axes:
        return density_zyx, mask_zyx, origin
    reference_axes = [
        axis
        for axis in numeric_axes
        if np.isclose(float(ratio[axis]), 1.0)
    ]
    if not reference_axes:
        raise ValueError(
            "normalize_aspect_ratio.ratio must contain one preserved axis with value 1"
        )
    coords = np.argwhere(active)
    lo_zyx = coords.min(axis=0).astype(np.int64)
    hi_zyx = (coords.max(axis=0) + 1).astype(np.int64)
    size_zyx = hi_zyx - lo_zyx
    spacing_zyx = np.asarray((spacing[2], spacing[1], spacing[0]), dtype=np.float64)
    physical_size_zyx = size_zyx.astype(np.float64) * spacing_zyx
    reference_axis = min(reference_axes, key=lambda axis: float(physical_size_zyx[axis]))
    reference_length_mm = float(size_zyx[reference_axis]) * float(spacing_zyx[reference_axis])
    crop_from = crop_from or (None, None, None)

    out_lo = lo_zyx.copy()
    out_hi = hi_zyx.copy()
    for axis, axis_ratio in enumerate(ratio):
        if axis_ratio is None:
            continue
        target_mm = reference_length_mm * float(axis_ratio)
        requested_voxels = max(1, int(round(target_mm / float(spacing_zyx[axis]))))
        available_voxels = int(size_zyx[axis])
        if requested_voxels > available_voxels:
            axis_name = ("z", "y", "x")[axis]
            warnings.warn(
                "bbox_ratio cannot reach requested bbox_ratio on "
                f"{axis_name} axis: requested {target_mm:g} mm "
                f"({requested_voxels} voxels) exceeds foreground extent "
                f"{float(physical_size_zyx[axis]):g} mm ({available_voxels} voxels); "
                f"using the full available {axis_name} extent.",
                RuntimeWarning,
                stacklevel=2,
            )
        target_voxels = min(available_voxels, requested_voxels)
        mode = crop_from[axis]
        if mode == "min":
            start = int(hi_zyx[axis]) - target_voxels
        elif mode == "max":
            start = int(lo_zyx[axis])
        else:
            center = 0.5 * (float(lo_zyx[axis]) + float(hi_zyx[axis]))
            start = int(round(center - 0.5 * float(target_voxels)))
        start = max(int(lo_zyx[axis]), min(start, int(hi_zyx[axis]) - target_voxels))
        out_lo[axis] = start
        out_hi[axis] = start + target_voxels

    slices = tuple(slice(int(out_lo[axis]), int(out_hi[axis])) for axis in range(3))
    lo_xyz = out_lo[[2, 1, 0]]
    cropped_origin = tuple(
        float(origin[index]) + float(lo_xyz[index]) * float(spacing[index])
        for index in range(3)
    )
    return density_zyx[slices], mask_zyx[slices], cropped_origin


def _aspect_ratio_zyx(value: Any) -> tuple[float | None, float | None, float | None]:
    if isinstance(value, dict):
        raw = value.get("ratio", value.get("ratios", value.get("aspect_ratio")))
        if raw is None:
            raw = value
    else:
        raw = value
    if isinstance(raw, dict):
        ordered = [raw.get("z"), raw.get("y"), raw.get("x")]
    else:
        ordered = list(raw) if isinstance(raw, (list, tuple)) else []
    if len(ordered) != 3:
        raise ValueError("normalize_aspect_ratio.ratio must contain three z/y/x values")
    parsed: list[float | None] = []
    for item in ordered:
        if item is None:
            parsed.append(None)
            continue
        token = str(item).strip().lower()
        if token in {"", "none", "null", "auto"}:
            parsed.append(None)
            continue
        value_float = float(item)
        if value_float <= 0:
            raise ValueError("normalize_aspect_ratio.ratio values must be positive or null")
        parsed.append(value_float)
    return parsed[0], parsed[1], parsed[2]


def _bbox_ratio_to_zyx(value: Any) -> tuple[float | None, float | None, float | None]:
    """Convert recipe-facing bbox_ratio order to the cropper's z/y/x order."""
    if isinstance(value, dict):
        raw = value.get("ratio", value.get("ratios", value.get("bbox_ratio")))
        if raw is None:
            raw = value
    else:
        raw = value
    if isinstance(raw, dict):
        ordered = [
            raw.get("reference", raw.get("first")),
            raw.get("constrained", raw.get("second", raw.get("cropped"))),
            raw.get("free", raw.get("third")),
        ]
        if all(item is None for item in ordered):
            # Accept explicit z/y/x dictionaries as a convenience, then display
            # them through the same recipe order used by saved workflows.
            ratio_zyx = _aspect_ratio_zyx(raw)
            return ratio_zyx
    else:
        ordered = list(raw) if isinstance(raw, (list, tuple)) else []
    if len(ordered) != 3:
        raise ValueError("bbox_ratio must contain three reference/constrained/free values")
    parsed: list[float | None] = []
    for item in ordered:
        if item is None:
            parsed.append(None)
            continue
        token = str(item).strip().lower()
        if token in {"", "none", "null", "auto"}:
            parsed.append(None)
            continue
        value_float = float(item)
        if value_float <= 0:
            raise ValueError("bbox_ratio values must be positive or null")
        parsed.append(value_float)
    reference, constrained, free = parsed
    return constrained, reference, free


def _bbox_crop_from_to_zyx(value: Any) -> tuple[str | None, str | None, str | None]:
    if isinstance(value, dict):
        raw = value.get("crop_from", value.get("bbox_crop_from", value))
    else:
        raw = value
    if isinstance(raw, dict):
        ordered = [
            raw.get("reference", raw.get("first")),
            raw.get("constrained", raw.get("second", raw.get("cropped"))),
            raw.get("free", raw.get("third")),
        ]
        if all(item is None for item in ordered):
            ordered = [raw.get("z"), raw.get("y"), raw.get("x")]
            if any(item is not None for item in ordered):
                return tuple(_crop_from_value(item) for item in ordered)  # type: ignore[return-value]
    else:
        ordered = list(raw) if isinstance(raw, (list, tuple)) else []
    if not ordered:
        return None, None, None
    if len(ordered) != 3:
        raise ValueError("bbox_crop_from must contain three reference/constrained/free values")
    reference, constrained, free = (_crop_from_value(item) for item in ordered)
    return constrained, reference, free


def _crop_from_value(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"", "none", "null", "auto", "center", "centre"}:
        return None
    if token in {"min", "low", "lo", "start"}:
        return "min"
    if token in {"max", "high", "hi", "end"}:
        return "max"
    raise ValueError("bbox_crop_from values must be min, max, center, or null")


def pad_arrays_to_foreground_margin(
    *,
    anchor_mask_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    margin_voxels: int,
    arrays: dict[str, np.ndarray],
    constant_values: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[float, float, float]]:
    active = np.asarray(anchor_mask_zyx, dtype=bool)
    if not np.any(active):
        raise ValueError("foreground margin padding requires a non-empty anchor mask")
    target = max(0, int(margin_voxels))
    if target == 0:
        return {name: np.asarray(value) for name, value in arrays.items()}, origin
    coords = np.argwhere(active)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    shape = np.asarray(active.shape, dtype=int)
    lower = np.maximum(0, target - lo)
    upper = np.maximum(0, target - ((shape - 1) - hi))
    if not np.any(lower) and not np.any(upper):
        return {name: np.asarray(value) for name, value in arrays.items()}, origin
    pad_width = tuple((int(lower[i]), int(upper[i])) for i in range(3))
    constants = {} if constant_values is None else constant_values
    padded: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        padded[name] = np.pad(
            array,
            pad_width,
            mode="constant",
            constant_values=constants.get(name, 0),
        )
    padded_origin = (
        float(origin[0]) - float(lower[2]) * float(spacing[0]),
        float(origin[1]) - float(lower[1]) * float(spacing[1]),
        float(origin[2]) - float(lower[0]) * float(spacing[2]),
    )
    return padded, padded_origin


def fixture_margin_voxels(
    model_config: dict[str, Any],
    *,
    spacing: tuple[float, float, float],
    default_axis: str = "z",
    default_intrusion_scale: float = 2.5,
) -> int:
    geometry = model_config.get("geometry", {})
    axis = str(geometry.get("cap_axis", geometry.get("axis", default_axis))).strip().lower()
    if axis not in AXIS_TO_INDEX:
        axis = default_axis
    axis_index = AXIS_TO_INDEX[axis]
    feature = geometry.get("cap")
    if not isinstance(feature, dict):
        feature = geometry.get("disk")
    if not isinstance(feature, dict):
        feature = geometry

    def _value_voxels(
        *voxel_keys: str,
        mm_keys: tuple[str, ...] = (),
    ) -> int | None:
        for key in voxel_keys:
            if key in feature:
                return int(feature[key])
            if key in geometry:
                return int(geometry[key])
        for key in mm_keys:
            if key in feature:
                return int(round(float(feature[key]) / spacing[axis_index]))
            if key in geometry:
                return int(round(float(geometry[key]) / spacing[axis_index]))
        return None

    thickness = _value_voxels(
        "thickness_voxels",
        "pmma_thickness_voxels",
        mm_keys=("thickness_mm", "pmma_thickness_mm"),
    )
    if thickness is None:
        return 0
    thickness = max(1, int(thickness))
    intrusion = _value_voxels(
        "intrusion_depth_voxels",
        "endplate_depth_voxels",
        "surface_depth_voxels",
        mm_keys=("intrusion_depth_mm", "endplate_depth_mm", "surface_depth_mm"),
    )
    if intrusion is None:
        intrusion = int(round(float(thickness) * float(default_intrusion_scale)))
    return max(0, thickness + max(1, int(intrusion)))


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


def _smooth_spacing_guard_allows(
    smooth_spec: Any,
    *,
    input_spacing: tuple[float, float, float],
) -> bool:
    if not isinstance(smooth_spec, dict):
        return True
    threshold = smooth_spec.get(
        "when_spacing_above_mm",
        smooth_spec.get(
            "spacing_threshold_mm",
            smooth_spec.get("mask_smoothing_spacing_threshold_mm"),
        ),
    )
    if threshold is None:
        return True
    return any(float(value) > float(threshold) for value in input_spacing)


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


def _target_preprocessing_resample_spacing(
    resample_spec: Any,
    *,
    geometry: dict[str, Any],
    spacing: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    if not _enabled(resample_spec):
        return None
    spec = resample_spec if isinstance(resample_spec, dict) else {}
    tolerance = float(
        spec.get(
            "spacing_tolerance_mm",
            geometry.get(
                "spacing_tolerance_mm",
                geometry.get("resample_tolerance_mm", 1.0e-3),
            ),
        )
    )
    rtol = float(
        spec.get(
            "spacing_tolerance_relative",
            geometry.get("spacing_tolerance_relative", 1.0e-5),
        )
    )
    if "target_spacing" in spec:
        target = _triple(
            spec["target_spacing"],
            "preprocessing.resample_isotropic.target_spacing",
        )
    else:
        raw_value = spec.get(
            "target_spacing_mm",
            spec.get("spacing_mm", spec.get("spacing")),
        )
        if raw_value is None:
            mode = str(spec.get("mode", "auto")).strip().lower()
            if mode in {"", "auto", "isotropic"}:
                if np.allclose(spacing, spacing[0], rtol=rtol, atol=tolerance):
                    return None
                raw_value = min(float(value) for value in spacing)
            else:
                raw_value = mode
        target_value = float(raw_value)
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
        sitk.sitkNearestNeighbor
        if interpolation == "nearest"
        else sitk.sitkBSpline
        if interpolation == "bspline"
        else sitk.sitkLinear
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
    density_values = _apply_density_input_transform(
        np.asarray(density_zyx, dtype=np.float64),
        density_cfg=density_cfg,
    )
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
        density_values,
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
        bin_material=_enabled(
            density_cfg.get(
                "bin_material",
                e_cfg.get("bin_material", e_cfg.get("binned_material", False)),
            )
        ),
        number_bins=int(
            density_cfg.get(
                "number_bins",
                density_cfg.get(
                    "bins",
                    e_cfg.get("number_bins", e_cfg.get("bins", 128)),
                ),
            )
        ),
        bin_value=density_cfg.get(
            "bin_value",
            density_cfg.get(
                "bin_assignment",
                e_cfg.get("bin_value", e_cfg.get("bin_assignment", "center")),
            ),
        ),
        **{
            key: value
            for key, value in e_cfg.items()
            if key
            not in {
                "equation",
                "bin_material",
                "binned_material",
                "number_bins",
                "bins",
                "bin_value",
                "bin_assignment",
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


def nonlinear_material_from_density(
    density_zyx: np.ndarray,
    active_mask_zyx: np.ndarray | None,
    *,
    material_config: dict[str, Any],
    poisson_ratio: float | np.ndarray,
):
    parsed = nonlinear_preset_from_material_config(material_config)
    if parsed is None:
        return None
    preset, nonlinear_cfg = parsed

    density_cfg = dict(material_config.get("density", {}))
    density_values = _apply_density_input_transform(
        np.asarray(density_zyx, dtype=np.float64),
        density_cfg=density_cfg,
    )
    bin_material = _enabled(nonlinear_cfg.get("bin_material", False))
    number_bins = int(nonlinear_cfg.get("number_bins", nonlinear_cfg.get("bins", 128)))
    if preset == "spine_nonlinear":
        return spine_nonlinear(
            density_values,
            active_mask=active_mask_zyx,
            poisson_ratio=poisson_ratio,
            bin_material=bin_material,
            number_bins=number_bins,
        )

    if preset == "manual":
        return manual_nonlinear(
            density_values,
            elastic=_required_manual_nonlinear_law(nonlinear_cfg, "elastic"),
            compressive_yield=_required_manual_nonlinear_law(
                nonlinear_cfg, "compressive_yield"
            ),
            tensile_yield=_required_manual_nonlinear_law(nonlinear_cfg, "tensile_yield"),
            plateau=nonlinear_cfg.get("plateau"),
            active_mask=active_mask_zyx,
            poisson_ratio=poisson_ratio,
            bin_material=bin_material,
            number_bins=number_bins,
        )

    basis = _density_basis(density_cfg)
    if basis != "rho_app":
        raise ValueError(
            "materials.nonlinear.preset='hip_nonlinear' requires "
            "materials.density.basis='rho_app'"
        )
    return hip_nonlinear(
        density_values,
        site=str(nonlinear_cfg.get("site", "femoral_neck")),
        active_mask=active_mask_zyx,
        poisson_ratio=poisson_ratio,
        bin_material=bin_material,
        number_bins=number_bins,
    )


def nonlinear_preset_from_material_config(
    material_config: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    nonlinear_cfg = material_config.get("nonlinear")
    if not nonlinear_cfg:
        return None
    if not isinstance(nonlinear_cfg, dict):
        raise ValueError("materials.nonlinear must be an object")
    preset = str(nonlinear_cfg.get("preset", "")).strip().lower()
    if preset not in {"spine_nonlinear", "hip_nonlinear", "manual"}:
        raise ValueError(NONLINEAR_PRESET_ERROR)
    return preset, nonlinear_cfg


def _required_manual_nonlinear_law(
    nonlinear_cfg: dict[str, Any], key: str
) -> dict[str, Any]:
    value = nonlinear_cfg.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"materials.nonlinear.{key} must be an object")
    return value


def _density_basis(density_cfg: dict[str, Any]) -> str | None:
    for key in ("basis", "density_basis", "units", "unit"):
        value = density_cfg.get(key)
        if value is not None:
            return str(value).strip().lower().replace("-", "_")
    return None


def _apply_density_input_transform(
    density_zyx: np.ndarray,
    *,
    density_cfg: dict[str, Any],
) -> np.ndarray:
    return apply_density_input_transform(density_zyx, density_cfg.get("input_transform"))


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
    intrusion = max(
        0,
        int(round(thickness * 2.5))
        if intrusion_depth_voxels is None
        else int(intrusion_depth_voxels),
    )
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
    inferior_start = max(0, global_lo + intrusion - thickness)
    inferior_stop = min(mask_xyz.shape[axis_index], global_lo + intrusion)
    superior_start = max(0, global_hi - intrusion + 1)
    superior_stop = min(mask_xyz.shape[axis_index], superior_start + thickness)

    for lateral in np.ndindex(lateral_shape):
        if inferior_footprint[lateral]:
            selector = _column_selector(lateral, axis_index)
            selector[axis_index] = slice(inferior_start, inferior_stop)
            inferior[tuple(selector)] = True
        if superior_footprint[lateral]:
            selector = _column_selector(lateral, axis_index)
            selector[axis_index] = slice(superior_start, superior_stop)
            superior[tuple(selector)] = True
    inferior[mask_xyz] = False
    superior[mask_xyz] = False
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


def _fill_short_1d_gaps(line: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(line, dtype=bool).copy()
    false_runs = np.flatnonzero(~out)
    if false_runs.size == 0:
        return out
    start = 0
    while start < false_runs.size:
        stop = start + 1
        while stop < false_runs.size and false_runs[stop] == false_runs[stop - 1] + 1:
            stop += 1
        run = false_runs[start:stop]
        if (
            run.size <= int(max_gap)
            and int(run[0]) > 0
            and int(run[-1]) < out.size - 1
            and out[int(run[0]) - 1]
            and out[int(run[-1]) + 1]
        ):
            out[run] = True
        start = stop
    return out


def _fill_short_2d_gaps(values: np.ndarray, max_gap: int = 2) -> np.ndarray:
    out = np.asarray(values, dtype=bool).copy()
    for row in range(out.shape[0]):
        out[row, :] = _fill_short_1d_gaps(out[row, :], max_gap)
    for col in range(out.shape[1]):
        out[:, col] = _fill_short_1d_gaps(out[:, col], max_gap)
    return out


def _clean_largest_2d_component(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_fill_holes

    values = np.asarray(mask, dtype=bool)
    values = _fill_short_2d_gaps(values, max_gap=2)
    values = binary_fill_holes(values).astype(bool)
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
    if mode in {"rectangle", "rectangular", "square"}:
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
            "disk shape must be one of anatomy, rectangle, square, round, or hex"
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


def export_model_artifacts(
    *,
    material_xyz: np.ndarray,
    labels_xyz: np.ndarray,
    nodeset_labels_xyz: np.ndarray | None = None,
    disk_labels_xyz: np.ndarray | None = None,
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
        nodeset_labels = (
            labels_xyz
            if nodeset_labels_xyz is None
            else np.asarray(nodeset_labels_xyz)
        )
        if nodeset_labels.shape != labels_xyz.shape:
            raise ValueError("nodeset_labels_xyz shape must match labels_xyz")
        exported["nodeset_image"] = export_scalar_image(
            ImageGrid(nodeset_labels.astype(np.float32), spacing, origin),
            resolve_path(output_cfg["nodeset_image"], base_dir=base_dir),
        )
    if "disk_label_image" in output_cfg:
        disk_labels = (
            np.zeros_like(labels_xyz)
            if disk_labels_xyz is None
            else np.asarray(disk_labels_xyz)
        )
        if disk_labels.shape != labels_xyz.shape:
            raise ValueError("disk_labels_xyz shape must match labels_xyz")
        exported["disk_label_image"] = export_scalar_image(
            ImageGrid(disk_labels.astype(np.float32), spacing, origin),
            resolve_path(output_cfg["disk_label_image"], base_dir=base_dir),
        )
    if "qc_image" in output_cfg:
        qc_path = resolve_path(output_cfg["qc_image"], base_dir=base_dir)
        qc_material, qc_labels, qc_origin, qc_offset = _crop_for_qc(
            material_xyz,
            labels_xyz,
            spacing=spacing,
            origin=origin,
        )
        qc_boundary_conditions = _shift_boundary_conditions_for_crop(
            boundary_conditions,
            offset_xyz=qc_offset,
        )
        exported["qc_image"] = write_case_overview(
            qc_material,
            output_path=qc_path,
            spacing=spacing,
            origin=qc_origin,
            field_xyz=qc_labels.astype(np.float32),
            field_name="MODEL LABELS",
            material_labels_xyz=qc_labels,
            boundary_conditions=qc_boundary_conditions,
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
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], tuple[int, int, int]]:
    active = np.asarray(material_xyz) > 0
    if not np.any(active):
        return material_xyz, labels_xyz, origin, (0, 0, 0)
    coords = np.argwhere(active)
    lo = np.maximum(coords.min(axis=0) - int(margin), 0)
    hi = np.minimum(coords.max(axis=0) + int(margin) + 1, material_xyz.shape)
    slices = tuple(slice(int(lo[idx]), int(hi[idx])) for idx in range(3))
    cropped_origin = tuple(
        float(origin[idx]) + float(lo[idx]) * float(spacing[idx]) for idx in range(3)
    )
    return (
        material_xyz[slices],
        labels_xyz[slices],
        cropped_origin,
        tuple(int(v) for v in lo),
    )


def _shift_boundary_conditions_for_crop(
    boundary_conditions: BoundaryConditionSet,
    *,
    offset_xyz: tuple[int, int, int],
) -> BoundaryConditionSet:
    offset = np.asarray(offset_xyz, dtype=np.int64)
    if not np.any(offset):
        return boundary_conditions

    def _shift(coords: np.ndarray) -> np.ndarray:
        values = np.asarray(coords, dtype=np.int64).copy()
        if values.size == 0:
            return values.astype(np.uint16).reshape((-1, 4))
        values[:, :3] -= offset
        keep = np.all(values[:, :3] >= 0, axis=1)
        return values[keep].astype(np.uint16, copy=False).reshape((-1, 4))

    fixed = _shift(boundary_conditions.fixed_coordinates)
    loaded = _shift(boundary_conditions.loaded_coordinates)
    fixed_keep = np.all(
        np.asarray(boundary_conditions.fixed_coordinates, dtype=np.int64)[:, :3] >= offset,
        axis=1,
    ) if boundary_conditions.fixed_coordinates.size else np.zeros((0,), dtype=bool)
    loaded_keep = np.all(
        np.asarray(boundary_conditions.loaded_coordinates, dtype=np.int64)[:, :3] >= offset,
        axis=1,
    ) if boundary_conditions.loaded_coordinates.size else np.zeros((0,), dtype=bool)
    node_sets = {
        name: [
            tuple(int(coord[idx]) - int(offset[idx]) for idx in range(3))
            for coord in coords
            if all(int(coord[idx]) >= int(offset[idx]) for idx in range(3))
        ]
        for name, coords in boundary_conditions.node_sets.items()
    }
    return BoundaryConditionSet(
        fixed_coordinates=fixed,
        fixed_values=boundary_conditions.fixed_values[fixed_keep],
        loaded_coordinates=loaded,
        loaded_values=boundary_conditions.loaded_values[loaded_keep],
        node_sets=node_sets,
    )


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
