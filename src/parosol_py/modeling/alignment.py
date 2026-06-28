from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .io import resolve_path


@dataclass(frozen=True)
class AlignmentResult:
    density_zyx: np.ndarray
    body_mask_zyx: np.ndarray
    process_mask_zyx: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MaskAlignmentResult:
    density_zyx: np.ndarray
    mask_zyx: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    metadata: dict[str, Any]


def align_mask_to_reference(
    *,
    density_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    registration_config: dict[str, Any],
    base_dir: Path,
) -> MaskAlignmentResult:
    """Rigidly align a binary mask and density image to a reference point cloud."""

    if not registration_config.get("enabled", True):
        return MaskAlignmentResult(
            density_zyx=density_zyx,
            mask_zyx=mask_zyx,
            spacing=spacing,
            origin=origin,
            metadata={"enabled": False},
        )

    reference_path = resolve_path(
        registration_config["reference_points"], base_dir=base_dir
    )
    max_points = int(registration_config.get("max_points", 8000))
    iterations = int(registration_config.get("iterations", 50))
    tolerance = float(registration_config.get("tolerance", 1.0e-4))
    margin_voxels = int(registration_config.get("margin_voxels", 4))
    method = str(registration_config.get("method", "lightweight_icp")).strip().lower()
    reference_points = _reference_points_from_config(
        reference_path,
        registration_config=registration_config,
        max_points=max_points,
    )
    moving_points = surface_points_from_mask(
        mask_zyx,
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
    reference_points, scaling_meta = _scale_reference_points_to_sample(
        moving_points=moving_points,
        reference_points=reference_points,
        registration_config=registration_config,
    )
    transform = estimate_rigid_icp(
        moving_points=moving_points,
        fixed_points=reference_points,
        iterations=iterations,
        tolerance=tolerance,
        start_by_matching_centroids_only=_icp_centroid_start(registration_config),
        convergence=str(registration_config.get("convergence", "delta")),
        distance_mode=str(registration_config.get("distance_mode", "mean")),
    )
    full_surface_points = surface_points_from_mask(
        mask_zyx,
        spacing=spacing,
        origin=origin,
        max_points=None,
    )
    output_origin, output_size = _output_grid_for_transform(
        full_surface_points,
        rotation=transform["rotation"],
        translation=transform["translation"],
        spacing=spacing,
        margin_voxels=margin_voxels,
    )
    density = _resample_with_transform(
        density_zyx,
        spacing=spacing,
        origin=origin,
        output_spacing=spacing,
        output_origin=output_origin,
        output_size=output_size,
        rotation=transform["rotation"],
        translation=transform["translation"],
        interpolation="bspline",
    )
    mask = _resample_with_transform(
        mask_zyx.astype(np.uint8),
        spacing=spacing,
        origin=origin,
        output_spacing=spacing,
        output_origin=output_origin,
        output_size=output_size,
        rotation=transform["rotation"],
        translation=transform["translation"],
        interpolation="nearest",
    ) > 0
    if registration_config.get("crop_to_bbox", True):
        density, mask, output_origin = _crop_arrays_to_mask_bbox(
            density,
            mask,
            spacing=spacing,
            origin=output_origin,
            margin_voxels=int(registration_config.get("crop_margin_voxels", 2)),
        )
    return MaskAlignmentResult(
        density_zyx=np.asarray(density, dtype=np.float64),
        mask_zyx=mask,
        spacing=spacing,
        origin=output_origin,
        metadata={
            "enabled": True,
            "method": method,
            "reference_points": str(reference_path),
            "iterations": transform["iterations"],
            "mean_distance": transform["mean_distance"],
            "rotation": transform["rotation"].tolist(),
            "translation": transform["translation"].tolist(),
            "output_size_xyz": list(output_size),
            "reference_scaling": scaling_meta,
        },
    )


def _crop_arrays_to_mask_bbox(
    density_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    margin_voxels: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    active = np.asarray(mask_zyx, dtype=bool)
    if not np.any(active):
        raise ValueError("registered mask has no foreground voxels")
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


def align_spine_body_to_reference(
    *,
    density_zyx: np.ndarray,
    body_mask_zyx: np.ndarray,
    process_mask_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    registration_config: dict[str, Any],
    base_dir: Path,
) -> AlignmentResult:
    """Align a vertebral body to a reference point cloud without VTK."""

    if not registration_config.get("enabled", True):
        return AlignmentResult(
            density_zyx=density_zyx,
            body_mask_zyx=body_mask_zyx,
            process_mask_zyx=process_mask_zyx,
            spacing=spacing,
            origin=origin,
            metadata={"enabled": False},
        )

    reference_path = resolve_path(
        registration_config["reference_points"], base_dir=base_dir
    )
    max_points = int(registration_config.get("max_points", 8000))
    iterations = int(registration_config.get("iterations", 50))
    tolerance = float(registration_config.get("tolerance", 1.0e-4))
    margin_voxels = int(registration_config.get("margin_voxels", 4))
    method = str(registration_config.get("method", "lightweight_icp")).strip().lower()
    reference_points = _reference_points_from_config(
        reference_path,
        registration_config=registration_config,
        max_points=max_points,
    )
    body_points = surface_points_from_mask(
        body_mask_zyx,
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
    reference_points, scaling_meta = _scale_reference_points_to_sample(
        moving_points=body_points,
        reference_points=reference_points,
        registration_config=registration_config,
    )
    transform = estimate_rigid_icp(
        moving_points=body_points,
        fixed_points=reference_points,
        iterations=iterations,
        tolerance=tolerance,
        start_by_matching_centroids_only=_icp_centroid_start(registration_config),
        convergence=str(registration_config.get("convergence", "delta")),
        distance_mode=str(registration_config.get("distance_mode", "mean")),
    )
    active_points = surface_points_from_mask(
        body_mask_zyx | process_mask_zyx,
        spacing=spacing,
        origin=origin,
        max_points=None,
    )
    output_origin, output_size = _output_grid_for_transform(
        active_points,
        rotation=transform["rotation"],
        translation=transform["translation"],
        spacing=spacing,
        margin_voxels=margin_voxels,
    )
    density = _resample_with_transform(
        density_zyx,
        spacing=spacing,
        origin=origin,
        output_spacing=spacing,
        output_origin=output_origin,
        output_size=output_size,
        rotation=transform["rotation"],
        translation=transform["translation"],
        interpolation="bspline",
    )
    body = _resample_with_transform(
        body_mask_zyx.astype(np.uint8),
        spacing=spacing,
        origin=origin,
        output_spacing=spacing,
        output_origin=output_origin,
        output_size=output_size,
        rotation=transform["rotation"],
        translation=transform["translation"],
        interpolation="nearest",
    ) > 0
    process = _resample_with_transform(
        process_mask_zyx.astype(np.uint8),
        spacing=spacing,
        origin=origin,
        output_spacing=spacing,
        output_origin=output_origin,
        output_size=output_size,
        rotation=transform["rotation"],
        translation=transform["translation"],
        interpolation="nearest",
    ) > 0
    return AlignmentResult(
        density_zyx=np.asarray(density, dtype=np.float64),
        body_mask_zyx=body,
        process_mask_zyx=process,
        spacing=spacing,
        origin=output_origin,
        metadata={
            "enabled": True,
            "method": method,
            "reference_points": str(reference_path),
            "iterations": transform["iterations"],
            "mean_distance": transform["mean_distance"],
            "rotation": transform["rotation"].tolist(),
            "translation": transform["translation"].tolist(),
            "output_size_xyz": list(output_size),
            "reference_scaling": scaling_meta,
        },
    )


def read_reference_points(path: str | Path, *, max_points: int | None = None) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        points = np.asarray(np.load(path), dtype=float)
    elif suffixes.endswith(".npz"):
        with np.load(path) as data:
            key = "points" if "points" in data else data.files[0]
            points = np.asarray(data[key], dtype=float)
    elif path.suffix.lower() == ".vtk":
        points = _read_ascii_vtk_points(path)
    else:
        points = np.loadtxt(path, dtype=float)
    points = _points_array(points, "reference points")
    return _sample_points(points, max_points=max_points)


def _reference_points_from_config(
    path: str | Path,
    *,
    registration_config: dict[str, Any],
    max_points: int | None,
) -> np.ndarray:
    points = read_reference_points(path, max_points=None)
    points = orient_reference_points(
        points,
        axis_order=registration_config.get(
            "reference_axis_order",
            registration_config.get("reference_order", "xyz"),
        ),
        flips=registration_config.get("reference_flips", (False, False, False)),
    )
    mirror_axis = registration_config.get("mirror_axis")
    if mirror_axis is not None:
        points = _mirror_points(points, axis=str(mirror_axis))
    return _sample_points(
        points,
        max_points=max_points,
        mode=str(
            registration_config.get(
                "reference_landmark_mode",
                registration_config.get("landmark_mode", "linspace"),
            )
        ),
        offset=int(registration_config.get("reference_landmark_offset", 0)),
    )


def orient_reference_points(
    points: np.ndarray,
    *,
    axis_order: str | tuple[str, str, str] | list[str] = "xyz",
    flips: Any = (False, False, False),
) -> np.ndarray:
    """Convert stored reference points into physical x/y/z coordinates."""

    values = _points_array(points, "reference points").copy()
    if isinstance(axis_order, str):
        order = tuple(axis_order.lower().replace(",", "").replace(" ", ""))
    else:
        order = tuple(str(axis).lower() for axis in axis_order)
    if order != ("x", "y", "z"):
        if sorted(order) != ["x", "y", "z"] or len(order) != 3:
            raise ValueError("reference_axis_order must be a permutation of x/y/z")
        source_index = {axis: idx for idx, axis in enumerate(order)}
        values = values[:, [source_index["x"], source_index["y"], source_index["z"]]]
    flags = _reference_flip_flags(flips)
    if any(flags):
        lo = values.min(axis=0)
        hi = values.max(axis=0)
        for axis, enabled in enumerate(flags):
            if enabled:
                values[:, axis] = lo[axis] + hi[axis] - values[:, axis]
    return values


def surface_points_from_mask(
    mask_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    max_points: int | None = None,
    sample_mode: str = "linspace",
    sample_offset: int = 0,
) -> np.ndarray:
    mask = np.asarray(mask_zyx, dtype=bool)
    if not np.any(mask):
        raise ValueError("mask contains no foreground voxels")
    surface = mask & ~_binary_erosion_6(mask)
    coords_zyx = np.argwhere(surface if np.any(surface) else mask)
    coords_xyz = coords_zyx[:, [2, 1, 0]].astype(float)
    points = np.asarray(origin, dtype=float) + coords_xyz * np.asarray(spacing)
    return _sample_points(
        points,
        max_points=max_points,
        mode=sample_mode,
        offset=sample_offset,
    )


def estimate_rigid_icp(
    *,
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    iterations: int = 50,
    tolerance: float = 1.0e-4,
    start_by_matching_centroids_only: bool = False,
    convergence: str = "delta",
    distance_mode: str = "mean",
) -> dict[str, Any]:
    moving = _points_array(moving_points, "moving_points")
    fixed = _points_array(fixed_points, "fixed_points")
    if start_by_matching_centroids_only:
        rotation = np.eye(3)
        translation = fixed.mean(axis=0) - moving.mean(axis=0)
    else:
        rotation, translation = _best_initial_transform(moving, fixed)
    previous_error = np.inf
    used_iterations = 0
    convergence_token = str(convergence).strip().lower()
    distance_token = str(distance_mode).strip().lower()
    for used_iterations in range(1, max(1, int(iterations)) + 1):
        transformed = moving @ rotation.T + translation
        matched = fixed[_nearest_indices(transformed, fixed)]
        step_rotation, step_translation = _kabsch(transformed, matched)
        rotation = step_rotation @ rotation
        translation = step_rotation @ translation + step_translation
        distances = np.linalg.norm(transformed - matched, axis=1)
        if distance_token == "mean":
            error = float(np.mean(distances))
        elif distance_token == "rms":
            error = float(np.sqrt(np.mean(distances * distances)))
        else:
            raise ValueError("distance_mode must be 'mean' or 'rms'")
        if convergence_token == "delta":
            converged = abs(previous_error - error) <= float(tolerance)
        elif convergence_token in {"absolute", "abs"}:
            converged = error <= float(tolerance)
        else:
            raise ValueError("convergence must be 'delta' or 'absolute'")
        if converged:
            previous_error = error
            break
        previous_error = error
    return {
        "rotation": rotation,
        "translation": translation,
        "iterations": used_iterations,
        "mean_distance": previous_error,
    }


def _best_initial_transform(
    moving: np.ndarray, fixed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    candidates = [
        (np.eye(3), fixed_center - moving_center),
        _initial_pca_transform(moving, fixed),
    ]
    return min(candidates, key=lambda item: _nearest_mean_distance(moving, fixed, *item))


def _read_ascii_vtk_points(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    header_lines = raw.splitlines(keepends=True)[:6]
    header = b"".join(header_lines).decode("ascii", errors="ignore")
    if "BINARY" in header.upper():
        return _read_binary_vtk_points(raw, path)
    tokens = raw.decode("utf-8", errors="ignore").split()
    try:
        idx = tokens.index("POINTS")
    except ValueError as exc:
        raise ValueError(f"{path} does not contain an ASCII VTK POINTS block") from exc
    count = int(tokens[idx + 1])
    start = idx + 3
    values = np.asarray(tokens[start : start + count * 3], dtype=float)
    if values.size != count * 3:
        raise ValueError(f"{path} POINTS block is incomplete")
    return values.reshape((count, 3))


def _read_binary_vtk_points(raw: bytes, path: Path) -> np.ndarray:
    lines = raw.splitlines(keepends=True)
    offset = 0
    for line in lines:
        decoded = line.decode("ascii", errors="ignore").strip().split()
        offset += len(line)
        if len(decoded) >= 3 and decoded[0].upper() == "POINTS":
            count = int(decoded[1])
            dtype_token = decoded[2].lower()
            dtype = ">f4" if dtype_token == "float" else ">f8"
            values = np.frombuffer(raw, dtype=dtype, count=count * 3, offset=offset)
            if values.size != count * 3:
                raise ValueError(f"{path} binary POINTS block is incomplete")
            return values.astype(float).reshape((count, 3))
    raise ValueError(f"{path} does not contain a VTK POINTS block")


def _points_array(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if array.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three points")
    return array[np.all(np.isfinite(array), axis=1)]


def _sample_points(
    points: np.ndarray,
    *,
    max_points: int | None,
    mode: str = "linspace",
    offset: int = 0,
) -> np.ndarray:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    token = str(mode).strip().lower()
    if token == "stride":
        step = max(1, int(np.ceil(points.shape[0] / int(max_points))))
        start = int(offset) % max(step, 1)
        indices = np.arange(start, points.shape[0], step, dtype=int)
        if indices.size < int(max_points):
            fallback = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=int)
            indices = np.unique(np.concatenate([indices, fallback]))
        indices = indices[: int(max_points)]
    else:
        indices = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=int)
    return points[indices]


def _scale_reference_points_to_sample(
    *,
    moving_points: np.ndarray,
    reference_points: np.ndarray,
    registration_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    scaling_cfg = registration_config.get("reference_scaling", {})
    if scaling_cfg is False:
        return reference_points, {"enabled": False}
    if scaling_cfg is True:
        scaling_cfg = {}
    if not isinstance(scaling_cfg, dict):
        raise ValueError("registration.reference_scaling must be a mapping or boolean")
    enabled = bool(scaling_cfg) or bool(registration_config.get("scale_reference", False))
    method = str(registration_config.get("method", "")).strip().lower()
    if not enabled and method not in {"scaled_icp", "vtk_scaled_icp"}:
        return reference_points, {"enabled": False}

    sample_lengths = _principal_axis_lengths(moving_points)
    reference_lengths = _principal_axis_lengths(reference_points)
    factors = scaling_cfg.get("factors", registration_config.get("registration_scale"))
    if factors is None or str(factors).strip().lower() == "auto":
        scale_factors = sample_lengths / np.maximum(reference_lengths, 1.0e-12)
        min_factors = _scale_triplet(
            scaling_cfg.get(
                "min_factors",
                registration_config.get("registration_min_scale", (0.8, 0.8, 0.75)),
            ),
            name="registration.reference_scaling.min_factors",
        )
        max_factors = _scale_triplet(
            scaling_cfg.get(
                "max_factors",
                registration_config.get("registration_max_scale", (1.2, 1.2, 1.3)),
            ),
            name="registration.reference_scaling.max_factors",
        )
        scale_factors = np.minimum(scale_factors, max_factors)
        scale_factors = np.maximum(scale_factors, min_factors)
        source = "pca_axis_lengths"
    else:
        scale_factors = _scale_triplet(
            factors,
            name="registration.reference_scaling.factors",
        )
        min_factors = None
        max_factors = None
        source = "manual"

    center = reference_points.mean(axis=0)
    scaled = (reference_points - center) * scale_factors + center
    return scaled, {
        "enabled": True,
        "source": source,
        "sample_axis_lengths": sample_lengths.tolist(),
        "reference_axis_lengths": reference_lengths.tolist(),
        "scale_factors": scale_factors.tolist(),
        "min_factors": None if min_factors is None else min_factors.tolist(),
        "max_factors": None if max_factors is None else max_factors.tolist(),
    }


def _principal_axis_lengths(points: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, dtype=float) - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    return np.sqrt(eigvals) * 2.0


def _scale_triplet(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, str):
        tokens = [token for token in value.replace(",", " ").split() if token]
        if len(tokens) == 1:
            tokens = tokens * 3
        values = [float(token) for token in tokens]
    else:
        values = [float(token) for token in value]
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"{name} must contain one or three numeric values")
    return np.asarray(values, dtype=float)


def _mirror_points(points: np.ndarray, *, axis: str) -> np.ndarray:
    values = np.asarray(points, dtype=float).copy()
    axis_token = axis.strip().lower()
    if axis_token not in {"x", "y", "z"}:
        raise ValueError("registration.mirror_axis must be one of x, y, z")
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_token]
    lo = float(values[:, axis_index].min())
    hi = float(values[:, axis_index].max())
    values[:, axis_index] = lo + hi - values[:, axis_index]
    return values


def _binary_erosion_6(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    eroded = center.copy()
    eroded &= padded[:-2, 1:-1, 1:-1]
    eroded &= padded[2:, 1:-1, 1:-1]
    eroded &= padded[1:-1, :-2, 1:-1]
    eroded &= padded[1:-1, 2:, 1:-1]
    eroded &= padded[1:-1, 1:-1, :-2]
    eroded &= padded[1:-1, 1:-1, 2:]
    return eroded


def _initial_pca_transform(
    moving: np.ndarray, fixed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    moving_axes = _principal_axes(moving - moving_center)
    fixed_axes = _principal_axes(fixed - fixed_center)
    rotation = fixed_axes @ moving_axes.T
    if np.linalg.det(rotation) < 0:
        fixed_axes[:, -1] *= -1
        rotation = fixed_axes @ moving_axes.T
    translation = fixed_center - rotation @ moving_center
    return rotation, translation


def _principal_axes(points: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(points, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def _nearest_indices(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return _nearest_indices_numpy(query, target)
    return cKDTree(target).query(query, workers=-1)[1]


def _reference_flip_flags(flips: Any) -> tuple[bool, bool, bool]:
    if isinstance(flips, str):
        tokens = {token.strip().lower() for token in flips.replace(",", " ").split()}
        flags = tuple(axis in tokens for axis in ("x", "y", "z"))
    else:
        flags = tuple(bool(v) for v in flips)
    if len(flags) != 3:
        raise ValueError("reference_flips must contain three booleans or axis names")
    return flags


def _icp_centroid_start(registration_config: dict[str, Any]) -> bool:
    if "start_by_matching_centroids_only" in registration_config:
        return bool(registration_config["start_by_matching_centroids_only"])
    mode = str(registration_config.get("initialization", "")).strip().lower()
    method = str(registration_config.get("method", "")).strip().lower()
    return mode in {"centroid", "centroids", "vtk"} or method == "vtk_icp"


def _nearest_mean_distance(
    moving: np.ndarray,
    fixed: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    transformed = moving @ rotation.T + translation
    matched = fixed[_nearest_indices(transformed, fixed)]
    return float(np.mean(np.linalg.norm(transformed - matched, axis=1)))


def _nearest_indices_numpy(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.empty((query.shape[0],), dtype=int)
    chunk = 512
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        diff = query[start:stop, None, :] - target[None, :, :]
        out[start:stop] = np.argmin(np.sum(diff * diff, axis=2), axis=1)
    return out


def _kabsch(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    h = (moving - moving_center).T @ (fixed - fixed_center)
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = fixed_center - rotation @ moving_center
    return rotation, translation


def _output_grid_for_transform(
    points: np.ndarray,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    spacing: tuple[float, float, float],
    margin_voxels: int,
) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    transformed = points @ rotation.T + translation
    spacing_arr = np.asarray(spacing, dtype=float)
    lo = transformed.min(axis=0) - int(margin_voxels) * spacing_arr
    hi = transformed.max(axis=0) + int(margin_voxels) * spacing_arr
    size = np.maximum(1, np.ceil((hi - lo) / spacing_arr).astype(int) + 1)
    return tuple(float(v) for v in lo), tuple(int(v) for v in size)


def _resample_with_transform(
    array_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    output_spacing: tuple[float, float, float],
    output_origin: tuple[float, float, float],
    output_size: tuple[int, int, int],
    rotation: np.ndarray,
    translation: np.ndarray,
    interpolation: str,
) -> np.ndarray:
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetOrigin(tuple(float(v) for v in origin))
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(tuple(float(v) for v in rotation.reshape(-1)))
    transform.SetTranslation(tuple(float(v) for v in translation))
    inverse = transform.GetInverse()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize([int(v) for v in output_size])
    resampler.SetOutputSpacing(tuple(float(v) for v in output_spacing))
    resampler.SetOutputOrigin(tuple(float(v) for v in output_origin))
    resampler.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    resampler.SetTransform(inverse)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor
        if interpolation == "nearest"
        else sitk.sitkBSpline
        if interpolation == "bspline"
        else sitk.sitkLinear
    )
    resampler.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(resampler.Execute(image))
