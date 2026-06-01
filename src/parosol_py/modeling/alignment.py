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
    reference_points = read_reference_points(reference_path, max_points=max_points)
    moving_points = surface_points_from_mask(
        mask_zyx,
        spacing=spacing,
        origin=origin,
        max_points=max_points,
    )
    transform = estimate_rigid_icp(
        moving_points=moving_points,
        fixed_points=reference_points,
        iterations=iterations,
        tolerance=tolerance,
    )
    output_origin, output_size = _output_grid_for_transform(
        moving_points,
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
        interpolation="linear",
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
            "method": "lightweight_icp",
            "reference_points": str(reference_path),
            "iterations": transform["iterations"],
            "mean_distance": transform["mean_distance"],
            "rotation": transform["rotation"].tolist(),
            "translation": transform["translation"].tolist(),
            "output_size_xyz": list(output_size),
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
    reference_points = read_reference_points(reference_path, max_points=max_points)
    body_points = surface_points_from_mask(
        body_mask_zyx,
        spacing=spacing,
        origin=origin,
        max_points=max_points,
    )
    transform = estimate_rigid_icp(
        moving_points=body_points,
        fixed_points=reference_points,
        iterations=iterations,
        tolerance=tolerance,
    )
    active_points = surface_points_from_mask(
        body_mask_zyx | process_mask_zyx,
        spacing=spacing,
        origin=origin,
        max_points=max_points,
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
        interpolation="linear",
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
            "method": "lightweight_icp",
            "reference_points": str(reference_path),
            "iterations": transform["iterations"],
            "mean_distance": transform["mean_distance"],
            "rotation": transform["rotation"].tolist(),
            "translation": transform["translation"].tolist(),
            "output_size_xyz": list(output_size),
        },
    )


def read_reference_points(path: str | Path, *, max_points: int | None = None) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    if "".join(path.suffixes).lower().endswith(".npz"):
        with np.load(path) as data:
            key = "points" if "points" in data else data.files[0]
            points = np.asarray(data[key], dtype=float)
    elif path.suffix.lower() == ".vtk":
        points = _read_ascii_vtk_points(path)
    else:
        points = np.loadtxt(path, dtype=float)
    points = _points_array(points, "reference points")
    return _sample_points(points, max_points=max_points)


def surface_points_from_mask(
    mask_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    max_points: int | None = None,
) -> np.ndarray:
    mask = np.asarray(mask_zyx, dtype=bool)
    if not np.any(mask):
        raise ValueError("mask contains no foreground voxels")
    surface = mask & ~_binary_erosion_6(mask)
    coords_zyx = np.argwhere(surface if np.any(surface) else mask)
    coords_xyz = coords_zyx[:, [2, 1, 0]].astype(float)
    points = np.asarray(origin, dtype=float) + coords_xyz * np.asarray(spacing)
    return _sample_points(points, max_points=max_points)


def estimate_rigid_icp(
    *,
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    iterations: int = 50,
    tolerance: float = 1.0e-4,
) -> dict[str, Any]:
    moving = _points_array(moving_points, "moving_points")
    fixed = _points_array(fixed_points, "fixed_points")
    rotation, translation = _initial_pca_transform(moving, fixed)
    previous_error = np.inf
    used_iterations = 0
    for used_iterations in range(1, max(1, int(iterations)) + 1):
        transformed = moving @ rotation.T + translation
        matched = fixed[_nearest_indices(transformed, fixed)]
        step_rotation, step_translation = _kabsch(transformed, matched)
        rotation = step_rotation @ rotation
        translation = step_rotation @ translation + step_translation
        error = float(np.mean(np.linalg.norm(transformed - matched, axis=1)))
        if abs(previous_error - error) <= float(tolerance):
            previous_error = error
            break
        previous_error = error
    return {
        "rotation": rotation,
        "translation": translation,
        "iterations": used_iterations,
        "mean_distance": previous_error,
    }


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


def _sample_points(points: np.ndarray, *, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=int)
    return points[indices]


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
        sitk.sitkNearestNeighbor if interpolation == "nearest" else sitk.sitkLinear
    )
    resampler.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(resampler.Execute(image))
