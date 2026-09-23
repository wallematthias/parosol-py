from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque

import numpy as np
import SimpleITK as sitk

SLICER_RAS_TO_SITK_LPS_DIRECTION = (
    -1.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


@dataclass(frozen=True)
class ImageGrid:
    array_xyz: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]


def _triple(
    values: tuple[float, float, float] | list[float] | np.ndarray, name: str
) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return tuple(float(v) for v in values)


def normalize_array(
    array,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    array_order: str = "zyx",
) -> ImageGrid:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"array must be 3D, got shape {arr.shape}")
    order = array_order.strip().lower()
    if order == "zyx":
        arr_xyz = np.transpose(arr, (2, 1, 0))
    elif order == "xyz":
        arr_xyz = arr
    else:
        raise ValueError("array_order must be 'zyx' or 'xyz'")
    return ImageGrid(
        array_xyz=np.ascontiguousarray(arr_xyz),
        spacing=_triple(spacing, "spacing"),
        origin=_triple(origin, "origin"),
    )


def to_output_order(array_xyz: np.ndarray, *, array_order: str = "zyx") -> np.ndarray:
    order = array_order.strip().lower()
    if order == "zyx":
        return np.ascontiguousarray(np.transpose(array_xyz, (2, 1, 0)))
    if order == "xyz":
        return np.ascontiguousarray(array_xyz)
    raise ValueError("array_order must be 'zyx' or 'xyz'")


def export_scalar_image(grid: ImageGrid, output_path: str | Path) -> Path:
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    arr_zyx = np.asarray(
        to_output_order(grid.array_xyz, array_order="zyx"), dtype=np.float32
    )
    img = sitk.GetImageFromArray(arr_zyx, isVector=False)
    img.SetSpacing(grid.spacing)
    img.SetOrigin(_ras_origin_to_sitk_lps(grid.origin))
    img.SetDirection(SLICER_RAS_TO_SITK_LPS_DIRECTION)
    sitk.WriteImage(img, str(out))
    return out


def restore_scalar_image_to_reference_grid(
    field_path: str | Path,
    reference_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Write a scalar field on the reference image grid, preserving crop position."""
    field = sitk.ReadImage(str(Path(field_path).expanduser().resolve()))
    reference = _read_reference_grid_image(reference_path)
    if field.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("field image must be scalar")
    if reference.GetDimension() != 3 or field.GetDimension() != 3:
        raise ValueError("field and reference images must be three-dimensional")

    same_grid = (
        field.GetSize() == reference.GetSize()
        and np.allclose(field.GetSpacing(), reference.GetSpacing(), rtol=1e-6, atol=1e-7)
        and np.allclose(field.GetOrigin(), reference.GetOrigin(), rtol=1e-6, atol=1e-6)
        and np.allclose(field.GetDirection(), reference.GetDirection(), rtol=1e-6, atol=1e-7)
    )
    if same_grid:
        restored = field
    else:
        continuous_start = np.asarray(
            reference.TransformPhysicalPointToContinuousIndex(field.GetOrigin()),
            dtype=float,
        )
        start = np.rint(continuous_start).astype(np.int64)
        end = start + np.asarray(field.GetSize(), dtype=np.int64)
        exact_crop = (
            np.allclose(field.GetSpacing(), reference.GetSpacing(), rtol=1e-6, atol=1e-7)
            and np.allclose(field.GetDirection(), reference.GetDirection(), rtol=1e-6, atol=1e-7)
            and np.allclose(continuous_start, start, rtol=0.0, atol=1e-4)
            and np.all(start >= 0)
            and np.all(end <= np.asarray(reference.GetSize(), dtype=np.int64))
        )
        if exact_crop:
            restored = sitk.Image(reference.GetSize(), field.GetPixelID())
            restored.CopyInformation(reference)
            restored = sitk.Paste(
                restored,
                field,
                field.GetSize(),
                (0, 0, 0),
                tuple(int(value) for value in start),
            )
        else:
            restored = sitk.Resample(
                field,
                reference,
                sitk.Transform(3, sitk.sitkIdentity),
                sitk.sitkLinear,
                0.0,
                field.GetPixelID(),
            )

    out = Path(output_path or field_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(restored, str(out))
    return out


def _read_reference_grid_image(reference_path: str | Path) -> sitk.Image:
    path = Path(reference_path).expanduser().resolve()
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError:
        from .modeling.io import read_image_zyx

        array_zyx, spacing, origin_ras = read_image_zyx(path)
        reference = sitk.Image(
            tuple(int(value) for value in reversed(array_zyx.shape)),
            sitk.sitkUInt8,
        )
        reference.SetSpacing(spacing)
        reference.SetOrigin(_ras_origin_to_sitk_lps(origin_ras))
        reference.SetDirection(SLICER_RAS_TO_SITK_LPS_DIRECTION)
        return reference


def _ras_origin_to_sitk_lps(
    origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (-float(origin[0]), -float(origin[1]), float(origin[2]))


def largest_connected_component(array, *, background: int | float = 0):
    values = np.asarray(array)
    mask = values != background
    if mask.ndim != 3:
        raise ValueError(f"array must be 3D, got shape {mask.shape}")
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int, int]] = []
    dims = tuple(int(v) for v in mask.shape)
    for start_array in np.argwhere(mask):
        start = tuple(int(v) for v in start_array)
        if visited[start]:
            continue
        component = _component(mask, visited, start, dims)
        if len(component) > len(best):
            best = component
    out = np.zeros(values.shape, dtype=values.dtype)
    if best:
        coords = tuple(np.asarray(best, dtype=np.int64).T)
        out[coords] = values[coords]
    return out


def coarsen_array(array, *, factor: int, reducer: str = "mean") -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 3:
        raise ValueError(f"array must be 3D, got shape {values.shape}")
    factor = int(factor)
    if factor < 1:
        raise ValueError("coarsen factor must be >= 1")
    if factor == 1:
        return np.array(values, copy=True)
    cropped_shape = tuple((size // factor) * factor for size in values.shape)
    if any(size == 0 for size in cropped_shape):
        raise ValueError("coarsen factor is larger than at least one image dimension")
    cropped = values[tuple(slice(0, size) for size in cropped_shape)]
    blocks = cropped.reshape(
        cropped_shape[0] // factor,
        factor,
        cropped_shape[1] // factor,
        factor,
        cropped_shape[2] // factor,
        factor,
    )
    token = reducer.strip().lower()
    if token == "mean":
        return blocks.mean(axis=(1, 3, 5))
    if token == "max":
        return blocks.max(axis=(1, 3, 5))
    if token == "min":
        return blocks.min(axis=(1, 3, 5))
    if token in {"nearest", "first"}:
        return blocks[:, 0, :, 0, :, 0]
    raise ValueError("coarsen reducer must be mean, max, min, or nearest")


def _component(
    mask: np.ndarray,
    visited: np.ndarray,
    start: tuple[int, int, int],
    dims: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    queue: deque[tuple[int, int, int]] = deque([start])
    visited[start] = True
    out: list[tuple[int, int, int]] = []
    while queue:
        coord = queue.popleft()
        out.append(coord)
        for axis in range(3):
            for offset in (-1, 1):
                neighbor = list(coord)
                neighbor[axis] += offset
                if neighbor[axis] < 0 or neighbor[axis] >= dims[axis]:
                    continue
                token = tuple(neighbor)
                if not visited[token] and mask[token]:
                    visited[token] = True
                    queue.append(token)
    return out
