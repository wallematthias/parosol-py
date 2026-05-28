from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class ImageGrid:
    array_xyz: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]


def _triple(values: tuple[float, float, float] | list[float] | np.ndarray, name: str) -> tuple[float, float, float]:
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
    arr_zyx = np.asarray(to_output_order(grid.array_xyz, array_order="zyx"), dtype=np.float32)
    img = sitk.GetImageFromArray(arr_zyx, isVector=False)
    img.SetSpacing(grid.spacing)
    img.SetOrigin(grid.origin)
    sitk.WriteImage(img, str(out))
    return out
