from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ImageGridMetadata:
    shape_zyx: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_ras: tuple[float, float, float]
    direction_ras: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    array_order: str = "zyx"
    coordinate_system: str = "RAS"
    units: str = "mm"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ImageGridMetadata:
        direction = data.get(
            "direction_ras",
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        return cls(
            shape_zyx=_int_triple(data["shape_zyx"], "shape_zyx"),
            spacing_xyz=_float_triple(data["spacing_xyz"], "spacing_xyz"),
            origin_ras=_float_triple(data["origin_ras"], "origin_ras"),
            direction_ras=_direction_tuple(direction),
            array_order=str(data.get("array_order", "zyx")),
            coordinate_system=str(data.get("coordinate_system", "RAS")),
            units=str(data.get("units", "mm")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_zyx": list(self.shape_zyx),
            "spacing_xyz": list(self.spacing_xyz),
            "origin_ras": list(self.origin_ras),
            "direction_ras": [list(row) for row in self.direction_ras],
            "array_order": self.array_order,
            "coordinate_system": self.coordinate_system,
            "units": self.units,
        }


def voxel_indices_zyx_to_ras(
    indices_zyx: np.ndarray,
    grid: ImageGridMetadata,
) -> np.ndarray:
    indices = np.asarray(indices_zyx, dtype=float)
    if indices.ndim == 1:
        indices = indices.reshape(1, 3)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices_zyx must have shape (n, 3)")
    indices_xyz = indices[:, ::-1]
    scaled = indices_xyz * np.asarray(grid.spacing_xyz, dtype=float)
    direction = np.asarray(grid.direction_ras, dtype=float)
    return np.asarray(grid.origin_ras, dtype=float) + scaled @ direction.T


def ras_to_voxel_indices_zyx(
    points_ras: np.ndarray,
    grid: ImageGridMetadata,
) -> np.ndarray:
    points = np.asarray(points_ras, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ras must have shape (n, 3)")
    direction = np.asarray(grid.direction_ras, dtype=float)
    local = (points - np.asarray(grid.origin_ras, dtype=float)) @ direction
    indices_xyz = local / np.asarray(grid.spacing_xyz, dtype=float)
    return np.rint(indices_xyz[:, ::-1]).astype(np.int64)


def ras_affine_from_grid(grid: ImageGridMetadata) -> np.ndarray:
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = np.asarray(grid.direction_ras, dtype=float) @ np.diag(
        np.asarray(grid.spacing_xyz, dtype=float)
    )
    affine[:3, 3] = np.asarray(grid.origin_ras, dtype=float)
    return affine


def identity_transform_record(
    name: str,
    *,
    source_space: str,
    target_space: str,
    interpolation: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": str(name),
        "source_space": str(source_space),
        "target_space": str(target_space),
        "matrix": np.eye(4, dtype=float).tolist(),
    }
    if interpolation is not None:
        record["interpolation"] = str(interpolation)
    if note is not None:
        record["note"] = str(note)
    return record


def transform_record(
    name: str,
    *,
    source_space: str,
    target_space: str,
    matrix: np.ndarray,
    interpolation: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")
    record = identity_transform_record(
        name,
        source_space=source_space,
        target_space=target_space,
        interpolation=interpolation,
        note=note,
    )
    record["matrix"] = array.tolist()
    return record


def _float_triple(values: Any, name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values")
    return tuple(float(value) for value in values)


def _int_triple(values: Any, name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values")
    return tuple(int(value) for value in values)


def _direction_tuple(values: Any) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    rows = tuple(_float_triple(row, "direction_ras row") for row in values)
    if len(rows) != 3:
        raise ValueError("direction_ras must contain three rows")
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("direction_ras must have shape (3, 3)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("direction_ras must be finite")
    return rows  # type: ignore[return-value]
