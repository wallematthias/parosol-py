from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .images import normalize_array


@dataclass(frozen=True)
class Model:
    material_xyz: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material_unit: str = "MPa"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_array(
        cls,
        material,
        *,
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        array_order: str = "zyx",
        material_unit: str = "MPa",
        metadata: dict[str, Any] | None = None,
    ) -> "Model":
        grid = normalize_array(
            material,
            spacing=spacing,
            origin=origin,
            array_order=array_order,
        )
        return cls(
            material_xyz=grid.array_xyz,
            spacing=grid.spacing,
            origin=grid.origin,
            material_unit=material_unit,
            metadata={} if metadata is None else dict(metadata),
        )


@dataclass(frozen=True)
class BoundaryConditionSet:
    fixed_coordinates: np.ndarray
    fixed_values: np.ndarray
    loaded_coordinates: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.uint16)
    )
    loaded_values: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    node_sets: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixed_coordinates",
            _coordinates(self.fixed_coordinates, "fixed_coordinates"),
        )
        object.__setattr__(self, "fixed_values", _values(self.fixed_values, "fixed_values"))
        object.__setattr__(
            self,
            "loaded_coordinates",
            _coordinates(self.loaded_coordinates, "loaded_coordinates"),
        )
        object.__setattr__(
            self,
            "loaded_values",
            _values(self.loaded_values, "loaded_values"),
        )
        if self.fixed_coordinates.shape[0] != self.fixed_values.shape[0]:
            raise ValueError("fixed coordinate/value counts differ")
        if self.loaded_coordinates.shape[0] != self.loaded_values.shape[0]:
            raise ValueError("loaded coordinate/value counts differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed_coordinates": self.fixed_coordinates.tolist(),
            "fixed_values": self.fixed_values.tolist(),
            "loaded_coordinates": self.loaded_coordinates.tolist(),
            "loaded_values": self.loaded_values.tolist(),
            "node_sets": {
                name: [list(coord) for coord in coords]
                for name, coords in self.node_sets.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundaryConditionSet":
        return cls(
            fixed_coordinates=np.asarray(data["fixed_coordinates"], dtype=np.uint16),
            fixed_values=np.asarray(data["fixed_values"], dtype=np.float32),
            loaded_coordinates=np.asarray(
                data.get("loaded_coordinates", []), dtype=np.uint16
            ).reshape((-1, 4)),
            loaded_values=np.asarray(data.get("loaded_values", []), dtype=np.float32),
            node_sets={
                name: [tuple(int(v) for v in coord) for coord in coords]
                for name, coords in data.get("node_sets", {}).items()
            },
        )


@dataclass(frozen=True)
class SolverProfile:
    tolerance: float = 1e-6
    level: int = 6
    mpi_processes: int = 1
    mpi_launcher: str = "mpirun"
    outputs: tuple[str, ...] = ("sed",)


@dataclass(frozen=True)
class OutputProfile:
    export_fields: bool = True
    image_fields: tuple[str, ...] = ("sed",)
    summary_name: str = "summary.json"
    retain_hdf5: bool = True


def _coordinates(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.uint16)
    if array.size == 0:
        return array.reshape((0, 4))
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (n, 4)")
    return array


def _values(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array
