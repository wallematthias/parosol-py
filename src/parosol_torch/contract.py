from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class BackendStatus(str, Enum):
    """Lifecycle state for a parosol_torch backend."""

    NOT_IMPLEMENTED = "not_implemented"
    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"


@dataclass(frozen=True)
class SolverSettings:
    """Backend-independent solve controls for future voxel elasticity solvers."""

    tolerance: float = 1e-6
    max_iterations: int | None = None
    device: str | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class VoxelElasticityProblem:
    """Explicit input contract for a future voxel elasticity backend.

    Coordinates use ParOSol's xyz array convention. Displacement coordinates are
    expected to be integer rows of ``(x, y, z, component)`` with component in
    ``0..2``. This is only a data contract; it does not imply that a torch
    elasticity solve is implemented.
    """

    stiffness_gpa_xyz: np.ndarray
    voxel_size_mm: float
    poisson_ratio: float
    fixed_displacement_coordinates: np.ndarray
    fixed_displacement_values: np.ndarray
    loaded_node_coordinates: np.ndarray | None = None
    loaded_node_values: np.ndarray | None = None
    requested_outputs: tuple[str, ...] = ("forces", "displacements")

    @property
    def dimensions_xyz(self) -> tuple[int, int, int]:
        """Return the voxel grid shape in xyz order."""

        shape = np.asarray(self.stiffness_gpa_xyz).shape
        if len(shape) != 3:
            raise ValueError("stiffness_gpa_xyz must be a 3D xyz array")
        return tuple(int(value) for value in shape)


@dataclass(frozen=True)
class VoxelElasticityResult:
    """Output contract for future torch-backed voxel elasticity results."""

    fields: dict[str, Any]
    diagnostics: dict[str, Any]
    converged: bool
    iterations: int | None = None
    residual_norm: float | None = None


class VoxelElasticityBackend(Protocol):
    """Small protocol implemented by future parosol_torch solver backends."""

    name: str
    status: BackendStatus

    def solve(
        self,
        problem: VoxelElasticityProblem,
        settings: SolverSettings | None = None,
    ) -> VoxelElasticityResult:
        """Solve a voxel elasticity problem."""
