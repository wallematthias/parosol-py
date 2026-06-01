from __future__ import annotations

from typing import Final

from .contract import (
    SolverSettings,
    VoxelElasticityBackend,
    VoxelElasticityProblem,
    VoxelElasticityResult,
)
from .elasticity import TorchVoxelElasticityBackend


_BACKENDS: Final[dict[str, VoxelElasticityBackend]] = {
    "torch-experimental": TorchVoxelElasticityBackend(),
}


def available_backends() -> tuple[str, ...]:
    """Return explicitly registered parosol_torch backend names."""

    return tuple(_BACKENDS)


def get_backend(name: str) -> VoxelElasticityBackend:
    """Return a registered torch backend without falling back to native ParOSol."""

    key = name.strip().lower()
    try:
        return _BACKENDS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown parosol_torch backend '{name}'. "
            f"Available backends: {', '.join(available_backends())}"
        ) from exc


def solve(
    problem: VoxelElasticityProblem,
    settings: SolverSettings | None = None,
    *,
    backend: str = "torch-experimental",
) -> VoxelElasticityResult:
    """Explicit experimental torch solve entry point.

    The default backend is experimental and remains separate from the validated
    native ParOSol solver. It is intended for development and small validation
    problems until reference parity is established.
    """

    return get_backend(backend).solve(problem, settings)
