from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .contract import (
    BackendStatus,
    SolverSettings,
    VoxelElasticityBackend,
    VoxelElasticityProblem,
    VoxelElasticityResult,
)


@dataclass(frozen=True)
class NotImplementedVoxelElasticityBackend:
    """Placeholder backend that documents the contract without solving FEA."""

    name: str = "torch-experimental"
    status: BackendStatus = BackendStatus.NOT_IMPLEMENTED

    def solve(
        self,
        problem: VoxelElasticityProblem,
        settings: SolverSettings | None = None,
    ) -> VoxelElasticityResult:
        """Fail clearly until the torch elasticity backend is validated."""

        _ = problem, settings
        raise NotImplementedError(
            "parosol_torch is not a validated ParOSol solver yet. "
            "It defines the future backend contract and prototype operators, "
            "but it must not be used for production FEA."
        )


_BACKENDS: Final[dict[str, VoxelElasticityBackend]] = {
    "torch-experimental": NotImplementedVoxelElasticityBackend(),
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

    The default backend is intentionally a ``NotImplemented`` backend, not a
    production solver. This keeps the namespace separate while making accidental
    use fail with a precise explanation.
    """

    return get_backend(backend).solve(problem, settings)
