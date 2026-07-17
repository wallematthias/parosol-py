from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VonMisesMaterial:
    youngs_modulus_mpa: float
    poisson_ratio: float
    yield_strength_mpa: float

    def __post_init__(self) -> None:
        if self.youngs_modulus_mpa <= 0.0:
            raise ValueError("youngs_modulus_mpa must be positive")
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must satisfy -1.0 < nu < 0.5")
        if self.yield_strength_mpa <= 0.0:
            raise ValueError("yield_strength_mpa must be positive")

    def to_hdf5_attrs(self) -> dict[str, float | str]:
        return {
            "type": "VonMisesIsotropic",
            "youngs_modulus_mpa": float(self.youngs_modulus_mpa),
            "poisson_ratio": float(self.poisson_ratio),
            "yield_strength_mpa": float(self.yield_strength_mpa),
        }


@dataclass(frozen=True)
class NonlinearSolverOptions:
    convergence_tolerance: float = 1.0e-6
    maximum_plastic_iterations: int = 50
    plastic_convergence_window: int = 2

    def __post_init__(self) -> None:
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive")
        if self.maximum_plastic_iterations <= 0:
            raise ValueError("maximum_plastic_iterations must be positive")
        if self.plastic_convergence_window <= 0:
            raise ValueError("plastic_convergence_window must be positive")
