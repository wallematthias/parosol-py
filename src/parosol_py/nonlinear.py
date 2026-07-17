from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral


@dataclass(frozen=True)
class VonMisesMaterial:
    youngs_modulus_mpa: float
    poisson_ratio: float
    yield_strength_mpa: float

    def __post_init__(self) -> None:
        if not isfinite(self.youngs_modulus_mpa) or self.youngs_modulus_mpa <= 0.0:
            raise ValueError("youngs_modulus_mpa must be finite and positive")
        if not isfinite(self.poisson_ratio) or not (
            -1.0 < self.poisson_ratio < 0.5
        ):
            raise ValueError("poisson_ratio must be finite and satisfy -1.0 < nu < 0.5")
        if not isfinite(self.yield_strength_mpa) or self.yield_strength_mpa <= 0.0:
            raise ValueError("yield_strength_mpa must be finite and positive")

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
        if (
            not isfinite(self.convergence_tolerance)
            or self.convergence_tolerance <= 0.0
        ):
            raise ValueError("convergence_tolerance must be finite and positive")
        if not _is_positive_integer(self.maximum_plastic_iterations):
            raise ValueError(
                "maximum_plastic_iterations must be a finite positive integer"
            )
        if not _is_positive_integer(self.plastic_convergence_window):
            raise ValueError(
                "plastic_convergence_window must be a finite positive integer"
            )


def _is_positive_integer(value: object) -> bool:
    return (
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and isfinite(value)
        and value > 0
    )
