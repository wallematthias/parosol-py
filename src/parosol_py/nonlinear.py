from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Any

import numpy as np


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


@dataclass(frozen=True)
class KeavenyNonlinearMaterialMap:
    youngs_modulus_mpa: np.ndarray
    poisson_ratio: float | np.ndarray
    compressive_yield_mpa: np.ndarray
    tensile_yield_mpa: np.ndarray
    plateau_mpa: np.ndarray
    material_id: np.ndarray
    metadata: dict[str, Any]

    def to_hdf5_attrs(self) -> dict[str, float | str]:
        return {
            "type": "AsymmetricPerfectPlasticDensityMap",
            "source": str(self.metadata["preset"]),
        }


def spine_keaveny_nonlinear(
    rho_qct,
    *,
    active_mask=None,
    poisson_ratio: float | np.ndarray = 0.3,
    bin_material: bool = False,
    number_bins: int = 128,
) -> KeavenyNonlinearMaterialMap:
    """Return the end-to-end vertebral nonlinear material map."""
    rho = _prepared_rho(rho_qct, active_mask=active_mask)
    rho_eval, bin_metadata = _maybe_bin_rho(
        rho,
        active=rho.active,
        bin_material=bin_material,
        number_bins=number_bins,
    )
    youngs = 3814.4 * np.power(np.maximum(rho_eval, 0.0), 1.05)
    plateau = 57.4464 * np.power(np.maximum(rho_eval, 0.0), 1.39)
    return _keaveny_map(
        youngs_mpa=youngs,
        compressive_yield_mpa=plateau,
        tensile_yield_mpa=plateau,
        plateau_mpa=plateau,
        poisson_ratio=poisson_ratio,
        active=rho.active,
        metadata={
            "preset": "spine_keaveny",
            "density_basis": "rho_qct",
            "side_multiplier": 1.28,
            "spine_tensile_policy": "tensile_yield_equals_compressive_plateau",
            **bin_metadata,
        },
    )


def hip_keaveny_nonlinear(
    rho_app,
    *,
    site: str = "femoral_neck",
    active_mask=None,
    poisson_ratio: float | np.ndarray = 0.3,
    bin_material: bool = False,
    number_bins: int = 128,
) -> KeavenyNonlinearMaterialMap:
    """Return the end-to-end hip nonlinear material map."""
    site_name = site.strip().lower().replace("-", "_")
    if site_name == "femoral_neck":
        coefficient = 8768.0
        exponent = 1.49
        compressive_strain = 0.0085
    elif site_name == "greater_trochanter":
        coefficient = 19212.8
        exponent = 2.18
        compressive_strain = 0.0070
    else:
        raise ValueError("site must be 'femoral_neck' or 'greater_trochanter'")
    tensile_strain = 0.0061
    rho = _prepared_rho(rho_app, active_mask=active_mask)
    rho_eval, bin_metadata = _maybe_bin_rho(
        rho,
        active=rho.active,
        bin_material=bin_material,
        number_bins=number_bins,
    )
    youngs = coefficient * np.power(np.maximum(rho_eval, 0.0), exponent)
    compressive = compressive_strain * youngs
    tensile = tensile_strain * youngs
    return _keaveny_map(
        youngs_mpa=youngs,
        compressive_yield_mpa=compressive,
        tensile_yield_mpa=tensile,
        plateau_mpa=compressive,
        poisson_ratio=poisson_ratio,
        active=rho.active,
        metadata={
            "preset": "hip_keaveny",
            "site": site_name,
            "density_basis": "rho_app",
            "side_multiplier": 1.28,
            "compressive_yield_strain": compressive_strain,
            "tensile_yield_strain": tensile_strain,
            **bin_metadata,
        },
    )


def _is_positive_integer(value: object) -> bool:
    return (
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and isfinite(value)
        and value > 0
    )


@dataclass(frozen=True)
class _PreparedRho:
    values: np.ndarray
    active: np.ndarray


def _prepared_rho(values, *, active_mask) -> _PreparedRho:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"rho input must be 3D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("rho input values must be finite")
    if active_mask is None:
        active = arr > 0.0
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != arr.shape:
            raise ValueError("active_mask must match rho input shape")
    return _PreparedRho(values=arr, active=active)


def _maybe_bin_rho(
    prepared: _PreparedRho,
    *,
    active: np.ndarray,
    bin_material: bool,
    number_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not bin_material:
        return prepared.values, {"bin_material": False}
    n_bins = int(number_bins)
    if n_bins <= 0:
        raise ValueError("number_bins must be positive")
    values = prepared.values
    sample = values[(values != 0.0) & active]
    if sample.size == 0:
        return values.copy(), {
            "bin_material": True,
            "number_bins": n_bins,
            "binning": "global_nonzero_active_density",
            "bin_edges": [],
            "bin_centers": [],
        }
    lo = float(np.min(sample))
    hi = float(np.max(sample))
    if np.isclose(lo, hi):
        centers = np.array([lo], dtype=np.float64)
        binned = np.where(active & (values != 0.0), lo, values)
        edges = np.array([lo, hi], dtype=np.float64)
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        indices = np.clip(np.digitize(values, edges, right=False) - 1, 0, n_bins - 1)
        binned = np.where(active & (values != 0.0), centers[indices], values)
    return binned, {
        "bin_material": True,
        "number_bins": n_bins,
        "binning": "global_nonzero_active_density",
        "bin_edges": [float(v) for v in edges],
        "bin_centers": [float(v) for v in centers],
    }


def _keaveny_map(
    *,
    youngs_mpa: np.ndarray,
    compressive_yield_mpa: np.ndarray,
    tensile_yield_mpa: np.ndarray,
    plateau_mpa: np.ndarray,
    poisson_ratio: float | np.ndarray,
    active: np.ndarray,
    metadata: dict[str, Any],
) -> KeavenyNonlinearMaterialMap:
    youngs = np.where(active, youngs_mpa, 0.0)
    compressive = np.where(active, compressive_yield_mpa, 0.0)
    tensile = np.where(active, tensile_yield_mpa, 0.0)
    plateau = np.where(active, plateau_mpa, 0.0)
    material_id = np.where(active, 1, 0).astype(np.uint16)
    return KeavenyNonlinearMaterialMap(
        youngs_modulus_mpa=youngs,
        poisson_ratio=poisson_ratio,
        compressive_yield_mpa=compressive,
        tensile_yield_mpa=tensile,
        plateau_mpa=plateau,
        material_id=material_id,
        metadata=metadata,
    )
