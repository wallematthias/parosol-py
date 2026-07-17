from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Any

import numpy as np

from parosol_py.materials import density_to_material_map


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
    convergence_tolerance: float = 1.0e-4
    maximum_plastic_iterations: int = 150
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
class DensityNonlinearMaterialMap:
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


def spine_nonlinear(
    rho_qct,
    *,
    active_mask=None,
    poisson_ratio: float | np.ndarray = 0.3,
    bin_material: bool = False,
    number_bins: int = 128,
) -> DensityNonlinearMaterialMap:
    """Return the end-to-end vertebral nonlinear material map.

    ``rho_qct`` is calibrated QCT density in mg/cc. The material fields are
    evaluated with ``rho = rho_qct / 1000`` in g/cc:
    ``E = 3814.4 * rho**1.05`` and
    ``sigma_c = plateau = 57.4464 * rho**1.39``. The 1.28 side multiplier is
    already included in the modulus and stress coefficients; it is not applied
    to density. For this first implementation, ``sigma_t = sigma_c = plateau``
    because a separate tensile law is not published in the supplied material
    law.
    """
    rho = _prepared_rho(rho_qct, active_mask=active_mask)
    rho_eval, bin_metadata = _maybe_bin_rho(
        rho,
        active=rho.active,
        bin_material=bin_material,
        number_bins=number_bins,
    )
    rho_gcc = np.maximum(rho_eval, 0.0) / 1000.0
    youngs = 3814.4 * np.power(rho_gcc, 1.05)
    plateau = 57.4464 * np.power(rho_gcc, 1.39)
    return _nonlinear_map(
        youngs_mpa=youngs,
        compressive_yield_mpa=plateau,
        tensile_yield_mpa=plateau,
        plateau_mpa=plateau,
        poisson_ratio=poisson_ratio,
        active=rho.active,
        metadata={
            "preset": "spine_nonlinear",
            "anatomic_site": "spine",
            "constitutive_model": "asymmetric_perfect_plastic",
            "elastic_law": "3814.4 * rho_qct_gcc ** 1.05",
            "compressive_yield_law": "57.4464 * rho_qct_gcc ** 1.39",
            "tensile_yield_law": "57.4464 * rho_qct_gcc ** 1.39",
            "density_basis": "rho_qct_mgcc",
            "equation_density_units": "g/cc",
            "side_multiplier": 1.28,
            "spine_tensile_policy": "tensile_yield_equals_compressive_plateau",
            **bin_metadata,
        },
    )


def hip_nonlinear(
    rho_app,
    *,
    site: str = "femoral_neck",
    active_mask=None,
    poisson_ratio: float | np.ndarray = 0.3,
    bin_material: bool = False,
    number_bins: int = 128,
) -> DensityNonlinearMaterialMap:
    """Return the end-to-end hip nonlinear material map.

    ``rho_app`` is the apparent density used directly by the hip law. For the
    femoral neck, ``E = 8768.0 * rho_app**1.49`` and
    ``sigma_c = plateau = 0.0085 * E``. For the greater trochanter,
    ``E = 19212.8 * rho_app**2.18`` and
    ``sigma_c = plateau = 0.0070 * E``. Both sites use
    ``sigma_t = 0.0061 * E``. The 1.28 side multiplier is already included in
    the modulus coefficients; it is not applied to density.
    """
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
    return _nonlinear_map(
        youngs_mpa=youngs,
        compressive_yield_mpa=compressive,
        tensile_yield_mpa=tensile,
        plateau_mpa=compressive,
        poisson_ratio=poisson_ratio,
        active=rho.active,
        metadata={
            "preset": "hip_nonlinear",
            "anatomic_site": "hip",
            "constitutive_model": "asymmetric_perfect_plastic",
            "elastic_law": f"{coefficient} * rho_app ** {exponent}",
            "compressive_yield_law": f"{compressive_strain} * E",
            "tensile_yield_law": f"{tensile_strain} * E",
            "site": site_name,
            "density_basis": "rho_app",
            "side_multiplier": 1.28,
            "compressive_yield_strain": compressive_strain,
            "tensile_yield_strain": tensile_strain,
            **bin_metadata,
        },
    )


def manual_nonlinear(
    density,
    *,
    elastic: dict[str, Any],
    compressive_yield: dict[str, Any],
    tensile_yield: dict[str, Any],
    plateau: dict[str, Any] | None = None,
    active_mask=None,
    poisson_ratio: float | np.ndarray = 0.3,
    bin_material: bool = False,
    number_bins: int = 128,
) -> DensityNonlinearMaterialMap:
    """Return a user-defined density-based asymmetric plastic material map.

    ``elastic``, ``compressive_yield``, ``tensile_yield``, and optional
    ``plateau`` use the same density-equation schema as linear density
    materials: for example ``{"equation": "power", "coefficient": 5000,
    "exponent": 1, "reference_density": 1000}``. If ``plateau`` is omitted,
    the compressive yield stress is used as the plateau stress.
    """
    rho = _prepared_rho(density, active_mask=active_mask)
    rho_eval, bin_metadata = _maybe_bin_rho(
        rho,
        active=rho.active,
        bin_material=bin_material,
        number_bins=number_bins,
    )
    youngs = _manual_density_law(rho_eval, elastic, active=rho.active)
    compressive = _manual_density_law(rho_eval, compressive_yield, active=rho.active)
    tensile = _manual_density_law(rho_eval, tensile_yield, active=rho.active)
    plateau_values = (
        compressive
        if plateau is None
        else _manual_density_law(rho_eval, plateau, active=rho.active)
    )
    return _nonlinear_map(
        youngs_mpa=youngs,
        compressive_yield_mpa=compressive,
        tensile_yield_mpa=tensile,
        plateau_mpa=plateau_values,
        poisson_ratio=poisson_ratio,
        active=rho.active,
        metadata={
            "preset": "manual",
            "constitutive_model": "asymmetric_perfect_plastic",
            "elastic_law": _manual_law_description(elastic),
            "compressive_yield_law": _manual_law_description(compressive_yield),
            "tensile_yield_law": _manual_law_description(tensile_yield),
            "plateau_law": (
                "compressive_yield"
                if plateau is None
                else _manual_law_description(plateau)
            ),
            **bin_metadata,
        },
    )


def _manual_density_law(
    density: np.ndarray,
    spec: dict[str, Any],
    *,
    active: np.ndarray,
) -> np.ndarray:
    if not isinstance(spec, dict):
        raise ValueError("manual nonlinear material laws must be objects")
    equation = str(spec.get("equation", "power"))
    mapped = density_to_material_map(
        density,
        equation=equation,
        poisson_ratio=0.3,
        active_mask=active,
        minimum_e_mpa=_manual_law_floor(spec),
        **{
            key: value
            for key, value in spec.items()
            if key
            not in {
                "equation",
                "minimum_e_mpa",
                "floor_e_mpa",
                "floor_mpa",
                "floor",
            }
        },
    )
    return np.asarray(mapped.youngs_modulus_mpa, dtype=np.float64)


def _manual_law_floor(spec: dict[str, Any]) -> float | None:
    for key in ("minimum_e_mpa", "floor_e_mpa", "floor_mpa", "floor"):
        if spec.get(key) is not None:
            return float(spec[key])
    return 0.0


def _manual_law_description(spec: dict[str, Any]) -> str:
    equation = str(spec.get("equation", "power"))
    if equation.strip().lower() == "power":
        coefficient = float(spec.get("coefficient", spec.get("e_max", 10000.0)))
        exponent = float(spec.get("exponent", 1.7))
        reference = float(spec.get("reference_density", spec.get("rho_max", 1.0)))
        return f"{coefficient:g} * (density / {reference:g}) ** {exponent:g}"
    if equation.strip().lower() == "linear":
        slope = float(spec.get("slope", spec.get("a", 1.0)))
        intercept = float(spec.get("intercept", spec.get("b", 0.0)))
        return f"{slope:g} * density + {intercept:g}"
    return equation


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


def _nonlinear_map(
    *,
    youngs_mpa: np.ndarray,
    compressive_yield_mpa: np.ndarray,
    tensile_yield_mpa: np.ndarray,
    plateau_mpa: np.ndarray,
    poisson_ratio: float | np.ndarray,
    active: np.ndarray,
    metadata: dict[str, Any],
) -> DensityNonlinearMaterialMap:
    youngs = np.where(active, youngs_mpa, 0.0)
    compressive = np.where(active, compressive_yield_mpa, 0.0)
    tensile = np.where(active, tensile_yield_mpa, 0.0)
    plateau = np.where(active, plateau_mpa, 0.0)
    material_id = np.where(active, 1, 0).astype(np.uint16)
    return DensityNonlinearMaterialMap(
        youngs_modulus_mpa=youngs,
        poisson_ratio=poisson_ratio,
        compressive_yield_mpa=compressive,
        tensile_yield_mpa=tensile,
        plateau_mpa=plateau,
        material_id=material_id,
        metadata=metadata,
    )
