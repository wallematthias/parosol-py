from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LinearIsotropicMaterials:
    youngs_modulus_mpa: dict[int, float]
    poisson_ratio: dict[int, float]


@dataclass(frozen=True)
class MaterialMap:
    youngs_modulus_mpa: np.ndarray
    poisson_ratio: float | np.ndarray
    metadata: dict[str, Any]


def material_to_stiffness_gpa(material, *, material_unit: str = "MPa") -> np.ndarray:
    arr = np.asarray(material, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"material must be 3D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("material values must be finite")
    if np.any(arr < 0.0):
        raise ValueError("material values must be non-negative")
    unit = material_unit.strip().lower()
    if unit == "mpa":
        out = arr / 1000.0
    elif unit == "gpa":
        out = arr
    else:
        raise ValueError("material_unit must be 'MPa' or 'GPa'")
    return np.ascontiguousarray(out.astype(np.float32, copy=False))


def labels_to_material_map(
    labels,
    table: LinearIsotropicMaterials,
    *,
    poisson_ratio: float | None = None,
) -> MaterialMap:
    label_array = np.asarray(labels)
    youngs = np.zeros(label_array.shape, dtype=np.float64)
    for label, value in table.youngs_modulus_mpa.items():
        youngs[label_array == label] = float(value)
    nu = _poisson_ratio_map(label_array, table, override=poisson_ratio)
    return MaterialMap(
        youngs_modulus_mpa=youngs,
        poisson_ratio=nu,
        metadata={
            "source": "labels",
            "labels": sorted(int(v) for v in table.youngs_modulus_mpa),
        },
    )


def density_to_material_map(
    density,
    *,
    equation: str = "power",
    poisson_ratio: float | dict[str, Any] = 0.3,
    mask_threshold: float = 0.0,
    active_threshold: float | None = None,
    active_mask=None,
    minimum_e_mpa: float | None = None,
    maximum_e_mpa: float | None = None,
    **parameters: Any,
) -> MaterialMap:
    density_array = np.asarray(density, dtype=np.float64)
    if density_array.ndim != 3:
        raise ValueError(f"density must be 3D, got shape {density_array.shape}")
    if not np.all(np.isfinite(density_array)):
        raise ValueError("density values must be finite")

    threshold = float(mask_threshold if active_threshold is None else active_threshold)
    active = _density_active_mask(
        density_array,
        active_mask=active_mask,
        threshold=threshold,
        combine_with_threshold=bool(
            parameters.get(
                "combine_active_mask_with_threshold",
                parameters.get("mask_and_threshold", False),
            )
        ),
    )

    equation_name = equation.strip().lower()
    if equation_name in {"power", "homminga"}:
        coefficient = float(
            parameters.get("coefficient", parameters.get("e_max", 10000.0))
        )
        exponent = float(parameters.get("exponent", 1.7))
        reference = float(
            parameters.get("reference_density", parameters.get("rho_max", 1.0))
        )
        if np.isclose(reference, 0.0):
            raise ValueError("reference_density must be non-zero")
        youngs = coefficient * np.power(
            np.maximum(density_array, 0.0) / reference, exponent
        )
        default_floor = 0.0
    elif equation_name in {
        "mulder",
        "mulder2007",
        "mulder_2007",
        "framework_mulder",
        "framework_mulder2007",
    }:
        equation_name = "mulder2007"
        slope = float(parameters.get("slope", parameters.get("a", 25.0)))
        intercept = float(parameters.get("intercept", parameters.get("b", -5830.0)))
        youngs = slope * density_array + intercept
        default_floor = 2.0
    elif equation_name == "linear":
        slope = float(parameters.get("slope", parameters.get("a", 1.0)))
        intercept = float(parameters.get("intercept", parameters.get("b", 0.0)))
        youngs = slope * density_array + intercept
        default_floor = 0.0
    elif equation_name == "polynomial":
        coefficients = parameters.get("coefficients")
        if coefficients is None:
            raise ValueError("polynomial density mapping requires coefficients")
        youngs = np.zeros(density_array.shape, dtype=np.float64)
        for power, coefficient in enumerate(coefficients):
            youngs += float(coefficient) * np.power(density_array, power)
        default_floor = 0.0
    else:
        raise ValueError(
            "density equation must be one of: power, homminga, mulder2007, linear, polynomial"
        )

    floor_e_mpa = _density_floor_e_mpa(
        parameters, minimum_e_mpa=minimum_e_mpa, default=default_floor
    )
    youngs = np.where(active, youngs, 0.0)
    youngs = np.where(active, np.maximum(youngs, floor_e_mpa), 0.0)
    if maximum_e_mpa is not None:
        youngs = np.where(active, np.minimum(youngs, float(maximum_e_mpa)), 0.0)

    nu = poisson_ratio_from_spec(poisson_ratio, density_array, active_mask=youngs > 0.0)
    return MaterialMap(
        youngs_modulus_mpa=youngs,
        poisson_ratio=nu,
        metadata={
            "source": "density",
            "equation": equation_name,
            "mask_threshold": threshold,
            "floor_e_mpa": floor_e_mpa,
            "active_source": "mask" if active_mask is not None else "density_threshold",
        },
    )


def _density_active_mask(
    density_array: np.ndarray,
    *,
    active_mask,
    threshold: float,
    combine_with_threshold: bool,
) -> np.ndarray:
    threshold_mask = density_array > float(threshold)
    if active_mask is None:
        return threshold_mask
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != density_array.shape:
        raise ValueError("active_mask must match density shape")
    if combine_with_threshold:
        return active & threshold_mask
    return active


def _density_floor_e_mpa(
    parameters: dict[str, Any],
    *,
    minimum_e_mpa: float | None,
    default: float,
) -> float:
    if minimum_e_mpa is not None:
        return float(minimum_e_mpa)
    for key in ("floor_e_mpa", "floor_mpa", "floor", "minimum_e_mpa"):
        if parameters.get(key) is not None:
            return float(parameters[key])
    return float(default)


def poisson_ratio_from_spec(
    spec: float | dict[str, Any],
    values,
    *,
    active_mask=None,
) -> float:
    if not isinstance(spec, dict):
        return float(spec)
    value_array = np.asarray(values, dtype=np.float64)
    mask = (
        np.asarray(active_mask) if active_mask is not None else np.isfinite(value_array)
    )
    if mask.shape != value_array.shape:
        raise ValueError("active_mask must match values shape")
    equation = str(spec.get("equation", spec.get("type", "constant"))).strip().lower()
    if equation == "constant":
        field = np.full(
            value_array.shape, float(spec.get("value", spec.get("nu", 0.3)))
        )
    elif equation == "linear":
        field = float(spec.get("slope", 0.0)) * value_array + float(
            spec.get("intercept", spec.get("value", 0.3))
        )
    elif equation == "power":
        coefficient = float(spec.get("coefficient", 0.3))
        exponent = float(spec.get("exponent", 0.0))
        reference = float(spec.get("reference", spec.get("reference_density", 1.0)))
        if np.isclose(reference, 0.0):
            raise ValueError("poisson reference density must be non-zero")
        field = coefficient * np.power(
            np.maximum(value_array, 0.0) / reference, exponent
        )
    else:
        raise ValueError("poisson_ratio equation must be constant, linear, or power")

    active = field[mask & np.isfinite(field)]
    if active.size == 0:
        return float(spec.get("fallback", 0.3))
    reducer = str(spec.get("reduce", "mean")).strip().lower()
    if reducer in {"mean", "volume_weighted_mean"}:
        out = float(np.mean(active))
    elif reducer == "median":
        out = float(np.median(active))
    elif reducer == "min":
        out = float(np.min(active))
    elif reducer == "max":
        out = float(np.max(active))
    else:
        raise ValueError("poisson_ratio.reduce must be mean, median, min, or max")
    if not 0.0 <= out < 0.5:
        raise ValueError(f"reduced poisson_ratio must be in [0, 0.5), got {out}")
    return out


def parse_linear_isotropic_materials(text: str) -> LinearIsotropicMaterials:
    blocks = re.finditer(
        r"(?P<name>[A-Za-z0-9_]+):\s*\n\s*Type:\s*LinearIsotropic\s*\n\s*E:\s*(?P<E>[-+0-9.eE]+)\s*\n\s*nu:\s*(?P<nu>[-+0-9.eE]+)",
        text,
    )
    definitions: dict[str, tuple[float, float]] = {}
    for match in blocks:
        definitions[match.group("name")] = (
            float(match.group("E")),
            float(match.group("nu")),
        )
    if not definitions:
        raise ValueError("No LinearIsotropic material definitions found")

    table_match = re.search(r"^MaterialTable:[ \t]*$", text, flags=re.M)
    if table_match is None:
        raise ValueError("MaterialTable section not found")

    youngs: dict[int, float] = {}
    poisson: dict[int, float] = {}
    for line in text[table_match.end() :].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if line == line.lstrip(" \t"):
            break
        if ":" not in stripped:
            continue
        label_text, name = [part.strip() for part in stripped.split(":", 1)]
        if not label_text.isdigit():
            continue
        if name not in definitions:
            raise ValueError(f"MaterialTable references undefined material '{name}'")
        label = int(label_text)
        youngs[label], poisson[label] = definitions[name]
    if not youngs:
        raise ValueError("MaterialTable contains no numeric labels")
    return LinearIsotropicMaterials(youngs_modulus_mpa=youngs, poisson_ratio=poisson)


def linear_isotropic_materials_from_config(
    config: dict[str, Any],
) -> LinearIsotropicMaterials:
    labels_cfg = config.get("labels")
    if labels_cfg is not None:
        if not isinstance(labels_cfg, dict):
            raise ValueError("materials.labels must be a table/object")
        youngs_by_label: dict[int, float] = {}
        poisson_by_label: dict[int, float] = {}
        default_nu = config.get("poisson_ratio", config.get("nu", 0.3))
        for label, spec in labels_cfg.items():
            if not isinstance(spec, dict):
                raise ValueError(f"materials.labels.{label} must be an object")
            youngs = spec.get(
                "E",
                spec.get(
                    "youngs_modulus",
                    spec.get("youngs_modulus_mpa", spec.get("modulus_mpa")),
                ),
            )
            if youngs is None:
                raise ValueError(f"materials.labels.{label} requires E")
            nu = spec.get("nu", spec.get("poisson_ratio", default_nu))
            numeric_label = int(label)
            youngs_by_label[numeric_label] = float(youngs)
            poisson_by_label[numeric_label] = float(nu)
        if not youngs_by_label:
            raise ValueError("materials.labels contains no labels")
        return LinearIsotropicMaterials(
            youngs_modulus_mpa=youngs_by_label,
            poisson_ratio=poisson_by_label,
        )

    definitions_cfg = config.get("definitions", config.get("MaterialDefinitions"))
    table_cfg = config.get("table", config.get("MaterialTable"))
    if not isinstance(definitions_cfg, dict) or not isinstance(table_cfg, dict):
        raise ValueError("inline materials require definitions and table sections")

    definitions: dict[str, tuple[float, float]] = {}
    for name, spec in definitions_cfg.items():
        if not isinstance(spec, dict):
            raise ValueError(f"material definition '{name}' must be an object")
        material_type = str(spec.get("Type", spec.get("type", "LinearIsotropic")))
        if material_type.strip().lower() != "linearisotropic":
            raise ValueError(f"unsupported material type for '{name}': {material_type}")
        youngs = spec.get(
            "E", spec.get("youngs_modulus", spec.get("youngs_modulus_mpa"))
        )
        nu = spec.get("nu", spec.get("poisson_ratio"))
        if youngs is None or nu is None:
            raise ValueError(f"material definition '{name}' requires E and nu")
        definitions[str(name)] = (float(youngs), float(nu))

    youngs_by_label: dict[int, float] = {}
    poisson_by_label: dict[int, float] = {}
    for label, name in table_cfg.items():
        token = str(name)
        if token not in definitions:
            raise ValueError(f"MaterialTable references undefined material '{token}'")
        numeric_label = int(label)
        youngs_by_label[numeric_label], poisson_by_label[numeric_label] = definitions[
            token
        ]
    if not youngs_by_label:
        raise ValueError("inline MaterialTable contains no labels")
    return LinearIsotropicMaterials(
        youngs_modulus_mpa=youngs_by_label,
        poisson_ratio=poisson_by_label,
    )


def _poisson_ratio_map(
    label_array: np.ndarray,
    table: LinearIsotropicMaterials,
    *,
    override: float | None,
) -> float | np.ndarray:
    if override is not None:
        return float(override)
    unique = sorted({round(float(value), 12) for value in table.poisson_ratio.values()})
    if len(unique) == 1:
        return float(unique[0])
    fill = float(unique[0])
    out = np.full(label_array.shape, fill, dtype=np.float64)
    for label, value in table.poisson_ratio.items():
        out[label_array == label] = float(value)
    return out
