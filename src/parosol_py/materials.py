from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearIsotropicMaterials:
    youngs_modulus_mpa: dict[int, float]
    poisson_ratio: dict[int, float]


def material_to_stiffness_gpa(material, *, material_unit: str = "MPa") -> np.ndarray:
    arr = np.asarray(material, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"material must be 3D, got shape {arr.shape}")
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


def parse_linear_isotropic_materials(text: str) -> LinearIsotropicMaterials:
    blocks = re.finditer(
        r"(?P<name>[A-Za-z0-9_]+):\s*\n\s*Type:\s*LinearIsotropic\s*\n\s*E:\s*(?P<E>[-+0-9.eE]+)\s*\n\s*nu:\s*(?P<nu>[-+0-9.eE]+)",
        text,
    )
    definitions: dict[str, tuple[float, float]] = {}
    for match in blocks:
        definitions[match.group("name")] = (float(match.group("E")), float(match.group("nu")))
    if not definitions:
        raise ValueError("No LinearIsotropic material definitions found")

    table_match = re.search(r"MaterialTable:\s*(?P<table>.*)", text, flags=re.S)
    if table_match is None:
        raise ValueError("MaterialTable section not found")

    youngs: dict[int, float] = {}
    poisson: dict[int, float] = {}
    for line in table_match.group("table").splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
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
