from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from parosol_py.core import BoundaryConditionSet


@dataclass(frozen=True)
class BuiltModel:
    material: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    poisson_ratio: float
    boundary_conditions: BoundaryConditionSet
    node_sets: dict[str, list[tuple[int, int, int]]]
    element_sets: dict[str, int]
    postprocess_mask: np.ndarray | None = None
    nonlinear_material: Any | None = None
    exported: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
