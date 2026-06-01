from __future__ import annotations

import numpy as np


def apply_scalar_poisson_7point(values, *, spacing: float = 1.0) -> np.ndarray:
    """Apply a tiny CPU 7-point Poisson stencil for backend prototyping.

    This scalar operator is useful for testing structured-grid indexing and
    boundary handling. It is not an elasticity operator and is not equivalent
    to native ParOSol.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("values must be a 3D array")
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    padded = np.pad(array, pad_width=1, mode="constant")
    result = 6.0 * array
    result -= padded[:-2, 1:-1, 1:-1]
    result -= padded[2:, 1:-1, 1:-1]
    result -= padded[1:-1, :-2, 1:-1]
    result -= padded[1:-1, 2:, 1:-1]
    result -= padded[1:-1, 1:-1, :-2]
    result -= padded[1:-1, 1:-1, 2:]
    return result / (float(spacing) ** 2)
