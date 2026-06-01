from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldComparison:
    passed: bool
    max_abs_error: float
    mean_abs_error: float
    rtol: float
    atol: float


def compare_field(
    reference,
    candidate,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> FieldComparison:
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)

    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "reference and candidate must have matching shapes: "
            f"{reference_array.shape} != {candidate_array.shape}"
        )

    abs_error = np.abs(reference_array - candidate_array)
    if abs_error.size == 0:
        max_abs_error = 0.0
        mean_abs_error = 0.0
    else:
        max_abs_error = float(np.max(abs_error))
        mean_abs_error = float(np.mean(abs_error))

    return FieldComparison(
        passed=bool(
            np.allclose(reference_array, candidate_array, rtol=rtol, atol=atol)
        ),
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        rtol=rtol,
        atol=atol,
    )
