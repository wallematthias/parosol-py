from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from .api import solve


@dataclass(frozen=True)
class NonlinearLoadHistoryResult:
    steps: list[dict[str, Any]]


_REQUIRED_OUTPUTS = ("forces", "displacements", "von_mises", "plastic_strain")


def run_nonlinear_load_history(
    *,
    material,
    spacing: tuple[float, float, float],
    final_strain: float,
    steps: int,
    nonlinear_material,
    nonlinear_solver,
    work_dir,
    **solve_kwargs,
) -> NonlinearLoadHistoryResult:
    if isinstance(steps, bool) or not isinstance(steps, Integral):
        raise ValueError("steps must be a positive integer")
    if steps <= 0:
        raise ValueError("steps must be positive")

    caller_outputs = tuple(solve_kwargs.pop("outputs", ()))
    outputs = tuple(dict.fromkeys((*_REQUIRED_OUTPUTS, *caller_outputs)))
    root = Path(work_dir)
    records: list[dict[str, Any]] = []
    for index in range(1, steps + 1):
        strain = final_strain * index / steps
        step_dir = root / f"step_{index:03d}"
        result = solve(
            material=material,
            spacing=spacing,
            strain=strain,
            nonlinear_material=nonlinear_material,
            nonlinear_solver=nonlinear_solver,
            work_dir=step_dir,
            outputs=outputs,
            **solve_kwargs,
        )
        mechanics = result.diagnostics.get("mechanics", {})
        records.append(
            {
                "step": index,
                "strain": strain,
                "generalized_load": mechanics.get("generalized_load"),
                "reaction_force": mechanics.get("reaction_force"),
                "plastic_iterations": result.diagnostics.get("nonlinear", {}).get(
                    "plastic_iterations"
                ),
            }
        )

    return NonlinearLoadHistoryResult(steps=records)
