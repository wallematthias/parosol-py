from __future__ import annotations

from .core import OutputProfile, SolverProfile

SOLVER_PROFILES = {
    "legacy_axial": SolverProfile(
        tolerance=1e-6,
        level=6,
        mpi_processes=1,
        outputs=("sed",),
    ),
    "batch": SolverProfile(
        tolerance=1e-6,
        level=6,
        mpi_processes=6,
        outputs=("sed",),
    ),
    "debug": SolverProfile(
        tolerance=1e-6,
        level=6,
        mpi_processes=1,
        outputs=("sed", "effective_strain", "von_mises"),
    ),
}

OUTPUT_PROFILES = {
    "quick_summary": OutputProfile(export_fields=False, image_fields=()),
    "standard_fields": OutputProfile(export_fields=True, image_fields=("sed",)),
    "debug": OutputProfile(
        export_fields=True,
        image_fields=("sed", "effective_strain", "von_mises"),
    ),
}


def get_solver_profile(name: str | None) -> SolverProfile:
    if name is None:
        return SolverProfile()
    try:
        return SOLVER_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown solver profile: {name}") from exc


def get_output_profile(name: str | None) -> OutputProfile:
    if name is None:
        return OutputProfile()
    try:
        return OUTPUT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown output profile: {name}") from exc
