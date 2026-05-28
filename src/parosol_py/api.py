from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .boundary_conditions import axial_compression
from .hdf5_io import write_parosol_input
from .images import ImageGrid, export_scalar_image, normalize_array
from .materials import material_to_stiffness_gpa
from .results import read_solution_fields
from .runner import RunSummary, build_parosol_command, packaged_executable, run_parosol

try:
    from py_aimio import read_aim
except ImportError:

    def read_aim(path):
        raise ImportError(
            "py_aimio is required to read AIM files. Install py_aimio or "
            "pass material arrays directly to solve()."
        ) from None


@dataclass(frozen=True)
class SolveSummary:
    dimensions_xyz: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    run: RunSummary | None = None


@dataclass(frozen=True)
class SolveResult:
    input_file: Path
    command: list[str]
    fields: dict[str, Any]
    summary: SolveSummary
    stdout: str = ""
    stderr: str = ""
    exported: dict[str, Path] = field(default_factory=dict)


def solve(
    *,
    material,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    array_order: str = "zyx",
    material_unit: str = "MPa",
    poisson_ratio: float = 0.3,
    test: str = "axial",
    test_axis: str = "z",
    strain: float = -0.01,
    outputs: tuple[str, ...] = ("sed",),
    tolerance: float = 1e-6,
    level: int = 6,
    executable: str | Path | None = None,
    work_dir: str | Path | None = None,
    export_dir: str | Path | None = None,
    dry_run: bool = False,
) -> SolveResult:
    if test.strip().lower() != "axial":
        raise ValueError("only test='axial' is supported")

    grid = normalize_array(
        material,
        spacing=spacing,
        origin=origin,
        array_order=array_order,
    )
    if not np.allclose(grid.spacing, grid.spacing[0], rtol=1e-9, atol=1e-12):
        raise ValueError(
            "solve() requires isotropic spacing; anisotropic spacing is not supported"
        )
    stiffness_gpa_xyz = material_to_stiffness_gpa(
        grid.array_xyz,
        material_unit=material_unit,
    )
    fixed_coords, fixed_values = axial_compression(
        stiffness_gpa_xyz,
        axis=test_axis,
        strain=strain,
    )

    case_dir = _prepare_work_dir(work_dir)
    input_file = write_parosol_input(
        case_dir / "parosol_input.h5",
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coords,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=float(grid.spacing[0]),
        poisson_ratio=poisson_ratio,
    )
    command = build_parosol_command(
        executable=executable if executable is not None else packaged_executable(),
        input_file=input_file,
        outputs=tuple(outputs),
        tolerance=tolerance,
        level=level,
    )
    summary = SolveSummary(
        dimensions_xyz=tuple(int(v) for v in grid.array_xyz.shape),
        spacing=grid.spacing,
        origin=grid.origin,
    )

    if dry_run:
        return SolveResult(
            input_file=input_file,
            command=command,
            fields={},
            summary=summary,
        )

    run = run_parosol(command, cwd=case_dir)
    if run.returncode != 0:
        raise RuntimeError(
            f"ParOSol failed with return code {run.returncode}\n"
            f"stdout:\n{run.stdout}\n"
            f"stderr:\n{run.stderr}"
        )

    fields = read_solution_fields(input_file, outputs=tuple(outputs))
    exported: dict[str, Path] = {}
    if export_dir is not None:
        export_root = Path(export_dir).expanduser().resolve()
        for name, field_values in fields.items():
            field_array = np.asarray(field_values)
            if field_array.ndim == 1 and field_array.size == stiffness_gpa_xyz.size:
                exported[name] = export_scalar_image(
                    ImageGrid(
                        array_xyz=field_array.reshape(stiffness_gpa_xyz.shape),
                        spacing=grid.spacing,
                        origin=grid.origin,
                    ),
                    export_root / f"{name}.nii.gz",
                )

    return SolveResult(
        input_file=input_file,
        command=run.command,
        fields=fields,
        summary=SolveSummary(
            dimensions_xyz=summary.dimensions_xyz,
            spacing=summary.spacing,
            origin=summary.origin,
            run=run.summary,
        ),
        stdout=run.stdout,
        stderr=run.stderr,
        exported=exported,
    )


def solve_aim(path: str | Path, **kwargs: Any) -> SolveResult:
    spacing = kwargs.pop("spacing", None)

    material, meta = read_aim(str(path))
    if spacing is None:
        spacing = meta.get("element_size", (1.0, 1.0, 1.0))
    origin = kwargs.pop("origin", meta.get("position", (0.0, 0.0, 0.0)))

    return solve(
        material=material,
        spacing=spacing,
        origin=origin,
        array_order="zyx",
        **kwargs,
    )


def _prepare_work_dir(work_dir: str | Path | None) -> Path:
    if work_dir is None:
        return Path(tempfile.mkdtemp(prefix="parosol_py_")).resolve()
    out = Path(work_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out
