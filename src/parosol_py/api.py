from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .boundary_conditions import axial_compression
from .core import BoundaryConditionSet
from .diagnostics import build_fea_diagnostics
from .field_export import NativeFieldMapper
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
    diagnostics: dict[str, Any] = field(default_factory=dict)


def solve(
    *,
    material,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    array_order: str = "zyx",
    material_unit: str = "MPa",
    poisson_ratio: float | np.ndarray = 0.3,
    test: str = "axial",
    test_axis: str = "z",
    strain: float = -0.01,
    load_case_type: str = "constrained_axial",
    load_direction: str | None = None,
    rotation_degrees: float | None = None,
    load_case_center: tuple[float, float] | None = None,
    outputs: tuple[str, ...] = ("sed",),
    tolerance: float = 1e-6,
    level: int = 6,
    mpi_processes: int = 1,
    mpi_launcher: str | Path = "mpirun",
    stream_output: bool = False,
    executable: str | Path | None = None,
    work_dir: str | Path | None = None,
    export_dir: str | Path | None = None,
    failure_criterion: str = "pistoia",
    critical_strain: float | None = 0.007,
    critical_volume_percent: float | None = 2.0,
    linear_failure_deformation: float = 0.002,
    crawford_coefficient: float = 0.0068,
    linear_failure_estimates: bool = False,
    boundary_conditions: BoundaryConditionSet | None = None,
    postprocess_mask=None,
    nonlinear_material=None,
    nonlinear_solver=None,
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
    if not np.allclose(grid.spacing, grid.spacing[0], rtol=1e-3, atol=1e-6):
        raise ValueError(
            "solve() requires isotropic spacing; anisotropic spacing is not supported"
        )
    stiffness_gpa_xyz = material_to_stiffness_gpa(
        grid.array_xyz,
        material_unit=material_unit,
    )
    nonlinear_material_xyz = _nonlinear_material_for_solve(
        nonlinear_material,
        array_order=array_order,
    )
    mask_xyz = _postprocess_mask_xyz(
        postprocess_mask,
        spacing=spacing,
        origin=origin,
        array_order=array_order,
        expected_shape=grid.array_xyz.shape,
    )
    if boundary_conditions is None:
        fixed_coords, fixed_values = axial_compression(
            stiffness_gpa_xyz,
            axis=test_axis,
            strain=strain,
            voxel_size_mm=float(grid.spacing[0]),
        )
        loaded_coords = None
        loaded_values = None
    else:
        fixed_coords = boundary_conditions.fixed_coordinates
        fixed_values = boundary_conditions.fixed_values
        loaded_coords = boundary_conditions.loaded_coordinates
        loaded_values = boundary_conditions.loaded_values

    case_dir = _prepare_work_dir(work_dir)
    input_file = write_parosol_input(
        path=case_dir / "parosol_input.h5",
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        fixed_displacement_coordinates=fixed_coords,
        fixed_displacement_values=fixed_values,
        voxel_size_mm=float(grid.spacing[0]),
        poisson_ratio=poisson_ratio,
        loaded_node_coordinates=loaded_coords,
        loaded_node_values=loaded_values,
        nonlinear_material=nonlinear_material_xyz,
        nonlinear_solver=nonlinear_solver,
    )
    command = build_parosol_command(
        executable=executable if executable is not None else packaged_executable(),
        input_file=input_file,
        outputs=tuple(outputs),
        tolerance=tolerance,
        level=level,
        mpi_processes=mpi_processes,
        mpi_launcher=mpi_launcher,
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

    run = run_parosol(command, cwd=case_dir, stream=stream_output)
    exported = _write_run_logs(case_dir, command=command, run=run)
    if run.returncode != 0:
        raise RuntimeError(
            f"ParOSol failed with return code {run.returncode}\n"
            f"logs: {exported}\n"
            f"stdout:\n{run.stdout}\n"
            f"stderr:\n{run.stderr}"
        )

    result_outputs = _summary_outputs(outputs)
    fields = read_solution_fields(input_file, outputs=result_outputs)
    if export_dir is not None:
        export_root = Path(export_dir).expanduser().resolve()
        active_size = int(np.count_nonzero(stiffness_gpa_xyz > 0))
        mapper = NativeFieldMapper(stiffness_gpa_xyz)
        requested_exports = {str(output).strip().lower() for output in outputs}
        for name, field_values in fields.items():
            if name not in requested_exports:
                continue
            field_array = _native_scalar_field(
                field_values,
                expected_sizes=(stiffness_gpa_xyz.size, active_size),
            )
            if field_array is not None:
                exported[name] = export_scalar_image(
                    ImageGrid(
                        array_xyz=_apply_postprocess_mask(
                            mapper.scalar_to_dense(field_array), mask_xyz
                        ),
                        spacing=grid.spacing,
                        origin=grid.origin,
                    ),
                    export_root / f"{name}.nii.gz",
                )
                continue
            vector_array = _native_vector_field(field_values)
            if vector_array is not None and name == "displacements":
                dense_vector = mapper.nodal_vector_to_dense_element(vector_array)
                for component, axis in enumerate(("x", "y", "z")):
                    exported[f"displacement_{axis}"] = export_scalar_image(
                        ImageGrid(
                            array_xyz=_apply_postprocess_mask(
                                dense_vector[..., component], mask_xyz
                            ),
                            spacing=grid.spacing,
                            origin=grid.origin,
                        ),
                        export_root / f"displacement_{axis}.nii.gz",
                    )

    diagnostics = build_fea_diagnostics(
        fields=fields,
        stiffness_gpa_xyz=stiffness_gpa_xyz,
        axis=test_axis,
        strain=strain,
        voxel_size_mm=float(grid.spacing[0]),
        load_case_type=load_case_type,
        load_direction=load_direction,
        rotation_degrees=rotation_degrees,
        load_case_center=load_case_center,
        failure_criterion=failure_criterion,
        critical_strain=critical_strain,
        critical_volume_percent=critical_volume_percent,
        boundary_conditions=boundary_conditions,
        evaluation_mask_xyz=mask_xyz,
        analysis_dimensions_xyz=_analysis_dimensions_xyz(mask_xyz),
        linear_failure_deformation=linear_failure_deformation,
        crawford_coefficient=crawford_coefficient,
        linear_failure_estimates=linear_failure_estimates,
    )
    nonlinear_diagnostics = _read_nonlinear_diagnostics(input_file)
    if nonlinear_diagnostics:
        diagnostics["nonlinear"] = nonlinear_diagnostics

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
        diagnostics=diagnostics,
    )


def _nonlinear_material_for_solve(nonlinear_material, *, array_order: str):
    if nonlinear_material is None:
        return None
    if not hasattr(nonlinear_material, "compressive_yield_mpa"):
        return nonlinear_material

    order = array_order.strip().lower()
    if order in {"xyz", "x-y-z"}:
        return nonlinear_material
    if order not in {"zyx", "z-y-x"}:
        raise ValueError("array_order must be 'zyx' or 'xyz'")

    def to_xyz(array):
        return np.transpose(np.asarray(array), (2, 1, 0))

    poisson = nonlinear_material.poisson_ratio
    if isinstance(poisson, np.ndarray):
        poisson = to_xyz(poisson)
    return replace(
        nonlinear_material,
        youngs_modulus_mpa=to_xyz(nonlinear_material.youngs_modulus_mpa),
        poisson_ratio=poisson,
        compressive_yield_mpa=to_xyz(nonlinear_material.compressive_yield_mpa),
        tensile_yield_mpa=to_xyz(nonlinear_material.tensile_yield_mpa),
        plateau_mpa=to_xyz(nonlinear_material.plateau_mpa),
        material_id=to_xyz(nonlinear_material.material_id),
    )


def _write_run_logs(
    case_dir: Path, *, command: list[str], run
) -> dict[str, Path]:
    command_path = case_dir / "parosol_command.txt"
    stdout_path = case_dir / "parosol_stdout.log"
    stderr_path = case_dir / "parosol_stderr.log"
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")
    stdout_path.write_text(run.stdout, encoding="utf-8")
    stderr_path.write_text(run.stderr, encoding="utf-8")
    return {
        "command_log": command_path,
        "stdout_log": stdout_path,
        "stderr_log": stderr_path,
    }


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


def _summary_outputs(outputs: tuple[str, ...]) -> tuple[str, ...]:
    requested: list[str] = []
    for output in (*outputs, "forces", "displacements"):
        token = str(output).strip().lower()
        if token not in requested:
            requested.append(token)
    return tuple(requested)


def _read_nonlinear_diagnostics(path: Path) -> dict[str, float | int]:
    names = (
        "plastic_iterations",
        "yielded_last",
        "plastic_convergence_last",
    )
    with h5py.File(path, "r") as h5:
        if "NonlinearResults" not in h5:
            return {}
        group = h5["NonlinearResults"]
        values: dict[str, float | int] = {}
        for name in names:
            if name in group.attrs:
                value = group.attrs[name]
            elif name in group:
                value = group[name][()]
            else:
                continue
            values[name] = (
                float(value) if name == "plastic_convergence_last" else int(value)
            )
    return values


def _native_scalar_field(
    values,
    expected_sizes: tuple[int, ...],
) -> np.ndarray | None:
    array = np.asarray(values)
    expected_sizes = tuple(int(size) for size in expected_sizes)
    if array.ndim == 1 and array.shape[0] in expected_sizes:
        return array.reshape(-1)
    if array.ndim == 2 and array.shape[1] == 1 and array.shape[0] in expected_sizes:
        return array.reshape(-1)
    return None


def _native_vector_field(values) -> np.ndarray | None:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 3:
        return array
    return None


def _postprocess_mask_xyz(
    mask,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    array_order: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray | None:
    if mask is None:
        return None
    grid = normalize_array(
        mask,
        spacing=spacing,
        origin=origin,
        array_order=array_order,
    )
    mask_xyz = np.asarray(grid.array_xyz, dtype=bool)
    if mask_xyz.shape != expected_shape:
        raise ValueError(
            f"postprocess_mask shape {mask_xyz.shape} does not match material shape {expected_shape}"
        )
    return mask_xyz


def _apply_postprocess_mask(
    field_xyz: np.ndarray,
    mask_xyz: np.ndarray | None,
) -> np.ndarray:
    if mask_xyz is None:
        return field_xyz
    return np.where(mask_xyz, field_xyz, 0)


def _analysis_dimensions_xyz(mask_xyz: np.ndarray | None) -> tuple[int, int, int] | None:
    if mask_xyz is None or not np.any(mask_xyz):
        return None
    coords = np.argwhere(np.asarray(mask_xyz, dtype=bool))
    return tuple(int(v) for v in (coords.max(axis=0) - coords.min(axis=0) + 1))
