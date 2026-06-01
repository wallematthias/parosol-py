from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from parosol_torch import SolverSettings, VoxelElasticityProblem, solve


def main() -> int:
    cases = [
        (
            "tiny_1x1x1",
            face_tension_problem(
                np.ones((1, 1, 1), dtype=np.float32),
                displacement=0.01,
            ),
            1e-8,
            100,
        )
    ]
    trab = Path("tests/fixtures/trab1240/trab1240_labels.npz")
    if trab.exists():
        cases.extend(trab_dense_cases(trab, sizes=(3, 5)))

    for name, problem, cpu_tolerance, max_iterations in cases:
        cpu = timed(problem, "cpu", cpu_tolerance, max_iterations)
        mps = timed(problem, "mps", 1e-5, max_iterations)
        speedup = cpu["time"] / mps["time"] if mps["time"] > 0 else float("inf")
        print(f"CASE {name}")
        for item in (cpu, mps):
            print(
                f"  {item['device']:>3} time={item['time']:.4f}s "
                f"conv={item['converged']} iter={item['iterations']} "
                f"residual={item['residual']:.3e} "
                f"reaction={item['reaction_left_x']:.8g} "
                f"sed_mean={item['sed_mean_active']:.8g} "
                f"sed_max={item['sed_max']:.8g}"
            )
        print(f"  speedup_mps_vs_cpu={speedup:.2f}x")
    return 0


def face_tension_problem(
    stiffness_xyz: np.ndarray,
    *,
    spacing: float = 1.0,
    displacement: float = 0.001,
) -> VoxelElasticityProblem:
    nx, ny, nz = stiffness_xyz.shape
    fixed = []
    loaded = []
    for y in range(ny + 1):
        for z in range(nz + 1):
            for component in range(3):
                fixed.append([0, y, z, component])
            loaded.append([nx, y, z, 0])
    return VoxelElasticityProblem(
        stiffness_gpa_xyz=stiffness_xyz.astype(np.float32),
        voxel_size_mm=float(spacing),
        poisson_ratio=0.3,
        fixed_displacement_coordinates=np.asarray(fixed, dtype=np.int64),
        fixed_displacement_values=np.zeros(len(fixed), dtype=np.float64),
        loaded_node_coordinates=np.asarray(loaded, dtype=np.int64),
        loaded_node_values=np.full(len(loaded), displacement, dtype=np.float64),
        requested_outputs=("forces", "displacements", "sed"),
    )


def trab_dense_cases(path: Path, *, sizes: tuple[int, ...]):
    with np.load(path) as data:
        labels_zyx = np.asarray(data["labels"])
        spacing_xyz = tuple(
            float(v) for v in np.asarray(data["spacing_xyz"]).reshape(-1)
        )
    labels_xyz = np.transpose(labels_zyx, (2, 1, 0))
    cases = []
    for size in sizes:
        found = dense_crop(labels_xyz, size)
        if found is None:
            continue
        x, y, z, crop = found
        stiffness = np.where(crop > 0, 8.748, 0.0).astype(np.float32)
        print(
            f"crop_{size} origin_xyz={(x, y, z)} "
            f"active={int(np.count_nonzero(crop))}"
        )
        cases.append(
            (
                f"trab_dense_{size}x{size}x{size}",
                face_tension_problem(stiffness, spacing=spacing_xyz[0]),
                1e-8,
                1000,
            )
        )
    return cases


def dense_crop(labels_xyz: np.ndarray, size: int):
    for x in range(labels_xyz.shape[0] - size + 1):
        for y in range(labels_xyz.shape[1] - size + 1):
            for z in range(labels_xyz.shape[2] - size + 1):
                crop = labels_xyz[x : x + size, y : y + size, z : z + size]
                if np.all(crop > 0):
                    return x, y, z, crop.copy()
    return None


def timed(problem, device: str, tolerance: float, max_iterations: int):
    started = perf_counter()
    result = solve(
        problem,
        SolverSettings(
            device=device,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
    )
    elapsed = perf_counter() - started
    sed = result.fields["sed"]
    active_sed = sed[sed > 0]
    return {
        "device": device,
        "time": elapsed,
        "converged": result.converged,
        "iterations": result.iterations,
        "residual": result.residual_norm,
        "reaction_left_x": float(result.fields["forces"][0, :, :, 0].sum()),
        "sed_mean_active": float(active_sed.mean()) if active_sed.size else 0.0,
        "sed_max": float(sed.max()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
