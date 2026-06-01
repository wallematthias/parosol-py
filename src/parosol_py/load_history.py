from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoadHistoryResult:
    loading_history: np.ndarray
    scaling_factors: np.ndarray
    residual: float
    mean: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scaling_factors": self.scaling_factors.tolist(),
            "residual": self.residual,
            "mean": self.mean,
            "std": self.std,
        }


def estimate_load_history(
    load_cases,
    bone_mask,
    *,
    evaluation_region=None,
    target_average: float = 0.02,
    cutoff_percentile: float = 95.0,
) -> LoadHistoryResult:
    """Estimate a non-negative linear load history from solved SED fields."""
    arrays = [
        np.asarray(load_case, dtype=np.float64).copy() for load_case in load_cases
    ]
    if not arrays:
        raise ValueError("at least one load case is required")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all load cases must have the same shape")

    mask = np.asarray(bone_mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError("bone_mask must match load case shape")
    if evaluation_region is not None:
        region = np.asarray(evaluation_region, dtype=bool)
        if region.shape != shape:
            raise ValueError("evaluation_region must match load case shape")
        mask = np.logical_and(mask, region)
    if not np.any(mask):
        raise ValueError("bone/evaluation mask contains no voxels")

    matrix_rows: list[np.ndarray] = []
    for index, array in enumerate(arrays):
        active = array[mask].copy()
        cutoff = np.percentile(active, float(cutoff_percentile))
        active[active > cutoff] = cutoff
        array[array > cutoff] = cutoff
        arrays[index] = array
        matrix_rows.append(active)
    matrix = np.stack(matrix_rows, axis=1)
    target = np.full(matrix.shape[0], float(target_average), dtype=np.float64)
    scaling, residual = _nnls(matrix, target)
    history = np.zeros(shape, dtype=np.float64)
    for array, factor in zip(arrays, scaling, strict=True):
        history += array * factor
    final_cutoff = np.percentile(history[mask], float(cutoff_percentile))
    history[history > final_cutoff] = final_cutoff
    return LoadHistoryResult(
        loading_history=history,
        scaling_factors=np.asarray(scaling, dtype=np.float64),
        residual=float(residual),
        mean=float(np.mean(history[mask])),
        std=float(np.std(history[mask])),
    )


def estimate_load_history_from_files(
    load_case_paths,
    *,
    bone_mask_path: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    target_average: float = 0.02,
    cutoff_percentile: float = 95.0,
) -> LoadHistoryResult:
    load_cases = [_read_array(path) for path in load_case_paths]
    result = estimate_load_history(
        load_cases,
        _read_array(bone_mask_path) > 0,
        target_average=target_average,
        cutoff_percentile=cutoff_percentile,
    )
    if output_path is not None:
        _write_array(output_path, result.loading_history)
    if summary_path is not None:
        from .reports import write_summary_json

        write_summary_json(summary_path, {"load_history": result.to_dict()})
    return result


def _nnls(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        from scipy.optimize import nnls
    except ImportError:
        return _projected_least_squares(matrix, target)
    scaling, residual = nnls(matrix, target)
    return np.asarray(scaling, dtype=np.float64), float(residual)


def _projected_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 5000,
) -> tuple[np.ndarray, float]:
    scale = np.zeros(matrix.shape[1], dtype=np.float64)
    spectral = np.linalg.norm(matrix, ord=2)
    step = 1.0 / max(float(spectral * spectral), 1e-12)
    for _ in range(iterations):
        gradient = matrix.T @ (matrix @ scale - target)
        scale = np.maximum(0.0, scale - step * gradient)
    residual = np.linalg.norm(matrix @ scale - target)
    return scale, float(residual)


def _read_array(path: str | Path) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".npy"):
        return np.load(p)
    if suffixes.endswith(".npz"):
        with np.load(p) as data:
            if "image" in data:
                return np.asarray(data["image"])
            if "sed" in data:
                return np.asarray(data["sed"])
            if len(data.files) == 1:
                return np.asarray(data[data.files[0]])
            raise ValueError(
                f"NPZ must contain image, sed, or one array; got {data.files}"
            )
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))


def _write_array(path: str | Path, array: np.ndarray) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".npy"):
        np.save(p, array)
    elif suffixes.endswith(".npz"):
        np.savez_compressed(p, image=array)
    else:
        import SimpleITK as sitk

        sitk.WriteImage(
            sitk.GetImageFromArray(np.asarray(array, dtype=np.float32)), str(p)
        )
    return p
