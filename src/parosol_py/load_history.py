from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoadHistoryResult:
    loading_history: np.ndarray
    scaling_factors: np.ndarray
    load_amplitudes: np.ndarray
    input_load_amplitudes: np.ndarray
    residual: float
    mean: float
    std: float
    failure: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        details = {
            "scaling_factors": self.scaling_factors.tolist(),
            "load_amplitudes": self.load_amplitudes.tolist(),
            "input_load_amplitudes": self.input_load_amplitudes.tolist(),
            "residual": self.residual,
            "mean": self.mean,
            "std": self.std,
        }
        data = {
            "results": {
                "load_amplitudes": self.load_amplitudes.tolist(),
            },
            "details": details,
        }
        if self.failure is not None:
            data["failure"] = self.failure
        return data


def estimate_load_history(
    load_cases,
    bone_mask,
    *,
    evaluation_region=None,
    target_average: float = 0.02,
    cutoff_percentile: float = 95.0,
    max_fit_voxels: int | None = 200_000,
    stiffness_gpa=None,
    critical_strain: float = 0.007,
    critical_volume_percent: float = 2.0,
    input_load_amplitudes=None,
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
    input_amplitudes = _input_load_amplitudes(
        input_load_amplitudes, expected_count=len(arrays)
    )
    unit_arrays = [
        array / float(amplitude * amplitude)
        for array, amplitude in zip(arrays, input_amplitudes, strict=True)
    ]

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
    for index, array in enumerate(unit_arrays):
        active = array[mask].copy()
        cutoff = np.percentile(active, float(cutoff_percentile))
        active[active > cutoff] = cutoff
        array[array > cutoff] = cutoff
        unit_arrays[index] = array
        matrix_rows.append(active)
    matrix = np.stack(matrix_rows, axis=1)
    fit_indices = _fit_indices(matrix.shape[0], max_fit_voxels=max_fit_voxels)
    fit_matrix = matrix[fit_indices]
    target = np.full(fit_matrix.shape[0], float(target_average), dtype=np.float64)
    scaling, _ = _nnls(fit_matrix, target)
    residual = float(np.linalg.norm(matrix @ scaling - float(target_average)))
    history = np.zeros(shape, dtype=np.float64)
    for array, factor in zip(unit_arrays, scaling, strict=True):
        history += array * factor
    final_cutoff = np.percentile(history[mask], float(cutoff_percentile))
    history[history > final_cutoff] = final_cutoff
    failure = None
    if stiffness_gpa is not None:
        failure = pistoia_from_sed(
            history,
            stiffness_gpa,
            bone_mask=mask,
            critical_strain=critical_strain,
            critical_volume_percent=critical_volume_percent,
        )
    load_amplitudes = np.sqrt(np.maximum(0.0, scaling))
    return LoadHistoryResult(
        loading_history=history,
        scaling_factors=np.asarray(scaling, dtype=np.float64),
        load_amplitudes=load_amplitudes,
        input_load_amplitudes=input_amplitudes,
        residual=float(residual),
        mean=float(np.mean(history[mask])),
        std=float(np.std(history[mask])),
        failure=failure,
    )


def estimate_load_history_from_files(
    load_case_paths,
    *,
    bone_mask_path: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    target_average: float = 0.02,
    cutoff_percentile: float = 95.0,
    max_fit_voxels: int | None = 200_000,
    stiffness_path: str | Path | None = None,
    critical_strain: float = 0.007,
    critical_volume_percent: float = 2.0,
    input_load_amplitudes=None,
) -> LoadHistoryResult:
    load_cases = [_read_array(path) for path in load_case_paths]
    if bone_mask_path is None:
        bone_mask = np.logical_or.reduce([np.asarray(case) > 0 for case in load_cases])
    else:
        bone_mask = _read_array(bone_mask_path) > 0
    result = estimate_load_history(
        load_cases,
        bone_mask,
        target_average=target_average,
        cutoff_percentile=cutoff_percentile,
        max_fit_voxels=max_fit_voxels,
        stiffness_gpa=None if stiffness_path is None else _read_stiffness(stiffness_path),
        critical_strain=critical_strain,
        critical_volume_percent=critical_volume_percent,
        input_load_amplitudes=input_load_amplitudes,
    )
    if output_path is not None:
        _write_array(output_path, result.loading_history)
    if summary_path is not None:
        from .reports import write_summary_json

        write_summary_json(summary_path, {"load_history": result.to_dict()})
    return result


def pistoia_from_sed(
    sed,
    stiffness_gpa,
    *,
    bone_mask=None,
    critical_strain: float = 0.007,
    critical_volume_percent: float = 2.0,
) -> dict[str, Any]:
    sed_array = np.asarray(sed, dtype=np.float64)
    stiffness_mpa = np.asarray(stiffness_gpa, dtype=np.float64) * 1000.0
    if sed_array.shape != stiffness_mpa.shape:
        raise ValueError("stiffness image must match SED image shape")
    mask = stiffness_mpa > 0.0
    if bone_mask is not None:
        mask &= np.asarray(bone_mask, dtype=bool)
    valid = mask & np.isfinite(sed_array) & np.isfinite(stiffness_mpa)
    if not np.any(valid):
        return {"status": "not_computed", "reason": "no active finite SED/modulus values"}
    ees = np.sqrt(np.maximum(0.0, 2.0 * sed_array[valid] / stiffness_mpa[valid]))
    percentile = max(0.0, min(100.0, 100.0 - float(critical_volume_percent)))
    ees_at_critical_volume = float(np.percentile(ees, percentile))
    factor = (
        None
        if np.isclose(ees_at_critical_volume, 0.0)
        else float(critical_strain) / ees_at_critical_volume
    )
    return {
        "status": "computed" if factor is not None else "not_computed",
        "criterion": "pistoia",
        "critical_strain": float(critical_strain),
        "critical_volume_percent": float(critical_volume_percent),
        "ees_at_critical_volume": ees_at_critical_volume,
        "factor": factor,
    }


def _fit_indices(count: int, *, max_fit_voxels: int | None) -> np.ndarray:
    if max_fit_voxels is None or int(max_fit_voxels) <= 0 or count <= int(max_fit_voxels):
        return np.arange(count)
    return np.linspace(0, count - 1, int(max_fit_voxels), dtype=np.int64)


def _input_load_amplitudes(values, *, expected_count: int) -> np.ndarray:
    if values is None:
        return np.ones(expected_count, dtype=np.float64)
    amplitudes = np.asarray(values, dtype=np.float64).reshape(-1)
    if amplitudes.shape != (expected_count,):
        raise ValueError("input_load_amplitudes must match the number of load cases")
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes <= 0.0):
        raise ValueError("input_load_amplitudes must contain positive finite values")
    return amplitudes


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


def _read_stiffness(path: str | Path) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() == ".h5":
        import h5py

        with h5py.File(p, "r") as h5:
            if "Image_Data/Image" not in h5:
                raise ValueError(f"HDF5 file does not contain Image_Data/Image: {p}")
            return np.asarray(h5["Image_Data/Image"][...])
    return _read_array(p)


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
