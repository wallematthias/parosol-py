from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from parosol_py.paths import suffix_text


def resolve_path(value, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def read_image_zyx(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    suffixes = suffix_text(path)
    if suffixes.endswith(".npy"):
        return np.load(path), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    if suffixes.endswith(".npz"):
        with np.load(path) as data:
            key = (
                "image"
                if "image" in data
                else "labels"
                if "labels" in data
                else data.files[0]
            )
            spacing = _npz_triple(data, "spacing_xyz", "spacing") or (1.0, 1.0, 1.0)
            origin = _npz_triple(data, "origin_xyz", "origin") or (0.0, 0.0, 0.0)
            return np.asarray(data[key]), spacing, origin
    if suffixes.endswith((".mha", ".mhd", ".nii", ".nii.gz")):
        image = sitk.ReadImage(str(path))
        image = sitk.DICOMOrient(image, "LPS")
        return (
            sitk.GetArrayFromImage(image),
            tuple(float(v) for v in image.GetSpacing()),
            tuple(float(v) for v in image.GetOrigin()),
        )
    if suffixes.endswith(".aim"):
        from parosol_py.api import read_aim

        array, meta = read_aim(str(path))
        spacing = tuple(float(v) for v in meta.get("element_size", (1.0, 1.0, 1.0)))
        origin = tuple(float(v) for v in meta.get("position", (0.0, 0.0, 0.0)))
        return np.asarray(array), spacing, origin
    raise ValueError(f"Unsupported model image format: {path}")


def _npz_triple(data: np.lib.npyio.NpzFile, preferred: str, fallback: str):
    key = preferred if preferred in data else fallback
    if key not in data:
        return None
    values = np.asarray(data[key]).reshape(-1)
    if values.size != 3:
        raise ValueError(f"{key} must contain exactly three values")
    return tuple(float(v) for v in values)
