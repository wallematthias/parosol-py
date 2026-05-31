from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .api import SolveResult, solve, solve_aim
from .materials import parse_linear_isotropic_materials
from .reports import solve_summary_dict, write_summary_json


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        return tomllib.loads(config_path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML config files") from exc
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return {} if loaded is None else loaded
    raise ValueError("config file must be .json, .toml, .yaml, or .yml")


def run_case_config(
    path: str | Path,
    *,
    dry_run: bool | None = None,
    work_dir: str | Path | None = None,
) -> SolveResult:
    config_path = Path(path).expanduser().resolve()
    config = load_config(config_path)
    base_dir = config_path.parent

    case_cfg = _section(config, "case")
    input_cfg = _section(config, "input")
    material_cfg = _section(config, "materials")
    load_case_cfg = _section(config, "load_case")
    solver_cfg = _section(config, "solver")
    output_cfg = _section(config, "output")
    failure_cfg = _section(config, "failure")

    case_name = str(case_cfg.get("name") or config_path.stem)
    run_dir = _resolve_path(
        work_dir
        if work_dir is not None
        else case_cfg.get("work_dir", output_cfg.get("work_dir", case_name)),
        base_dir=base_dir,
    )
    export_dir = _resolve_path(output_cfg.get("fields_dir", run_dir), base_dir=base_dir)
    summary_path = _resolve_path(
        output_cfg.get("summary", run_dir / f"{case_name}_summary.json"),
        base_dir=base_dir,
    )

    load_type = str(load_case_cfg.get("type", "axial")).strip().lower()
    if load_type not in {"axial", "compression"}:
        raise NotImplementedError(
            "Only axial/compression load cases are implemented in this CLI pass"
        )

    dry = bool(output_cfg.get("dry_run", False) if dry_run is None else dry_run)
    outputs = tuple(str(v) for v in solver_cfg.get("outputs", ("sed",)))
    image_path = _resolve_path(input_cfg["image"], base_dir=base_dir)

    common = {
        "spacing": _spacing(input_cfg, image_path=image_path),
        "origin": _origin(input_cfg, image_path=image_path),
        "material_unit": str(material_cfg.get("units", "MPa")),
        "poisson_ratio": float(material_cfg.get("poisson_ratio", material_cfg.get("nu", 0.3))),
        "test": "axial",
        "test_axis": str(load_case_cfg.get("axis", "z")),
        "strain": float(load_case_cfg.get("strain", load_case_cfg.get("normal_strain", -0.01))),
        "outputs": outputs,
        "tolerance": float(solver_cfg.get("tolerance", solver_cfg.get("convergence_tolerance", 1e-6))),
        "level": int(solver_cfg.get("level", 6)),
        "work_dir": run_dir,
        "export_dir": None if dry else export_dir,
        "dry_run": dry,
    }

    image_type = str(input_cfg.get("image_type", "material_mpa")).strip().lower()
    if image_path.suffix.lower() == ".aim" and image_type in {"material_mpa", "mpa", "gpa"}:
        result = solve_aim(image_path, **common)
    else:
        material = _load_material_array(
            image_path,
            image_type=image_type,
            material_cfg=material_cfg,
            base_dir=base_dir,
        )
        result = solve(material=material, array_order="zyx", **common)

    summary = solve_summary_dict(
        result,
        extra={
            "case": {"name": case_name},
            "load_case": {
                "type": load_type,
                "axis": common["test_axis"],
                "strain": common["strain"],
            },
            "failure": {
                "criterion": failure_cfg.get("criterion", "pistoia"),
                "critical_strain": failure_cfg.get("critical_strain"),
                "critical_volume_percent": failure_cfg.get("critical_volume_percent"),
                "status": "not_computed",
            },
        },
    )
    write_summary_json(summary_path, summary)
    return result


def _load_material_array(
    image_path: Path,
    *,
    image_type: str,
    material_cfg: dict[str, Any],
    base_dir: Path,
) -> np.ndarray:
    array_zyx = _read_image_array_zyx(image_path)
    if image_type in {"material_mpa", "mpa", "material"}:
        return array_zyx.astype(np.float64, copy=False)
    if image_type in {"material_gpa", "gpa"}:
        return array_zyx.astype(np.float64, copy=False)
    if image_type not in {"material_labels", "labels", "segmentation"}:
        raise ValueError(
            "input.image_type must be material_mpa, material_gpa, or material_labels"
        )

    materials_file = material_cfg.get("file")
    if materials_file is None:
        raise ValueError("materials.file is required when input.image_type is material_labels")
    table = parse_linear_isotropic_materials(
        _resolve_path(materials_file, base_dir=base_dir).read_text(encoding="utf-8")
    )
    labels = np.asarray(array_zyx)
    out = np.zeros(labels.shape, dtype=np.float64)
    for label, youngs_mpa in table.youngs_modulus_mpa.items():
        out[labels == label] = float(youngs_mpa)
    return out


def _read_image_array_zyx(path: Path) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        return np.load(path)
    if suffixes.endswith((".mha", ".mhd", ".nii", ".nii.gz")):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if suffixes.endswith(".aim"):
        from .api import read_aim

        array, _meta = read_aim(str(path))
        return np.asarray(array)
    raise ValueError(f"Unsupported input image format: {path}")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section '{name}' must be a table/object")
    return value


def _resolve_path(value, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _spacing(input_cfg: dict[str, Any], *, image_path: Path) -> tuple[float, float, float]:
    value = input_cfg.get("spacing", (1.0, 1.0, 1.0))
    if str(value).strip().lower() == "auto":
        spacing, _origin = _image_metadata(image_path)
        if spacing is None:
            raise ValueError("spacing='auto' is available only for image formats with metadata")
        return spacing
    return _triple(value, "input.spacing")


def _origin(input_cfg: dict[str, Any], *, image_path: Path) -> tuple[float, float, float]:
    value = input_cfg.get("origin", (0.0, 0.0, 0.0))
    if str(value).strip().lower() == "auto":
        _spacing, origin = _image_metadata(image_path)
        return (0.0, 0.0, 0.0) if origin is None else origin
    return _triple(value, "input.origin")


def _triple(value, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    return tuple(float(v) for v in value)


def _image_metadata(path: Path) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".mha", ".mhd", ".nii", ".nii.gz")):
        image = sitk.ReadImage(str(path))
        return _triple(image.GetSpacing(), "image spacing"), _triple(image.GetOrigin(), "image origin")
    if suffixes.endswith(".aim"):
        from .api import read_aim

        _array, meta = read_aim(str(path))
        spacing = meta.get("element_size")
        origin = meta.get("position")
        return (
            None if spacing is None else _triple(spacing, "AIM element_size"),
            None if origin is None else _triple(origin, "AIM position"),
        )
    return None, None
