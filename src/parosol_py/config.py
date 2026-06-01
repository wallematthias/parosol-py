from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .api import SolveResult, solve, solve_aim
from .core import Model
from .images import normalize_array
from .load_cases import (
    BodyWeightCompression,
    ConfinedCompression,
    SimpleShear,
    UniaxialCompression,
)
from .materials import parse_linear_isotropic_materials
from .nodesets import boundary_conditions_from_nodesets, nodes_from_labeled_voxels
from .profiles import get_output_profile, get_solver_profile
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
    nodeset_cfg = _section(config, "nodesets")
    load_case_cfg = _section(config, "load_case")
    solver_cfg = _section(config, "solver")
    output_cfg = _section(config, "output")
    failure_cfg = _section(config, "failure")
    solver_profile = get_solver_profile(config.get("solver_profile"))
    output_profile = get_output_profile(config.get("output_profile"))

    case_name = str(case_cfg.get("name") or config_path.stem)
    run_dir = _resolve_path(
        work_dir
        if work_dir is not None
        else case_cfg.get("work_dir", output_cfg.get("work_dir", case_name)),
        base_dir=base_dir,
    )
    export_dir = _resolve_path(output_cfg.get("fields_dir", run_dir), base_dir=base_dir)
    export_fields = bool(
        output_cfg.get(
            "export_fields",
            output_cfg.get("fields", output_profile.export_fields),
        )
    )
    summary_path = _resolve_path(
        output_cfg.get("summary", run_dir / f"{case_name}_summary.json"),
        base_dir=base_dir,
    )

    dry = bool(output_cfg.get("dry_run", False) if dry_run is None else dry_run)
    outputs = tuple(str(v) for v in solver_cfg.get("outputs", solver_profile.outputs))
    image_path = _resolve_path(input_cfg["image"], base_dir=base_dir)
    image_type = str(input_cfg.get("image_type", "material_mpa")).strip().lower()
    spacing = _spacing(input_cfg, image_path=image_path)
    origin = _origin(input_cfg, image_path=image_path)
    poisson_ratio = _poisson_ratio(
        material_cfg,
        image_type=image_type,
        base_dir=base_dir,
    )

    common = {
        "spacing": spacing,
        "origin": origin,
        "material_unit": str(material_cfg.get("units", "MPa")),
        "poisson_ratio": poisson_ratio,
        "test": "axial",
        "test_axis": str(load_case_cfg.get("axis", "z")),
        "strain": float(
            load_case_cfg.get("strain", load_case_cfg.get("normal_strain", -0.01))
        ),
        "outputs": outputs,
        "tolerance": float(
            solver_cfg.get(
                "tolerance",
                solver_cfg.get("convergence_tolerance", solver_profile.tolerance),
            )
        ),
        "level": int(solver_cfg.get("level", solver_profile.level)),
        "mpi_processes": int(
            solver_cfg.get(
                "mpi_processes",
                solver_cfg.get("processes", solver_profile.mpi_processes),
            )
        ),
        "mpi_launcher": str(solver_cfg.get("mpi_launcher", solver_profile.mpi_launcher)),
        "work_dir": run_dir,
        "export_dir": export_dir if export_fields and not dry else None,
        "failure_criterion": str(failure_cfg.get("criterion", "pistoia")),
        "critical_strain": _optional_float(failure_cfg.get("critical_strain", 0.007)),
        "critical_volume_percent": _optional_float(
            failure_cfg.get("critical_volume_percent", 2.0)
        ),
        "dry_run": dry,
    }

    if image_path.suffix.lower() == ".aim" and image_type in {
        "material_mpa",
        "mpa",
        "gpa",
    }:
        if nodeset_cfg:
            raise ValueError(
                "nodeset load cases with AIM inputs must use image_type='material_labels' "
                "or another path that can be read as an array"
            )
        result = solve_aim(image_path, **common)
    else:
        material = _load_material_array(
            image_path,
            image_type=image_type,
            material_cfg=material_cfg,
            base_dir=base_dir,
        )
        boundary_conditions = _boundary_conditions_from_config(
            material,
            spacing=spacing,
            origin=origin,
            array_order="zyx",
            nodeset_cfg=nodeset_cfg,
            load_case_cfg=load_case_cfg,
            base_dir=base_dir,
        )
        if boundary_conditions is not None:
            common["boundary_conditions"] = boundary_conditions
        result = solve(material=material, array_order="zyx", **common)

    load_type = str(load_case_cfg.get("type", "axial")).strip().lower()
    extra: dict[str, Any] = {
        "case": {"name": case_name},
        "load_case": {
            "type": load_type,
            "axis": common["test_axis"],
            "strain": common["strain"],
        },
    }
    if dry:
        extra["failure"] = {
            "criterion": common["failure_criterion"],
            "critical_strain": common["critical_strain"],
            "critical_volume_percent": common["critical_volume_percent"],
            "status": "not_computed",
        }
    summary = solve_summary_dict(result, extra=extra)
    write_summary_json(summary_path, summary)
    return result


def _boundary_conditions_from_config(
    material_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    array_order: str,
    nodeset_cfg: dict[str, Any],
    load_case_cfg: dict[str, Any],
    base_dir: Path,
):
    load_type = str(load_case_cfg.get("type", "axial")).strip().lower()
    if load_type in {"axial", "compression"}:
        if nodeset_cfg:
            raise ValueError(
                "nodesets were configured but load_case.type is axial; use type='nodeset'"
            )
        return None
    if load_type in {"uniaxial", "uniaxial_compression"}:
        if nodeset_cfg:
            raise ValueError(
                "nodesets were configured but load_case.type is uniaxial; "
                "use type='nodeset'"
            )
        model = Model.from_array(
            material_zyx,
            spacing=spacing,
            origin=origin,
            array_order=array_order,
        )
        return UniaxialCompression(
            axis=str(load_case_cfg.get("axis", "z")),
            strain=float(
                load_case_cfg.get("strain", load_case_cfg.get("normal_strain", -0.01))
            ),
        ).generate(model)
    if load_type in {"shear", "simple_shear", "directional_shear"}:
        if nodeset_cfg:
            raise ValueError(
                "nodesets were configured but load_case.type is shear; use type='nodeset'"
            )
        model = Model.from_array(
            material_zyx,
            spacing=spacing,
            origin=origin,
            array_order=array_order,
        )
        return SimpleShear(
            axis=str(load_case_cfg.get("axis", "z")),
            direction=str(load_case_cfg.get("direction", "x")),
            strain=float(load_case_cfg.get("strain", 0.01)),
        ).generate(model)
    if load_type in {"body_weight", "force", "force_compression"}:
        if nodeset_cfg:
            raise ValueError(
                "nodesets were configured but load_case.type is body_weight; "
                "use type='nodeset'"
            )
        model = Model.from_array(
            material_zyx,
            spacing=spacing,
            origin=origin,
            array_order=array_order,
        )
        force = load_case_cfg.get(
            "force_n",
            load_case_cfg.get("force", load_case_cfg.get("value", -1.0)),
        )
        return BodyWeightCompression(
            axis=str(load_case_cfg.get("axis", "z")),
            force_n=float(force),
        ).generate(model)
    if load_type in {"confined", "confined_compression"}:
        if nodeset_cfg:
            raise ValueError(
                "nodesets were configured but load_case.type is confined; "
                "use type='nodeset'"
            )
        model = Model.from_array(
            material_zyx,
            spacing=spacing,
            origin=origin,
            array_order=array_order,
        )
        return ConfinedCompression(
            axis=str(load_case_cfg.get("axis", "z")),
            strain=float(
                load_case_cfg.get("strain", load_case_cfg.get("normal_strain", -0.01))
            ),
        ).generate(model)
    if load_type not in {"nodeset", "custom"}:
        raise NotImplementedError(
            "load_case.type must be axial/compression, shear, body_weight, "
            "confined, uniaxial, or nodeset/custom"
        )
    if not nodeset_cfg:
        raise ValueError("load_case.type='nodeset' requires a nodesets section")

    material_grid = normalize_array(
        material_zyx,
        spacing=spacing,
        origin=origin,
        array_order=array_order,
    )
    node_sets = _load_node_sets(
        nodeset_cfg,
        material_xyz=material_grid.array_xyz,
        spacing=spacing,
        origin=origin,
        base_dir=base_dir,
    )
    return boundary_conditions_from_nodesets(
        node_sets,
        fixed=list(load_case_cfg.get("fixed", ())),
        prescribed=list(load_case_cfg.get("prescribed", ())),
        loaded=list(load_case_cfg.get("loaded", ())),
        dimensions_xyz=tuple(int(v) for v in material_grid.array_xyz.shape),
        spacing=material_grid.spacing,
    )


def _load_node_sets(
    nodeset_cfg: dict[str, Any],
    *,
    material_xyz: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    base_dir: Path,
) -> dict[str, list[tuple[int, int, int]]]:
    out: dict[str, list[tuple[int, int, int]]] = {}
    for name, spec in nodeset_cfg.items():
        if not isinstance(spec, dict):
            raise ValueError(f"nodesets.{name} must be a table/object")
        nodeset_type = str(spec.get("type", "label_image")).strip().lower()
        if nodeset_type != "label_image":
            raise NotImplementedError("Only label_image nodesets are implemented")
        labels_path = _resolve_path(spec["image"], base_dir=base_dir)
        labels_zyx = _read_image_array_zyx(labels_path)
        label_grid = normalize_array(
            labels_zyx,
            spacing=spacing,
            origin=origin,
            array_order="zyx",
        )
        if label_grid.array_xyz.shape != material_xyz.shape:
            raise ValueError(
                f"nodeset image '{labels_path}' shape {label_grid.array_xyz.shape} "
                f"does not match material image shape {material_xyz.shape}"
            )
        out[str(name)] = nodes_from_labeled_voxels(
            label_grid.array_xyz,
            label=int(spec["label"]),
            selection=str(spec.get("selection", "surface_nodes")),
            material=material_xyz,
        )
    return out


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
        raise ValueError(
            "materials.file is required when input.image_type is material_labels"
        )
    table = parse_linear_isotropic_materials(
        _resolve_path(materials_file, base_dir=base_dir).read_text(encoding="utf-8")
    )
    labels = np.asarray(array_zyx)
    out = np.zeros(labels.shape, dtype=np.float64)
    for label, youngs_mpa in table.youngs_modulus_mpa.items():
        out[labels == label] = float(youngs_mpa)
    return out


def _poisson_ratio(
    material_cfg: dict[str, Any],
    *,
    image_type: str,
    base_dir: Path,
) -> float:
    explicit = material_cfg.get("poisson_ratio", material_cfg.get("nu"))
    if explicit is not None:
        return float(explicit)
    if image_type not in {"material_labels", "labels", "segmentation"}:
        return 0.3
    materials_file = material_cfg.get("file")
    if materials_file is None:
        return 0.3
    table = parse_linear_isotropic_materials(
        _resolve_path(materials_file, base_dir=base_dir).read_text(encoding="utf-8")
    )
    values = sorted({round(float(value), 12) for value in table.poisson_ratio.values()})
    if len(values) > 1:
        raise ValueError(
            "native ParOSol currently supports one global Poisson's ratio; "
            f"material table contains multiple values: {values}"
        )
    return float(values[0])


def _read_image_array_zyx(path: Path) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        return np.load(path)
    if suffixes.endswith(".npz"):
        with np.load(path) as data:
            if "labels" in data:
                return np.asarray(data["labels"])
            if "image" in data:
                return np.asarray(data["image"])
            keys = list(data.files)
            if len(keys) == 1:
                return np.asarray(data[keys[0]])
            raise ValueError(
                f"NPZ image files must contain 'labels', 'image', or one array; got {keys}"
            )
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


def _spacing(
    input_cfg: dict[str, Any], *, image_path: Path
) -> tuple[float, float, float]:
    value = input_cfg.get("spacing", (1.0, 1.0, 1.0))
    if str(value).strip().lower() == "auto":
        spacing, _origin = _image_metadata(image_path)
        if spacing is None:
            raise ValueError(
                "spacing='auto' is available only for image formats with metadata"
            )
        return spacing
    return _triple(value, "input.spacing")


def _origin(
    input_cfg: dict[str, Any], *, image_path: Path
) -> tuple[float, float, float]:
    value = input_cfg.get("origin", (0.0, 0.0, 0.0))
    if str(value).strip().lower() == "auto":
        _spacing, origin = _image_metadata(image_path)
        return (0.0, 0.0, 0.0) if origin is None else origin
    return _triple(value, "input.origin")


def _triple(value, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    return tuple(float(v) for v in value)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _image_metadata(
    path: Path,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npz"):
        with np.load(path) as data:
            spacing = _npz_metadata_triple(data, "spacing_xyz", "spacing")
            origin = _npz_metadata_triple(data, "origin_xyz", "origin")
            return spacing, origin
    if suffixes.endswith((".mha", ".mhd", ".nii", ".nii.gz")):
        image = sitk.ReadImage(str(path))
        return _triple(image.GetSpacing(), "image spacing"), _triple(
            image.GetOrigin(), "image origin"
        )
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


def _npz_metadata_triple(
    data: np.lib.npyio.NpzFile,
    preferred_key: str,
    fallback_key: str,
) -> tuple[float, float, float] | None:
    key = preferred_key if preferred_key in data else fallback_key
    if key not in data:
        return None
    values = np.asarray(data[key]).reshape(-1)
    return _triple(values.tolist(), key)
