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
from .images import coarsen_array, largest_connected_component, normalize_array
from .load_cases import (
    Bending,
    BodyWeightCompression,
    ConfinedCompression,
    ConstrainedAxialCompression,
    SimpleShear,
    Torsion,
    UniaxialCompression,
)
from .materials import (
    density_to_material_map,
    labels_to_material_map,
    linear_isotropic_materials_from_config,
    parse_linear_isotropic_materials,
    poisson_ratio_from_spec,
)
from .modeling import build_model
from .nodesets import boundary_conditions_from_nodesets, nodes_from_labeled_voxels
from .paths import suffix_text
from .profiles import get_output_profile, get_solver_profile
from .reports import compact_summary_dict, solve_summary_dict, write_summary_json
from .set_export import write_element_sets, write_node_sets
from .visualization import dense_scalar_field, write_case_overview


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
    model_cfg = _section(config, "model")
    material_cfg = _section(config, "materials")
    nodeset_cfg = _section(config, "nodesets")
    load_case_cfg = _section(config, "load_case")
    solver_cfg = _section(config, "solver")
    output_cfg = _section(config, "output")
    preprocessing_cfg = _section(config, "preprocessing")
    postprocess_cfg = _section(config, "postprocess")
    pistoia_cfg = _pistoia_config(postprocess_cfg)
    failure_load_cfg = _failure_load_config(postprocess_cfg)
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
    output_fields = _output_fields(output_cfg, output_profile)
    export_fields = bool(output_cfg.get("export_fields", output_profile.export_fields))
    result_path = _resolve_path(
        output_cfg.get(
            "result",
            output_cfg.get("summary", run_dir / "result.json"),
        ),
        base_dir=base_dir,
    )

    dry = bool(output_cfg.get("dry_run", False) if dry_run is None else dry_run)
    outputs = tuple(
        str(v)
        for v in solver_cfg.get(
            "outputs",
            output_fields if output_fields else solver_profile.outputs,
        )
    )
    image_path = None
    if model_cfg:
        image_type = "model"
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
    else:
        image_path = _resolve_path(input_cfg["image"], base_dir=base_dir)
        image_type = str(input_cfg.get("image_type", "material_mpa")).strip().lower()
        spacing = _spacing(input_cfg, image_path=image_path)
        origin = _origin(input_cfg, image_path=image_path)
    poisson_ratio = _poisson_ratio(
        material_cfg, image_type=image_type, base_dir=base_dir
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
        "load_case_type": str(load_case_cfg.get("type", "constrained_axial"))
        .strip()
        .lower(),
        "load_direction": load_case_cfg.get("direction"),
        "rotation_degrees": _load_case_rotation_degrees(load_case_cfg),
        "load_case_center": _load_case_center(load_case_cfg),
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
        "mpi_launcher": str(
            solver_cfg.get("mpi_launcher", solver_profile.mpi_launcher)
        ),
        "stream_output": bool(solver_cfg.get("stream_output", not dry)),
        "work_dir": run_dir,
        "export_dir": export_dir if export_fields and not dry else None,
        "failure_criterion": str(pistoia_cfg.get("criterion", "pistoia")),
        "critical_strain": _optional_float(pistoia_cfg.get("critical_strain", 0.007)),
        "critical_volume_percent": _optional_float(
            pistoia_cfg.get("critical_volume_percent", 2.0)
        ),
        "linear_failure_deformation": float(
            failure_load_cfg.get("linear_deformation", 0.002)
        ),
        "crawford_coefficient": float(
            failure_load_cfg.get("crawford_coefficient", 0.0068)
        ),
        "linear_failure_estimates": _linear_failure_estimates_enabled(
            postprocess_cfg
        ),
        "dry_run": dry,
    }

    built_model = None
    if model_cfg:
        built_model = build_model(
            model_cfg,
            base_dir=base_dir,
            material_config=material_cfg,
            load_case_config=load_case_cfg,
            preprocessing_config=preprocessing_cfg,
        )
        material = built_model.material
        spacing = built_model.spacing
        origin = built_model.origin
        common["spacing"] = spacing
        common["origin"] = origin
        common["poisson_ratio"] = built_model.poisson_ratio
        model_meta = built_model.metadata.get("model", {})
        common["test_axis"] = str(model_meta.get("load_axis", common["test_axis"]))
        common["load_direction"] = model_meta.get(
            "load_direction", common["load_direction"]
        )
        common["strain"] = _effective_strain_for_displacement(
            material,
            spacing=spacing,
            origin=origin,
            array_order="zyx",
            load_case_cfg=load_case_cfg,
            fallback=float(common["strain"]),
            axis=str(common["test_axis"]),
        )
        common["boundary_conditions"] = built_model.boundary_conditions
        if _mask_fields_to_segmentation(postprocess_cfg):
            common["postprocess_mask"] = built_model.postprocess_mask
        result = solve(material=material, array_order="zyx", **common)
        result.exported.update(built_model.exported)
        result.exported.update(
            _export_overview(
                material,
                spacing=spacing,
                origin=origin,
                output_cfg=output_cfg,
                base_dir=base_dir,
                run_dir=run_dir,
                case_name=case_name,
                result=result,
                boundary_conditions=built_model.boundary_conditions,
                field_mask_zyx=built_model.postprocess_mask
                if _mask_fields_to_segmentation(postprocess_cfg)
                else None,
            )
        )
    elif suffix_text(image_path).endswith(".aim") and image_type in {
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
        material, mapped_poisson_ratio = _load_material_array(
            image_path,
            image_type=image_type,
            material_cfg=material_cfg,
            base_dir=base_dir,
            fallback_poisson_ratio=poisson_ratio,
        )
        if _connectivity_filter_enabled(preprocessing_cfg):
            material = largest_connected_component(material)
        material, spacing = _coarsen_material(
            material,
            spacing=spacing,
            preprocessing_cfg=preprocessing_cfg,
        )
        common["spacing"] = spacing
        common["poisson_ratio"] = mapped_poisson_ratio
        common["strain"] = _effective_strain_for_displacement(
            material,
            spacing=spacing,
            origin=origin,
            array_order="zyx",
            load_case_cfg=load_case_cfg,
            fallback=float(common["strain"]),
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
            if output_cfg.get("export_boundary_conditions", False):
                bc_path = _resolve_path(
                    output_cfg.get(
                        "boundary_conditions",
                        run_dir / f"{case_name}_boundary_conditions.json",
                    ),
                    base_dir=base_dir,
                )
                write_summary_json(bc_path, boundary_conditions.to_dict())
        debug_boundary_conditions = boundary_conditions
        if debug_boundary_conditions is None and (
            output_cfg.get("export_sets", False) or _visualization_enabled(output_cfg)
        ):
            debug_boundary_conditions = _default_boundary_conditions_for_export(
                material,
                spacing=spacing,
                origin=origin,
                load_case_cfg=load_case_cfg,
            )
        set_exports = _export_debug_sets(
            material,
            spacing=spacing,
            origin=origin,
            output_cfg=output_cfg,
            base_dir=base_dir,
            boundary_conditions=debug_boundary_conditions,
        )
        result = solve(material=material, array_order="zyx", **common)
        result.exported.update(set_exports)
        result.exported.update(
            _export_overview(
                material,
                spacing=spacing,
                origin=origin,
                output_cfg=output_cfg,
                base_dir=base_dir,
                run_dir=run_dir,
                case_name=case_name,
                result=result,
                boundary_conditions=debug_boundary_conditions,
            )
        )

    load_type = str(load_case_cfg.get("type", "constrained_axial")).strip().lower()
    extra: dict[str, Any] = {
        "case": {"name": case_name},
        "load_case": {
            "type": load_type,
            "axis": common["test_axis"],
            "strain": common["strain"],
        },
    }
    extra["execution"] = {
        "config": str(config_path),
        "work_dir": str(run_dir),
        "dry_run": dry,
        **_section(config, "execution"),
    }
    if dry:
        extra["failure"] = {
            "criterion": common["failure_criterion"],
            "critical_strain": common["critical_strain"],
            "critical_volume_percent": common["critical_volume_percent"],
            "status": "not_computed",
        }
    if built_model is not None:
        extra["model"] = _model_summary(built_model)
    extra["quality"] = _quality_config(solver_cfg)
    summary_path = _resolve_path(
        output_cfg.get("run_summary", result_path.with_name("summary.json")),
        base_dir=base_dir,
    )
    summary = solve_summary_dict(result, extra=extra)
    exported = summary.setdefault("outputs", {}).setdefault("exported", {})
    exported["result"] = str(result_path)
    exported["summary"] = str(summary_path)
    write_summary_json(summary_path, summary)
    write_summary_json(result_path, compact_summary_dict(summary))
    result.exported["result"] = result_path
    result.exported["summary"] = summary_path
    return result


def _output_fields(output_cfg: dict[str, Any], output_profile) -> tuple[str, ...]:
    fields = output_cfg.get("fields", output_cfg.get("image_fields"))
    if fields is None:
        return tuple(output_profile.image_fields)
    return tuple(str(value) for value in fields)


def _pistoia_config(postprocess_cfg: dict[str, Any]) -> dict[str, Any]:
    pistoia = postprocess_cfg.get("pistoia", {})
    if pistoia is False:
        return {
            "criterion": "none",
            "critical_strain": None,
            "critical_volume_percent": None,
        }
    if pistoia is True:
        return {}
    if not isinstance(pistoia, dict):
        raise ValueError("postprocess.pistoia must be a table/object or boolean")
    return pistoia


def _failure_load_config(postprocess_cfg: dict[str, Any]) -> dict[str, Any]:
    failure_load = postprocess_cfg.get("failure_load", {})
    if failure_load is False:
        return {"linear_deformation": 0.002, "crawford_coefficient": 0.0068}
    if failure_load is True:
        return {}
    if not isinstance(failure_load, dict):
        raise ValueError("postprocess.failure_load must be a table/object or boolean")
    return failure_load


def _linear_failure_estimates_enabled(postprocess_cfg: dict[str, Any]) -> bool:
    failure_load = postprocess_cfg.get("failure_load", False)
    if failure_load is False or failure_load is None:
        return False
    return bool(failure_load)


def _mask_fields_to_segmentation(postprocess_cfg: dict[str, Any]) -> bool:
    fields = postprocess_cfg.get("fields", {})
    if isinstance(fields, dict):
        return bool(fields.get("mask_to_segmentation", False))
    return bool(postprocess_cfg.get("mask_to_segmentation", False))


def _coarsen_material(
    material: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    preprocessing_cfg: dict[str, Any],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    coarsen = preprocessing_cfg.get("coarsen")
    if not coarsen:
        return material, spacing
    if isinstance(coarsen, dict):
        factor = int(coarsen.get("factor", 1))
        reducer = str(coarsen.get("reducer", "mean"))
    else:
        factor = int(coarsen)
        reducer = "mean"
    if factor == 1:
        return material, spacing
    return coarsen_array(material, factor=factor, reducer=reducer), tuple(
        float(value) * factor for value in spacing
    )


def _export_debug_sets(
    material_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    output_cfg: dict[str, Any],
    base_dir: Path,
    boundary_conditions,
) -> dict[str, Path]:
    if not output_cfg.get("export_sets", False):
        return {}
    formats = tuple(str(value) for value in output_cfg.get("set_formats", ("json",)))
    sets_dir = _resolve_path(
        output_cfg.get("sets_dir", output_cfg.get("fields_dir", "sets")),
        base_dir=base_dir,
    )
    grid = normalize_array(
        material_zyx,
        spacing=spacing,
        origin=origin,
        array_order="zyx",
    )
    out = write_element_sets(
        grid.array_xyz,
        directory=sets_dir,
        spacing=grid.spacing,
        origin=grid.origin,
        formats=formats,
    )
    if boundary_conditions is not None:
        out.update(
            write_node_sets(
                boundary_conditions.node_sets,
                directory=sets_dir,
                spacing=grid.spacing,
                origin=grid.origin,
                formats=formats,
            )
        )
    return out


def _export_overview(
    material_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    output_cfg: dict[str, Any],
    base_dir: Path,
    run_dir: Path,
    case_name: str,
    result: SolveResult,
    boundary_conditions,
    field_mask_zyx=None,
) -> dict[str, Path]:
    if not _visualization_enabled(output_cfg):
        return {}
    path_value = output_cfg.get("overview", output_cfg.get("visualization"))
    if path_value is None or isinstance(path_value, bool):
        path_value = run_dir / f"{case_name}_overview.png"
    overview_path = _resolve_path(path_value, base_dir=base_dir)
    grid = normalize_array(
        material_zyx,
        spacing=spacing,
        origin=origin,
        array_order="zyx",
    )
    field_mask = None
    if field_mask_zyx is not None:
        field_mask = normalize_array(
            field_mask_zyx,
            spacing=spacing,
            origin=origin,
            array_order="zyx",
        ).array_xyz.astype(bool)
    field_name = str(output_cfg.get("visualization_field", "sed")).strip().lower()
    field = _overview_field_from_export(
        result.exported.get(field_name),
        expected_shape=grid.array_xyz.shape,
    )
    if field is None:
        field = dense_scalar_field(grid.array_xyz, result.fields.get(field_name))
    out = write_case_overview(
        grid.array_xyz,
        output_path=overview_path,
        spacing=grid.spacing,
        origin=grid.origin,
        field_xyz=field,
        field_name=field_name.upper(),
        field_mask_xyz=field_mask,
        boundary_conditions=boundary_conditions,
        title=case_name,
    )
    return {"overview": out}


def _overview_field_from_export(
    path: Path | None,
    *,
    expected_shape: tuple[int, int, int],
) -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    image = sitk.ReadImage(str(path))
    field_xyz = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    if field_xyz.shape != expected_shape:
        return None
    field_xyz = np.asarray(field_xyz, dtype=np.float64)
    return np.where(field_xyz != 0.0, field_xyz, np.nan)


def _visualization_enabled(output_cfg: dict[str, Any]) -> bool:
    return bool(output_cfg.get("visualize", output_cfg.get("visualization", True)))


def _default_boundary_conditions_for_export(
    material_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    load_case_cfg: dict[str, Any],
):
    load_type = str(load_case_cfg.get("type", "constrained_axial")).strip().lower()
    if load_type not in {
        "axial",
        "compression",
        "constrained_axial",
        "plate_compression",
    }:
        return None
    model = Model.from_array(
        material_zyx,
        spacing=spacing,
        origin=origin,
        array_order="zyx",
    )
    return ConstrainedAxialCompression(
        axis=str(load_case_cfg.get("axis", "z")),
        strain=float(
            load_case_cfg.get("strain", load_case_cfg.get("normal_strain", -0.01))
        ),
        displacement=_load_case_displacement(load_case_cfg),
        surface=_load_case_surface(load_case_cfg),
    ).generate(model)


def _quality_config(solver_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "checks": {
            "max_relative_residual": _optional_float(
                solver_cfg.get("max_relative_residual")
            ),
            "max_iterations": (
                None
                if solver_cfg.get("max_iterations") is None
                else int(solver_cfg["max_iterations"])
            ),
        }
    }


def _model_summary(built_model) -> dict[str, Any]:
    metadata = dict(built_model.metadata.get("model", {}))
    return {
        **metadata,
        "node_sets": {
            name: len(nodes) for name, nodes in built_model.node_sets.items()
        },
        "element_sets": dict(built_model.element_sets),
        "exported": {name: str(path) for name, path in built_model.exported.items()},
    }


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
    load_type = str(load_case_cfg.get("type", "constrained_axial")).strip().lower()
    surface = _load_case_surface(load_case_cfg)
    if load_type in {"axial", "compression", "constrained_axial", "plate_compression"}:
        if nodeset_cfg:
            raise ValueError(
                "nodesets were configured but load_case.type is constrained axial; "
                "use type='nodeset'"
            )
        displacement = _load_case_displacement(load_case_cfg)
        if displacement is not None:
            model = Model.from_array(
                material_zyx,
                spacing=spacing,
                origin=origin,
                array_order=array_order,
            )
            return ConstrainedAxialCompression(
                axis=str(load_case_cfg.get("axis", "z")),
                strain=float(
                    load_case_cfg.get(
                        "strain", load_case_cfg.get("normal_strain", -0.01)
                    )
                ),
                displacement=displacement,
                surface=surface,
            ).generate(model)
        if surface is not None:
            model = Model.from_array(
                material_zyx,
                spacing=spacing,
                origin=origin,
                array_order=array_order,
            )
            return ConstrainedAxialCompression(
                axis=str(load_case_cfg.get("axis", "z")),
                strain=float(
                    load_case_cfg.get(
                        "strain", load_case_cfg.get("normal_strain", -0.01)
                    )
                ),
                surface=surface,
            ).generate(model)
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
            displacement=_load_case_displacement(load_case_cfg),
            surface=surface,
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
            displacement=_load_case_displacement(load_case_cfg),
            vector=_load_case_vector(load_case_cfg),
            surface=surface,
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
            surface=surface,
        ).generate(model)
    if load_type in {"torsion", "twist"}:
        if nodeset_cfg:
            raise ValueError(
                "nodesets were configured but load_case.type is torsion; "
                "use type='nodeset'"
            )
        model = Model.from_array(
            material_zyx,
            spacing=spacing,
            origin=origin,
            array_order=array_order,
        )
        return Torsion(
            axis=str(load_case_cfg.get("axis", "z")),
            twist_angle_degrees=float(
                load_case_cfg.get(
                    "twist_angle_degrees",
                    load_case_cfg.get("twist_angle", load_case_cfg.get("angle", 1.0)),
                )
            ),
            center=_load_case_center(load_case_cfg),
        ).generate(model)
    if load_type in {"bending", "bend"}:
        if nodeset_cfg:
            raise ValueError(
                "nodesets were configured but load_case.type is bending; "
                "use type='nodeset'"
            )
        model = Model.from_array(
            material_zyx,
            spacing=spacing,
            origin=origin,
            array_order=array_order,
        )
        return Bending(
            axis=str(load_case_cfg.get("axis", "z")),
            bending_angle_degrees=float(
                load_case_cfg.get(
                    "bending_angle_degrees",
                    load_case_cfg.get("bending_angle", load_case_cfg.get("angle", 1.0)),
                )
            ),
            neutral_axis_angle_degrees=float(
                load_case_cfg.get(
                    "neutral_axis_angle_degrees",
                    load_case_cfg.get("neutral_axis_angle", 90.0),
                )
            ),
            center=_load_case_center(load_case_cfg),
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
            displacement=_load_case_displacement(load_case_cfg),
            surface=surface,
        ).generate(model)
    if load_type not in {"nodeset", "custom"}:
        raise NotImplementedError(
            "load_case.type must be constrained_axial/plate_compression, "
            "uniaxial, confined, shear, torsion, bending, body_weight, "
            "or nodeset/custom"
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


def _load_case_displacement(load_case_cfg: dict[str, Any]) -> float | None:
    value = load_case_cfg.get("displacement", load_case_cfg.get("normal_displacement"))
    if value is None:
        return None
    return float(value)


def _load_case_vector(load_case_cfg: dict[str, Any]) -> tuple[float, float] | None:
    value = load_case_cfg.get("shear_vector", load_case_cfg.get("vector"))
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("load_case.shear_vector must contain exactly two values")
    return tuple(float(v) for v in value)


def _load_case_center(load_case_cfg: dict[str, Any]) -> tuple[float, float] | None:
    value = load_case_cfg.get("center", load_case_cfg.get("central_axis"))
    if value is None or str(value).strip().lower() in {
        "center",
        "center_of_mass",
        "center_of_bounds",
    }:
        return None
    if len(value) != 2:
        raise ValueError("load_case.center must contain exactly two values")
    return tuple(float(v) for v in value)


def _load_case_rotation_degrees(load_case_cfg: dict[str, Any]) -> float | None:
    load_type = str(load_case_cfg.get("type", "")).strip().lower()
    if load_type in {"torsion", "twist"}:
        return float(
            load_case_cfg.get(
                "twist_angle_degrees",
                load_case_cfg.get("twist_angle", load_case_cfg.get("angle", 1.0)),
            )
        )
    if load_type in {"bending", "bend"}:
        return float(
            load_case_cfg.get(
                "bending_angle_degrees",
                load_case_cfg.get("bending_angle", load_case_cfg.get("angle", 1.0)),
            )
        )
    return None


def _load_case_surface(load_case_cfg: dict[str, Any]):
    surface = load_case_cfg.get("surface", load_case_cfg.get("surfaces"))
    if surface is None:
        return None
    if isinstance(surface, dict) and ("top" in surface or "bottom" in surface):
        common = {
            key: value for key, value in surface.items() if key not in {"top", "bottom"}
        }
        top = surface.get("top", {})
        if isinstance(top, dict):
            common.update(top)
        return common
    return surface


def _effective_strain_for_displacement(
    material_zyx: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    array_order: str,
    load_case_cfg: dict[str, Any],
    fallback: float,
    axis: str | None = None,
) -> float:
    displacement = _load_case_displacement(load_case_cfg)
    if displacement is None:
        return fallback
    axis_token = str(
        axis if axis is not None else load_case_cfg.get("axis", "z")
    ).strip().lower()
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_token]
    grid = normalize_array(
        material_zyx,
        spacing=spacing,
        origin=origin,
        array_order=array_order,
    )
    height = grid.array_xyz.shape[axis_index] * grid.spacing[axis_index]
    if np.isclose(height, 0.0):
        raise ValueError("cannot convert displacement to strain for zero height")
    return float(displacement) / float(height)


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
    fallback_poisson_ratio: float,
) -> tuple[np.ndarray, float | np.ndarray]:
    array_zyx = _read_image_array_zyx(image_path)
    if image_type in {"material_mpa", "mpa", "material"}:
        return array_zyx.astype(np.float64, copy=False), _continuous_poisson_ratio(
            material_cfg,
            array_zyx,
            fallback=fallback_poisson_ratio,
        )
    if image_type in {"material_gpa", "gpa"}:
        return array_zyx.astype(np.float64, copy=False), _continuous_poisson_ratio(
            material_cfg,
            array_zyx,
            fallback=fallback_poisson_ratio,
        )
    if image_type in {"density", "density_mg_ha", "density_mgcm3", "rho"}:
        density_cfg = _section(material_cfg, "density")
        e_cfg = density_cfg.get("E", density_cfg.get("youngs_modulus", density_cfg))
        if not isinstance(e_cfg, dict):
            raise ValueError("materials.density.E must be an object")
        poisson_spec = density_cfg.get(
            "nu",
            density_cfg.get(
                "poisson_ratio",
                material_cfg.get("poisson_ratio", material_cfg.get("nu", 0.3)),
            ),
        )
        mapped = density_to_material_map(
            array_zyx,
            equation=str(e_cfg.get("equation", "power")),
            poisson_ratio=poisson_spec,
            mask_threshold=float(
                density_cfg.get(
                    "active_threshold", density_cfg.get("mask_threshold", 0.0)
                )
            ),
            minimum_e_mpa=float(
                e_cfg.get("minimum_e_mpa", density_cfg.get("minimum_e_mpa", 0.0))
            ),
            maximum_e_mpa=_optional_float(
                e_cfg.get("maximum_e_mpa", density_cfg.get("maximum_e_mpa"))
            ),
            **{
                key: value
                for key, value in e_cfg.items()
                if key
                not in {
                    "equation",
                    "minimum_e_mpa",
                    "maximum_e_mpa",
                }
            },
        )
        return mapped.youngs_modulus_mpa, mapped.poisson_ratio
    if image_type not in {"material_labels", "labels", "segmentation"}:
        raise ValueError(
            "input.image_type must be material_mpa, material_gpa, material_labels, or density"
        )

    materials_file = material_cfg.get("file")
    if materials_file is None:
        table = linear_isotropic_materials_from_config(material_cfg)
    else:
        table = parse_linear_isotropic_materials(
            _resolve_path(materials_file, base_dir=base_dir).read_text(encoding="utf-8")
        )
    mapped = labels_to_material_map(
        array_zyx,
        table,
        poisson_ratio=material_cfg.get("poisson_ratio", material_cfg.get("nu")),
    )
    return mapped.youngs_modulus_mpa, mapped.poisson_ratio


def _continuous_poisson_ratio(
    material_cfg: dict[str, Any],
    values: np.ndarray,
    *,
    fallback: float,
) -> float:
    density_cfg = material_cfg.get("density")
    if isinstance(density_cfg, dict):
        spec = density_cfg.get(
            "nu",
            density_cfg.get(
                "poisson_ratio",
                material_cfg.get("poisson_ratio", material_cfg.get("nu")),
            ),
        )
    else:
        spec = material_cfg.get("poisson_ratio", material_cfg.get("nu"))
    if spec is None:
        return fallback
    return poisson_ratio_from_spec(spec, values, active_mask=np.asarray(values) > 0)


def _connectivity_filter_enabled(preprocessing_cfg: dict[str, Any]) -> bool:
    value = preprocessing_cfg.get(
        "largest_cc", preprocessing_cfg.get("connectivity_filter", False)
    )
    if isinstance(value, str):
        return value.strip().lower() in {"on", "true", "yes", "largest"}
    return bool(value)


def _poisson_ratio(
    material_cfg: dict[str, Any],
    *,
    image_type: str,
    base_dir: Path,
) -> float:
    density_cfg = material_cfg.get("density")
    if isinstance(density_cfg, dict) and image_type in {
        "density",
        "density_mg_ha",
        "density_mgcm3",
        "rho",
    }:
        explicit = density_cfg.get(
            "nu",
            density_cfg.get(
                "poisson_ratio",
                material_cfg.get("poisson_ratio", material_cfg.get("nu")),
            ),
        )
    else:
        explicit = material_cfg.get("poisson_ratio", material_cfg.get("nu"))
    if explicit is not None:
        if isinstance(explicit, dict):
            return float(
                explicit.get(
                    "fallback", explicit.get("value", explicit.get("intercept", 0.3))
                )
            )
        return float(explicit)
    if image_type not in {"material_labels", "labels", "segmentation"}:
        return 0.3
    materials_file = material_cfg.get("file")
    if materials_file is None:
        try:
            table = linear_isotropic_materials_from_config(material_cfg)
        except ValueError:
            return 0.3
    else:
        table = parse_linear_isotropic_materials(
            _resolve_path(materials_file, base_dir=base_dir).read_text(encoding="utf-8")
        )
    values = sorted({round(float(value), 12) for value in table.poisson_ratio.values()})
    return float(values[0])


def _read_image_array_zyx(path: Path) -> np.ndarray:
    suffixes = suffix_text(path)
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
    suffixes = suffix_text(path)
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
