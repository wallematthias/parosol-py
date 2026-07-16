from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, run_case_config
from .load_history import estimate_load_history_from_files
from .reports import write_summary_json


def run_batch_config(
    path: str | Path,
    *,
    dry_run: bool | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    batch_path = Path(path).expanduser().resolve()
    config = load_config(batch_path)
    base_dir = batch_path.parent
    batch_cfg = _section(config, "batch")
    cases = list(batch_cfg.get("cases", ()))
    batch_work_dir = _resolve_path(
        work_dir
        if work_dir is not None
        else batch_cfg.get(
            "work_dir", _section(config, "case").get("work_dir", "batch")
        ),
        base_dir=base_dir,
    )
    summary_path = _resolve_path(
        batch_cfg.get("summary", batch_work_dir / "batch_summary.json"),
        base_dir=base_dir,
    )

    full_case_configs: list[dict[str, Any]] = []
    full_case_summaries: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for index, case_override in enumerate(cases):
        case_config = _case_config(
            config,
            case_override,
            index=index,
            base_dir=base_dir,
        )
        case_path = _write_case_config(
            case_config,
            batch_work_dir=batch_work_dir,
            case_name=case_config["case"]["name"],
        )
        full_case_configs.append(case_config)
        run_case_config(case_path, dry_run=dry_run, work_dir=None)
        case_summary_path = (
            case_config["output"].get("result") or case_config["output"]["summary"]
        )
        case_summary = _read_case_summary(Path(case_summary_path))
        full_case_summaries.append(case_summary)
        case_summaries.append(_compact_case_summary(case_summary))

    summary = {
        "batch": {
            "name": str(
                batch_cfg.get(
                    "name", _section(config, "case").get("name", batch_path.stem)
                )
            ),
            "case_count": len(case_summaries),
            "summary": str(summary_path),
            "work_dir": str(batch_work_dir),
        },
        "cases": case_summaries,
    }
    postprocess_cfg = _section(config, "postprocess")
    if postprocess_cfg:
        summary["postprocess"] = copy.deepcopy(postprocess_cfg)
        _run_batch_postprocess(
            summary["postprocess"],
            base_config=config,
            case_configs=full_case_configs,
            case_summaries=full_case_summaries,
            base_dir=base_dir,
            batch_work_dir=batch_work_dir,
            dry_run=bool(dry_run),
        )
    write_summary_json(summary_path, summary)
    return summary


def _run_batch_postprocess(
    postprocess_cfg: dict[str, Any],
    *,
    base_config: dict[str, Any],
    case_configs: list[dict[str, Any]],
    case_summaries: list[dict[str, Any]],
    base_dir: Path,
    batch_work_dir: Path,
    dry_run: bool,
) -> None:
    load_history_cfg = postprocess_cfg.get("load_history")
    if not isinstance(load_history_cfg, dict) or not load_history_cfg.get(
        "enabled", False
    ):
        return
    summary_path = _resolve_path(
        load_history_cfg.get("summary", "load_history_summary.json"),
        base_dir=base_dir,
    )
    output_path = _resolve_path(
        load_history_cfg.get("output", "load_history.nii.gz"),
        base_dir=base_dir,
    )
    load_history_cfg["summary"] = str(summary_path)
    load_history_cfg["output"] = str(output_path)
    if dry_run:
        load_history_cfg["status"] = "dry_run"
        return

    field = _load_history_field(load_history_cfg)
    load_case_paths = _load_history_field_paths(
        case_summaries,
        field=field,
        requested_cases=load_history_cfg.get("cases"),
    )
    selected_case_summaries = _load_history_case_summaries(
        case_summaries, requested_cases=load_history_cfg.get("cases")
    )
    bone_mask_path = load_history_cfg.get("bone_mask")
    resolved_bone_mask = (
        None
        if bone_mask_path is None
        else _resolve_path(bone_mask_path, base_dir=base_dir)
    )
    result = estimate_load_history_from_files(
        load_case_paths,
        bone_mask_path=resolved_bone_mask,
        output_path=output_path,
        summary_path=summary_path,
        target_average=float(load_history_cfg.get("target_average", 0.02)),
        cutoff_percentile=float(load_history_cfg.get("cutoff_percentile", 95.0)),
        max_fit_voxels=int(load_history_cfg.get("max_fit_voxels", 200_000)),
        stiffness_path=_first_input_file(case_summaries),
        critical_strain=float(load_history_cfg.get("critical_strain", 0.007)),
        critical_volume_percent=float(
            load_history_cfg.get("critical_volume_percent", 2.0)
        ),
        input_load_amplitudes=_case_input_load_amplitudes(
            selected_case_summaries,
            case_configs=_load_history_case_configs(
                case_configs, selected_summaries=selected_case_summaries
            ),
        ),
    )
    load_history_cfg["status"] = "computed"
    load_history_cfg["input_fields"] = [str(path) for path in load_case_paths]
    load_history_cfg.update(
        _compact_load_history_summary(
            result.to_dict(),
            case_summaries=selected_case_summaries,
        )
    )
    _run_load_history_final_rerun(
        load_history_cfg,
        base_config=base_config,
        case_configs=case_configs,
        case_summaries=case_summaries,
        base_dir=base_dir,
        batch_work_dir=batch_work_dir,
        dry_run=dry_run,
    )


def _run_load_history_final_rerun(
    load_history_cfg: dict[str, Any],
    *,
    base_config: dict[str, Any],
    case_configs: list[dict[str, Any]],
    case_summaries: list[dict[str, Any]],
    base_dir: Path,
    batch_work_dir: Path,
    dry_run: bool,
) -> None:
    final_cfg = load_history_cfg.get("final_rerun")
    if not isinstance(final_cfg, dict) or not final_cfg.get("enabled", False):
        return
    requested_cases = load_history_cfg.get("cases")
    selected_summaries = _load_history_case_summaries(
        case_summaries, requested_cases=requested_cases
    )
    selected_configs = _load_history_case_configs(
        case_configs, selected_summaries=selected_summaries
    )
    details = load_history_cfg.get("details", {})
    load_amplitudes = list(details.get("load_amplitudes", ()))
    input_amplitudes = list(details.get("input_load_amplitudes", ()))
    if len(load_amplitudes) != len(selected_configs):
        raise ValueError(
            "load-history final rerun needs one estimated amplitude per selected unit case"
        )
    case_override = copy.deepcopy(final_cfg.get("case", {}))
    case_override.setdefault(
        "name_suffix", final_cfg.get("name_suffix", "final_combined")
    )
    rerun_config = _case_config(
        base_config,
        case_override,
        index=len(case_configs),
        base_dir=base_dir,
    )
    rerun_config["load_case"] = _combined_nodeset_load_case(
        selected_configs,
        load_amplitudes=load_amplitudes,
        input_amplitudes=input_amplitudes,
    )
    output_cfg = rerun_config.setdefault("output", {})
    if final_cfg.get("fields") is not None:
        output_cfg["fields"] = list(final_cfg.get("fields") or ["sed"])
        output_cfg["export_fields"] = True
    if final_cfg.get("visualization_field") is not None:
        output_cfg["visualization_field"] = str(final_cfg.get("visualization_field"))
    if dry_run:
        final_cfg["status"] = "dry_run"
        final_cfg["load_case"] = rerun_config["load_case"]
        return
    case_path = _write_case_config(
        rerun_config,
        batch_work_dir=batch_work_dir,
        case_name=rerun_config["case"]["name"],
    )
    run_case_config(case_path, dry_run=False, work_dir=None)
    summary_path = (
        rerun_config["output"].get("result") or rerun_config["output"]["summary"]
    )
    final_summary = _read_case_summary(Path(summary_path))
    final_output = final_cfg.get("output")
    final_field = str(final_cfg.get("field", "sed"))
    if final_output:
        exported = final_summary.get("outputs", {}).get("exported", {})
        field_path = exported.get(final_field)
        if field_path is not None:
            destination = _resolve_path(final_output, base_dir=base_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(field_path, destination)
            final_cfg["output"] = str(destination)
    final_cfg["status"] = "computed"
    final_cfg["case"] = _compact_case_summary(final_summary)
    final_cfg["load_case"] = rerun_config["load_case"]


def _load_history_case_configs(
    case_configs: list[dict[str, Any]],
    *,
    selected_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    configs = []
    for summary in selected_summaries:
        name = str(summary.get("case", {}).get("name", ""))
        match = next(
            (
                config
                for config in case_configs
                if str(config.get("case", {}).get("name", "")) == name
            ),
            None,
        )
        if match is None:
            raise ValueError(f"load-history case config {name!r} was not found")
        configs.append(match)
    return configs


def _combined_nodeset_load_case(
    case_configs: list[dict[str, Any]],
    *,
    load_amplitudes,
    input_amplitudes,
) -> dict[str, Any]:
    combined = {"type": "nodeset", "fixed": [], "prescribed": [], "loaded": []}
    seen_fixed = set()
    for index, case_config in enumerate(case_configs):
        load_case = case_config.get("load_case", {})
        if str(load_case.get("type", "")).strip().lower() not in {"nodeset", "custom"}:
            raise ValueError(
                "load-history final rerun currently requires nodeset unit cases"
            )
        scale = _load_history_case_scale(
            load_amplitudes[index],
            input_amplitudes[index] if index < len(input_amplitudes) else 1.0,
        )
        for spec in load_case.get("fixed", ()):
            frozen = json.dumps(spec, sort_keys=True)
            if frozen not in seen_fixed:
                combined["fixed"].append(copy.deepcopy(spec))
                seen_fixed.add(frozen)
        combined["prescribed"].extend(
            _scaled_load_spec(spec, scale) for spec in load_case.get("prescribed", ())
        )
        combined["loaded"].extend(
            _scaled_load_spec(spec, scale) for spec in load_case.get("loaded", ())
        )
    for key in ("fixed", "prescribed", "loaded"):
        if not combined[key]:
            combined.pop(key)
    return combined


def _load_history_case_scale(load_amplitude, input_amplitude) -> float:
    input_value = float(input_amplitude)
    if abs(input_value) <= 1.0e-12:
        return 0.0
    return float(load_amplitude) / input_value


def _scaled_load_spec(spec: dict[str, Any], scale: float) -> dict[str, Any]:
    scaled = copy.deepcopy(spec)
    if "value" in scaled:
        scaled["value"] = _scaled_load_value(scaled["value"], scale)
    return scaled


def _scaled_load_value(value, scale: float):
    if isinstance(value, str):
        text = value.strip()
        suffix = ""
        for candidate in (
            "degrees",
            "degree",
            "deg",
            "radians",
            "radian",
            "rad",
            "mm",
            "%",
        ):
            if text.lower().endswith(candidate):
                suffix = text[-len(candidate) :]
                text = text[: -len(candidate)].strip()
                break
        return f"{float(text) * float(scale):g}{suffix}"
    return float(value) * float(scale)


def _load_history_field(load_history_cfg: dict[str, Any]) -> str:
    fields = load_history_cfg.get("fields", ["sed"])
    if isinstance(fields, str):
        return fields
    if not fields:
        return "sed"
    return str(list(fields)[0])


def _load_history_field_paths(
    case_summaries: list[dict[str, Any]],
    *,
    field: str,
    requested_cases,
) -> list[Path]:
    selected = _load_history_case_summaries(
        case_summaries, requested_cases=requested_cases
    )
    paths: list[Path] = []
    for summary in selected:
        exported = summary.get("outputs", {}).get("exported", {})
        path = exported.get(field)
        if path is None:
            name = summary.get("case", {}).get("name", "<unknown>")
            raise ValueError(
                f"load-history postprocess requires exported field {field!r} "
                f"for case {name!r}; set output.export_fields=true and include "
                f"{field!r} in output.fields"
            )
        paths.append(Path(path).expanduser().resolve())
    return paths


def _load_history_case_summaries(
    case_summaries: list[dict[str, Any]],
    *,
    requested_cases,
) -> list[dict[str, Any]]:
    if not requested_cases:
        return case_summaries
    return [
        _find_case_summary(case_summaries, str(requested))
        for requested in requested_cases
    ]


def _case_input_load_amplitudes(
    case_summaries: list[dict[str, Any]],
    *,
    case_configs: list[dict[str, Any]] | None = None,
) -> list[float]:
    amplitudes = []
    configs = list(case_configs or [])
    for index, summary in enumerate(case_summaries):
        amplitude = _generalized_load_amplitude(summary)
        if amplitude is None or amplitude <= 0.0 or not np.isfinite(amplitude):
            config = configs[index] if index < len(configs) else None
            amplitude = _configured_load_amplitude(config)
        amplitudes.append(float(amplitude))
    return amplitudes


def _generalized_load_amplitude(summary: dict[str, Any]) -> float | None:
    generalized = summary.get("mechanics", {}).get("generalized_load", {})
    if not isinstance(generalized, dict):
        return None
    if str(generalized.get("name", "")).strip().lower() == "moment":
        vector = generalized.get("vector")
        if isinstance(vector, dict):
            values = [float(vector.get(axis, 0.0) or 0.0) for axis in ("x", "y", "z")]
            magnitude = float(np.linalg.norm(values))
            if np.isfinite(magnitude) and magnitude > 0.0:
                return magnitude
    value = generalized.get("value")
    return None if value is None else abs(float(value))


def _configured_load_amplitude(case_config: dict[str, Any] | None) -> float:
    if not isinstance(case_config, dict):
        return 1.0
    load_case = case_config.get("load_case", {})
    values = []
    if isinstance(load_case, dict):
        for section in ("prescribed", "loaded"):
            for spec in load_case.get(section, ()) or ():
                if isinstance(spec, dict) and "value" in spec:
                    try:
                        values.append(abs(_numeric_load_value(spec["value"])))
                    except (TypeError, ValueError):
                        continue
    finite = [value for value in values if np.isfinite(value) and value > 0.0]
    return float(finite[0]) if finite else 1.0


def _numeric_load_value(value) -> float:
    if isinstance(value, str):
        text = value.strip()
        for suffix in (
            "degrees",
            "degree",
            "deg",
            "radians",
            "radian",
            "rad",
            "mm",
            "%",
            "N",
            "n",
        ):
            if text.lower().endswith(suffix.lower()):
                text = text[: -len(suffix)].strip()
                break
        return float(text)
    return float(value)


def _find_case_summary(
    case_summaries: list[dict[str, Any]], requested: str
) -> dict[str, Any]:
    for summary in case_summaries:
        name = str(summary.get("case", {}).get("name", ""))
        if name == requested or name.endswith(f"_{requested}"):
            return summary
    available = [
        str(summary.get("case", {}).get("name", "")) for summary in case_summaries
    ]
    raise ValueError(
        f"load-history case {requested!r} was not found; available cases: {available}"
    )


def _first_input_file(case_summaries: list[dict[str, Any]]) -> Path | None:
    for summary in case_summaries:
        path = summary.get("outputs", {}).get("input_file")
        if path:
            return Path(path).expanduser().resolve()
    return None


def _compact_load_history_summary(
    result: dict[str, Any], *, case_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    details = result.get("details", {})
    failure = result.get("failure", {})
    factor = failure.get("factor") if isinstance(failure, dict) else None
    amplitudes = details.get("load_amplitudes", [])
    estimated_loads = []
    failure_loads = []
    for amplitude, summary in zip(amplitudes, case_summaries, strict=False):
        generalized = summary.get("mechanics", {}).get("generalized_load")
        estimated = _load_history_amplitude_entry(
            amplitude,
            generalized,
            case=summary.get("case", {}).get("name"),
        )
        estimated_loads.append(estimated)
        if factor is not None and generalized and generalized.get("value") is not None:
            failure_loads.append(
                _load_history_amplitude_entry(
                    float(amplitude) * float(factor),
                    generalized,
                    case=summary.get("case", {}).get("name"),
                )
            )
    output = {
        "results": {
            "estimated_loads": estimated_loads,
            "failure_loads": failure_loads,
        },
        "details": details,
    }
    if failure:
        output["failure"] = failure
    return output


def _load_history_amplitude_entry(
    amplitude, generalized: dict[str, Any] | None, *, case
) -> dict[str, Any]:
    value = _signed_amplitude(amplitude, generalized)
    units = None if generalized is None else generalized.get("units")
    component = None if generalized is None else generalized.get("component")
    load_type = None if generalized is None else generalized.get("name")
    entry = {
        "case": case,
        "value": value,
        "units": units,
        "component": component,
        "load_type": load_type,
    }
    if isinstance(generalized, dict):
        vector = _scaled_generalized_vector(float(amplitude), generalized)
        if vector is not None:
            entry["vector"] = vector
    return entry


def _scaled_generalized_vector(
    amplitude: float, generalized: dict[str, Any]
) -> dict[str, float] | None:
    load_type = str(generalized.get("name", "")).strip().lower()
    if load_type == "moment" and isinstance(generalized.get("vector"), dict):
        source = np.asarray(
            [
                float(generalized["vector"].get(axis, 0.0) or 0.0)
                for axis in ("x", "y", "z")
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(source))
        if np.isfinite(norm) and norm > 0.0:
            scaled = source * (float(amplitude) / norm)
            return {
                axis: float(scaled[index]) for index, axis in enumerate(("x", "y", "z"))
            }
    if load_type == "force":
        component = str(generalized.get("component", "")).strip().lower()
        if component in {"x", "y", "z"}:
            vector = {"x": 0.0, "y": 0.0, "z": 0.0}
            vector[component] = _signed_amplitude(amplitude, generalized)
            return vector
    return None


def _signed_amplitude(amplitude, generalized: dict[str, Any] | None) -> float:
    value = None if generalized is None else generalized.get("value")
    sign = -1.0 if value is not None and float(value) < 0.0 else 1.0
    return sign * float(amplitude)


def _case_config(
    base_config: dict[str, Any],
    case_override: dict[str, Any],
    *,
    index: int,
    base_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(
        {key: value for key, value in base_config.items() if key != "batch"}
    )
    _absolutize_referenced_paths(config, base_dir=base_dir)
    base_case = _section(config, "case")
    base_name = str(base_case.get("name", "case"))
    suffix = str(
        case_override.get("name_suffix", case_override.get("name", f"case_{index + 1}"))
    )
    case_name = suffix if suffix.startswith(base_name) else f"{base_name}_{suffix}"

    _deep_update(
        config,
        {
            key: value
            for key, value in case_override.items()
            if key not in {"name", "name_suffix"}
        },
    )
    config.setdefault("case", {})
    config["case"]["name"] = case_name
    parent_work_dir = Path(str(base_case.get("work_dir", base_name))).parent
    case_work_dir = _resolve_path(parent_work_dir / case_name, base_dir=base_dir)
    config["case"]["work_dir"] = str(case_work_dir)
    config.setdefault("output", {})
    config["output"]["result"] = str(case_work_dir / "result.json")
    config["output"]["summary"] = config["output"]["result"]
    config["output"]["run_summary"] = str(case_work_dir / "summary.json")
    config["output"]["fields_dir"] = str(case_work_dir / "fields")
    config["output"]["visualization"] = str(case_work_dir / "overview.png")
    if "material_image" in config["output"]:
        config["output"]["material_image"] = str(
            case_work_dir / "model" / "material.nii.gz"
        )
    if "boundary_conditions" in config["output"]:
        config["output"]["boundary_conditions"] = str(
            case_work_dir / "boundary_conditions.json"
        )
    if "sets_dir" in config["output"]:
        config["output"]["sets_dir"] = str(case_work_dir / "sets")
    return config


def _write_case_config(
    config: dict[str, Any],
    *,
    batch_work_dir: Path,
    case_name: str,
) -> Path:
    config_dir = batch_work_dir / "_cases"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{case_name}.json"
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _read_case_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _compact_case_summary(summary: dict[str, Any]) -> dict[str, Any]:
    mechanics = summary.get("mechanics", {})
    failure = summary.get("failure", {})
    results = summary.get("results", {})
    return {
        "case": summary.get("case", {}),
        "load_case": summary.get("load_case", {}),
        "result": summary.get("outputs", {}).get("result")
        or summary.get("outputs", {}).get("summary"),
        "results": results
        or {
            "generalized_load": mechanics.get("generalized_load"),
            "generalized_stiffness": mechanics.get("generalized_stiffness"),
            "reaction_force": mechanics.get("reaction_force"),
            "stiffness": mechanics.get("stiffness"),
            "failure_load": failure.get("failure_load"),
            "failure_generalized_load": failure.get("failure_generalized_load"),
            "pistoia_factor": failure.get("factor"),
            "ees_at_critical_volume": failure.get("ees_at_critical_volume"),
        },
        "generalized_load": mechanics.get("generalized_load"),
        "generalized_stiffness": mechanics.get("generalized_stiffness"),
        "stiffness": mechanics.get("stiffness"),
        "failure_load": failure.get("failure_load"),
        "failure_generalized_load": failure.get("failure_generalized_load"),
        "failure": {
            "criterion": failure.get("criterion"),
            "critical_strain": failure.get("critical_strain"),
            "critical_volume_percent": failure.get("critical_volume_percent"),
            "ees_at_critical_volume": failure.get("ees_at_critical_volume"),
            "factor": failure.get("factor"),
            "status": failure.get("status"),
        },
    }


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _absolutize_referenced_paths(config: dict[str, Any], *, base_dir: Path) -> None:
    input_cfg = _section(config, "input")
    if "image" in input_cfg:
        input_cfg["image"] = str(_resolve_path(input_cfg["image"], base_dir=base_dir))
    material_cfg = _section(config, "materials")
    if "file" in material_cfg:
        material_cfg["file"] = str(
            _resolve_path(material_cfg["file"], base_dir=base_dir)
        )
    for spec in _section(config, "nodesets").values():
        if isinstance(spec, dict) and "image" in spec:
            spec["image"] = str(_resolve_path(spec["image"], base_dir=base_dir))


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
