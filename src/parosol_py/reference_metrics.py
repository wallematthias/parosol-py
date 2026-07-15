from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


CORE_METRICS_SCHEMA_VERSION = 1


def build_core_metric_lock(index_path: str | Path) -> dict[str, Any]:
    source = Path(index_path).expanduser().resolve()
    index = json.loads(source.read_text(encoding="utf-8"))
    records = [_core_metric_record(run) for run in index.get("runs", ())]
    return {
        "schema_version": CORE_METRICS_SCHEMA_VERSION,
        "source_index": str(source),
        "output_root": index.get("output_root"),
        "records": records,
    }


def write_core_metric_lock(
    index_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    json_name: str = "core_metrics.json",
    csv_name: str = "core_metrics.csv",
) -> tuple[Path, Path]:
    lock = build_core_metric_lock(index_path)
    source = Path(index_path).expanduser().resolve()
    out_dir = (
        source.parent if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / json_name
    csv_path = out_dir / csv_name
    json_path.write_text(
        json.dumps(_jsonable(lock), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_records_csv(csv_path, lock["records"])
    return json_path, csv_path


def _core_metric_record(run: dict[str, Any]) -> dict[str, Any]:
    record = {
        "engine": run.get("engine"),
        "fixture": run.get("fixture"),
        "profile": run.get("profile"),
        "name": run.get("name"),
        "status": run.get("status"),
        "output_dir": run.get("output_dir"),
        "log": run.get("log"),
    }
    profile = str(run.get("profile", ""))
    fixture = str(run.get("fixture", ""))
    engine = str(run.get("engine", ""))
    artifacts = (
        run.get("artifacts", {}) if isinstance(run.get("artifacts"), dict) else {}
    )
    result = _load_result_json(artifacts)
    if profile.startswith("XtremeCT"):
        record.update(_xtremect_metrics(result))
    elif profile.startswith("load_history"):
        record.update(_load_history_metrics(result))
    elif engine == "ogo":
        record.update(_ogo_deformation_metrics(artifacts.get("metrics", {})))
    elif fixture in {"vertebra_l4_mini", "femur_left_mini"}:
        record.update(_parosol_deformation_metrics(result))
    else:
        record["metric_family"] = "generic"
    return record


def _xtremect_metrics(result: dict[str, Any]) -> dict[str, Any]:
    mechanics = _mapping(result.get("mechanics"))
    failure = _mapping(result.get("failure"))
    load_case = _mapping(result.get("load_case"))
    axis = str(
        load_case.get("axis")
        or _nested(mechanics, "generalized_load", "component")
        or ""
    )
    strain = _float_or_none(load_case.get("strain"))
    reaction_force = _axis_value(mechanics.get("reaction_force"), axis)
    return {
        "metric_family": "xtremect_pistoia",
        "target_axis": axis or None,
        "target_deformation_percent": None if strain is None else abs(strain) * 100.0,
        "applied_displacement_mm": _axis_value(
            mechanics.get("applied_displacement"), axis
        ),
        "reaction_force_n": reaction_force,
        "generalized_load_n": _nested_float(mechanics, "generalized_load", "value"),
        "stiffness_n_per_mm": _nested_float(
            mechanics, "generalized_stiffness", "value"
        ),
        "pistoia_factor": _float_or_none(failure.get("factor")),
        "pistoia_failure_load_n": _nested_float(
            failure, "failure_generalized_load", "value"
        ),
        "ees_at_critical_volume": _float_or_none(failure.get("ees_at_critical_volume")),
        "critical_strain": _float_or_none(failure.get("critical_strain")),
        "critical_volume_percent": _float_or_none(
            failure.get("critical_volume_percent")
        ),
        "top_node_count": _int_or_none(mechanics.get("top_node_count")),
        "bottom_node_count": _int_or_none(mechanics.get("bottom_node_count")),
    }


def _parosol_deformation_metrics(result: dict[str, Any]) -> dict[str, Any]:
    mechanics = _mapping(result.get("mechanics"))
    failure = _mapping(result.get("failure"))
    load_case = _mapping(result.get("load_case"))
    generalized = _mapping(mechanics.get("generalized_load"))
    axis = str(load_case.get("axis") or generalized.get("component") or "")
    reference_length = _float_or_none(mechanics.get("reference_length_mm"))
    applied = _axis_value(mechanics.get("applied_displacement"), axis)
    target_percent = _deformation_percent(applied, reference_length)
    return {
        "metric_family": "reference_deformation",
        "target_axis": axis or None,
        "target_deformation_percent": target_percent,
        "applied_displacement_mm": applied,
        "reference_length_mm": reference_length,
        "reaction_force_n": _axis_value(mechanics.get("reaction_force"), axis)
        or _nested_float(mechanics, "generalized_load", "value"),
        "generalized_load_n": _nested_float(mechanics, "generalized_load", "value"),
        "stiffness_n_per_mm": _nested_float(
            mechanics, "generalized_stiffness", "value"
        ),
        "pistoia_factor": _float_or_none(failure.get("factor")),
        "pistoia_failure_load_n": _nested_float(
            failure, "failure_generalized_load", "value"
        ),
        "top_node_count": _int_or_none(mechanics.get("top_node_count")),
        "bottom_node_count": _int_or_none(mechanics.get("bottom_node_count")),
    }


def _ogo_deformation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = _mapping(metrics)
    axis = _axis_from_analysis_var(metrics.get("analysis_var"))
    applied = _float_or_none(metrics.get("applied_displacement"))
    reference_length = _float_or_none(metrics.get("characteristic_length_mm"))
    return {
        "metric_family": "reference_deformation",
        "target_axis": axis,
        "target_deformation_percent": _deformation_percent(applied, reference_length),
        "applied_displacement_mm": applied,
        "reference_length_mm": reference_length,
        "reaction_force_n": _float_or_none(metrics.get("reaction_force_N")),
        "stiffness_n_per_mm": _float_or_none(metrics.get("stiffness_N_per_mm")),
        "analysis_var": metrics.get("analysis_var"),
        "model_file": metrics.get("model_file"),
    }


def _load_history_metrics(result: dict[str, Any]) -> dict[str, Any]:
    load_history = _mapping(_nested(result, "postprocess", "load_history"))
    details = _mapping(load_history.get("details"))
    results = _mapping(load_history.get("results"))
    estimated_loads = list(results.get("estimated_loads") or ())
    failure_loads = list(results.get("failure_loads") or ())
    estimated_force = _sum_vectors(estimated_loads, load_type="force")
    estimated_moment = _sum_vectors(estimated_loads, load_type="moment")
    failure_force = _sum_vectors(failure_loads, load_type="force")
    failure_moment = _sum_vectors(failure_loads, load_type="moment")
    failure = _mapping(load_history.get("failure"))
    return {
        "metric_family": "load_history",
        "load_history_method": load_history.get("method"),
        "load_history_case_count": len(load_history.get("cases") or ()),
        "load_history_cases": list(load_history.get("cases") or ()),
        "load_history_mean": _float_or_none(details.get("mean")),
        "load_history_std": _float_or_none(details.get("std")),
        "load_history_residual": _float_or_none(details.get("residual")),
        "load_history_load_amplitudes": list(details.get("load_amplitudes") or ()),
        "load_history_input_load_amplitudes": list(
            details.get("input_load_amplitudes") or ()
        ),
        "load_history_scaling_factors": list(details.get("scaling_factors") or ()),
        "estimated_loads": estimated_loads,
        "estimated_total_force_vector_n": estimated_force,
        "estimated_total_force_magnitude_n": _vector_magnitude(estimated_force),
        "estimated_total_moment_vector_nmm": estimated_moment,
        "estimated_total_moment_magnitude_nmm": _vector_magnitude(estimated_moment),
        "failure_loads": failure_loads,
        "failure_total_force_vector_n": failure_force,
        "failure_total_force_magnitude_n": _vector_magnitude(failure_force),
        "failure_total_moment_vector_nmm": failure_moment,
        "failure_total_moment_magnitude_nmm": _vector_magnitude(failure_moment),
        "load_history_pistoia_factor": _float_or_none(failure.get("factor")),
        "load_history_ees_at_critical_volume": _float_or_none(
            failure.get("ees_at_critical_volume")
        ),
        "load_history_failure_status": failure.get("status"),
        "final_rerun": _final_rerun_metrics(load_history.get("final_rerun")),
    }


def _final_rerun_metrics(value: Any) -> dict[str, Any]:
    final = _mapping(value)
    if not final:
        return {
            "status": "not_run",
            "reaction_force_n": None,
            "stiffness_n_per_mm": None,
            "pistoia_failure_load_n": None,
            "pistoia_factor": None,
        }
    case = _mapping(final.get("case"))
    results = _mapping(case.get("results"))
    return {
        "status": final.get("status", "unknown"),
        "reaction_force_n": _nested_float(results, "generalized_load", "value"),
        "stiffness_n_per_mm": _nested_float(results, "generalized_stiffness", "value"),
        "pistoia_failure_load_n": _nested_float(
            results, "failure_generalized_load", "value"
        ),
        "pistoia_factor": _float_or_none(results.get("pistoia_factor")),
        "case_name": _nested(case, "case", "name"),
    }


def _load_result_json(artifacts: dict[str, Any]) -> dict[str, Any]:
    path = artifacts.get("result_json")
    if not path:
        return {}
    result_path = Path(str(path)).expanduser()
    if not result_path.exists():
        return {}
    return json.loads(result_path.read_text(encoding="utf-8"))


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = [_flatten(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            row.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            row[name] = json.dumps(_jsonable(value), sort_keys=True)
        else:
            row[name] = value
    return row


def _sum_vectors(loads: list[Any], *, load_type: str) -> dict[str, float]:
    total = {"x": 0.0, "y": 0.0, "z": 0.0}
    for entry in loads:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("load_type", "")).strip().lower() != load_type:
            continue
        vector = _mapping(entry.get("vector"))
        for axis in total:
            total[axis] += float(vector.get(axis) or 0.0)
    return total


def _vector_magnitude(vector: dict[str, float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector.values()))


def _axis_value(values: Any, axis: str) -> float | None:
    mapping = _mapping(values)
    if not axis or axis not in mapping:
        return None
    return _float_or_none(mapping.get(axis))


def _deformation_percent(
    applied: float | None, reference_length: float | None
) -> float | None:
    if applied is None or reference_length in {None, 0.0}:
        return None
    return abs(float(applied)) / abs(float(reference_length)) * 100.0


def _axis_from_analysis_var(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    for axis in ("x", "y", "z"):
        if text.startswith(f"f{axis}") or text.startswith(f"m{axis}"):
            return axis
    return None


def _nested(data: Any, *keys: str) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _nested_float(data: Any, *keys: str) -> float | None:
    return _float_or_none(_nested(data, *keys))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
