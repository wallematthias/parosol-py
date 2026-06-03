from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def field_statistics(
    values, *, percentiles: tuple[float, ...] = (5, 25, 50, 75, 95)
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    out: dict[str, Any] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
    }
    if finite.size == 0:
        out.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "median": None,
                "percentiles": {},
            }
        )
        return out
    out.update(
        {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "median": float(np.median(finite)),
            "percentiles": {
                _percentile_key(p): float(np.percentile(finite, p)) for p in percentiles
            },
        }
    )
    return out


def solve_summary_dict(
    result, *, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    run = result.summary.run
    data: dict[str, Any] = {
        "solver": {
            "iterations": None if run is None else run.iterations,
            "relative_residual": None if run is None else run.relative_residual,
            "absolute_residual": None if run is None else run.absolute_residual,
            "runtime_seconds": None if run is None else run.overall_time_seconds,
        },
        "image": {
            "dimensions_xyz": list(result.summary.dimensions_xyz),
            "spacing": list(result.summary.spacing),
            "origin": list(result.summary.origin),
        },
        "outputs": {
            "input_file": str(result.input_file),
            "exported": {name: str(path) for name, path in result.exported.items()},
        },
        "fields": _summarize_fields(result.fields),
    }
    data["quality"] = _solution_quality(data["solver"])
    if getattr(result, "diagnostics", None):
        data.update(_jsonable(result.diagnostics))
    if extra:
        extra_json = _jsonable(extra)
        if "quality" in extra_json:
            data["quality"].update(extra_json.pop("quality"))
            data["quality"].update(
                _solution_quality(
                    data["solver"], checks=data["quality"].get("checks", {})
                )
            )
        data.update(extra_json)
    return data


def compact_summary_dict(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the user-facing subset of a full solve summary."""

    mechanics = summary.get("mechanics", {})
    failure = summary.get("failure", {})
    outputs = summary.get("outputs", {})
    compact: dict[str, Any] = {
        "case": summary.get("case", {}),
        "execution": summary.get("execution", {}),
        "image": summary.get("image", {}),
        "load_case": summary.get("load_case", {}),
        "results": {
            "generalized_load": mechanics.get("generalized_load"),
            "generalized_stiffness": mechanics.get("generalized_stiffness"),
            "reaction_force": mechanics.get("reaction_force"),
            "stiffness": mechanics.get("stiffness"),
            "failure_load": failure.get("failure_load"),
            "failure_generalized_load": failure.get("failure_generalized_load"),
            "pistoia_factor": failure.get("factor"),
            "ees_at_critical_volume": failure.get("ees_at_critical_volume"),
        },
        "mechanics": {
            "generalized_load": mechanics.get("generalized_load"),
            "generalized_stiffness": mechanics.get("generalized_stiffness"),
            "reaction_force": mechanics.get("reaction_force"),
            "stiffness": mechanics.get("stiffness"),
            "applied_displacement": mechanics.get("applied_displacement"),
            "applied_rotation_degrees": mechanics.get("applied_rotation_degrees"),
            "top_node_count": mechanics.get("top_node_count"),
            "bottom_node_count": mechanics.get("bottom_node_count"),
            "status": mechanics.get("status"),
        },
        "failure": {
            "status": failure.get("status"),
            "criterion": failure.get("criterion"),
            "critical_strain": failure.get("critical_strain"),
            "critical_volume_percent": failure.get("critical_volume_percent"),
            "ees_at_critical_volume": failure.get("ees_at_critical_volume"),
            "factor": failure.get("factor"),
            "failure_load": failure.get("failure_load"),
            "failure_generalized_load": failure.get("failure_generalized_load"),
        },
        "solver": summary.get("solver", {}),
        "quality": summary.get("quality", {}),
        "outputs": {
            "input_file": outputs.get("input_file"),
            "exported": outputs.get("exported", {}),
        },
    }
    for key in ("linear_reaction_at_deformation", "crawford_stiffness_height"):
        if key in failure:
            compact["results"][key] = failure[key]
            compact["failure"][key] = failure[key]
    if "model" in summary:
        compact["model"] = summary["model"]
    return _jsonable(compact)


def write_summary_json(path: str | Path, data: dict[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def write_solve_summary_json(
    result, path: str | Path, *, extra: dict[str, Any] | None = None
) -> Path:
    return write_summary_json(path, solve_summary_dict(result, extra=extra))


def parse_legacy_analysis_text(text: str) -> dict[str, Any]:
    return {
        "model_input": _parse_key_value_table(text, "Table 1: Model Input"),
        "materials": _parse_materials(text),
        "strain_energy_density": _parse_scalar_stats_table(
            text, "Table 6: Strain Energy Density"
        ),
        "von_mises_stress": _parse_scalar_stats_table(
            text, "Table 7: Von Mises Stress"
        ),
        "nodal_displacements": _parse_vector_node_table(
            text, "Table 8: Nodal Displacements"
        ),
        "nodal_forces": _parse_vector_node_table(text, "Table 9: Nodal Forces"),
    }


def parse_legacy_analysis_file(path: str | Path) -> dict[str, Any]:
    return parse_legacy_analysis_text(Path(path).read_text(encoding="utf-8"))


def parse_pistoia_text(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "critical_volume_percent": None,
        "critical_ees": None,
        "ees_at_critical_volume": None,
        "factor": None,
        "failure_load": {},
        "reaction_force_node_set_1": {},
        "displacement_node_set_1": {},
        "axial_stiffness": {},
        "ees_distribution": {},
        "failed_materials": [],
    }
    for line in text.splitlines():
        if "Critical volume (%)" in line:
            out["critical_volume_percent"] = float(line.split(":")[-1].strip())
        elif "Critical EES" in line:
            out["critical_ees"] = float(line.split(":")[-1].strip())
        elif "EES at vol_crit" in line:
            out["ees_at_critical_volume"] = float(line.split(":")[-1].strip())
        elif "Factor (from table)" in line:
            out["factor"] = float(line.split(":")[-1].strip())
        elif "Failure load (RF * factor)" in line:
            out["failure_load"] = _xyz_from_line(line, ("fx", "fy", "fz"))
        elif "RF (node set 1)" in line:
            out["reaction_force_node_set_1"] = _xyz_from_line(line, ("fx", "fy", "fz"))
        elif "U (node set 1)" in line:
            out["displacement_node_set_1"] = _xyz_from_line(line, ("ux", "uy", "uz"))
        elif "Axial stiffness:" in line:
            out["axial_stiffness"] = _xyz_from_line(line, ("x", "y", "z"))
        else:
            stat_match = re.match(
                r"^\s*(average|std_dev|minimum|maximum|median)\s+(\S+)\s*$", line
            )
            if stat_match:
                out["ees_distribution"][stat_match.group(1)] = _to_number(
                    stat_match.group(2)
                )
                continue
            failed_match = re.match(r"^\s*(\d+)\s+(\d+)\s+([-\d.]+)\s*$", line)
            if failed_match:
                out["failed_materials"].append(
                    {
                        "material_id": int(failed_match.group(1)),
                        "elements": int(failed_match.group(2)),
                        "percent": float(failed_match.group(3)),
                    }
                )
    return out


def parse_pistoia_file(path: str | Path) -> dict[str, Any]:
    return parse_pistoia_text(Path(path).read_text(encoding="utf-8"))


def _summarize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in fields.items():
        if isinstance(value, dict):
            out[name] = {
                component: field_statistics(component_values)
                for component, component_values in value.items()
            }
            continue
        array = np.asarray(value)
        if array.size and np.issubdtype(array.dtype, np.number):
            out[name] = field_statistics(array)
    return out


def _solution_quality(
    solver: dict[str, Any],
    *,
    checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = {} if checks is None else checks
    issues: list[str] = []
    max_relative = checks.get("max_relative_residual")
    max_iterations = checks.get("max_iterations")
    if solver.get("relative_residual") is not None and max_relative is not None:
        if float(solver["relative_residual"]) > float(max_relative):
            issues.append("relative_residual")
    if solver.get("iterations") is not None and max_iterations is not None:
        if int(solver["iterations"]) > int(max_iterations):
            issues.append("iterations")
    if solver.get("iterations") is None and solver.get("runtime_seconds") is None:
        status = "not_computed"
    else:
        status = "passed" if not issues else "warning"
    return {
        "status": status,
        "issues": issues,
        "checks": checks,
    }


def _parse_key_value_table(text: str, title: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in _table_lines(text, title):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[_key(key)] = _to_number(value.strip())
    return out


def _parse_materials(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in _table_lines(text, "Table 2: Materials"):
        match = re.match(
            r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+([-\d.E+]+)\s+(\d+)\s*$", line
        )
        if not match:
            continue
        out.append(
            {
                "index": int(match.group(1)),
                "material_id": int(match.group(2)),
                "name": match.group(3),
                "type": match.group(4),
                "e_ii_max": float(match.group(5)),
                "elements": int(match.group(6)),
            }
        )
    return out


def _parse_scalar_stats_table(text: str, title: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    current = "all"
    for line in _table_lines(text, title):
        material_match = re.match(r"^\s*m:\s*(\S+)\s*$", line)
        if material_match:
            token = material_match.group(1).strip()
            current = "all" if token.upper() == "ALL" else token
            groups.setdefault(current, {})
            continue
        stat_match = re.match(
            r"^\s*(average|std_dev|minimum|maximum|median|perc\d+)\s+(\S+)\s*$", line
        )
        if stat_match:
            groups.setdefault(current, {})[stat_match.group(1)] = _to_number(
                stat_match.group(2)
            )
    return groups


def _parse_vector_node_table(text: str, title: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in _table_lines(text, title):
        if "Node set:" in line:
            if current is not None:
                out.append(current)
            current = {
                "node_set": int(line.split(":")[-1].strip()),
                "name": "",
                "stats": {},
            }
            continue
        if current is None:
            continue
        if "Name:" in line:
            current["name"] = line.split(":")[-1].strip()
            continue
        match = re.match(
            r"^\s*(total|average|std_dev|minimum|maximum|median)\s+(\S+)\s+(\S+)\s+(\S+)\s*$",
            line,
        )
        if match:
            current["stats"][match.group(1)] = {
                "x": _to_number(match.group(2)),
                "y": _to_number(match.group(3)),
                "z": _to_number(match.group(4)),
            }
    if current is not None:
        out.append(current)
    return out


def _table_lines(text: str, title: str) -> list[str]:
    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if title in line)
    except StopIteration:
        return []
    section = lines[start + 1 :]
    out: list[str] = []
    seen_content = False
    for line in section:
        if line.startswith("====") and seen_content:
            break
        if (
            line.startswith("----")
            or line.startswith("....")
            or line.startswith("====")
        ):
            continue
        if line.strip():
            seen_content = True
        out.append(line)
    return out


def _xyz_from_line(line: str, keys: tuple[str, str, str]) -> dict[str, Any]:
    values = re.findall(
        r"[-+]?(?:INF|\d+\.\d+E[-+]?\d+|\d+\.\d+|\d+)", line, flags=re.I
    )
    return {key: _to_number(value) for key, value in zip(keys, values[-3:])}


def _to_number(text: str) -> Any:
    token = text.strip()
    if token.upper() == "INF":
        return float("inf")
    if token.upper() == "-INF":
        return float("-inf")
    try:
        if any(ch in token for ch in (".", "E", "e")):
            return float(token)
        return int(token)
    except ValueError:
        return token


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _percentile_key(value: float) -> str:
    if float(value).is_integer():
        return f"p{int(value):02d}"
    return f"p{value:g}"
