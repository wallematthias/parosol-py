from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import load_config, run_case_config
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
        run_case_config(case_path, dry_run=dry_run, work_dir=None)
        case_summary = _read_case_summary(Path(case_config["output"]["summary"]))
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
    write_summary_json(summary_path, summary)
    return summary


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
    config["case"]["work_dir"] = str(
        _resolve_path(parent_work_dir / case_name, base_dir=base_dir)
    )
    config.setdefault("output", {})
    config["output"]["summary"] = str(Path(config["case"]["work_dir"]) / "summary.json")
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
    return {
        "case": summary.get("case", {}),
        "load_case": summary.get("load_case", {}),
        "summary": summary.get("outputs", {}).get("summary"),
        "generalized_load": mechanics.get("generalized_load"),
        "generalized_stiffness": mechanics.get("generalized_stiffness"),
        "failure_generalized_load": failure.get("failure_generalized_load"),
        "failure": {
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
