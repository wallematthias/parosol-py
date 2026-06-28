from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess

from .workflow_contracts import EXPECTED_PUBLIC_PROFILES, validate_builtin_profile
from .workflow_registry import builtin_profile_path
from .workflow_template import load_workflow_template


def build_builtin_workflow_baseline() -> dict[str, Any]:
    workflows: dict[str, Any] = {}
    for profile in EXPECTED_PUBLIC_PROFILES:
        path = builtin_profile_path(profile)
        if path is None:
            workflows[profile] = {"status": "missing"}
            continue
        config, _source = load_workflow_template(path)
        workflows[profile] = _workflow_summary(profile, path, config)
    return {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "profiles": list(EXPECTED_PUBLIC_PROFILES),
        "workflows": workflows,
    }


def _workflow_summary(profile: str, path: Path, config: dict[str, Any]) -> dict[str, Any]:
    template = config.get("workflow_template", {})
    editor = config.get("slicer_editor", {})
    model = config.get("model", {})
    registration = model.get("registration", {}) if isinstance(model, dict) else {}
    replay = model.get("workflow_replay", {}) if isinstance(model, dict) else {}
    solver = config.get("solver", {})
    load_case = config.get("load_case", {})
    batch = config.get("batch", {})
    planes = editor.get("planes", []) if isinstance(editor, dict) else []
    loads = editor.get("loads", []) if isinstance(editor, dict) else []
    issues = validate_builtin_profile(profile)
    return {
        "path": str(path),
        "workflow_type": template.get("type"),
        "profile": template.get("profile", profile),
        "plane_count": len(planes) if isinstance(planes, list) else 0,
        "load_count": len(loads) if isinstance(loads, list) else 0,
        "load_case_type": load_case.get("type") if isinstance(load_case, dict) else None,
        "batch_case_count": len(batch.get("cases", [])) if isinstance(batch, dict) else 0,
        "model_type": model.get("type") if isinstance(model, dict) else None,
        "registration_enabled": bool(registration.get("enabled", False))
        if isinstance(registration, dict)
        else False,
        "workflow_replay_enabled": bool(replay.get("enabled", False))
        if isinstance(replay, dict)
        else False,
        "workflow_replay_model_space": replay.get("model_space")
        if isinstance(replay, dict)
        else None,
        "solver_tolerance": solver.get("tolerance") if isinstance(solver, dict) else None,
        "contract_issue_count": len(issues),
        "contract_issues": [issue.__dict__ for issue in issues],
    }


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"
