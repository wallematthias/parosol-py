from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import zipfile

from .workflow_registry import available_profiles, builtin_profile_path
from .workflow_template import load_workflow_template


EXPECTED_PUBLIC_PROFILES = (
    "XtremeCTI",
    "XtremeCTII",
    "hip-sideways-fall-left",
    "hip-sideways-fall-right",
    "load_history_3",
    "load_history_6",
    "spine-compression",
)

FORBIDDEN_PUBLIC_KEYS = frozenset({"protrusion_depth_mm"})


@dataclass(frozen=True)
class WorkflowContractIssue:
    code: str
    message: str


def validate_all_builtin_workflows() -> dict[str, list[WorkflowContractIssue]]:
    results: dict[str, list[WorkflowContractIssue]] = {}
    for profile in EXPECTED_PUBLIC_PROFILES:
        results[profile] = validate_builtin_profile(profile)
    return results


def validate_builtin_profile(profile: str) -> list[WorkflowContractIssue]:
    path = builtin_profile_path(profile)
    if path is None:
        return [
            WorkflowContractIssue(
                code="missing_profile",
                message=f"Built-in workflow profile is missing: {profile}",
            )
        ]
    config, _source = load_workflow_template(path)
    return validate_workflow_config(
        config,
        profile=profile,
        bundle_members=_bundle_members(path),
    )


def validate_workflow_config(
    config: dict[str, Any],
    *,
    profile: str,
    bundle_members: Iterable[str],
) -> list[WorkflowContractIssue]:
    issues: list[WorkflowContractIssue] = []
    members = tuple(bundle_members)
    template = config.get("workflow_template", {})
    template_type = str(template.get("type", "")).strip()
    slicer_editor = config.get("slicer_editor", {})
    planes = slicer_editor.get("planes") if isinstance(slicer_editor, dict) else None

    if profile in EXPECTED_PUBLIC_PROFILES and profile not in set(available_profiles()):
        issues.append(
            WorkflowContractIssue(
                code="missing_public_profile",
                message=f"Public profile is not registered: {profile}",
            )
        )
    if template_type == "single_case_fea" and not (
        isinstance(planes, list) and len(planes) > 0
    ):
        issues.append(
            WorkflowContractIssue(
                code="missing_editor_planes",
                message=f"Workflow {profile} has no canonical slicer_editor.planes",
            )
        )
    for key_path in _find_forbidden_keys(config):
        issues.append(
            WorkflowContractIssue(
                code="forbidden_key",
                message=f"Forbidden workflow key {key_path[-1]} at {'.'.join(key_path)}",
            )
        )
    if profile in {
        "spine-compression",
        "hip-sideways-fall-left",
        "hip-sideways-fall-right",
    }:
        if not any(member == "reference/slicer_reference_points.npy" for member in members):
            issues.append(
                WorkflowContractIssue(
                    code="missing_reference_points",
                    message=f"Workflow {profile} must include reference/slicer_reference_points.npy",
                )
            )
        if any(member.lower().endswith(".vtk") for member in members):
            issues.append(
                WorkflowContractIssue(
                    code="vtk_reference_packaged",
                    message=f"Workflow {profile} must not package VTK reference assets",
                )
            )
    return issues


def _bundle_members(path: str | Path) -> tuple[str, ...]:
    workflow_path = Path(path)
    if not workflow_path.is_file() or not workflow_path.name.endswith(".parosol-workflow"):
        return ()
    with zipfile.ZipFile(workflow_path) as archive:
        return tuple(sorted(archive.namelist()))


def _find_forbidden_keys(
    value: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            next_path = (*path, key_text)
            if key_text in FORBIDDEN_PUBLIC_KEYS:
                found.append(next_path)
            found.extend(_find_forbidden_keys(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, (*path, str(index))))
    return found
