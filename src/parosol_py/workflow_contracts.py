from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

from .workflow_registry import available_profiles, builtin_profile_path
from .workflow_template import load_workflow_template


EXPECTED_PUBLIC_PROFILES = (
    "XtremeCTI",
    "XtremeCTII",
    "hip-sideways-fall-left",
    "hip-sideways-fall-left-nonlinear",
    "hip-sideways-fall-right",
    "hip-sideways-fall-right-nonlinear",
    "load_history_3",
    "load_history_6",
    "spine-compression",
    "spine-compression-nonlinear",
)

FORBIDDEN_PUBLIC_KEYS = frozenset({"protrusion_depth_mm"})
REFERENCE_POINT_MEMBERS = frozenset(
    {
        "reference/slicer_reference_points.npy",
        "reference/slicer_reference_points.npz",
    }
)
PRIMARY_WORKFLOW_YAML_MEMBERS = frozenset({"workflow.yaml", "workflow.yml"})
WORKFLOW_TEMPLATE_KEY = "workflow_template"
STRONG_WORKFLOW_SECTION_KEYS = frozenset(
    {
        "slicer_editor",
        "solver",
        "load_case",
        "model",
        "batch",
        "case",
        "input",
        "nodesets",
        "output",
    }
)
SOLVER_TOLERANCE_PROFILES = frozenset(
    {
        "XtremeCTI",
        "XtremeCTII",
        "spine-compression",
        "spine-compression-nonlinear",
        "hip-sideways-fall-left",
        "hip-sideways-fall-left-nonlinear",
        "hip-sideways-fall-right",
        "hip-sideways-fall-right-nonlinear",
    }
)
XTREMECT_PROFILES = frozenset({"XtremeCTI", "XtremeCTII"})
REFERENCE_NODESET_PROFILES = frozenset(
    {
        "spine-compression",
        "spine-compression-nonlinear",
        "hip-sideways-fall-left",
        "hip-sideways-fall-left-nonlinear",
        "hip-sideways-fall-right",
        "hip-sideways-fall-right-nonlinear",
    }
)
LOAD_HISTORY_SUFFIXES = {
    "load_history_3": ("compression_z", "shear_zx", "shear_zy"),
    "load_history_6": (
        "compression_z",
        "shear_zx",
        "shear_zy",
        "bending_x",
        "bending_y",
        "torsion_z",
    ),
}
LOAD_HISTORY_3_MATERIALS = {
    100: {"E": 8748.0, "nu": 0.3},
    127: {"E": 8748.0, "nu": 0.3},
}


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
    members = _bundle_members(path)
    issues = validate_workflow_config(
        config,
        profile=profile,
        bundle_members=members,
    )
    issues.extend(_validate_secondary_workflow_yaml_members(path, profile, members))
    return issues


def validate_workflow_config(
    config: dict[str, Any],
    *,
    profile: str,
    bundle_members: Iterable[str],
) -> list[WorkflowContractIssue]:
    issues: list[WorkflowContractIssue] = []
    members = tuple(bundle_members)
    config_mapping = _mapping(config)
    template = _mapping(config_mapping.get("workflow_template", {}))
    template_type = str(template.get("type", "")).strip()
    slicer_editor = config_mapping.get("slicer_editor", {})
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
    for key_path in _find_forbidden_keys(config_mapping):
        issues.append(
            WorkflowContractIssue(
                code="forbidden_key",
                message=f"Forbidden workflow key {key_path[-1]} at {'.'.join(key_path)}",
            )
        )
    if profile in {
        "spine-compression",
        "spine-compression-nonlinear",
        "hip-sideways-fall-left",
        "hip-sideways-fall-left-nonlinear",
        "hip-sideways-fall-right",
        "hip-sideways-fall-right-nonlinear",
    }:
        if not any(member in REFERENCE_POINT_MEMBERS for member in members):
            issues.append(
                WorkflowContractIssue(
                    code="missing_reference_points",
                    message=f"Workflow {profile} must include a slicer reference points asset",
                )
            )
        if any(member.lower().endswith(".vtk") for member in members):
            issues.append(
                WorkflowContractIssue(
                    code="vtk_reference_packaged",
                    message=f"Workflow {profile} must not package VTK reference assets",
                )
            )
    _validate_family_semantics(profile, config_mapping, issues)
    return issues


def _validate_family_semantics(
    profile: str,
    config: dict[str, Any],
    issues: list[WorkflowContractIssue],
) -> None:
    solver = _mapping(config.get("solver", {}))
    load_case = _mapping(config.get("load_case", {}))

    if profile in SOLVER_TOLERANCE_PROFILES:
        _require_float(
            issues,
            profile=profile,
            code="invalid_solver_tolerance",
            path="solver.tolerance",
            actual=solver.get("tolerance"),
            expected=1.0e-4,
        )

    if profile in XTREMECT_PROFILES:
        _require_equal(
            issues,
            profile=profile,
            code="invalid_load_case_type",
            path="load_case.type",
            actual=load_case.get("type"),
            expected="constrained_axial",
        )
        _require_equal(
            issues,
            profile=profile,
            code="invalid_load_case_axis",
            path="load_case.axis",
            actual=load_case.get("axis"),
            expected="z",
        )
        _require_float(
            issues,
            profile=profile,
            code="invalid_load_case_strain",
            path="load_case.strain",
            actual=load_case.get("strain"),
            expected=-0.01,
        )

    if profile in REFERENCE_NODESET_PROFILES:
        model = _mapping(config.get("model", {}))
        replay = _mapping(model.get("workflow_replay", {}))
        _require_equal(
            issues,
            profile=profile,
            code="invalid_model_type",
            path="model.type",
            actual=model.get("type"),
            expected="workflow_replay",
        )
        _require_equal(
            issues,
            profile=profile,
            code="invalid_workflow_replay_enabled",
            path="model.workflow_replay.enabled",
            actual=replay.get("enabled"),
            expected=True,
        )
        _require_equal(
            issues,
            profile=profile,
            code="invalid_workflow_replay_model_space",
            path="model.workflow_replay.model_space",
            actual=replay.get("model_space"),
            expected="reference",
        )
        _require_equal(
            issues,
            profile=profile,
            code="invalid_load_case_type",
            path="load_case.type",
            actual=load_case.get("type"),
            expected="nodeset",
        )

    expected_suffixes = LOAD_HISTORY_SUFFIXES.get(profile)
    if expected_suffixes is not None:
        batch = _mapping(config.get("batch", {}))
        actual_suffixes = _batch_name_suffixes(batch.get("cases"))
        if actual_suffixes != expected_suffixes:
            issues.append(
                WorkflowContractIssue(
                    code="invalid_batch_name_suffixes",
                    message=(
                        f"Workflow {profile} must define batch.cases name_suffixes "
                        f"{list(expected_suffixes)!r}; got {list(actual_suffixes)!r}"
                    ),
                )
            )
    if profile == "load_history_3":
        _validate_label_materials(
            profile=profile,
            config=config,
            expected=LOAD_HISTORY_3_MATERIALS,
            issues=issues,
        )


def _bundle_members(path: str | Path) -> tuple[str, ...]:
    workflow_path = Path(path)
    if not workflow_path.is_file() or not workflow_path.name.endswith(".parosol-workflow"):
        return ()
    with zipfile.ZipFile(workflow_path) as archive:
        return tuple(sorted(archive.namelist()))


def _validate_secondary_workflow_yaml_members(
    path: str | Path,
    profile: str,
    bundle_members: Iterable[str],
) -> list[WorkflowContractIssue]:
    workflow_path = Path(path)
    if not workflow_path.is_file() or not workflow_path.name.endswith(".parosol-workflow"):
        return []

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read workflow templates") from exc

    issues: list[WorkflowContractIssue] = []
    members = tuple(bundle_members)
    with zipfile.ZipFile(workflow_path) as archive:
        primary_members = _primary_workflow_members(archive)
        for member in members:
            if not _is_secondary_yaml_member(member, primary_members):
                continue
            try:
                loaded = yaml.safe_load(archive.read(member))
            except yaml.YAMLError as exc:
                issues.append(
                    WorkflowContractIssue(
                        code="malformed_workflow_yaml",
                        message=f"{member}: malformed YAML workflow config: {exc}",
                    )
                )
                continue
            if not _looks_like_workflow_config(loaded):
                continue
            for issue in validate_workflow_config(
                loaded,
                profile=profile,
                bundle_members=members,
            ):
                issues.append(
                    WorkflowContractIssue(
                        code=issue.code,
                        message=f"{member}: {issue.message}",
                    )
                )
    return issues


def _primary_workflow_members(archive: zipfile.ZipFile) -> frozenset[str]:
    primary_members = set(PRIMARY_WORKFLOW_YAML_MEMBERS)
    try:
        manifest = json.loads(archive.read("manifest.json"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return frozenset(primary_members)
    workflow = manifest.get("workflow") if isinstance(manifest, dict) else None
    if isinstance(workflow, str) and workflow:
        primary_members.add(workflow)
    return frozenset(primary_members)


def _is_secondary_yaml_member(
    member: str,
    primary_members: Iterable[str],
) -> bool:
    member_path = Path(member)
    if member in set(primary_members):
        return False
    if member_path.name.lower() in PRIMARY_WORKFLOW_YAML_MEMBERS:
        return False
    return member_path.suffix.lower() in {".yaml", ".yml"}


def _looks_like_workflow_config(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = {str(key) for key in value}
    return WORKFLOW_TEMPLATE_KEY in keys or len(keys & STRONG_WORKFLOW_SECTION_KEYS) >= 2


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _batch_name_suffixes(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item.get("name_suffix") if isinstance(item, dict) else None for item in value
    )


def _validate_label_materials(
    *,
    profile: str,
    config: dict[str, Any],
    expected: dict[int, dict[str, float]],
    issues: list[WorkflowContractIssue],
) -> None:
    labels = _mapping(_mapping(config.get("materials", {})).get("labels"))
    for label, expected_values in expected.items():
        material = _mapping(labels.get(label, labels.get(str(label))))
        for key, expected_value in expected_values.items():
            actual = material.get(key)
            actual_float = _to_float(actual)
            if actual_float is not None and actual_float == expected_value:
                continue
            issues.append(
                WorkflowContractIssue(
                    code="invalid_load_history_material",
                    message=(
                        f"Workflow {profile} must set materials.labels.{label}.{key} "
                        f"to {expected_value!r}; got {actual!r}"
                    ),
                )
            )


def _require_equal(
    issues: list[WorkflowContractIssue],
    *,
    profile: str,
    code: str,
    path: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual == expected:
        return
    issues.append(
        WorkflowContractIssue(
            code=code,
            message=f"Workflow {profile} must set {path} to {expected!r}; got {actual!r}",
        )
    )


def _require_float(
    issues: list[WorkflowContractIssue],
    *,
    profile: str,
    code: str,
    path: str,
    actual: Any,
    expected: float,
) -> None:
    actual_float = _to_float(actual)
    if actual_float is not None and actual_float == expected:
        return
    issues.append(
        WorkflowContractIssue(
            code=code,
            message=f"Workflow {profile} must set {path} to {expected!r}; got {actual!r}",
        )
    )


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
