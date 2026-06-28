from __future__ import annotations

import json
import zipfile

import pytest

from parosol_py import workflow_contracts
from parosol_py.workflow_contracts import (
    EXPECTED_PUBLIC_PROFILES,
    WorkflowContractIssue,
    validate_all_builtin_workflows,
    validate_workflow_config,
)


def test_validator_rejects_public_protrusion_schema():
    config = {
        "workflow_template": {"profile": "example", "type": "single_case_fea"},
        "slicer_editor": {
            "planes": [
                {
                    "name": "Support",
                    "contact": "Material disks",
                    "protrusion_depth_mm": 4.0,
                }
            ]
        },
    }

    issues = validate_workflow_config(config, profile="example", bundle_members=())

    assert WorkflowContractIssue(
        code="forbidden_key",
        message="Forbidden workflow key protrusion_depth_mm at slicer_editor.planes.0.protrusion_depth_mm",
    ) in issues


def test_validator_requires_editor_planes_for_single_case_workflows():
    config = {
        "workflow_template": {"profile": "example", "type": "single_case_fea"},
        "model": {"workflow_replay": {"enabled": True}},
    }

    issues = validate_workflow_config(config, profile="example", bundle_members=())

    assert any(issue.code == "missing_editor_planes" for issue in issues)


def test_validator_accepts_transitional_npz_reference_points_member():
    issues = validate_workflow_config(
        {},
        profile="spine-compression",
        bundle_members=("reference/slicer_reference_points.npz",),
    )

    assert not any(issue.code == "missing_reference_points" for issue in issues)


def test_validator_accepts_npy_reference_points_member():
    issues = validate_workflow_config(
        {},
        profile="spine-compression",
        bundle_members=("reference/slicer_reference_points.npy",),
    )

    assert not any(issue.code == "missing_reference_points" for issue in issues)


def test_validator_rejects_vtk_reference_assets():
    issues = validate_workflow_config(
        {},
        profile="spine-compression",
        bundle_members=(
            "reference/slicer_reference_points.npy",
            "reference/slicer_reference_points.vtk",
        ),
    )

    assert any(issue.code == "vtk_reference_packaged" for issue in issues)


def test_validator_handles_non_mapping_workflow_template():
    config = {
        "workflow_template": "malformed",
        "slicer_editor": {
            "planes": [
                {
                    "name": "Support",
                    "contact": "Material disks",
                    "protrusion_depth_mm": 4.0,
                }
            ]
        },
    }

    issues = validate_workflow_config(config, profile="example", bundle_members=())

    assert WorkflowContractIssue(
        code="forbidden_key",
        message="Forbidden workflow key protrusion_depth_mm at slicer_editor.planes.0.protrusion_depth_mm",
    ) in issues


def test_expected_public_profiles_are_the_workflow_only_surface():
    assert EXPECTED_PUBLIC_PROFILES == (
        "XtremeCTI",
        "XtremeCTII",
        "hip-sideways-fall-left",
        "hip-sideways-fall-right",
        "load_history_3",
        "load_history_6",
        "spine-compression",
    )


def test_all_public_builtin_workflows_satisfy_contracts():
    results = validate_all_builtin_workflows()

    failures = {
        profile: [issue.message for issue in issues]
        for profile, issues in results.items()
        if issues
    }

    assert failures == {}


def test_builtin_validator_checks_secondary_workflow_yaml(tmp_path, monkeypatch):
    bundle = _workflow_bundle(
        tmp_path,
        workflow_yaml=_clean_spine_workflow_yaml(),
        secondary_members={
            "parosol_slicer_case.yaml": """
workflow_template:
  type: single_case_fea
slicer_editor:
  planes:
  - name: Superior disk
    protrusion_depth_mm: 3.5
solver:
  tolerance: 1.0e-06
load_case:
  type: nodeset
model:
  workflow_replay: {}
""",
        },
    )
    monkeypatch.setattr(
        workflow_contracts,
        "builtin_profile_path",
        lambda profile: str(bundle),
    )
    monkeypatch.setattr(
        workflow_contracts,
        "available_profiles",
        lambda: ("spine-compression",),
    )

    issues = workflow_contracts.validate_builtin_profile("spine-compression")

    assert WorkflowContractIssue(
        code="invalid_solver_tolerance",
        message=(
            "parosol_slicer_case.yaml: Workflow spine-compression must set "
            "solver.tolerance to 0.0001; got 1e-06"
        ),
    ) in issues
    assert any(
        issue.message
        == (
            "parosol_slicer_case.yaml: Forbidden workflow key "
            "protrusion_depth_mm at slicer_editor.planes.0.protrusion_depth_mm"
        )
        for issue in issues
    )
    assert any(
        issue.message
        == (
            "parosol_slicer_case.yaml: Workflow spine-compression must set "
            "model.workflow_replay.model_space to 'reference'; got None"
        )
        for issue in issues
    )


def test_builtin_validator_reports_malformed_secondary_workflow_yaml(
    tmp_path,
    monkeypatch,
):
    bundle = _workflow_bundle(
        tmp_path,
        workflow_yaml=_clean_spine_workflow_yaml(),
        secondary_members={"parosol_slicer_case.yaml": "solver: ["},
    )
    monkeypatch.setattr(
        workflow_contracts,
        "builtin_profile_path",
        lambda profile: str(bundle),
    )
    monkeypatch.setattr(
        workflow_contracts,
        "available_profiles",
        lambda: ("spine-compression",),
    )

    issues = workflow_contracts.validate_builtin_profile("spine-compression")

    assert any(
        issue.code == "malformed_workflow_yaml"
        and issue.message.startswith("parosol_slicer_case.yaml: malformed YAML")
        for issue in issues
    )


def test_builtin_validator_ignores_secondary_metadata_yaml_with_generic_model_key(
    tmp_path,
    monkeypatch,
):
    bundle = _workflow_bundle(
        tmp_path,
        workflow_yaml=_clean_spine_workflow_yaml(),
        secondary_members={"metadata.yaml": "model: scanner\n"},
    )
    monkeypatch.setattr(
        workflow_contracts,
        "builtin_profile_path",
        lambda profile: str(bundle),
    )
    monkeypatch.setattr(
        workflow_contracts,
        "available_profiles",
        lambda: ("spine-compression",),
    )

    issues = workflow_contracts.validate_builtin_profile("spine-compression")

    assert issues == []


def test_builtin_validator_skips_manifest_primary_workflow_member(
    tmp_path,
    monkeypatch,
):
    bundle = tmp_path / "fixture.parosol-workflow"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": "parosol-py-workflow",
                    "version": 1,
                    "workflow": "parosol_slicer_case.yaml",
                    "files": [
                        "parosol_slicer_case.yaml",
                        "reference/slicer_reference_points.npy",
                    ],
                }
            ),
        )
        archive.writestr(
            "parosol_slicer_case.yaml",
            """
workflow_template:
  type: single_case_fea
slicer_editor:
  planes:
  - name: Superior disk
    protrusion_depth_mm: 3.5
solver:
  tolerance: 1.0e-06
load_case:
  type: nodeset
model:
  workflow_replay: {}
""",
        )
        archive.writestr("reference/slicer_reference_points.npy", b"reference")
    monkeypatch.setattr(
        workflow_contracts,
        "builtin_profile_path",
        lambda profile: str(bundle),
    )
    monkeypatch.setattr(
        workflow_contracts,
        "available_profiles",
        lambda: ("spine-compression",),
    )

    issues = workflow_contracts.validate_builtin_profile("spine-compression")

    assert any(issue.code == "invalid_solver_tolerance" for issue in issues)
    assert not any(
        issue.message.startswith("parosol_slicer_case.yaml:") for issue in issues
    )


@pytest.mark.parametrize(
    "profile",
    [
        "XtremeCTI",
        "XtremeCTII",
        "spine-compression",
        "hip-sideways-fall-left",
        "hip-sideways-fall-right",
    ],
)
def test_validator_requires_public_solver_tolerance(profile):
    issues = validate_workflow_config(
        {"solver": {"tolerance": 1.0e-3}},
        profile=profile,
        bundle_members=("reference/slicer_reference_points.npy",),
    )

    assert any(issue.code == "invalid_solver_tolerance" for issue in issues)


@pytest.mark.parametrize("profile", ["XtremeCTI", "XtremeCTII"])
def test_validator_requires_xtremect_axial_load_case(profile):
    issues = validate_workflow_config(
        {
            "solver": {"tolerance": 1.0e-4},
            "load_case": {"type": "nodeset", "axis": "x", "strain": -0.02},
        },
        profile=profile,
        bundle_members=(),
    )

    assert {issue.code for issue in issues} >= {
        "invalid_load_case_type",
        "invalid_load_case_axis",
        "invalid_load_case_strain",
    }


@pytest.mark.parametrize(
    "profile",
    ["spine-compression", "hip-sideways-fall-left", "hip-sideways-fall-right"],
)
def test_validator_requires_reference_nodeset_replay_workflows(profile):
    issues = validate_workflow_config(
        {
            "solver": {"tolerance": 1.0e-4},
            "model": {"workflow_replay": {"model_space": "scanner"}},
            "load_case": {"type": "constrained_axial"},
        },
        profile=profile,
        bundle_members=("reference/slicer_reference_points.npy",),
    )

    assert {issue.code for issue in issues} >= {
        "invalid_workflow_replay_model_space",
        "invalid_load_case_type",
    }


@pytest.mark.parametrize(
    ("profile", "expected_suffixes"),
    [
        ("load_history_3", ["compression_z", "shear_zx", "shear_zy"]),
        (
            "load_history_6",
            [
                "compression_z",
                "shear_zx",
                "shear_zy",
                "bending_x",
                "bending_y",
                "torsion_z",
            ],
        ),
    ],
)
def test_validator_requires_load_history_suffix_order(profile, expected_suffixes):
    issues = validate_workflow_config(
        {
            "batch": {
                "cases": [
                    {"name_suffix": suffix}
                    for suffix in reversed(expected_suffixes)
                ]
            }
        },
        profile=profile,
        bundle_members=(),
    )

    assert any(issue.code == "invalid_batch_name_suffixes" for issue in issues)


def _workflow_bundle(
    tmp_path,
    *,
    workflow_yaml: str,
    secondary_members: dict[str, str],
):
    bundle = tmp_path / "fixture.parosol-workflow"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("workflow.yaml", workflow_yaml)
        archive.writestr("reference/slicer_reference_points.npy", b"reference")
        for member, content in secondary_members.items():
            archive.writestr(member, content)
    return bundle


def _clean_spine_workflow_yaml() -> str:
    return """
workflow_template:
  type: single_case_fea
slicer_editor:
  planes:
  - name: Superior disk
    intrusion_depth_mm: 3.5
solver:
  tolerance: 1.0e-04
load_case:
  type: nodeset
model:
  workflow_replay:
    model_space: reference
"""
