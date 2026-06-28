from __future__ import annotations

from parosol_py.workflow_contracts import (
    EXPECTED_PUBLIC_PROFILES,
    WorkflowContractIssue,
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
