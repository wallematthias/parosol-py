from __future__ import annotations

import json
from pathlib import Path
import subprocess

from parosol_py.workflow_baseline import build_builtin_workflow_baseline
from parosol_py.workflow_baseline import _canonicalize_loaded_workflow
from parosol_py.workflow_contracts import EXPECTED_PUBLIC_PROFILES


def test_build_builtin_workflow_baseline_has_public_profiles():
    baseline = build_builtin_workflow_baseline()

    assert tuple(sorted(baseline["profiles"])) == EXPECTED_PUBLIC_PROFILES
    assert baseline["schema_version"] == 2
    assert "git_sha" in baseline
    assert set(baseline["workflows"]) == set(EXPECTED_PUBLIC_PROFILES)


def test_baseline_is_json_serializable():
    baseline = build_builtin_workflow_baseline()

    encoded = json.dumps(baseline, sort_keys=True)

    assert "spine-compression" in encoded


def test_baseline_records_workflow_fingerprints():
    baseline = build_builtin_workflow_baseline()
    spine = baseline["workflows"]["spine-compression"]

    assert "workflow.yaml" in spine["bundle_members"]
    assert "workflow.yaml" in spine["bundle_member_sha256"]
    assert len(spine["bundle_member_sha256"]["workflow.yaml"]) == 64
    assert len(spine["config_sha256"]) == 64
    assert spine["workflow_replay"]["enabled"] is True
    assert spine["workflow_replay"]["model_space"] == "reference"


def test_baseline_config_hashes_are_stable_across_temp_extractions():
    first = build_builtin_workflow_baseline()
    second = build_builtin_workflow_baseline()

    for profile in first["profiles"]:
        assert (
            first["workflows"][profile]["config_sha256"]
            == second["workflows"][profile]["config_sha256"]
        )


def test_baseline_canonicalizes_workflow_bundle_temp_roots():
    paths = {
        "linux": "/tmp/parosol_workflow_abc123/reference/slicer_reference_points.npy",
        "mac": "/var/folders/zz/parosol_workflow_def456/reference/slicer_reference_points.npy",
    }

    canonical = _canonicalize_loaded_workflow(paths)

    assert canonical == {
        "linux": "<workflow_bundle>/reference/slicer_reference_points.npy",
        "mac": "<workflow_bundle>/reference/slicer_reference_points.npy",
    }


def test_baseline_git_sha_ignores_caller_cwd_git_repo(tmp_path, monkeypatch):
    unrelated_repo = tmp_path / "unrelated"
    unrelated_repo.mkdir()
    subprocess.run(
        ["git", "init"],
        check=True,
        capture_output=True,
        cwd=unrelated_repo,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=ParOSol Test",
            "-c",
            "user.email=parosol@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "unrelated",
        ],
        check=True,
        capture_output=True,
        cwd=unrelated_repo,
        text=True,
    )
    unrelated_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        cwd=unrelated_repo,
        text=True,
    ).stdout.strip()

    monkeypatch.chdir(unrelated_repo)

    baseline = build_builtin_workflow_baseline()
    baseline_git_sha = baseline["git_sha"]

    assert baseline_git_sha != unrelated_sha


def test_builtin_workflow_behavior_matches_locked_fixture():
    expected_path = Path(__file__).parent / "fixtures" / "builtin_workflow_baseline_locked.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    assert _locked_workflow_baseline() == expected


def _locked_workflow_baseline():
    baseline = build_builtin_workflow_baseline()
    return {
        "schema_version": 1,
        "profiles": baseline["profiles"],
        "workflows": {
            profile: _locked_workflow_summary(workflow)
            for profile, workflow in baseline["workflows"].items()
        },
    }


def _locked_workflow_summary(workflow):
    locked_member_hashes = {
        member: digest
        for member, digest in workflow["bundle_member_sha256"].items()
        if member == "workflow.yaml" or member.startswith("reference/")
    }
    return {
        "config_sha256": workflow["config_sha256"],
        "bundle_member_sha256": locked_member_hashes,
        "workflow_type": workflow["workflow_type"],
        "profile": workflow["profile"],
        "plane_count": workflow["plane_count"],
        "load_count": workflow["load_count"],
        "load_case_type": workflow["load_case_type"],
        "batch_case_count": workflow["batch_case_count"],
        "model_type": workflow["model_type"],
        "registration_enabled": workflow["registration_enabled"],
        "workflow_replay_enabled": workflow["workflow_replay_enabled"],
        "workflow_replay_model_space": workflow["workflow_replay_model_space"],
        "solver_tolerance": workflow["solver_tolerance"],
    }
