from __future__ import annotations

import json
import subprocess

from parosol_py.workflow_baseline import build_builtin_workflow_baseline
from parosol_py.workflow_contracts import EXPECTED_PUBLIC_PROFILES


def test_build_builtin_workflow_baseline_has_public_profiles():
    baseline = build_builtin_workflow_baseline()

    assert tuple(sorted(baseline["profiles"])) == EXPECTED_PUBLIC_PROFILES
    assert baseline["schema_version"] == 1
    assert "git_sha" in baseline
    assert set(baseline["workflows"]) == set(EXPECTED_PUBLIC_PROFILES)


def test_baseline_is_json_serializable():
    baseline = build_builtin_workflow_baseline()

    encoded = json.dumps(baseline, sort_keys=True)

    assert "spine-compression" in encoded


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
