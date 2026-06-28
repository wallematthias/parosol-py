from __future__ import annotations

import json

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
