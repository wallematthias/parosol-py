# ParOSol Workflow Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize `parosol-py` workflow replay so `.parosol-workflow` bundles, `parosol_case.yaml`, generated geometry, built-in recipes, load-history workflows, and Ogo/n88 parity are protected by explicit contracts and regression tests.

**Architecture:** Add contract and baseline tooling first, then ratchet geometry and replay tests around the existing `parosol_py.workflow_geometry`, `parosol_py.workflow_template`, and `parosol_py.modeling.workflow_replay` modules. Keep built-in workflows as packaged workflow bundles; update bundle contents only when contract tests force a deliberate schema or metadata correction.

**Tech Stack:** Python, NumPy, SimpleITK, PyYAML, pytest, zipfile, existing `parosol-py` workflow/modeling/batch APIs.

---

## File Structure

- Create `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_contracts.py`
  Owns workflow contract validation for public profiles and arbitrary workflow configs.
- Create `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_workflow_contracts.py`
  Tests validator behavior and packaged workflow contracts.
- Create `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_baseline.py`
  Builds compact JSON-serializable workflow baseline snapshots without running expensive solves.
- Create `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/scripts/workflow_baseline.py`
  Thin command-line wrapper around `parosol_py.workflow_baseline`.
- Create `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_workflow_baseline.py`
  Tests baseline snapshot structure using packaged workflows.
- Modify `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_geometry.py`
  Add focused helpers only where exact geometry tests require implementation changes.
- Modify `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_workflow_geometry.py`
  Add exact geometry tests for disk/bone exclusion, bbox scaling, intrusion, and axis-aligned equivalence.
- Modify `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/modeling/workflow_replay.py`
  Tighten metadata and replay precedence only where replay tests prove drift.
- Modify `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_modeling.py`
  Add replay tests for plane precedence, artifact-label consistency, and percent displacement reference length.
- Modify `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/*.parosol-workflow`
  Update packaged workflow YAML inside bundles only after contract tests expose concrete drift.
- Create `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/parity.py`
  Provides lightweight comparison helpers for Ogo/n88 parity reports.
- Create `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/scripts/run_ogo_parity.py`
  Optional harness for local parity runs when the reference bundle is available.
- Create `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_parity.py`
  Tests parity metric comparison without running the solver.

## Working Tree Rule

The current branch has unrelated dirty files. Each task must stage only the files named in that task. Do not reset, checkout, or remove existing dirty files unless the user explicitly asks.

---

### Task 1: Add Workflow Contract Validator

**Files:**
- Create: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_contracts.py`
- Create: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_workflow_contracts.py`

- [ ] **Step 1: Write failing validator unit tests**

Add this file:

```python
# /Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_workflow_contracts.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_workflow_contracts.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'parosol_py.workflow_contracts'`.

- [ ] **Step 3: Implement the validator module**

Create:

```python
# /Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_contracts.py
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
    if profile in {"spine-compression", "hip-sideways-fall-left", "hip-sideways-fall-right"}:
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


def _find_forbidden_keys(value: Any, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
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
```

- [ ] **Step 4: Run validator tests**

Run:

```bash
pytest -q tests/test_workflow_contracts.py -v
```

Expected: PASS for the three validator behavior tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/parosol_py/workflow_contracts.py tests/test_workflow_contracts.py
git commit -m "test: add workflow contract validator"
```

Expected: commit includes only the new validator module and tests.

---

### Task 2: Add Workflow Baseline Snapshot Tool

**Files:**
- Create: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_baseline.py`
- Create: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/scripts/workflow_baseline.py`
- Create: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_workflow_baseline.py`

- [ ] **Step 1: Write failing baseline tests**

Create:

```python
# /Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_workflow_baseline.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_workflow_baseline.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'parosol_py.workflow_baseline'`.

- [ ] **Step 3: Implement baseline module**

Create:

```python
# /Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_baseline.py
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
```

- [ ] **Step 4: Add command-line wrapper**

Create:

```python
# /Users/matthias.walle/Documents/14_GitHub/active/parosol-py/scripts/workflow_baseline.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from parosol_py.workflow_baseline import build_builtin_workflow_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a compact built-in workflow baseline JSON file.")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_builtin_workflow_baseline(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run baseline tests**

Run:

```bash
pytest -q tests/test_workflow_baseline.py -v
```

Expected: PASS.

- [ ] **Step 6: Capture local baseline artifact**

Run:

```bash
python scripts/workflow_baseline.py --output scratch/workflow_baseline_current.json
```

Expected: writes `scratch/workflow_baseline_current.json`. Do not stage `scratch/`.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/parosol_py/workflow_baseline.py scripts/workflow_baseline.py tests/test_workflow_baseline.py
git commit -m "test: add workflow baseline snapshot tool"
```

Expected: commit includes only the baseline module, script, and tests.

---

### Task 3: Enforce Public Built-In Workflow Contracts

**Files:**
- Modify: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_workflow_contracts.py`
- Modify if test failures require it: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_contracts.py`
- Modify if test failures require it: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/XtremeCTI.parosol-workflow`
- Modify if test failures require it: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/XtremeCTII.parosol-workflow`
- Modify if test failures require it: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/spine-compression.parosol-workflow`
- Modify if test failures require it: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/hip-sideways-fall-left.parosol-workflow`
- Modify if test failures require it: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/hip-sideways-fall-right.parosol-workflow`
- Modify if test failures require it: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/load_history_3.parosol-workflow`
- Modify if test failures require it: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/load_history_6.parosol-workflow`

- [ ] **Step 1: Add built-in contract tests**

Append to `tests/test_workflow_contracts.py`:

```python
from parosol_py.workflow_contracts import validate_all_builtin_workflows


def test_all_public_builtin_workflows_satisfy_contracts():
    results = validate_all_builtin_workflows()

    failures = {
        profile: [issue.message for issue in issues]
        for profile, issues in results.items()
        if issues
    }

    assert failures == {}
```

- [ ] **Step 2: Run tests to expose current contract drift**

Run:

```bash
pytest -q tests/test_workflow_contracts.py -v
```

Expected: FAIL only if bundled workflows or validator expectations disagree. The failure output lists exact profile and issue messages.

- [ ] **Step 3: Tighten validator expectations for family-specific contracts**

If the built-in contract test passes without checking family semantics, extend `workflow_contracts.py` with:

```python
def _validate_family_semantics(
    profile: str,
    config: dict[str, Any],
    issues: list[WorkflowContractIssue],
) -> None:
    solver = config.get("solver", {})
    load_case = config.get("load_case", {})
    batch = config.get("batch", {})
    model = config.get("model", {})
    replay = model.get("workflow_replay", {}) if isinstance(model, dict) else {}

    if profile in {"XtremeCTI", "XtremeCTII", "spine-compression", "hip-sideways-fall-left", "hip-sideways-fall-right"}:
        if float(solver.get("tolerance", 0.0)) != 1.0e-4:
            issues.append(WorkflowContractIssue("wrong_solver_tolerance", f"{profile} must use solver.tolerance=1.0e-4"))

    if profile in {"XtremeCTI", "XtremeCTII"}:
        if load_case.get("type") != "constrained_axial":
            issues.append(WorkflowContractIssue("wrong_load_case", f"{profile} must use constrained_axial"))
        if load_case.get("axis") != "z":
            issues.append(WorkflowContractIssue("wrong_load_axis", f"{profile} must load along z"))
        if float(load_case.get("strain", 0.0)) != -0.01:
            issues.append(WorkflowContractIssue("wrong_strain", f"{profile} must use -0.01 strain"))

    if profile == "spine-compression":
        if replay.get("model_space") != "reference":
            issues.append(WorkflowContractIssue("wrong_model_space", "spine-compression must replay in reference model space"))
        if load_case.get("type") != "nodeset":
            issues.append(WorkflowContractIssue("wrong_load_case", "spine-compression must use nodeset load_case"))

    if profile.startswith("hip-sideways-fall"):
        if replay.get("model_space") != "reference":
            issues.append(WorkflowContractIssue("wrong_model_space", f"{profile} must replay in reference model space"))
        if load_case.get("type") != "nodeset":
            issues.append(WorkflowContractIssue("wrong_load_case", f"{profile} must use nodeset load_case"))

    if profile == "load_history_3":
        names = [case.get("name_suffix") for case in batch.get("cases", [])]
        if names != ["compression_z", "shear_zx", "shear_zy"]:
            issues.append(WorkflowContractIssue("wrong_batch_cases", "load_history_3 must contain compression_z, shear_zx, shear_zy"))

    if profile == "load_history_6":
        names = [case.get("name_suffix") for case in batch.get("cases", [])]
        expected = ["compression_z", "shear_zx", "shear_zy", "bending_x", "bending_y", "torsion_z"]
        if names != expected:
            issues.append(WorkflowContractIssue("wrong_batch_cases", f"load_history_6 must contain {expected}"))
```

Call it from `validate_workflow_config(...)` before returning:

```python
    _validate_family_semantics(profile, config, issues)
```

- [ ] **Step 4: Fix concrete packaged workflow drift**

If a workflow bundle needs YAML changes, use this exact Python helper from the repo root and change only the YAML keys indicated by test failures:

```bash
python - <<'PY'
from pathlib import Path
import shutil
import tempfile
import zipfile
import yaml

bundle = Path("src/parosol_py/workflows/spine-compression.parosol-workflow")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(root)
    workflow = root / "workflow.yaml"
    data = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    data.setdefault("solver", {})["tolerance"] = 1.0e-4
    workflow.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    backup = bundle.with_suffix(bundle.suffix + ".bak")
    shutil.copy2(bundle, backup)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                out.write(path, path.relative_to(root).as_posix())
    backup.unlink()
PY
```

Replace `spine-compression.parosol-workflow` and the YAML assignment with the specific workflow and key from the failing contract. Do not bulk-normalize unrelated YAML formatting unless the test requires that file.

- [ ] **Step 5: Run contract and packaging tests**

Run:

```bash
pytest -q tests/test_workflow_contracts.py tests/test_config_templates.py tests/test_packaging.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/parosol_py/workflow_contracts.py tests/test_workflow_contracts.py tests/test_config_templates.py tests/test_packaging.py src/parosol_py/workflows/*.parosol-workflow
git commit -m "test: enforce public workflow contracts"
```

Expected: commit contains only validator/test changes and workflow bundles changed by failing contract tests.

---

### Task 4: Ratchet Workflow Geometry Tests

**Files:**
- Modify: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_workflow_geometry.py`
- Modify if needed: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_geometry.py`

- [ ] **Step 1: Add exact disk/bone exclusion test**

Append to `tests/test_workflow_geometry.py`:

```python
def test_projected_material_disk_never_labels_bone_voxels():
    mask_xyz = np.zeros((7, 7, 7), dtype=bool)
    mask_xyz[2:5, 2:5, 2:5] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    editor = {
        "planes": [
            {
                "name": "Superior disk",
                "contact": "Material disks",
                "surface_mode": "project_bounded",
                "shape": "anatomy",
                "thickness_mm": 2.0,
                "intrusion_depth_mm": 2.0,
                "center_ras": [3.0, 3.0, 6.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [5.0, 5.0],
            }
        ]
    }

    geometry = generate_disk_and_nodeset_geometry(
        editor,
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_labels={"superior_disk": 201},
        nodeset_names={"Superior disk": "superior_disk"},
        disk_labels={"Superior disk": 22},
    )

    disk = geometry.disk_labels_xyz == 22
    assert np.count_nonzero(disk) > 0
    assert np.count_nonzero(disk & mask_xyz) == 0
```

- [ ] **Step 2: Add intrusion monotonicity test**

Append:

```python
def test_larger_intrusion_wraps_more_anatomy_columns_without_entering_bone():
    mask_xyz = np.zeros((9, 9, 9), dtype=bool)
    mask_xyz[3:6, 3:6, 4:6] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0

    def build(intrusion_depth_mm: float):
        editor = {
            "planes": [
                {
                    "name": "Support disk",
                    "contact": "Material disks",
                    "surface_mode": "project_bounded",
                    "shape": "anatomy",
                    "thickness_mm": 2.0,
                    "intrusion_depth_mm": intrusion_depth_mm,
                    "center_ras": [4.0, 4.0, 8.0],
                    "normal_ras": [0.0, 0.0, -1.0],
                    "u_axis_ras": [1.0, 0.0, 0.0],
                    "v_axis_ras": [0.0, 1.0, 0.0],
                    "size_mm": [6.0, 6.0],
                }
            ]
        }
        return generate_disk_and_nodeset_geometry(
            editor,
            mask_xyz=mask_xyz,
            material_xyz=material_xyz,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            nodeset_labels={"support_disk": 202},
            nodeset_names={"Support disk": "support_disk"},
            disk_labels={"Support disk": 22},
        ).disk_labels_xyz == 22

    shallow = build(0.0)
    wrapped = build(3.0)

    assert np.count_nonzero(wrapped) >= np.count_nonzero(shallow)
    assert np.count_nonzero(wrapped & mask_xyz) == 0
```

- [ ] **Step 3: Add axis-aligned equivalence test**

Append:

```python
def test_axis_aligned_projection_matches_general_projected_surface():
    mask_xyz = np.zeros((8, 8, 8), dtype=bool)
    mask_xyz[2:6, 2:6, 3:6] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    base_plane = {
        "name": "Top",
        "contact": "Bone surface",
        "surface_mode": "project_bounded",
        "shape": "anatomy",
        "thickness_mm": 0.0,
        "intrusion_depth_mm": 0.0,
        "center_ras": [3.5, 3.5, 7.0],
        "normal_ras": [0.0, 0.0, -1.0],
        "u_axis_ras": [1.0, 0.0, 0.0],
        "v_axis_ras": [0.0, 1.0, 0.0],
        "size_mm": [5.0, 5.0],
    }

    geometry = generate_disk_and_nodeset_geometry(
        {"planes": [base_plane]},
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_labels={"top": 201},
        nodeset_names={"Top": "top"},
    )

    nodes = np.argwhere(geometry.nodeset_labels_xyz == 201)
    assert nodes.size > 0
    assert np.unique(nodes[:, 2]).tolist() == [5]
```

- [ ] **Step 4: Run geometry tests**

Run:

```bash
pytest -q tests/test_workflow_geometry.py -v
```

Expected: PASS if current geometry already satisfies the ratchet, otherwise FAIL on the precise behavior to fix.

- [ ] **Step 5: Fix geometry only if tests fail**

If disk labels enter bone voxels, ensure `_generate_projected_disk_mask(...)` keeps the current `empty_mask` exclusion and applies it after every branch:

```python
    final = inside_all & d_ok & empty_mask & bucket_mask
```

If the axis-aligned test fails because projection chooses the wrong surface, fix `_first_surface_points_by_bucket(...)` so the selected voxel is the nearest candidate along the plane normal:

```python
    for voxel, dist, uu, vv in zip(idx, distance, u, v, strict=True):
        key = _bucket_key(float(uu), float(vv), spacing=spacing)
        current = best.get(key)
        if current is None or float(dist) < current[0]:
            best[key] = (float(dist), np.asarray(voxel, dtype=np.int64))
```

- [ ] **Step 6: Run geometry tests again**

Run:

```bash
pytest -q tests/test_workflow_geometry.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add tests/test_workflow_geometry.py src/parosol_py/workflow_geometry.py
git commit -m "test: ratchet workflow geometry behavior"
```

Expected: commit contains geometry tests and only required geometry fixes.

---

### Task 5: Ratchet Workflow Replay Model Tests

**Files:**
- Modify: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_modeling.py`
- Modify if needed: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/modeling/workflow_replay.py`
- Modify if needed: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/nodesets.py`

- [ ] **Step 1: Add exported artifact label consistency test**

Update the imports in `tests/test_modeling.py`:

```python
from parosol_py.nodesets import nodes_from_labeled_voxels
```

and extend the existing `parosol_py.modeling.common` import to include:

```python
occupied_length_mm
```

Append to `tests/test_modeling.py`:

```python
def test_workflow_replay_exports_generated_nodeset_labels_from_planes(tmp_path: Path):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 20
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"body": 20},
            "outputs": {
                "material_image": str(tmp_path / "model" / "material.nii.gz"),
                "nodeset_image": str(tmp_path / "model" / "nodesets.nii.gz"),
                "manifest": str(tmp_path / "model" / "model.json"),
            },
            "workflow_replay": {"enabled": True},
            "registration": {"enabled": False},
            "slicer_editor": {
                "planes": [
                    {
                        "name": "Superior disk",
                        "relative_to": "model_bbox",
                        "center_fraction": [0.5, 0.5, 1.25],
                        "size_fraction": [1.5, 1.5],
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "anatomy",
                        "thickness_mm": 2.0,
                        "intrusion_depth_mm": 1.0,
                        "normal_ras": [0.0, 0.0, -1.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 1.0, 0.0],
                    }
                ]
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={
            "type": "nodeset",
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": "-10%"}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    labels_zyx, _spacing, _origin = read_image_zyx(tmp_path / "model" / "nodesets.nii.gz")
    labels_xyz = np.transpose(labels_zyx, (2, 1, 0))
    material_xyz = np.transpose(built.material, (2, 1, 0))
    reconstructed = nodes_from_labeled_voxels(
        labels_xyz,
        label=201,
        selection="surface_nodes",
        material=material_xyz,
    )
    assert built.metadata["model"]["workflow_replay"]["geometry_mode"] == "plane_driven"
    assert reconstructed == built.node_sets["superior_disk"]
```

- [ ] **Step 2: Add percent-displacement reference length test**

Append:

```python
def test_workflow_replay_percent_displacement_uses_occupied_model_length_with_disks(tmp_path: Path):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 20
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"body": 20},
            "workflow_replay": {"enabled": True},
            "registration": {"enabled": False},
            "slicer_editor": {
                "planes": [
                    {
                        "name": "Superior disk",
                        "relative_to": "model_bbox",
                        "center_fraction": [0.5, 0.5, 1.25],
                        "size_fraction": [1.5, 1.5],
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "anatomy",
                        "thickness_mm": 2.0,
                        "intrusion_depth_mm": 0.0,
                        "normal_ras": [0.0, 0.0, -1.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 1.0, 0.0],
                    }
                ]
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={
            "type": "nodeset",
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": "-10%"}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    values = built.boundary_conditions.fixed_values
    prescribed = values[np.abs(values) > 0.0]
    material_xyz = np.transpose(built.material, (2, 1, 0))
    occupied = occupied_length_mm(material_xyz, axis="z", spacing=built.spacing)
    assert prescribed.size > 0
    assert float(np.min(prescribed)) == pytest.approx(-0.10 * occupied)
```

- [ ] **Step 3: Run replay tests**

Run:

```bash
pytest -q tests/test_modeling.py -k "workflow_replay and (exports_generated or percent_displacement or prefers_plane)" -v
```

Expected: PASS if replay already satisfies these contracts, otherwise FAIL on the precise replay behavior.

- [ ] **Step 4: Fix replay only if tests fail**

If exported labels do not match built node sets, ensure `labels_xyz[node_label_xyz > 0]` is written after plane-driven `node_label_xyz` generation and before `export_model_artifacts(...)`.

If percent displacement does not include disks, keep this existing pattern in `build_workflow_replay_model(...)` and fix any regression around `material_xyz` population before boundary conditions are built:

```python
        percent_reference_lengths_mm={
            axis: occupied_length_mm(material_xyz, axis=axis, spacing=spacing)
            for axis in ("x", "y", "z")
        },
```

- [ ] **Step 5: Run replay-focused tests again**

Run:

```bash
pytest -q tests/test_modeling.py -k "workflow_replay" -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tests/test_modeling.py src/parosol_py/modeling/workflow_replay.py src/parosol_py/nodesets.py
git commit -m "test: ratchet workflow replay contracts"
```

Expected: commit contains replay tests and only required replay fixes.

---

### Task 6: Protect Load-History Workflow Recipes

**Files:**
- Modify: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_batch.py`
- Modify if needed: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/batch.py`
- Modify if needed: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/load_history_3.parosol-workflow`
- Modify if needed: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/load_history_6.parosol-workflow`

- [ ] **Step 1: Add load-history recipe contract test**

Append to `tests/test_batch.py`:

```python
def test_load_history_6_workflow_recipe_keeps_unit_case_order_and_rotational_units():
    from parosol_py.workflow_registry import builtin_profile_path
    from parosol_py.workflow_template import load_workflow_template

    config, _source = load_workflow_template(builtin_profile_path("load_history_6"))
    cases = config["batch"]["cases"]

    assert [case["name_suffix"] for case in cases] == [
        "compression_z",
        "shear_zx",
        "shear_zy",
        "bending_x",
        "bending_y",
        "torsion_z",
    ]
    assert cases[3]["load_case"]["type"] == "bending"
    assert cases[4]["load_case"]["type"] == "bending"
    assert cases[5]["load_case"]["type"] == "torsion"
    assert config["postprocess"]["load_history"]["cases"] == [
        "compression_z",
        "shear_zx",
        "shear_zy",
        "bending_x",
        "bending_y",
        "torsion_z",
    ]
```

- [ ] **Step 2: Run load-history tests**

Run:

```bash
pytest -q tests/test_batch.py -k "load_history" -v
```

Expected: PASS if recipe and current load-history machinery are intact.

- [ ] **Step 3: Fix only if the new recipe test fails**

If the batch order or postprocess cases drifted, update the YAML inside `load_history_6.parosol-workflow` to:

```yaml
batch:
  cases:
    - name_suffix: compression_z
      load_case: {type: constrained_axial, axis: z, strain: -0.01}
    - name_suffix: shear_zx
      load_case: {type: shear, axis: z, direction: x, strain: -0.01}
    - name_suffix: shear_zy
      load_case: {type: shear, axis: z, direction: y, strain: -0.01}
    - name_suffix: bending_x
      load_case:
        type: bending
        axis: z
        bending_angle_degrees: -1
        neutral_axis_angle_degrees: 90
        center: center_of_mass
    - name_suffix: bending_y
      load_case:
        type: bending
        axis: z
        bending_angle_degrees: -1
        neutral_axis_angle_degrees: 0
        center: center_of_mass
    - name_suffix: torsion_z
      load_case:
        type: torsion
        axis: z
        twist_angle_degrees: -1
        center: center_of_mass
postprocess:
  load_history:
    cases: [compression_z, shear_zx, shear_zy, bending_x, bending_y, torsion_z]
```

- [ ] **Step 4: Run combined load-history and contract tests**

Run:

```bash
pytest -q tests/test_batch.py -k "load_history" tests/test_workflow_contracts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/test_batch.py src/parosol_py/batch.py src/parosol_py/workflows/load_history_3.parosol-workflow src/parosol_py/workflows/load_history_6.parosol-workflow
git commit -m "test: protect load-history workflow recipes"
```

Expected: commit contains the recipe test and only required recipe or batch fixes.

---

### Task 7: Add Ogo/N88 Parity Comparison Harness

**Files:**
- Create: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/parity.py`
- Create: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/scripts/run_ogo_parity.py`
- Create: `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_parity.py`

- [ ] **Step 1: Write failing parity tests**

Create:

```python
# /Users/matthias.walle/Documents/14_GitHub/active/parosol-py/tests/test_parity.py
from __future__ import annotations

from parosol_py.parity import compare_metric, summarize_metric_comparisons


def test_compare_metric_reports_absolute_and_percent_error():
    comparison = compare_metric("stiffness_N_per_mm", observed=1505.58, expected=1671.01)

    assert comparison["name"] == "stiffness_N_per_mm"
    assert comparison["observed"] == 1505.58
    assert comparison["expected"] == 1671.01
    assert comparison["absolute_error"] == abs(1505.58 - 1671.01)
    assert comparison["relative_error_percent"] == abs(1505.58 - 1671.01) / 1671.01 * 100.0


def test_summarize_metric_comparisons_flags_tolerance_failures():
    summary = summarize_metric_comparisons(
        [
            compare_metric("force", observed=90.0, expected=100.0, tolerance_percent=15.0),
            compare_metric("stiffness", observed=80.0, expected=100.0, tolerance_percent=5.0),
        ]
    )

    assert summary["status"] == "failed"
    assert summary["failed_metrics"] == ["stiffness"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_parity.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'parosol_py.parity'`.

- [ ] **Step 3: Implement parity helpers**

Create:

```python
# /Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/parity.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json


def compare_metric(
    name: str,
    *,
    observed: float,
    expected: float,
    tolerance_percent: float = 10.0,
) -> dict[str, Any]:
    absolute_error = abs(float(observed) - float(expected))
    relative = None if float(expected) == 0.0 else absolute_error / abs(float(expected)) * 100.0
    passed = relative is not None and relative <= float(tolerance_percent)
    return {
        "name": name,
        "observed": float(observed),
        "expected": float(expected),
        "absolute_error": absolute_error,
        "relative_error_percent": relative,
        "tolerance_percent": float(tolerance_percent),
        "status": "passed" if passed else "failed",
    }


def summarize_metric_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [item["name"] for item in comparisons if item.get("status") != "passed"]
    return {
        "status": "passed" if not failed else "failed",
        "failed_metrics": failed,
        "comparisons": comparisons,
    }


def read_first_results_csv_row(path: str | Path) -> dict[str, str]:
    with Path(path).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return dict(row)
    raise ValueError(f"results CSV has no data rows: {path}")


def write_parity_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output
```

- [ ] **Step 4: Add optional parity script**

Create:

```python
# /Users/matthias.walle/Documents/14_GitHub/active/parosol-py/scripts/run_ogo_parity.py
from __future__ import annotations

import argparse
from pathlib import Path

from parosol_py.parity import compare_metric, summarize_metric_comparisons, write_parity_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare compact ParOSol and Ogo/n88 parity metrics.")
    parser.add_argument("--case", required=True, choices=("hip_10001", "spine_10001"))
    parser.add_argument("--observed-force", required=True, type=float)
    parser.add_argument("--expected-force", required=True, type=float)
    parser.add_argument("--observed-stiffness", required=True, type=float)
    parser.add_argument("--expected-stiffness", required=True, type=float)
    parser.add_argument("--tolerance-percent", type=float, default=10.0)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    summary = summarize_metric_comparisons(
        [
            compare_metric(
                "reaction_force_N",
                observed=args.observed_force,
                expected=args.expected_force,
                tolerance_percent=args.tolerance_percent,
            ),
            compare_metric(
                "stiffness_N_per_mm",
                observed=args.observed_stiffness,
                expected=args.expected_stiffness,
                tolerance_percent=args.tolerance_percent,
            ),
        ]
    )
    summary["case"] = args.case
    write_parity_summary(Path(args.summary), summary)
    print(summary["status"])
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run parity tests**

Run:

```bash
pytest -q tests/test_parity.py -v
```

Expected: PASS.

- [ ] **Step 6: Run a local comparison using known hip parity numbers**

Run:

```bash
python scripts/run_ogo_parity.py \
  --case hip_10001 \
  --observed-force 5721.20549955838 \
  --expected-force 6283.0 \
  --observed-stiffness 1505.5804135131707 \
  --expected-stiffness 1671.0106382978724 \
  --tolerance-percent 12 \
  --summary scratch/hip_10001_parity_summary.json
```

Expected: exits 0 and prints `passed` for the current known roughly 9-10% hip delta. Do not stage `scratch/`.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/parosol_py/parity.py scripts/run_ogo_parity.py tests/test_parity.py
git commit -m "test: add Ogo parity comparison harness"
```

Expected: commit contains parity helpers, optional script, and tests.

---

### Task 8: Run Stabilization Verification Suite

**Files:**
- Modify only if failures require targeted fixes:
  `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_geometry.py`
  `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/modeling/workflow_replay.py`
  `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflow_template.py`
  `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/batch.py`
  `/Users/matthias.walle/Documents/14_GitHub/active/parosol-py/src/parosol_py/workflows/*.parosol-workflow`

- [ ] **Step 1: Run focused stabilization tests**

Run:

```bash
pytest -q \
  tests/test_workflow_contracts.py \
  tests/test_workflow_baseline.py \
  tests/test_workflow_geometry.py \
  tests/test_modeling.py -k "workflow_replay or bbox_relative" \
  tests/test_batch.py -k "load_history" \
  tests/test_parity.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run packaging and CLI workflow tests**

Run:

```bash
pytest -q tests/test_config_templates.py tests/test_packaging.py tests/test_config_cli.py -k "workflow or profile or shortcut or batch" -v
```

Expected: PASS.

- [ ] **Step 3: Run local baseline capture again**

Run:

```bash
python scripts/workflow_baseline.py --output scratch/workflow_baseline_after_stabilization.json
```

Expected: writes `scratch/workflow_baseline_after_stabilization.json`. Compare with the first baseline and inspect any contract issue count changes:

```bash
python - <<'PY'
import json
from pathlib import Path
before = json.loads(Path("scratch/workflow_baseline_current.json").read_text())
after = json.loads(Path("scratch/workflow_baseline_after_stabilization.json").read_text())
print("before git:", before["git_sha"])
print("after git:", after["git_sha"])
for profile, item in after["workflows"].items():
    print(profile, "contract issues:", item["contract_issue_count"])
PY
```

Expected: every printed `contract issues` count is `0`.

- [ ] **Step 4: Run broad non-solver test suite**

Run:

```bash
pytest -q tests/test_workflow_contracts.py tests/test_workflow_baseline.py tests/test_workflow_geometry.py tests/test_modeling.py tests/test_batch.py tests/test_config_templates.py tests/test_packaging.py tests/test_config_cli.py tests/test_parity.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit final targeted fixes if any were needed**

If Step 1 through Step 4 required additional fixes, commit only those files:

```bash
git add src/parosol_py tests scripts
git commit -m "fix: stabilize workflow replay contracts"
```

Expected: no commit is created if no files changed during Task 8.

---

## Self-Review Checklist

- Spec coverage: Tasks 1-3 cover workflow contracts and built-in profile protection. Tasks 4-5 cover geometry and replay ratchets. Task 6 covers load-history compatibility. Task 7 covers Ogo/n88 parity reporting. Task 8 covers verification.
- Scope: The plan avoids solver rewrites, VTK dependencies, and spine/hip-specific modeling modules.
- Dirty tree protection: Every task stages named files only and does not reset unrelated changes.
- Acceptance: Completion requires contract tests, geometry tests, replay tests, load-history tests, parity helper tests, packaging tests, and CLI workflow tests to pass.
