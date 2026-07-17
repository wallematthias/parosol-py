### Task 7 Report: Workflow Configuration Integration

Status: complete

Summary:
- Added `materials.nonlinear.preset` parsing for `spine_keaveny` and `hip_keaveny`.
- Unknown presets raise exactly: `materials.nonlinear.preset must be 'spine_keaveny' or 'hip_keaveny'`.
- Direct density config runs now build the selected Keaveny nonlinear map, use its Young's modulus for stiffness, transpose it for the solve/write path, and write `/Nonlinear`.
- Workflow replay now builds a nonlinear material map from the calibrated density and active model mask, carries it on `BuiltModel`, crops it with the final workflow material grid, and passes it to `solve`.
- Hip nonlinear config requires `materials.density.basis: rho_app`; no conversion chain was added.
- Linear behavior remains unchanged when `materials.nonlinear` is absent.

Tests:
- `conda run -n ogoloco-n88 pytest tests/test_config_cli.py -k nonlinear -v`
  - 4 passed, 51 deselected
- `conda run -n ogoloco-n88 pytest tests/test_modeling.py -k nonlinear -v`
  - 1 passed, 60 deselected
- New specific tests were also run directly during red/green work.

Adjacent changes:
- `src/parosol_py/modeling/types.py`: added optional `BuiltModel.nonlinear_material`.
- `src/parosol_py/modeling/workflow_replay.py`: required to build and carry the workflow nonlinear material map from calibrated density.

Concerns:
- Existing unrelated local modification remains in `.superpowers/sdd/keaveny/task-5-6-report.md`; it was not touched for this task.
