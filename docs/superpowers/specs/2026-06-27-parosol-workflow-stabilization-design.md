# ParOSol Workflow Stabilization Design

**Date:** 2026-06-27

## Goal

Make the `parosol-py` workflow system water-tight and maintainable by turning the interactive workflow path into the only primary user-facing path, with strong contracts and regression gates around geometry, replay, built-in workflows, and Ogo/n88 parity.

The desired steady state is:

1. Users create workflows interactively in Slicer.
2. Users save those workflows as `.parosol-workflow` bundles.
3. Users replay those workflows headlessly on new samples.
4. Built-in spine, hip, XtremeCTI, XtremeCTII, load-history-3, and load-history-6 workflows are ordinary workflow bundles, not special hidden code paths.
5. `parosol_case.yaml` is the compiled sample-specific case contract.

## Current State

The previous migration and geometry unification work moved the code in the right direction:

- public profiles are backed by packaged `.parosol-workflow` bundles,
- `slicer_editor.planes` are present in built-in workflows,
- `parosol_py.workflow_geometry` exists as a pure-Python geometry module,
- `build_workflow_replay_model(...)` prefers plane-driven geometry when editor planes exist,
- spine and hip workflows use `.npy` reference points rather than `.vtk` reference files,
- public workflow schemas now use `intrusion_depth_mm` rather than `protrusion_depth_mm`.

The remaining problem is not one missing feature. The problem is that the behavior is not protected tightly enough. Tests often assert that some disk or nodeset exists, but not that the exact plane resolution, disk footprint, labels, displacement reference length, load semantics, replay mode, and parity output stayed stable.

## Non-Goals

- Do not add VTK as a runtime dependency to `parosol-py`.
- Do not preserve spine or hip as bespoke modeling code paths.
- Do not remove legacy cached-label workflow fallback before the compatibility tests prove it is safe.
- Do not rewrite the solver or load-history optimizer.
- Do not require full Ogo/n88 artifact compatibility when `results.csv`, `result.json`, `summary.json`, and optional field images preserve the mechanics users need.

## Core Decisions

### 1. `.parosol-workflow` Is The User-Facing Reproducibility Contract

The workflow bundle is the saved recipe. It must contain:

- preprocessing settings,
- material settings,
- registration settings and reference assets when needed,
- canonical `slicer_editor.planes`,
- canonical `slicer_editor.loads`,
- load-case or batch definitions,
- postprocess settings.

Saved `disk_labels.nii.gz` and `nodesets.nii.gz` inside a workflow bundle are derived debug or legacy fallback artifacts. They must not control replay when valid editor planes are present.

### 2. `parosol_case.yaml` Is The Single Compiled Case Contract

`parosol_case.yaml` is the sample-specific resolved contract used by the runner. It may contain absolute paths, generated output paths, compiled `model.workflow_replay` settings, load-case definitions, and execution metadata.

The case file is allowed to differ from the reusable workflow bundle because it is bound to one input sample.

### 3. Planes And Loads Are Canonical

For workflow-based cases:

- planes define contact geometry, disk generation, surface projection, and nodeset regions,
- loads define fixed, displacement, force, bending, torsion, shear, compression, or load-history unit cases tied to nodesets,
- generated disks, nodeset labels, material labels, and boundary conditions are outputs of the workflow engine.

### 4. BBox-Relative Plane Definitions Are Required For Reusable Replay

Reusable workflow planes should be authored relative to `model_bbox`, `active_bbox`, or `image_bbox` using:

- `center_fraction`,
- `size_fraction`,
- `normal_ras`,
- `u_axis_ras`,
- `v_axis_ras`.

Absolute `center_ras` and `size_mm` can remain as editor/reference metadata, but replay must prefer bbox-relative definitions when they exist. This lets the same workflow scale onto larger or smaller samples.

### 5. Public Disk Terminology Uses Thickness And Intrusion

The public schema uses:

- `thickness_mm`: disk thickness in the load-facing fixture direction,
- `intrusion_depth_mm`: how far the generated disk follows or wraps around the contacted anatomy along projection columns.

`intrusion_depth_mm` does not mean the disk physically penetrates the segmented object. Generated disk material still occupies non-bone voxels unless a future workflow explicitly defines another behavior.

`protrusion_depth_mm` must not appear in public workflow bundles, config templates, or documentation.

### 6. Special Workflows Are Recipes

Spine compression, hip sideways fall, XtremeCTI, XtremeCTII, load-history-3, and load-history-6 must be represented as ordinary workflow bundles.

Workflow-specific behavior belongs in:

- bundled workflow YAML,
- reference assets,
- generic registration settings,
- generic geometry settings,
- generic load-case and postprocess machinery.

No new spine-only or hip-only geometry code should be added.

## Architecture

### Geometry Engine

`parosol_py.workflow_geometry` owns pure-Python workflow geometry:

- reference-space and sample-space plane resolution,
- bbox-relative plane resolution,
- plane transform and optional surface snapping,
- projected disk generation,
- intersect surface nodesets,
- project-bounded surface nodesets,
- anatomy, rectangle, oval, and polygonal footprint behavior supported by the workflow schema,
- axis-aligned fast paths where equivalent to general projection,
- disk and nodeset label image generation.

This module must stay independent of Slicer, VTK, MRML objects, and solver execution.

### Replay Pipeline

`parosol_py.modeling.workflow_replay` compiles a workflow into a model:

1. load and preprocess image and mask,
2. choose registration and model masks,
3. pad/crop/resample without losing required fixture space,
4. resolve editor planes in the target replay space,
5. regenerate disk and nodeset labels from planes,
6. derive node sets and boundary conditions,
7. export model artifacts and metadata.

When `slicer_editor.planes` exists, the replay metadata must report `geometry_mode: plane_driven`.

### Workflow Template Loader

`parosol_py.workflow_template` specializes reusable workflow bundles into sample-specific `parosol_case.yaml` files. It resolves input paths, output paths, model artifact paths, and reference paths. It must not silently convert a plane-driven workflow into a cached-label workflow.

### Built-In Workflow Registry

The public profile registry exposes only workflow-backed profiles:

- `XtremeCTI`
- `XtremeCTII`
- `spine-compression`
- `hip-sideways-fall-left`
- `hip-sideways-fall-right`
- `load_history_3`
- `load_history_6`

Legacy names such as `ct-spine-compression`, `ct-hip-sideways-fall`, and `vertebra` stay out of the public registry unless they are deliberately reintroduced as aliases with tests.

## Validation Matrix

### Unit Geometry Tests

Synthetic geometry tests must assert exact behavior, not only non-empty outputs:

- bbox-relative center and size resolution on scaled masks,
- projected disk generation on flat and uneven surfaces,
- anatomy footprint wrapping controlled by `intrusion_depth_mm`,
- load-facing disk face labels are flat where intended,
- intersect mode returns only the plane intersection surface,
- project-bounded mode returns the nearest projected surface,
- generated disks extend beyond the original image when needed,
- generated disks do not occupy bone voxels,
- axis-aligned fast paths match the general ray projection result.

### Workflow Contract Tests

Each packaged workflow must be checked for:

- valid `.parosol-workflow` bundle structure,
- no `.vtk` references in packaged spine or hip workflows,
- required `.npy` reference points where registration is enabled,
- no public `protrusion_depth_mm`,
- expected `intrusion_depth_mm`,
- expected registration settings,
- expected public profile names,
- expected load-case or batch definitions,
- expected solver tolerance,
- expected output fields.

### Replay Tests

Replay tests must assert:

- plane-driven workflows ignore intentionally wrong cached labels,
- replay metadata records `geometry_mode: plane_driven`,
- resolved plane summaries include bbox-relative source metadata,
- node set names and labels match the workflow contract,
- percent displacements use occupied active model length including generated disks,
- model artifact label images match generated geometry,
- cached-label fallback is used only for legacy workflows without valid editor planes.

### Compatibility Tests

XtremeCTI and XtremeCTII need deterministic protection:

- top and bottom nodesets remain surface-intersection driven,
- axial strain direction and magnitude remain stable,
- material label mappings remain stable,
- load directions remain stable.

Load-history workflows need recipe protection:

- load-history-3 contains compression, shear-x, and shear-y unit cases,
- load-history-6 also contains bending-x, bending-y, and torsion unit cases,
- rotational unit cases report moments in `N*mm`,
- final rerun scales nodeset load definitions consistently.

### Ogo/N88 Parity Harness

The parity harness must promote the existing reference bundle into repeatable acceptance checks:

- reference bundle: `/Users/matthias.walle/Downloads/n88_ogo_reference_test_bundle_20260625`,
- spine input: `spine-sub-001/input/density.nii.gz` and `spine-sub-001/input/segmentation.nii.gz`,
- hip input: `hip-sub-RETRO2_10001/input/density.nii.gz` and `hip-sub-RETRO2_10001/input/segmentation.nii.gz`,
- reference assets: `references/L4_BODY_SPINE_COMPRESSION_REF.vtk` and `references/LT_FEMUR_SIDEWAYS_FALL_REF.vtk`.

Acceptance comparisons should include:

- applied displacement,
- reaction force,
- stiffness,
- failure load where configured,
- generated model dimensions,
- disk and nodeset label counts,
- optional SED field comparison when field output is enabled and grids are comparable.

The parity harness should store expected values and tolerances in versioned data, while large generated fields remain local artifacts.

## Implementation Sequence

### Phase 1: Baseline Freeze

Create a stabilization branch and record the current behavior before changing implementation. Capture:

- public workflow list,
- workflow bundle schema snapshots,
- representative `parosol_case.yaml` files,
- model artifact summaries,
- `results.csv` values for XtremeCTI, XtremeCTII, spine, hip, load-history-3, and load-history-6 dry-run or smoke runs.

The baseline should be explicit about which values are accepted and which values are known failures.

### Phase 2: Contract Validator

Add a workflow contract validator used by tests and optional CLI diagnostics. It should validate public workflow bundles and produce clear errors for schema drift.

### Phase 3: Geometry Ratchet

Strengthen `tests/test_workflow_geometry.py` and replay geometry tests until the disk and nodeset behavior is exact enough to catch drift.

### Phase 4: Replay Ratchet

Strengthen replay tests so the case compiler and model builder prove that:

- planes win over cached labels,
- bbox-relative definitions are preserved in metadata,
- artifact labels match generated geometry,
- percent displacement uses the intended reference length.

### Phase 5: Workflow Family Protection

Lock the built-in workflows in this order:

1. XtremeCTI and XtremeCTII,
2. load-history-3 and load-history-6,
3. spine-compression,
4. hip-sideways-fall-left and hip-sideways-fall-right.

This order protects previously validated workflows before revisiting the more delicate Ogo/n88 parity work.

### Phase 6: Parity Harness

Create a script or pytest marker for Ogo/n88 parity runs. The default unit test suite can skip large solves unless the reference bundle is present and an explicit environment flag is set.

The harness should emit compact CSV/JSON comparison reports and should not require committing large generated fields.

### Phase 7: Cleanup

After contract and parity gates pass, remove or quarantine legacy paths that can silently bypass plane-driven replay. Keep compatibility fallback only where a test proves it is needed.

## Acceptance Gates

Work is complete when:

- public profiles are workflow-backed and validated,
- no packaged public workflow uses `protrusion_depth_mm`,
- valid editor-plane workflows always replay with `geometry_mode: plane_driven`,
- cached labels cannot override editor-plane replay,
- bbox-relative planes scale correctly across synthetic sample sizes,
- XtremeCTI and XtremeCTII remain deterministic,
- load-history-3 and load-history-6 retain their validated unit-case semantics,
- spine and hip parity reports are repeatable and stored with tolerances,
- generated model artifacts are exported as derived outputs,
- `parosol_case.yaml` is the only sample-specific case contract.

## Risks And Mitigations

### Risk: Geometry drift during test strengthening

Mitigation: capture baseline outputs first and classify differences as accepted, rejected, or unresolved before changing implementation.

### Risk: Parity targets are not directly comparable

Mitigation: compare applied displacement, reaction force, stiffness, and failure load first. Add SED comparisons only after grid alignment and masking are explicit.

### Risk: Cached-label fallback hides replay bugs

Mitigation: add tests with intentionally wrong cached labels and require plane-driven outputs to ignore them.

### Risk: BBox-relative planes conflict with registered reference-space planes

Mitigation: define precedence clearly. If a plane has `relative_to` with valid fractions, replay resolves that plane from the selected sample/model bbox. Absolute reference-space pose remains metadata unless no relative definition exists.

### Risk: Dirty working state makes baseline unclear

Mitigation: start stabilization by committing or shelving unrelated work, then capture baseline outputs from a named commit. The baseline report must record the exact git SHA and command set.

