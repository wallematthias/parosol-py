# FEA API and Profiles Design

## Goal

Transform `parosol-py` from a thin ParOSol wrapper into a modular Python FEA API
with friendly command-line profiles. The core package should stay general and
testable, while bone-specific helpers and legacy solver-compatible workflows are layered
on top.

The main user workflows are:

- Run standard 1% axial compression on segmented bone images.
- Match legacy solver stiffness, reaction force, Pistoia factor, and failure load for
  legacy cases.
- Export useful standard outputs, including compressed `.nii.gz` field images
  and compact summary JSON.
- Reuse solver/load/output profiles across large batches.
- Leave room for harder boundary conditions, such as curved top/bottom plates,
  body-weight force loading, shear, and bending.

## Design Principles

- Keep the core API solver-oriented, not bone-specific.
- Represent boundary conditions as explicit data objects that can be inspected,
  serialized, tested, and eventually edited by a GUI such as 3D Slicer.
- Make bone helpers generate core boundary-condition objects rather than writing
  solver files directly.
- Keep the CLI as a profile/config adapter over the Python API.
- Treat legacy solver outputs as validation references, not as the internal data model.
- Make field image export a standard output, but avoid unnecessary dense
  reconstruction when a profile only asks for summary metrics.

## Layered Architecture

### Core FEA Layer

The core layer should know about images, material properties, boundary
conditions, solver execution, and results. It should not know about femurs,
tibias, endplates, or body-weight assumptions.

Core objects:

- `Model`: material/stiffness image, spacing, origin, axis order, and optional
  mask metadata.
- `MaterialMap`: label-to-material mapping, linear isotropic properties, and
  unit conversion.
- `BoundaryConditionSet`: fixed displacement constraints and nodal/face force
  loads in a single serializable object.
- `LoadCase`: a generator that maps a `Model` to a `BoundaryConditionSet`.
- `SolverProfile`: solver settings such as tolerance, MPI process count, level,
  executable, and postprocessing outputs.
- `OutputProfile`: summary fields, image fields, compression, and export paths.
- `Result`: structured solver metrics, mechanics, failure metrics, field paths,
  and optional in-memory arrays.

### Bone Helper Layer

The bone helper layer provides convenience constructors for common biomedical
FEA use cases. These helpers consume segmentations, masks, landmarks, or surface
definitions and emit core `LoadCase` or `BoundaryConditionSet` objects.

Initial helpers:

- `AxialCompression`: fixed bottom surface, prescribed top displacement.
- `BodyWeightCompression`: fixed/stabilized support plus distributed top or
  surface force derived from body weight and a multiplier.
- `SimpleShear`: top surface displacement in a lateral direction with bottom
  support.
- `Bending`: linear displacement gradient or equivalent distributed nodal loads.
- `SurfaceProjectedCompression`: project constraints or loads onto segmented or
  user-defined curved surface regions.

### CLI/Profile Layer

The CLI should load a case config and profile files, instantiate the same Python
objects, and run the same core API. Inexperienced users should be able to select
profiles such as `legacy_axial`, `standard_fields`, or `batch_summary` without
knowing every solver option.

Example config direction:

```yaml
case:
  name: CABHS_5001_RL_V1_HOM_LS

model:
  image: CABHS_5001_RL_V1_HOM_LS.AIM
  image_type: material_labels
  spacing: auto

materials:
  file: material_cort_trab.txt

load_case:
  preset: axial_compression
  axis: z
  strain: -0.01
  surfaces: auto_extrema

solver_profile: legacy_axial
output_profile: standard_fields
```

## Boundary Conditions

Boundary conditions are the hardest part and should be a first-class API, not
hidden inside `solve()`.

`BoundaryConditionSet` should support:

- fixed nodal displacements;
- nodal forces, matching native ParOSol `Loaded_Nodes`;
- face or surface force helpers that distribute a total force over nodes;
- fixed, free, or weakly constrained lateral directions;
- metadata for named node sets such as `top`, `bottom`, `support`, and
  `loaded_surface`;
- JSON/YAML serialization for reuse in batch runs and future Slicer integration.

The current native ParOSol code supports fixed displacements and loaded nodes.
The current Python writer exposes only fixed displacements. The next API slice
should expose `Loaded_Nodes` in `hdf5_io.py` and tests should validate that force
loads are written in the expected native coordinate order.

## Slicer Integration Path

Slicer should eventually be a boundary-condition editor and visual QA tool, not
a required runtime dependency.

The exchange format should be plain files:

- image or material volume: `.aim`, `.nii.gz`, or `.mha`;
- optional surface/plate labels: `.nii.gz`;
- landmarks and axes: `.json`;
- boundary-condition template: `.json` or `.yaml`;
- solver/output profiles: `.yaml`.

A user could visually define top/bottom plates or surface regions once in
Slicer, export the BC template, then run the same template headlessly over a
batch of similar bones.

## Standard Profiles

Profiles should be named bundles, with all fields overridable by a case config.

- `legacy_axial`: axial compression defaults, legacy-compatible material parsing, Pistoia
  JSON metrics, tolerance `1e-6`, standard legacy solver validation fields.
- `quick_summary`: SED and force/displacement fields only, no dense image export,
  summary JSON only.
- `standard_fields`: summary JSON plus compressed `.nii.gz` for SED and selected
  strain/stress fields.
- `batch`: MPI/process defaults, deterministic output names, minimal logs, and
  no interactive assumptions.
- `debug`: richer logs, HDF5 retention, optional `.mha` outputs, and extra
  solver fields.

## Standard Outputs

Every complete run should be able to produce:

- `summary.json` with solver metrics, image/model metadata, mechanics, Pistoia
  values, field statistics, output paths, and profile information;
- selected compressed `.nii.gz` fields, especially SED;
- retained ParOSol HDF5 input/output when requested;
- captured solver stdout/stderr logs for reproducibility.

The summary JSON should be generated directly from ParOSol results. Old legacy solver
text parsing remains useful only for validation and migration.

## Fast Field Reconstruction

Field export should stay standard, but it must not make summary-only runs slow.

Implementation direction:

- Separate native result reading from dense image reconstruction.
- Only reconstruct/export fields requested by `OutputProfile`.
- Vectorize native-to-dense mapping and avoid Python per-element loops.
- Cache coordinate order/mapping for multiple fields from the same model.
- Support compressed `.nii.gz` as the default external image format.
- Keep `.mha` available for debugging and comparison.

The old `BoneMechanoregulation` fast extractor used vectorized element-center
mapping for `.n88model` fields. `parosol-py` should use the equivalent idea for
ParOSol HDF5 native element order.

## legacy solver Validation Matrix

The current local validation folder contains five real legacy solver reference cases:

- `VITD_0003_RL_M06_HOM_LS`
- `CABHS_5001_RL_V1_HOM_LS`
- `CABHS_5001_TL_V1_HOM_LS`
- `CABHS_5002_RL_V1_HOM_LS`
- `CABHS_5002_TL_V1_HOM_LS`

All use axial z compression at `-0.01` strain. CABHS cases use Pistoia critical
volume `2.0%`; VITD uses `7.5%`. These should become a local validation command
or pytest-marked integration test that compares:

- reaction force along z;
- axial stiffness along z;
- EES at critical volume;
- Pistoia factor;
- failure load along z;
- SED field orientation and summary statistics where practical.

The validation command should emit a compact comparison table and JSON report.

## Packaging Direction

The package should remain pip-installable with modern wheels as the release
goal. Local conda environments are useful for development, but should not be the
user-facing installation story.

Packaging targets:

- source build for developers with CMake, MPI, HDF5, and Eigen;
- macOS and Linux wheels first;
- Windows evaluated after the native dependency story is clear;
- bundled native ParOSol executable discoverable by the Python package;
- CI wheel builds once local native build flags are stable.

MPI is the current scalability path. Thread-style ergonomics can be exposed as a
profile option, but internally it should map to `mpirun -np N` unless native
threading is added.

## Implementation Slices

1. Stabilize current compatibility work:
   - keep MPI launch support;
   - keep physical displacement scaling;
   - keep summary-only output profiles;
   - finish a successful legacy solver comparison run on at least one sample case.

2. Introduce explicit core data objects:
   - `Model`;
   - `BoundaryConditionSet`;
   - `SolverProfile`;
   - `OutputProfile`;
   - `Result`.

3. Add force boundary condition support:
   - write `Loaded_Nodes` datasets;
   - add nodal force tests;
   - add face force distribution helper.

4. Refactor load cases:
   - implement axial compression as a `LoadCase`;
   - add simple shear;
   - define bending and curved/projection APIs, even if advanced helpers arrive
     later.

5. Optimize field export:
   - move native-to-dense reconstruction into a dedicated module;
   - vectorize and cache coordinate mappings;
   - benchmark against current export on the sample cases.

6. Add profiles and validation CLI:
   - profile files or built-in profile registry;
   - `parosol validate-reference` over the five local references;
   - JSON and table outputs.

## Acceptance Criteria

- Existing API and CLI behavior remain usable.
- The core API can run axial compression through explicit `Model`,
  `LoadCase`, and profile objects.
- The HDF5 writer supports both fixed displacements and nodal force loads.
- `.nii.gz` export remains a standard output profile and is measurably faster
  than the current per-field reconstruction path.
- Summary-only runs do not reconstruct dense images.
- At least one full local sample case completes and reports comparison against
  legacy solver pistoia values.
- The five local legacy solver cases are represented in a validation manifest.
- The CLI can run a named profile without requiring inexperienced users to
  understand the lower-level API.

