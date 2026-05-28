# parosol-py Design

## Goal

Create `parosol-py`, a standalone, pip-installable Python package that extracts ParOSol from the old `framework-main` repository and makes it usable as a modern finite-element solver library. The first pass focuses on a clean Python API and reproducible FAIM-vs-ParOSol validation. A FAIM-like command-line compatibility layer is a second pass built on top of the validated API.

The migration should let existing workflows move from:

```text
input image + material.txt -> FAIM 10.0 -> .txt + .n88model outputs
```

to:

```text
input image or material array -> parosol-py -> structured Python results + .nii.gz image outputs
```

`aimio-py` owns AIM input/output. `parosol-py` owns FE preparation, ParOSol execution, result extraction, image export, and validation against FAIM.

## Scope

### First Pass

- Package ParOSol as `parosol-py`, with import name `parosol_py`.
- Provide a clean Python API for solving from NumPy arrays and, through `aimio-py`, AIM files.
- Generate ParOSol HDF5 input from material arrays, spacing, Poisson ratio, boundary conditions, and requested outputs.
- Run ParOSol locally and capture logs, timings, iterations, and residuals.
- Read ParOSol output fields into structured Python objects.
- Export scalar and tensor-derived fields as `.nii.gz`, with `.mha` allowed as a debugging or comparison format.
- Build synthetic FAIM golden fixtures from small examples in `BoneMechanoregulation`.
- Compare ParOSol outputs against FAIM outputs using explicit numeric tolerances.

### Second Pass

- Add FAIM-ish command-line entry points using the Python API.
- Support existing FAIM-style options where they map cleanly, such as test type, test axis, strain, material definitions, convergence tolerance, maximum iterations, and output fields.
- Keep the CLI as a thin adapter so FAIM naming and `.n88model` assumptions do not leak into the core API.

## Non-Goals

- Reimplement all of FAIM 10.0 in the first pass.
- Preserve `.n88model` as the primary output format.
- Copy old `ifb_framework` AIM IO into the new package.
- Build a GUI.
- Support every FAIM material model immediately. The first pass targets linear isotropic material definitions needed by the synthetic validation fixtures.

## Package Boundaries

### External Dependencies

- `aimio-py`: read AIM files and AIM metadata through `py_aimio`.
- `numpy`: array representation and numeric tests.
- `h5py`: ParOSol HDF5 input/output.
- `SimpleITK`: `.nii.gz` and `.mha` export with spacing/origin metadata. This matches the existing `BoneMechanoregulation` experiments and keeps one image IO path for validation fixtures.
- CMake, MPI, HDF5, and Eigen-compatible headers/libraries for building ParOSol.

### Internal Package Modules

```text
parosol_py/
  __init__.py
  api.py
  images.py
  materials.py
  boundary_conditions.py
  hdf5_io.py
  runner.py
  results.py
  validation.py
  cli.py                # second pass
```

Module responsibilities:

- `api.py`: public `solve`, `solve_aim`, and dataclasses.
- `images.py`: axis-order normalization, spacing/origin handling, and image export.
- `materials.py`: material array to ParOSol stiffness image conversion; material table parsing for FAIM-style material files.
- `boundary_conditions.py`: standard axial compression and simple shear condition generation.
- `hdf5_io.py`: ParOSol HDF5 schema read/write.
- `runner.py`: executable discovery, subprocess execution, MPI/local run configuration, and log capture.
- `results.py`: field decoding, metadata, tensor field layout, and output formatting.
- `validation.py`: helpers for FAIM golden fixture comparison.
- `cli.py`: second-pass FAIM-ish command-line adapter.

## Public API

The first-pass API should be small and explicit:

```python
from parosol_py import solve, solve_aim

result = solve(
    material=material_mpa,
    spacing=(0.061, 0.061, 0.061),
    origin=(0.0, 0.0, 0.0),
    material_unit="MPa",
    poisson_ratio=0.3,
    test="axial",
    test_axis="z",
    strain=-0.01,
    outputs=("sed", "strain", "stress", "forces"),
)
```

`solve` accepts NumPy arrays first. The default array order is `array_order="zyx"` because this matches `SimpleITK.GetArrayFromImage`, `aimio-py` arrays, and common NumPy image conventions. Internally, geometry-facing code may normalize to `(x, y, z)`, but every public result must declare and preserve the requested output order.

`solve_aim` delegates AIM reading to `py_aimio`:

```python
result = solve_aim(
    "segmented.aim",
    material_definitions="material.txt",
    test="axial",
    test_axis="z",
    outputs=("sed",),
)
```

The returned result should include:

- `fields`: requested fields as NumPy arrays with declared axis order.
- `summary`: iterations, residuals, solve time, dimensions, spacing, and material summary.
- `paths`: optional exported files.
- `log`: captured ParOSol stdout/stderr.

## Data Flow

1. Normalize input image or material array into an internal lattice representation.
2. Convert material units to ParOSol stiffness image units. ParOSol expects Young's modulus in GPa.
3. Generate boundary condition datasets for the requested test.
4. Write a temporary or user-specified ParOSol HDF5 input file.
5. Run the packaged ParOSol executable.
6. Read output datasets from the same HDF5 file.
7. Decode fields into the requested axis order.
8. Export requested fields to `.nii.gz` and optionally `.mha`.
9. Return a structured result object.

## FAIM Validation Strategy

Use `BoneMechanoregulation` as the source of small synthetic FAIM cases. The first fixture set should include:

- A homogeneous cube under axial compression.
- An asymmetric block, such as `(9, 7, 5)`, to catch axis-order errors.
- A two-material block to validate material mapping.
- An H-shape or disconnected-component case to validate preprocessing and geometry handling.
- One optional AIM-derived tiny case read through `aimio-py`.

For each fixture:

1. Generate the input material or label image deterministically in Python.
2. Run FAIM 10.0 once to produce golden outputs.
3. Store compact golden data as `.nii.gz` and JSON summaries.
4. Run `parosol-py` on the same case.
5. Compare fields and summaries with explicit tolerances.

The validation should focus on equivalence at the workflow level: field shape, orientation, material assignment, boundary-condition direction, force balance, and numeric field agreement. Exact bitwise equality is not expected unless a fixture proves it is realistic.

## Packaging Design

`parosol-py` should use modern Python packaging:

- `pyproject.toml` with a PEP 517 build backend.
- CMake-based build for the ParOSol executable.
- Wheel builds for supported platforms where practical.
- The ParOSol executable installed as package data or an entry-point-accessible binary.
- Tests runnable with `pytest`.

The package should use `scikit-build-core` for the CMake build. A setuptools fallback is out of scope for the first pass unless `scikit-build-core` proves unable to install the ParOSol executable cleanly.

## CLI Pass Two

The second-pass CLI should wrap the API, not duplicate solver logic.

Example shape:

```bash
parosol-solve input.aim \
  --material_definitions material.txt \
  --test axial \
  --test_axis z \
  --normal_strain -0.01 \
  --outputs sed,strain,stress \
  --output_dir out
```

Compatibility aliases can be added after the API validation is stable. The CLI should emit `.nii.gz` and JSON summaries by default. It may optionally emit FAIM-like text reports, but those reports are not the core data model.

## Risks

- ParOSol and FAIM may not be numerically identical for some fields because solver formulations, convergence behavior, or post-processing definitions differ.
- ParOSol's HDF5 schema and output ordering must be handled carefully to avoid silent axis flips.
- Building MPI/HDF5-backed C++ code in wheels may be harder than packaging pure Python.
- FAIM's material definition grammar may be broader than the first-pass parser.
- `.nii.gz` affine/orientation choices must be explicit so outputs remain inspectable and comparable.

## Acceptance Criteria

- `pip install -e .` builds and installs `parosol-py` locally.
- `import parosol_py` works.
- A small NumPy cube can be solved through `parosol_py.solve`.
- Requested fields can be returned as NumPy arrays and exported to `.nii.gz`.
- Synthetic validation tests compare ParOSol against FAIM golden outputs for at least cube compression, asymmetric geometry, and two-material geometry.
- The public API does not depend on `ifb_framework`.
- AIM input uses `py_aimio`, not copied AIM reader code.
- FAIM-ish CLI work is documented as pass two and does not block the first-pass API package.
