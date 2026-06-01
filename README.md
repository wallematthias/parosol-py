# parosol-py

Standalone Python package for running the ParOSol micro-FE solver from Python.

## Install

```bash
pip install -e .
```

Native build requirements:

- CMake
- MPI C++ compiler/runtime
- HDF5 C++ libraries
- Eigen headers

If editable installation fails during CMake configuration, check that `mpicxx`
or an equivalent MPI C++ compiler wrapper is available on `PATH`.

## Python API

```python
import numpy as np
from parosol_py import solve

material = np.ones((10, 10, 10), dtype=float) * 1000.0  # z, y, x; MPa

result = solve(
    material=material,
    spacing=(0.061, 0.061, 0.061),
    material_unit="MPa",
    test="axial",
    test_axis="z",
    strain=-0.01,
    outputs=("sed",),
    export_dir="outputs",
)

print(result.summary)
print(result.exported)
```

Use `dry_run=True` to write the ParOSol HDF5 input and inspect the generated
command without launching the solver.

Material helpers are available for both label images and continuous density
images:

```python
from parosol_py import density_to_material_map

mapped = density_to_material_map(
    density_image,
    equation="power",
    coefficient=10000,
    exponent=1.7,
    reference_density=1000,
    poisson_ratio=0.3,
)
```

Label-image material tables can define different Poisson ratios per material.
ParOSol-py writes those as an optional native per-element Poisson ratio image.
For continuous density inputs, `materials.poisson_ratio` equations are currently
reduced to one scalar value before solve.

## Command Line

The primary command is profile-driven:

```bash
parosol distal-radius.AIM --profile XtremeCTII --output outputs/distal-radius
```

If `--output` is omitted, ParOSol-py writes to a sibling directory named
`<input>_parosol`. Every run writes the generated `parosol_case.yaml`,
`summary.json`, field images when enabled, and an overview PNG. The summary
contains an `execution` section with the resolved input paths, profile, generated
config path, output directory, dry-run flag, and exact shortcut command.

Model profiles use the same command and accept a standard `--mask` argument:

```bash
parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile vertebra \
  --output outputs/10001_vertebra

parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile proximal_femur_sideways_fall \
  --side left \
  --output outputs/10001_left_femur_fall
```

Use `--dry-run` to write the generated model, solver input, overview, and JSON
without launching the native solver:

```bash
parosol distal-radius.AIM --profile XtremeCTII --output outputs/distal-radius --dry-run
```

Advanced users can still keep the generated YAML, edit any section, and run it
directly:

```yaml
case:
  name: VITD_0003_RL_M06_HOM_LS
  work_dir: outputs/VITD_0003_RL_M06_HOM_LS

input:
  image: VITD_0003_RL_M06_HOM_LS.AIM
  image_type: material_labels
  spacing: [0.061, 0.061, 0.061]

materials:
  units: MPa
  definitions:
    TrabecularBone:
      Type: LinearIsotropic
      E: 8748
      nu: 0.3
    CorticalBone:
      Type: LinearIsotropic
      E: 8748
      nu: 0.3
  table:
    100: TrabecularBone
    127: CorticalBone

load_case:
  type: constrained_axial
  axis: z
  strain: -0.01
  surface:
    mode: smart
    depth: auto

solver:
  tolerance: 1e-6
  level: 6
  outputs: [sed]

postprocess:
  pistoia:
    criterion: pistoia
    critical_volume_percent: 2.0
    critical_strain: 0.007

output:
  summary: outputs/VITD_0003_RL_M06_HOM_LS/summary.json
  fields: [sed]
  export_fields: true
  visualize: true
```

The Pistoia-style failure factor is computed from the strain/SED field for all
load cases. Compression reports a failure force, while shear, bending, and
torsion also report a `failure_generalized_load` whose interpretation follows
the load case, for example force or moment.

Config-driven runs write a professional overview PNG by default. It shows
axial, sagittal, and coronal mid-slices of the material image, the selected
result field on the second row when available, and boundary-condition markers
and direction arrows for quick debugging.

Run an explicit config:

```bash
parosol run case.yaml
```

Run a multi-case direct-mechanics batch:

```bash
parosol batch direct_mechanics.yaml
```

A batch config uses the same top-level input/material/solver/output sections as
a normal case, plus a `batch.cases` list. Each case override is expanded into an
individual run directory and summarized in one `batch_summary.json`.

Scanner/load-case profiles are available as `XtremeCTI` and `XtremeCTII`. Each
profile defines the standard binary bone material table, constrained z-axis
compression at 1% strain, SED output, and Pistoia post-processing defaults.

Load-history profiles are available as `load_history_3` and `load_history_6`.
They declare a `postprocess.load_history` block and generate the solved SED
fields for the NNLS load-history post-processing step:

```bash
parosol load-history compression_sed.nii.gz shear_x_sed.nii.gz shear_y_sed.nii.gz \
  --bone-mask bone_mask.nii.gz \
  --summary load_history_summary.json \
  -o load_history.nii.gz
```

Useful output controls:

```yaml
output:
  fields: [sed, effective_strain, von_mises]
  export_fields: true
  export_sets: true
  set_formats: [json, vtk]

solver:
  max_relative_residual: 1.0e-6
```

`output.fields` selects image-field outputs. `output.export_sets` writes node-set
and material element-set debug files for Slicer/ParaView inspection.

Convert old legacy solver text outputs into compact JSON:

```bash
parosol summarize-legacy \
  --analysis VITD_0003_RL_M06_HOM_LS_analysis.txt \
  --pistoia VITD_0003_RL_M06_HOM_LS_pistoia.txt \
  -o VITD_0003_RL_M06_HOM_LS_summary.json
```

## AIM Input

```python
from parosol_py import solve_aim

result = solve_aim("segmented.aim", outputs=("sed",), export_dir="outputs")
```

AIM IO is provided by `aimio-py` through `py_aimio`.

## Scope

This first pass provides the clean Python API, HDF5 input writing, solver
command construction, result reading, `.nii.gz` scalar export, and validation
helpers. Legacy-compatible command-line compatibility is planned as a second pass.

## GPU Backends

The native ParOSol C++/MPI solver remains the validated reference backend.
Accelerator work lives in a separate optional `parosol_torch` namespace rather
than inside `src/parosol_native`. This package currently exposes capability
checks only; it does not yet provide a validated numerical solver.

```python
from parosol_torch import backend_info

print(backend_info())
```

See `docs/gpu-backend-roadmap.md` for the implementation and validation plan.

## Validation

The test suite includes a connected-component-filtered TRAB_1240 reference fixture.
By default, tests validate the compressed fixture metadata without running the solver.
Set `PAROSOL_RUN_REFERENCE_TESTS=1` to run the ParOSol axial-compression regression
and compare stiffness, reaction force, and Pistoia failure load against the fixed
reference JSON.

See `docs/legacy-feature-map.md` for the compatibility roadmap and the cleaner
ParOSol-py equivalents of useful legacy workflow features.
See `docs/n88-reference-verification.md` for the current local reference
verification results for standard load cases.
