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

## Command Line

Create a case config:

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

failure:
  criterion: pistoia
  critical_volume_percent: 2.0
  critical_strain: 0.007

output:
  summary: outputs/VITD_0003_RL_M06_HOM_LS/summary.json
```

The Pistoia-style failure factor is computed from the strain/SED field for all
load cases. Compression reports a failure force, while shear, bending, and
torsion also report a `failure_generalized_load` whose interpretation follows
the load case, for example force or moment.

Run it:

```bash
parosol run case.yaml
```

Dry-run without launching the solver:

```bash
parosol run case.yaml --dry-run
```

Run a multi-case direct-mechanics batch:

```bash
parosol batch direct_mechanics.yaml
```

A batch config uses the same top-level input/material/solver/output sections as
a normal case, plus a `batch.cases` list. Each case override is expanded into an
individual run directory and summarized in one `batch_summary.json`.

Load-history profiles are available as `load_history_3` and `load_history_6`.
They generate the solved SED fields for the NNLS load-history post-processing
step:

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
