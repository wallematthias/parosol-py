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
  file: material_cort_trab.txt
  units: MPa
  poisson_ratio: 0.3

load_case:
  type: axial
  axis: z
  strain: -0.01

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

Run it:

```bash
parosol run case.yaml
```

Dry-run without launching the solver:

```bash
parosol run case.yaml --dry-run
```

Convert old FAIM text outputs into compact JSON:

```bash
parosol summarize-faim \
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
helpers. FAIM-ish command-line compatibility is planned as a second pass.

## Validation

The test suite includes optional FAIM 10.0 reference checks. When FAIM and a
packaged ParOSol executable are available, a tiny axial-compression cube is
generated with `n88modelgenerator`, solved with FAIM, solved with ParOSol, and
the dense SED fields are compared.
