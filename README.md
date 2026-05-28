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
