# parosol-py

`parosol-py` is the Python package and runtime wrapper for the ParOSol
micro-FE solver. It provides Python helpers for creating solver inputs, running
the bundled native executable, reading outputs, and mapping label or density
images to material stiffness.

This repository keeps the package documentation intentionally small.

The bundled native ParOSol solver was written by Cyril Flaig and is distributed
under the GNU General Public License, version 2 or later. The Python package is
therefore distributed as `GPL-2.0-or-later`.

## Install

Prebuilt wheels include the native ParOSol executable. For local development:

```bash
python -m pip install -e .[dev]
```

Source installs require CMake, an MPI C++ compiler/runtime, HDF5 C++ libraries,
and Eigen headers. Release wheels are built for Python 3.11, 3.12, and 3.13.

## Local Check

Before relying on GitHub Actions, run the local verification gate from an
environment that has the native build tools:

```bash
python scripts/local_check.py
```

Add `--smoke-install` to install the built wheel into a temporary virtual
environment and import it.

## Python API

```python
import numpy as np

from parosol_py import solve

material = np.ones((10, 10, 10), dtype=np.float32) * 1000.0

result = solve(
    material=material,
    spacing=(0.061, 0.061, 0.061),
    material_unit="MPa",
    test="axial",
    test_axis="z",
    strain=-0.01,
    outputs=("sed",),
    export_dir="outputs/example",
)

print(result.summary)
print(result.exported)
```

Use `dry_run=True` to write the ParOSol HDF5 input and command without launching
the solver:

```python
result = solve(
    material=material,
    spacing=(0.061, 0.061, 0.061),
    material_unit="MPa",
    test="axial",
    test_axis="z",
    strain=-0.01,
    dry_run=True,
    export_dir="outputs/dry_run",
)
```

## Material Mapping

Label images can be mapped through an explicit material table:

```python
import numpy as np

from parosol_py import LinearIsotropicMaterials, labels_to_material_map

labels = np.array([[[100, 127]]], dtype=np.uint16)
table = LinearIsotropicMaterials(
    youngs_modulus_mpa={100: 8748.0, 127: 8748.0},
    poisson_ratio={100: 0.3, 127: 0.3},
)

mapped = labels_to_material_map(labels, table)
```

Continuous density images can be converted with one of the supported equations:

```python
from parosol_py import density_to_material_map

mapped = density_to_material_map(
    density_image,
    equation="power",
    coefficient=10000.0,
    exponent=1.7,
    reference_density=1000.0,
    poisson_ratio=0.3,
)
```

The Mulder grayscale BMD law is available as `mulder2007`:

```python
mapped = density_to_material_map(
    density_image,
    equation="mulder2007",
    active_mask=outer_contour,
    floor_e_mpa=2.0,
    poisson_ratio=0.3,
)
```
