# parosol-py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-pass `parosol-py` package: installable ParOSol executable, clean Python API, HDF5 IO, image export, and synthetic FAIM-oriented validation scaffolding.

**Architecture:** Package ParOSol as a native CMake executable inside a modern `scikit-build-core` Python package, then wrap it with focused Python modules. Python normalizes image/material inputs, writes ParOSol HDF5 input, runs the solver, reads `/Solution` datasets, exports image fields, and provides validation helpers. FAIM-ish CLI compatibility remains out of scope for this first implementation plan.

**Tech Stack:** Python 3.10+, NumPy, h5py, SimpleITK, pytest, scikit-build-core, CMake, MPI, HDF5, Eigen, ParOSol C++ source from `framework-main/src/parOsol`.

---

## File Structure

Create this package structure:

```text
/Users/matthias.walle/Documents/fea/
  pyproject.toml
  CMakeLists.txt
  README.md
  src/
    parosol_py/
      __init__.py
      api.py
      boundary_conditions.py
      hdf5_io.py
      images.py
      materials.py
      results.py
      runner.py
      validation.py
      _version.py
    parosol_native/
      CMakeLists.txt
      LICENSE
      src/
        ... copied ParOSol C++ sources ...
  tests/
    test_boundary_conditions.py
    test_hdf5_io.py
    test_images.py
    test_materials.py
    test_results.py
    test_runner.py
    test_api.py
    test_validation.py
```

Keep `framework-main/` untracked exploration material. Copy only `framework-main/src/parOsol` into `src/parosol_native` when implementing Task 1.

## Task 1: Package Scaffold and Native Source Copy

**Files:**
- Create: `/Users/matthias.walle/Documents/fea/pyproject.toml`
- Create: `/Users/matthias.walle/Documents/fea/CMakeLists.txt`
- Create: `/Users/matthias.walle/Documents/fea/README.md`
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/__init__.py`
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/_version.py`
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_native/CMakeLists.txt`
- Copy: `/Users/matthias.walle/Documents/fea/framework-main/src/parOsol/LICENSE` to `/Users/matthias.walle/Documents/fea/src/parosol_native/LICENSE`
- Copy: `/Users/matthias.walle/Documents/fea/framework-main/src/parOsol/src/*` to `/Users/matthias.walle/Documents/fea/src/parosol_native/src/`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_api.py`

- [ ] **Step 1: Copy the ParOSol native source**

Run:

```bash
mkdir -p /Users/matthias.walle/Documents/fea/src/parosol_native
cp -R /Users/matthias.walle/Documents/fea/framework-main/src/parOsol/src /Users/matthias.walle/Documents/fea/src/parosol_native/src
cp /Users/matthias.walle/Documents/fea/framework-main/src/parOsol/LICENSE /Users/matthias.walle/Documents/fea/src/parosol_native/LICENSE
```

Expected: `/Users/matthias.walle/Documents/fea/src/parosol_native/src/main.cpp` exists.

- [ ] **Step 2: Create packaging files**

Create `/Users/matthias.walle/Documents/fea/pyproject.toml`:

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11>=2.12"]
build-backend = "scikit_build_core.build"

[project]
name = "parosol-py"
version = "0.1.0"
description = "Python package and API wrapper for the ParOSol micro-FE solver"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "GPL-2.0-or-later" }
authors = [{ name = "Matthias Walle" }]
dependencies = [
  "numpy>=1.24",
  "h5py>=3.10",
  "SimpleITK>=2.3",
  "aimio-py>=0.1.1",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-cov>=5",
  "build>=1",
]

[tool.scikit-build]
cmake.version = ">=3.18"
wheel.packages = ["src/parosol_py"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Create `/Users/matthias.walle/Documents/fea/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(parosol_py LANGUAGES CXX)

add_subdirectory(src/parosol_native)

install(TARGETS parosol
        RUNTIME DESTINATION parosol_py/bin)
```

Create `/Users/matthias.walle/Documents/fea/src/parosol_native/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)

set(PAROSOL_SOURCES
    src/GReader.cpp
    src/GWriter.cpp
    src/JacobiSmoother.cpp
    src/Toolbox.cpp
    src/AsciiImage.cpp
    src/HDF5Image.cpp
    src/AsciiImageMirrored.cpp
    src/fem.cpp
    src/est_ev.cpp
    src/Chebyshev.cpp
    src/PCGSolver.cpp
    src/main.cpp)

add_executable(parosol ${PAROSOL_SOURCES})
target_compile_features(parosol PRIVATE cxx_std_11)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(HDF5 REQUIRED COMPONENTS CXX)

target_include_directories(parosol PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${HDF5_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(parosol PRIVATE
    ${HDF5_LIBRARIES}
    MPI::MPI_CXX)
```

Create `/Users/matthias.walle/Documents/fea/README.md`:

```markdown
# parosol-py

Standalone Python package for running the ParOSol micro-FE solver from Python.

First-pass scope:

- NumPy and AIM inputs
- ParOSol HDF5 input/output
- Local solver execution
- NumPy and `.nii.gz` field outputs
- Synthetic validation against FAIM fixtures
```

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/_version.py`:

```python
__version__ = "0.1.0"
```

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/__init__.py`:

```python
from ._version import __version__

__all__ = ["__version__"]
```

- [ ] **Step 3: Write import smoke test**

Create `/Users/matthias.walle/Documents/fea/tests/test_api.py`:

```python
import parosol_py


def test_package_imports():
    assert parosol_py.__version__ == "0.1.0"
```

- [ ] **Step 4: Run import smoke test**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_api.py -v
```

Expected: PASS.

- [ ] **Step 5: Try editable install**

Run:

```bash
python -m pip install -e /Users/matthias.walle/Documents/fea
```

Expected: package installs. If CMake cannot find MPI or HDF5, record the missing dependency in the final task notes and continue with pure-Python tests by setting the build dependency issue aside.

- [ ] **Step 6: Commit scaffold**

```bash
git add pyproject.toml CMakeLists.txt README.md src/parosol_py src/parosol_native tests/test_api.py
git commit -m "feat: scaffold parosol-py package"
```

## Task 2: Image Axis Normalization and Export

**Files:**
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/images.py`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_images.py`

- [ ] **Step 1: Write failing image tests**

Create `/Users/matthias.walle/Documents/fea/tests/test_images.py`:

```python
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from parosol_py.images import ImageGrid, export_scalar_image, normalize_array


def test_normalize_array_accepts_zyx_default():
    arr_zyx = np.zeros((2, 3, 4), dtype=np.float32)
    arr_zyx[1, 2, 3] = 9.0

    grid = normalize_array(arr_zyx, spacing=(0.1, 0.2, 0.3), origin=(1.0, 2.0, 3.0))

    assert grid.array_xyz.shape == (4, 3, 2)
    assert grid.array_xyz[3, 2, 1] == 9.0
    assert grid.spacing == (0.1, 0.2, 0.3)
    assert grid.origin == (1.0, 2.0, 3.0)


def test_normalize_array_accepts_xyz():
    arr_xyz = np.zeros((4, 3, 2), dtype=np.float32)
    arr_xyz[3, 2, 1] = 11.0

    grid = normalize_array(
        arr_xyz,
        spacing=(0.1, 0.2, 0.3),
        origin=(0.0, 0.0, 0.0),
        array_order="xyz",
    )

    assert grid.array_xyz.shape == (4, 3, 2)
    assert grid.array_xyz[3, 2, 1] == 11.0


def test_export_scalar_image_roundtrips_nii_gz(tmp_path: Path):
    arr_xyz = np.zeros((4, 3, 2), dtype=np.float32)
    arr_xyz[3, 2, 1] = 7.0
    grid = ImageGrid(array_xyz=arr_xyz, spacing=(0.1, 0.2, 0.3), origin=(1.0, 2.0, 3.0))
    out = tmp_path / "sed.nii.gz"

    export_scalar_image(grid, out)

    img = sitk.ReadImage(str(out))
    arr_zyx = sitk.GetArrayFromImage(img)
    assert arr_zyx.shape == (2, 3, 4)
    assert arr_zyx[1, 2, 3] == 7.0
    assert tuple(round(v, 6) for v in img.GetSpacing()) == (0.1, 0.2, 0.3)
    assert tuple(round(v, 6) for v in img.GetOrigin()) == (1.0, 2.0, 3.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_images.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing `parosol_py.images`.

- [ ] **Step 3: Implement image module**

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/images.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class ImageGrid:
    array_xyz: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]


def _triple(values: tuple[float, float, float] | list[float] | np.ndarray, name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return tuple(float(v) for v in values)


def normalize_array(
    array,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    array_order: str = "zyx",
) -> ImageGrid:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"array must be 3D, got shape {arr.shape}")
    order = array_order.strip().lower()
    if order == "zyx":
        arr_xyz = np.transpose(arr, (2, 1, 0))
    elif order == "xyz":
        arr_xyz = arr
    else:
        raise ValueError("array_order must be 'zyx' or 'xyz'")
    return ImageGrid(
        array_xyz=np.ascontiguousarray(arr_xyz),
        spacing=_triple(spacing, "spacing"),
        origin=_triple(origin, "origin"),
    )


def to_output_order(array_xyz: np.ndarray, *, array_order: str = "zyx") -> np.ndarray:
    order = array_order.strip().lower()
    if order == "zyx":
        return np.ascontiguousarray(np.transpose(array_xyz, (2, 1, 0)))
    if order == "xyz":
        return np.ascontiguousarray(array_xyz)
    raise ValueError("array_order must be 'zyx' or 'xyz'")


def export_scalar_image(grid: ImageGrid, output_path: str | Path) -> Path:
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    arr_zyx = np.asarray(to_output_order(grid.array_xyz, array_order="zyx"), dtype=np.float32)
    img = sitk.GetImageFromArray(arr_zyx, isVector=False)
    img.SetSpacing(grid.spacing)
    img.SetOrigin(grid.origin)
    sitk.WriteImage(img, str(out))
    return out
```

- [ ] **Step 4: Run image tests**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_images.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit image support**

```bash
git add src/parosol_py/images.py tests/test_images.py
git commit -m "feat: add image normalization and export"
```

## Task 3: Material Conversion

**Files:**
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/materials.py`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_materials.py`

- [ ] **Step 1: Write failing material tests**

Create `/Users/matthias.walle/Documents/fea/tests/test_materials.py`:

```python
import numpy as np
import pytest

from parosol_py.materials import material_to_stiffness_gpa, parse_linear_isotropic_materials


def test_material_to_stiffness_gpa_from_mpa():
    material_mpa = np.array([[[0.0, 1000.0], [2000.0, 0.0]]], dtype=np.float64)
    out = material_to_stiffness_gpa(material_mpa, material_unit="MPa")
    assert out.dtype == np.float32
    assert np.allclose(out, np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=np.float32))


def test_material_to_stiffness_gpa_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        material_to_stiffness_gpa(np.array([[[-1.0]]]), material_unit="MPa")


def test_parse_linear_isotropic_materials():
    text = """MaterialDefinitions:
    Material_001:
        Type: LinearIsotropic
        E: 8748
        nu: 0.3
    Material_002:
        Type: LinearIsotropic
        E: 10000
        nu: 0.25
MaterialTable:
    1: Material_001
    2: Material_002
"""
    parsed = parse_linear_isotropic_materials(text)
    assert parsed.youngs_modulus_mpa == {1: 8748.0, 2: 10000.0}
    assert parsed.poisson_ratio == {1: 0.3, 2: 0.25}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_materials.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement material module**

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/materials.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearIsotropicMaterials:
    youngs_modulus_mpa: dict[int, float]
    poisson_ratio: dict[int, float]


def material_to_stiffness_gpa(material, *, material_unit: str = "MPa") -> np.ndarray:
    arr = np.asarray(material, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"material must be 3D, got shape {arr.shape}")
    if np.any(arr < 0.0):
        raise ValueError("material values must be non-negative")
    unit = material_unit.strip().lower()
    if unit == "mpa":
        out = arr / 1000.0
    elif unit == "gpa":
        out = arr
    else:
        raise ValueError("material_unit must be 'MPa' or 'GPa'")
    return np.ascontiguousarray(out.astype(np.float32, copy=False))


def parse_linear_isotropic_materials(text: str) -> LinearIsotropicMaterials:
    blocks = re.finditer(
        r"(?P<name>[A-Za-z0-9_]+):\s*\n\s*Type:\s*LinearIsotropic\s*\n\s*E:\s*(?P<E>[-+0-9.eE]+)\s*\n\s*nu:\s*(?P<nu>[-+0-9.eE]+)",
        text,
    )
    definitions: dict[str, tuple[float, float]] = {}
    for match in blocks:
        definitions[match.group("name")] = (float(match.group("E")), float(match.group("nu")))
    if not definitions:
        raise ValueError("No LinearIsotropic material definitions found")

    table_match = re.search(r"MaterialTable:\s*(?P<table>.*)", text, flags=re.S)
    if table_match is None:
        raise ValueError("MaterialTable section not found")

    youngs: dict[int, float] = {}
    poisson: dict[int, float] = {}
    for line in table_match.group("table").splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        label_text, name = [part.strip() for part in stripped.split(":", 1)]
        if not label_text.isdigit():
            continue
        if name not in definitions:
            raise ValueError(f"MaterialTable references undefined material '{name}'")
        label = int(label_text)
        youngs[label], poisson[label] = definitions[name]
    if not youngs:
        raise ValueError("MaterialTable contains no numeric labels")
    return LinearIsotropicMaterials(youngs_modulus_mpa=youngs, poisson_ratio=poisson)
```

- [ ] **Step 4: Run material tests**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_materials.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit material support**

```bash
git add src/parosol_py/materials.py tests/test_materials.py
git commit -m "feat: add material conversion helpers"
```

## Task 4: Boundary Conditions and HDF5 Input Writer

**Files:**
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/boundary_conditions.py`
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/hdf5_io.py`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_boundary_conditions.py`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_hdf5_io.py`

- [ ] **Step 1: Write failing boundary condition tests**

Create `/Users/matthias.walle/Documents/fea/tests/test_boundary_conditions.py`:

```python
import numpy as np

from parosol_py.boundary_conditions import axial_compression


def test_axial_compression_z_generates_bottom_and_top_constraints():
    stiffness = np.ones((3, 2, 4), dtype=np.float32)
    coords, values = axial_compression(stiffness, axis="z", strain=-0.01)

    assert coords.shape[1] == 4
    assert values.shape == (coords.shape[0],)
    assert coords.dtype == np.uint16
    assert values.dtype == np.float32

    bottom_z = coords[:, 2] == 0
    top_z = coords[:, 2] == 4
    assert np.any(bottom_z)
    assert np.any(top_z)
    assert np.all(values[bottom_z] == 1e-16)
    assert np.allclose(values[top_z], -0.04)


def test_axial_compression_ignores_empty_columns():
    stiffness = np.zeros((2, 2, 2), dtype=np.float32)
    stiffness[0, 0, :] = 1.0

    coords, _values = axial_compression(stiffness, axis="z", strain=-0.01)

    unique_xy = set(map(tuple, coords[:, :2]))
    assert unique_xy == {(0, 0)}
```

- [ ] **Step 2: Write failing HDF5 IO test**

Create `/Users/matthias.walle/Documents/fea/tests/test_hdf5_io.py`:

```python
from pathlib import Path

import h5py
import numpy as np

from parosol_py.boundary_conditions import axial_compression
from parosol_py.hdf5_io import write_parosol_input


def test_write_parosol_input_schema(tmp_path: Path):
    stiffness_xyz = np.ones((3, 2, 4), dtype=np.float32)
    coords, values = axial_compression(stiffness_xyz, axis="z", strain=-0.01)
    out = tmp_path / "case.h5"

    write_parosol_input(
        out,
        stiffness_gpa_xyz=stiffness_xyz,
        fixed_displacement_coordinates=coords,
        fixed_displacement_values=values,
        voxel_size_mm=0.061,
        poisson_ratio=0.3,
    )

    with h5py.File(out, "r") as h5:
        group = h5["Image_Data"]
        assert set(group.keys()) == {
            "Fixed_Displacement_Coordinates",
            "Fixed_Displacement_Values",
            "Image",
            "Poisons_ratio",
            "Voxelsize",
        }
        assert group["Image"].shape == (4, 2, 3)
        assert np.array_equal(group["Image"][...], np.swapaxes(stiffness_xyz, 0, 2))
        assert float(group["Voxelsize"][()]) == 0.061
        assert float(group["Poisons_ratio"][()]) == 0.3
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_boundary_conditions.py /Users/matthias.walle/Documents/fea/tests/test_hdf5_io.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement boundary conditions**

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/boundary_conditions.py`:

```python
from __future__ import annotations

import numpy as np

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def axial_compression(stiffness_gpa_xyz, *, axis: str = "z", strain: float = -0.01) -> tuple[np.ndarray, np.ndarray]:
    stiffness = np.asarray(stiffness_gpa_xyz)
    if stiffness.ndim != 3:
        raise ValueError(f"stiffness_gpa_xyz must be 3D, got shape {stiffness.shape}")
    token = axis.strip().lower()
    if token not in AXIS_TO_INDEX:
        raise ValueError("axis must be one of: x, y, z")

    axis_index = AXIS_TO_INDEX[token]
    dims = np.asarray(stiffness.shape, dtype=np.int64)
    node_max = int(dims[axis_index])
    displacement = float(strain) * float(node_max)

    coords: list[list[int]] = []
    values: list[float] = []

    occupied = stiffness > 0.0
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    projected = np.any(occupied, axis=axis_index)
    for lateral_index in np.argwhere(projected):
        base = [0, 0, 0]
        base[lateral_axes[0]] = int(lateral_index[0])
        base[lateral_axes[1]] = int(lateral_index[1])
        for node_coord, value in ((0, 1e-16), (node_max, displacement)):
            coord = base.copy()
            coord[axis_index] = int(node_coord)
            coord.append(axis_index)
            coords.append(coord)
            values.append(float(value))

    if not coords:
        raise ValueError("No non-zero stiffness voxels found for boundary conditions")
    return np.asarray(coords, dtype=np.uint16), np.asarray(values, dtype=np.float32)
```

- [ ] **Step 5: Implement HDF5 writer**

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/hdf5_io.py`:

```python
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_parosol_input(
    path: str | Path,
    *,
    stiffness_gpa_xyz,
    fixed_displacement_coordinates,
    fixed_displacement_values,
    voxel_size_mm: float,
    poisson_ratio: float,
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    stiffness = np.asarray(stiffness_gpa_xyz, dtype=np.float32)
    coords = np.asarray(fixed_displacement_coordinates, dtype=np.uint16)
    values = np.asarray(fixed_displacement_values, dtype=np.float32)
    if stiffness.ndim != 3:
        raise ValueError(f"stiffness_gpa_xyz must be 3D, got shape {stiffness.shape}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("fixed_displacement_coordinates must have shape (n, 4)")
    if values.shape != (coords.shape[0],):
        raise ValueError("fixed_displacement_values must have shape (n,)")

    with h5py.File(out, "w") as h5:
        group = h5.create_group("Image_Data")
        group.create_dataset("Fixed_Displacement_Coordinates", data=coords)
        group.create_dataset("Fixed_Displacement_Values", data=values)
        group.create_dataset("Poisons_ratio", data=float(poisson_ratio))
        group.create_dataset("Voxelsize", data=float(voxel_size_mm))
        group.create_dataset("Image", data=np.swapaxes(stiffness, 0, 2))
    return out
```

- [ ] **Step 6: Run boundary/HDF5 tests**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_boundary_conditions.py /Users/matthias.walle/Documents/fea/tests/test_hdf5_io.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit boundary and HDF5 IO**

```bash
git add src/parosol_py/boundary_conditions.py src/parosol_py/hdf5_io.py tests/test_boundary_conditions.py tests/test_hdf5_io.py
git commit -m "feat: write parosol hdf5 inputs"
```

## Task 5: Runner and Result Reader

**Files:**
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/runner.py`
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/results.py`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_runner.py`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_results.py`

- [ ] **Step 1: Write failing result tests**

Create `/Users/matthias.walle/Documents/fea/tests/test_results.py`:

```python
from pathlib import Path

import h5py
import numpy as np

from parosol_py.results import read_solution_fields


def test_read_solution_fields_scalar_and_tensor(tmp_path: Path):
    h5_path = tmp_path / "solved.h5"
    with h5py.File(h5_path, "w") as h5:
        sol = h5.create_group("Solution")
        sol.create_dataset("SED", data=np.array([1.0, 2.0], dtype=np.float32))
        sol.create_dataset("e_xx", data=np.array([0.1, 0.2], dtype=np.float32))
        sol.create_dataset("e_yy", data=np.array([0.3, 0.4], dtype=np.float32))
        sol.create_dataset("e_zz", data=np.array([0.5, 0.6], dtype=np.float32))
        sol.create_dataset("e_xy", data=np.array([0.7, 0.8], dtype=np.float32))
        sol.create_dataset("e_yz", data=np.array([0.9, 1.0], dtype=np.float32))
        sol.create_dataset("e_xz", data=np.array([1.1, 1.2], dtype=np.float32))

    fields = read_solution_fields(h5_path, outputs=("sed", "strain"))

    assert np.allclose(fields["sed"], [1.0, 2.0])
    assert set(fields["strain"]) == {"xx", "yy", "zz", "xy", "yz", "xz"}
    assert np.allclose(fields["strain"]["xz"], [1.1, 1.2])
```

- [ ] **Step 2: Write failing runner tests**

Create `/Users/matthias.walle/Documents/fea/tests/test_runner.py`:

```python
from pathlib import Path

import pytest

from parosol_py.runner import build_parosol_command, parse_run_summary


def test_build_parosol_command_maps_outputs():
    cmd = build_parosol_command(
        executable=Path("/opt/parosol"),
        input_file=Path("/tmp/case.h5"),
        outputs=("sed", "strain", "stress"),
        tolerance=1e-7,
        level=4,
    )
    assert cmd == [
        "/opt/parosol",
        "--SED",
        "--strain",
        "--stress",
        "--tol",
        "1e-07",
        "--level",
        "4",
        "/tmp/case.h5",
    ]


def test_parse_run_summary_extracts_solver_metrics():
    text = """#  Nr of It: 123
#  Relative residuum: 4.5e-08
#  Absolute residuum: 2.3e-04
#  Overall:  1.25
"""
    summary = parse_run_summary(text)
    assert summary.iterations == 123
    assert summary.relative_residual == pytest.approx(4.5e-8)
    assert summary.absolute_residual == pytest.approx(2.3e-4)
    assert summary.overall_time_seconds == pytest.approx(1.25)
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_runner.py /Users/matthias.walle/Documents/fea/tests/test_results.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement result reader**

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/results.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

OUTPUT_DATASETS = {
    "sed": "SED",
    "von_mises": "VonMises",
    "effective_strain": "EFF",
    "deviatoric_strain": "e_dev",
    "volumetric_strain": "e_vol",
}
TENSOR_AXES = ("xx", "yy", "zz", "xy", "yz", "xz")


def read_solution_fields(path: str | Path, *, outputs: tuple[str, ...]) -> dict[str, Any]:
    requested = tuple(token.strip().lower() for token in outputs)
    fields: dict[str, Any] = {}
    with h5py.File(Path(path), "r") as h5:
        if "Solution" not in h5:
            raise ValueError("ParOSol output does not contain /Solution")
        solution = h5["Solution"]
        for output in requested:
            if output in OUTPUT_DATASETS:
                dataset = OUTPUT_DATASETS[output]
                if dataset not in solution:
                    raise ValueError(f"Requested output '{output}' not found in /Solution/{dataset}")
                fields[output] = np.asarray(solution[dataset][...])
            elif output == "strain":
                fields[output] = _read_tensor(solution, prefix="e_")
            elif output == "stress":
                fields[output] = _read_tensor(solution, prefix="s_")
            elif output in {"forces", "force"}:
                fields["forces"] = np.asarray(solution["force"][...])
            elif output in {"displacements", "disp"}:
                fields["displacements"] = np.asarray(solution["disp"][...])
            else:
                raise ValueError(f"Unsupported output '{output}'")
    return fields


def _read_tensor(solution, *, prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for axis in TENSOR_AXES:
        name = f"{prefix}{axis}"
        if name not in solution:
            raise ValueError(f"Tensor component missing from solution: {name}")
        out[axis] = np.asarray(solution[name][...])
    return out
```

- [ ] **Step 5: Implement runner**

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/runner.py`:

```python
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

OUTPUT_FLAGS = {
    "sed": "--SED",
    "strain": "--strain",
    "stress": "--stress",
    "von_mises": "--VonMises",
    "effective_strain": "--EFF",
    "deviatoric_strain": "--e_dev",
    "volumetric_strain": "--e_vol",
}


@dataclass(frozen=True)
class RunSummary:
    iterations: int | None = None
    relative_residual: float | None = None
    absolute_residual: float | None = None
    overall_time_seconds: float | None = None


@dataclass(frozen=True)
class RunResult:
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    summary: RunSummary


def packaged_executable() -> Path:
    return resources.files("parosol_py").joinpath("bin/parosol")


def build_parosol_command(
    *,
    executable: str | Path,
    input_file: str | Path,
    outputs: tuple[str, ...],
    tolerance: float = 1e-6,
    level: int = 6,
) -> list[str]:
    cmd = [str(Path(executable))]
    for output in outputs:
        token = output.strip().lower()
        if token in {"forces", "force", "displacements", "disp"}:
            continue
        if token not in OUTPUT_FLAGS:
            raise ValueError(f"Unsupported ParOSol output '{output}'")
        flag = OUTPUT_FLAGS[token]
        if flag not in cmd:
            cmd.append(flag)
    cmd.extend(["--tol", f"{float(tolerance):g}", "--level", str(int(level)), str(Path(input_file))])
    return cmd


def parse_run_summary(stdout: str) -> RunSummary:
    patterns = {
        "iterations": (r"#\s+Nr of It:\s+([0-9]+)", int),
        "relative_residual": (r"#\s+Relative residuum:\s+([-+0-9.eE]+)", float),
        "absolute_residual": (r"#\s+Absolute residuum:\s+([-+0-9.eE]+)", float),
        "overall_time_seconds": (r"#\s+Overall:\s+([-+0-9.eE]+)", float),
    }
    values = {}
    for name, (pattern, cast) in patterns.items():
        match = re.search(pattern, stdout)
        values[name] = cast(match.group(1)) if match else None
    return RunSummary(**values)


def run_parosol(command: list[str], *, cwd: str | Path | None = None) -> RunResult:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    summary = parse_run_summary(proc.stdout)
    return RunResult(command=command, stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode, summary=summary)
```

- [ ] **Step 6: Run runner/result tests**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_runner.py /Users/matthias.walle/Documents/fea/tests/test_results.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit runner/result support**

```bash
git add src/parosol_py/runner.py src/parosol_py/results.py tests/test_runner.py tests/test_results.py
git commit -m "feat: add parosol runner and result reader"
```

## Task 6: Public API Integration

**Files:**
- Modify: `/Users/matthias.walle/Documents/fea/src/parosol_py/__init__.py`
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/api.py`
- Modify: `/Users/matthias.walle/Documents/fea/tests/test_api.py`

- [ ] **Step 1: Replace API tests with dry-run integration tests**

Modify `/Users/matthias.walle/Documents/fea/tests/test_api.py`:

```python
from pathlib import Path

import numpy as np

import parosol_py
from parosol_py import solve


def test_package_imports():
    assert parosol_py.__version__ == "0.1.0"


def test_solve_dry_run_writes_input_and_returns_result(tmp_path: Path):
    material_zyx = np.zeros((4, 3, 2), dtype=np.float64)
    material_zyx[:, 1, 1] = 1000.0

    result = solve(
        material=material_zyx,
        spacing=(0.061, 0.061, 0.061),
        origin=(1.0, 2.0, 3.0),
        material_unit="MPa",
        test="axial",
        test_axis="z",
        strain=-0.01,
        outputs=("sed",),
        work_dir=tmp_path,
        dry_run=True,
    )

    assert result.input_file.exists()
    assert result.command[-1] == str(result.input_file)
    assert "--SED" in result.command
    assert result.fields == {}
    assert result.summary.dimensions_xyz == (2, 3, 4)
    assert result.summary.spacing == (0.061, 0.061, 0.061)
```

- [ ] **Step 2: Run API tests to verify failure**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_api.py -v
```

Expected: FAIL with missing `solve`.

- [ ] **Step 3: Implement API module**

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/api.py`:

```python
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .boundary_conditions import axial_compression
from .hdf5_io import write_parosol_input
from .images import normalize_array
from .materials import material_to_stiffness_gpa
from .results import read_solution_fields
from .runner import RunSummary, build_parosol_command, packaged_executable, run_parosol


@dataclass(frozen=True)
class SolveSummary:
    dimensions_xyz: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    run: RunSummary | None = None


@dataclass(frozen=True)
class SolveResult:
    input_file: Path
    command: list[str]
    fields: dict[str, Any]
    summary: SolveSummary
    stdout: str = ""
    stderr: str = ""
    exported: dict[str, Path] = field(default_factory=dict)


def solve(
    *,
    material,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    array_order: str = "zyx",
    material_unit: str = "MPa",
    poisson_ratio: float = 0.3,
    test: str = "axial",
    test_axis: str = "z",
    strain: float = -0.01,
    outputs: tuple[str, ...] = ("sed",),
    tolerance: float = 1e-6,
    level: int = 6,
    executable: str | Path | None = None,
    work_dir: str | Path | None = None,
    dry_run: bool = False,
) -> SolveResult:
    if test.strip().lower() != "axial":
        raise ValueError("First-pass solve supports only test='axial'")

    grid = normalize_array(material, spacing=spacing, origin=origin, array_order=array_order)
    stiffness = material_to_stiffness_gpa(grid.array_xyz, material_unit=material_unit)
    coords, values = axial_compression(stiffness, axis=test_axis, strain=strain)

    run_dir = Path(work_dir).expanduser().resolve() if work_dir is not None else Path(tempfile.mkdtemp(prefix="parosol-py-")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    input_file = run_dir / "parosol_input.h5"
    write_parosol_input(
        input_file,
        stiffness_gpa_xyz=stiffness,
        fixed_displacement_coordinates=coords,
        fixed_displacement_values=values,
        voxel_size_mm=float(grid.spacing[0]),
        poisson_ratio=float(poisson_ratio),
    )

    exe = Path(executable).expanduser().resolve() if executable is not None else packaged_executable()
    command = build_parosol_command(executable=exe, input_file=input_file, outputs=outputs, tolerance=tolerance, level=level)
    summary = SolveSummary(dimensions_xyz=tuple(int(v) for v in stiffness.shape), spacing=grid.spacing, origin=grid.origin)
    if dry_run:
        return SolveResult(input_file=input_file, command=command, fields={}, summary=summary)

    run = run_parosol(command, cwd=run_dir)
    if run.returncode != 0:
        raise RuntimeError(f"ParOSol failed with code {run.returncode}\\nSTDOUT:\\n{run.stdout}\\nSTDERR:\\n{run.stderr}")
    fields = read_solution_fields(input_file, outputs=outputs)
    return SolveResult(
        input_file=input_file,
        command=command,
        fields=fields,
        summary=SolveSummary(dimensions_xyz=summary.dimensions_xyz, spacing=summary.spacing, origin=summary.origin, run=run.summary),
        stdout=run.stdout,
        stderr=run.stderr,
    )
```

- [ ] **Step 4: Export API from package**

Modify `/Users/matthias.walle/Documents/fea/src/parosol_py/__init__.py`:

```python
from ._version import __version__
from .api import SolveResult, SolveSummary, solve

__all__ = ["SolveResult", "SolveSummary", "__version__", "solve"]
```

- [ ] **Step 5: Run API tests**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit public API**

```bash
git add src/parosol_py/__init__.py src/parosol_py/api.py tests/test_api.py
git commit -m "feat: add dry-run solve api"
```

## Task 7: AIM Entry Point and Validation Helpers

**Files:**
- Modify: `/Users/matthias.walle/Documents/fea/src/parosol_py/api.py`
- Create: `/Users/matthias.walle/Documents/fea/src/parosol_py/validation.py`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_validation.py`
- Modify: `/Users/matthias.walle/Documents/fea/tests/test_api.py`

- [ ] **Step 1: Add failing validation tests**

Create `/Users/matthias.walle/Documents/fea/tests/test_validation.py`:

```python
import numpy as np
import pytest

from parosol_py.validation import compare_field


def test_compare_field_passes_within_tolerance():
    summary = compare_field(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.001, 2.999]),
        rtol=1e-2,
        atol=1e-3,
    )
    assert summary.max_abs_error == pytest.approx(0.001)
    assert summary.passed is True


def test_compare_field_fails_outside_tolerance():
    summary = compare_field(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.5, 3.0]),
        rtol=1e-3,
        atol=1e-6,
    )
    assert summary.passed is False
    assert summary.max_abs_error == pytest.approx(0.5)
```

- [ ] **Step 2: Add failing `solve_aim` API test with monkeypatch**

Append to `/Users/matthias.walle/Documents/fea/tests/test_api.py`:

```python
def test_solve_aim_delegates_to_py_aimio(monkeypatch, tmp_path: Path):
    from parosol_py.api import solve_aim

    calls = {}

    def fake_read_aim(path):
        calls["path"] = path
        arr = np.zeros((4, 3, 2), dtype=np.float64)
        arr[:, 1, 1] = 1000.0
        return arr, {"element_size": (0.061, 0.061, 0.061), "position": (1.0, 2.0, 3.0)}

    monkeypatch.setattr("parosol_py.api.read_aim", fake_read_aim)
    result = solve_aim("case.aim", work_dir=tmp_path, dry_run=True)

    assert calls["path"] == "case.aim"
    assert result.input_file.exists()
    assert result.summary.spacing == (0.061, 0.061, 0.061)
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_validation.py /Users/matthias.walle/Documents/fea/tests/test_api.py -v
```

Expected: FAIL with missing `validation` and missing `solve_aim`.

- [ ] **Step 4: Implement validation helpers**

Create `/Users/matthias.walle/Documents/fea/src/parosol_py/validation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldComparison:
    passed: bool
    max_abs_error: float
    mean_abs_error: float
    rtol: float
    atol: float


def compare_field(reference, candidate, *, rtol: float = 1e-5, atol: float = 1e-8) -> FieldComparison:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape}, candidate {cand.shape}")
    abs_error = np.abs(ref - cand)
    return FieldComparison(
        passed=bool(np.allclose(ref, cand, rtol=rtol, atol=atol)),
        max_abs_error=float(abs_error.max(initial=0.0)),
        mean_abs_error=float(abs_error.mean() if abs_error.size else 0.0),
        rtol=float(rtol),
        atol=float(atol),
    )
```

- [ ] **Step 5: Implement `solve_aim`**

Modify `/Users/matthias.walle/Documents/fea/src/parosol_py/api.py` imports:

```python
from py_aimio import read_aim
```

Add this function below `solve`:

```python
def solve_aim(path: str | Path, **kwargs: Any) -> SolveResult:
    arr, meta = read_aim(str(path))
    spacing = kwargs.pop("spacing", None)
    if spacing is None:
        spacing = tuple(float(v) for v in meta.get("element_size", (1.0, 1.0, 1.0)))
    origin = kwargs.pop("origin", None)
    if origin is None:
        origin = tuple(float(v) for v in meta.get("position", (0.0, 0.0, 0.0)))
    return solve(material=arr, spacing=spacing, origin=origin, array_order="zyx", **kwargs)
```

Modify `/Users/matthias.walle/Documents/fea/src/parosol_py/__init__.py`:

```python
from ._version import __version__
from .api import SolveResult, SolveSummary, solve, solve_aim

__all__ = ["SolveResult", "SolveSummary", "__version__", "solve", "solve_aim"]
```

- [ ] **Step 6: Run API and validation tests**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_validation.py /Users/matthias.walle/Documents/fea/tests/test_api.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit AIM and validation helpers**

```bash
git add src/parosol_py/api.py src/parosol_py/__init__.py src/parosol_py/validation.py tests/test_api.py tests/test_validation.py
git commit -m "feat: add aim entry point and validation helpers"
```

## Task 8: Real Solver Smoke Test and Exported Field Support

**Files:**
- Modify: `/Users/matthias.walle/Documents/fea/src/parosol_py/api.py`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_api.py`
- Test: `/Users/matthias.walle/Documents/fea/tests/test_solver_smoke.py`

- [ ] **Step 1: Add export support test**

Append to `/Users/matthias.walle/Documents/fea/tests/test_api.py`:

```python
def test_solve_dry_run_accepts_export_dir(tmp_path: Path):
    material_zyx = np.ones((4, 3, 2), dtype=np.float64) * 1000.0
    result = solve(
        material=material_zyx,
        spacing=(1.0, 1.0, 1.0),
        outputs=("sed",),
        work_dir=tmp_path,
        export_dir=tmp_path / "exports",
        dry_run=True,
    )
    assert result.exported == {}
```

- [ ] **Step 2: Add optional real solver smoke test**

Create `/Users/matthias.walle/Documents/fea/tests/test_solver_smoke.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from parosol_py import solve
from parosol_py.runner import packaged_executable


def test_real_solver_cube_smoke(tmp_path: Path):
    executable = packaged_executable()
    if not executable.exists():
        pytest.skip(f"Packaged ParOSol executable not available: {executable}")

    material_zyx = np.ones((3, 3, 3), dtype=np.float64) * 1000.0
    result = solve(
        material=material_zyx,
        spacing=(1.0, 1.0, 1.0),
        test="axial",
        test_axis="z",
        strain=-0.01,
        outputs=("sed",),
        work_dir=tmp_path,
        tolerance=1e-4,
        level=2,
    )

    assert "sed" in result.fields
    assert result.summary.run is not None
    assert result.summary.run.iterations is None or result.summary.run.iterations > 0
```

- [ ] **Step 3: Run tests to verify export-dir failure**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_api.py::test_solve_dry_run_accepts_export_dir -v
```

Expected: FAIL with unexpected `export_dir` argument.

- [ ] **Step 4: Add export_dir argument and post-run scalar export**

Modify `solve` signature in `/Users/matthias.walle/Documents/fea/src/parosol_py/api.py`:

```python
    export_dir: str | Path | None = None,
```

Add import:

```python
import numpy as np
from .images import ImageGrid, export_scalar_image, normalize_array
```

Replace the final return block after reading fields:

```python
    exported: dict[str, Path] = {}
    if export_dir is not None:
        out_dir = Path(export_dir).expanduser().resolve()
        for name, value in fields.items():
            if isinstance(value, np.ndarray) and value.ndim == 1 and value.size == stiffness.size:
                field_xyz = np.asarray(value, dtype=np.float32).reshape(stiffness.shape, order="C")
                exported[name] = export_scalar_image(
                    ImageGrid(array_xyz=field_xyz, spacing=grid.spacing, origin=grid.origin),
                    out_dir / f"{name}.nii.gz",
                )
    return SolveResult(
        input_file=input_file,
        command=command,
        fields=fields,
        summary=SolveSummary(dimensions_xyz=summary.dimensions_xyz, spacing=summary.spacing, origin=summary.origin, run=run.summary),
        stdout=run.stdout,
        stderr=run.stderr,
        exported=exported,
    )
```

- [ ] **Step 5: Run API tests and optional smoke test**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests/test_api.py /Users/matthias.walle/Documents/fea/tests/test_solver_smoke.py -v
```

Expected: API tests PASS. Smoke test either PASS if native executable builds, or SKIP if executable is unavailable.

- [ ] **Step 6: Run full test suite**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests -v
```

Expected: PASS, with the real solver smoke test allowed to SKIP only when the packaged executable is unavailable.

- [ ] **Step 7: Commit smoke/export support**

```bash
git add src/parosol_py/api.py tests/test_api.py tests/test_solver_smoke.py
git commit -m "feat: add field export and solver smoke test"
```

## Task 9: Documentation and Final Verification

**Files:**
- Modify: `/Users/matthias.walle/Documents/fea/README.md`
- Create: `/Users/matthias.walle/Documents/fea/docs/superpowers/specs/2026-05-28-parosol-py-design.md` already exists
- Create: `/Users/matthias.walle/Documents/fea/docs/superpowers/plans/2026-05-28-parosol-py-implementation.md` already exists

- [ ] **Step 1: Update README usage example**

Modify `/Users/matthias.walle/Documents/fea/README.md`:

```markdown
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

## AIM Input

```python
from parosol_py import solve_aim

result = solve_aim("segmented.aim", outputs=("sed",), export_dir="outputs")
```

AIM IO is provided by `aimio-py` through `py_aimio`.

## Scope

This first pass provides the clean Python API. FAIM-ish command-line compatibility is planned as a second pass.
```

- [ ] **Step 2: Run final verification**

Run:

```bash
python -m pytest /Users/matthias.walle/Documents/fea/tests -v
python -m pip install -e /Users/matthias.walle/Documents/fea
python - <<'PY'
import parosol_py
print(parosol_py.__version__)
PY
```

Expected: tests PASS, editable install succeeds or reports a clearly documented native dependency issue, import prints `0.1.0`.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: only intentional changes remain. The previously untracked `framework-main/` may still be present and should not be staged unless the implementation explicitly copied needed native files into `src/parosol_native/`.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/superpowers/plans/2026-05-28-parosol-py-implementation.md
git commit -m "docs: add parosol-py implementation plan and usage"
```

## Self-Review

- Spec coverage: the plan covers package scaffold, ParOSol native build, NumPy API, AIM delegation through `aimio-py`, HDF5 input, solver runner, field reading, `.nii.gz` export, validation helpers, tests, and documentation. FAIM-ish CLI is intentionally excluded as second-pass work.
- Placeholder scan: no task contains unresolved `TBD`, `TODO`, or "implement later" steps. The only conditional path is native dependency failure during editable install, which is explicitly captured as a verification outcome.
- Type consistency: public names are consistent across tasks: `solve`, `solve_aim`, `SolveResult`, `SolveSummary`, `ImageGrid`, `normalize_array`, `material_to_stiffness_gpa`, `write_parosol_input`, `build_parosol_command`, `read_solution_fields`, and `compare_field`.
