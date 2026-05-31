# FEA API and Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular FEA API with explicit models, boundary conditions, load cases, solver/output profiles, fast field export hooks, and legacy solver validation scaffolding.

**Architecture:** Add typed core objects beside the existing compatibility API, then progressively route existing `solve()` and CLI config through those objects. Keep ParOSol HDF5 writing and result parsing as the solver adapter, while load cases generate inspectable `BoundaryConditionSet` instances.

**Tech Stack:** Python dataclasses, NumPy, h5py, SimpleITK, pytest, Ruff, existing scikit-build-core ParOSol packaging.

---

## File Structure

- Create `src/parosol_py/core.py`: `Model`, `BoundaryConditionSet`, `SolverProfile`, `OutputProfile`, and serialization helpers.
- Create `src/parosol_py/load_cases.py`: `AxialCompression`, `SimpleShear`, and force distribution helpers that generate `BoundaryConditionSet`.
- Create `src/parosol_py/field_export.py`: native-to-dense field mapping cache and fast scalar image export.
- Create `src/parosol_py/profiles.py`: built-in solver/output profile registry.
- Create `src/parosol_py/reference_validation.py`: local validation manifest, legacy solver reference parsing, ParOSol summary comparison.
- Modify `src/parosol_py/hdf5_io.py`: write optional `Loaded_Nodes_*` datasets.
- Modify `src/parosol_py/api.py`: accept core objects and preserve existing `solve()` compatibility.
- Modify `src/parosol_py/config.py`: load profile names and new `model:` config shape while preserving current `input:` shape.
- Modify `src/parosol_py/cli.py`: add `validate-reference` after validation helpers exist.
- Modify `src/parosol_py/__init__.py`: export public core/load/profile classes.
- Add tests in `tests/test_core.py`, `tests/test_load_cases.py`, `tests/test_hdf5_io.py`, `tests/test_field_export.py`, `tests/test_profiles.py`, and `tests/test_reference_validation.py`.

---

### Task 1: Core Data Objects

**Files:**
- Create: `src/parosol_py/core.py`
- Modify: `src/parosol_py/__init__.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing tests for model, boundary condition, and profile objects**

Create `tests/test_core.py`:

```python
import numpy as np

from parosol_py import BoundaryConditionSet, Model, OutputProfile, SolverProfile


def test_model_normalizes_material_to_xyz_and_tracks_spacing():
    material_zyx = np.arange(24, dtype=np.float32).reshape((2, 3, 4))

    model = Model.from_array(
        material_zyx,
        spacing=(0.061, 0.061, 0.061),
        origin=(1.0, 2.0, 3.0),
        array_order="zyx",
        material_unit="MPa",
    )

    assert model.material_xyz.shape == (4, 3, 2)
    assert model.spacing == (0.061, 0.061, 0.061)
    assert model.origin == (1.0, 2.0, 3.0)
    assert model.material_unit == "MPa"
    np.testing.assert_array_equal(model.material_xyz, np.transpose(material_zyx, (2, 1, 0)))


def test_boundary_condition_set_serializes_fixed_and_loaded_nodes():
    bc = BoundaryConditionSet(
        fixed_coordinates=np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint16),
        fixed_values=np.array([1e-16, 1e-16], dtype=np.float32),
        loaded_coordinates=np.array([[1, 1, 1, 2]], dtype=np.uint16),
        loaded_values=np.array([-12.5], dtype=np.float32),
        node_sets={"top": [(1, 1, 1)]},
    )

    data = bc.to_dict()
    restored = BoundaryConditionSet.from_dict(data)

    np.testing.assert_array_equal(restored.fixed_coordinates, bc.fixed_coordinates)
    np.testing.assert_array_equal(restored.loaded_coordinates, bc.loaded_coordinates)
    assert restored.node_sets == {"top": [(1, 1, 1)]}


def test_profiles_have_stable_defaults():
    solver = SolverProfile()
    output = OutputProfile()

    assert solver.tolerance == 1e-6
    assert solver.level == 6
    assert solver.mpi_processes == 1
    assert output.export_fields is True
    assert output.image_fields == ("sed",)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_core.py -q
```

Expected: import failure for missing `BoundaryConditionSet`, `Model`, `OutputProfile`, or `SolverProfile`.

- [ ] **Step 3: Implement core data objects**

Create `src/parosol_py/core.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .images import normalize_array


@dataclass(frozen=True)
class Model:
    material_xyz: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material_unit: str = "MPa"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_array(
        cls,
        material,
        *,
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        array_order: str = "zyx",
        material_unit: str = "MPa",
        metadata: dict[str, Any] | None = None,
    ) -> "Model":
        grid = normalize_array(
            material,
            spacing=spacing,
            origin=origin,
            array_order=array_order,
        )
        return cls(
            material_xyz=grid.array_xyz,
            spacing=grid.spacing,
            origin=grid.origin,
            material_unit=material_unit,
            metadata={} if metadata is None else dict(metadata),
        )


@dataclass(frozen=True)
class BoundaryConditionSet:
    fixed_coordinates: np.ndarray
    fixed_values: np.ndarray
    loaded_coordinates: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.uint16)
    )
    loaded_values: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    node_sets: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixed_coordinates",
            _coordinates(self.fixed_coordinates, "fixed_coordinates"),
        )
        object.__setattr__(self, "fixed_values", _values(self.fixed_values, "fixed_values"))
        object.__setattr__(
            self,
            "loaded_coordinates",
            _coordinates(self.loaded_coordinates, "loaded_coordinates"),
        )
        object.__setattr__(
            self,
            "loaded_values",
            _values(self.loaded_values, "loaded_values"),
        )
        if self.fixed_coordinates.shape[0] != self.fixed_values.shape[0]:
            raise ValueError("fixed coordinate/value counts differ")
        if self.loaded_coordinates.shape[0] != self.loaded_values.shape[0]:
            raise ValueError("loaded coordinate/value counts differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed_coordinates": self.fixed_coordinates.tolist(),
            "fixed_values": self.fixed_values.tolist(),
            "loaded_coordinates": self.loaded_coordinates.tolist(),
            "loaded_values": self.loaded_values.tolist(),
            "node_sets": {
                name: [list(coord) for coord in coords]
                for name, coords in self.node_sets.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundaryConditionSet":
        return cls(
            fixed_coordinates=np.asarray(data["fixed_coordinates"], dtype=np.uint16),
            fixed_values=np.asarray(data["fixed_values"], dtype=np.float32),
            loaded_coordinates=np.asarray(
                data.get("loaded_coordinates", []), dtype=np.uint16
            ).reshape((-1, 4)),
            loaded_values=np.asarray(data.get("loaded_values", []), dtype=np.float32),
            node_sets={
                name: [tuple(int(v) for v in coord) for coord in coords]
                for name, coords in data.get("node_sets", {}).items()
            },
        )


@dataclass(frozen=True)
class SolverProfile:
    tolerance: float = 1e-6
    level: int = 6
    mpi_processes: int = 1
    mpi_launcher: str = "mpirun"
    outputs: tuple[str, ...] = ("sed",)


@dataclass(frozen=True)
class OutputProfile:
    export_fields: bool = True
    image_fields: tuple[str, ...] = ("sed",)
    summary_name: str = "summary.json"
    retain_hdf5: bool = True


def _coordinates(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.uint16)
    if array.size == 0:
        return array.reshape((0, 4))
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (n, 4)")
    return array


def _values(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array
```

Modify `src/parosol_py/__init__.py`:

```python
from ._version import __version__
from .api import SolveResult, SolveSummary, solve, solve_aim
from .core import BoundaryConditionSet, Model, OutputProfile, SolverProfile

__all__ = [
    "BoundaryConditionSet",
    "Model",
    "OutputProfile",
    "SolveResult",
    "SolveSummary",
    "SolverProfile",
    "__version__",
    "solve",
    "solve_aim",
]
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/test_core.py -q
PYTHONPATH=src python -m ruff check src tests
```

Expected: tests pass and Ruff passes.

Commit:

```bash
git add src/parosol_py/core.py src/parosol_py/__init__.py tests/test_core.py
git commit -m "feat: add core fea data objects"
```

---

### Task 2: HDF5 Loaded Nodes Support

**Files:**
- Modify: `src/parosol_py/hdf5_io.py`
- Modify: `src/parosol_py/api.py`
- Test: `tests/test_hdf5_io.py`

- [ ] **Step 1: Write failing HDF5 test for loaded nodes**

Append to `tests/test_hdf5_io.py`:

```python
def test_write_parosol_input_writes_loaded_nodes(tmp_path):
    path = write_parosol_input(
        tmp_path / "case.h5",
        stiffness_gpa_xyz=np.ones((2, 2, 2), dtype=np.float32),
        fixed_displacement_coordinates=np.array([[0, 0, 0, 0]], dtype=np.uint16),
        fixed_displacement_values=np.array([1e-16], dtype=np.float32),
        loaded_node_coordinates=np.array([[2, 2, 2, 2]], dtype=np.uint16),
        loaded_node_values=np.array([-10.0], dtype=np.float32),
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
    )

    with h5py.File(path, "r") as h5:
        group = h5["Image_Data"]
        np.testing.assert_array_equal(
            group["Loaded_Nodes_Coordinates"][...],
            np.array([[2, 2, 2, 2]], dtype=np.uint16),
        )
        np.testing.assert_allclose(group["Loaded_Nodes_Values"][...], [-10.0])
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_hdf5_io.py::test_write_parosol_input_writes_loaded_nodes -q
```

Expected: `TypeError` for unexpected `loaded_node_coordinates` or missing dataset assertion.

- [ ] **Step 3: Implement loaded node writing**

Modify `write_parosol_input()` signature in `src/parosol_py/hdf5_io.py`:

```python
def write_parosol_input(
    path: str | Path,
    *,
    stiffness_gpa_xyz,
    fixed_displacement_coordinates,
    fixed_displacement_values,
    voxel_size_mm: float,
    poisson_ratio: float,
    loaded_node_coordinates=None,
    loaded_node_values=None,
) -> Path:
```

Inside the function, after fixed coordinates validation:

```python
    loaded_coords = np.zeros((0, 4), dtype=np.uint16) if loaded_node_coordinates is None else np.asarray(loaded_node_coordinates)
    loaded_values = np.zeros((0,), dtype=np.float32) if loaded_node_values is None else np.asarray(loaded_node_values, dtype=np.float32)
    if loaded_coords.size == 0:
        loaded_coords = loaded_coords.reshape((0, 4))
    if loaded_coords.ndim != 2 or loaded_coords.shape[1] != 4:
        raise ValueError("loaded_node_coordinates must have shape (n, 4)")
    if loaded_values.shape != (loaded_coords.shape[0],):
        raise ValueError("loaded_node_values must have shape (n,)")
    if np.any(loaded_coords[:, :3] > node_max_xyz):
        raise ValueError("loaded_node_coordinates exceed node bounds")
```

Write datasets:

```python
        loaded_coords_zyx = loaded_coords.astype(np.uint16, copy=False)[:, [2, 1, 0, 3]]
        group.create_dataset("Loaded_Nodes_Coordinates", data=loaded_coords_zyx)
        group.create_dataset("Loaded_Nodes_Values", data=loaded_values)
```

Modify `api.solve()` to pass empty loaded nodes for now:

```python
        loaded_node_coordinates=None,
        loaded_node_values=None,
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/test_hdf5_io.py -q
PYTHONPATH=src python -m ruff check src tests
```

Commit:

```bash
git add src/parosol_py/hdf5_io.py src/parosol_py/api.py tests/test_hdf5_io.py
git commit -m "feat: write parosol loaded node forces"
```

---

### Task 3: Load Case Objects

**Files:**
- Create: `src/parosol_py/load_cases.py`
- Modify: `src/parosol_py/__init__.py`
- Test: `tests/test_load_cases.py`

- [ ] **Step 1: Write failing load case tests**

Create `tests/test_load_cases.py`:

```python
import numpy as np

from parosol_py import AxialCompression, BodyWeightCompression, Model, SimpleShear


def test_axial_compression_generates_named_top_and_bottom_sets():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(0.5, 0.5, 0.5))

    bc = AxialCompression(axis="z", strain=-0.01).generate(model)

    assert "top" in bc.node_sets
    assert "bottom" in bc.node_sets
    top_z_values = bc.fixed_coordinates[bc.fixed_coordinates[:, 2] == 2]
    assert np.any(top_z_values[:, 3] == 2)
    assert np.min(bc.fixed_values) == np.float32(-0.01)


def test_body_weight_compression_distributes_total_force_over_top_nodes():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = BodyWeightCompression(axis="z", force_n=-90.0).generate(model)

    assert bc.loaded_coordinates.shape[0] == 9
    assert np.all(bc.loaded_coordinates[:, 3] == 2)
    assert np.sum(bc.loaded_values) == np.float32(-90.0)


def test_simple_shear_moves_top_in_lateral_direction():
    model = Model.from_array(np.ones((2, 2, 2)), spacing=(1, 1, 1))

    bc = SimpleShear(axis="z", direction="x", strain=0.02).generate(model)

    top_x_values = bc.fixed_values[
        (bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_coordinates[:, 3] == 0)
    ]
    assert np.any(np.isclose(top_x_values, 0.04))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_load_cases.py -q
```

Expected: import failure for missing load case classes.

- [ ] **Step 3: Implement load cases**

Create `src/parosol_py/load_cases.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary_conditions import AXIS_TO_INDEX, axial_compression
from .core import BoundaryConditionSet, Model


@dataclass(frozen=True)
class AxialCompression:
    axis: str = "z"
    strain: float = -0.01

    def generate(self, model: Model) -> BoundaryConditionSet:
        coords, values = axial_compression(
            model.material_xyz,
            axis=self.axis,
            strain=self.strain,
            voxel_size_mm=model.spacing[AXIS_TO_INDEX[self.axis]],
        )
        return BoundaryConditionSet(
            fixed_coordinates=coords,
            fixed_values=values,
            node_sets=_top_bottom_sets(model.material_xyz, self.axis),
        )


@dataclass(frozen=True)
class BodyWeightCompression:
    axis: str = "z"
    force_n: float = -1.0

    def generate(self, model: Model) -> BoundaryConditionSet:
        base = AxialCompression(axis=self.axis, strain=0.0).generate(model)
        axis_index = AXIS_TO_INDEX[self.axis]
        top = base.node_sets["top"]
        values = np.full((len(top),), float(self.force_n) / len(top), dtype=np.float32)
        coords = np.asarray([(*coord, axis_index) for coord in top], dtype=np.uint16)
        return BoundaryConditionSet(
            fixed_coordinates=base.fixed_coordinates,
            fixed_values=base.fixed_values,
            loaded_coordinates=coords,
            loaded_values=values,
            node_sets=base.node_sets,
        )


@dataclass(frozen=True)
class SimpleShear:
    axis: str = "z"
    direction: str = "x"
    strain: float = 0.01

    def generate(self, model: Model) -> BoundaryConditionSet:
        bc = AxialCompression(axis=self.axis, strain=0.0).generate(model)
        axis_index = AXIS_TO_INDEX[self.axis]
        direction_index = AXIS_TO_INDEX[self.direction]
        height = model.material_xyz.shape[axis_index] * model.spacing[axis_index]
        displacement = float(self.strain) * float(height)
        fixed = bc.fixed_coordinates.copy()
        values = bc.fixed_values.copy()
        top_axis_value = model.material_xyz.shape[axis_index]
        mask = (fixed[:, axis_index] == top_axis_value) & (fixed[:, 3] == direction_index)
        values[mask] = displacement
        return BoundaryConditionSet(
            fixed_coordinates=fixed,
            fixed_values=values,
            loaded_coordinates=bc.loaded_coordinates,
            loaded_values=bc.loaded_values,
            node_sets=bc.node_sets,
        )


def _top_bottom_sets(stiffness_xyz: np.ndarray, axis: str) -> dict[str, list[tuple[int, int, int]]]:
    axis_index = AXIS_TO_INDEX[axis]
    dims = tuple(int(v) for v in stiffness_xyz.shape)
    occupied = stiffness_xyz > 0
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    out = {"bottom": set(), "top": set()}
    for label, element_axis_value, node_axis_value in (
        ("bottom", 0, 0),
        ("top", dims[axis_index] - 1, dims[axis_index]),
    ):
        surface = np.take(occupied, indices=element_axis_value, axis=axis_index)
        for lateral_index in np.argwhere(surface):
            base = [0, 0, 0]
            base[axis_index] = node_axis_value
            base[lateral_axes[0]] = int(lateral_index[0])
            base[lateral_axes[1]] = int(lateral_index[1])
            for du in (0, 1):
                for dv in (0, 1):
                    node = base.copy()
                    node[lateral_axes[0]] += du
                    node[lateral_axes[1]] += dv
                    out[label].add(tuple(node))
    return {name: sorted(coords) for name, coords in out.items()}
```

Modify `src/parosol_py/__init__.py` to export `AxialCompression`, `BodyWeightCompression`, and `SimpleShear`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/test_load_cases.py -q
PYTHONPATH=src python -m ruff check src tests
```

Commit:

```bash
git add src/parosol_py/load_cases.py src/parosol_py/__init__.py tests/test_load_cases.py
git commit -m "feat: add explicit load case objects"
```

---

### Task 4: Route Compatibility API Through BoundaryConditionSet

**Files:**
- Modify: `src/parosol_py/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write failing test for solving with explicit boundary conditions**

Append to `tests/test_api.py`:

```python
def test_solve_accepts_explicit_boundary_condition_set(monkeypatch, tmp_path):
    from parosol_py import BoundaryConditionSet

    captured = {}
    bc = BoundaryConditionSet(
        fixed_coordinates=np.array([[0, 0, 0, 0]], dtype=np.uint16),
        fixed_values=np.array([1e-16], dtype=np.float32),
        loaded_coordinates=np.array([[2, 2, 2, 2]], dtype=np.uint16),
        loaded_values=np.array([-10.0], dtype=np.float32),
    )

    def fake_write_parosol_input(**kwargs):
        captured.update(kwargs)
        path = kwargs["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("h5")
        return path

    monkeypatch.setattr("parosol_py.api.write_parosol_input", fake_write_parosol_input)

    result = solve(
        material=np.ones((2, 2, 2)),
        spacing=(1, 1, 1),
        boundary_conditions=bc,
        work_dir=tmp_path,
        dry_run=True,
    )

    assert result.input_file.exists()
    np.testing.assert_array_equal(captured["loaded_node_coordinates"], bc.loaded_coordinates)
    np.testing.assert_array_equal(captured["loaded_node_values"], bc.loaded_values)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_api.py::test_solve_accepts_explicit_boundary_condition_set -q
```

Expected: `TypeError` for unexpected `boundary_conditions`.

- [ ] **Step 3: Implement compatibility routing**

Modify `solve()` signature:

```python
    boundary_conditions: BoundaryConditionSet | None = None,
```

Import `BoundaryConditionSet`. Replace direct axial call with:

```python
    if boundary_conditions is None:
        fixed_coords, fixed_values = axial_compression(...)
        loaded_coords = None
        loaded_values = None
    else:
        fixed_coords = boundary_conditions.fixed_coordinates
        fixed_values = boundary_conditions.fixed_values
        loaded_coords = boundary_conditions.loaded_coordinates
        loaded_values = boundary_conditions.loaded_values
```

Pass loaded values to `write_parosol_input()`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/test_api.py tests/test_hdf5_io.py -q
PYTHONPATH=src python -m ruff check src tests
```

Commit:

```bash
git add src/parosol_py/api.py tests/test_api.py
git commit -m "feat: allow explicit boundary conditions in solve"
```

---

### Task 5: Built-In Profiles

**Files:**
- Create: `src/parosol_py/profiles.py`
- Modify: `src/parosol_py/config.py`
- Modify: `src/parosol_py/__init__.py`
- Test: `tests/test_profiles.py`
- Test: `tests/test_config_cli.py`

- [ ] **Step 1: Write failing profile tests**

Create `tests/test_profiles.py`:

```python
from parosol_py import get_output_profile, get_solver_profile


def test_builtin_profiles_are_available():
    solver = get_solver_profile("legacy_axial")
    output = get_output_profile("quick_summary")

    assert solver.tolerance == 1e-6
    assert "sed" in solver.outputs
    assert output.export_fields is False
    assert output.image_fields == ()
```

Append to `tests/test_config_cli.py`:

```python
def test_run_case_config_applies_named_profiles(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "solver_profile": "legacy_axial",
                "output_profile": "quick_summary",
                "output": {"summary": "summary.json"},
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_solve(**kwargs):
        captured.update(kwargs)
        from parosol_py.api import SolveResult, SolveSummary

        return SolveResult(
            input_file=tmp_path / "input.h5",
            command=["parosol"],
            fields={},
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    assert captured["outputs"] == ("sed",)
    assert captured["export_dir"] is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_profiles.py tests/test_config_cli.py::test_run_case_config_applies_named_profiles -q
```

Expected: import failure for profile helpers or assertion failure.

- [ ] **Step 3: Implement profile registry and config application**

Create `src/parosol_py/profiles.py`:

```python
from __future__ import annotations

from .core import OutputProfile, SolverProfile

SOLVER_PROFILES = {
    "legacy_axial": SolverProfile(tolerance=1e-6, level=6, mpi_processes=1, outputs=("sed",)),
    "batch": SolverProfile(tolerance=1e-6, level=6, mpi_processes=6, outputs=("sed",)),
    "debug": SolverProfile(
        tolerance=1e-6,
        level=6,
        mpi_processes=1,
        outputs=("sed", "effective_strain", "von_mises"),
    ),
}

OUTPUT_PROFILES = {
    "quick_summary": OutputProfile(export_fields=False, image_fields=()),
    "standard_fields": OutputProfile(export_fields=True, image_fields=("sed",)),
    "debug": OutputProfile(export_fields=True, image_fields=("sed", "effective_strain", "von_mises")),
}


def get_solver_profile(name: str | None) -> SolverProfile:
    if name is None:
        return SolverProfile()
    try:
        return SOLVER_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown solver profile: {name}") from exc


def get_output_profile(name: str | None) -> OutputProfile:
    if name is None:
        return OutputProfile()
    try:
        return OUTPUT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown output profile: {name}") from exc
```

In `config.py`, import helpers and apply profile defaults before section overrides:

```python
from .profiles import get_output_profile, get_solver_profile
```

Inside `run_case_config()`:

```python
    solver_profile = get_solver_profile(config.get("solver_profile"))
    output_profile = get_output_profile(config.get("output_profile"))
```

Use profile defaults:

```python
    outputs = tuple(str(v) for v in solver_cfg.get("outputs", solver_profile.outputs))
    export_fields = bool(output_cfg.get("export_fields", output_profile.export_fields))
```

Use solver profile defaults for tolerance, level, and MPI.

Modify `__init__.py` to export `get_solver_profile` and `get_output_profile`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/test_profiles.py tests/test_config_cli.py -q
PYTHONPATH=src python -m ruff check src tests
```

Commit:

```bash
git add src/parosol_py/profiles.py src/parosol_py/config.py src/parosol_py/__init__.py tests/test_profiles.py tests/test_config_cli.py
git commit -m "feat: add solver and output profiles"
```

---

### Task 6: Fast Field Export Module

**Files:**
- Create: `src/parosol_py/field_export.py`
- Modify: `src/parosol_py/api.py`
- Test: `tests/test_field_export.py`

- [ ] **Step 1: Write failing tests for cached native-to-dense mapping**

Create `tests/test_field_export.py`:

```python
import numpy as np

from parosol_py.field_export import NativeFieldMapper


def test_native_field_mapper_maps_active_values_to_dense_xyz():
    stiffness = np.zeros((3, 2, 1), dtype=np.float32)
    stiffness[0, 1, 0] = 1
    stiffness[1, 1, 0] = 1
    stiffness[2, 0, 0] = 1
    mapper = NativeFieldMapper(stiffness)

    dense = mapper.scalar_to_dense(np.array([21.0, 31.0, 40.0], dtype=np.float32))

    assert dense.shape == (3, 2, 1)
    assert dense[0, 1, 0] == 21.0
    assert dense[1, 1, 0] == 31.0
    assert dense[2, 0, 0] == 40.0


def test_native_field_mapper_reuses_coordinate_arrays():
    stiffness = np.ones((2, 2, 2), dtype=np.float32)
    mapper = NativeFieldMapper(stiffness)

    first = mapper.active_coordinates
    second = mapper.active_coordinates

    assert first is second
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_field_export.py -q
```

Expected: missing module import failure.

- [ ] **Step 3: Implement mapper and route API export through it**

Create `src/parosol_py/field_export.py` with a `NativeFieldMapper` dataclass that caches sorted active/dense coordinates and provides `scalar_to_dense(values)`.

Move the current `_morton_key` and `_native_scalar_to_dense_xyz` logic from `api.py` into this module. Preserve behavior for active-length and dense-length fields.

In `api.py`, replace direct `_native_scalar_to_dense_xyz()` calls with:

```python
mapper = NativeFieldMapper(stiffness_gpa_xyz)
...
array_xyz=mapper.scalar_to_dense(field_array)
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/test_field_export.py tests/test_api.py -q
PYTHONPATH=src python -m ruff check src tests
```

Commit:

```bash
git add src/parosol_py/field_export.py src/parosol_py/api.py tests/test_field_export.py
git commit -m "feat: add cached native field mapper"
```

---

### Task 7: legacy solver Validation Manifest and Comparison

**Files:**
- Create: `src/parosol_py/reference_validation.py`
- Modify: `src/parosol_py/cli.py`
- Test: `tests/test_reference_validation.py`

- [ ] **Step 1: Write failing validation tests**

Create `tests/test_reference_validation.py`:

```python
from pathlib import Path

from parosol_py.reference_validation import ReferenceCase, compare_pistoia_summary, discover_reference_cases


def test_discover_reference_cases_finds_local_reference_set():
    cases = discover_reference_cases(Path("/Users/matthias.walle/Documents/10_Data/fea_test"))

    names = {case.name for case in cases}
    assert "VITD_0003_RL_M06_HOM_LS" in names
    assert "CABHS_5001_RL_V1_HOM_LS" in names


def test_compare_pistoia_summary_reports_relative_errors():
    case = ReferenceCase(
        name="sample",
        aim_path=Path("sample.AIM"),
        analysis_path=Path("sample_analysis.txt"),
        pistoia_path=Path("sample_pistoia.txt"),
        critical_volume_percent=2.0,
        critical_strain=0.007,
    )
    parosol = {
        "failure": {
            "factor": 0.5,
            "ees_at_critical_volume": 0.014,
            "failure_load": {"z": -100.0},
        },
        "mechanics": {"stiffness": {"z": 1000.0}, "reaction_force": {"z": -200.0}},
    }
    reference = {
        "factor": 0.4,
        "ees_at_critical_volume": 0.01,
        "failure_load": {"fz": -80.0},
        "axial_stiffness": {"z": 800.0},
        "reaction_force_node_set_1": {"fz": -160.0},
    }

    comparison = compare_pistoia_summary(case, parosol, reference)

    assert comparison["factor"]["absolute_error"] == 0.1
    assert comparison["failure_load_z"]["relative_error"] == 0.25
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_reference_validation.py -q
```

Expected: missing module import failure.

- [ ] **Step 3: Implement validation helpers**

Create `src/parosol_py/reference_validation.py` with:

- `ReferenceCase` dataclass;
- `discover_reference_cases(root)` that pairs `*.AIM`, `*_analysis.txt`, and `*_pistoia.txt`;
- critical volume parsed from pistoia file using existing parser;
- `compare_pistoia_summary(case, parosol_summary, reference_pistoia)` returning absolute and relative errors for factor, EES critical volume, failure load z, stiffness z, and reaction force z.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
PYTHONPATH=src pytest tests/test_reference_validation.py -q
PYTHONPATH=src python -m ruff check src tests
```

Commit:

```bash
git add src/parosol_py/reference_validation.py tests/test_reference_validation.py
git commit -m "feat: add reference validation manifest helpers"
```

---

## Self-Review

- Spec coverage: Tasks cover core objects, explicit BCs, loaded node force support, load cases, profiles, fast field mapping, and legacy solver validation scaffolding.
- Known deferred item: full Slicer integration remains a file-format/API design path, not implemented in this first plan.
- Known deferred item: bending and curved surface projection are named in load case design but not implemented beyond API direction; this plan implements axial, body-weight force compression, and simple shear first.
- Placeholder scan: no task uses `TBD` or an undefined implementation step.
- Type consistency: `Model`, `BoundaryConditionSet`, `SolverProfile`, `OutputProfile`, `AxialCompression`, `BodyWeightCompression`, and `SimpleShear` are defined before downstream use.

