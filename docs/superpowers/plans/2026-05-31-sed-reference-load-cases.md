# SED Reference and Load Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a field-wise TRAB_1240 SED regression fixture and expose additional simple load cases through the clean config/API layer.

**Architecture:** Keep reference data in compressed NumPy fixtures, using dense image-aligned SED for portability and extracting active values in the same order as ParOSol diagnostics. Add load-case config mapping in `config.py` that delegates to existing focused load-case classes, with one new confined-compression class where needed.

**Tech Stack:** Python, NumPy, h5py for one-time legacy reference extraction, pytest, ParOSol HDF5 runner.

---

### Task 1: TRAB_1240 SED Reference Fixture

**Files:**
- Create: `tests/fixtures/trab1240/reference_sed.npz`
- Modify: `tests/test_trab1240_reference.py`

- [x] **Step 1: Generate dense SED fixture**

Use `/tmp/parosol_trab1240_reference/trab1240.n88model` to read `Solutions/Solution1/ElementValues/StrainEnergyDensity`, map each element center into the connected-component filtered TRAB_1240 grid, and save `sed_zyx` plus metadata to `tests/fixtures/trab1240/reference_sed.npz`.

- [x] **Step 2: Add failing field-wise SED test**

Add an assertion in the opt-in TRAB_1240 solver test that loads `reference_sed.npz`, extracts active values in ParOSol element order, and checks relative L2 error, mean, p95, p98, and max.

- [x] **Step 3: Run the opt-in test and verify it fails if the fixture is missing or mapping is wrong**

Run: `PAROSOL_RUN_REFERENCE_TESTS=1 PAROSOL_REFERENCE_MPI=6 pytest tests/test_trab1240_reference.py -q`

- [x] **Step 4: Fix the fixture/test mapping until the field comparison passes**

The comparison should pass with tight but realistic tolerances around `5e-4` relative for aggregate metrics and `1e-3` for relative L2 error.

### Task 2: Load Case Config Mapping

**Files:**
- Modify: `src/parosol_py/load_cases.py`
- Modify: `src/parosol_py/config.py`
- Modify: `tests/test_load_cases.py`
- Modify: `tests/test_config_cli.py`
- Modify: `src/parosol_py/config_templates/default.yaml`

- [x] **Step 1: Add failing tests for config load cases**

Add tests that `load_case.type: shear`, `load_case.type: body_weight`, and `load_case.type: confined` pass explicit `BoundaryConditionSet` objects into `solve`.

- [x] **Step 2: Add `ConfinedCompression`**

Implement a class that fixes all displacement degrees on the bottom surface, prescribes axial displacement on the top surface, and fixes lateral degrees on both top and bottom surfaces.

- [x] **Step 3: Route config load-case types**

In `_boundary_conditions_from_config`, construct `SimpleShear`, `BodyWeightCompression`, or `ConfinedCompression` for their respective `load_case.type` values when no user nodesets are supplied.

- [x] **Step 4: Update config template**

Document concise examples for axial, shear, body-weight force, confined compression, and custom nodeset loading.

### Task 3: Manual-Derived Roadmap

**Files:**
- Create: `docs/legacy-feature-map.md`

- [x] **Step 1: Add feature map**

Summarize useful legacy manual concepts as modern ParOSol-py features: image inputs, material tables, connectivity, standard load cases, uneven/label-defined surfaces, force loads, derived fields, JSON summaries, field export, profiles, batch execution, and nonlinear features deferred out of scope.

- [x] **Step 2: Link from README**

Add one short pointer from `README.md` to the roadmap.
