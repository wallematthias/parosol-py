### Task 4 Report: Native HDF5 Reader For Material Map Datasets

**Status:** Complete.

**Files changed:**
- `src/parosol_native/src/HDF5Image.h`
- `src/parosol_native/src/HDF5Image.cpp`
- `tests/test_nonlinear_solver_smoke.py`

**Implementation notes:**
- Added native material-map storage fields for:
  - `nonlinear_map_E_mpa`
  - `nonlinear_map_nu`
  - `nonlinear_map_sigma_c_mpa`
  - `nonlinear_map_sigma_t_mpa`
  - `nonlinear_map_plateau_mpa`
  - `nonlinear_map_material_id`
- Preserved scalar `VonMisesIsotropic` validation/read behavior by keeping scalar attribute validation in the non-map branch.
- Added `AsymmetricPerfectPlasticDensityMap` handling in `HDF5Image::Scan`.
- Required all six `/Nonlinear` datasets:
  - `YoungsModulusMPa`
  - `PoissonRatio`
  - `CompressiveYieldStressMPa`
  - `TensileYieldStressMPa`
  - `PlateauStressMPa`
  - `MaterialID`
- Validated each map dataset shape against the HDF5 `Image` dimensions.
- Read valid map datasets with the same `my_offset`/`my_count` distributed layout used for `Image`.
- Added destructor cleanup for the new native arrays.

**Test evidence:**
- Red check before implementation:
  - `conda run -n ogoloco-n88 pytest tests/test_nonlinear_solver_smoke.py::test_asymmetric_density_map_requires_all_datasets -v`
  - Failed because native still reported scalar-attribute validation errors instead of `missing TensileYieldStressMPa`.
- Green focused check after implementation:
  - `conda run -n ogoloco-n88 pytest tests/test_nonlinear_solver_smoke.py::test_asymmetric_density_map_requires_all_datasets -v`
  - Passed: `1 passed, 3 warnings`.
- Scalar regression check:
  - `conda run -n ogoloco-n88 pytest tests/test_nonlinear_solver_smoke.py::test_native_rejects_invalid_nonlinear_hdf5_config -v`
  - Passed: `1 passed, 3 warnings`.
- Whitespace check:
  - `git diff --check`
  - Passed with no output.

**Scope note:**
- `main.cpp` was not edited. Valid `AsymmetricPerfectPlasticDensityMap` solve support remains intentionally blocked by the existing non-`VonMisesIsotropic` rejection in `main.cpp`, as allowed by the task brief.

---

### Review Finding Fix: Material Map Dataset Rank Validation

**Status:** Complete.

**Files changed:**
- `src/parosol_native/src/HDF5Image.cpp`
- `tests/test_nonlinear_solver_smoke.py`

**Implementation notes:**
- Added rank-aware validation for each required `AsymmetricPerfectPlasticDensityMap` dataset under `/Nonlinear`.
- Required each map dataset to have rank exactly 3 and dimensions equal to the `Image` dataset dimensions.
- Zero-initialized HDF5 dimension buffers before dataset-size calls touched in `HDF5Image.cpp`.
- Converted map dataset read failures into the existing `nonlinear_config_error` path with `failed to read <dataset>` messages.
- Preserved scalar `VonMisesIsotropic` behavior by leaving scalar validation in the existing non-map branch.

**Regression test:**
- Added `test_asymmetric_density_map_rejects_rank_two_dataset`, which replaces `TensileYieldStressMPa` with a rank-2 `(3, 3)` dataset and asserts the native executable returns nonzero with `invalid nonlinear configuration` and `TensileYieldStressMPa rank must be 3`, without falling through to the unsupported material-type message.

**Test evidence:**
- Red check before native fix:
  - `conda run -n ogoloco-n88 pytest tests/test_nonlinear_solver_smoke.py::test_asymmetric_density_map_rejects_rank_two_dataset -v`
  - Failed because native returned `ERROR: only VonMisesIsotropic nonlinear material is currently supported`.
- Rebuilt editable package:
  - `conda run -n ogoloco-n88 python -m pip install -e .`
  - Succeeded.
- Green focused check:
  - `conda run -n ogoloco-n88 pytest tests/test_nonlinear_solver_smoke.py::test_asymmetric_density_map_rejects_rank_two_dataset -v`
  - Passed: `1 passed, 3 warnings`.
- Requested smoke checks plus new malformed-rank test:
  - `conda run -n ogoloco-n88 pytest tests/test_nonlinear_solver_smoke.py::test_asymmetric_density_map_requires_all_datasets tests/test_nonlinear_solver_smoke.py::test_native_rejects_invalid_nonlinear_hdf5_config tests/test_nonlinear_solver_smoke.py::test_asymmetric_density_map_rejects_rank_two_dataset -v`
  - Passed: `3 passed, 3 warnings`.
- Whitespace check:
  - `git diff --check -- src/parosol_native/src/HDF5Image.cpp tests/test_nonlinear_solver_smoke.py`
  - Passed with no output.

**Concerns:**
- None beyond the existing intentional `main.cpp` guard that still rejects valid `AsymmetricPerfectPlasticDensityMap` solves until that material type is implemented.
