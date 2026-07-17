# Final Fix Report: Public Solve Nonlinear Map Array Order

## Status

Fixed and verified.

## Scope

- `src/parosol_py/api.py`
- `tests/test_nonlinear_config.py`

## Change Summary

- Added a private `api.py` helper that converts nonlinear material maps from zyx to xyz when public `solve(..., array_order="zyx")` is used.
- The helper transposes `youngs_modulus_mpa`, `compressive_yield_mpa`, `tensile_yield_mpa`, `plateau_mpa`, `material_id`, and array-valued `poisson_ratio`.
- Scalar `poisson_ratio` is preserved.
- Scalar `VonMisesMaterial` behavior is preserved by returning non-map materials unchanged.
- Added a non-cubic direct `solve()` dry-run regression using `spine_keaveny_nonlinear` with zyx shape `(2, 3, 4)`, verifying HDF5 nonlinear map shape alignment with `Image_Data/Image` and `Image == YoungsModulusMPa / 1000` after reversing writer `swapaxes`.

## Verification

Red check before fix:

```text
conda run -n ogoloco-n88 pytest tests/test_nonlinear_config.py::test_solve_dry_run_transposes_zyx_keaveny_nonlinear_material_map -v
```

Result: failed with `ValueError: nonlinear material dataset YoungsModulusMPa must match stiffness shape (4, 3, 2), got (2, 3, 4)`.

Green checks after fix:

```text
conda run -n ogoloco-n88 pytest tests/test_nonlinear_config.py::test_solve_dry_run_transposes_zyx_keaveny_nonlinear_material_map -v
```

Result: 1 passed, 3 warnings.

```text
conda run -n ogoloco-n88 pytest tests/test_nonlinear_config.py tests/test_api.py -k nonlinear -v
```

Result: 35 passed, 11 deselected, 3 warnings.

Warnings were existing SWIG deprecation warnings from imported dependencies.

## Concerns

No native C++ was edited. The public direct solve path now duplicates the config helper behavior locally to avoid importing config workflow helpers into `api.py`.

## Commit

Included in the final fix commit reported by Codex.
