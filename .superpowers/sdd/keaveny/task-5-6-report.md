# Task 5-6 Report: Native Asymmetric Perfect Plastic Nonlinear Solve

## Summary

Implemented the native `AsymmetricPerfectPlasticDensityMap` solve path alongside the existing scalar `VonMisesIsotropic` path.

## Changes

- Added `AsymmetricMaterialProperties` and `AsymmetricPerfectPlasticMaterial`.
- Implemented a first asymmetric perfect-plastic update using principal stress limits:
  - tensile yield when max principal stress exceeds `sigma_t`
  - compressive yield when negative min principal stress exceeds `sigma_c`
  - no tensile/compressive averaging
- Capped yielded principal stresses with zero hardening and stored plastic strain as `total_strain - D^-1 * capped_stress`, preserving elastic unloading through the accumulated plastic strain state.
- Added a map-mode `NonlinearProblem` constructor that reorders local dense map arrays into active octree element order by decoding each active element Morton key.
- Routed `main.cpp` to construct `NonlinearProblem` for both `VonMisesIsotropic` and `AsymmetricPerfectPlasticDensityMap`.
- Added native smoke coverage for:
  - asymmetric tension/compression with `E=1000 MPa`, `nu=0.3`, `sigma_t=5 MPa`, `sigma_c=20 MPa`, `plateau=20 MPa`
  - two-material cube where low-strength voxels accumulate plastic strain while high-strength voxels remain elastic
  - native rejection of mismatched map `YoungsModulusMPa` versus `/Image_Data/Image` stiffness
  - native rejection of invalid active map values for Poisson ratio, yield stresses, and plateau stress
  - accepted `plateau != sigma_c` behavior, where `sigma_c` controls compression yield onset and `plateau` controls the post-yield compressive cap

## Extra Helper Change

One small change was required outside the initial ownership list in `src/parosol_native/src/HDF5Image.cpp`: the native map reader now reads float map datasets through float buffers and copies into double arrays, and reads `MaterialID` through a signed-short buffer before copying to `unsigned short`. This avoids the existing generic HDF5 reader choosing the wrong native type for `unsigned short`.

## Limitations / Concerns

- Matrix stiffness still comes from `/Image_Data/Image`; the asymmetric constitutive update uses map `E` and `nu`. Native loading now rejects active voxels where `/Nonlinear/YoungsModulusMPa / 1000.0` does not match `/Image_Data/Image` stiffness.
- The two-material smoke test can assert exported final plastic strain localization, which is the strongest current-output assertion available. Exact first-yield iteration ordering per voxel is deferred because current native outputs expose final averaged element plastic strain and `yielded_last`, but not a per-iteration/per-region yield history.

## Verification

- `conda run -n ogoloco-n88 python -m pip install -e .`
- `conda run -n ogoloco-n88 pytest tests/test_nonlinear_solver_smoke.py -v`

Result: 10 passed, 3 warnings.
