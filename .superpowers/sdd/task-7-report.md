# Task 7 Report: Gauss-Point Plastic State and Output

## Scope

Implemented Task 7 from base commit `9593cc5e149ab3f8453ca1784ff5d3b5f8ad960a` on branch `parosol-nonlinear`.

- Replaced element-only plastic history updates with eight independent six-component plastic strain vectors per local element.
- Retained `/Solution/PlasticStrain` as the element-average dataset with shape `(global_elements, 6)`.
- Added flattened detailed state at `/Solution/GaussPoint8Values/PlasticStrain` with shape `(global_elements, 48)`. Each row is Gauss point 0 through 7, with six tensor components per point.
- Kept `parosol_py.results.read_solution_fields(..., outputs=("plastic_strain",))` unchanged, so the Python API continues to return `(n, 6)`.
- Added the necessary `main.cpp` output handoff because the new printer method requires the nonlinear problem's Gauss history.

## Implementation Notes

`NonlinearProblem` owns `_plastic_gauss` as `std::vector<Eigen::Matrix<double, 6, 1> >`, sized to `grid.GetNrElem() * 8` and initialized to zero. For every element and Gauss point, the material update receives that point's total strain and prior state, then stores the returned state back to the same point.

The element average is rebuilt from the eight updated point states for the existing `/Solution/PlasticStrain` output. `yielded_last` counts each element once if any point yielded. `plastic_convergence_last` is the MPI maximum of the absolute change across every updated Gauss-point component.

The existing HDF5 writer only writes rank-2 block datasets. The detailed data is therefore intentionally flattened to 48 columns rather than emitted as rank 3.

## RED Evidence

After adding the direct HDF5 assertions but before native implementation, ran:

```text
/Users/matthias.walle/miniforge3/envs/ogoloco-n88/bin/python -m pytest tests/test_nonlinear_solver_smoke.py::test_native_nonlinear_cube_writes_plastic_state_and_diagnostics -v
```

Result: failed as expected with `KeyError: Unable to synchronously open object (component not found)` for `Solution/GaussPoint8Values/PlasticStrain`. The pre-change Task 6 output had the element-average data but no detailed Gauss-point dataset.

The retained Task 6 RED-run file recorded `plastic_iterations = 43`, `yielded_last = 27`, and `plastic_convergence_last = 9.501534016659319e-07`.

## GREEN Evidence

Rebuilt the editable native package:

```text
/Users/matthias.walle/miniforge3/envs/ogoloco-n88/bin/python -m pip install -e . --no-build-isolation
```

Result: exit 0; editable wheel built and installed successfully.

Ran the required smoke suite:

```text
/Users/matthias.walle/miniforge3/envs/ogoloco-n88/bin/python -m pytest tests/test_nonlinear_solver_smoke.py -v
```

Result: `2 passed`.

Direct HDF5 inspection of the fresh nonlinear solve confirmed:

```text
/Solution/PlasticStrain: (27, 6), 27 rows with nonzero norm
/Solution/GaussPoint8Values/PlasticStrain: (27, 48), 27 rows with nonzero norm
```

## Iteration Count

The nonlinear iteration count changed from 43 in the retained Task 6 RED run to 2 with independent Gauss-point state. This follows the required convergence definition: the maximum absolute change now tracks individual point-state components rather than the element-averaged history.

## Excluded Scope

No external-solver references were added to tracked files.
