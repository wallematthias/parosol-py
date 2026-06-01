# GPU Backend Roadmap

ParOSol-py should keep the bundled native ParOSol solver as the validated CPU
reference backend. GPU work belongs in a separate `parosol_torch` namespace so
it can evolve without changing the scientific contract of `parosol_py`.

## Boundary

`parosol_py` owns the stable model-building API, material mapping, boundary
conditions, summaries, field export, batch execution, and validation helpers.
`src/parosol_native` remains the wrapped GPL ParOSol C++/MPI implementation.
`parosol_torch` is a separate optional package namespace for accelerator
experiments and, eventually, a validated GPU solver backend.

No GPU backend should be selected implicitly. Users should ask for it explicitly
through a future setting such as:

```yaml
solver:
  backend: torch
  device: mps
```

Until the GPU backend passes the reference suite, production profiles should
continue to use the native backend.

## Why Not a Quick MPS Port?

Apple MPS is accessible through torch tensor kernels, but ParOSol is not a
tensor program. Its performance comes from a matrix-free 8-node hexahedral
elasticity operator, MPI domain decomposition, and multilevel preconditioning.
A faithful GPU backend needs to reproduce the same operator, boundary-condition
treatment, post-processing fields, and convergence behavior.

## Proper Implementation Milestones

1. Define the backend contract.
   The backend input is a voxel material image, spacing, Poisson ratio policy,
   boundary conditions, requested fields, convergence settings, and output
   directory. The backend output is the existing `SolveResult` shape.

2. Extract reusable FE kernels.
   The 24x24 linear hexahedral element stiffness matrix and element
   stress/strain post-processing must be reproduced in a testable Python/C++ API
   independent of file IO.

3. Build a validated CPU prototype.
   A single-process matrix-free backend should match native ParOSol on tiny
   cubes before any GPU acceleration is trusted.

4. Add `parosol_torch` operator kernels.
   The first torch implementation should be matrix-free and batched over active
   elements. It should support `cpu`, `mps`, and `cuda` devices where available,
   but remain optional through `parosol-py[torch]`.

5. Add preconditioning deliberately.
   Plain CG is unlikely to be competitive for real HR-pQCT volumes. The GPU path
   needs a documented preconditioner strategy before it is considered useful for
   production-size FEAs.

6. Validate mechanics and fields.
   Acceptance requires matching native ParOSol/FAIM reference cases for
   stiffness, reaction force, Pistoia failure load, SED, and effective strain
   within predefined tolerances.

7. Benchmark honestly.
   Benchmark solve time, post-processing time, field export time, peak memory,
   and total wall time for representative small, medium, and large scans.

## Initial Package State

The repository now includes `src/parosol_torch` with runtime capability checks,
an explicit backend contract, a backend registry, a tiny scalar Poisson
prototype operator, and an experimental matrix-free 8-node hexahedral elasticity
solver. The registered `torch-experimental` backend supports explicit `cpu`,
`mps`, and `cuda` device requests through torch when available.

The prototype operator in `parosol_torch.prototype` remains only a CPU 7-point
scalar stencil for checking structured-grid indexing and boundary conventions.
The elasticity backend in `parosol_torch.elasticity` is the first real torch
operator path, but it is still experimental and not yet a validated replacement
for native ParOSol.

## Current Backend Contract

`VoxelElasticityProblem` describes the future solver input:

- `stiffness_gpa_xyz`: 3D xyz stiffness image.
- `voxel_size_mm`: isotropic voxel size.
- `poisson_ratio`: homogeneous Poisson ratio for the first milestone.
- `fixed_displacement_coordinates` and `fixed_displacement_values`: Dirichlet
  constraints using `(x, y, z, component)` rows.
- Optional loaded-node coordinates/values for force or displacement loading.
- Requested output fields such as `forces`, `displacements`, and eventually
  `sed` or strain-derived fields.

`SolverSettings` holds backend-independent controls: tolerance, optional maximum
iterations, optional device, and optional output directory. `VoxelElasticityResult`
is intentionally small until CPU parity defines the final field and diagnostics
shape.

The registry exposes only `torch-experimental`. It does not register native
ParOSol and it is not wired into `parosol_py.solve()` or the default `parosol`
CLI.

## Next Steps to a Real GPU Solver

1. CPU parity contract tests.
   Build tiny native-backed fixtures with known boundary conditions and expected
   result fields. Define tolerances for reaction force, displacement residuals,
   SED, and failure-load post-processing before implementing torch kernels.

2. Element kernel extraction.
   Reproduce the 8-node hexahedral stiffness action and stress/strain recovery
   in pure NumPy first. Tests should compare element-level forces and energies
   against native/reference outputs for one element, a two-element stack, and a
   small heterogeneous cube.

3. Matrix-free torch solver.
   The first matrix-free conjugate-gradient path now exists in
   `parosol_torch.elasticity`. It supports prescribed displacement constraints,
   homogeneous Poisson ratio, SED, nodal displacement, and nodal force outputs
   for small explicit problems. It stays marked experimental until it matches
   native references.

4. Native parity tests.
   Add tiny native-backed fixtures for one element, a two-element stack,
   heterogeneous materials, and standard constrained axial compression. Compare
   reaction forces, displacement fields, SED, and convergence diagnostics before
   using the torch backend for scientific results.

5. Preconditioning and memory strategy.
   Add documented preconditioning, active-voxel indexing, chunking decisions, and
   peak-memory checks before testing production-sized HR-pQCT scans.

6. Production integration gate.
   Only after validation and benchmarks should a non-default `parosol_py` backend
   selector be considered. The native backend must remain the default reference
   path.
