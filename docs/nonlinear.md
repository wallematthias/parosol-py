# Native Nonlinear Solver

ParOSol supports an experimental small-strain native nonlinear solve for
`VonMisesIsotropic` perfect plasticity. The implementation stores plastic strain
state and repeatedly solves the linearized equilibrium problem with the existing
PCG solver. Native tiny-cube smoke tests validate the current implementation.

## Supported

- Isotropic voxel grids
- `VonMisesIsotropic`
- Displacement-controlled axial tests
- Per-element plastic strain output
- Optional load-history wrapper

## Single Solve

```python
import numpy as np

from parosol_py import solve
from parosol_py.nonlinear import NonlinearSolverOptions, VonMisesMaterial

result = solve(
    material=np.full((3, 3, 3), 6829.0, dtype=np.float32),
    spacing=(1.0, 1.0, 1.0),
    strain=-0.05,
    test="axial",
    load_case_type="constrained_axial",
    outputs=("plastic_strain",),
    nonlinear_material=VonMisesMaterial(6829.0, 0.3, 50.0),
    nonlinear_solver=NonlinearSolverOptions(
        convergence_tolerance=1.0e-6,
        maximum_plastic_iterations=50,
    ),
)

print(result.diagnostics["nonlinear"])
```

## Load History

Use `run_nonlinear_load_history` to run repeated displacement-controlled solves
across a monotonic strain history:

```python
import numpy as np

from parosol_py.nonlinear import NonlinearSolverOptions, VonMisesMaterial
from parosol_py.nonlinear_workflow import run_nonlinear_load_history

history = run_nonlinear_load_history(
    material=np.full((3, 3, 3), 6829.0, dtype=np.float32),
    spacing=(1.0, 1.0, 1.0),
    final_strain=-0.05,
    steps=5,
    load_case_type="constrained_axial",
    nonlinear_material=VonMisesMaterial(6829.0, 0.3, 50.0),
    nonlinear_solver=NonlinearSolverOptions(convergence_tolerance=1.0e-6),
)

print(history.points[-1].diagnostics["nonlinear"])
```

Each history point records the applied strain, generalized load, reaction force,
plastic iteration count, and the underlying `SolveResult`.

## Not Supported Yet

- Mohr-Coulomb
- Maximum principal strain
- Geometric nonlinearity
- Contact
- Bit-for-bit parity with other nonlinear implementations

## Validation

Run the focused native smoke test with:

```bash
python -m pytest tests/test_nonlinear_solver_smoke.py -v
```

The test verifies plastic-state output, yielded-element count, convergence at or
below the configured tolerance, and convergence before the iteration cap.

## Known Limitations

This solver is experimental and currently intended for the supported small,
displacement-controlled voxel cases above. Plasticity is idealized as isotropic
von Mises perfect plasticity, so there is no hardening model. Do not use it for
contact, large-deformation behavior, or unsupported failure criteria.
