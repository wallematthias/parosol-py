# Legacy Feature Map

This map translates useful concepts from the Numerics88 bone FEA workflow into
the cleaner ParOSol-py API/config vocabulary. The goal is compatibility of
scientific outputs, not a one-to-one clone of the old command-line interface.

## Implemented Core

| Legacy concept | ParOSol-py equivalent | Status |
| --- | --- | --- |
| Segmented image input with material IDs | `input.image` plus `image_type: material_labels` | Implemented for AIM, NIfTI, MetaImage, NPY, NPZ |
| Material definition/table files | `materials.file` with `MaterialDefinitions` and `MaterialTable` | Implemented for linear isotropic E/nu tables |
| Continuous density input | `input.image_type: density` plus `materials.density` equation | Implemented for power, linear, and polynomial E mappings |
| Poisson ratio from equation | `materials.poisson_ratio` scalar or equation reduced to one value | Implemented; native solve still uses one global value |
| Connectivity filtering | `preprocessing.connectivity_filter: true` | Implemented as largest non-zero component |
| Constrained axial plate compression | `load_case.type: constrained_axial` or `plate_compression` | Implemented |
| Uniaxial compression | `load_case.type: uniaxial` | Implemented |
| Confined compression | `load_case.type: confined` | Implemented |
| Absolute compression displacement | `load_case.displacement` or `normal_displacement` | Implemented |
| Directional shear | `load_case.type: shear` with `axis` and `direction`; common profiles `shear_zx` and `shear_zy` | Implemented |
| Two-component shear vector | `load_case.shear_vector: [x, y]` for z-normal shear | Implemented |
| Force-driven compression | `load_case.type: body_weight` with `force_n` | Implemented |
| Custom node sets | `nodesets` label image plus `load_case.type: nodeset` | Implemented |
| Pistoia failure estimate | `failure.criterion: pistoia` in summary JSON | Implemented |
| Derived SED field | `solver.outputs: [sed]` and optional `.nii.gz` export | Implemented |
| Compact analysis output | `output.summary` JSON | Implemented |
| Boundary-condition debug export | `output.export_boundary_conditions: true` | Implemented as JSON |
| Visible/uneven top-bottom surfaces | `load_case.surface: {mode: smart, depth: auto}` | Implemented |
| Bending tests | `load_case.type: bending` with axis, neutral axis, and angle | Boundary conditions implemented; moment summary implemented |
| Torsion tests | `load_case.type: torsion` with axis and twist angle | Boundary conditions implemented; torque summary implemented |

## Next High-Value Features

| Legacy concept | Proposed ParOSol-py equivalent | Notes |
| --- | --- | --- |
| Direct mechanics batches | `profiles/direct_mechanics_manifest.yaml` plus `parosol batch` | Implemented for x/y/z compression and z-normal shear |
| Export node/element sets | `.vtp`/`.vtk` or `.json` debug exports for selected node sets | Useful before Slicer integration |
| Field export selection | `output.fields: [sed, von_mises, effective_strain]` | Mostly present, needs profile polishing |
| Solution quality report | Residual/iteration/runtime plus optional checks | Summary already has core solver values |
| Coarsen/interpolate workflow | Downsampled solve plus interpolated full-resolution initial guess | Deferred optimization feature |
| Nonlinear/progressive loading | Separate nonlinear engine/profile | Out of first stable linear API scope |

## User-Facing Direction

The preferred workflow should stay small:

1. Provide a label image and material table.
2. Pick a named load case/profile or provide label-image node sets.
3. Run `parosol run case.yaml`, or `parosol batch batch.yaml` for multi-case mechanics.
4. Read `summary.json` and optional `.nii.gz` fields.

Advanced users should be able to build the same model through Python objects,
inspect generated boundary conditions, and batch over profiles without changing
the core solver API.
