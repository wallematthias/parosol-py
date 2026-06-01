# Legacy Feature Map

This map translates useful concepts from the Numerics88 bone FEA workflow into
the cleaner ParOSol-py API/config vocabulary. The goal is compatibility of
scientific outputs, not a one-to-one clone of the old command-line interface.

## Implemented Core

| Legacy concept | ParOSol-py equivalent | Status |
| --- | --- | --- |
| Segmented image input with material IDs | `input.image` plus `image_type: material_labels` | Implemented for AIM, NIfTI, MetaImage, NPY, NPZ |
| Material definition/table files | Inline `materials.definitions` and `materials.table`, or optional `materials.file` | Implemented for linear isotropic E/nu tables |
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
| Export node/element sets | `output.export_sets: true` with `set_formats: [json, vtk]` | Implemented for node sets and material element sets |
| Field export selection | `output.fields: [sed, von_mises, effective_strain]` | Implemented |
| Solution quality report | `quality` section with residual/runtime/iteration checks | Implemented |
| Coarse preview solve | `preprocessing.coarsen` or `profiles/coarse_preview.yaml` | Implemented as downsampled solve grid |
| Progressive loading sequence | `profiles/progressive_loading_manifest.yaml` plus `parosol batch` | Implemented as linear load increments |
| Load-history estimation | `profiles/load_history_3.yaml`, `profiles/load_history_6.yaml`, and `parosol load-history` | Implemented as NNLS SED post-processing |
| Visible/uneven top-bottom surfaces | `load_case.surface: {mode: smart, depth: auto}` | Implemented |
| Bending tests | `load_case.type: bending` with axis, neutral axis, and angle | Boundary conditions implemented; moment summary implemented |
| Torsion tests | `load_case.type: torsion` with axis and twist angle | Boundary conditions implemented; torque summary implemented |

## Next High-Value Features

| Legacy concept | Proposed ParOSol-py equivalent | Notes |
| --- | --- | --- |
| Direct mechanics batches | `profiles/direct_mechanics_manifest.yaml` plus `parosol batch` | Implemented for x/y/z compression and z-normal shear |
| Full-resolution interpolation from coarse solve | Initial guess/reprojection API | Requires native solver support for restart/initial displacement fields |
| Nonlinear material behavior | Separate nonlinear engine/profile | Out of first stable linear API scope |

## User-Facing Direction

The preferred workflow should stay small:

1. Provide a label image and material table.
2. Pick a named load case/profile or provide label-image node sets.
3. Run `parosol run case.yaml`, or `parosol batch batch.yaml` for multi-case mechanics.
4. Read `summary.json` and optional `.nii.gz` fields.

Advanced users should be able to build the same model through Python objects,
inspect generated boundary conditions, and batch over profiles without changing
the core solver API.
