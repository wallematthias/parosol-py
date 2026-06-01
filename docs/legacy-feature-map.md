# Legacy Feature Map

This map translates useful concepts from the Numerics88 bone FEA workflow into
the cleaner ParOSol-py API/config vocabulary. The goal is compatibility of
scientific outputs, not a one-to-one clone of the old command-line interface.

## Implemented Core

| Legacy concept | ParOSol-py equivalent | Status |
| --- | --- | --- |
| Segmented image input with material IDs | `input.image` plus `image_type: material_labels` | Implemented for AIM, NIfTI, MetaImage, NPY, NPZ |
| Material definition/table files | `materials.file` with `MaterialDefinitions` and `MaterialTable` | Implemented for linear isotropic E/nu tables |
| Connectivity filtering | Store connected fixtures and add preprocessing helpers | Fixture implemented; helper API planned |
| Axial plate compression | `load_case.type: axial` | Implemented |
| Uniaxial compression | `load_case.type: uniaxial` | Implemented |
| Confined compression | `load_case.type: confined` | Implemented |
| Directional shear | `load_case.type: shear` with `axis` and `direction`; common profiles `shear_zx` and `shear_zy` | Implemented |
| Force-driven compression | `load_case.type: body_weight` with `force_n` | Implemented |
| Custom node sets | `nodesets` label image plus `load_case.type: nodeset` | Implemented |
| Pistoia failure estimate | `failure.criterion: pistoia` in summary JSON | Implemented |
| Derived SED field | `solver.outputs: [sed]` and optional `.nii.gz` export | Implemented |
| Compact analysis output | `output.summary` JSON | Implemented |

## Next High-Value Features

| Legacy concept | Proposed ParOSol-py equivalent | Notes |
| --- | --- | --- |
| Visible/uneven top-bottom surfaces | `surface: visible` or label-image plate helpers | Needed for curved plates and anatomy-aware BCs |
| Bending tests | `load_case.type: bending` with axis, neutral axis, angle/curvature | Needs careful torque and reaction summary definitions |
| Torsion tests | `load_case.type: torsion` with axis and angle | Similar to bending: must define summary moments |
| Direct mechanics batches | `profiles/direct_mechanics.yaml` generating x/y/z compression and shear cases | Good batch feature after load cases are stable |
| Export node/element sets | `.vtp`/`.vtk` or `.json` debug exports for selected node sets | Useful before Slicer integration |
| Field export selection | `output.fields: [sed, von_mises, effective_strain]` | Mostly present, needs profile polishing |
| Solution quality report | Residual/iteration/runtime plus optional checks | Summary already has core solver values |
| Coarsen/interpolate workflow | Downsampled solve plus interpolated full-resolution initial guess | Deferred optimization feature |
| Nonlinear/progressive loading | Separate nonlinear engine/profile | Out of first stable linear API scope |

## User-Facing Direction

The preferred workflow should stay small:

1. Provide a label image and material table.
2. Pick a named load case/profile or provide label-image node sets.
3. Run `parosol run case.yaml`.
4. Read `summary.json` and optional `.nii.gz` fields.

Advanced users should be able to build the same model through Python objects,
inspect generated boundary conditions, and batch over profiles without changing
the core solver API.
