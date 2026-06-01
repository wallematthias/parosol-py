# Anatomic FE Modeling Layer Design

## Goal

Add a `model` layer to `parosol-py` that can build solver-ready voxel FE models from anatomy-specific clinical inputs before running the ParOSol solver. The first anatomy targets are spine compression and proximal femur sideways fall, using the stable concepts from OgoLoco without depending on OgoLoco as a runtime package.

## Context From OgoLoco

The updated OgoLoco `main` branch keeps FE generation in `src/ogoloco/plugins/simulators/model_generation.py` and delegates legacy model creation to `src/ogoloco/vendor/ogo/cli/ref/SpineCompressionFe.py` and `SidewaysFallFe.py`.

The spine workflow reads a calibrated density image plus a body/process mask, crops and aligns the vertebra, generates or uses a cortical mask, converts density to material bins, pads the model, extrudes PMMA disks, derives top/bottom node sets, and writes an `.n88model`. The current OgoLoco version improves disk stability by extruding raytraced body-cap disks from the body mask rather than only the boundary label image.

The femur workflow reads a calibrated density image plus a femur mask, resamples to isotropic spacing, pre-rotates left/right femurs, aligns to a reference femur, converts density to material bins, generates PMMA caps for the femoral head and greater trochanter, derives femoral-head/trochanter/distal node sets, and applies the sideways-fall boundary conditions.

## Proposed Configuration

`model` is a top-level section because these steps define the FE model geometry and boundary regions. They are not generic image cleanup.

```yaml
model:
  type: spine_compression
  density_image: density.nii.gz
  mask_image: spine_body_process.nii.gz
  labels:
    body: 2
    process: 1
  geometry:
    isotropic_spacing: 1.0
    reference: built_in
    pmma_thickness_mm: 3
    disk_method: raytrace_body_cap
  outputs:
    material_image: model/material.nii.gz
    nodeset_image: model/nodesets.nii.gz
    manifest: model/model.json
    qc_image: model/qc.png
```

```yaml
model:
  type: proximal_femur_sideways_fall
  density_image: hip_density.nii.gz
  mask_image: femur_mask.nii.gz
  side: left
  geometry:
    isotropic_spacing: 1.0
    reference: built_in
    pmma_thickness_mm: 3
    cap_method: pmma_contact_caps
  outputs:
    material_image: model/material.nii.gz
    nodeset_image: model/nodesets.nii.gz
    manifest: model/model.json
    qc_image: model/qc.png
```

Existing direct material-label workflows remain valid by treating them as `model.type: direct_voxel` internally.

## Python API

The public API should make model generation explicit and reusable:

```python
from parosol_py.modeling import build_model
from parosol_py import solve

built = build_model(config["model"], base_dir=config_path.parent)
result = solve(
    material=built.material,
    spacing=built.spacing,
    origin=built.origin,
    poisson_ratio=built.poisson_ratio,
    boundary_conditions=built.boundary_conditions,
    outputs=("sed",),
)
```

`BuiltModel` should contain:

- `material`: dense material/stiffness image in z-y-x order for the existing solver path.
- `spacing` and `origin`: physical metadata after resampling/model construction.
- `boundary_conditions`: a `BoundaryConditionSet` built from generated node sets.
- `materials`: resolved material definitions, including PMMA.
- `node_sets`: named node sets for QC/export.
- `element_sets`: named material/body/process/disk/cap sets for QC/export.
- `exported`: paths to model artifacts.
- `metadata`: anatomy, labels, model generation parameters, warnings, and source paths.

## Architecture

Create `src/parosol_py/modeling/` with small modules:

- `types.py`: `BuiltModel`, warnings, metadata dataclasses.
- `builder.py`: `build_model()` dispatcher and path resolution.
- `direct.py`: adapter for current material-image workflows.
- `density.py`: density preprocessing and density-to-modulus/bin helpers shared by spine/femur.
- `spine.py`: spine compression model builder.
- `femur.py`: proximal femur sideways-fall model builder.
- `caps.py`: PMMA disk/cap image generation.
- `surfaces.py`: voxel surface and node-set selection helpers.
- `exports.py`: model manifest, model image, nodeset image, and QC image exports.

The solver stays anatomy-agnostic. `run_case_config()` should first call the model builder when `model` is present, then pass the resulting material image and boundary conditions into the existing `solve()` path.

## Spine Builder

Inputs:

- Density image.
- Body/process label image or separate body/process masks.
- Optional cortical mask.
- Optional reference path. For the first implementation, an explicit reference path is required for reference-aligned clinical models; synthetic tests may use reference-free geometry.

Outputs:

- Material image with bone and PMMA regions.
- Node-set labels for inferior and superior PMMA contact surfaces.
- Element-set labels for trabecular body, cortical body, trabecular process, cortical process, inferior disk, superior disk.
- QC image showing density, segmentation, material model, disks, and BC nodes.

Initial implementation should port the OgoLoco raytrace body-cap disk idea and avoid writing `.n88model`.

## Femur Builder

Inputs:

- Density image.
- Left or right femur mask.
- Side: `left` or `right`.
- Optional reference path.

Outputs:

- Material image with bone and PMMA caps.
- Node-set labels for femoral head PMMA, greater trochanter PMMA, and distal femur.
- Element-set labels for bone, femoral head cap, greater trochanter cap.
- QC image showing alignment, caps, and BC nodes.

Initial implementation should reproduce the OgoLoco sideways-fall boundary-condition intent with ParOSol displacement BCs.

## Error Handling

Model builders should fail before solving if:

- The input image and mask cannot be read.
- Image and mask geometry cannot be reconciled.
- Required labels are absent.
- The model has no active material voxels.
- Required generated node sets are empty.
- PMMA disk/cap thickness rounds to zero voxels.
- The generated spacing is anisotropic in a way the current ParOSol writer cannot represent.

Warnings should be stored in the model manifest and summary JSON for recoverable issues such as small node sets, clipped caps, or fallback surface detection.

## Testing

Start with synthetic tests rather than full clinical regression cases:

- Direct model builder preserves current material-image behavior.
- Spine builder on a small labeled block generates two PMMA disks and non-empty superior/inferior node sets.
- Femur builder on a small synthetic femur-like mask generates three named node sets.
- `run_case_config()` uses `model` output when present.
- Model artifacts are written deterministically: `material.nii.gz`, `nodesets.nii.gz`, `model.json`, `qc.png`.

Clinical verification should follow with OgoLoco-derived spine/femur cases once reference inputs and expected stiffness/failure values are selected.

## Out Of Scope For First Slice

- Nonlinear/progressive damage.
- Native material-specific Poisson ratio in ParOSol.
- Full Slicer visual authoring of BCs.
- Replacing every OgoLoco FE recipe in one pass.
- Writing `.n88model`.
