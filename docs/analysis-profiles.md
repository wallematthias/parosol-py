# Analysis Profiles and Installation

ParOSol-py is a profile-driven finite-element analysis tool for voxel bone
models. The usual workflow is:

1. Start from a material label image or a density image plus segmentation mask.
2. Select a built-in profile.
3. Run `parosol`.
4. Read `summary.json`, optional field images, and the overview/debug figure.

The profiles are YAML snippets stored in
`src/parosol_py/config_templates/profiles`. They can be used through the
shortcut CLI or printed and edited as full config files.

## Installation

### Recommended: Install a Prebuilt Wheel

The easiest installation should be from a wheel built by the GitHub Actions
matrix. Wheels include the native ParOSol executable, so users do not need CMake,
MPI, HDF5, Eigen, or a compiler on their workstation.

Once a wheel index or release artifact is available:

```bash
python -m pip install parosol-py
```

For a private package index, use the authenticated index URL provided by the
project:

```bash
python -m pip install parosol-py --extra-index-url https://<private-index>
```

For a private GitHub Release wheel:

```bash
python -m pip install https://github.com/<owner>/parosol-py/releases/download/v0.1.0/parosol_py-<version>-<tags>.whl
```

The repository currently builds wheel artifacts for:

- Linux x86_64
- Windows AMD64
- macOS arm64
- macOS x86_64

The repository can stay private while wheels are distributed through GitHub
Releases, GitHub Packages, a private package index, or later public PyPI.

### Developer Install from Source

Source installs compile the native solver and are meant for development:

```bash
git clone git@github.com:wallematthias/parosol-py.git
cd parosol-py
python -m pip install -e .[dev]
```

Native build requirements for source installs:

- CMake
- MPI C++ compiler/runtime
- HDF5 C++ libraries
- Eigen headers

If CMake cannot find dependencies, make sure `mpicxx`, HDF5, and Eigen are
available on `PATH`/`CMAKE_PREFIX_PATH`.

## CLI Basics

The shortcut form runs one image with one profile:

```bash
parosol IMAGE --profile PROFILE --output OUTPUT_DIR
```

Examples:

```bash
parosol distal-radius.AIM --profile XtremeCTII --output runs/distal-radius

parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile vertebra \
  --output runs/10001_vertebra

parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile proximal_femur_sideways_fall \
  --side left \
  --output runs/10001_left_femur_fall
```

If `--output` is omitted, the output directory is written next to the input as
`<input>_parosol`.

Use `--dry-run` to build the input/model files and overview without launching
the solver:

```bash
parosol distal-radius.AIM --profile XtremeCTII --output runs/check --dry-run
```

Print a commented full config plus a profile override:

```bash
parosol config-template --profile XtremeCTII > xtremectii_case.yaml
```

Run an explicit config:

```bash
parosol run xtremectii_case.yaml
```

Run a batch config:

```bash
parosol batch load_history_3.yaml
```

## Output Files

A normal run writes:

- `parosol_case.yaml`: the resolved config that was run.
- `summary.json`: compact mechanics, failure, solver, field, quality, and
  execution metadata.
- `overview.png`: axial, sagittal, and coronal mid-slice QC figure with
  boundary-condition markers.
- `parosol_input.h5`: native solver input.
- `parosol_stdout.log` and `parosol_stderr.log`: native solver logs.
- `fields/*.nii.gz`: optional field exports, usually `sed.nii.gz`.
- `model/*`: model-building outputs for vertebra and femur profiles.

The summary includes the exact command, generated config path, input paths,
profile, dry-run flag, image spacing/origin, mechanics, Pistoia failure metrics,
solver iterations/residual/runtime, and exported file paths.

## Config Sections

ParOSol-py configs are organized into clear sections:

- `preprocessing`: generic image preparation such as `largest_cc`, `crop_to_bb`,
  `coarsen`, and smoothing for lower-resolution CT workflows.
- `model`: optional FE model construction from a density image and mask. Used by
  vertebra and proximal femur profiles.
- `materials`: label material definitions or density-to-E conversion. Supports
  scalar and label-specific Poisson ratios.
- `load_case`: boundary-condition definition, for example constrained axial
  compression, shear, bending, torsion, spine compression, or sideways fall.
- `solver`: native solver settings, MPI process count, tolerance, level, and
  native field outputs.
- `postprocess`: Pistoia failure load, vertebral strength estimates, field
  masking, and load-history estimation.
- `output`: summary, field exports, set/debug exports, and visualization paths.

## Most Important Profiles

### XtremeCTI

Use this for standard first-generation HR-pQCT binary material-label scans.

Command:

```bash
parosol scan.AIM --profile XtremeCTI --output runs/scan_XtremeCTI
```

Defaults:

- Input type: material labels.
- Material labels: `100 = TrabecularBone`, `127 = CorticalBone`.
- Young's modulus: `6829 MPa` for both trabecular and cortical labels.
- Poisson ratio: `0.3` for both labels.
- Load case: constrained axial compression along `z`.
- Applied strain: `-0.01`.
- Solver output: `sed`.
- Postprocess: Pistoia criterion with critical strain `0.007` and critical
  volume `2%`.

This profile is intentionally minimal: it runs at native image spacing and does
not smooth or resample the segmented HR-pQCT label image.

### XtremeCTII

Use this for standard second-generation HR-pQCT binary material-label scans.

Command:

```bash
parosol scan.AIM --profile XtremeCTII --output runs/scan_XtremeCTII
```

Defaults:

- Input type: material labels.
- Material labels: `100 = TrabecularBone`, `127 = CorticalBone`.
- Young's modulus: `8748 MPa` for both trabecular and cortical labels.
- Poisson ratio: `0.3` for both labels.
- Load case: constrained axial compression along `z`.
- Applied strain: `-0.01`.
- Solver output: `sed`.
- Postprocess: Pistoia criterion with critical strain `0.007` and critical
  volume `2%`.

For AIM files, spacing and origin are read from `aimio-py`. The output summary
reports physical height/displacement in millimeters, not voxel counts.

### Vertebra

Use this for QCT vertebral body compression from a density image plus
segmentation mask.

Command:

```bash
parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile vertebra \
  --output runs/10001_vertebra
```

Expected mask labels by default:

- `20`: vertebral body.
- `48`: posterior elements/process.

Modeling defaults:

- Model type: `spine_compression`.
- Isotropic model spacing: `1.0 mm`, with tolerance to avoid unnecessary
  resampling when already close.
- Uses the body label to generate projected PMMA caps/disks.
- Disk target label: `20`.
- Disk shape: `anatomy`.
- Disk thickness: `3 mm`.
- Disk intrusion depth: `6 mm`.
- Disk material: PMMA, `E = 2500 MPa`, `nu = 0.3`.
- Optional lightweight ICP registration can align the vertebral body to a
  reference point cloud without requiring VTK.

Preprocessing defaults:

- Keep largest connected component.
- Smooth density and labels with `sigma_mm: 1.0`.
- Crop to bounding box with margin.

Materials:

- Density-to-E conversion: linear, `E = 10 * density`.
- Poisson ratio: `0.3`.

Load case:

- Type: `spine_compression`.
- Axis: `z`.
- Displacement: `-0.2 mm`.

Postprocessing:

- Pistoia failure estimate.
- Linear reaction at `0.2%` deformation.
- Crawford-style stiffness-height strength estimate using coefficient `0.0068`.
- Field masking to the segmentation so generated disks do not contaminate the
  bone field statistics.

Outputs:

- `model/material.nii.gz`: generated material image including caps.
- `model/nodesets.nii.gz`: generated node/boundary-condition labels.
- `model/model.json`: modeling manifest.
- `model/qc.png`: model-building QC.
- `fields/sed.nii.gz` and optional `effective_strain`.
- `overview.png`: material and SED slices with boundary-condition markers.

### Proximal Femur Sideways Fall

Use this for proximal femur sideways-fall modeling from a density image plus
left/right femur mask.

Command:

```bash
parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile proximal_femur_sideways_fall \
  --side left \
  --output runs/10001_left_femur_fall
```

Expected mask labels by default:

- `2`: femur.

Modeling defaults:

- Model type: `proximal_femur_sideways_fall`.
- Side: `left` unless overridden with `--side right`.
- Cap axis: `y`.
- Isotropic model spacing: `1.0 mm`.
- Projected PMMA cap target label: `2`.
- Cap shape: `anatomy`.
- Cap thickness: `3 mm`.
- Cap intrusion depth: `6 mm`.
- Optional lightweight ICP registration is available but disabled by default.

Preprocessing defaults:

- Keep largest connected component.
- Smooth density and labels with `sigma_mm: 1.0`.
- Crop to bounding box with margin.

Materials:

- Density-to-E conversion: linear, `E = 10 * density`.
- Poisson ratio: `0.3`.
- PMMA cap material: `E = 2500 MPa`, `nu = 0.3`.

Load case:

- Type: `sideways_fall`.
- Applied displacement: `1.0 mm`.

Postprocessing:

- Pistoia failure estimate.
- SED field masked to the segmentation by default.

The older `proximal_femur` profile currently uses the same modeling defaults and
sideways-fall load case. Prefer `proximal_femur_sideways_fall` because the name
states the intended loading mode.

## Other Direct Mechanics Profiles

### constrained_axial_z

Constrained plate compression along `z` at `-1%` strain. Bottom nodes are fixed
in all directions; top nodes are laterally constrained and displaced along `z`.
This is the profile closest to the standard FAIM axial setup.

### smart_bone_compression_z

Constrained z compression with `surface.mode: smart` and `depth: auto`. This is
intended for uneven bone surfaces where strict first/last slices may be a poor
choice. It exports SED fields by default.

### shear_zx and shear_zy

Shear with the top z-normal surface moving laterally:

- `shear_zx`: top surface moves in `x`.
- `shear_zy`: top surface moves in `y`.

Both use `strain: 0.01` and export SED.

### bending_z

Opposing top/bottom plate tilt around the z-oriented model using
`bending_angle_degrees: 1`. The `neutral_axis_angle_degrees` setting controls
the bending direction. The summary reports generalized moment/stiffness values.

### torsion_z

Top plate twist about `z` using `twist_angle_degrees: 1`. The summary reports
generalized torque/stiffness values.

### density_power

Continuous density input profile. It changes `input.image_type` to `density`
and maps density to Young's modulus with:

```yaml
equation: power
coefficient: 10000
exponent: 1.7
reference_density: 1000
mask_threshold: 0
```

Use this when voxel values are density-like rather than material labels.

### standard_fields

Routine field-export profile. Exports SED as a NIfTI field.

### standard_mechanics_fields

Exports SED, effective strain, and von Mises fields.

### debug

Enables extra solver outputs and debug visualization fields:

- SED
- effective strain
- von Mises

### debug_sets

Exports node/element set debug files as JSON and VTK for inspection in tools
such as ParaView or Slicer.

### coarse_preview

Runs a fast downsampled preview solve using:

```yaml
preprocessing:
  coarsen:
    factor: 2
    reducer: mean
```

Use this only for quick exploratory runs, not final mechanics.

### batch

Optimized for larger batch runs where compact JSON summaries are the main
output. It uses four MPI processes, tolerance `1e-6`, and disables field export
unless another profile or config section enables it.

## Batch and Load-History Profiles

### direct_mechanics_manifest

Batch manifest with five direct mechanics cases:

- x compression.
- y compression.
- z compression.
- z-normal shear in x.
- z-normal shear in y.

### load_history_3

Three-case load-history batch:

- compression_z.
- shear_zx.
- shear_zy.

It exports SED for each case and declares a `postprocess.load_history` block for
NNLS load-history estimation.

### load_history_6

Six-case load-history batch:

- compression_z.
- shear_zx.
- shear_zy.
- bending_x.
- bending_y.
- torsion_z.

It exports SED for each case and declares the same NNLS load-history
postprocessing interface.

After the SED fields exist, load-history estimation can be run directly:

```bash
parosol load-history compression_sed.nii.gz shear_x_sed.nii.gz shear_y_sed.nii.gz \
  --bone-mask bone_mask.nii.gz \
  --summary load_history_summary.json \
  --output load_history.nii.gz
```

### progressive_loading_manifest

Runs linear compression increments:

- `-0.25%`
- `-0.50%`
- `-0.75%`
- `-1.00%`

This is a sequence of linear solves, not nonlinear or progressive damage
mechanics.

## Material and Poisson Ratio Handling

Label images can define a different `E` and `nu` per material:

```yaml
materials:
  units: MPa
  definitions:
    TrabecularBone:
      Type: LinearIsotropic
      E: 8748
      nu: 0.25
    CorticalBone:
      Type: LinearIsotropic
      E: 12000
      nu: 0.3
  table:
    100: TrabecularBone
    127: CorticalBone
```

ParOSol-py writes a native per-element Poisson-ratio image when material labels
use different `nu` values. Scalar `nu = 0.3` behavior remains compatible with the
standard reference runs.

For continuous density input, `materials.poisson_ratio` may be a scalar or an
equation; equation-based values are currently reduced to one scalar before solve.

## Validation and Reference Behavior

The repository includes a connected-component-filtered TRAB_1240 fixture and a
fixed `nu = 0.3` axial-compression reference. By default the normal test suite
does not run the native reference solve. To run it locally:

```bash
PAROSOL_RUN_REFERENCE_TESTS=1 pytest tests/test_trab1240_reference.py
```

For the CABHS HR-pQCT sample, the XtremeCTII profile has been checked against
legacy axial/Pistoia output with close agreement in stiffness and failure load.

