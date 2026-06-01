# parosol-py

Standalone Python package for running the ParOSol micro-FE solver from Python.

See [docs/analysis-profiles.md](docs/analysis-profiles.md) for the full
installation guide, profile reference, and workflow documentation.

## Install

### Prebuilt Wheels

The recommended installation is from prebuilt wheels. Wheels include the native
ParOSol executable, so users do not need to compile C++ code locally.

For now, parosol-py is distributed privately from GitHub Release wheels. Clone
the private repository and use the helper script:

```bash
git clone git@github.com:wallematthias/parosol-py.git
cd parosol-py
python scripts/install_prebuilt.py
```

The helper downloads all release wheels and lets `pip` choose the matching wheel
for the active Python/platform. This is different from `pip install -e .`, which
is an editable source install and compiles the native solver locally.
If GitHub CLI is not authenticated yet, run `gh auth login` once before using
the helper.

Manual equivalent:

```bash
tmpdir="$(mktemp -d)"
gh release download --repo wallematthias/parosol-py --pattern "*.whl" --dir "$tmpdir"
python -m pip install --no-index --find-links "$tmpdir" parosol-py
```

The GitHub Actions wheel matrix builds private Linux x86_64, Windows AMD64,
macOS arm64, and macOS x86_64 artifacts for supported Python versions.
Downloading all wheels from a release avoids manual wheel selection.

That same command works from Python 3.10, 3.11, 3.12, or 3.13 environments as
long as the release contains a wheel for that Python/platform tag. `pip` picks
the compatible wheel automatically.

### Developer Install

Developer/source installs compile the native solver:

```bash
pip install -e .[dev]
```

Native build requirements:

- CMake
- MPI C++ compiler/runtime
- HDF5 C++ libraries
- Eigen headers

If editable installation fails during CMake configuration, check that `mpicxx`
or an equivalent MPI C++ compiler wrapper is available on `PATH`.

## Quick Workflows

### HR-pQCT XtremeCTI / XtremeCTII

Use the scanner profiles for standard binary material-label HR-pQCT scans:

```bash
parosol distal-radius.AIM --profile XtremeCTII --output outputs/distal-radius
```

`XtremeCTI` uses `E = 6829 MPa`; `XtremeCTII` uses `E = 8748 MPa`. Both map
labels `100 = TrabecularBone` and `127 = CorticalBone`, use `nu = 0.3`, run
constrained z-axis axial compression at `-1%` strain, export SED, and compute
the Pistoia failure load.

### Vertebra CT

Use the vertebra model profile for density image plus segmentation mask:

```bash
parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile vertebra \
  --output outputs/10001_vertebra
```

The default mask labels are `20 = vertebral body` and
`48 = vertebral process`. The profile builds PMMA disks/caps, optionally supports
lightweight ICP alignment to a reference point cloud, runs spine compression,
and reports Pistoia plus linear vertebral strength estimates.

### Proximal Femur Sideways Fall

Use the proximal femur profile for a density image plus femur mask:

```bash
parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile proximal_femur_sideways_fall \
  --side left \
  --output outputs/10001_left_femur_fall
```

The default femur label is `2`. The profile resamples/crops/smooths as needed,
builds PMMA caps, runs a sideways-fall load case, exports SED, and writes model
QC images.

## Python API

```python
import numpy as np
from parosol_py import solve

material = np.ones((10, 10, 10), dtype=float) * 1000.0  # z, y, x; MPa

result = solve(
    material=material,
    spacing=(0.061, 0.061, 0.061),
    material_unit="MPa",
    test="axial",
    test_axis="z",
    strain=-0.01,
    outputs=("sed",),
    export_dir="outputs",
)

print(result.summary)
print(result.exported)
```

Use `dry_run=True` to write the ParOSol HDF5 input and inspect the generated
command without launching the solver.

Material helpers are available for both label images and continuous density
images:

```python
from parosol_py import density_to_material_map

mapped = density_to_material_map(
    density_image,
    equation="power",
    coefficient=10000,
    exponent=1.7,
    reference_density=1000,
    poisson_ratio=0.3,
)
```

Label-image material tables can define different Poisson ratios per material.
ParOSol-py writes those as an optional native per-element Poisson ratio image.
For config files, prefer the project-native material syntax:

```yaml
materials:
  units: MPa
  labels:
    100: {name: trabecular_bone, E: 8748, nu: 0.3}
    127: {name: cortical_bone, E: 8748, nu: 0.3}
```

Continuous density profiles keep the modulus conversion and Poisson ratio
together under `materials.density`.

## Command Line

The primary command is profile-driven:

```bash
parosol distal-radius.AIM --profile XtremeCTII --output outputs/distal-radius
```

If `--output` is omitted, ParOSol-py writes to a sibling directory named
`<input>_parosol`. Every run writes the generated `parosol_case.yaml`,
`summary.json`, field images when enabled, and an overview PNG. The summary
contains an `execution` section with the resolved input paths, profile, generated
config path, output directory, dry-run flag, and exact shortcut command.

Model profiles use the same command and accept a standard `--mask` argument:

```bash
parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile vertebra \
  --output outputs/10001_vertebra

parosol 10001_QCT.nii.gz \
  --mask 10001_SEG.nii.gz \
  --profile proximal_femur_sideways_fall \
  --side left \
  --output outputs/10001_left_femur_fall
```

Use `--dry-run` to write the generated model, solver input, overview, and JSON
without launching the native solver:

```bash
parosol distal-radius.AIM --profile XtremeCTII --output outputs/distal-radius --dry-run
```

Advanced users can still keep the generated YAML, edit any section, and run it
directly:

```yaml
case:
  name: VITD_0003_RL_M06_HOM_LS
  work_dir: outputs/VITD_0003_RL_M06_HOM_LS

input:
  image: VITD_0003_RL_M06_HOM_LS.AIM
  image_type: material_labels
  spacing: [0.061, 0.061, 0.061]

materials:
  units: MPa
  labels:
    100:
      name: trabecular_bone
      E: 8748
      nu: 0.3
    127:
      name: cortical_bone
      E: 8748
      nu: 0.3

load_case:
  type: constrained_axial
  axis: z
  strain: -0.01
  surface:
    mode: smart
    depth: auto

solver:
  tolerance: 1e-6
  level: 6
  outputs: [sed]

postprocess:
  pistoia:
    criterion: pistoia
    critical_volume_percent: 2.0
    critical_strain: 0.007

output:
  summary: outputs/VITD_0003_RL_M06_HOM_LS/summary.json
  fields: [sed]
  export_fields: true
  visualize: true
```

The Pistoia-style failure factor is computed from the strain/SED field for all
load cases. Compression reports a failure force, while shear, bending, and
torsion also report a `failure_generalized_load` whose interpretation follows
the load case, for example force or moment.

Config-driven runs write a professional overview PNG by default. It shows
axial, sagittal, and coronal mid-slices of the material image, the selected
result field on the second row when available, and boundary-condition markers
and direction arrows for quick debugging.

Run an explicit config:

```bash
parosol run case.yaml
```

Run a multi-case direct-mechanics batch:

```bash
parosol batch direct_mechanics.yaml
```

A batch config uses the same top-level input/material/solver/output sections as
a normal case, plus a `batch.cases` list. Each case override is expanded into an
individual run directory and summarized in one `batch_summary.json`.

Run a whole folder with the same profile:

```bash
parosol batch /data/xtremectii_inputs \
  --profile XtremeCTII \
  --output /data/xtremectii_results
```

For model-building profiles, point the batch at a mask folder:

```bash
parosol batch /data/qct_images \
  --profile vertebra \
  --mask-dir /data/segmentations \
  --mask-pattern "{stem}_SEG.nii.gz" \
  --output /data/vertebra_results
```

Scanner/load-case profiles are available as `XtremeCTI` and `XtremeCTII`. Each
profile defines the standard binary bone material labels, constrained z-axis
compression at 1% strain, SED output, and Pistoia post-processing defaults.

All built-in profiles are documented in
[docs/analysis-profiles.md](docs/analysis-profiles.md). The current profile set
includes:

- `XtremeCTI`, `XtremeCTII`
- `vertebra`
- `proximal_femur`, `proximal_femur_sideways_fall`
- `constrained_axial_z`, `smart_bone_compression_z`
- `shear_zx`, `shear_zy`, `bending_z`, `torsion_z`
- `density_power`, `standard_fields`, `standard_mechanics_fields`
- `debug`, `debug_sets`, `coarse_preview`, `batch`
- `direct_mechanics_manifest`, `load_history_3`, `load_history_6`
- `progressive_loading_manifest`

Load-history profiles are available as `load_history_3` and `load_history_6`.
They declare a `postprocess.load_history` block and generate the solved SED
fields for the NNLS load-history post-processing step:

```bash
parosol load-history compression_sed.nii.gz shear_x_sed.nii.gz shear_y_sed.nii.gz \
  --bone-mask bone_mask.nii.gz \
  --summary load_history_summary.json \
  -o load_history.nii.gz
```

Useful output controls:

```yaml
output:
  fields: [sed, effective_strain, von_mises]
  export_fields: true
  export_sets: true
  set_formats: [json, vtk]

solver:
  max_relative_residual: 1.0e-6
```

`output.fields` selects image-field outputs. `output.export_sets` writes node-set
and material element-set debug files for Slicer/ParaView inspection.

Convert old legacy solver text outputs into compact JSON:

```bash
parosol summarize-legacy \
  --analysis VITD_0003_RL_M06_HOM_LS_analysis.txt \
  --pistoia VITD_0003_RL_M06_HOM_LS_pistoia.txt \
  -o VITD_0003_RL_M06_HOM_LS_summary.json
```

## AIM Input

```python
from parosol_py import solve_aim

result = solve_aim("segmented.aim", outputs=("sed",), export_dir="outputs")
```

AIM IO is provided by `aimio-py` through `py_aimio`.

## GPU Backends

The native ParOSol C++/MPI solver remains the validated reference backend.
Accelerator work lives in a separate optional `parosol_torch` namespace rather
than inside `src/parosol_native`. This package currently exposes capability
checks only; it does not yet provide a validated numerical solver.

```python
from parosol_torch import backend_info

print(backend_info())
```

See `docs/gpu-backend-roadmap.md` for the implementation and validation plan.

## Validation

The test suite includes a connected-component-filtered TRAB_1240 reference fixture.
By default, tests validate the compressed fixture metadata without running the solver.
Set `PAROSOL_RUN_REFERENCE_TESTS=1` to run the ParOSol axial-compression regression
and compare stiffness, reaction force, and Pistoia failure load against the fixed
reference JSON.

See `docs/legacy-feature-map.md` for the compatibility roadmap and the cleaner
ParOSol-py equivalents of useful legacy workflow features.
See `docs/n88-reference-verification.md` for the current local reference
verification results for standard load cases.
