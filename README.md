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

### HR-pQCT Load-History Estimation

For HR-pQCT load-history estimation, solve a small basis of mechanical load
cases, export SED for each case, then combine those fields with the NNLS
load-history estimator. Two built-in YAML profile templates describe the common
bases:

- `load_history_3.yaml`: compression, shear in x, and shear in y.
- `load_history_6.yaml`: compression, shear x/y, bending x/y, and torsion z.

Run it with the same shortcut structure as any other profile. In the CLI,
omit the `.yaml` suffix from the profile name:

```bash
parosol distal-radius.AIM \
  --profile load_history_3 \
  --output outputs/distal-radius_load_history
```

This writes `parosol_batch.yaml`, solves the three or six load cases into
separate case folders, and writes `batch_summary.json`. After the SED fields
exist, run the optimizer step:

```bash
parosol load-history \
  outputs/compression_z/fields/sed.nii.gz \
  outputs/shear_zx/fields/sed.nii.gz \
  outputs/shear_zy/fields/sed.nii.gz \
  --bone-mask bone_mask.nii.gz \
  --summary outputs/load_history_summary.json \
  --output outputs/load_history.nii.gz
```

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
than inside `src/parosol_native`. The torch backend is experimental and must be
called explicitly; it is not selected by normal `parosol` profiles.

```python
import numpy as np
from parosol_torch import SolverSettings, VoxelElasticityProblem, backend_info, solve

print(backend_info())

fixed = []
loaded = []
for y in range(2):
    for z in range(2):
        for component in range(3):
            fixed.append([0, y, z, component])
        loaded.append([1, y, z, 0])

problem = VoxelElasticityProblem(
    stiffness_gpa_xyz=np.ones((1, 1, 1), dtype=np.float32),
    voxel_size_mm=1.0,
    poisson_ratio=0.3,
    fixed_displacement_coordinates=np.asarray(fixed),
    fixed_displacement_values=np.zeros(len(fixed)),
    loaded_node_coordinates=np.asarray(loaded),
    loaded_node_values=np.full(len(loaded), 0.01),
)
result = solve(problem, SolverSettings(device="mps"))
```

The experimental solver uses a matrix-free 8-node hexahedral elasticity operator
and torch conjugate gradients on `cpu`, `mps`, or `cuda`. It is meant for small
backend-development and validation cases until it matches native ParOSol
reference runs. Install torch separately with `pip install -e .[torch]`.

See `docs/gpu-backend-roadmap.md` for the validation plan.

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
