# Numerics88 Reference Verification

Local reference checks were run against a Numerics88 10.0 installation using
a homogeneous 3 x 3 x 3 voxel cube, `E = 1000 MPa`, `nu = 0.3`, and unit
spacing. The checks compare ParOSol-py generated boundary conditions and solved
SED fields against `n88modelgenerator`, `n88solver_slt`, and `n88derivedfields`.

## Boundary Conditions

Boundary-condition coordinate/value sets matched exactly for:

| Load case | Reference command options | Result |
| --- | --- | --- |
| Axial compression | `--test axial --test_axis z --normal_strain -0.01` | Exact match |
| Uniaxial compression | `--test uniaxial --test_axis z --normal_strain -0.01` | Exact match |
| Confined compression | `--test confined --test_axis z --normal_strain -0.01` | Exact match |
| Directional shear x | `--test dshear --test_axis z --shear_vector 0.02,0` | Exact match |
| Directional shear y | `--test dshear --test_axis z --shear_vector 0,0.02` | Exact match |

## SED Fields

| Load case | Reference mean | ParOSol-py mean | Relative L2 error |
| --- | ---: | ---: | ---: |
| Axial compression | 0.0537145913 | 0.0537145834 | 2.48e-7 |
| Uniaxial compression | 0.0500000059 | 0.0499999978 | 2.36e-7 |
| Confined compression | 0.0673077026 | 0.0673076893 | 2.00e-7 |
| Directional shear x | 0.0549606691 | 0.0549606620 | 2.33e-7 |
| Directional shear y | 0.0549606677 | 0.0549606620 | 2.27e-7 |

These checks validate the standard plate compression and directional shear
boundary conditions before scaling to real bone images. The TRAB_1240 fixture
adds the larger real-image regression for axial compression, including Pistoia
summary metrics and a dense SED field comparison.
