# FE Reference Fixtures

This folder contains compact real-world fixtures for the mini FE workflow
lockdown tests.

- `core_metric_lock.json` is the machine-readable core metric lock.
- `core_metric_lock.csv` is the flattened tabular companion for inspection.
- `field_data/manifest.json` indexes rediscoverable NIfTI field outputs.
- `field_data/<fixture>/<profile>/<engine>/<case>/material.nii.gz` stores the
  material field for a solved case.
- `field_data/<fixture>/<profile>/<engine>/<case>/sed.nii.gz` stores the strain
  energy density field for the same solved case.

The field archive currently includes Ogo and ParOSol outputs for spine and hip,
FAIM and ParOSol outputs for XtremeCT I/II and load-history 3, and ParOSol
outputs for load-history 6.

The metric lock stores rich records for rediscovery, but the regression tests
only hard-assert the canonical values needed to catch workflow drift: stiffness,
headline XtremeCT failure values, prescribed deformation percentages, and
load-history unit-case amplitudes.
