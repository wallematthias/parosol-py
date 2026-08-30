"""Output layout helpers for FEA derivatives."""

from __future__ import annotations

from pathlib import Path

from bone_imaging_derivatives.layout import record_output_path


_FEA_FILENAMES = {
    "solver_input": "parosol_input.h5",
    "solver_config": "parosol_case.yaml",
    "material_image": "material.nii.gz",
    "boundary_conditions": "boundary_conditions.json",
    "sed_map": "sed.nii.gz",
    "strain_map": "strain.nii.gz",
    "stress_map": "stress.nii.gz",
    "displacement_map": "displacement.nii.gz",
    "summary_table": "summary.json",
    "diagnostic_log": "parosol.log",
}


def fea_derivative_output_paths(
    dataset_root: str | Path,
    *,
    subject_id: str,
    site: str,
    case_id: str,
) -> dict[str, Path]:
    """Return standard output paths for one FEA derivative case."""
    output_dir = record_output_path(
        Path(dataset_root),
        "FEA",
        subject_id,
        site,
        "runs",
        case_id,
    )
    return {
        "output_dir": output_dir,
        **{role: output_dir / filename for role, filename in _FEA_FILENAMES.items()},
    }
