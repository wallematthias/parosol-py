from __future__ import annotations

import argparse
from pathlib import Path

from parosol_py.cli import _shortcut_config
from parosol_py.derivatives import fea_derivative_output_paths


def test_fea_derivative_output_paths_use_the_shared_family_layout(tmp_path: Path) -> None:
    """Changing the derivative family or artifact names must not silently move FEA outputs."""
    paths = fea_derivative_output_paths(
        tmp_path,
        subject_id="001",
        site="tibia",
        case_id="baseline",
    )

    output_dir = tmp_path / "derivatives" / "FEA" / "sub-001" / "site-tibia" / "runs" / "baseline"
    assert paths["output_dir"] == output_dir
    assert paths["solver_input"] == output_dir / "parosol_input.h5"
    assert paths["sed_map"] == output_dir / "sed.nii.gz"
    assert paths["summary_table"] == output_dir / "summary.json"
    assert paths["diagnostic_log"] == output_dir / "parosol.log"


def test_shortcut_uses_fea_derivative_output_when_dataset_context_is_provided(tmp_path: Path) -> None:
    """Dropping the dataset context would return the shortcut to its legacy sibling output folder."""
    image_path = tmp_path / "material.npy"
    args = argparse.Namespace(
        image=str(image_path),
        profile="XtremeCTII",
        mask=None,
        output=None,
        name="baseline",
        dry_run=True,
        side=None,
        reference_points=None,
        template=None,
        dataset_root=str(tmp_path),
        subject="001",
        site="tibia",
        _argv=[str(image_path), "--profile", "XtremeCTII"],
    )

    config = _shortcut_config(args)

    assert config["execution"]["output_dir"] == str(
        tmp_path / "derivatives" / "FEA" / "sub-001" / "site-tibia" / "runs" / "baseline"
    )
