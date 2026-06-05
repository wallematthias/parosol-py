from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import yaml

from parosol_py.bundle import create_bundle, inspect_bundle, run_bundle
from parosol_py.cli import main


def _write_minimal_case(tmp_path: Path) -> Path:
    image = np.zeros((4, 4, 4), dtype=np.float32)
    image[:, 1:3, 1:3] = 1000.0
    image_path = tmp_path / "material.npy"
    np.save(image_path, image)

    config = {
        "case": {"name": "bundle_cube", "work_dir": str(tmp_path / "source_run")},
        "input": {
            "image": str(image_path),
            "image_type": "material_mpa",
            "spacing": [1.0, 1.0, 1.0],
            "origin": [0.0, 0.0, 0.0],
        },
        "materials": {"units": "MPa", "nu": 0.3},
        "load_case": {"type": "constrained_axial", "axis": "z", "strain": -0.01},
        "solver": {"outputs": ["sed"], "mpi_processes": 1, "mpi_launcher": ""},
        "output": {
            "result": "result.json",
            "run_summary": "summary.json",
            "fields": ["sed"],
            "export_fields": True,
            "fields_dir": "fields",
            "visualization": False,
        },
        "postprocess": {"pistoia": {"criterion": "pistoia"}},
    }
    config_path = tmp_path / "parosol_case.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_bundle_create_writes_single_portable_file_with_manifest_and_inputs(tmp_path: Path):
    config_path = _write_minimal_case(tmp_path)
    bundle_path = tmp_path / "named_case.parosol"

    created = create_bundle(config_path, bundle_path)

    assert created == bundle_path
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "parosol_case.yaml" in names
        assert "inputs/material.npy" in names
        bundled_config = yaml.safe_load(archive.read("parosol_case.yaml"))
        manifest = json.loads(archive.read("manifest.json"))

    assert bundled_config["input"]["image"] == "inputs/material.npy"
    assert bundled_config["output"]["result"] == "result.json"
    assert manifest["format"] == "parosol-py-bundle"
    assert manifest["case"]["name"] == "bundle_cube"


def test_bundle_run_dry_run_generates_postprocessing_outputs_in_output_dir(tmp_path: Path):
    config_path = _write_minimal_case(tmp_path)
    bundle_path = create_bundle(config_path, tmp_path / "named_case.parosol")
    output_dir = tmp_path / "remote_run"

    result = run_bundle(bundle_path, output_dir=output_dir, dry_run=True)

    assert result.input_file == output_dir / "parosol_input.h5"
    assert (output_dir / "parosol_case.yaml").exists()
    assert (output_dir / "result.json").exists()
    assert (output_dir / "summary.json").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["execution"]["interface"] == "bundle"
    assert summary["outputs"]["input_file"] == str(output_dir / "parosol_input.h5")


def test_bundle_inspect_reports_case_and_manifest(tmp_path: Path):
    config_path = _write_minimal_case(tmp_path)
    bundle_path = create_bundle(config_path, tmp_path / "named_case.parosol")

    info = inspect_bundle(bundle_path)

    assert info["path"] == str(bundle_path)
    assert info["manifest"]["case"]["name"] == "bundle_cube"
    assert "inputs/material.npy" in info["files"]


def test_cli_creates_and_runs_portable_bundle(tmp_path: Path):
    config_path = _write_minimal_case(tmp_path)
    bundle_path = tmp_path / "cli_case.parosol"
    output_dir = tmp_path / "cli_remote_run"

    assert main(["bundle", "create", str(config_path), "--output", str(bundle_path)]) == 0
    assert bundle_path.exists()

    assert main(["run", str(bundle_path), "--output", str(output_dir), "--dry-run"]) == 0
    assert (output_dir / "parosol_input.h5").exists()
    assert (output_dir / "result.json").exists()
    assert (output_dir / "summary.json").exists()
