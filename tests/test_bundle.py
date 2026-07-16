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


def _write_minimal_batch_case(tmp_path: Path) -> Path:
    config_path = _write_minimal_case(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["case"]["name"] = "bundle_batch"
    config["case"]["work_dir"] = str(tmp_path / "source_batch" / "bundle_batch")
    config["batch"] = {
        "work_dir": str(tmp_path / "source_batch"),
        "summary": str(tmp_path / "source_batch" / "result.json"),
        "cases": [
            {
                "name_suffix": "compression_z",
                "load_case": config["load_case"],
            }
        ],
    }
    config["postprocess"] = {
        "load_history": {
            "enabled": True,
            "summary": str(tmp_path / "source_batch" / "load_history_summary.json"),
            "output": str(tmp_path / "source_batch" / "fields" / "estimated_sed.nii.gz"),
            "final_rerun": {
                "enabled": True,
                "field": "sed",
                "fields": ["sed"],
                "output": str(tmp_path / "source_batch" / "fields" / "final_sed.nii.gz"),
            },
        }
    }
    batch_path = tmp_path / "parosol_batch.yaml"
    batch_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return batch_path


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


def test_bundle_create_preserves_batch_config_with_portable_outputs(tmp_path: Path):
    config_path = _write_minimal_batch_case(tmp_path)
    bundle_path = tmp_path / "batch_case.parosol"

    create_bundle(config_path, bundle_path)

    with zipfile.ZipFile(bundle_path) as archive:
        bundled_config = yaml.safe_load(archive.read("parosol_case.yaml"))

    assert bundled_config["case"]["work_dir"] == "bundle_batch"
    assert bundled_config["output"]["result"] == "bundle_batch/result.json"
    assert bundled_config["output"]["fields_dir"] == "bundle_batch/fields"
    assert bundled_config["batch"]["work_dir"] == "."
    assert bundled_config["batch"]["summary"] == "result.json"
    assert bundled_config["postprocess"]["load_history"]["summary"] == "load_history_summary.json"
    assert bundled_config["postprocess"]["load_history"]["output"] == "fields/estimated_sed.nii.gz"
    assert (
        bundled_config["postprocess"]["load_history"]["final_rerun"]["output"]
        == "fields/final_sed.nii.gz"
    )


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


def test_batch_bundle_run_dry_run_generates_batch_outputs_in_output_dir(tmp_path: Path):
    config_path = _write_minimal_batch_case(tmp_path)
    bundle_path = create_bundle(config_path, tmp_path / "batch_case.parosol")
    output_dir = tmp_path / "remote_batch_run"

    summary = run_bundle(bundle_path, output_dir=output_dir, dry_run=True)

    assert summary["batch"]["summary"] == str(output_dir / "result.json")
    assert (output_dir / "result.json").exists()
    assert (output_dir / "parosol_case.yaml").exists()
    runtime_config = yaml.safe_load((output_dir / "parosol_case.yaml").read_text())
    assert runtime_config["execution"]["interface"] == "bundle-batch"
    assert runtime_config["case"]["work_dir"] == str(output_dir / "bundle_batch")
    assert runtime_config["batch"]["work_dir"] == str(output_dir)
    assert runtime_config["batch"]["summary"] == str(output_dir / "result.json")
    assert runtime_config["postprocess"]["load_history"]["summary"] == str(
        output_dir / "load_history_summary.json"
    )
    assert runtime_config["postprocess"]["load_history"]["output"] == str(
        output_dir / "fields" / "estimated_sed.nii.gz"
    )
    assert runtime_config["postprocess"]["load_history"]["final_rerun"]["output"] == str(
        output_dir / "fields" / "final_sed.nii.gz"
    )
    assert summary["postprocess"]["load_history"]["status"] == "dry_run"
    assert (output_dir / "bundle_batch_compression_z" / "summary.json").exists()


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


def test_cli_runs_portable_batch_bundle(tmp_path: Path, capsys):
    config_path = _write_minimal_batch_case(tmp_path)
    bundle_path = create_bundle(config_path, tmp_path / "cli_batch_case.parosol")
    output_dir = tmp_path / "cli_remote_batch_run"

    assert main(["run", str(bundle_path), "--output", str(output_dir), "--dry-run"]) == 0

    assert (output_dir / "result.json").exists()
    assert (output_dir / "bundle_batch_compression_z" / "summary.json").exists()
    assert str(output_dir / "result.json") in capsys.readouterr().out
