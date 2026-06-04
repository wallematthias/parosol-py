import json
from pathlib import Path

import numpy as np

from parosol_py.batch import run_batch_config
from parosol_py.cli import main


def test_run_batch_config_expands_cases_and_writes_combined_summary(
    monkeypatch,
    tmp_path: Path,
):
    np.save(tmp_path / "material.npy", np.ones((2, 2, 2), dtype=np.float32) * 1000)
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(
        json.dumps(
            {
                "case": {"name": "sample", "work_dir": "runs/sample"},
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "output": {"dry_run": True},
                "postprocess": {
                    "load_history": {
                        "enabled": True,
                        "method": "nnls",
                        "fields": ["sed"],
                        "summary": "runs/load_history_summary.json",
                        "output": "runs/load_history.nii.gz",
                    }
                },
                "batch": {
                    "summary": "runs/batch_summary.json",
                    "cases": [
                        {
                            "name_suffix": "compression_z",
                            "load_case": {
                                "type": "constrained_axial",
                                "axis": "z",
                                "strain": -0.01,
                            },
                        },
                        {
                            "name_suffix": "shear_zx",
                            "load_case": {
                                "type": "shear",
                                "axis": "z",
                                "direction": "x",
                                "strain": 0.01,
                            },
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    seen = []

    def fake_run_case_config(path, *, dry_run=None, work_dir=None):
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        seen.append(config)
        summary_path = Path(config["output"]["summary"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "case": {"name": config["case"]["name"]},
                    "load_case": config["load_case"],
                    "mechanics": {
                        "generalized_load": {
                            "name": "force",
                            "value": len(seen),
                            "units": "N",
                        },
                        "generalized_stiffness": {
                            "name": "stiffness",
                            "value": 100 * len(seen),
                            "units": "N/mm",
                        },
                    },
                    "failure": {
                        "criterion": "pistoia",
                        "factor": 0.5 * len(seen),
                        "failure_load": {
                            "x": None,
                            "y": None,
                            "z": -5 * len(seen),
                        },
                        "failure_generalized_load": {
                            "name": "force",
                            "value": 10 * len(seen),
                            "units": "N",
                        },
                        "status": "computed",
                    },
                }
            ),
            encoding="utf-8",
        )
        return object()

    monkeypatch.setattr("parosol_py.batch.run_case_config", fake_run_case_config)

    summary = run_batch_config(batch_path, dry_run=True)

    assert [config["case"]["name"] for config in seen] == [
        "sample_compression_z",
        "sample_shear_zx",
    ]
    assert seen[0]["case"]["work_dir"].endswith("sample_compression_z")
    assert seen[0]["output"]["summary"].endswith("sample_compression_z/result.json")
    assert seen[0]["output"]["run_summary"].endswith(
        "sample_compression_z/summary.json"
    )
    assert seen[0]["output"]["fields_dir"].endswith("sample_compression_z/fields")
    assert seen[0]["output"]["visualization"].endswith(
        "sample_compression_z/overview.png"
    )
    assert summary["batch"]["case_count"] == 2
    assert summary["postprocess"]["load_history"]["method"] == "nnls"
    assert summary["cases"][1]["load_case"]["direction"] == "x"
    assert summary["cases"][1]["generalized_stiffness"]["value"] == 200
    assert summary["cases"][1]["failure_load"]["z"] == -10
    assert summary["cases"][1]["failure"]["factor"] == 1.0
    assert summary["cases"][1]["failure_generalized_load"]["value"] == 20
    assert (tmp_path / "runs" / "batch_summary.json").exists()


def test_cli_batch_runs_manifest(monkeypatch, tmp_path: Path, capsys):
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(
        json.dumps({"batch": {"summary": "summary.json", "cases": []}}),
        encoding="utf-8",
    )

    def fake_run_batch_config(path, *, dry_run=None, work_dir=None):
        assert Path(path) == batch_path
        assert dry_run is True
        return {"batch": {"case_count": 0, "summary": str(tmp_path / "summary.json")}}

    monkeypatch.setattr("parosol_py.cli.run_batch_config", fake_run_batch_config)

    assert main(["batch", str(batch_path), "--dry-run"]) == 0

    assert "summary.json" in capsys.readouterr().out


def test_run_batch_config_computes_load_history_postprocess(
    monkeypatch,
    tmp_path: Path,
):
    np.save(tmp_path / "material.npy", np.ones((2, 2, 2), dtype=np.float32) * 1000)
    sed_a = tmp_path / "compression_sed.npy"
    sed_b = tmp_path / "shear_sed.npy"
    np.save(sed_a, np.ones((2, 2, 2), dtype=np.float32))
    np.save(sed_b, np.ones((2, 2, 2), dtype=np.float32) * 2.0)
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(
        json.dumps(
            {
                "case": {"name": "sample", "work_dir": "runs/sample"},
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "output": {"fields": ["sed"], "export_fields": True},
                "postprocess": {
                    "load_history": {
                        "enabled": True,
                        "method": "nnls",
                        "fields": ["sed"],
                        "cases": ["compression_z", "shear_zx"],
                        "summary": "runs/load_history_summary.json",
                        "output": "runs/load_history.npz",
                    }
                },
                "batch": {
                    "summary": "runs/batch_summary.json",
                    "cases": [
                        {
                            "name_suffix": "compression_z",
                            "load_case": {
                                "type": "constrained_axial",
                                "axis": "z",
                                "strain": -0.01,
                            },
                        },
                        {
                            "name_suffix": "shear_zx",
                            "load_case": {
                                "type": "shear",
                                "axis": "z",
                                "direction": "x",
                                "strain": 0.01,
                            },
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    sed_paths = [sed_a, sed_b]

    def fake_run_case_config(path, *, dry_run=None, work_dir=None):
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        index = 0 if config["case"]["name"].endswith("compression_z") else 1
        summary_path = Path(config["output"]["summary"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "case": {"name": config["case"]["name"]},
                    "load_case": config["load_case"],
                    "outputs": {"exported": {"sed": str(sed_paths[index])}},
                    "mechanics": {
                        "generalized_load": {
                            "name": "force",
                            "component": "z",
                            "value": 10.0 * (index + 1),
                            "units": "N",
                        }
                    },
                    "failure": {},
                }
            ),
            encoding="utf-8",
        )
        return object()

    monkeypatch.setattr("parosol_py.batch.run_case_config", fake_run_case_config)

    summary = run_batch_config(batch_path)

    load_history = summary["postprocess"]["load_history"]
    assert load_history["status"] == "computed"
    assert Path(load_history["summary"]).exists()
    assert Path(load_history["output"]).exists()
    assert len(load_history["details"]["scaling_factors"]) == 2
    assert load_history["details"]["input_load_amplitudes"] == [10.0, 20.0]
    assert len(load_history["results"]["estimated_loads"]) == 2
    assert (
        load_history["results"]["estimated_loads"][0]["value"]
        == load_history["details"]["load_amplitudes"][0]
    )
    assert load_history["results"]["estimated_loads"][0]["units"] == "N"
    saved = np.load(load_history["output"])
    assert saved["image"].shape == (2, 2, 2)


def test_cli_shortcut_routes_load_history_profile_to_batch(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    image_path = tmp_path / "distal_radius.mha"
    output_dir = tmp_path / "distal_radius_load_history"
    image_path.write_bytes(b"placeholder")
    seen = {}

    def fake_run_batch_config(path, *, dry_run=None, work_dir=None):
        import yaml

        config_path = Path(path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        seen["path"] = config_path
        seen["config"] = config
        seen["dry_run"] = dry_run
        seen["work_dir"] = work_dir
        return {
            "batch": {
                "case_count": 3,
                "summary": str(output_dir / "batch_summary.json"),
            },
            "cases": [],
        }

    monkeypatch.setattr("parosol_py.cli.run_batch_config", fake_run_batch_config)

    assert (
        main(
            [
                str(image_path),
                "--profile",
                "load_history_3",
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    config = seen["config"]
    assert seen["path"] == output_dir / "parosol_batch.yaml"
    assert seen["dry_run"] is True
    assert seen["work_dir"] == output_dir
    assert config["execution"]["interface"] == "shortcut"
    assert config["execution"]["profile"] == "load_history_3"
    assert config["input"]["image"] == str(image_path.resolve())
    assert config["input"]["spacing"] == "auto"
    assert config.get("nodesets", {}) == {}
    assert config["case"]["name"] == "distal_radius"
    assert config["case"]["work_dir"] == str(output_dir / "distal_radius")
    assert config["batch"]["summary"] == str(output_dir / "result.json")
    assert [case["name_suffix"] for case in config["batch"]["cases"]] == [
        "compression_z",
        "shear_zx",
        "shear_zy",
    ]
    assert "batch_summary.json" in capsys.readouterr().out


def test_cli_batch_runs_folder_with_profile(monkeypatch, tmp_path: Path, capsys):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    np.save(input_dir / "sample_a.npy", np.ones((2, 2, 2), dtype=np.uint8) * 100)
    np.save(input_dir / "sample_b.npy", np.ones((2, 2, 2), dtype=np.uint8) * 127)
    (input_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    _stub_cli_case_runner(monkeypatch)

    assert (
        main(
            [
                "batch",
                str(input_dir),
                "--profile",
                "XtremeCTII",
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    batch_summary = output_dir / "result.json"
    assert str(batch_summary) in stdout
    summary = json.loads(batch_summary.read_text(encoding="utf-8"))
    assert summary["batch"]["mode"] == "folder"
    assert summary["batch"]["profile"] == "XtremeCTII"
    assert summary["batch"]["case_count"] == 2
    assert [case["case"]["name"] for case in summary["cases"]] == [
        "sample_a",
        "sample_b",
    ]
    assert (output_dir / "sample_a" / "parosol_case.yaml").exists()
    assert (output_dir / "sample_b" / "result.json").exists()


def test_cli_batch_folder_uses_mask_pattern_for_model_profile(
    monkeypatch,
    tmp_path: Path,
):
    input_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "runs"
    input_dir.mkdir()
    mask_dir.mkdir()
    np.save(input_dir / "case_01.npy", np.ones((2, 2, 2), dtype=np.float32))
    np.save(mask_dir / "case_01_SEG.npy", np.ones((2, 2, 2), dtype=np.uint8) * 20)
    _stub_cli_case_runner(monkeypatch)

    assert (
        main(
            [
                "batch",
                str(input_dir),
                "--profile",
                "vertebra",
                "--mask-dir",
                str(mask_dir),
                "--mask-pattern",
                "{stem}_SEG.npy",
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    case_summary = json.loads(
        (output_dir / "case_01" / "result.json").read_text(encoding="utf-8")
    )
    assert case_summary["execution"]["mask"] == str(mask_dir / "case_01_SEG.npy")
    assert case_summary["execution"]["interface"] == "batch-folder"


def test_cli_batch_folder_applies_workflow_template(
    monkeypatch,
    tmp_path: Path,
):
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "runs"
    template_dir = tmp_path / "template"
    input_dir.mkdir()
    template_dir.mkdir()
    np.save(input_dir / "case_a.npy", np.ones((2, 2, 2), dtype=np.uint8) * 100)
    np.save(template_dir / "reference.npy", np.ones((2, 2, 2), dtype=np.uint8) * 100)
    np.save(template_dir / "nodesets.npy", np.ones((2, 2, 2), dtype=np.uint8))
    (template_dir / "workflow.yaml").write_text(
        json.dumps(
            {
                "case": {"name": "reference", "work_dir": "old"},
                "input": {
                    "image": "reference.npy",
                    "image_type": "material_labels",
                    "spacing": [1, 1, 1],
                },
                "materials": {
                    "units": "MPa",
                    "labels": {100: {"name": "bone", "E": 1000, "nu": 0.3}},
                },
                "nodesets": {
                    "fixed": {
                        "type": "label_image",
                        "image": "nodesets.npy",
                        "label": 1,
                    }
                },
                "load_case": {"type": "nodeset", "fixed": []},
                "output": {"export_fields": False},
                "solver": {"mpi_processes": 1},
            }
        ),
        encoding="utf-8",
    )
    _stub_cli_case_runner(monkeypatch)

    assert (
        main(
            [
                "batch",
                str(input_dir),
                "--profile",
                "interactive_custom",
                "--template",
                str(template_dir),
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    generated = (output_dir / "case_a" / "parosol_case.yaml").read_text(
        encoding="utf-8"
    )
    batch_summary = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert "interface: batch-folder" in generated
    assert f"template: {template_dir.resolve()}" in generated
    assert batch_summary["batch"]["profile"] == "interactive_custom"
    assert batch_summary["batch"]["template"] == str(template_dir.resolve())


def _stub_cli_case_runner(monkeypatch):
    def fake_run_case_config(path, *, dry_run=None, work_dir=None):
        import yaml

        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        summary_path = Path(config["output"]["summary"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "case": {"name": config["case"]["name"]},
                    "execution": config["execution"],
                    "load_case": config.get("load_case", {}),
                    "mechanics": {},
                    "failure": {"status": "dry_run"},
                }
            ),
            encoding="utf-8",
        )
        return object()

    monkeypatch.setattr("parosol_py.cli.run_case_config", fake_run_case_config)


def test_run_batch_config_dry_run_executes_real_case_expansion(tmp_path: Path):
    np.save(tmp_path / "material.npy", np.ones((2, 2, 2), dtype=np.float32) * 1000)
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(
        json.dumps(
            {
                "case": {"name": "cube", "work_dir": "runs/cube"},
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "output": {"dry_run": True},
                "batch": {
                    "summary": "runs/batch_summary.json",
                    "cases": [
                        {
                            "name_suffix": "compression_z",
                            "load_case": {
                                "type": "constrained_axial",
                                "axis": "z",
                                "strain": -0.01,
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = run_batch_config(batch_path, dry_run=True)

    case_summary = tmp_path / "runs" / "cube_compression_z" / "summary.json"
    case_config = tmp_path / "runs" / "cube" / "_cases" / "cube_compression_z.json"
    assert case_config.exists()
    assert case_summary.exists()
    assert summary["batch"]["case_count"] == 1
    assert summary["cases"][0]["case"]["name"] == "cube_compression_z"
