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
                        }
                    },
                    "failure": {
                        "failure_generalized_load": {
                            "name": "force",
                            "value": 10 * len(seen),
                            "units": "N",
                        }
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
    assert seen[0]["output"]["summary"].endswith("sample_compression_z/summary.json")
    assert seen[0]["output"]["fields_dir"].endswith("sample_compression_z/fields")
    assert seen[0]["output"]["visualization"].endswith(
        "sample_compression_z/overview.png"
    )
    assert summary["batch"]["case_count"] == 2
    assert summary["postprocess"]["load_history"]["method"] == "nnls"
    assert summary["cases"][1]["load_case"]["direction"] == "x"
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
    assert config["case"]["name"] == "distal_radius"
    assert config["case"]["work_dir"] == str(output_dir / "distal_radius")
    assert config["batch"]["summary"] == str(output_dir / "batch_summary.json")
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
    batch_summary = output_dir / "batch_summary.json"
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
    assert (output_dir / "sample_b" / "summary.json").exists()


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
        (output_dir / "case_01" / "summary.json").read_text(encoding="utf-8")
    )
    assert case_summary["execution"]["mask"] == str(mask_dir / "case_01_SEG.npy")
    assert case_summary["execution"]["interface"] == "batch-folder"


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
