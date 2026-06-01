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
    assert summary["batch"]["case_count"] == 2
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
