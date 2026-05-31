import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from parosol_py.cli import main
from parosol_py.config import run_case_config


def test_run_case_config_dry_run_writes_summary_json(tmp_path: Path):
    material = np.ones((3, 3, 3), dtype=np.float64) * 1000.0
    material_path = tmp_path / "material.npy"
    np.save(material_path, material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "case": {"name": "cube", "work_dir": "run"},
                "input": {
                    "image": "material.npy",
                    "image_type": "material_mpa",
                    "spacing": [1.0, 1.0, 1.0],
                },
                "load_case": {"type": "axial", "axis": "z", "strain": -0.01},
                "solver": {"outputs": ["sed"], "tolerance": 1e-6, "level": 2},
                "output": {"summary": "run/cube_summary.json", "dry_run": True},
            }
        ),
        encoding="utf-8",
    )

    result = run_case_config(config_path)

    assert result.input_file.exists()
    summary = json.loads(
        (tmp_path / "run" / "cube_summary.json").read_text(encoding="utf-8")
    )
    assert summary["case"]["name"] == "cube"
    assert summary["load_case"]["axis"] == "z"
    assert summary["failure"]["status"] == "not_computed"


def test_run_case_config_reads_image_metadata_for_auto_spacing(tmp_path: Path):
    material_zyx = np.ones((2, 2, 2), dtype=np.float32) * 1000.0
    image = sitk.GetImageFromArray(material_zyx)
    image.SetSpacing((0.5, 0.5, 0.5))
    image.SetOrigin((1.0, 2.0, 3.0))
    image_path = tmp_path / "material.mha"
    sitk.WriteImage(image, str(image_path))
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {
                    "image": "material.mha",
                    "image_type": "material_mpa",
                    "spacing": "auto",
                    "origin": "auto",
                },
                "output": {"summary": "summary.json", "dry_run": True},
            }
        ),
        encoding="utf-8",
    )

    result = run_case_config(config_path)

    assert result.summary.spacing == (0.5, 0.5, 0.5)
    assert result.summary.origin == (1.0, 2.0, 3.0)


def test_run_case_config_can_disable_field_export(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "output": {"summary": "summary.json", "export_fields": False},
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_solve(**kwargs):
        captured.update(kwargs)
        from parosol_py.api import SolveResult, SolveSummary

        return SolveResult(
            input_file=tmp_path / "input.h5",
            command=["parosol"],
            fields={},
            summary=SolveSummary(
                dimensions_xyz=(2, 2, 2),
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
            ),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    assert captured["export_dir"] is None


def test_cli_run_and_summarize_faim(tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "output": {"summary": "summary.json"},
            }
        ),
        encoding="utf-8",
    )

    assert main(["run", str(config_path), "--dry-run"]) == 0
    assert (tmp_path / "summary.json").exists()

    old_root = Path("/Users/matthias.walle/Documents/10_Data/fea_test")
    out = tmp_path / "old_summary.json"
    assert (
        main(
            [
                "summarize-faim",
                "--analysis",
                str(old_root / "VITD_0003_RL_M06_HOM_LS_analysis.txt"),
                "--pistoia",
                str(old_root / "VITD_0003_RL_M06_HOM_LS_pistoia.txt"),
                "-o",
                str(out),
            ]
        )
        == 0
    )
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["faim"]["pistoia"]["failure_load"]["fz"] == -4741.0
