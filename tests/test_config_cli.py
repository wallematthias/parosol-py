import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from parosol_py.cli import main
from parosol_py.config import run_case_config

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "reference"


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
                "load_case": {
                    "type": "constrained_axial",
                    "axis": "z",
                    "strain": -0.01,
                },
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


def test_run_case_config_reads_compressed_npz_label_image(tmp_path: Path):
    labels = np.ones((2, 2, 2), dtype=np.uint8)
    np.savez_compressed(
        tmp_path / "labels.npz",
        labels=labels,
        spacing_xyz=np.asarray([0.5, 0.5, 0.5], dtype=np.float64),
        origin_xyz=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
    )
    (tmp_path / "materials.yaml").write_text(
        "MaterialDefinitions:\n"
        "    Bone:\n"
        "        Type: LinearIsotropic\n"
        "        E: 10000\n"
        "        nu: 0.3\n"
        "MaterialTable:\n"
        "    1: Bone\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {
                    "image": "labels.npz",
                    "image_type": "material_labels",
                    "spacing": "auto",
                    "origin": "auto",
                },
                "materials": {"file": "materials.yaml"},
                "output": {"summary": "summary.json", "dry_run": True},
            }
        ),
        encoding="utf-8",
    )

    result = run_case_config(config_path)

    assert result.summary.dimensions_xyz == (2, 2, 2)
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


def test_run_case_config_applies_named_profiles(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "solver_profile": "standard",
                "output_profile": "quick_summary",
                "output": {"summary": "summary.json"},
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
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    assert captured["outputs"] == ("sed",)
    assert captured["export_dir"] is None


def test_run_case_config_builds_boundary_conditions_from_voxel_nodeset_labels(
    monkeypatch, tmp_path: Path
):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    nodesets = np.zeros((2, 2, 2), dtype=np.uint8)
    nodesets[0, :, :] = 2
    nodesets[1, :, :] = 1
    np.save(tmp_path / "material.npy", material)
    np.save(tmp_path / "nodesets.npy", nodesets)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "nodesets": {
                    "top_plate": {
                        "type": "label_image",
                        "image": "nodesets.npy",
                        "label": 1,
                        "selection": "surface_nodes",
                    },
                    "bottom_plate": {
                        "type": "label_image",
                        "image": "nodesets.npy",
                        "label": 2,
                        "selection": "surface_nodes",
                    },
                },
                "load_case": {
                    "type": "nodeset",
                    "prescribed": [
                        {"nodeset": "top_plate", "dof": "z", "value": "-1%"}
                    ],
                    "fixed": [
                        {"nodeset": "bottom_plate", "dofs": ["x", "y", "z"], "value": 0}
                    ],
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert bc.node_sets["top_plate"]
    assert bc.node_sets["bottom_plate"]
    assert np.any((bc.fixed_coordinates[:, 2] == 2) & (bc.fixed_values == -0.02))
    assert np.any((bc.fixed_coordinates[:, 2] == 0) & (bc.fixed_values == 0.0))


def test_run_case_config_builds_shear_load_case(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "shear",
                    "axis": "z",
                    "direction": "x",
                    "strain": 0.02,
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert captured["load_case_type"] == "shear"
    assert captured["load_direction"] == "x"
    assert np.any(
        (bc.fixed_coordinates[:, 2] == 2)
        & (bc.fixed_coordinates[:, 3] == 0)
        & np.isclose(bc.fixed_values, 0.04)
    )


def test_run_case_config_builds_shear_vector_load_case(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "shear",
                    "axis": "z",
                    "shear_vector": [0.02, 0.03],
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert np.any(
        (bc.fixed_coordinates[:, 2] == 2)
        & (bc.fixed_coordinates[:, 3] == 0)
        & np.isclose(bc.fixed_values, 0.04)
    )
    assert np.any(
        (bc.fixed_coordinates[:, 2] == 2)
        & (bc.fixed_coordinates[:, 3] == 1)
        & np.isclose(bc.fixed_values, 0.06)
    )


def test_run_case_config_maps_density_input_with_poisson_equation(
    monkeypatch, tmp_path: Path
):
    density = np.array([[[0.0, 500.0, 1000.0]]])
    np.save(tmp_path / "density.npy", density)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {
                    "image": "density.npy",
                    "image_type": "density",
                    "spacing": [1, 1, 1],
                },
                "materials": {
                    "density": {
                        "equation": "power",
                        "coefficient": 10000,
                        "exponent": 2,
                        "reference_density": 1000,
                    },
                    "poisson_ratio": {
                        "equation": "linear",
                        "slope": 0.0001,
                        "intercept": 0.2,
                    },
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((3, 1, 1), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    assert captured["material"].tolist() == [[[0.0, 2500.0, 10000.0]]]
    assert captured["poisson_ratio"] == pytest.approx(0.275)


def test_run_case_config_uses_smart_visible_surfaces(monkeypatch, tmp_path: Path):
    material = np.zeros((4, 2, 2), dtype=np.float64)
    material[1:3, 0, 0] = 1000.0
    material[0:4, 0, 1] = 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "constrained_axial",
                    "axis": "z",
                    "displacement": -0.2,
                    "surface": {"mode": "smart", "depth": "auto"},
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((2, 2, 4), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert (0, 0, 3) in bc.node_sets["top"]
    assert (1, 0, 4) in bc.node_sets["top"]


def test_run_case_config_can_export_boundary_conditions(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "uniaxial",
                    "axis": "z",
                    "strain": -0.01,
                },
                "output": {
                    "summary": "summary.json",
                    "dry_run": True,
                    "export_boundary_conditions": True,
                    "boundary_conditions": "bc.json",
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_solve(**kwargs):
        from parosol_py.api import SolveResult, SolveSummary

        return SolveResult(
            input_file=tmp_path / "input.h5",
            command=["parosol"],
            fields={},
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    exported = json.loads((tmp_path / "bc.json").read_text(encoding="utf-8"))
    assert exported["fixed_coordinates"]
    assert "top" in exported["node_sets"]


def test_run_case_config_builds_absolute_displacement_load_case(
    monkeypatch, tmp_path: Path
):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "confined",
                    "axis": "z",
                    "displacement": -0.25,
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert np.any(
        (bc.fixed_coordinates[:, 2] == 2)
        & (bc.fixed_coordinates[:, 3] == 2)
        & np.isclose(bc.fixed_values, -0.25)
    )


def test_run_case_config_builds_axial_absolute_displacement_load_case(
    monkeypatch, tmp_path: Path
):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "constrained_axial",
                    "axis": "z",
                    "displacement": -0.25,
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert captured["strain"] == -0.125
    assert np.any(
        (bc.fixed_coordinates[:, 2] == 2)
        & (bc.fixed_coordinates[:, 3] == 2)
        & np.isclose(bc.fixed_values, -0.25)
    )


def test_run_case_config_builds_uniaxial_load_case(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "uniaxial",
                    "axis": "z",
                    "strain": -0.01,
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert np.all(bc.fixed_coordinates[:, 3] == 2)
    assert np.any(np.isclose(bc.fixed_values, -0.02))


def test_run_case_config_builds_body_weight_load_case(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "body_weight",
                    "axis": "z",
                    "force_n": -90.0,
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert bc.loaded_coordinates.shape[0] == 9
    assert np.sum(bc.loaded_values) == np.float32(-90.0)


def test_run_case_config_builds_torsion_load_case(monkeypatch, tmp_path: Path):
    material = np.ones((3, 3, 3), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "torsion",
                    "axis": "z",
                    "twist_angle_degrees": 1.0,
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((3, 3, 3), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert np.any(
        (bc.fixed_coordinates[:, 2] == 3)
        & (bc.fixed_coordinates[:, 3] == 0)
        & np.isclose(bc.fixed_values, 0.0264070667)
    )


def test_run_case_config_builds_bending_load_case(monkeypatch, tmp_path: Path):
    material = np.ones((3, 3, 3), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "bending",
                    "axis": "z",
                    "bending_angle_degrees": 1.0,
                    "neutral_axis_angle_degrees": 90.0,
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((3, 3, 3), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert np.any(
        (bc.fixed_coordinates[:, 0] == 0)
        & (bc.fixed_coordinates[:, 2] == 3)
        & (bc.fixed_coordinates[:, 3] == 2)
        & np.isclose(bc.fixed_values, -0.0130903013)
    )


def test_run_case_config_builds_confined_load_case(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "load_case": {
                    "type": "confined",
                    "axis": "z",
                    "strain": -0.01,
                },
                "output": {"summary": "summary.json", "dry_run": True},
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
            summary=SolveSummary((2, 2, 2), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    bc = captured["boundary_conditions"]
    assert np.any(
        (bc.fixed_coordinates[:, 2] == 2)
        & (bc.fixed_coordinates[:, 3] == 2)
        & np.isclose(bc.fixed_values, -0.02)
    )
    assert np.any(
        (bc.fixed_coordinates[:, 2] == 2)
        & (bc.fixed_coordinates[:, 3] == 0)
        & np.isclose(bc.fixed_values, 0.0)
    )


def test_cli_run_and_summarize_legacy(tmp_path: Path):
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

    out = tmp_path / "old_summary.json"
    assert (
        main(
            [
                "summarize-legacy",
                "--analysis",
                str(FIXTURE_ROOT / "SAMPLE_HOM_LS_analysis.txt"),
                "--pistoia",
                str(FIXTURE_ROOT / "SAMPLE_HOM_LS_pistoia.txt"),
                "-o",
                str(out),
            ]
        )
        == 0
    )
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["reference"]["pistoia"]["failure_load"]["fz"] == -4741.0
