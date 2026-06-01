import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from parosol_py.api import SolveResult, SolveSummary
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


def test_run_case_config_prefers_postprocess_pistoia(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "postprocess": {
                    "pistoia": {
                        "criterion": "pistoia",
                        "critical_strain": 0.012,
                        "critical_volume_percent": 5.0,
                    }
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

    assert captured["failure_criterion"] == "pistoia"
    assert captured["critical_strain"] == pytest.approx(0.012)
    assert captured["critical_volume_percent"] == pytest.approx(5.0)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["failure"]["critical_strain"] == pytest.approx(0.012)
    assert summary["failure"]["critical_volume_percent"] == pytest.approx(5.0)


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


def test_run_case_config_uses_model_section_for_dry_run(tmp_path: Path):
    density = np.zeros((6, 5, 4), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[1:5, 1:4, 1:3] = 800.0
    mask[1:5, 1:4, 1:3] = 2
    mask[2:4, 2:3, 2:4] = 1
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "case": {"name": "vertebra_model"},
                "model": {
                    "type": "vertebra",
                    "density_image": "density.npy",
                    "mask_image": "mask.npy",
                    "labels": {"body": 2, "process": 1},
                    "geometry": {"pmma_thickness_mm": 1, "axis": "z"},
                    "outputs": {
                        "material_image": "model/material.nii.gz",
                        "nodeset_image": "model/nodesets.nii.gz",
                        "manifest": "model/model.json",
                        "qc_image": "model/qc.png",
                    },
                },
                "materials": {
                    "density": {
                        "E": {
                            "equation": "linear",
                            "slope": 10.0,
                            "intercept": 0.0,
                        },
                        "nu": 0.3,
                    },
                    "pmma": {"E": 2500, "nu": 0.3},
                },
                "load_case": {"type": "spine_compression", "displacement": -0.2},
                "output": {
                    "summary": "summary.json",
                    "dry_run": True,
                    "visualization": "overview.png",
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_case_config(config_path)

    assert result.summary.dimensions_xyz == (4, 5, 8)
    assert (tmp_path / "model" / "material.nii.gz").exists()
    assert (tmp_path / "model" / "nodesets.nii.gz").exists()
    assert (tmp_path / "model" / "qc.png").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["model"]["type"] == "spine_compression"
    assert summary["model"]["node_sets"]["inferior"] > 0


def test_model_overview_uses_postprocess_mask_for_fields(monkeypatch, tmp_path: Path):
    density = np.zeros((6, 5, 4), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[1:5, 1:4, 1:3] = 800.0
    mask[1:5, 1:4, 1:3] = 2
    mask[2:4, 2:3, 2:4] = 1
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "case": {"name": "vertebra_model"},
                "model": {
                    "type": "vertebra",
                    "density_image": "density.npy",
                    "mask_image": "mask.npy",
                    "labels": {"body": 2, "process": 1},
                    "geometry": {"pmma_thickness_mm": 1, "axis": "z"},
                },
                "materials": {
                    "density": {"equation": "linear", "slope": 10.0},
                    "poisson_ratio": 0.3,
                    "pmma": {"E": 2500, "nu": 0.3},
                },
                "load_case": {"type": "spine_compression", "displacement": -0.2},
                "postprocess": {"fields": {"mask_to_segmentation": True}},
                "output": {
                    "summary": "summary.json",
                    "visualization": "overview.png",
                    "fields": ["sed"],
                    "export_fields": True,
                    "fields_dir": "fields",
                },
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_solve(**kwargs):
        captured["postprocess_mask"] = kwargs.get("postprocess_mask")
        export_dir = Path(kwargs["export_dir"])
        export_dir.mkdir(parents=True, exist_ok=True)
        sed_path = export_dir / "sed.nii.gz"
        field = np.ones_like(kwargs["material"], dtype=np.float32)
        field = np.where(kwargs["postprocess_mask"], field, 0.0)
        image = sitk.GetImageFromArray(field)
        image.SetSpacing(kwargs["spacing"])
        image.SetOrigin(kwargs["origin"])
        sitk.WriteImage(image, str(sed_path))
        return SolveResult(
            input_file=tmp_path / "input.h5",
            command=["parosol"],
            fields={"sed": np.ones(int(np.count_nonzero(kwargs["material"] > 0)))},
            summary=SolveSummary(
                tuple(int(v) for v in np.transpose(kwargs["material"], (2, 1, 0)).shape),
                kwargs["spacing"],
                kwargs["origin"],
            ),
            exported={"sed": sed_path},
            diagnostics={},
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)
    seen = {}

    def fake_write_case_overview(material_xyz, **kwargs):
        seen["field_mask"] = kwargs.get("field_mask_xyz")
        seen["field"] = kwargs.get("field_xyz")
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x89PNG\r\n\x1a\n")
        return out

    monkeypatch.setattr("parosol_py.config.write_case_overview", fake_write_case_overview)

    run_case_config(config_path)

    assert captured["postprocess_mask"] is not None
    assert seen["field_mask"] is not None
    assert np.all(np.isnan(seen["field"][~seen["field_mask"]]))


def test_model_section_uses_model_load_axis_for_sideways_fall(
    monkeypatch, tmp_path: Path
):
    density = np.zeros((7, 8, 9), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[1:6, 2:6, 2:7] = 700.0
    mask[1:6, 2:6, 2:7] = 2
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "type": "proximal_femur",
                    "density_image": "density.npy",
                    "mask_image": "mask.npy",
                    "labels": {"femur": 2},
                    "geometry": {"cap_axis": "y", "pmma_thickness_mm": 1},
                },
                "materials": {
                    "density": {"equation": "linear", "slope": 10.0},
                    "poisson_ratio": 0.3,
                    "pmma": {"E": 2500, "nu": 0.3},
                },
                "load_case": {"type": "sideways_fall", "displacement": 1.0},
                "output": {"summary": "summary.json", "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_solve(**kwargs):
        captured.update(kwargs)
        return SolveResult(
            input_file=tmp_path / "input.h5",
            command=["parosol"],
            fields={},
            summary=SolveSummary(
                tuple(int(v) for v in np.transpose(kwargs["material"], (2, 1, 0)).shape),
                kwargs["spacing"],
                kwargs["origin"],
            ),
            diagnostics={},
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    assert captured["test_axis"] == "y"
    assert captured["load_direction"] == "y"
    assert captured["load_case_type"] == "sideways_fall"


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


def test_run_case_config_accepts_label_material_map(monkeypatch, tmp_path: Path):
    labels = np.asarray([[[100, 127]]], dtype=np.uint8)
    np.save(tmp_path / "labels.npy", labels)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {
                    "image": "labels.npy",
                    "image_type": "material_labels",
                    "spacing": [1, 1, 1],
                },
                "materials": {
                    "labels": {
                        100: {
                            "name": "trabecular_bone",
                            "E": 6829,
                            "nu": 0.25,
                        },
                        127: {
                            "name": "cortical_bone",
                            "E": 8748,
                            "nu": 0.3,
                        },
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
            summary=SolveSummary((2, 1, 1), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    run_case_config(config_path)

    assert captured["material"].tolist() == [[[6829.0, 8748.0]]]
    np.testing.assert_allclose(captured["poisson_ratio"], [[[0.25, 0.3]]])


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


def test_run_case_config_writes_overview_png_for_dry_run(monkeypatch, tmp_path: Path):
    material = np.ones((3, 3, 3), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "case": {"name": "cube"},
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "output": {
                    "summary": "summary.json",
                    "dry_run": True,
                    "visualization": "cube_overview.png",
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
            summary=SolveSummary((3, 3, 3), (1, 1, 1), (0, 0, 0)),
        )

    monkeypatch.setattr("parosol_py.config.solve", fake_solve)

    result = run_case_config(config_path)

    overview = tmp_path / "cube_overview.png"
    assert overview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.exported["overview"] == overview.resolve()


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


def test_run_case_config_uses_output_fields_as_solver_outputs(
    monkeypatch, tmp_path: Path
):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "output": {
                    "summary": "summary.json",
                    "fields": ["sed", "effective_strain", "von_mises"],
                },
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

    assert captured["outputs"] == ("sed", "effective_strain", "von_mises")


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
                        "E": {
                            "equation": "power",
                            "coefficient": 10000,
                            "exponent": 2,
                            "reference_density": 1000,
                        },
                        "nu": {
                            "equation": "linear",
                            "slope": 0.0001,
                            "intercept": 0.2,
                        },
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


def test_run_case_config_can_export_node_and_element_sets(monkeypatch, tmp_path: Path):
    material = np.ones((2, 2, 2), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [1, 1, 1]},
                "output": {
                    "summary": "summary.json",
                    "dry_run": True,
                    "export_sets": True,
                    "set_formats": ["json", "vtk"],
                    "sets_dir": "sets",
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

    result = run_case_config(config_path)

    assert (tmp_path / "sets" / "element_sets.json").exists()
    assert (tmp_path / "sets" / "node_sets.json").exists()
    assert any(path.name.endswith("_nodes.vtk") for path in result.exported.values())


def test_run_case_config_can_coarsen_material_before_solving(
    monkeypatch, tmp_path: Path
):
    material = np.ones((4, 4, 4), dtype=np.float64) * 1000.0
    np.save(tmp_path / "material.npy", material)
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "input": {"image": "material.npy", "spacing": [0.5, 0.5, 0.5]},
                "preprocessing": {"coarsen": {"factor": 2, "reducer": "mean"}},
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

    assert captured["material"].shape == (2, 2, 2)
    assert captured["spacing"] == (1.0, 1.0, 1.0)


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


def test_cli_shortcut_runs_direct_profile_and_records_execution(tmp_path: Path):
    labels = np.ones((3, 3, 3), dtype=np.uint8) * 100
    image_path = tmp_path / "distal_radius.npy"
    output_dir = tmp_path / "out"
    np.save(image_path, labels)

    assert (
        main(
            [
                str(image_path),
                "--profile",
                "XtremeCTII",
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    generated = output_dir / "parosol_case.yaml"
    summary_path = output_dir / "summary.json"
    assert generated.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["case"]["name"] == "distal_radius"
    assert summary["execution"]["interface"] == "shortcut"
    assert summary["execution"]["profile"] == "XtremeCTII"
    assert summary["execution"]["image"] == str(image_path.resolve())
    assert summary["execution"]["generated_config"] == str(generated)


def test_cli_shortcut_direct_profile_uses_auto_spacing_for_metadata_images(
    tmp_path: Path,
):
    image = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.uint8) * 100)
    image.SetSpacing((0.0607, 0.0607, 0.0607))
    image_path = tmp_path / "distal_radius.mha"
    output_dir = tmp_path / "out"
    sitk.WriteImage(image, str(image_path))

    assert (
        main(
            [
                str(image_path),
                "--profile",
                "XtremeCTII",
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    generated_text = (output_dir / "parosol_case.yaml").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "spacing: auto" in generated_text
    assert "origin: auto" in generated_text
    assert summary["image"]["spacing"] == [0.0607, 0.0607, 0.0607]


def test_cli_shortcut_accepts_aim_version_suffix(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "STRAMBO_0003_RL_Y04.AIM;1"
    output_dir = tmp_path / "out"

    def fake_read_aim(path):
        assert path.endswith("STRAMBO_0003_RL_Y04.AIM;1")
        return (
            np.ones((3, 3, 3), dtype=np.uint8) * 100,
            {"element_size": (0.0607, 0.0607, 0.0607), "position": (0, 0, 0)},
        )

    monkeypatch.setattr("parosol_py.api.read_aim", fake_read_aim)

    assert (
        main(
            [
                str(image_path),
                "--profile",
                "XtremeCTI",
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["case"]["name"] == "STRAMBO_0003_RL_Y04"
    assert summary["execution"]["image"] == str(image_path.resolve())
    assert summary["image"]["spacing"] == [0.0607, 0.0607, 0.0607]


def test_cli_shortcut_runs_model_profile_with_standard_mask_argument(tmp_path: Path):
    density = np.zeros((8, 7, 6), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:5, 2:4] = 800.0
    mask[2:6, 2:5, 2:4] = 20
    mask[3:6, 3:5, 4:6] = 48
    density_path = tmp_path / "10001_QCT.npy"
    mask_path = tmp_path / "10001_SEG.npy"
    output_dir = tmp_path / "vertebra_out"
    np.save(density_path, density)
    np.save(mask_path, mask)

    assert (
        main(
            [
                str(density_path),
                "--mask",
                str(mask_path),
                "--profile",
                "vertebra",
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    generated = output_dir / "parosol_case.yaml"
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert generated.exists()
    assert summary["execution"]["profile"] == "vertebra"
    assert summary["execution"]["mask"] == str(mask_path.resolve())
    assert summary["model"]["type"] == "spine_compression"
    assert (output_dir / "model" / "material.nii.gz").exists()
