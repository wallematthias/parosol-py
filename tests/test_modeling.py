import json
from pathlib import Path

import numpy as np
import pytest

from parosol_py.modeling import build_model


def test_spine_compression_model_generates_pmma_disks_and_bc_sets(tmp_path: Path):
    density = np.zeros((8, 7, 6), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:5, 2:4] = 800.0
    mask[2:6, 2:5, 2:4] = 2
    mask[3:5, 3:4, 3:5] = 1
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 2, "process": 1},
            "geometry": {"pmma_thickness_mm": 2, "axis": "z"},
            "outputs": {
                "material_image": "model/material.nii.gz",
                "nodeset_image": "model/nodesets.nii.gz",
                "manifest": "model/model.json",
                "qc_image": "model/qc.png",
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {
                "equation": "linear",
                "slope": 10.0,
                "intercept": 0.0,
                "mask_threshold": 0.0,
            },
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    assert built.material.shape[0] == density.shape[0] + 4
    assert set(built.node_sets) >= {"inferior", "superior"}
    assert len(built.node_sets["inferior"]) > 0
    assert len(built.node_sets["superior"]) > 0
    assert built.boundary_conditions.fixed_coordinates.shape[0] > 0
    assert built.element_sets["inferior_disk"] > 0
    assert built.element_sets["superior_disk"] > 0
    assert built.exported["material_image"].exists()
    assert built.exported["nodeset_image"].exists()
    assert built.exported["qc_image"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    manifest = json.loads(built.exported["manifest"].read_text(encoding="utf-8"))
    assert manifest["model"]["type"] == "spine_compression"
    assert manifest["materials"]["pmma"]["E"] == pytest.approx(2500.0)


def test_proximal_femur_model_generates_caps_and_sideways_fall_sets(tmp_path: Path):
    density = np.zeros((7, 8, 9), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[1:6, 2:6, 2:7] = 700.0
    mask[1:6, 2:6, 2:7] = 2
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "proximal_femur",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "side": "left",
            "geometry": {"pmma_thickness_mm": 2},
            "outputs": {
                "material_image": "model/material.nii.gz",
                "nodeset_image": "model/nodesets.nii.gz",
                "manifest": "model/model.json",
                "qc_image": "model/qc.png",
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {
                "equation": "linear",
                "slope": 12.0,
                "intercept": 0.0,
                "mask_threshold": 0.0,
            },
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "sideways_fall", "displacement": -1.0},
    )

    assert set(built.node_sets) >= {
        "femoral_head_pmma",
        "greater_trochanter_pmma",
        "distal_femur",
    }
    assert all(len(nodes) > 0 for nodes in built.node_sets.values())
    assert built.element_sets["femoral_head_cap"] > 0
    assert built.element_sets["greater_trochanter_cap"] > 0
    assert built.boundary_conditions.fixed_coordinates.shape[0] > 0
    assert np.min(built.boundary_conditions.fixed_values) < 0.0
    assert built.exported["qc_image"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_model_builder_rejects_missing_spine_labels(tmp_path: Path):
    density = np.ones((4, 4, 4), dtype=np.float32)
    mask = np.full_like(density, 2, dtype=np.uint8)
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    with pytest.raises(ValueError, match="process"):
        build_model(
            {
                "type": "vertebra",
                "density_image": "density.npy",
                "mask_image": "mask.npy",
                "labels": {"body": 2, "process": 1},
            },
            base_dir=tmp_path,
            material_config={"density": {"equation": "linear"}},
        )
