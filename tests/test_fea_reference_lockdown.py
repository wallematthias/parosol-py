from __future__ import annotations

from pathlib import Path

import numpy as np

from parosol_py.reference_comparison import write_reference_comparison_bundle
from parosol_py.reference_fixtures import (
    EXPECTED_REFERENCE_FIXTURES,
    ImageGridMetadata,
    fixture_array_xyz,
    load_fixture_array,
    load_reference_fixture,
    validate_reference_fixture,
)
from parosol_py.reference_geometry import (
    ras_to_voxel_indices_zyx,
    voxel_indices_zyx_to_ras,
)
from parosol_py.reference_masks import split_filled_segmentation_compartments
from parosol_py.workflow_geometry import generate_disk_and_nodeset_geometry


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "fea_reference"


def test_reference_fixture_families_are_committed_and_ras():
    assert EXPECTED_REFERENCE_FIXTURES == (
        "xtremect_tibia_mini",
        "vertebra_l4_mini",
        "femur_left_mini",
    )

    expected_labels = {
        "xtremect_tibia_mini": {0, 100, 127},
        "vertebra_l4_mini": {0, 20, 48},
        "femur_left_mini": {0, 2},
    }

    for name in EXPECTED_REFERENCE_FIXTURES:
        fixture = load_reference_fixture(name, fixture_root=FIXTURE_ROOT)

        assert fixture.name == name
        assert fixture.grid.coordinate_system == "RAS"
        assert fixture.grid.units == "mm"
        assert fixture.grid.array_order == "zyx"
        assert fixture.grid.shape_zyx == load_fixture_array(fixture, "labels").shape
        assert fixture.transform_chain[0]["name"] == "source_image_to_input_ras"
        assert fixture.transform_chain[-1]["target_space"] == "visualization_ras_grid"
        assert validate_reference_fixture(fixture) == []

        labels = load_fixture_array(fixture, "labels")
        assert set(np.unique(labels).astype(int)) == expected_labels[name]
        assert labels.dtype == np.uint8
        assert (
            int(np.count_nonzero(labels)) == fixture.provenance["nonzero_label_voxels"]
        )
        assert fixture.workflows


def test_filled_segmentation_split_fills_then_erodes_in_plane():
    segmentation = np.zeros((2, 9, 9), dtype=np.uint8)
    segmentation[:, 2:7, 2:7] = 1
    segmentation[:, 4, 4] = 0

    split = split_filled_segmentation_compartments(segmentation, erosion_iterations=1)

    assert split.full_zyx[:, 4, 4].all()
    assert split.trabecular_zyx[:, 4, 4].all()
    assert split.cortical_zyx[:, 2, 2].all()
    assert not split.trabecular_zyx[:, 2, 2].any()
    assert not np.any(split.trabecular_zyx & split.cortical_zyx)
    np.testing.assert_array_equal(
        split.trabecular_zyx | split.cortical_zyx, split.full_zyx
    )


def test_xtremect_tibia_fixture_records_mini_source_compartment_generation():
    fixture = load_reference_fixture("xtremect_tibia_mini", fixture_root=FIXTURE_ROOT)
    labels = load_fixture_array(fixture, "labels")
    generation = fixture.provenance.get("source_compartment_generation", {})
    source_masks = fixture.provenance.get("source_compartment_masks", {})

    assert fixture.provenance["source_dataset"].startswith(
        "AdvectionModel-grayscale-tibia-density-step-40"
    )
    assert generation["method"] == "filled_segmentation_inplane_erosion"
    assert generation["erosion_iterations"] == 8
    assert generation["source_segmentation_nonzero_voxels"] == 252477
    assert generation["filled_segmentation_voxels"] == 662744
    assert generation["trabecular_contour_voxels"] == 464208
    assert generation["cortical_contour_voxels"] == 198536
    assert set(source_masks) >= {"trabecular", "cortical"}
    for compartment, label_value in (("trabecular", 100), ("cortical", 127)):
        mask_record = source_masks[compartment]
        assert mask_record["label_value"] == label_value
        assert mask_record["generation_method"] == "filled_segmentation_inplane_erosion"
        assert mask_record["fixture_voxels"] == int(
            np.count_nonzero(labels == label_value)
        )
        assert mask_record["fixture_voxels"] > 0
    assert (
        fixture.provenance["fixture_correction"][
            "different_voxels_from_previous_fixture"
        ]
        == 13154
    )
    assert int(np.count_nonzero(labels == 100)) == 97543
    assert int(np.count_nonzero(labels == 127)) == 154934


def test_voxel_coordinate_helpers_use_ras_mm_for_asymmetric_grid():
    grid = ImageGridMetadata(
        shape_zyx=(3, 4, 5),
        spacing_xyz=(2.0, 3.0, 5.0),
        origin_ras=(10.0, 20.0, 30.0),
        direction_ras=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        array_order="zyx",
    )
    indices_zyx = np.asarray([[0, 0, 0], [1, 2, 3], [2, 3, 4]], dtype=np.int64)

    points = voxel_indices_zyx_to_ras(indices_zyx, grid)

    np.testing.assert_allclose(
        points,
        np.asarray(
            [
                [10.0, 20.0, 30.0],
                [16.0, 26.0, 35.0],
                [18.0, 29.0, 40.0],
            ]
        ),
    )
    np.testing.assert_array_equal(ras_to_voxel_indices_zyx(points, grid), indices_zyx)


def test_reference_comparison_bundle_writes_json_and_visual_artifacts(tmp_path):
    grid = ImageGridMetadata(
        shape_zyx=(5, 6, 7),
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_ras=(0.0, 0.0, 0.0),
        direction_ras=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        array_order="zyx",
    )
    anatomy = np.zeros(grid.shape_zyx, dtype=np.float32)
    reference_labels = np.zeros(grid.shape_zyx, dtype=np.uint8)
    replay_labels = np.zeros(grid.shape_zyx, dtype=np.uint8)
    anatomy[1:4, 1:5, 2:6] = 100.0
    reference_labels[1:4, 1:5, 2:6] = 20
    replay_labels[1:4, 1:5, 2:6] = 20
    replay_labels[2, 2, 3] = 0

    bundle = write_reference_comparison_bundle(
        tmp_path,
        fixture_name="toy",
        grid=grid,
        reference_summary={
            "source": "ogo_faim",
            "mechanics": {"top_node_count": 24},
            "transform_chain": [{"name": "input_ras", "matrix": np.eye(4).tolist()}],
        },
        replay_summary={
            "source": "parosol_py",
            "mechanics": {"top_node_count": 23},
            "transform_chain": [{"name": "input_ras", "matrix": np.eye(4).tolist()}],
        },
        anatomy_zyx=anatomy,
        reference_labels_zyx=reference_labels,
        replay_labels_zyx=replay_labels,
        tolerances={"label_dice_min": 0.99, "node_count_delta_max": 0},
    )

    assert bundle.reference_json.name == "reference.json"
    assert bundle.replay_json.name == "replay.json"
    assert bundle.equivalence_json.name == "equivalence.json"
    assert bundle.visual_report.name == "visual_report.html"
    assert bundle.visual_report.exists()
    assert bundle.png_paths
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in bundle.png_paths)
    assert not list(tmp_path.glob("*.csv"))

    html = bundle.visual_report.read_text(encoding="utf-8")
    assert "toy" in html
    assert "label_dice" in bundle.equivalence
    assert bundle.equivalence["passed"] is False
    assert bundle.equivalence["label_overlap"]["different_voxels"] == 1


def test_generic_disk_generator_runs_for_all_reference_fixtures():
    for fixture_name in EXPECTED_REFERENCE_FIXTURES:
        fixture = load_reference_fixture(fixture_name, fixture_root=FIXTURE_ROOT)
        labels_xyz = fixture_array_xyz(fixture, "labels")
        mask_xyz = labels_xyz > 0
        padded_mask = np.pad(mask_xyz, ((0, 0), (0, 0), (0, 4)), constant_values=False)
        padded_material = np.pad(
            labels_xyz.astype(np.uint16),
            ((0, 0), (0, 0), (0, 4)),
            constant_values=0,
        )
        active = np.argwhere(padded_mask)
        lower = active.min(axis=0)
        upper = active.max(axis=0)
        center_xyz = (lower + upper) / 2.0
        center_ras = np.asarray(fixture.grid.origin_ras) + center_xyz * np.asarray(
            fixture.grid.spacing_xyz
        )
        size_xyz = np.maximum(
            (upper - lower + 1) * np.asarray(fixture.grid.spacing_xyz),
            np.asarray(fixture.grid.spacing_xyz),
        )
        top_z = (
            fixture.grid.origin_ras[2] + float(upper[2]) * fixture.grid.spacing_xyz[2]
        )
        editor = {
            "planes": [
                {
                    "name": "Superior support",
                    "contact": "Material disks",
                    "shape": "rectangle",
                    "surface_mode": "project_bounded",
                    "center_ras": [
                        float(center_ras[0]),
                        float(center_ras[1]),
                        float(top_z),
                    ],
                    "normal_ras": [0.0, 0.0, -1.0],
                    "u_axis_ras": [1.0, 0.0, 0.0],
                    "v_axis_ras": [0.0, 1.0, 0.0],
                    "size_mm": [float(size_xyz[0]), float(size_xyz[1])],
                    "thickness_mm": float(fixture.grid.spacing_xyz[2] * 2.0),
                    "intrusion_depth_mm": float(fixture.grid.spacing_xyz[2]),
                }
            ]
        }

        geometry = generate_disk_and_nodeset_geometry(
            editor,
            mask_xyz=padded_mask,
            material_xyz=padded_material,
            spacing=fixture.grid.spacing_xyz,
            origin=fixture.grid.origin_ras,
            disk_labels={"Superior support": 10001},
            nodeset_names={"Superior support": "superior_support"},
            nodeset_labels={"superior_support": 10002},
            nodeset_specs={"superior_support": {"selection": "outer_face_nodes"}},
        )

        assert int(np.count_nonzero(geometry.disk_labels_xyz == 10001)) > 0
        assert int(np.count_nonzero(geometry.nodeset_labels_xyz == 10002)) > 0
        assert geometry.node_sets["superior_support"]


def test_durable_package_code_does_not_use_banned_reference_word():
    banned = "par" + "ity"
    source_root = Path(__file__).resolve().parents[1] / "src" / "parosol_py"
    offenders = []
    for path in source_root.rglob("*.py"):
        if banned in path.read_text(encoding="utf-8").lower():
            offenders.append(path.relative_to(source_root).as_posix())

    assert offenders == []
