import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from parosol_py.core import BoundaryConditionSet
from parosol_py.modeling import build_model
from parosol_py.modeling.common import (
    _shift_boundary_conditions_for_crop,
    load_density_and_mask,
    material_from_density,
    projected_caps_from_mask,
)
from parosol_py.modeling.io import read_image_zyx
from parosol_py.modeling.alignment import (
    align_mask_to_reference,
    estimate_rigid_icp,
    orient_reference_points,
    read_reference_points,
    surface_points_from_mask,
)
from parosol_py.modeling.common import displacement_from_load_case
from parosol_py.modeling.femur import (
    detect_lesser_trochanter_cut_z,
    standardize_femur_shaft_length,
)
from parosol_py.modeling.workflow_replay import (
    _resolve_bbox_relative_editor,
    _scale_reference_space_editor,
    _workflow_active_mask,
    _workflow_model_mask,
    build_workflow_replay_model,
)


def test_model_image_reader_canonicalizes_nifti_direction(tmp_path: Path):
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 2.0, 3.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    path = tmp_path / "flipped_y.nii.gz"
    sitk.WriteImage(image, str(path))

    data, spacing, origin = read_image_zyx(path)

    expected = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS"))
    np.testing.assert_array_equal(data, expected)
    assert spacing == pytest.approx((1.0, 2.0, 3.0))
    assert origin == pytest.approx((10.0, 16.0, 30.0))


def test_qc_crop_shifts_boundary_condition_coordinates():
    bc = BoundaryConditionSet(
        fixed_coordinates=np.asarray(
            [
                [10, 20, 30, 2],
                [2, 20, 30, 2],
            ],
            dtype=np.uint16,
        ),
        fixed_values=np.asarray([1.0, 2.0], dtype=np.float32),
        loaded_coordinates=np.asarray([[11, 21, 31, 0]], dtype=np.uint16),
        loaded_values=np.asarray([3.0], dtype=np.float32),
        node_sets={"top": [(10, 20, 30), (2, 20, 30)]},
    )

    shifted = _shift_boundary_conditions_for_crop(bc, offset_xyz=(4, 5, 6))

    np.testing.assert_array_equal(
        shifted.fixed_coordinates,
        np.asarray([[6, 15, 24, 2]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        shifted.loaded_coordinates,
        np.asarray([[7, 16, 25, 0]], dtype=np.uint16),
    )
    assert shifted.fixed_values.tolist() == [1.0]
    assert shifted.loaded_values.tolist() == [3.0]
    assert shifted.node_sets["top"] == [(6, 15, 24)]


def test_model_geometry_numeric_isotropic_spacing_resamples_to_target(
    tmp_path: Path,
):
    density = np.ones((5, 6, 7), dtype=np.float32)
    mask = np.ones_like(density, dtype=np.uint8)
    density_image = sitk.GetImageFromArray(density)
    mask_image = sitk.GetImageFromArray(mask)
    density_image.SetSpacing((0.8, 0.8, 0.8))
    mask_image.SetSpacing((0.8, 0.8, 0.8))
    sitk.WriteImage(density_image, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_image, str(tmp_path / "mask.nii.gz"))

    resampled_density, resampled_mask, spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "geometry": {"isotropic_spacing": 1.0},
        },
        base_dir=tmp_path,
    )

    assert spacing == pytest.approx((1.0, 1.0, 1.0))
    assert resampled_density.shape != density.shape
    assert resampled_mask.shape == resampled_density.shape


def test_model_geometry_spacing_tolerance_skips_unnecessary_resampling(
    tmp_path: Path,
):
    density = np.ones((5, 6, 7), dtype=np.float32)
    mask = np.ones_like(density, dtype=np.uint8)
    density_image = sitk.GetImageFromArray(density)
    mask_image = sitk.GetImageFromArray(mask)
    density_image.SetSpacing((0.6069, 0.6069, 0.6069))
    mask_image.SetSpacing((0.6069, 0.6069, 0.6069))
    sitk.WriteImage(density_image, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_image, str(tmp_path / "mask.nii.gz"))

    resampled_density, _resampled_mask, spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "geometry": {
                "resample_spacing": [0.607, 0.607, 0.607],
                "spacing_tolerance_mm": 0.001,
            },
        },
        base_dir=tmp_path,
    )

    assert spacing == pytest.approx((0.6069, 0.6069, 0.6069))
    assert resampled_density.shape == density.shape


def test_model_preprocessing_smooths_density_and_labels_together(tmp_path: Path):
    density = np.zeros((9, 9, 9), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[4, 4, 4] = 100.0
    mask[4, 4, 4] = 2
    image = sitk.GetImageFromArray(density)
    labels = sitk.GetImageFromArray(mask)
    image.SetSpacing((1.0, 1.0, 1.0))
    labels.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(labels, str(tmp_path / "mask.nii.gz"))

    smoothed_density, smoothed_mask, _spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
        },
        base_dir=tmp_path,
        preprocessing_config={
            "smooth": {
                "enabled": True,
                "sigma_mm": 1.0,
                "density": True,
                "labels": True,
                "label_threshold": 0.02,
            }
        },
    )

    assert 0.0 < smoothed_density[4, 4, 3] < 100.0
    assert smoothed_density[4, 4, 4] < 100.0
    assert np.count_nonzero(smoothed_mask == 2) > 1


def test_model_preprocessing_smooth_spacing_guard(tmp_path: Path):
    density = np.zeros((9, 9, 9), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[4, 4, 4] = 100.0
    mask[4, 4, 4] = 2

    for spacing, folder in (((1.0, 1.0, 1.0), "fine"), ((3.0, 3.0, 3.0), "coarse")):
        case_dir = tmp_path / folder
        case_dir.mkdir()
        image = sitk.GetImageFromArray(density)
        labels = sitk.GetImageFromArray(mask)
        image.SetSpacing(spacing)
        labels.SetSpacing(spacing)
        sitk.WriteImage(image, str(case_dir / "density.nii.gz"))
        sitk.WriteImage(labels, str(case_dir / "mask.nii.gz"))

    smooth_cfg = {
        "enabled": True,
        "when_spacing_above_mm": 2.0,
        "sigma_mm": 3.0,
        "density": True,
        "labels": True,
        "label_threshold": 0.02,
    }

    fine_density, fine_mask, _spacing, _origin = load_density_and_mask(
        {"density_image": "density.nii.gz", "mask_image": "mask.nii.gz"},
        base_dir=tmp_path / "fine",
        preprocessing_config={"smooth": smooth_cfg},
    )
    coarse_density, coarse_mask, _spacing, _origin = load_density_and_mask(
        {"density_image": "density.nii.gz", "mask_image": "mask.nii.gz"},
        base_dir=tmp_path / "coarse",
        preprocessing_config={"smooth": smooth_cfg},
    )

    assert fine_density[4, 4, 4] == pytest.approx(100.0)
    assert np.count_nonzero(fine_mask == 2) == 1
    assert coarse_density[4, 4, 4] < 100.0
    assert np.count_nonzero(coarse_mask == 2) > 1


def test_material_from_density_uses_nested_mulder_law_inside_active_contour():
    density = np.array(
        [
            [[0.0, 500.0, 750.0]],
            [[0.0, 500.0, 750.0]],
        ],
        dtype=np.float64,
    )
    active_contour = np.array(
        [
            [[True, True, True]],
            [[False, False, False]],
        ],
        dtype=bool,
    )

    material, nu = material_from_density(
        density,
        active_contour,
        material_config={
            "density": {
                "E": {
                    "equation": "mulder2007",
                    "floor_e_mpa": 2.0,
                },
                "nu": 0.29,
            },
        },
    )

    assert material.tolist() == [
        [[2.0, 6670.0, 12920.0]],
        [[0.0, 0.0, 0.0]],
    ]
    assert nu == pytest.approx(0.29)


def test_material_from_density_applies_optional_input_transform():
    density = np.array([[[1000.0]]], dtype=np.float64)
    active = np.array([[[True]]], dtype=bool)

    material, nu = material_from_density(
        density,
        active,
        material_config={
            "density": {
                "input_transform": {
                    "equation": "linear",
                    "slope": 1.06,
                    "intercept": 38.9,
                },
                "E": {
                    "equation": "power",
                    "coefficient": 10500.0,
                    "exponent": 2.29,
                    "reference_density": 1000.0,
                },
                "nu": 0.3,
            },
        },
    )

    assert material[0, 0, 0] > 10500.0
    assert nu == pytest.approx(0.3)


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

    assert built.material.shape[0] >= density.shape[0] + 4
    assert set(built.node_sets) >= {"inferior", "superior"}
    assert len(built.node_sets["inferior"]) > 0
    assert len(built.node_sets["superior"]) > 0
    axis_values = np.asarray(built.node_sets["inferior"])[:, 2]
    assert np.all(axis_values == np.min(axis_values))
    axis_values = np.asarray(built.node_sets["superior"])[:, 2]
    assert np.all(axis_values == np.max(axis_values))
    assert built.boundary_conditions.fixed_coordinates.shape[0] > 0
    assert built.element_sets["inferior_disk"] > 0
    assert built.element_sets["superior_disk"] > 0
    assert built.exported["material_image"].exists()
    assert built.exported["nodeset_image"].exists()
    assert built.exported["qc_image"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    manifest = json.loads(built.exported["manifest"].read_text(encoding="utf-8"))
    assert manifest["model"]["type"] == "spine_compression"
    assert manifest["materials"]["pmma"]["E"] == pytest.approx(2500.0)


def test_spine_pmma_disks_use_flat_outer_faces_not_side_walls(tmp_path: Path):
    density = np.zeros((10, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:8, 2:6, 2:6] = 800.0
    mask[2:8, 2:6, 2:6] = 20
    mask[4:8, 1, 2:6] = 20
    mask[3:6, 3:5, 6] = 48
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 20, "process": 48},
            "geometry": {
                "pmma_thickness_voxels": 2,
                "endplate_depth_voxels": 1,
                "axis": "z",
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    inferior_nodes = np.asarray(built.node_sets["inferior"])
    superior_nodes = np.asarray(built.node_sets["superior"])
    assert np.ptp(inferior_nodes[:, 2]) == 0
    assert np.ptp(superior_nodes[:, 2]) == 0
    assert built.element_sets["inferior_disk"] <= 48
    assert built.element_sets["superior_disk"] <= 48


def test_projected_caps_from_mask_fills_short_internal_footprint_gaps():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[5:7, 2:5, 2:10] = True
    mask[5:7, 7:10, 2:10] = True

    inferior, superior = projected_caps_from_mask(
        mask,
        axis="x",
        thickness_voxels=4,
        intrusion_depth_voxels=2,
        shape="anatomy",
    )

    assert inferior.any()
    assert superior.any()
    assert not np.any(inferior & mask)
    assert not np.any(superior & mask)
    assert np.all(superior[7:9, 5:7, 2:10])
    cap_x = np.where(superior)[0]
    assert cap_x.max() - cap_x.min() + 1 <= 4


def test_projected_caps_from_mask_intrusion_keeps_requested_total_thickness():
    mask = np.zeros((24, 12, 12), dtype=bool)
    mask[8:10, 2:10, 2:10] = True
    mask[14:16, 5:7, 5:7] = True

    _inferior, superior = projected_caps_from_mask(
        mask,
        axis="x",
        thickness_voxels=6,
        intrusion_depth_voxels=5,
        shape="anatomy",
    )

    assert superior.any()
    assert not np.any(superior & mask)
    cap_x = np.where(superior)[0]
    assert cap_x.max() - cap_x.min() + 1 <= 6


def test_spine_disk_geometry_accepts_explicit_target_thickness_and_intrusion(
    tmp_path: Path,
):
    density = np.zeros((9, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:7, 2:6, 2:5] = 800.0
    mask[2:7, 2:6, 2:5] = 20
    mask[4:6, 3:5, 5:7] = 48
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 20, "process": 48},
            "geometry": {
                "axis": "z",
                "disk": {
                    "target_label": 20,
                    "thickness_voxels": 1,
                    "intrusion_depth_voxels": 2,
                },
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    assert built.metadata["model"]["disk"] == {
        "target_label": "20",
        "shape": "anatomy",
        "thickness_voxels": 1,
        "intrusion_depth_voxels": 2,
        "method": "projected_cap",
    }
    assert built.element_sets["inferior_disk"] > 0
    assert built.element_sets["superior_disk"] > 0
    assert np.all(built.material[built.postprocess_mask] != 2500.0)


@pytest.mark.parametrize("shape", ["anatomy", "square", "round", "hex"])
def test_spine_disk_geometry_supports_contact_shapes(tmp_path: Path, shape: str):
    density = np.zeros((9, 10, 10), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:7, 2:7, 2:6] = 800.0
    mask[2:7, 2:7, 2:6] = 20
    density[2:5, 7:9, 2:4] = 800.0
    mask[2:5, 7:9, 2:4] = 20
    mask[4:6, 4:6, 6:8] = 48
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 20, "process": 48},
            "geometry": {
                "axis": "z",
                "disk": {
                    "target_label": 20,
                    "shape": shape,
                    "thickness_voxels": 1,
                    "intrusion_depth_voxels": 2,
                },
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    assert built.metadata["model"]["disk"]["shape"] == shape
    assert built.element_sets["inferior_disk"] > 0
    assert built.element_sets["superior_disk"] > 0


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
        load_case_config={"type": "sideways_fall", "displacement": 1.0},
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
    assert np.max(built.boundary_conditions.fixed_values) > 0.0
    femoral_head = np.asarray(built.node_sets["femoral_head_pmma"])[:, :3]
    greater_trochanter = np.asarray(built.node_sets["greater_trochanter_pmma"])[:, :3]
    fixed_coords = np.asarray(built.boundary_conditions.fixed_coordinates)
    gt_nodes = {tuple(node) for node in built.node_sets["greater_trochanter_pmma"]}
    distal_nodes = {tuple(node) for node in built.node_sets["distal_femur"]}
    gt_dofs = {int(coord[3]) for coord in fixed_coords if tuple(coord[:3]) in gt_nodes}
    distal_dofs = {
        int(coord[3]) for coord in fixed_coords if tuple(coord[:3]) in distal_nodes
    }
    assert femoral_head[:, 1].mean() < greater_trochanter[:, 1].mean()
    assert gt_dofs == {1}
    assert distal_dofs == {0, 2}
    assert built.metadata["model"]["load_axis"] == "y"
    assert built.metadata["model"]["load_direction"] == "y"
    assert built.metadata["model"]["caps"]["target_label"] == "2"
    assert built.metadata["model"]["caps"]["shape"] == "anatomy"
    assert built.exported["qc_image"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_model_crop_to_bb_uses_declared_model_labels(tmp_path: Path):
    density = np.zeros((20, 20, 20), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 2
    mask[15:18, 15:18, 15:18] = 99
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "proximal_femur",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"femur": 2},
            "geometry": {"pmma_thickness_voxels": 1, "cap_axis": "y"},
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "sideways_fall", "displacement": 1.0},
        preprocessing_config={
            "crop_to_bb": {"enabled": True, "margin_voxels": 1},
        },
    )

    assert built.element_sets["bone"] == 4 * 4 * 4
    active = np.argwhere(built.postprocess_mask)
    lo = active.min(axis=0)
    hi = active.max(axis=0)
    shape = np.asarray(built.postprocess_mask.shape)
    assert int(lo[0]) >= 3
    assert int(shape[0] - 1 - hi[0]) >= 3
    assert int(lo[2]) >= 3
    assert int(shape[2] - 1 - hi[2]) >= 3


def test_proximal_femur_model_pads_foreground_margin_before_caps(tmp_path: Path):
    density = np.zeros((12, 12, 12), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:10, 2:8, 4:12] = 700.0
    mask[2:10, 2:8, 4:12] = 2
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "proximal_femur",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"femur": 2},
            "geometry": {
                "cap_axis": "y",
                "cap": {
                    "thickness_voxels": 2,
                    "intrusion_depth_voxels": 3,
                },
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "sideways_fall", "displacement": 1.0},
        preprocessing_config={
            "crop_to_bb": {"enabled": True, "margin_voxels": 0},
        },
    )

    active = np.argwhere(built.postprocess_mask)
    lo = active.min(axis=0)
    hi = active.max(axis=0)
    shape = np.asarray(built.postprocess_mask.shape)

    # The builder should keep enough z/x margin for later fixture generation
    # even when the loaded foreground originally touched the crop boundary.
    assert int(lo[0]) >= 5
    assert int(shape[0] - 1 - hi[0]) >= 5
    assert int(lo[2]) >= 5
    assert int(shape[2] - 1 - hi[2]) >= 5


def test_femur_lesser_trochanter_cut_uses_distal_area_peak():
    data = np.zeros((60, 80, 90), dtype=bool)
    x = np.arange(data.shape[0])[:, None]
    y = np.arange(data.shape[1])[None, :]
    for z in range(10, 86):
        radius = 10
        y_center = 35
        if 58 <= z <= 64:
            radius = 19
        if 73 <= z <= 79:
            y_center = 50
            radius = 12
        section = ((x - 30) ** 2 + (y - y_center) ** 2) <= radius**2
        data[:, :, z] = section

    meta = detect_lesser_trochanter_cut_z(
        data,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )

    assert meta["greater_trochanter_z"] == pytest.approx(76.0)
    assert meta["lesser_trochanter_z"] == pytest.approx(61.0)
    assert meta["cut_z"] == pytest.approx(61.0)


def test_femur_lesser_trochanter_cut_uses_percent_offset_in_z_only():
    data = np.zeros((80, 90, 100), dtype=bool)
    x = np.arange(data.shape[0])[:, None]
    y = np.arange(data.shape[1])[None, :]
    for z in range(10, 96):
        radius = 10
        y_center = 35
        if 54 <= z <= 62:
            radius = 20
        if 78 <= z <= 82:
            y_center = 58
            radius = 12
        section = ((x - 40) ** 2 + (y - y_center) ** 2) <= radius**2
        data[:, :, z] = section

    meta = detect_lesser_trochanter_cut_z(
        data,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        distal_offset_percent=50.0,
        max_distal_to_greater_mm=45.0,
    )

    assert meta["greater_trochanter_z"] == pytest.approx(81.0)
    assert meta["lesser_trochanter_z"] == pytest.approx(58.0)
    assert meta["distal_offset_mm"] == pytest.approx(11.5)
    assert meta["cut_z"] == pytest.approx(46.5)


def test_standardize_femur_shaft_length_zeroes_voxels_below_cut():
    density = np.ones((5, 3, 3), dtype=np.float32)
    mask = np.ones_like(density, dtype=bool)

    cropped_density, cropped_mask, meta = standardize_femur_shaft_length(
        density_xyz=np.transpose(density, (2, 1, 0)),
        mask_xyz=np.transpose(mask, (2, 1, 0)),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        cut_mode="fixed_length",
        retained_length_mm=2.0,
    )

    out = np.transpose(cropped_mask, (2, 1, 0))
    assert meta["cut_z"] == pytest.approx(2.0)
    assert not np.any(out[:2])
    assert np.all(out[2:])
    assert np.all(np.transpose(cropped_density, (2, 1, 0))[2:] == 1.0)


def test_standardize_femur_shaft_length_supports_proportional_length_mode():
    density_xyz = np.ones((3, 4, 10), dtype=np.float32)
    mask_xyz = np.ones_like(density_xyz, dtype=bool)

    cropped_density, cropped_mask, meta = standardize_femur_shaft_length(
        density_xyz=density_xyz,
        mask_xyz=mask_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        cut_mode="proportional_length",
        cut_axis="z",
        cut_side="low",
        reference_extent_axis="y",
        retain_multiplier=1.0,
    )

    assert meta["cut_mode"] == "proportional_length"
    assert meta["cut_axis"] == "z"
    assert meta["cut_side"] == "low"
    assert meta["reference_extent_axis"] == "y"
    assert meta["reference_extent_mm"] == pytest.approx(4.0)
    assert meta["cut_coordinate_mm"] == pytest.approx(5.0)
    assert meta["cut_z"] == pytest.approx(5.0)
    out = np.transpose(cropped_mask, (2, 1, 0))
    assert not np.any(out[:5])
    assert np.all(out[5:])
    assert np.all(np.transpose(cropped_density, (2, 1, 0))[5:] == 1.0)


def test_proximal_femur_model_standardizes_distal_shaft_with_proportional_length(
    tmp_path: Path,
):
    density_xyz = np.zeros((60, 80, 90), dtype=np.float32)
    femur_xyz = np.zeros_like(density_xyz, dtype=bool)
    x = np.arange(density_xyz.shape[0])[:, None]
    y = np.arange(density_xyz.shape[1])[None, :]
    for z in range(10, 86):
        radius = 10
        y_center = 35
        if 58 <= z <= 64:
            radius = 19
        if 73 <= z <= 79:
            y_center = 50
            radius = 12
        section = ((x - 30) ** 2 + (y - y_center) ** 2) <= radius**2
        femur_xyz[:, :, z] = section
        density_xyz[:, :, z][section] = 700.0

    density = np.transpose(density_xyz, (2, 1, 0))
    mask = np.transpose(femur_xyz.astype(np.uint8) * 2, (2, 1, 0))
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "proximal_femur_sideways_fall",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"femur": 2},
            "geometry": {
                "cap_axis": "y",
                "pmma_thickness_voxels": 2,
                "shaft_standardization": {
                    "enabled": True,
                    "cut_mode": "proportional_length",
                    "cut_axis": "z",
                    "cut_side": "low",
                    "reference_extent_axis": "y",
                    "retain_multiplier": 1.35,
                },
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "sideways_fall", "displacement": 1.0},
    )

    z = np.argwhere(built.postprocess_mask)[:, 0]
    assert int(z.min()) > 10
    assert len(built.node_sets["distal_femur"]) > 0
    assert built.element_sets["distal_femur"] > 0
    shaft = built.metadata["model"]["shaft_standardization"]
    assert shaft["cut_mode"] == "proportional_length"
    assert shaft["cut_axis"] == "z"
    assert shaft["cut_side"] == "low"
    assert shaft["reference_extent_axis"] == "y"
    assert shaft["retain_multiplier"] == pytest.approx(1.35)
    assert shaft["reference_extent_mm"] > 0.0
    assert shaft["cut_z"] == pytest.approx(shaft["cut_coordinate_mm"])


def test_spine_model_pads_foreground_margin_before_disk_projection(tmp_path: Path):
    density = np.zeros((12, 12, 12), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[0:8, 2:10, 4:12] = 800.0
    mask[0:8, 2:10, 4:12] = 20
    mask[2:6, 3:5, 10:12] = 48
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 20, "process": 48},
            "geometry": {
                "axis": "z",
                "disk": {
                    "target_label": 20,
                    "shape": "anatomy",
                    "thickness_voxels": 2,
                    "intrusion_depth_voxels": 3,
                },
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
        preprocessing_config={
            "crop_to_bb": {"enabled": True, "margin_voxels": 0},
        },
    )

    active = np.argwhere(built.postprocess_mask)
    lo = active.min(axis=0)
    hi = active.max(axis=0)
    shape = np.asarray(built.postprocess_mask.shape)

    assert int(lo[0]) >= 5
    assert int(shape[0] - 1 - hi[0]) >= 5
    assert int(lo[2]) >= 5
    assert int(shape[2] - 1 - hi[2]) >= 5


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


def test_lightweight_icp_estimates_translation_and_reads_vtk_points(tmp_path: Path):
    fixed = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    moving = fixed + np.asarray([2.0, -1.0, 0.5])
    transform = estimate_rigid_icp(
        moving_points=moving,
        fixed_points=fixed,
        iterations=10,
        tolerance=1e-8,
    )

    np.testing.assert_allclose(transform["translation"], [-2.0, 1.0, -0.5], atol=1e-6)

    vtk_path = tmp_path / "points.vtk"
    vtk_path.write_text(
        "\n".join(
            [
                "# vtk DataFile Version 3.0",
                "points",
                "ASCII",
                "DATASET POLYDATA",
                "POINTS 3 float",
                "0 0 0",
                "1 2 3",
                "4 5 6",
            ]
        ),
        encoding="utf-8",
    )
    np.testing.assert_allclose(
        read_reference_points(vtk_path),
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )


def test_reference_points_reader_supports_binary_vtk(tmp_path: Path):
    vtk_path = tmp_path / "points_binary.vtk"
    points = np.asarray(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=">f4"
    )
    vtk_path.write_bytes(
        b"# vtk DataFile Version 5.1\n"
        b"points\n"
        b"BINARY\n"
        b"DATASET POLYDATA\n"
        b"POINTS 3 float\n"
        + points.tobytes()
    )

    np.testing.assert_allclose(read_reference_points(vtk_path), points.astype(float))


def test_reference_points_orientation_reorders_and_flips_axes():
    stored_zyx = np.asarray(
        [
            [0.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [4.0, 30.0, 300.0],
        ]
    )

    oriented = orient_reference_points(
        stored_zyx,
        axis_order="zyx",
        flips="x",
    )

    np.testing.assert_allclose(
        oriented,
        [
            [300.0, 10.0, 0.0],
            [200.0, 20.0, 2.0],
            [100.0, 30.0, 4.0],
        ],
    )


def test_icp_supports_vtk_like_centroid_initialization():
    fixed = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
        ]
    )
    moving = fixed + np.asarray([3.0, -2.0, 1.0])

    transform = estimate_rigid_icp(
        moving_points=moving,
        fixed_points=fixed,
        iterations=10,
        tolerance=1.0e-8,
        start_by_matching_centroids_only=True,
        convergence="absolute",
        distance_mode="rms",
    )

    np.testing.assert_allclose(transform["translation"], [-3.0, 2.0, -1.0], atol=1e-6)


def test_surface_point_sampling_supports_stride_mode():
    mask = np.ones((4, 4, 4), dtype=bool)

    linspace = surface_points_from_mask(
        mask,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        max_points=5,
        sample_mode="linspace",
    )
    stride = surface_points_from_mask(
        mask,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        max_points=5,
        sample_mode="stride",
        sample_offset=1,
    )

    assert linspace.shape == (5, 3)
    assert stride.shape == (5, 3)
    assert not np.allclose(linspace, stride)


def test_spine_registration_can_scale_reference_by_pca_axis_lengths(tmp_path: Path):
    density = np.zeros((10, 8, 6), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:8, 2:6, 2:4] = 800.0
    mask[2:8, 2:6, 2:4] = 2
    mask[4:7, 3:5, 4:6] = 1
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    reference_mask = np.zeros((6, 6, 6), dtype=bool)
    reference_mask[1:5, 1:5, 2:4] = True
    reference_points = surface_points_from_mask(
        reference_mask,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 2, "process": 1},
            "registration": {
                "enabled": True,
                "method": "vtk_icp",
                "reference_points": "reference_points.npz",
                "initialization": "centroid",
                "convergence": "delta",
                "distance_mode": "mean",
                "max_points": 2000,
                "iterations": 5,
                "source_landmark_mode": "stride",
                "source_landmark_offset": 0,
                "reference_scaling": {
                    "enabled": True,
                    "min_factors": [0.5, 0.5, 0.5],
                    "max_factors": [2.0, 2.0, 2.0],
                },
            },
            "geometry": {"pmma_thickness_mm": 2, "axis": "z"},
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "target_displacement_percent": -0.68},
    )

    scaling = built.metadata["model"]["registration"]["reference_scaling"]
    assert scaling["enabled"] is True
    assert scaling["source"] == "pca_axis_lengths"
    assert len(scaling["scale_factors"]) == 3
    assert all(0.5 <= value <= 2.0 for value in scaling["scale_factors"])


def test_model_percent_displacement_uses_padded_full_height():
    displacement = displacement_from_load_case(
        {"target_displacement_percent": -0.68},
        axis="z",
        dimensions_xyz=(10, 20, 42),
        spacing=(1.0, 1.0, 1.0),
        default=-0.01,
    )

    assert displacement == pytest.approx(-0.2856)


def test_workflow_replay_uses_body_for_registration_but_full_mask_for_model():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 20
    mask[2:4, 0:1, 1:3] = 48
    model_config = {"labels": {"body": 20, "process": 48}}
    replay_cfg = {"registration_target": "vertebral_body"}

    registration_mask = _workflow_active_mask(mask, model_config, replay_cfg)
    model_mask = _workflow_model_mask(mask, model_config)

    assert int(np.count_nonzero(registration_mask)) == int(np.count_nonzero(mask == 20))
    assert int(np.count_nonzero(model_mask)) == int(np.count_nonzero(mask > 0))
    assert np.count_nonzero(model_mask) > np.count_nonzero(registration_mask)


def test_proximal_femur_model_can_use_reference_registration(tmp_path: Path):
    density = np.zeros((8, 9, 10), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:7] = 700.0
    mask[2:6, 2:6, 2:7] = 2
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)
    reference_points = surface_points_from_mask(
        mask == 2,
        spacing=(1.0, 1.0, 1.0),
        origin=(5.0, -2.0, 1.0),
    )
    np.savez(tmp_path / "femur_reference.npz", points=reference_points)

    built = build_model(
        {
            "type": "proximal_femur_sideways_fall",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"femur": 2},
            "registration": {
                "enabled": True,
                "reference_points": "femur_reference.npz",
                "max_points": 2000,
                "iterations": 5,
            },
            "geometry": {"pmma_thickness_voxels": 1, "cap_axis": "y"},
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "sideways_fall", "displacement": 1.0},
    )

    assert built.metadata["model"]["registration"]["enabled"] is True
    assert built.metadata["model"]["registration"]["method"] == "lightweight_icp"
    assert built.metadata["model"]["load_axis"] == "y"
    nonzero = np.abs(built.boundary_conditions.fixed_values) > 1.0e-12
    assert np.all(built.boundary_conditions.fixed_values[nonzero] > 0.0)
    assert built.element_sets["femoral_head_cap"] > 0
    assert built.element_sets["greater_trochanter_cap"] > 0
    active = np.argwhere(built.postprocess_mask)
    assert int(active[:, 0].min()) <= 3
    assert int(built.postprocess_mask.shape[0] - active[:, 0].max()) <= 4


def test_mask_alignment_sizes_output_grid_from_full_surface_not_sampled_subset(
    tmp_path: Path,
):
    density = np.zeros((18, 18, 18), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    mask[2:16, 2:12, 2:12] = 1
    mask[16:18, 8:10, 8:10] = 1
    density[mask > 0] = 100.0
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    full_surface = surface_points_from_mask(
        mask > 0,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        max_points=None,
    )
    sampled_surface = surface_points_from_mask(
        mask > 0,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        max_points=40,
        sample_mode="stride",
        sample_offset=0,
    )
    assert float(full_surface[:, 2].max()) > float(sampled_surface[:, 2].max())
    np.savez(tmp_path / "reference_points.npz", points=full_surface)

    aligned = align_mask_to_reference(
        density_zyx=density,
        mask_zyx=mask > 0,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        registration_config={
            "enabled": True,
            "reference_points": "reference_points.npz",
            "method": "vtk_icp",
            "initialization": "centroid",
            "convergence": "delta",
            "distance_mode": "mean",
            "max_points": 40,
            "iterations": 5,
            "source_landmark_mode": "stride",
            "source_landmark_offset": 0,
            "margin_voxels": 4,
            "crop_to_bbox": False,
        },
        base_dir=tmp_path,
    )

    coords = np.argwhere(aligned.mask_zyx)
    high_margin = np.asarray(aligned.mask_zyx.shape) - 1 - coords.max(axis=0)
    assert int(high_margin[0]) >= 4


def test_workflow_replay_model_uses_saved_disk_and_nodeset_labels(tmp_path: Path):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 2

    disk_labels = np.zeros_like(mask, dtype=np.uint16)
    disk_labels[2:6, 1:2, 2:6] = 201
    disk_labels[2:6, 6:7, 2:6] = 202

    nodeset_labels = np.zeros_like(mask, dtype=np.uint16)
    nodeset_labels[2:6, 1:2, 2:6] = 101
    nodeset_labels[2:6, 6:7, 2:6] = 202
    nodeset_labels[2:6, 2:6, 2:3] = 103

    density_img = sitk.GetImageFromArray(density)
    mask_img = sitk.GetImageFromArray(mask)
    disk_img = sitk.GetImageFromArray(disk_labels)
    nodeset_img = sitk.GetImageFromArray(nodeset_labels)
    for image in (density_img, mask_img, disk_img, nodeset_img):
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))

    sitk.WriteImage(density_img, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_img, str(tmp_path / "mask.nii.gz"))
    sitk.WriteImage(disk_img, str(tmp_path / "disk_labels.nii.gz"))
    sitk.WriteImage(nodeset_img, str(tmp_path / "nodesets.nii.gz"))

    reference_points = surface_points_from_mask(
        mask == 2,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    built = build_model(
        {
            "type": "proximal_femur_sideways_fall",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
            "registration": {
                "enabled": True,
                "reference_points": "reference_points.npz",
                "max_points": 2000,
                "iterations": 5,
                "initialization": "centroid",
            },
            "workflow_replay": {
                "enabled": True,
                "disk_labels": "disk_labels.nii.gz",
                "nodesets": "nodesets.nii.gz",
                "reference_points": "reference_points.npz",
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={
            "type": "nodeset",
            "fixed": [
                {"nodeset": "impact_disk", "dofs": ["x", "y", "z"], "value": 0.0},
                {"nodeset": "distal_shaft_fixation", "dofs": ["x", "y", "z"], "value": 0.0},
            ],
            "prescribed": [
                {"nodeset": "support_disk", "dof": "y", "value": "4.0%", "units": "%"}
            ],
        },
        nodeset_config={
            "impact_disk": {
                "type": "label_image",
                "label": 101,
                "selection": "surface_nodes",
            },
            "support_disk": {
                "type": "label_image",
                "label": 202,
                "selection": "surface_nodes",
            },
            "distal_shaft_fixation": {
                "type": "label_image",
                "label": 103,
                "selection": "interface_nodes",
            },
        },
    )

    material_xyz = np.transpose(built.material, (2, 1, 0))
    assert np.count_nonzero(material_xyz == 2500.0) > 0
    assert built.metadata["model"]["workflow_replay"]["enabled"] is True
    assert built.metadata["model"]["registration"]["enabled"] is True
    assert set(built.node_sets) == {"impact_disk", "support_disk", "distal_shaft_fixation"}
    assert all(len(nodes) > 0 for nodes in built.node_sets.values())
    assert np.any(np.abs(built.boundary_conditions.fixed_values) > 0.0)
    prescribed_y = built.boundary_conditions.fixed_values[
        (built.boundary_conditions.fixed_coordinates[:, 3] == 1)
        & (~np.isclose(built.boundary_conditions.fixed_values, 0.0))
    ]
    assert np.unique(prescribed_y).tolist() == pytest.approx([0.20])


def test_workflow_replay_pads_sample_extent_before_resampling_saved_disks(
    tmp_path: Path,
):
    density = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.full_like(density, 2, dtype=np.uint8)

    disk_labels = np.zeros((6, 6, 6), dtype=np.uint16)
    disk_labels[1:5, 0:1, 1:5] = 201
    disk_labels[1:5, 5:6, 1:5] = 202

    nodeset_labels = np.zeros((6, 6, 6), dtype=np.uint16)
    nodeset_labels[1:5, 0:1, 1:5] = 101
    nodeset_labels[1:5, 5:6, 1:5] = 202
    nodeset_labels[1:5, 1:2, 1:5] = 103

    density_img = sitk.GetImageFromArray(density)
    mask_img = sitk.GetImageFromArray(mask)
    density_img.SetSpacing((1.0, 1.0, 1.0))
    mask_img.SetSpacing((1.0, 1.0, 1.0))
    density_img.SetOrigin((0.0, 0.0, 0.0))
    mask_img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(density_img, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_img, str(tmp_path / "mask.nii.gz"))

    disk_img = sitk.GetImageFromArray(disk_labels)
    node_img = sitk.GetImageFromArray(nodeset_labels)
    disk_img.SetSpacing((1.0, 1.0, 1.0))
    node_img.SetSpacing((1.0, 1.0, 1.0))
    disk_img.SetOrigin((-1.0, -1.0, -1.0))
    node_img.SetOrigin((-1.0, -1.0, -1.0))
    sitk.WriteImage(disk_img, str(tmp_path / "disk_labels.nii.gz"))
    sitk.WriteImage(node_img, str(tmp_path / "nodesets.nii.gz"))

    reference_points = surface_points_from_mask(
        mask == 2,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
            "geometry": {
                "cap_axis": "y",
                "cap": {
                    "thickness_voxels": 1,
                    "intrusion_depth_voxels": 1,
                },
            },
            "workflow_replay": {
                "enabled": True,
                "disk_labels": "disk_labels.nii.gz",
                "nodesets": "nodesets.nii.gz",
                "reference_points": "reference_points.npz",
            },
            "registration": {
                "enabled": True,
                "reference_points": "reference_points.npz",
                "initialization": "centroid",
                "max_points": 2000,
                "iterations": 5,
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={
            "type": "nodeset",
            "fixed": [
                {"nodeset": "impact_disk", "dofs": ["x", "y", "z"], "value": 0.0},
                {"nodeset": "shaft_fixation", "dofs": ["x", "y", "z"], "value": 0.0},
            ],
            "prescribed": [
                {"nodeset": "support_disk", "dof": "y", "value": 1.0},
            ],
        },
        nodeset_config={
            "impact_disk": {"type": "label_image", "label": 101, "selection": "surface_nodes"},
            "support_disk": {"type": "label_image", "label": 202, "selection": "surface_nodes"},
            "shaft_fixation": {"type": "label_image", "label": 103, "selection": "interface_nodes"},
        },
        preprocessing_config={
            "crop_to_bb": {"enabled": True, "margin_voxels": 0},
        },
    )

    material_xyz = np.transpose(built.material, (2, 1, 0))
    assert int(np.count_nonzero(material_xyz == 2500.0)) == 32


def test_workflow_replay_prefers_plane_driven_geometry_over_saved_labels(
    tmp_path: Path,
):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 2

    wrong_disk_labels = np.zeros_like(mask, dtype=np.uint16)
    wrong_nodeset_labels = np.zeros_like(mask, dtype=np.uint16)
    wrong_nodeset_labels[0:1, :, :] = 101

    density_img = sitk.GetImageFromArray(density)
    mask_img = sitk.GetImageFromArray(mask)
    disk_img = sitk.GetImageFromArray(wrong_disk_labels)
    nodes_img = sitk.GetImageFromArray(wrong_nodeset_labels)
    for image in (density_img, mask_img, disk_img, nodes_img):
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(density_img, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_img, str(tmp_path / "mask.nii.gz"))
    sitk.WriteImage(disk_img, str(tmp_path / "disk_labels.nii.gz"))
    sitk.WriteImage(nodes_img, str(tmp_path / "nodesets.nii.gz"))

    reference_points = surface_points_from_mask(
        mask == 2,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    built = build_model(
        {
            "type": "proximal_femur_sideways_fall",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
            "registration": {
                "enabled": True,
                "reference_points": "reference_points.npz",
                "max_points": 2000,
                "iterations": 5,
                "initialization": "centroid",
            },
            "workflow_replay": {
                "enabled": True,
                "disk_labels": "disk_labels.nii.gz",
                "nodesets": "nodesets.nii.gz",
                "reference_points": "reference_points.npz",
            },
            "slicer_editor": {
                "planes": [
                    {
                        "name": "Support disk",
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "anatomy",
                        "thickness_mm": 2.0,
                        "protrusion_depth_mm": 1.0,
                        "use_plane_size": True,
                        "center_ras": [3.5, 3.5, 7.0],
                        "normal_ras": [0.0, 0.0, -1.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 1.0, 0.0],
                        "size_mm": [4.0, 4.0],
                    }
                ]
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={
            "type": "nodeset",
            "prescribed": [{"nodeset": "support_disk", "dof": "z", "value": -1.0}],
        },
        nodeset_config={
            "support_disk": {
                "type": "label_image",
                "label": 202,
                "selection": "surface_nodes",
            }
        },
    )

    assert "support_disk" in built.node_sets
    assert len(built.node_sets["support_disk"]) > 0
    material_xyz = np.transpose(built.material, (2, 1, 0))
    assert int(np.count_nonzero(material_xyz == 2500.0)) > 0


def test_workflow_replay_reference_model_space_keeps_reference_plane_axial(
    tmp_path: Path,
):
    density = np.zeros((10, 10, 10), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[3:7, 3:7, 3:7] = 700.0
    mask[3:7, 3:7, 3:7] = 20

    density_img = sitk.GetImageFromArray(density)
    mask_img = sitk.GetImageFromArray(mask)
    for image in (density_img, mask_img):
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(density_img, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_img, str(tmp_path / "mask.nii.gz"))

    reference_points = surface_points_from_mask(
        mask == 20,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"body": 20},
            "workflow_replay": {
                "enabled": True,
                "model_space": "reference",
                "reference_points": "reference_points.npz",
            },
            "registration": {
                "enabled": True,
                "reference_points": "reference_points.npz",
                "initialization": "centroid",
                "max_points": 2000,
                "iterations": 5,
            },
            "slicer_editor": {
                "planes": [
                    {
                        "name": "Superior disk",
                        "reference_space": True,
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "anatomy",
                        "thickness_mm": 2.0,
                        "protrusion_depth_mm": 1.0,
                        "use_plane_size": True,
                        "center_ras": [4.5, 4.5, 8.0],
                        "normal_ras": [0.0, 0.0, -1.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 1.0, 0.0],
                        "size_mm": [4.0, 4.0],
                    }
                ]
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={
            "type": "nodeset",
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": -1.0}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    assert built.metadata["model"]["workflow_replay"]["model_space"] == "reference"
    assert built.metadata["model"]["workflow_replay"]["geometry_mode"] == "plane_driven"
    assert built.metadata["model"]["registration"]["applied_to_model_grid"] is True
    z_values = [node[2] for node in built.node_sets["superior_disk"]]
    assert max(z_values) - min(z_values) <= 2


def test_workflow_replay_without_registration_stays_in_sample_space(tmp_path: Path):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 20

    density_img = sitk.GetImageFromArray(density)
    mask_img = sitk.GetImageFromArray(mask)
    for image in (density_img, mask_img):
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(density_img, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_img, str(tmp_path / "mask.nii.gz"))

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"body": 20},
            "workflow_replay": {"enabled": True},
            "registration": {"enabled": False},
            "slicer_editor": {
                "planes": [
                    {
                        "name": "Superior disk",
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "anatomy",
                        "thickness_mm": 2.0,
                        "protrusion_depth_mm": 1.0,
                        "use_plane_size": True,
                        "center_ras": [3.5, 3.5, 7.0],
                        "normal_ras": [0.0, 0.0, -1.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 1.0, 0.0],
                        "size_mm": [4.0, 4.0],
                    }
                ]
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={
            "type": "nodeset",
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": -1.0}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    assert built.metadata["model"]["registration"]["enabled"] is False
    assert built.metadata["model"]["workflow_replay"]["model_space"] == "sample"
    assert "superior_disk" in built.node_sets


def test_reference_space_editor_scales_bc_planes_with_reference_size():
    editor = {
        "planes": [
            {
                "name": "Superior disk",
                "center_ras": [10.0, 20.0, 30.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [12.0, 14.0],
            }
        ]
    }

    scaled = _scale_reference_space_editor(
        editor,
        scaling_meta={
            "enabled": True,
            "reference_center": [0.0, 0.0, 0.0],
            "scale_factors": [2.0, 3.0, 4.0],
        },
    )

    plane = scaled["planes"][0]
    assert plane["center_ras"] == pytest.approx([20.0, 60.0, 120.0])
    assert plane["size_mm"] == pytest.approx([24.0, 42.0])
    assert plane["normal_ras"] == [0.0, 0.0, -1.0]


def test_bbox_relative_editor_resolves_planes_from_model_bounds():
    mask = np.zeros((8, 12, 16), dtype=bool)
    mask[2:6, 3:9, 4:14] = True
    editor = {
        "planes": [
            {
                "name": "Superior disk",
                "relative_to": "model_bbox",
                "center_fraction": [0.5, 0.5, 1.25],
                "size_fraction": [1.5, 2.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
            }
        ]
    }

    resolved = _resolve_bbox_relative_editor(
        editor,
        model_mask_zyx=mask,
        spacing=(0.5, 2.0, 3.0),
        origin=(10.0, 20.0, 30.0),
    )

    plane = resolved["planes"][0]
    assert plane["center_ras"] == pytest.approx([14.25, 31.0, 47.25])
    assert plane["size_mm"] == pytest.approx([6.75, 20.0])
    assert plane["relative_to"] == "resolved_model_bbox"
    assert plane["relative_definition"]["relative_to"] == "model_bbox"


def test_workflow_replay_uses_bbox_relative_plane_for_scaled_model(tmp_path: Path):
    density = np.zeros((12, 14, 16), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[3:9, 4:10, 5:13] = 700.0
    mask[3:9, 4:10, 5:13] = 20

    density_img = sitk.GetImageFromArray(density)
    mask_img = sitk.GetImageFromArray(mask)
    for image in (density_img, mask_img):
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(density_img, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_img, str(tmp_path / "mask.nii.gz"))

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"body": 20},
            "workflow_replay": {"enabled": True},
            "registration": {"enabled": False},
            "slicer_editor": {
                "planes": [
                    {
                        "name": "Superior disk",
                        "relative_to": "model_bbox",
                        "center_fraction": [0.5, 0.5, 1.25],
                        "size_fraction": [1.5, 1.5],
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "anatomy",
                        "thickness_mm": 2.0,
                        "protrusion_depth_mm": 1.0,
                        "use_plane_size": True,
                        "normal_ras": [0.0, 0.0, -1.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 1.0, 0.0],
                    }
                ]
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={
            "type": "nodeset",
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": -1.0}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    assert "superior_disk" in built.node_sets
    z_values = [node[2] for node in built.node_sets["superior_disk"]]
    assert max(z_values) - min(z_values) <= 2
    resolved_plane = built.metadata["model"]["workflow_replay"]["resolved_planes"][0]
    assert resolved_plane["relative_to"] == "resolved_model_bbox"
    assert resolved_plane["relative_definition"]["center_fraction"] == [0.5, 0.5, 1.25]


def test_workflow_replay_pads_when_relative_disk_extends_outside_image(
    tmp_path: Path,
):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[4:8, 2:6, 2:6] = 700.0
    mask[4:8, 2:6, 2:6] = 20

    density_img = sitk.GetImageFromArray(density)
    mask_img = sitk.GetImageFromArray(mask)
    for image in (density_img, mask_img):
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(density_img, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_img, str(tmp_path / "mask.nii.gz"))

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"body": 20},
            "workflow_replay": {"enabled": True},
            "registration": {"enabled": False},
            "slicer_editor": {
                "planes": [
                    {
                        "name": "Superior disk",
                        "relative_to": "model_bbox",
                        "center_fraction": [0.5, 0.5, 1.5],
                        "size_fraction": [1.5, 1.5],
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "anatomy",
                        "thickness_mm": 2.0,
                        "protrusion_depth_mm": 0.0,
                        "use_plane_size": True,
                        "normal_ras": [0.0, 0.0, -1.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 1.0, 0.0],
                    }
                ]
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={
            "type": "nodeset",
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": -1.0}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    assert built.material.shape[0] > 8
    assert int(built.element_sets["workflow_disks"]) > 0
    assert len(built.node_sets["superior_disk"]) > 0


def test_spine_model_can_use_lightweight_icp_registration(tmp_path: Path):
    density = np.zeros((8, 7, 6), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:5, 2:4] = 800.0
    mask[2:6, 2:5, 2:4] = 2
    mask[3:5, 3:4, 3:5] = 1
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)
    reference_points = surface_points_from_mask(
        mask == 2,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 2, "process": 1},
            "registration": {
                "enabled": True,
                "reference_points": "reference_points.npz",
                "max_points": 2000,
                "iterations": 5,
            },
            "geometry": {"pmma_thickness_mm": 2, "axis": "z"},
            "outputs": {"manifest": "model/model.json"},
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    assert built.metadata["model"]["registration"]["enabled"] is True
    assert built.element_sets["inferior_disk"] > 0
    manifest = json.loads(built.exported["manifest"].read_text(encoding="utf-8"))
    assert manifest["model"]["registration"]["method"] == "lightweight_icp"
