from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from parosol_py.core import BoundaryConditionSet
from parosol_py.config import _image_metadata, _read_image_array_zyx
from parosol_py.modeling import build_model
from parosol_py.modeling.common import (
    _shift_boundary_conditions_for_crop,
    build_preprocessed_inputs_preview,
    load_density_and_mask,
    material_from_density,
    occupied_length_mm,
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
from parosol_py.modeling.workflow_replay import (
    _crop_workflow_model_to_material_bbox,
    _editor_disk_labels,
    _invert_rigid_transform,
    _reference_model_space_icp_direction,
    _resolve_bbox_relative_editor,
    _scale_reference_space_editor,
    _scale_reference_points_preserving_pose,
    _workflow_active_mask,
    _workflow_model_mask,
    build_workflow_replay_preview,
    build_workflow_replay_model,
)
from parosol_py.nodesets import nodes_from_labeled_voxels



def test_legacy_anatomy_builder_modules_are_removed():
    project_root = Path(__file__).resolve().parents[1]
    modeling_dir = project_root / "src" / "parosol_py" / "modeling"

    assert not (modeling_dir / "spine.py").exists()
    assert not (modeling_dir / "femur.py").exists()


def test_build_model_requires_workflow_replay_for_modeling(tmp_path: Path):
    with pytest.raises(NotImplementedError, match="model\\.workflow_replay\\.enabled"):
        build_model(
            {
                "type": "spine_compression",
                "density_image": "density.npy",
                "mask_image": "mask.npy",
            },
            base_dir=tmp_path,
            material_config={},
        )

def test_model_image_reader_canonicalizes_nifti_direction_to_slicer_ras(tmp_path: Path):
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 2.0, 3.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    path = tmp_path / "flipped_y.nii.gz"
    sitk.WriteImage(image, str(path))

    data, spacing, origin = read_image_zyx(path)

    expected_image = sitk.DICOMOrient(sitk.ReadImage(str(path)), "RAS")
    expected = sitk.GetArrayFromImage(expected_image)
    np.testing.assert_array_equal(data, expected)
    assert spacing == pytest.approx((1.0, 2.0, 3.0))
    assert origin == pytest.approx((-13.0, -20.0, 30.0))

    direct_data = _read_image_array_zyx(path)
    direct_spacing, direct_origin = _image_metadata(path)
    np.testing.assert_array_equal(direct_data, expected)
    assert direct_spacing == pytest.approx(expected_image.GetSpacing())
    assert direct_origin == pytest.approx(origin)


def _canonical_surface_points_from_mask_image(path: Path, label: int):
    labels_zyx, spacing, origin = read_image_zyx(path)
    return surface_points_from_mask(
        labels_zyx == label,
        spacing=spacing,
        origin=origin,
    )


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


def test_model_preprocessing_resample_isotropic_targets_requested_spacing(
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
        },
        base_dir=tmp_path,
        preprocessing_config={
            "resample_isotropic": {
                "enabled": True,
                "mode": "auto",
                "target_spacing_mm": 1.0,
            }
        },
    )

    assert spacing == pytest.approx((1.0, 1.0, 1.0))
    assert resampled_density.shape != density.shape
    assert resampled_mask.shape == resampled_density.shape


def test_model_preprocessing_resample_isotropic_uses_requested_density_interpolation(
    tmp_path: Path,
):
    z, y, x = np.indices((5, 6, 7), dtype=np.float32)
    density = (x**2 + 3.0 * y + 0.5 * z).astype(np.float32)
    mask = np.ones_like(density, dtype=np.uint8)
    density_image = sitk.GetImageFromArray(density)
    mask_image = sitk.GetImageFromArray(mask)
    density_image.SetSpacing((0.8, 0.8, 0.8))
    mask_image.SetSpacing((0.8, 0.8, 0.8))
    sitk.WriteImage(density_image, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_image, str(tmp_path / "mask.nii.gz"))

    model_config = {
        "density_image": "density.nii.gz",
        "mask_image": "mask.nii.gz",
    }
    base_resample = {
        "enabled": True,
        "mode": "fixed",
        "target_spacing_mm": 1.0,
    }

    linear_density, linear_mask, _spacing, _origin = load_density_and_mask(
        model_config,
        base_dir=tmp_path,
        preprocessing_config={"resample_isotropic": base_resample},
    )
    bspline_density, bspline_mask, _spacing, _origin = load_density_and_mask(
        model_config,
        base_dir=tmp_path,
        preprocessing_config={
            "resample_isotropic": {
                **base_resample,
                "density_interpolation": "bspline",
            }
        },
    )

    assert bspline_density.shape == linear_density.shape
    assert np.array_equal(bspline_mask, linear_mask)
    assert not np.allclose(bspline_density, linear_density)


def test_model_preprocessing_crop_to_bb_margin_mm_uses_image_spacing(
    tmp_path: Path,
):
    density = np.zeros((10, 10, 10), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[4:6, 4:6, 4:6] = 700.0
    mask[4:6, 4:6, 4:6] = 2
    density_image = sitk.GetImageFromArray(density)
    mask_image = sitk.GetImageFromArray(mask)
    density_image.SetSpacing((0.5, 1.0, 2.0))
    mask_image.SetSpacing((0.5, 1.0, 2.0))
    sitk.WriteImage(density_image, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_image, str(tmp_path / "mask.nii.gz"))

    cropped_density, cropped_mask, _spacing, origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
            "geometry": {"isotropic_spacing": False},
        },
        base_dir=tmp_path,
        preprocessing_config={"crop_to_bb": {"enabled": True, "margin_mm": 2.0}},
    )

    assert cropped_density.shape == (4, 6, 10)
    assert cropped_mask.shape == (4, 6, 10)
    assert origin == pytest.approx((-4.5, -7.0, 6.0))


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


def test_model_preprocessing_normalizes_mask_bbox_aspect_ratio(tmp_path: Path):
    density = np.zeros((24, 20, 24), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:18, 4:14, 2:22] = 700.0
    mask[2:18, 4:14, 2:22] = 2
    density_image = sitk.GetImageFromArray(density)
    mask_image = sitk.GetImageFromArray(mask)
    density_image.SetSpacing((1.0, 1.0, 1.0))
    mask_image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(density_image, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_image, str(tmp_path / "mask.nii.gz"))

    cropped_density, cropped_mask, _spacing, origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
        },
        base_dir=tmp_path,
        preprocessing_config={
            "normalize_aspect_ratio": {
                "enabled": True,
                "ratio": [1.2, 1.0, None],
            }
        },
    )

    assert cropped_density.shape == (12, 10, 20)
    assert cropped_mask.shape == (12, 10, 20)
    assert origin == pytest.approx((-21.0, -13.0, 4.0))
    assert int(np.count_nonzero(cropped_mask == 2)) == 12 * 10 * 20

    alias_density, alias_mask, _spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
        },
        base_dir=tmp_path,
        preprocessing_config={"aspect-ratio": [1.2, 1.0, None]},
    )

    assert alias_density.shape == (12, 10, 20)
    assert alias_mask.shape == (12, 10, 20)


def test_model_preprocessing_accepts_bbox_ratio_recipe_order(tmp_path: Path):
    density = np.zeros((24, 20, 24), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:18, 4:14, 2:22] = 700.0
    mask[2:18, 4:14, 2:22] = 2
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    cropped_density, cropped_mask, _spacing, origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
        },
        base_dir=tmp_path,
        preprocessing_config={"bbox_ratio": [1.0, 1.2, None]},
    )

    assert cropped_density.shape == (12, 10, 20)
    assert cropped_mask.shape == (12, 10, 20)
    assert origin == pytest.approx((-21.0, -13.0, 4.0))


def test_model_preprocessing_single_label_mask_can_replace_declared_label(
    tmp_path: Path,
):
    density = np.zeros((12, 14, 16), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:8, 3:9, 4:12] = 700.0
    mask[2:8, 3:9, 4:12] = 1
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    with pytest.warns(RuntimeWarning, match="using single foreground label 1"):
        cropped_density, cropped_mask, _spacing, origin = load_density_and_mask(
            {
                "density_image": "density.nii.gz",
                "mask_image": "mask.nii.gz",
                "labels": {"femur": 2},
            },
            base_dir=tmp_path,
            preprocessing_config={"crop_to_bb": {"enabled": True, "margin_voxels": 0}},
        )

    assert cropped_density.shape == (6, 6, 8)
    assert cropped_mask.shape == (6, 6, 8)
    assert origin == pytest.approx((-11.0, -8.0, 2.0))


def test_model_preprocessing_bbox_ratio_can_crop_from_constrained_min_end(
    tmp_path: Path,
):
    density = np.zeros((24, 20, 24), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:18, 4:14, 2:22] = 700.0
    mask[2:18, 4:14, 2:22] = 2
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    cropped_density, cropped_mask, _spacing, origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
        },
        base_dir=tmp_path,
        preprocessing_config={
            "bbox_ratio": [1.0, 1.2, None],
            "bbox_crop_from": [None, "min", None],
        },
    )

    assert cropped_density.shape == (12, 10, 20)
    assert cropped_mask.shape == (12, 10, 20)
    assert origin == pytest.approx((-21.0, -13.0, 6.0))


def test_model_preprocessing_bbox_ratio_recomputes_reference_after_crop(
    tmp_path: Path,
):
    density = np.zeros((90, 60, 12), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[0:10, 5:55, 2:10] = 700.0
    mask[0:10, 5:55, 2:10] = 2
    density[10:80, 20:40, 2:10] = 700.0
    mask[10:80, 20:40, 2:10] = 2
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    _cropped_density, cropped_mask, _spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
        },
        base_dir=tmp_path,
        preprocessing_config={
            "bbox_ratio": [1.0, 1.3, None],
            "bbox_crop_from": [None, "min", None],
        },
    )

    coords = np.argwhere(cropped_mask == 2)
    final_size = coords.max(axis=0) - coords.min(axis=0) + 1

    assert final_size[1] == 20
    assert final_size[0] <= 26


def test_model_preprocessing_proximal_box_ratio_uses_proximal_transverse_width(
    tmp_path: Path,
):
    density = np.zeros((120, 120, 110), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[70:110, 45:73, 30:82] = 700.0
    mask[70:110, 45:73, 30:82] = 2
    density[10:70, 45:105, 46:66] = 700.0
    mask[10:70, 45:105, 46:66] = 2
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    _cropped_density, cropped_mask, _spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
        },
        base_dir=tmp_path,
        preprocessing_config={
            "proximal_box_ratio": {
                "enabled": True,
                "ratio": 1.2,
                "proximal_fraction": 0.4,
                "reference_width": "max_xy",
                "crop_from": "min",
            }
        },
    )

    coords = np.argwhere(cropped_mask == 2)
    final_size = coords.max(axis=0) - coords.min(axis=0) + 1

    assert final_size[0] == 62
    assert final_size[1] == 60
    assert final_size[2] == 52


def test_workflow_replay_bbox_crop_from_uses_slicer_ijk_z_direction(
    tmp_path: Path,
):
    density = np.zeros((10, 6, 6), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[0:8, 1:5, 1:5] = 700.0
    mask[0:8, 1:5, 1:5] = 1
    for name, array in (("density", density), ("mask", mask)):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(image, str(tmp_path / f"{name}.nii.gz"))

    _cropped_density, cropped_mask, _spacing, origin = load_density_and_mask(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 1},
            "workflow_replay": {"enabled": True},
        },
        base_dir=tmp_path,
        preprocessing_config={
            "bbox_ratio": [1.0, 1.0, None],
            "bbox_crop_from": [None, "max", None],
        },
    )

    assert cropped_mask.shape == (4, 4, 4)
    assert origin == pytest.approx((-4.0, 1.0, 4.0))


def test_workflow_replay_bbox_crop_from_keeps_npy_array_direction(
    tmp_path: Path,
):
    density = np.zeros((10, 6, 6), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[0:8, 1:5, 1:5] = 700.0
    mask[0:8, 1:5, 1:5] = 1
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    _cropped_density, cropped_mask, _spacing, origin = load_density_and_mask(
        {
            "type": "workflow_replay",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"femur": 1},
            "workflow_replay": {"enabled": True},
        },
        base_dir=tmp_path,
        preprocessing_config={
            "bbox_ratio": [1.0, 1.0, None],
            "bbox_crop_from": [None, "max", None],
        },
    )

    assert cropped_mask.shape == (4, 4, 4)
    assert origin == pytest.approx((1.0, 1.0, 0.0))


def test_model_preprocessing_bbox_ratio_warns_when_target_axis_is_too_short(
    tmp_path: Path,
):
    density = np.zeros((18, 20, 24), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:10, 4:14, 2:22] = 700.0
    mask[2:10, 4:14, 2:22] = 2
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    with pytest.warns(RuntimeWarning, match="cannot reach requested bbox_ratio"):
        cropped_density, cropped_mask, _spacing, origin = load_density_and_mask(
            {
                "density_image": "density.nii.gz",
                "mask_image": "mask.nii.gz",
                "labels": {"femur": 2},
            },
            base_dir=tmp_path,
            preprocessing_config={
                "bbox_ratio": [1.0, 1.2, None],
                "bbox_crop_from": [None, "min", None],
            },
        )

    assert cropped_density.shape == (8, 10, 20)
    assert cropped_mask.shape == (8, 10, 20)
    assert origin == pytest.approx((-21.0, -13.0, 2.0))


def test_model_preprocessing_bbox_ratio_uses_shortest_one_axis_as_reference(
    tmp_path: Path,
):
    density = np.zeros((24, 20, 24), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:18, 4:14, 2:22] = 700.0
    mask[2:18, 4:14, 2:22] = 2
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    cropped_density, cropped_mask, _spacing, origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
        },
        base_dir=tmp_path,
        preprocessing_config={"bbox_ratio": [1.0, 1.0, 1.0]},
    )

    assert cropped_density.shape == (10, 10, 10)
    assert cropped_mask.shape == (10, 10, 10)
    assert origin == pytest.approx((-16.0, -13.0, 5.0))


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


def test_preprocessed_inputs_preview_uses_shared_model_preprocessing(tmp_path: Path):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    density[2:5, 2:5, 2:5] = 100.0
    density[7, 7, 7] = 200.0
    mask = np.zeros_like(density, dtype=np.uint8)
    mask[2:5, 2:5, 2:5] = 20
    mask[7, 7, 7] = 20

    density_img = sitk.GetImageFromArray(density)
    mask_img = sitk.GetImageFromArray(mask)
    for image in (density_img, mask_img):
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(density_img, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_img, str(tmp_path / "mask.nii.gz"))

    model_config = {
        "density_image": "density.nii.gz",
        "mask_image": "mask.nii.gz",
    }
    preprocessing_config = {
        "largest_cc": True,
        "crop_to_bb": {"enabled": True, "margin_voxels": 0},
    }
    preview = build_preprocessed_inputs_preview(
        model_config,
        base_dir=tmp_path,
        preprocessing_config=preprocessing_config,
    )
    expected_density, expected_mask, expected_spacing, expected_origin = load_density_and_mask(
        model_config,
        base_dir=tmp_path,
        preprocessing_config=preprocessing_config,
    )

    assert preview.density_zyx.shape == (3, 3, 3)
    assert preview.mask_zyx.shape == (3, 3, 3)
    np.testing.assert_allclose(preview.density_zyx, expected_density)
    np.testing.assert_array_equal(preview.mask_zyx, expected_mask)
    assert tuple(preview.spacing) == expected_spacing
    assert tuple(preview.origin) == expected_origin
    assert set(np.unique(preview.mask_zyx).astype(int)) == {20}
    assert preview.metadata["preprocessing"]["largest_cc"] is True


def test_model_custom_preprocessing_runs_after_standard_preprocessing(tmp_path: Path):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 10.0
    mask[2:6, 2:6, 2:6] = 1
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))
    (tmp_path / "custom_preprocessing.py").write_text(
        """
from parosol_py.images import ImageGrid


def custom_preprocessing(image, mask=None):
    assert image.array_xyz.shape == (4, 4, 4)
    cropped_image = ImageGrid(
        array_xyz=image.array_xyz[1:, :, :] + 5.0,
        spacing=image.spacing,
        origin=(image.origin[0] + image.spacing[0], image.origin[1], image.origin[2]),
    )
    cropped_mask = ImageGrid(
        array_xyz=mask.array_xyz[1:, :, :],
        spacing=mask.spacing,
        origin=cropped_image.origin,
    )
    return cropped_image, cropped_mask, {"step": "post_standard_crop"}
""",
        encoding="utf-8",
    )

    custom_density, custom_mask, spacing, origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
        },
        base_dir=tmp_path,
        preprocessing_config={"crop_to_bb": {"enabled": True, "margin_voxels": 0}},
        custom_preprocessing_config={"script": "custom_preprocessing.py"},
    )

    assert custom_density.shape == (4, 4, 3)
    assert custom_mask.shape == (4, 4, 3)
    assert spacing == pytest.approx((1.0, 1.0, 1.0))
    assert origin == pytest.approx((-4.0, -5.0, 2.0))
    assert float(custom_density[0, 0, 0]) == pytest.approx(15.0)


def test_model_custom_preprocessing_selects_named_option(tmp_path: Path):
    density = np.ones((4, 4, 4), dtype=np.float32)
    mask = np.ones_like(density, dtype=np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))
    (tmp_path / "first.py").write_text(
        """
def first_crop(image, mask=None):
    return image.array_xyz + 10.0, mask, {"selected": "first"}
""",
        encoding="utf-8",
    )
    (tmp_path / "second.py").write_text(
        """
def second_crop(image, mask=None):
    return image.array_xyz + 20.0, mask, {"selected": "second"}
""",
        encoding="utf-8",
    )

    custom_density, _custom_mask, _spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
        },
        base_dir=tmp_path,
        custom_preprocessing_config={
            "selected": "second",
            "options": [
                {"id": "first", "script": "first.py", "function": "first_crop"},
                {"id": "second", "script": "second.py", "function": "second_crop"},
            ],
        },
    )

    assert float(custom_density[0, 0, 0]) == pytest.approx(21.0)


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


def test_reference_points_reader_supports_npy_reference_cloud(tmp_path: Path):
    reference = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=float,
    )
    path = tmp_path / "slicer_reference_points.npy"
    np.save(path, reference)

    np.testing.assert_allclose(read_reference_points(path), reference)


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
    mask[0:1, 0:1, 0:1] = 2
    mask[1:3, 1:3, 1:3] = 20
    mask[2:4, 0:1, 1:3] = 48
    model_config = {
        "labels": {"body": 20, "process": 48},
        "targets": {"registration": "body", "model": ["body", "process"]},
    }
    replay_cfg = {}

    registration_mask = _workflow_active_mask(mask, model_config, replay_cfg)
    model_mask = _workflow_model_mask(mask, model_config)

    assert int(np.count_nonzero(registration_mask)) == int(np.count_nonzero(mask == 20))
    assert int(np.count_nonzero(model_mask)) == int(
        np.count_nonzero((mask == 20) | (mask == 48))
    )
    assert not np.any(model_mask[mask == 2])
    assert np.count_nonzero(model_mask) > np.count_nonzero(registration_mask)


def test_workflow_replay_uses_label_one_for_registration_when_labels_are_remapped():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask[2:4, 0:1, 1:3] = 2
    model_config = {
        "labels": {"body": 1, "process": 2},
        "targets": {"registration": "body", "model": ["body", "process"]},
    }
    replay_cfg = {}

    registration_mask = _workflow_active_mask(mask, model_config, replay_cfg)
    model_mask = _workflow_model_mask(mask, model_config)

    assert int(np.count_nonzero(registration_mask)) == int(np.count_nonzero(mask == 1))
    assert not np.any(registration_mask[mask == 2])
    assert int(np.count_nonzero(model_mask)) == int(np.count_nonzero(mask > 0))


def test_workflow_replay_single_label_mask_can_replace_declared_target_label():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    model_config = {
        "labels": {"femur": 2},
        "targets": {"registration": "femur", "model": ["femur"]},
    }
    replay_cfg = {}

    with pytest.warns(RuntimeWarning, match="using single foreground label 1"):
        registration_mask = _workflow_active_mask(mask, model_config, replay_cfg)
    with pytest.warns(RuntimeWarning, match="using single foreground label 1"):
        model_mask = _workflow_model_mask(mask, model_config)

    assert int(np.count_nonzero(registration_mask)) == int(np.count_nonzero(mask == 1))
    assert int(np.count_nonzero(model_mask)) == int(np.count_nonzero(mask == 1))


def test_workflow_replay_single_label_mask_can_replace_implicit_declared_label():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    model_config = {"labels": {"femur": 2}}
    replay_cfg = {}

    with pytest.warns(RuntimeWarning, match="using single foreground label 1"):
        registration_mask = _workflow_active_mask(mask, model_config, replay_cfg)
    with pytest.warns(RuntimeWarning, match="using single foreground label 1"):
        model_mask = _workflow_model_mask(mask, model_config)

    assert int(np.count_nonzero(registration_mask)) == int(np.count_nonzero(mask == 1))
    assert int(np.count_nonzero(model_mask)) == int(np.count_nonzero(mask == 1))


def test_workflow_replay_target_masks_are_selected_by_declared_label_keys():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 7
    mask[2:4, 0:1, 1:3] = 9
    model_config = {
        "labels": {"core": 7, "appendage": 9},
        "targets": {
            "registration": "core",
            "model": ["core", "appendage"],
        },
    }

    registration_mask = _workflow_active_mask(mask, model_config, {})
    model_mask = _workflow_model_mask(mask, model_config)

    assert int(np.count_nonzero(registration_mask)) == int(np.count_nonzero(mask == 7))
    assert not np.any(registration_mask[mask == 9])
    assert int(np.count_nonzero(model_mask)) == int(np.count_nonzero(mask > 0))


def test_reference_space_scaling_preserves_reference_pose_about_origin():
    reference = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [10.0, 0.0, 1.0],
        ]
    )
    sample = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    scaled, meta = _scale_reference_points_preserving_pose(
        reference_points=reference,
        sample_points=sample,
        registration_config={
            "reference_scaling": {
                "enabled": True,
                "min_factors": [0.0, 0.0, 0.0],
                "max_factors": [10.0, 10.0, 10.0],
            }
        },
    )

    scale = np.asarray(meta["scale_factors"])
    np.testing.assert_allclose(scaled, reference * scale)
    assert meta["source"] == "origin_covariance_axis_lengths_reference_pose"


def test_reference_space_scaling_uses_covariance_axis_lengths():
    reference = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [0.0, 24.0, 0.0],
            [12.0, 24.0, 0.0],
            [0.0, 0.0, 36.0],
            [12.0, 0.0, 36.0],
            [0.0, 24.0, 36.0],
            [12.0, 24.0, 36.0],
            [30.0, 0.0, 0.0],
        ]
    )
    sample = reference * np.asarray([1.5, 0.75, 1.25])

    _scaled, meta = _scale_reference_points_preserving_pose(
        reference_points=reference,
        sample_points=sample,
        registration_config={
            "reference_scaling": {
                "enabled": True,
                "min_factors": [0.0, 0.0, 0.0],
                "max_factors": [10.0, 10.0, 10.0],
            }
        },
    )

    def covariance_principal_axis_lengths(points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=0)
        eigvals = np.linalg.eigvalsh(np.cov(centered.T))
        return np.sqrt(np.maximum(eigvals, 0.0)) * 2.0

    reference_lengths = covariance_principal_axis_lengths(reference)
    sample_lengths = covariance_principal_axis_lengths(sample)
    np.testing.assert_allclose(meta["reference_axis_lengths"], reference_lengths)
    np.testing.assert_allclose(meta["sample_axis_lengths"], sample_lengths)
    np.testing.assert_allclose(
        meta["scale_factors"],
        sample_lengths / np.maximum(reference_lengths, 1.0e-6),
    )


def test_reference_model_space_icp_direction_defaults_to_reference_to_sample():
    assert _reference_model_space_icp_direction({}) == "reference_to_sample"
    assert (
        _reference_model_space_icp_direction({"icp_direction": "reference-to-sample"})
        == "reference_to_sample"
    )

    with pytest.raises(ValueError, match="no longer selectable"):
        _reference_model_space_icp_direction({"icp_direction": "sample_to_reference"})
    with pytest.raises(ValueError, match="registration\\.icp_direction"):
        _reference_model_space_icp_direction({"icp_direction": "sideways"})


def test_invert_rigid_transform_uses_row_vector_point_convention():
    angle = np.deg2rad(20.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.asarray([3.0, -2.0, 5.0])
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-4.0, 5.0, -6.0],
        ]
    )

    transformed = points @ rotation.T + translation
    inverse_rotation, inverse_translation = _invert_rigid_transform(
        rotation,
        translation,
    )

    recovered = transformed @ inverse_rotation.T + inverse_translation
    np.testing.assert_allclose(recovered, points)


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

    reference_points = _canonical_surface_points_from_mask_image(
        tmp_path / "mask.nii.gz",
        2,
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    built = build_model(
        {
            "type": "workflow_replay",
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


def test_workflow_replay_model_builds_spine_nonlinear_material(
    tmp_path: Path,
):
    density = np.ones((4, 4, 4), dtype=np.float32)
    density[2, 2, 2] = 0.0
    mask = np.ones_like(density, dtype=np.uint8) * 20
    disk_labels = np.zeros_like(mask, dtype=np.uint16)
    nodeset_labels = np.zeros_like(mask, dtype=np.uint16)
    nodeset_labels[0, :, :] = 101
    nodeset_labels[-1, :, :] = 202
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "disk_labels.npy", disk_labels)
    np.save(tmp_path / "nodesets.npy", nodeset_labels)

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 20},
            "workflow_replay": {
                "enabled": True,
                "disk_labels": "disk_labels.npy",
                "nodesets": "nodesets.npy",
            },
            "registration": {"enabled": False},
        },
        base_dir=tmp_path,
        material_config={
            "density": {
                "E": {
                    "equation": "power",
                    "coefficient": 3814.4,
                    "exponent": 1.05,
                },
                "nu": 0.3,
            },
            "nonlinear": {"preset": "spine_nonlinear"},
        },
        load_case_config={
            "type": "nodeset",
            "fixed": [{"nodeset": "inferior", "dofs": ["x", "y", "z"], "value": 0.0}],
            "prescribed": [{"nodeset": "superior", "dof": "z", "value": -0.1}],
        },
        nodeset_config={
            "inferior": {
                "type": "label_image",
                "label": 101,
                "selection": "surface_nodes",
            },
            "superior": {
                "type": "label_image",
                "label": 202,
                "selection": "surface_nodes",
            },
        },
    )

    assert built.nonlinear_material is not None
    assert (
        built.nonlinear_material.to_hdf5_attrs()["type"]
        == "AsymmetricPerfectPlasticDensityMap"
    )
    np.testing.assert_allclose(
        built.material,
        built.nonlinear_material.youngs_modulus_mpa,
    )
    assert built.nonlinear_material.compressive_yield_mpa.shape == built.material.shape
    assert built.nonlinear_material.tensile_yield_mpa.shape == built.material.shape


def test_workflow_replay_nonlinear_assigns_pmma_disks_as_elastic_fixture_material(
    tmp_path: Path,
):
    density = np.ones((4, 4, 4), dtype=np.float32)
    mask = np.ones_like(density, dtype=np.uint8) * 20
    disk_labels = np.zeros_like(mask, dtype=np.uint16)
    disk_labels[0, :, :] = 201
    nodeset_labels = np.zeros_like(mask, dtype=np.uint16)
    nodeset_labels[0, :, :] = 101
    nodeset_labels[-1, :, :] = 202
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "disk_labels.npy", disk_labels)
    np.save(tmp_path / "nodesets.npy", nodeset_labels)

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 20},
            "workflow_replay": {
                "enabled": True,
                "disk_labels": "disk_labels.npy",
                "nodesets": "nodesets.npy",
            },
            "registration": {"enabled": False},
        },
        base_dir=tmp_path,
        material_config={
            "density": {
                "E": {
                    "equation": "power",
                    "coefficient": 3814.4,
                    "exponent": 1.05,
                },
                "nu": 0.3,
            },
            "pmma": {"E": 2500.0, "nu": 0.31},
            "nonlinear": {"preset": "spine_nonlinear"},
        },
        load_case_config={
            "type": "nodeset",
            "fixed": [
                {"nodeset": "inferior", "dofs": ["x", "y", "z"], "value": 0.0}
            ],
            "prescribed": [{"nodeset": "superior", "dof": "z", "value": -0.1}],
        },
        nodeset_config={
            "inferior": {
                "type": "label_image",
                "label": 101,
                "selection": "surface_nodes",
            },
            "superior": {
                "type": "label_image",
                "label": 202,
                "selection": "surface_nodes",
            },
        },
    )

    disk_mask = np.isclose(built.material, 2500.0)
    assert np.count_nonzero(disk_mask) > 0
    assert built.nonlinear_material is not None
    assert np.all(built.nonlinear_material.material_id[disk_mask] == 2)
    assert np.all(built.nonlinear_material.compressive_yield_mpa[disk_mask] == 0.0)
    assert np.all(built.nonlinear_material.tensile_yield_mpa[disk_mask] == 0.0)
    assert np.all(built.nonlinear_material.plateau_mpa[disk_mask] == 0.0)
    assert np.allclose(built.nonlinear_material.poisson_ratio[disk_mask], 0.31)
    assert np.all(built.nonlinear_material.material_id[built.material == 0.0] == 0)


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

    reference_points = _canonical_surface_points_from_mask_image(
        tmp_path / "mask.nii.gz",
        2,
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

    reference_points = _canonical_surface_points_from_mask_image(
        tmp_path / "mask.nii.gz",
        2,
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    built = build_model(
        {
            "type": "workflow_replay",
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
                        "intrusion_depth_mm": 1.0,
                        "use_plane_size": True,
                        "center_ras": [-3.5, -3.5, 7.0],
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


def test_workflow_replay_exports_generated_nodeset_labels_from_planes(tmp_path: Path):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 20
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"body": 20},
            "outputs": {
                "material_image": str(tmp_path / "model" / "material.nii.gz"),
                "nodeset_image": str(tmp_path / "model" / "nodesets.nii.gz"),
                "manifest": str(tmp_path / "model" / "model.json"),
            },
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
                        "intrusion_depth_mm": 1.0,
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
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": "-10%"}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    labels_zyx, _spacing, _origin = read_image_zyx(tmp_path / "model" / "nodesets.nii.gz")
    labels_xyz = np.transpose(labels_zyx, (2, 1, 0))
    material_xyz = np.transpose(built.material, (2, 1, 0))
    assert 1 not in set(np.unique(labels_zyx).astype(int))
    reconstructed = nodes_from_labeled_voxels(
        labels_xyz,
        label=201,
        selection="surface_nodes",
        material=material_xyz,
    )
    assert built.metadata["model"]["workflow_replay"]["geometry_mode"] == "plane_driven"
    assert reconstructed == built.node_sets["superior_disk"]


def test_workflow_replay_exports_generated_disk_labels_for_visual_review(tmp_path: Path):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 20
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"body": 20},
            "outputs": {
                "material_image": str(tmp_path / "model" / "material.nii.gz"),
                "nodeset_image": str(tmp_path / "model" / "nodesets.nii.gz"),
                "disk_label_image": str(tmp_path / "model" / "disks.nii.gz"),
                "manifest": str(tmp_path / "model" / "model.json"),
            },
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
                        "intrusion_depth_mm": 1.0,
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
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": "-10%"}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    disk_labels_zyx, _spacing, _origin = read_image_zyx(
        tmp_path / "model" / "disks.nii.gz"
    )
    nodeset_labels_zyx, _spacing, _origin = read_image_zyx(
        tmp_path / "model" / "nodesets.nii.gz"
    )

    assert int(np.count_nonzero(disk_labels_zyx == 10001)) > int(
        np.count_nonzero(nodeset_labels_zyx == 201)
    )


def test_editor_disk_labels_default_to_reserved_range(tmp_path: Path):
    labels = _editor_disk_labels(
        resolved_editor={
            "planes": [
                {"name": "Top disk", "contact": "Material disks"},
                {"name": "Bottom disk", "contact": "PMMA caps"},
                {"name": "Bone surface", "contact": "Bone surface"},
            ],
        },
        replay_cfg={},
        base_dir=tmp_path,
    )

    assert labels == {"Top disk": 10001, "Bottom disk": 10002}


def test_workflow_replay_honors_explicit_plane_disk_labels_without_cached_labelmap(
    tmp_path: Path,
):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 20
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"body": 20},
            "outputs": {
                "material_image": str(tmp_path / "model" / "material.nii.gz"),
                "nodeset_image": str(tmp_path / "model" / "nodesets.nii.gz"),
                "manifest": str(tmp_path / "model" / "model.json"),
            },
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
                        "intrusion_depth_mm": 1.0,
                        "disk_label": 222,
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
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": "-10%"}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    assert built.metadata["model"]["workflow_replay"]["disk_labels"] is None
    assert built.element_sets["disk_label_222"] > 0
    assert "disk_label_201" not in built.element_sets


def test_workflow_replay_final_crop_removes_empty_canvas_and_shifts_nodes():
    material_xyz = np.zeros((5, 6, 9), dtype=np.float32)
    material_xyz[1:4, 2:5, 2:6] = 1000.0
    labels_xyz = np.zeros_like(material_xyz, dtype=np.uint16)
    labels_xyz[1:4, 2:5, 2:6] = 1
    node_label_xyz = np.zeros_like(labels_xyz)
    node_label_xyz[1:4, 2:5, 5:6] = 201

    cropped = _crop_workflow_model_to_material_bbox(
        material_xyz=material_xyz,
        labels_xyz=labels_xyz,
        node_label_xyz=node_label_xyz,
        spacing=(1.0, 2.0, 3.0),
        origin=(10.0, 20.0, 30.0),
        node_sets={
            "support": [
                (1, 2, 5),
                (4, 5, 6),
            ]
        },
        percent_reference_node_sets={
            "support": [
                (1, 2, 5),
                (4, 5, 6),
            ]
        },
    )

    assert cropped.material_xyz.shape == (3, 3, 4)
    assert cropped.labels_xyz.shape == (3, 3, 4)
    assert cropped.node_label_xyz.shape == (3, 3, 4)
    assert cropped.origin == pytest.approx((11.0, 24.0, 36.0))
    assert cropped.crop["lower_index_xyz"] == [1, 2, 2]
    assert cropped.crop["upper_index_xyz"] == [4, 5, 6]
    assert cropped.node_sets["support"] == [(0, 0, 3), (3, 3, 4)]
    assert cropped.percent_reference_node_sets["support"] == [(0, 0, 3), (3, 3, 4)]
    assert np.count_nonzero(cropped.material_xyz) == np.count_nonzero(material_xyz)


def test_workflow_replay_keeps_generated_outer_face_node_sets(tmp_path: Path):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 20
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

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
                        "intrusion_depth_mm": 1.0,
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
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": "-10%"}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "outer_face_nodes",
            }
        },
    )

    nodes = np.asarray(built.node_sets["superior_disk"], dtype=int)

    assert built.metadata["model"]["workflow_replay"]["geometry_mode"] == "plane_driven"
    assert np.unique(nodes[:, 2]).tolist() == [built.material.shape[0]]
    assert int(np.count_nonzero(built.postprocess_mask)) == built.element_sets["bone"]
    assert int(np.count_nonzero(built.postprocess_mask)) < int(np.count_nonzero(built.material))


def test_workflow_replay_resolves_editor_plane_normal_displacement_sign(
    tmp_path: Path,
):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 2
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

    built = build_workflow_replay_model(
        {
            "type": "workflow_replay",
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "labels": {"femur": 2},
            "workflow_replay": {"enabled": True},
            "registration": {"enabled": False},
            "slicer_editor": {
                "planes": [
                    {
                        "name": "Greater trochanter disk",
                        "relative_to": "model_bbox",
                        "center_fraction": [0.5, -0.25, 0.5],
                        "size_fraction": [1.5, 1.5],
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "rectangle",
                        "thickness_mm": 2.0,
                        "intrusion_depth_mm": 0.0,
                        "normal_ras": [0.0, 1.0, 0.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 0.0, 1.0],
                        "disk_label": 101,
                    },
                    {
                        "name": "Femoral head disk",
                        "relative_to": "model_bbox",
                        "center_fraction": [0.5, 1.25, 0.5],
                        "size_fraction": [1.5, 1.5],
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "rectangle",
                        "thickness_mm": 2.0,
                        "intrusion_depth_mm": 0.0,
                        "normal_ras": [0.0, -1.0, 0.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 0.0, 1.0],
                        "disk_label": 202,
                    },
                ],
                "loads": [
                    {
                        "nodeset": "Greater trochanter disk",
                        "mode": "Fixed",
                        "direction": "Plane normal",
                        "value": 0.0,
                        "units": "",
                        "fixed_dofs": ["y"],
                    },
                    {
                        "nodeset": "Femoral head disk",
                        "mode": "Displacement",
                        "direction": "Plane normal",
                        "value": 10.0,
                        "units": "%",
                    },
                ],
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
            "fixed": [{"nodeset": "greater_trochanter_disk", "dofs": ["y"], "value": 0.0}],
            "prescribed": [{"nodeset": "femoral_head_disk", "dof": "y", "value": "10%"}],
        },
        nodeset_config={
            "greater_trochanter_disk": {
                "type": "label_image",
                "label": 101,
                "selection": "outer_face_nodes",
            },
            "femoral_head_disk": {
                "type": "label_image",
                "label": 202,
                "selection": "outer_face_nodes",
            },
        },
    )

    head_nodes = set(built.node_sets["femoral_head_disk"])
    coords = built.boundary_conditions.fixed_coordinates
    values = built.boundary_conditions.fixed_values
    head_y_values = [
        float(value)
        for coord, value in zip(coords, values, strict=True)
        if tuple(int(item) for item in coord[:3]) in head_nodes and int(coord[3]) == 1
    ]

    assert head_y_values
    assert all(value < 0.0 for value in head_y_values)
    assert built.metadata["model"]["effective_load_case"]["prescribed"] == [
        {
            "nodeset": "femoral_head_disk",
            "dof": "y",
            "value": "-10%",
            "units": "%",
        }
    ]


def test_workflow_replay_outer_face_nodes_keep_surface_percent_length(
    tmp_path: Path,
):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 20
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

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
                        "intrusion_depth_mm": 0.0,
                        "normal_ras": [0.0, 0.0, -1.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 1.0, 0.0],
                    },
                    {
                        "name": "Inferior disk",
                        "relative_to": "model_bbox",
                        "center_fraction": [0.5, 0.5, -0.25],
                        "size_fraction": [1.5, 1.5],
                        "contact": "Material disks",
                        "surface_mode": "project_bounded",
                        "shape": "anatomy",
                        "thickness_mm": 2.0,
                        "intrusion_depth_mm": 0.0,
                        "normal_ras": [0.0, 0.0, 1.0],
                        "u_axis_ras": [1.0, 0.0, 0.0],
                        "v_axis_ras": [0.0, 1.0, 0.0],
                    },
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
            "fixed": [{"nodeset": "inferior_disk", "dofs": ["x", "y", "z"], "value": 0.0}],
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": "-10%"}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "outer_face_nodes",
            },
            "inferior_disk": {
                "type": "label_image",
                "label": 102,
                "selection": "outer_face_nodes",
            },
        },
    )

    superior_nodes = np.asarray(built.node_sets["superior_disk"], dtype=int)
    inferior_nodes = np.asarray(built.node_sets["inferior_disk"], dtype=int)
    prescribed = built.boundary_conditions.fixed_values[
        np.abs(built.boundary_conditions.fixed_values) > 0.0
    ]

    assert np.unique(superior_nodes[:, 2]).tolist() == [built.material.shape[0]]
    assert np.unique(inferior_nodes[:, 2]).tolist() == [0]
    assert built.boundary_conditions.reference_lengths_mm["z"] == pytest.approx(8.0)
    assert prescribed.size > 0
    assert np.unique(prescribed).tolist() == pytest.approx([-0.8])


def test_workflow_replay_percent_displacement_uses_occupied_model_length_with_disks(
    tmp_path: Path,
):
    density = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 20
    sitk.WriteImage(sitk.GetImageFromArray(density), str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.nii.gz"))

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
                        "intrusion_depth_mm": 0.0,
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
            "prescribed": [{"nodeset": "superior_disk", "dof": "z", "value": "-10%"}],
        },
        nodeset_config={
            "superior_disk": {
                "type": "label_image",
                "label": 201,
                "selection": "surface_nodes",
            }
        },
    )

    values = built.boundary_conditions.fixed_values
    prescribed = values[np.abs(values) > 0.0]
    material_xyz = np.transpose(built.material, (2, 1, 0))
    occupied = occupied_length_mm(material_xyz, axis="z", spacing=built.spacing)
    assert prescribed.size > 0
    assert float(np.min(prescribed)) == pytest.approx(-0.10 * occupied)


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

    reference_points = _canonical_surface_points_from_mask_image(
        tmp_path / "mask.nii.gz",
        20,
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
                        "intrusion_depth_mm": 1.0,
                        "use_plane_size": True,
                        "center_ras": [-4.5, -4.5, 8.0],
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


def test_workflow_replay_preview_uses_same_reference_grid_as_model_builder(
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

    reference_points = _canonical_surface_points_from_mask_image(
        tmp_path / "mask.nii.gz",
        20,
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    model_config = {
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
                    "intrusion_depth_mm": 1.0,
                    "use_plane_size": True,
                    "center_ras": [-4.5, -4.5, 8.0],
                    "normal_ras": [0.0, 0.0, -1.0],
                    "u_axis_ras": [1.0, 0.0, 0.0],
                    "v_axis_ras": [0.0, 1.0, 0.0],
                    "size_mm": [4.0, 4.0],
                }
            ]
        },
    }

    preview = build_workflow_replay_preview(
        model_config,
        base_dir=tmp_path,
    )
    built = build_workflow_replay_model(
        model_config,
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

    assert preview.metadata["model_space"] == "reference"
    assert preview.metadata["registration"]["applied_to_model_grid"] is True
    assert preview.spacing == pytest.approx(built.spacing)
    offset_xyz = np.rint(
        (np.asarray(built.origin) - np.asarray(preview.origin))
        / np.asarray(preview.spacing)
    ).astype(int)
    assert np.all(offset_xyz >= 0)
    offset_zyx = offset_xyz[::-1]
    shape_zyx = np.asarray(built.postprocess_mask.shape)
    upper_zyx = offset_zyx + shape_zyx
    assert np.all(upper_zyx <= np.asarray(preview.model_mask_zyx.shape))
    preview_subset = preview.model_mask_zyx[
        offset_zyx[0] : upper_zyx[0],
        offset_zyx[1] : upper_zyx[1],
        offset_zyx[2] : upper_zyx[2],
    ]
    np.testing.assert_array_equal(preview_subset, built.postprocess_mask)


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
                        "intrusion_depth_mm": 1.0,
                        "use_plane_size": True,
                        "center_ras": [-3.5, -3.5, 7.0],
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


def test_bbox_relative_editor_resolves_planes_from_fraction_bounds():
    mask = np.zeros((8, 12, 16), dtype=bool)
    mask[2:6, 3:9, 4:14] = True
    editor = {
        "planes": [
            {
                "name": "Greater trochanter disk",
                "relative_to": "model_bbox",
                "bbox_fraction_bounds": {
                    "x": [0.0, 1.0],
                    "y": [-0.1, -0.1],
                    "z": [0.0, 1.0],
                },
                "normal_ras": [0.0, 1.0, 0.0],
                "u_axis_ras": [0.0, 0.0, -1.0],
                "v_axis_ras": [-1.0, 0.0, 0.0],
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
    assert "center_fraction" not in plane
    assert "size_fraction" not in plane
    assert plane["center_ras"] == pytest.approx([14.25, 25.0, 40.5])
    assert plane["size_mm"] == pytest.approx([9.0, 4.5])
    assert plane["relative_definition"]["bbox_fraction_bounds"] == {
        "x": [0.0, 1.0],
        "y": [-0.1, -0.1],
        "z": [0.0, 1.0],
    }


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
                        "intrusion_depth_mm": 1.0,
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
    resolved_editor = built.metadata["model"]["workflow_replay"]["resolved_editor"]
    full_plane = resolved_editor["planes"][0]
    assert full_plane["name"] == "Superior disk"
    assert full_plane["relative_to"] == "resolved_model_bbox"
    assert full_plane["center_ras"] == pytest.approx(resolved_plane["center_ras"])
    assert full_plane["normal_ras"] == pytest.approx([0.0, 0.0, -1.0])
    assert full_plane["u_axis_ras"] == pytest.approx([1.0, 0.0, 0.0])
    assert full_plane["v_axis_ras"] == pytest.approx([0.0, 1.0, 0.0])
    assert full_plane["thickness_mm"] == pytest.approx(2.0)
    assert full_plane["intrusion_depth_mm"] == pytest.approx(1.0)


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
                        "intrusion_depth_mm": 0.0,
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

    crop = built.metadata["model"]["workflow_replay"]["final_material_crop"]
    assert crop["enabled"] is True
    assert crop["original_shape_xyz"][2] > crop["cropped_shape_xyz"][2]
    assert built.material.shape[0] == crop["cropped_shape_xyz"][2]
    assert int(built.element_sets["workflow_disks"]) > 0
    assert len(built.node_sets["superior_disk"]) > 0
