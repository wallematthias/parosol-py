from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from parosol_py.images import (
    ImageGrid,
    export_scalar_image,
    normalize_array,
    restore_scalar_image_to_reference_grid,
)
from parosol_py.modeling.io import read_image_zyx


def test_normalize_array_accepts_zyx_default():
    arr_zyx = np.zeros((2, 3, 4), dtype=np.float32)
    arr_zyx[1, 2, 3] = 9.0

    grid = normalize_array(arr_zyx, spacing=(0.1, 0.2, 0.3), origin=(1.0, 2.0, 3.0))

    assert grid.array_xyz.shape == (4, 3, 2)
    assert grid.array_xyz[3, 2, 1] == 9.0
    assert grid.spacing == (0.1, 0.2, 0.3)
    assert grid.origin == (1.0, 2.0, 3.0)


def test_normalize_array_accepts_xyz():
    arr_xyz = np.zeros((4, 3, 2), dtype=np.float32)
    arr_xyz[3, 2, 1] = 11.0

    grid = normalize_array(
        arr_xyz,
        spacing=(0.1, 0.2, 0.3),
        origin=(0.0, 0.0, 0.0),
        array_order="xyz",
    )

    assert grid.array_xyz.shape == (4, 3, 2)
    assert grid.array_xyz[3, 2, 1] == 11.0


def test_export_scalar_image_roundtrips_nii_gz(tmp_path: Path):
    arr_xyz = np.zeros((4, 3, 2), dtype=np.float32)
    arr_xyz[3, 2, 1] = 7.0
    grid = ImageGrid(
        array_xyz=arr_xyz,
        spacing=(0.1, 0.2, 0.3),
        origin=(-1.0, -2.0, 3.0),
    )
    out = tmp_path / "sed.nii.gz"

    export_scalar_image(grid, out)

    img = sitk.ReadImage(str(out))
    arr_zyx = sitk.GetArrayFromImage(img)
    assert arr_zyx.shape == (2, 3, 4)
    assert arr_zyx[1, 2, 3] == 7.0
    assert tuple(round(v, 6) for v in img.GetSpacing()) == (0.1, 0.2, 0.3)
    assert tuple(round(v, 6) for v in img.GetOrigin()) == (1.0, 2.0, 3.0)
    assert tuple(round(v, 6) for v in img.GetDirection()) == (
        -1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    data_zyx, spacing, origin = read_image_zyx(out)
    assert data_zyx.shape == (2, 3, 4)
    assert data_zyx[1, 2, 3] == 7.0
    assert tuple(round(v, 6) for v in spacing) == (0.1, 0.2, 0.3)
    assert origin == (-1.0, -2.0, 3.0)


def test_restore_scalar_image_to_reference_grid_preserves_position_and_array_size(tmp_path: Path):
    reference = sitk.Image((8, 7, 6), sitk.sitkFloat32)
    reference.SetSpacing((0.4, 0.5, 0.6))
    reference.SetOrigin((12.0, -8.0, 3.0))
    reference.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    reference_path = tmp_path / "reference.nii.gz"
    sitk.WriteImage(reference, str(reference_path))

    crop_start_xyz = (2, 1, 3)
    crop_array = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4)) + 1.0
    cropped = sitk.GetImageFromArray(crop_array)
    cropped.SetSpacing(reference.GetSpacing())
    cropped.SetDirection(reference.GetDirection())
    cropped.SetOrigin(reference.TransformIndexToPhysicalPoint(crop_start_xyz))
    field_path = tmp_path / "sed.nii.gz"
    sitk.WriteImage(cropped, str(field_path))

    restored_path = restore_scalar_image_to_reference_grid(field_path, reference_path)

    restored = sitk.ReadImage(str(restored_path))
    saved_reference = sitk.ReadImage(str(reference_path))
    restored_array = sitk.GetArrayFromImage(restored)
    expected = np.zeros((6, 7, 8), dtype=np.float32)
    expected[3:5, 1:4, 2:6] = crop_array
    np.testing.assert_array_equal(restored_array, expected)
    assert restored.GetSize() == saved_reference.GetSize()
    assert restored.GetSpacing() == saved_reference.GetSpacing()
    assert restored.GetOrigin() == saved_reference.GetOrigin()
    assert restored.GetDirection() == saved_reference.GetDirection()


def test_restore_scalar_image_to_reference_grid_exactly_reorients_signed_axes(tmp_path: Path):
    reference = sitk.Image((8, 7, 2), sitk.sitkFloat32)
    reference.SetSpacing((0.4, 0.5, 0.6))
    reference.SetOrigin((12.0, -8.0, 3.0))
    reference_path = tmp_path / "reference.nii.gz"
    sitk.WriteImage(reference, str(reference_path))

    crop_array = np.zeros((2, 3, 4), dtype=np.float32)
    crop_array[:, 1, 2] = (1.25, 2.5)
    cropped = sitk.GetImageFromArray(crop_array)
    cropped.SetSpacing(reference.GetSpacing())
    cropped.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    # Small header round-off must not force interpolation of an integer lattice mapping.
    cropped.SetOrigin((14.80001, -5.00001, 3.0))
    field_path = tmp_path / "sed.nii.gz"
    sitk.WriteImage(cropped, str(field_path))

    restore_scalar_image_to_reference_grid(field_path, reference_path)

    restored = sitk.ReadImage(str(field_path))
    saved_reference = sitk.ReadImage(str(reference_path))
    expected = np.zeros((2, 7, 8), dtype=np.float32)
    expected[:, 4:7, 4:8] = crop_array[:, ::-1, ::-1]
    np.testing.assert_array_equal(sitk.GetArrayFromImage(restored), expected)
    assert np.count_nonzero(sitk.GetArrayFromImage(restored)) == 2
    assert restored.GetSize() == saved_reference.GetSize()
    assert restored.GetSpacing() == saved_reference.GetSpacing()
    assert restored.GetOrigin() == saved_reference.GetOrigin()
    assert restored.GetDirection() == saved_reference.GetDirection()


def test_restore_scalar_image_uses_parosol_reference_reader_for_non_itk_inputs(tmp_path: Path):
    reference_path = tmp_path / "reference.npz"
    np.savez_compressed(
        reference_path,
        image=np.ones((5, 6, 7), dtype=np.float32),
        spacing_xyz=np.asarray((0.2, 0.3, 0.4)),
        origin_xyz=np.asarray((10.0, 20.0, 30.0)),
    )
    crop_array = np.ones((2, 3, 4), dtype=np.float32)
    cropped = sitk.GetImageFromArray(crop_array)
    cropped.SetSpacing((0.2, 0.3, 0.4))
    cropped.SetOrigin((-10.2, -20.6, 30.8))
    cropped.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    field_path = tmp_path / "sed.nii.gz"
    sitk.WriteImage(cropped, str(field_path))

    restore_scalar_image_to_reference_grid(field_path, reference_path)

    restored = sitk.ReadImage(str(field_path))
    expected = np.zeros((5, 6, 7), dtype=np.float32)
    expected[2:4, 2:5, 1:5] = crop_array
    np.testing.assert_array_equal(sitk.GetArrayFromImage(restored), expected)
    assert restored.GetSize() == (7, 6, 5)
    assert restored.GetSpacing() == pytest.approx((0.2, 0.3, 0.4))
    assert restored.GetOrigin() == pytest.approx((-10.0, -20.0, 30.0))
