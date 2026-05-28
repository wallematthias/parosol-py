from pathlib import Path

import numpy as np
import SimpleITK as sitk

from parosol_py.images import ImageGrid, export_scalar_image, normalize_array


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
    grid = ImageGrid(array_xyz=arr_xyz, spacing=(0.1, 0.2, 0.3), origin=(1.0, 2.0, 3.0))
    out = tmp_path / "sed.nii.gz"

    export_scalar_image(grid, out)

    img = sitk.ReadImage(str(out))
    arr_zyx = sitk.GetArrayFromImage(img)
    assert arr_zyx.shape == (2, 3, 4)
    assert arr_zyx[1, 2, 3] == 7.0
    assert tuple(round(v, 6) for v in img.GetSpacing()) == (0.1, 0.2, 0.3)
    assert tuple(round(v, 6) for v in img.GetOrigin()) == (1.0, 2.0, 3.0)
