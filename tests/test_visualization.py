from pathlib import Path

import numpy as np

from parosol_py.core import BoundaryConditionSet
from parosol_py.visualization import (
    _mid_slices,
    _panel_vector_scale,
    write_case_overview,
)


def test_write_case_overview_creates_png_with_material_field_and_bcs(tmp_path: Path):
    material = np.zeros((4, 4, 4), dtype=np.float32)
    material[1:3, 1:3, :] = 1000.0
    sed = np.linspace(0.0, 1.0, material.size, dtype=np.float32).reshape(material.shape)
    boundary_conditions = BoundaryConditionSet(
        fixed_coordinates=np.asarray(
            [[1, 1, 0, 2], [2, 2, 4, 2], [1, 2, 4, 0]], dtype=np.uint16
        ),
        fixed_values=np.asarray([0.0, -0.04, 0.01], dtype=np.float32),
        loaded_coordinates=np.asarray([[2, 1, 4, 1]], dtype=np.uint16),
        loaded_values=np.asarray([5.0], dtype=np.float32),
    )

    path = write_case_overview(
        material,
        output_path=tmp_path / "overview.png",
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        field_xyz=sed,
        field_name="SED",
        boundary_conditions=boundary_conditions,
        title="cube",
    )

    assert path == (tmp_path / "overview.png").resolve()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_write_case_overview_summarizes_large_boundary_condition_sets(
    tmp_path: Path,
):
    material = np.ones((32, 32, 8), dtype=np.float32) * 1000.0
    sed = np.ones_like(material, dtype=np.float32)
    bottom = []
    top = []
    for x in range(33):
        for y in range(33):
            bottom.append([x, y, 0, 2])
            top.append([x, y, 8, 2])
    boundary_conditions = BoundaryConditionSet(
        fixed_coordinates=np.asarray([*bottom, *top], dtype=np.uint16),
        fixed_values=np.asarray(
            [0.0] * len(bottom) + [-0.08] * len(top), dtype=np.float32
        ),
    )

    path = write_case_overview(
        material,
        output_path=tmp_path / "large_bc_overview.png",
        spacing=(1.0, 1.0, 1.0),
        field_xyz=sed,
        boundary_conditions=boundary_conditions,
        title="large_bc",
    )

    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_panel_vector_scale_makes_small_displacements_visible():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((126, 108)), extent=(-0.5, 107.5, -0.5, 125.5))

    scale = _panel_vector_scale(ax, np.asarray([0.0]), np.asarray([-0.076]))

    plt.close(fig)
    assert abs(scale * -0.076) > 8.0


def test_mid_slices_use_physical_spacing_for_extents_and_labels():
    material = np.ones((4, 5, 6), dtype=np.float32)

    slices = _mid_slices(material, spacing=(0.1, 0.2, 0.3), origin=(1.0, 2.0, 3.0))

    assert slices["axial"].extent == (0.95, 1.35, 1.9, 2.9)
    assert slices["sagittal"].extent == (1.9, 2.9, 2.85, 4.65)
    assert slices["coronal"].extent == (0.95, 1.35, 2.85, 4.65)
    assert slices["axial"].xlabel == "x (mm)"
    assert slices["sagittal"].ylabel == "z (mm)"
