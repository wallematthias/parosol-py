from __future__ import annotations

from pathlib import Path

import numpy as np

from parosol_py.modeling.workflow_replay import _plane_disk_required_bounds
from parosol_py.workflow_geometry import read_reference_points


def test_workflow_replay_padding_uses_intrusion_depth_key():
    bounds = _plane_disk_required_bounds(
        {
            "center_ras": [0.0, 0.0, 0.0],
            "normal_ras": [0.0, 0.0, 1.0],
            "u_axis_ras": [1.0, 0.0, 0.0],
            "v_axis_ras": [0.0, 1.0, 0.0],
            "size_mm": [10.0, 10.0],
            "thickness_mm": 2.0,
            "intrusion_depth_mm": 5.0,
        },
        active_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        spacing=(1.0, 1.0, 1.0),
    )

    assert bounds is not None
    _minimum, maximum = bounds
    assert maximum[2] == 6.0


def test_workflow_geometry_reference_reader_supports_npy(tmp_path: Path):
    reference = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=float,
    )
    path = tmp_path / "slicer_reference_points.npy"
    np.save(path, reference)

    np.testing.assert_allclose(read_reference_points(path), reference)
