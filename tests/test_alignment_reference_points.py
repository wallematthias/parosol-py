from __future__ import annotations

from pathlib import Path

import numpy as np

from parosol_py.modeling.alignment import read_reference_points


def test_reference_points_reader_supports_npy_reference_cloud(tmp_path: Path):
    reference = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=float,
    )
    path = tmp_path / "slicer_reference_points.npy"
    np.save(path, reference)

    np.testing.assert_allclose(read_reference_points(path), reference)
