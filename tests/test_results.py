from pathlib import Path

import h5py
import numpy as np

from parosol_py.results import read_solution_fields


def test_read_solution_fields_scalar_and_tensor(tmp_path: Path):
    h5_path = tmp_path / "solved.h5"
    with h5py.File(h5_path, "w") as h5:
        sol = h5.create_group("Solution")
        sol.create_dataset("SED", data=np.array([1.0, 2.0], dtype=np.float32))
        sol.create_dataset("e_xx", data=np.array([0.1, 0.2], dtype=np.float32))
        sol.create_dataset("e_yy", data=np.array([0.3, 0.4], dtype=np.float32))
        sol.create_dataset("e_zz", data=np.array([0.5, 0.6], dtype=np.float32))
        sol.create_dataset("e_xy", data=np.array([0.7, 0.8], dtype=np.float32))
        sol.create_dataset("e_yz", data=np.array([0.9, 1.0], dtype=np.float32))
        sol.create_dataset("e_xz", data=np.array([1.1, 1.2], dtype=np.float32))

    fields = read_solution_fields(h5_path, outputs=("sed", "strain"))

    assert np.allclose(fields["sed"], [1.0, 2.0])
    assert set(fields["strain"]) == {"xx", "yy", "zz", "xy", "yz", "xz"}
    assert np.allclose(fields["strain"]["xz"], [1.1, 1.2])
