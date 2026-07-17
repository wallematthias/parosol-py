from pathlib import Path

import h5py
import numpy as np
import pytest

from parosol_py.results import DEFAULT_OUTPUTS, read_solution_fields


def test_read_solution_fields_defaults_to_native_solution_outputs(tmp_path: Path):
    h5_path = tmp_path / "solved.h5"
    with h5py.File(h5_path, "w") as h5:
        sol = h5.create_group("Solution")
        sol.create_dataset("SED", data=np.array([1.0, 2.0], dtype=np.float32))
        sol.create_dataset("EFF", data=np.array([0.1, 0.2], dtype=np.float32))
        sol.create_dataset("VonMises", data=np.array([10.0, 20.0], dtype=np.float32))

    fields = read_solution_fields(h5_path)

    assert tuple(fields) == DEFAULT_OUTPUTS
    assert np.allclose(fields["sed"], [1.0, 2.0])
    assert np.allclose(fields["effective_strain"], [0.1, 0.2])
    assert np.allclose(fields["von_mises"], [10.0, 20.0])


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


def test_read_solution_fields_plastic_strain_tensor_and_magnitude(tmp_path: Path):
    h5_path = tmp_path / "solved.h5"
    with h5py.File(h5_path, "w") as h5:
        sol = h5.create_group("Solution")
        sol.create_dataset(
            "PlasticStrain",
            data=np.array(
                [
                    [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 12.0, 0.0, 0.0, 5.0],
                ],
                dtype=np.float32,
            ),
        )

    fields = read_solution_fields(
        h5_path,
        outputs=("plastic_strain", "plastic_strain_magnitude"),
    )

    assert fields["plastic_strain"].shape == (2, 6)
    assert np.allclose(fields["plastic_strain"][:, 1], [4.0, 0.0])
    assert np.allclose(fields["plastic_strain_magnitude"], [5.0, 13.0])


def test_read_solution_fields_nonlinear_work_scalars(tmp_path: Path):
    h5_path = tmp_path / "solved.h5"
    with h5py.File(h5_path, "w") as h5:
        sol = h5.create_group("Solution")
        sol.create_dataset("SED", data=np.array([1.0, 2.0], dtype=np.float32))
        sol.create_dataset(
            "PlasticDissipation",
            data=np.array([0.5, 0.25], dtype=np.float32),
        )

    fields = read_solution_fields(
        h5_path,
        outputs=("plastic_dissipation", "mechanical_work_density"),
    )

    assert np.allclose(fields["plastic_dissipation"], [0.5, 0.25])
    assert np.allclose(fields["mechanical_work_density"], [1.5, 2.25])


def test_read_solution_fields_reports_missing_nonlinear_outputs_actionably(
    tmp_path: Path,
):
    h5_path = tmp_path / "solved.h5"
    with h5py.File(h5_path, "w") as h5:
        sol = h5.create_group("Solution")
        sol.create_dataset("SED", data=np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match="requires a nonlinear ParOSol solve"):
        read_solution_fields(h5_path, outputs=("plastic_strain_magnitude",))
