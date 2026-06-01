import numpy as np
import pytest

from parosol_py.diagnostics import build_fea_diagnostics


def test_shear_failure_uses_same_factor_and_reports_generalized_force():
    stiffness = np.ones((1, 1, 1), dtype=np.float32)
    forces = np.zeros((8, 3), dtype=np.float32)
    for index, coord in enumerate(_unit_cube_nodes()):
        if coord[2] == 1:
            forces[index, 0] = 2.0
    fields = {
        "forces": forces,
        "sed": np.full((1,), 0.05, dtype=np.float32),
    }

    diagnostics = build_fea_diagnostics(
        fields=fields,
        stiffness_gpa_xyz=stiffness,
        axis="z",
        strain=0.01,
        load_case_type="shear",
        load_direction="x",
        critical_strain=0.02,
        critical_volume_percent=100,
    )

    assert diagnostics["mechanics"]["generalized_load"]["name"] == "force"
    assert diagnostics["mechanics"]["generalized_load"]["value"] == pytest.approx(8.0)
    assert diagnostics["failure"]["factor"] == pytest.approx(2.0)
    assert diagnostics["failure"]["failure_generalized_load"]["value"] == pytest.approx(
        16.0
    )


def test_torsion_reports_moment_and_rotational_failure_load():
    stiffness = np.ones((1, 1, 1), dtype=np.float32)
    forces = np.zeros((8, 3), dtype=np.float32)
    for index, coord in enumerate(_unit_cube_nodes()):
        if coord == (1, 0, 1):
            forces[index, 1] = 1.0
        if coord == (0, 1, 1):
            forces[index, 0] = -1.0
    fields = {
        "forces": forces,
        "sed": np.full((1,), 0.05, dtype=np.float32),
    }

    diagnostics = build_fea_diagnostics(
        fields=fields,
        stiffness_gpa_xyz=stiffness,
        axis="z",
        strain=0.0,
        load_case_type="torsion",
        rotation_degrees=1.0,
        critical_strain=0.02,
        critical_volume_percent=100,
    )

    assert diagnostics["mechanics"]["generalized_load"]["name"] == "moment"
    assert diagnostics["mechanics"]["generalized_load"]["value"] == pytest.approx(1.0)
    assert (
        diagnostics["mechanics"]["generalized_stiffness"]["name"]
        == "rotational_stiffness"
    )
    assert diagnostics["failure"]["failure_generalized_load"]["value"] == pytest.approx(
        2.0
    )


def _unit_cube_nodes():
    return sorted(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        key=lambda coord: coord[0] + 2 * coord[1] + 4 * coord[2],
    )
