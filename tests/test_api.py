import numpy as np
import pytest

import parosol_py
from parosol_py import solve
from parosol_py.api import solve_aim


def test_package_imports():
    assert parosol_py.__version__ == "0.1.0"


def test_solve_dry_run_writes_input_and_reports_summary(tmp_path):
    material_zyx = np.zeros((4, 3, 2))
    material_zyx[:, 1, 1] = 1000.0

    result = solve(
        material=material_zyx,
        spacing=(0.061, 0.061, 0.061),
        origin=(1.0, 2.0, 3.0),
        material_unit="MPa",
        test="axial",
        test_axis="z",
        strain=-0.01,
        outputs=("sed",),
        work_dir=tmp_path,
        dry_run=True,
    )

    assert result.input_file.exists()
    assert result.command[-1] == str(result.input_file)
    assert "--SED" in result.command
    assert result.fields == {}
    assert result.summary.dimensions_xyz == (2, 3, 4)
    assert result.summary.spacing == (0.061, 0.061, 0.061)


def test_solve_rejects_anisotropic_spacing_in_dry_run(tmp_path):
    material_zyx = np.zeros((4, 3, 2))
    material_zyx[:, 1, 1] = 1000.0

    with pytest.raises(ValueError, match="isotropic|anisotropic"):
        solve(
            material=material_zyx,
            spacing=(0.061, 0.08, 0.061),
            work_dir=tmp_path,
            dry_run=True,
        )


def test_solve_aim_reads_aim_and_preserves_metadata(monkeypatch, tmp_path):
    calls = {}

    def fake_read_aim(path):
        calls["path"] = path
        arr = np.zeros((4, 3, 2))
        arr[:, 1, 1] = 1000.0
        return arr, {
            "element_size": (0.061, 0.061, 0.061),
            "position": (1.0, 2.0, 3.0),
        }

    monkeypatch.setattr("parosol_py.api.read_aim", fake_read_aim)

    result = solve_aim("case.aim", work_dir=tmp_path, dry_run=True)

    assert calls["path"] == "case.aim"
    assert result.input_file.exists()
    assert result.summary.spacing == (0.061, 0.061, 0.061)
