import numpy as np
import pytest

from parosol_py.nonlinear import hip_keaveny_nonlinear, spine_keaveny_nonlinear


def test_spine_keaveny_formula_uses_end_to_end_qct_law():
    rho_qct = np.array([[[0.0, 1.0, 2.0]]], dtype=np.float64)

    mapped = spine_keaveny_nonlinear(rho_qct)

    expected_E = 3814.4 * np.power(rho_qct, 1.05)
    expected_plateau = 57.4464 * np.power(rho_qct, 1.39)
    np.testing.assert_allclose(mapped.youngs_modulus_mpa, expected_E)
    np.testing.assert_allclose(mapped.compressive_yield_mpa, expected_plateau)
    np.testing.assert_allclose(mapped.tensile_yield_mpa, expected_plateau)
    np.testing.assert_allclose(mapped.plateau_mpa, expected_plateau)
    assert mapped.metadata["preset"] == "spine_keaveny"
    assert mapped.metadata["density_basis"] == "rho_qct"
    assert mapped.metadata["side_multiplier"] == pytest.approx(1.28)


def test_hip_keaveny_femoral_neck_formula_uses_end_to_end_law():
    rho_app = np.array([[[0.0, 1.0, 2.0]]], dtype=np.float64)

    mapped = hip_keaveny_nonlinear(rho_app, site="femoral_neck")

    expected_E = 8768.0 * np.power(rho_app, 1.49)
    np.testing.assert_allclose(mapped.youngs_modulus_mpa, expected_E)
    np.testing.assert_allclose(mapped.compressive_yield_mpa, 0.0085 * expected_E)
    np.testing.assert_allclose(mapped.tensile_yield_mpa, 0.0061 * expected_E)
    np.testing.assert_allclose(mapped.plateau_mpa, 0.0085 * expected_E)
    assert mapped.metadata["preset"] == "hip_keaveny"
    assert mapped.metadata["site"] == "femoral_neck"
    assert mapped.metadata["density_basis"] == "rho_app"


def test_hip_keaveny_greater_trochanter_formula_uses_end_to_end_law():
    rho_app = np.array([[[0.0, 1.0, 2.0]]], dtype=np.float64)

    mapped = hip_keaveny_nonlinear(rho_app, site="greater_trochanter")

    expected_E = 19212.8 * np.power(rho_app, 2.18)
    np.testing.assert_allclose(mapped.youngs_modulus_mpa, expected_E)
    np.testing.assert_allclose(mapped.compressive_yield_mpa, 0.0070 * expected_E)
    np.testing.assert_allclose(mapped.tensile_yield_mpa, 0.0061 * expected_E)
    np.testing.assert_allclose(mapped.plateau_mpa, 0.0070 * expected_E)
    assert mapped.metadata["site"] == "greater_trochanter"


def test_keaveny_presets_zero_inactive_voxels_from_mask():
    rho_qct = np.ones((1, 1, 3), dtype=np.float64)
    active_mask = np.array([[[True, False, True]]])

    mapped = spine_keaveny_nonlinear(rho_qct, active_mask=active_mask)

    assert mapped.youngs_modulus_mpa[0, 0, 0] > 0.0
    assert mapped.youngs_modulus_mpa[0, 0, 1] == 0.0
    assert mapped.compressive_yield_mpa[0, 0, 1] == 0.0
    assert mapped.tensile_yield_mpa[0, 0, 1] == 0.0
    assert mapped.material_id[0, 0, 1] == 0


def test_hip_keaveny_rejects_unknown_site():
    rho_app = np.ones((1, 1, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="site must be"):
        hip_keaveny_nonlinear(rho_app, site="shaft")


def test_keaveny_presets_reject_non_3d_input():
    with pytest.raises(ValueError, match="rho input must be 3D"):
        spine_keaveny_nonlinear(np.ones((2, 2), dtype=np.float64))


def test_keaveny_binning_uses_same_binned_rho_for_all_outputs():
    rho_app = np.array([[[0.0, 1.0, 3.0]]], dtype=np.float64)

    mapped = hip_keaveny_nonlinear(
        rho_app,
        site="femoral_neck",
        bin_material=True,
        number_bins=1,
    )

    expected_E = np.array([[[0.0, 8768.0 * 2.0**1.49, 8768.0 * 2.0**1.49]]])
    np.testing.assert_allclose(mapped.youngs_modulus_mpa, expected_E)
    np.testing.assert_allclose(mapped.compressive_yield_mpa, 0.0085 * expected_E)
    np.testing.assert_allclose(mapped.tensile_yield_mpa, 0.0061 * expected_E)
    assert mapped.metadata["bin_material"] is True
