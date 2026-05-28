import numpy as np
import pytest

from parosol_py.materials import material_to_stiffness_gpa, parse_linear_isotropic_materials


def test_material_to_stiffness_gpa_from_mpa():
    material_mpa = np.array([[[0.0, 1000.0], [2000.0, 0.0]]], dtype=np.float64)
    out = material_to_stiffness_gpa(material_mpa, material_unit="MPa")
    assert out.dtype == np.float32
    assert np.allclose(out, np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=np.float32))


def test_material_to_stiffness_gpa_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        material_to_stiffness_gpa(np.array([[[-1.0]]]), material_unit="MPa")


def test_parse_linear_isotropic_materials():
    text = """MaterialDefinitions:
    Material_001:
        Type: LinearIsotropic
        E: 8748
        nu: 0.3
    Material_002:
        Type: LinearIsotropic
        E: 10000
        nu: 0.25
MaterialTable:
    1: Material_001
    2: Material_002
"""
    parsed = parse_linear_isotropic_materials(text)
    assert parsed.youngs_modulus_mpa == {1: 8748.0, 2: 10000.0}
    assert parsed.poisson_ratio == {1: 0.3, 2: 0.25}
