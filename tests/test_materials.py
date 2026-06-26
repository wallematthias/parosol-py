import numpy as np
import pytest

from parosol_py.materials import (
    density_to_material_map,
    labels_to_material_map,
    LinearIsotropicMaterials,
    linear_isotropic_materials_from_config,
    material_to_stiffness_gpa,
    parse_linear_isotropic_materials,
    poisson_ratio_from_spec,
)


def test_material_to_stiffness_gpa_from_mpa():
    material_mpa = np.array([[[0.0, 1000.0], [2000.0, 0.0]]], dtype=np.float64)
    out = material_to_stiffness_gpa(material_mpa, material_unit="MPa")
    assert out.dtype == np.float32
    assert np.allclose(out, np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=np.float32))


def test_material_to_stiffness_gpa_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        material_to_stiffness_gpa(np.array([[[-1.0]]]), material_unit="MPa")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_material_to_stiffness_gpa_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="finite"):
        material_to_stiffness_gpa(np.array([[[value]]]), material_unit="MPa")


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


def test_parse_linear_isotropic_materials_stops_table_at_next_top_level_section():
    text = """MaterialDefinitions:
    Material_001:
        Type: LinearIsotropic
        E: 8748
        nu: 0.3
MaterialTable:
    1: Material_001
OtherSection:
    2: NotAMaterial
"""
    parsed = parse_linear_isotropic_materials(text)
    assert parsed.youngs_modulus_mpa == {1: 8748.0}
    assert parsed.poisson_ratio == {1: 0.3}


def test_labels_to_material_map_preserves_material_specific_poisson_ratio():
    table = parse_linear_isotropic_materials(
        """MaterialDefinitions:
    Trab:
        Type: LinearIsotropic
        E: 500
        nu: 0.25
    Cort:
        Type: LinearIsotropic
        E: 10000
        nu: 0.3
MaterialTable:
    126: Trab
    127: Cort
"""
    )

    mapped = labels_to_material_map(np.array([[[126, 127]]]), table)

    assert mapped.youngs_modulus_mpa.tolist() == [[[500.0, 10000.0]]]
    np.testing.assert_allclose(mapped.poisson_ratio, [[[0.25, 0.3]]])


def test_labels_to_material_map_can_override_material_specific_poisson_ratio():
    table = LinearIsotropicMaterials(
        youngs_modulus_mpa={126: 500.0, 127: 10000.0},
        poisson_ratio={126: 0.25, 127: 0.3},
    )

    mapped = labels_to_material_map(np.array([[[126, 127]]]), table, poisson_ratio=0.3)

    assert mapped.poisson_ratio == 0.3


def test_density_to_material_map_uses_power_equation_and_reduced_poisson_ratio():
    density = np.array([[[0.0, 500.0, 1000.0]]])

    mapped = density_to_material_map(
        density,
        equation="power",
        coefficient=10000.0,
        exponent=2.0,
        reference_density=1000.0,
        poisson_ratio={"equation": "linear", "slope": 0.0001, "intercept": 0.2},
        mask_threshold=0.0,
    )

    assert mapped.youngs_modulus_mpa.tolist() == [[[0.0, 2500.0, 10000.0]]]
    assert mapped.poisson_ratio == pytest.approx(0.275)


def test_density_to_material_map_uses_mulder2007_law_with_floor():
    density = np.array([[[0.0, 500.0, 750.0]]])
    active = np.array([[[True, True, False]]])

    mapped = density_to_material_map(
        density,
        equation="mulder2007",
        active_mask=active,
        floor_e_mpa=2.0,
    )

    assert mapped.youngs_modulus_mpa.tolist() == [[[2.0, 6670.0, 0.0]]]
    assert mapped.metadata["equation"] == "mulder2007"
    assert mapped.metadata["floor_e_mpa"] == pytest.approx(2.0)


def test_density_to_material_map_keeps_zero_density_background_without_active_mask():
    density = np.array([[[0.0, 500.0, 750.0]]])

    mapped = density_to_material_map(
        density,
        equation="mulder2007",
        floor_e_mpa=2.0,
        mask_threshold=0.0,
    )

    assert mapped.youngs_modulus_mpa.tolist() == [[[0.0, 6670.0, 12920.0]]]


def test_density_to_material_map_can_use_ogo_compatible_global_bins():
    density = np.array([[[0.0, 10.0, 20.0, 30.0, 40.0]]])

    mapped = density_to_material_map(
        density,
        equation="linear",
        slope=2.0,
        intercept=1.0,
        mask_threshold=0.0,
        bin_material=True,
        number_bins=2,
    )

    assert mapped.youngs_modulus_mpa.tolist() == [[[0.0, 36.0, 36.0, 66.0, 66.0]]]
    assert mapped.metadata["bin_material"] is True
    assert mapped.metadata["number_bins"] == 2
    np.testing.assert_allclose(mapped.metadata["bin_centers"], [17.5, 32.5])
    np.testing.assert_allclose(mapped.metadata["bin_edges"], [10.0, 25.0, 40.0])


def test_poisson_ratio_from_spec_can_reduce_continuous_field():
    values = np.array([[[0.0, 1.0, 2.0]]])

    nu = poisson_ratio_from_spec(
        {"equation": "linear", "slope": 0.05, "intercept": 0.2, "reduce": "median"},
        values,
        active_mask=values > 0,
    )

    assert nu == pytest.approx(0.275)


def test_linear_isotropic_materials_from_inline_config():
    parsed = linear_isotropic_materials_from_config(
        {
            "definitions": {
                "TrabecularBone": {"Type": "LinearIsotropic", "E": 6829, "nu": 0.3},
                "CorticalBone": {"Type": "LinearIsotropic", "E": 8748, "nu": 0.3},
            },
            "table": {100: "TrabecularBone", 127: "CorticalBone"},
        }
    )

    assert parsed.youngs_modulus_mpa == {100: 6829.0, 127: 8748.0}
    assert parsed.poisson_ratio == {100: 0.3, 127: 0.3}


def test_linear_isotropic_materials_from_label_config():
    parsed = linear_isotropic_materials_from_config(
        {
            "units": "MPa",
            "labels": {
                100: {"name": "trabecular_bone", "E": 6829, "nu": 0.25},
                127: {"name": "cortical_bone", "E": 8748, "nu": 0.3},
            },
        }
    )

    assert parsed.youngs_modulus_mpa == {100: 6829.0, 127: 8748.0}
    assert parsed.poisson_ratio == {100: 0.25, 127: 0.3}
