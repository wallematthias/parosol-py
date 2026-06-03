from parosol_py.config_templates import available_config_profiles, read_config_template
from parosol_py.cli import main


def test_default_config_template_documents_material_and_nodeset_workflow():
    text = read_config_template("default")

    assert "image_type: material_labels" in text
    assert "nodesets:" in text
    assert "selection: surface_nodes" in text
    assert "postprocess:" in text
    assert "pistoia:" in text
    assert "load_history:" in text
    assert "Material-specific nu" in text
    assert "values are preserved" in text


def test_profile_override_templates_are_available():
    profiles = available_config_profiles()

    assert "constrained_axial_z" in profiles
    assert "shear_zx" in profiles
    assert "shear_zy" in profiles
    assert "torsion_z" in profiles
    assert "bending_z" in profiles
    assert "smart_bone_compression_z" in profiles
    assert "density_power" in profiles
    assert "direct_mechanics_manifest" in profiles
    assert "load_history_3" in profiles
    assert "load_history_6" in profiles
    assert "XtremeCTI" in profiles
    assert "XtremeCTII" in profiles
    assert "vertebra" in profiles
    assert "proximal_femur" in profiles
    assert "proximal_femur_sideways_fall" in profiles
    assert "standard_mechanics_fields" in profiles
    assert "debug_sets" in profiles
    assert "coarse_preview" in profiles
    assert "progressive_loading_manifest" in profiles
    assert "batch" in profiles
    assert "debug" in profiles
    assert "type: constrained_axial" in read_config_template("constrained_axial_z")
    assert "direction: x" in read_config_template("shear_zx")
    assert "direction: y" in read_config_template("shear_zy")
    assert "strain: -0.01" in read_config_template("shear_zx")
    assert "strain: -0.01" in read_config_template("shear_zy")
    assert "type: torsion" in read_config_template("torsion_z")
    assert "type: bending" in read_config_template("bending_z")
    assert "twist_angle_degrees: -1" in read_config_template("torsion_z")
    assert "bending_angle_degrees: -1" in read_config_template("bending_z")
    assert "mode: smart" in read_config_template("smart_bone_compression_z")
    assert "image_type: density" in read_config_template("density_power")
    assert "compression_x" in read_config_template("direct_mechanics_manifest")
    assert "load_history_3" in read_config_template("load_history_3")
    assert "postprocess:" in read_config_template("load_history_3")
    assert "direction: x, strain: -0.01" in read_config_template("load_history_3")
    assert "direction: y, strain: -0.01" in read_config_template("load_history_3")
    assert "load_history:" in read_config_template("load_history_6")
    assert "bending_x" in read_config_template("load_history_6")
    assert "direction: x, strain: -0.01" in read_config_template("load_history_6")
    assert "direction: y, strain: -0.01" in read_config_template("load_history_6")
    assert "bending_angle_degrees: -1" in read_config_template("load_history_6")
    assert "twist_angle_degrees: -1" in read_config_template("load_history_6")
    xtremecti = read_config_template("XtremeCTI")
    xtremectii = read_config_template("XtremeCTII")
    assert "E: 6829" in xtremecti
    assert "E: 8748" in xtremectii
    assert "type: constrained_axial" in xtremectii
    assert "strain: -0.01" in xtremectii
    assert "pistoia:" in xtremectii
    assert "type: spine_compression" in read_config_template("vertebra")
    assert "body: 20" in read_config_template("vertebra")
    assert "type: proximal_femur" in read_config_template("proximal_femur")
    assert "femur: 2" in read_config_template("proximal_femur")
    assert "type: proximal_femur_sideways_fall" in read_config_template(
        "proximal_femur_sideways_fall"
    )
    assert "type: sideways_fall" in read_config_template(
        "proximal_femur_sideways_fall"
    )
    assert "effective_strain" in read_config_template("standard_mechanics_fields")
    assert "set_formats: [json, vtk]" in read_config_template("debug_sets")
    assert "coarsen:" in read_config_template("coarse_preview")
    assert "progressive_loading" in read_config_template("progressive_loading_manifest")


def test_cli_prints_config_template(capsys):
    assert main(["config-template", "--profile", "constrained_axial_z"]) == 0

    out = capsys.readouterr().out
    assert "parosol-py default case settings" in out
    assert "type: constrained_axial" in out
