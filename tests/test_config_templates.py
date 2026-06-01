from parosol_py.config_templates import available_config_profiles, read_config_template
from parosol_py.cli import main


def test_default_config_template_documents_material_and_nodeset_workflow():
    text = read_config_template("default")

    assert "image_type: material_labels" in text
    assert "nodesets:" in text
    assert "selection: surface_nodes" in text
    assert "native ParOSol currently uses one global Poisson" in text


def test_profile_override_templates_are_available():
    profiles = available_config_profiles()

    assert "constrained_axial_z" in profiles
    assert "shear_zx" in profiles
    assert "shear_zy" in profiles
    assert "torsion_z" in profiles
    assert "bending_z" in profiles
    assert "batch" in profiles
    assert "debug" in profiles
    assert "type: constrained_axial" in read_config_template("constrained_axial_z")
    assert "direction: x" in read_config_template("shear_zx")
    assert "direction: y" in read_config_template("shear_zy")
    assert "type: torsion" in read_config_template("torsion_z")
    assert "type: bending" in read_config_template("bending_z")


def test_cli_prints_config_template(capsys):
    assert main(["config-template", "--profile", "constrained_axial_z"]) == 0

    out = capsys.readouterr().out
    assert "parosol-py default case settings" in out
    assert "type: constrained_axial" in out
