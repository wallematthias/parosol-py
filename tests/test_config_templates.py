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

    assert "legacy_axial" in profiles
    assert "batch" in profiles
    assert "debug" in profiles
    assert "solver_profile: legacy_axial" in read_config_template("legacy_axial")


def test_cli_prints_config_template(capsys):
    assert main(["config-template", "--profile", "legacy_axial"]) == 0

    out = capsys.readouterr().out
    assert "parosol-py default case settings" in out
    assert "solver_profile: legacy_axial" in out
