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

    assert "faim_compat" in profiles
    assert "batch" in profiles
    assert "debug" in profiles
    assert "solver_profile: faim_compat" in read_config_template("faim_compat")


def test_cli_prints_config_template(capsys):
    assert main(["config-template", "--profile", "faim_compat"]) == 0

    out = capsys.readouterr().out
    assert "parosol-py default case settings" in out
    assert "solver_profile: faim_compat" in out
