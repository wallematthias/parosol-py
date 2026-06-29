import pytest

from parosol_py.config_templates import available_config_profiles, read_config_template
from parosol_py.cli import main
from parosol_py.workflow_template import load_workflow_template


EXPECTED_WORKFLOW_PROFILES = {
    "XtremeCTI",
    "XtremeCTII",
    "spine-compression",
    "hip-sideways-fall-left",
    "hip-sideways-fall-right",
    "load_history_3",
    "load_history_6",
}


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


def test_profile_registry_is_workflow_only():
    profiles = available_config_profiles()

    assert set(profiles) == EXPECTED_WORKFLOW_PROFILES
    for removed in (
        "vertebra",
        "ct-spine-compression",
        "ct-hip-sideways-fall",
        "ct-hip-sideways-fall-left",
        "ct-hip-sideways-fall-right",
        "constrained_axial_z",
        "shear_zx",
        "density_power",
        "direct_mechanics_manifest",
    ):
        assert removed not in profiles


def test_workflow_templates_are_available_by_profile_name():
    spine = read_config_template("spine-compression")
    hip = read_config_template("hip-sideways-fall-left")
    xtremectii = read_config_template("XtremeCTII")

    assert "workflow_template:" in spine
    assert "value: -0.68%" in spine
    assert "workflow_template:" in hip
    assert "value: 4.0%" in hip
    assert "E: 8748" in xtremectii
    assert ("tolerance: 1.0e-4" in xtremectii) or ("tolerance: 0.0001" in xtremectii)
    assert "type: constrained_axial" in xtremectii
    assert "strain: -0.01" in xtremectii
    assert "pistoia:" in xtremectii


def test_packaged_workflows_use_npy_references_and_intrusion_schema():
    import zipfile

    from parosol_py.workflow_registry import builtin_profile_path

    for name in ("spine-compression", "hip-sideways-fall-left", "hip-sideways-fall-right"):
        path = builtin_profile_path(name)
        assert path is not None
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            assert "reference/slicer_reference_points.npy" in members
            assert not any(member.lower().endswith(".vtk") for member in members)
            assert not any(member.lower().endswith(".npz") for member in members)

        text = read_config_template(name)
        loaded, _source = load_workflow_template(path)
        replay = loaded["model"]["workflow_replay"]
        density = loaded["materials"]["density"]
        assert loaded["model"]["type"] == "workflow_replay"
        assert replay["enabled"] is True
        assert replay["model_space"] == "reference"
        assert density["bin_material"] is True
        assert density["number_bins"] == 128
        assert density["bin_value"] == "center"
        assert "reference/slicer_reference_points.npy" in text
        assert "method: vtk_icp" in text
        assert "initialization: centroid" in text
        assert "source_landmark_mode: stride" in text
        assert "intrusion_depth_mm:" in text
        assert "protrusion_depth_mm" not in text


def test_spine_workflow_contract_targets_body_registration_and_full_model():
    from parosol_py.workflow_registry import builtin_profile_path

    loaded, _source = load_workflow_template(builtin_profile_path("spine-compression"))
    model = loaded["model"]
    replay = model["workflow_replay"]

    assert model["type"] == "workflow_replay"
    assert model["labels"] == {"body": 20, "process": 48}
    assert model["targets"]["registration"] == "vertebral_body"
    assert model["targets"]["disk_projection"] == "vertebral_body"
    assert model["registration"]["reference_scaling"] == {
        "enabled": True,
        "min_factors": [0.8, 0.8, 0.75],
        "max_factors": [1.2, 1.2, 1.3],
    }
    assert replay["enabled"] is True
    assert replay["model_space"] == "reference"
    assert replay["reference_points"].endswith("reference/slicer_reference_points.npy")
    assert replay["editor_reference_points"].endswith("reference/slicer_reference_points.npy")
    assert loaded["slicer_editor"]["planes"][0]["relative_to"] == "model_bbox"
    assert loaded["slicer_editor"]["planes"][1]["relative_to"] == "model_bbox"


def test_load_history_workflows_remain_boundary_condition_recipes():
    assert "load_history_3" in read_config_template("load_history_3")
    assert "postprocess:" in read_config_template("load_history_3")
    assert "name_suffix: shear_zx" in read_config_template("load_history_3")
    assert "name_suffix: shear_zy" in read_config_template("load_history_3")
    assert "load_history:" in read_config_template("load_history_6")
    assert "bending_x" in read_config_template("load_history_6")
    assert "name_suffix: shear_zx" in read_config_template("load_history_6")
    assert "name_suffix: shear_zy" in read_config_template("load_history_6")
    assert "bending_angle_degrees: -1" in read_config_template("load_history_6")
    assert "neutral_axis_angle_degrees: 0" in read_config_template("load_history_6")
    assert "neutral_axis_angle_degrees: 90" in read_config_template("load_history_6")
    assert "twist_angle_degrees: -1" in read_config_template("load_history_6")


def test_profile_assets_can_be_loaded_from_dynamic_registry():
    from parosol_py.workflow_registry import available_profiles, builtin_profile_path

    assert set(available_profiles()) == EXPECTED_WORKFLOW_PROFILES
    for name in EXPECTED_WORKFLOW_PROFILES:
        path = builtin_profile_path(name)
        assert path is not None
        loaded, source = load_workflow_template(path)
        assert source.name.startswith(name)
        assert isinstance(loaded, dict)


def test_cli_prints_config_template(capsys):
    assert main(["config-template", "--profile", "spine-compression"]) == 0

    out = capsys.readouterr().out
    assert "parosol-py default case settings" in out
    assert "workflow_template:" in out


def test_legacy_modelling_yaml_profiles_are_not_public_templates():
    for profile in (
        "spine-batch",
        "proximal_femur",
        "hip-batch",
        "proximal_femur_sideways_fall",
        "vertebra",
        "ct-spine-compression",
        "ct-hip-sideways-fall",
        "constrained_axial_z",
        "density_power",
    ):
        with pytest.raises(ValueError, match="unknown config template/profile"):
            read_config_template(profile)
