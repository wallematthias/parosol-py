import numpy as np
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
    assert "bbox_ratio: [1, 1.2, null]" in text


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
    assert "bbox_ratio:" in hip
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
            assert "disk_labels.nii.gz" not in members
            assert "nodesets.nii.gz" not in members

        text = read_config_template(name)
        loaded, _source = load_workflow_template(path)
        replay = loaded["model"]["workflow_replay"]
        density = loaded["materials"]["density"]
        assert loaded["model"]["type"] == "workflow_replay"
        assert replay["enabled"] is True
        assert replay["model_space"] == "reference"
        assert "disk_labels" not in replay
        assert "nodesets" not in replay
        assert density["bin_material"] is True
        assert density["number_bins"] == 128
        assert density["bin_value"] == "center"
        for spec in loaded.get("nodesets", {}).values():
            assert spec["label"] >= 10001
        assert "reference/slicer_reference_points.npy" in text
        assert "method: vtk_icp" in text
        assert "initialization: centroid" in text
        assert "source_landmark_mode: stride" in text
        assert "intrusion_depth_mm:" in text
        assert "protrusion_depth_mm" not in text


def test_reference_space_workflows_use_canonical_icp_without_exposing_option():
    from parosol_py.modeling.workflow_replay import _reference_model_space_icp_direction
    from parosol_py.workflow_registry import builtin_profile_path

    for profile in (
        "spine-compression",
        "hip-sideways-fall-left",
        "hip-sideways-fall-right",
    ):
        loaded, _source = load_workflow_template(builtin_profile_path(profile))

        assert "icp_direction" not in loaded["workflow_template"]["registration"]
        assert "icp_direction" not in loaded["model"]["registration"]
        assert (
            _reference_model_space_icp_direction(loaded["model"]["registration"])
            == "reference_to_sample"
        )
        assert loaded["model"]["workflow_replay"]["model_space"] == "reference"


def test_spine_workflow_contract_targets_body_registration_and_full_model():
    from parosol_py.workflow_registry import builtin_profile_path

    loaded, _source = load_workflow_template(builtin_profile_path("spine-compression"))
    model = loaded["model"]
    replay = model["workflow_replay"]

    assert model["type"] == "workflow_replay"
    assert model["labels"] == {"body": 20, "process": 48}
    assert model["targets"]["registration"] == "body"
    assert model["targets"]["model"] == ["body", "process"]
    assert model["targets"]["disk_projection"] == "body"
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


def test_spine_workflow_cap_geometry_matches_locked_settings():
    from parosol_py.workflow_registry import builtin_profile_path

    loaded, _source = load_workflow_template(builtin_profile_path("spine-compression"))
    disk = loaded["model"]["geometry"]["disk"]
    smooth = loaded["preprocessing"]["smooth"]
    density = loaded["materials"]["density"]

    assert smooth["enabled"] is True
    assert smooth["labels"] is True
    assert smooth["density"] is False
    assert density["input_transform"] == {"equation": "linear", "clamp_min": -31.0}
    assert density["E"]["floor_e_mpa"] == 0.0001
    assert disk["thickness_mm"] == 10.0
    assert disk["intrusion_depth_mm"] == 6.0
    for plane in loaded["slicer_editor"]["planes"]:
        if plane["contact"] == "Material disks":
            assert plane["thickness_mm"] == 10.0
            assert plane["intrusion_depth_mm"] == 6.0


def test_spine_workflow_reference_asset_uses_l4_body_reference_in_slicer_ras():
    import numpy as np

    from parosol_py.workflow_registry import builtin_profile_path

    loaded, _source = load_workflow_template(builtin_profile_path("spine-compression"))
    reference_path = loaded["model"]["workflow_replay"]["reference_points"]
    reference_points = np.load(reference_path)
    reference_mean = reference_points.mean(axis=0)
    reference_extent = np.ptp(reference_points, axis=0)

    assert reference_points.shape == (15839, 3)
    assert reference_mean == pytest.approx(
        [-289.024492336167, 26.366880110037343, 271.640394192391]
    )
    assert reference_extent == pytest.approx(
        [41.162017822265625, 29.8794002532959, 30.0]
    )
    for plane in loaded["slicer_editor"]["planes"]:
        center = plane["center_ras"]
        assert center[0] < 0.0
        assert center[1] > 0.0


def test_hip_workflow_cap_geometry_matches_maintained_sideways_fall_settings():
    from parosol_py.workflow_registry import builtin_profile_path

    expected_sides = {
        "hip-sideways-fall-left": "left",
        "hip-sideways-fall-right": "right",
    }
    for profile, expected_side in expected_sides.items():
        loaded, _source = load_workflow_template(builtin_profile_path(profile))
        cap = loaded["model"]["geometry"]["cap"]
        density = loaded["materials"]["density"]

        assert loaded["model"]["side"] == expected_side
        assert loaded["output"]["fields"] == ["sed"]
        assert loaded["preprocessing"]["smooth"]["enabled"] is False
        assert density["input_transform"] == {
            "equation": "keyak1994_k2hpo4_to_ash",
            "clamp_min": -31.0,
        }
        assert density["E"] == {
            "equation": "power",
            "coefficient": 10500.0,
            "exponent": 2.29,
            "reference_density": 1000.0,
            "floor_e_mpa": 0.0,
        }
        assert loaded["preprocessing"]["bbox_ratio"] == [1.0, 1.3, None]
        assert loaded["preprocessing"]["bbox_crop_from"] == [None, "max", None]
        assert "normalize_aspect_ratio" not in loaded["preprocessing"]
        assert "icp_direction" not in loaded["model"]["registration"]
        assert loaded["nodesets"] == {
            "greater_trochanter_disk": {
                "type": "label_image",
                "label": 10001,
                "selection": "outer_face_nodes",
            },
            "femoral_head_disk": {
                "type": "label_image",
                "label": 10002,
                "selection": "outer_face_nodes",
            },
            "distal_shaft_fixation": {
                "type": "label_image",
                "label": 10003,
                "selection": "interface_nodes",
            },
        }
        assert cap["thickness_mm"] == 10.0
        assert cap["intrusion_depth_mm"] == 6.0
        for plane in loaded["slicer_editor"]["planes"]:
            assert plane["relative_to"] == "model_bbox"
            assert plane["thickness_mm"] == 10.0
            assert plane["intrusion_depth_mm"] == 6.0
            if plane["contact"] == "Material disks":
                assert plane["disk_label"] == loaded["nodesets"][
                    plane["name"].lower().replace(" ", "_")
                ]["label"]
        planes_by_name = {
            plane["name"]: plane for plane in loaded["slicer_editor"]["planes"]
        }
        expected_geometry = {
            "Greater trochanter disk": {
                "bbox_fraction_bounds": {
                    "x": [0.0, 1.0],
                    "y": [-0.1, -0.1],
                    "z": [0.0, 1.0],
                },
                "disk_label": 10001,
            },
            "Femoral head disk": {
                "bbox_fraction_bounds": {
                    "x": [0.0, 1.0],
                    "y": [1.1, 1.1],
                    "z": [0.0, 1.0],
                },
                "disk_label": 10002,
            },
            "Distal shaft fixation": {
                "bbox_fraction_bounds": {
                    "x": [0.0, 1.0],
                    "y": [0.0, 1.0],
                    "z": [-0.1, -0.1],
                },
            },
        }
        for name, expected in expected_geometry.items():
            plane = planes_by_name[name]
            assert plane["bbox_fraction_bounds"] == expected["bbox_fraction_bounds"]
            assert "center_fraction" not in plane
            assert "size_fraction" not in plane
            if "disk_label" in expected:
                assert plane["disk_label"] == expected["disk_label"]
                assert plane["shape"] == "anatomy"
                assert plane["anatomy_constrained"] is True


def test_hip_workflow_reference_assets_use_current_dense_reference():
    from parosol_py.workflow_registry import builtin_profile_path

    expected_lengths = np.asarray([31.46, 42.36, 64.19])
    for profile in ("hip-sideways-fall-left", "hip-sideways-fall-right"):
        loaded, _source = load_workflow_template(builtin_profile_path(profile))
        reference_path = loaded["model"]["workflow_replay"]["reference_points"]
        reference_points = np.load(reference_path)
        centered = reference_points - reference_points.mean(axis=0)
        axis_lengths = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(centered.T)), 0.0)) * 2.0

        assert reference_points.shape[0] > 30000
        assert axis_lengths == pytest.approx(expected_lengths, abs=0.25)


def test_hip_workflow_load_case_uses_anatomical_disk_names():
    from parosol_py.workflow_registry import builtin_profile_path

    for profile in ("hip-sideways-fall-left", "hip-sideways-fall-right"):
        loaded, _source = load_workflow_template(builtin_profile_path(profile))
        load_case = loaded["load_case"]
        planes_by_name = {
            plane["name"]: plane for plane in loaded["slicer_editor"]["planes"]
        }
        loads_by_name = {
            load["nodeset"]: load for load in loaded["slicer_editor"]["loads"]
        }

        assert load_case == {
            "type": "nodeset",
            "fixed": [
                {"nodeset": "greater_trochanter_disk", "dofs": ["y"], "value": 0.0},
                {
                    "nodeset": "distal_shaft_fixation",
                    "dofs": ["x", "z"],
                    "value": 0.0,
                },
            ],
            "prescribed": [
                {
                    "nodeset": "femoral_head_disk",
                    "dof": "y",
                    "value": "4.0%",
                    "units": "%",
                }
            ],
        }
        assert planes_by_name["Greater trochanter disk"]["bc_mode"] == "Fixed"
        assert planes_by_name["Greater trochanter disk"]["fixed_dofs"] == ["y"]
        assert planes_by_name["Femoral head disk"]["bc_mode"] == "Displacement"
        assert planes_by_name["Distal shaft fixation"]["fixed_dofs"] == ["x", "z"]
        assert loads_by_name["Greater trochanter disk"]["fixed_dofs"] == ["y"]
        assert loads_by_name["Femoral head disk"]["mode"] == "Displacement"
        assert loads_by_name["Femoral head disk"]["value"] == 4.0
        assert loads_by_name["Femoral head disk"]["units"] == "%"
        assert loads_by_name["Distal shaft fixation"]["fixed_dofs"] == ["x", "z"]


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
