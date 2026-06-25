import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from parosol_py.modeling import build_model
from parosol_py.modeling.common import load_density_and_mask, material_from_density
from parosol_py.modeling.io import read_image_zyx
from parosol_py.modeling.alignment import (
    estimate_rigid_icp,
    orient_reference_points,
    read_reference_points,
    surface_points_from_mask,
)
from parosol_py.modeling.common import displacement_from_load_case


def test_model_image_reader_canonicalizes_nifti_direction(tmp_path: Path):
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 2.0, 3.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    path = tmp_path / "flipped_y.nii.gz"
    sitk.WriteImage(image, str(path))

    data, spacing, origin = read_image_zyx(path)

    expected = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS"))
    np.testing.assert_array_equal(data, expected)
    assert spacing == pytest.approx((1.0, 2.0, 3.0))
    assert origin == pytest.approx((10.0, 16.0, 30.0))


def test_model_geometry_numeric_isotropic_spacing_resamples_to_target(
    tmp_path: Path,
):
    density = np.ones((5, 6, 7), dtype=np.float32)
    mask = np.ones_like(density, dtype=np.uint8)
    density_image = sitk.GetImageFromArray(density)
    mask_image = sitk.GetImageFromArray(mask)
    density_image.SetSpacing((0.8, 0.8, 0.8))
    mask_image.SetSpacing((0.8, 0.8, 0.8))
    sitk.WriteImage(density_image, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_image, str(tmp_path / "mask.nii.gz"))

    resampled_density, resampled_mask, spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "geometry": {"isotropic_spacing": 1.0},
        },
        base_dir=tmp_path,
    )

    assert spacing == pytest.approx((1.0, 1.0, 1.0))
    assert resampled_density.shape != density.shape
    assert resampled_mask.shape == resampled_density.shape


def test_model_geometry_spacing_tolerance_skips_unnecessary_resampling(
    tmp_path: Path,
):
    density = np.ones((5, 6, 7), dtype=np.float32)
    mask = np.ones_like(density, dtype=np.uint8)
    density_image = sitk.GetImageFromArray(density)
    mask_image = sitk.GetImageFromArray(mask)
    density_image.SetSpacing((0.6069, 0.6069, 0.6069))
    mask_image.SetSpacing((0.6069, 0.6069, 0.6069))
    sitk.WriteImage(density_image, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(mask_image, str(tmp_path / "mask.nii.gz"))

    resampled_density, _resampled_mask, spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
            "geometry": {
                "resample_spacing": [0.607, 0.607, 0.607],
                "spacing_tolerance_mm": 0.001,
            },
        },
        base_dir=tmp_path,
    )

    assert spacing == pytest.approx((0.6069, 0.6069, 0.6069))
    assert resampled_density.shape == density.shape


def test_model_preprocessing_smooths_density_and_labels_together(tmp_path: Path):
    density = np.zeros((9, 9, 9), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[4, 4, 4] = 100.0
    mask[4, 4, 4] = 2
    image = sitk.GetImageFromArray(density)
    labels = sitk.GetImageFromArray(mask)
    image.SetSpacing((1.0, 1.0, 1.0))
    labels.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(tmp_path / "density.nii.gz"))
    sitk.WriteImage(labels, str(tmp_path / "mask.nii.gz"))

    smoothed_density, smoothed_mask, _spacing, _origin = load_density_and_mask(
        {
            "density_image": "density.nii.gz",
            "mask_image": "mask.nii.gz",
        },
        base_dir=tmp_path,
        preprocessing_config={
            "smooth": {
                "enabled": True,
                "sigma_mm": 1.0,
                "density": True,
                "labels": True,
                "label_threshold": 0.02,
            }
        },
    )

    assert 0.0 < smoothed_density[4, 4, 3] < 100.0
    assert smoothed_density[4, 4, 4] < 100.0
    assert np.count_nonzero(smoothed_mask == 2) > 1


def test_material_from_density_uses_nested_mulder_law_inside_active_contour():
    density = np.array(
        [
            [[0.0, 500.0, 750.0]],
            [[0.0, 500.0, 750.0]],
        ],
        dtype=np.float64,
    )
    active_contour = np.array(
        [
            [[True, True, True]],
            [[False, False, False]],
        ],
        dtype=bool,
    )

    material, nu = material_from_density(
        density,
        active_contour,
        material_config={
            "density": {
                "E": {
                    "equation": "mulder2007",
                    "floor_e_mpa": 2.0,
                },
                "nu": 0.29,
            },
        },
    )

    assert material.tolist() == [
        [[2.0, 6670.0, 12920.0]],
        [[0.0, 0.0, 0.0]],
    ]
    assert nu == pytest.approx(0.29)


def test_spine_compression_model_generates_pmma_disks_and_bc_sets(tmp_path: Path):
    density = np.zeros((8, 7, 6), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:5, 2:4] = 800.0
    mask[2:6, 2:5, 2:4] = 2
    mask[3:5, 3:4, 3:5] = 1
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 2, "process": 1},
            "geometry": {"pmma_thickness_mm": 2, "axis": "z"},
            "outputs": {
                "material_image": "model/material.nii.gz",
                "nodeset_image": "model/nodesets.nii.gz",
                "manifest": "model/model.json",
                "qc_image": "model/qc.png",
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {
                "equation": "linear",
                "slope": 10.0,
                "intercept": 0.0,
                "mask_threshold": 0.0,
            },
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    assert built.material.shape[0] == density.shape[0] + 4
    assert set(built.node_sets) >= {"inferior", "superior"}
    assert len(built.node_sets["inferior"]) > 0
    assert len(built.node_sets["superior"]) > 0
    axis_values = np.asarray(built.node_sets["inferior"])[:, 2]
    assert np.all(axis_values == np.min(axis_values))
    axis_values = np.asarray(built.node_sets["superior"])[:, 2]
    assert np.all(axis_values == np.max(axis_values))
    assert built.boundary_conditions.fixed_coordinates.shape[0] > 0
    assert built.element_sets["inferior_disk"] > 0
    assert built.element_sets["superior_disk"] > 0
    assert built.exported["material_image"].exists()
    assert built.exported["nodeset_image"].exists()
    assert built.exported["qc_image"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    manifest = json.loads(built.exported["manifest"].read_text(encoding="utf-8"))
    assert manifest["model"]["type"] == "spine_compression"
    assert manifest["materials"]["pmma"]["E"] == pytest.approx(2500.0)


def test_spine_pmma_disks_use_flat_outer_faces_not_side_walls(tmp_path: Path):
    density = np.zeros((10, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:8, 2:6, 2:6] = 800.0
    mask[2:8, 2:6, 2:6] = 20
    mask[4:8, 1, 2:6] = 20
    mask[3:6, 3:5, 6] = 48
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 20, "process": 48},
            "geometry": {
                "pmma_thickness_voxels": 2,
                "endplate_depth_voxels": 1,
                "axis": "z",
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    inferior_nodes = np.asarray(built.node_sets["inferior"])
    superior_nodes = np.asarray(built.node_sets["superior"])
    assert np.ptp(inferior_nodes[:, 2]) == 0
    assert np.ptp(superior_nodes[:, 2]) == 0
    assert built.element_sets["inferior_disk"] <= 48
    assert built.element_sets["superior_disk"] <= 48


def test_spine_disk_geometry_accepts_explicit_target_thickness_and_intrusion(
    tmp_path: Path,
):
    density = np.zeros((9, 8, 8), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:7, 2:6, 2:5] = 800.0
    mask[2:7, 2:6, 2:5] = 20
    mask[4:6, 3:5, 5:7] = 48
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 20, "process": 48},
            "geometry": {
                "axis": "z",
                "disk": {
                    "target_label": 20,
                    "thickness_voxels": 1,
                    "intrusion_depth_voxels": 2,
                },
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    assert built.metadata["model"]["disk"] == {
        "target_label": "20",
        "shape": "anatomy",
        "thickness_voxels": 1,
        "intrusion_depth_voxels": 2,
        "method": "projected_cap",
    }
    assert built.element_sets["inferior_disk"] > 0
    assert built.element_sets["superior_disk"] > 0
    assert np.all(built.material[built.postprocess_mask] != 2500.0)


@pytest.mark.parametrize("shape", ["anatomy", "square", "round", "hex"])
def test_spine_disk_geometry_supports_contact_shapes(tmp_path: Path, shape: str):
    density = np.zeros((9, 10, 10), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:7, 2:7, 2:6] = 800.0
    mask[2:7, 2:7, 2:6] = 20
    density[2:5, 7:9, 2:4] = 800.0
    mask[2:5, 7:9, 2:4] = 20
    mask[4:6, 4:6, 6:8] = 48
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 20, "process": 48},
            "geometry": {
                "axis": "z",
                "disk": {
                    "target_label": 20,
                    "shape": shape,
                    "thickness_voxels": 1,
                    "intrusion_depth_voxels": 2,
                },
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    assert built.metadata["model"]["disk"]["shape"] == shape
    assert built.element_sets["inferior_disk"] > 0
    assert built.element_sets["superior_disk"] > 0


def test_proximal_femur_model_generates_caps_and_sideways_fall_sets(tmp_path: Path):
    density = np.zeros((7, 8, 9), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[1:6, 2:6, 2:7] = 700.0
    mask[1:6, 2:6, 2:7] = 2
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "proximal_femur",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "side": "left",
            "geometry": {"pmma_thickness_mm": 2},
            "outputs": {
                "material_image": "model/material.nii.gz",
                "nodeset_image": "model/nodesets.nii.gz",
                "manifest": "model/model.json",
                "qc_image": "model/qc.png",
            },
        },
        base_dir=tmp_path,
        material_config={
            "density": {
                "equation": "linear",
                "slope": 12.0,
                "intercept": 0.0,
                "mask_threshold": 0.0,
            },
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "sideways_fall", "displacement": 1.0},
    )

    assert set(built.node_sets) >= {
        "femoral_head_pmma",
        "greater_trochanter_pmma",
        "distal_femur",
    }
    assert all(len(nodes) > 0 for nodes in built.node_sets.values())
    assert built.element_sets["femoral_head_cap"] > 0
    assert built.element_sets["greater_trochanter_cap"] > 0
    assert built.boundary_conditions.fixed_coordinates.shape[0] > 0
    assert np.max(built.boundary_conditions.fixed_values) > 0.0
    femoral_head = np.asarray(built.node_sets["femoral_head_pmma"])[:, :3]
    greater_trochanter = np.asarray(built.node_sets["greater_trochanter_pmma"])[:, :3]
    assert femoral_head[:, 1].mean() < greater_trochanter[:, 1].mean()
    assert built.metadata["model"]["load_axis"] == "y"
    assert built.metadata["model"]["load_direction"] == "y"
    assert built.metadata["model"]["caps"]["target_label"] == "2"
    assert built.metadata["model"]["caps"]["shape"] == "anatomy"
    assert built.exported["qc_image"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_model_crop_to_bb_uses_declared_model_labels(tmp_path: Path):
    density = np.zeros((20, 20, 20), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:6] = 700.0
    mask[2:6, 2:6, 2:6] = 2
    mask[15:18, 15:18, 15:18] = 99
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    built = build_model(
        {
            "type": "proximal_femur",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"femur": 2},
            "geometry": {"pmma_thickness_voxels": 1, "cap_axis": "y"},
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "sideways_fall", "displacement": 1.0},
        preprocessing_config={
            "crop_to_bb": {"enabled": True, "margin_voxels": 1},
        },
    )

    assert built.material.shape == (6, 8, 6)
    assert built.origin == pytest.approx((1.0, 0.0, 1.0))
    assert built.element_sets["bone"] == 4 * 4 * 4


def test_model_builder_rejects_missing_spine_labels(tmp_path: Path):
    density = np.ones((4, 4, 4), dtype=np.float32)
    mask = np.full_like(density, 2, dtype=np.uint8)
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)

    with pytest.raises(ValueError, match="process"):
        build_model(
            {
                "type": "vertebra",
                "density_image": "density.npy",
                "mask_image": "mask.npy",
                "labels": {"body": 2, "process": 1},
            },
            base_dir=tmp_path,
            material_config={"density": {"equation": "linear"}},
        )


def test_lightweight_icp_estimates_translation_and_reads_vtk_points(tmp_path: Path):
    fixed = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    moving = fixed + np.asarray([2.0, -1.0, 0.5])
    transform = estimate_rigid_icp(
        moving_points=moving,
        fixed_points=fixed,
        iterations=10,
        tolerance=1e-8,
    )

    np.testing.assert_allclose(transform["translation"], [-2.0, 1.0, -0.5], atol=1e-6)

    vtk_path = tmp_path / "points.vtk"
    vtk_path.write_text(
        "\n".join(
            [
                "# vtk DataFile Version 3.0",
                "points",
                "ASCII",
                "DATASET POLYDATA",
                "POINTS 3 float",
                "0 0 0",
                "1 2 3",
                "4 5 6",
            ]
        ),
        encoding="utf-8",
    )
    np.testing.assert_allclose(
        read_reference_points(vtk_path),
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )


def test_reference_points_reader_supports_binary_vtk(tmp_path: Path):
    vtk_path = tmp_path / "points_binary.vtk"
    points = np.asarray(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=">f4"
    )
    vtk_path.write_bytes(
        b"# vtk DataFile Version 5.1\n"
        b"points\n"
        b"BINARY\n"
        b"DATASET POLYDATA\n"
        b"POINTS 3 float\n"
        + points.tobytes()
    )

    np.testing.assert_allclose(read_reference_points(vtk_path), points.astype(float))


def test_reference_points_orientation_reorders_and_flips_axes():
    stored_zyx = np.asarray(
        [
            [0.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [4.0, 30.0, 300.0],
        ]
    )

    oriented = orient_reference_points(
        stored_zyx,
        axis_order="zyx",
        flips="x",
    )

    np.testing.assert_allclose(
        oriented,
        [
            [300.0, 10.0, 0.0],
            [200.0, 20.0, 2.0],
            [100.0, 30.0, 4.0],
        ],
    )


def test_icp_supports_vtk_like_centroid_initialization():
    fixed = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
        ]
    )
    moving = fixed + np.asarray([3.0, -2.0, 1.0])

    transform = estimate_rigid_icp(
        moving_points=moving,
        fixed_points=fixed,
        iterations=10,
        tolerance=1.0e-8,
        start_by_matching_centroids_only=True,
        convergence="absolute",
        distance_mode="rms",
    )

    np.testing.assert_allclose(transform["translation"], [-3.0, 2.0, -1.0], atol=1e-6)


def test_model_percent_displacement_uses_padded_full_height():
    displacement = displacement_from_load_case(
        {"target_displacement_percent": -0.68},
        axis="z",
        dimensions_xyz=(10, 20, 42),
        spacing=(1.0, 1.0, 1.0),
        default=-0.01,
    )

    assert displacement == pytest.approx(-0.2856)


def test_proximal_femur_model_can_use_reference_registration(tmp_path: Path):
    density = np.zeros((8, 9, 10), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:6, 2:7] = 700.0
    mask[2:6, 2:6, 2:7] = 2
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)
    reference_points = surface_points_from_mask(
        mask == 2,
        spacing=(1.0, 1.0, 1.0),
        origin=(5.0, -2.0, 1.0),
    )
    np.savez(tmp_path / "femur_reference.npz", points=reference_points)

    built = build_model(
        {
            "type": "proximal_femur_sideways_fall",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"femur": 2},
            "registration": {
                "enabled": True,
                "reference_points": "femur_reference.npz",
                "max_points": 2000,
                "iterations": 5,
            },
            "geometry": {"pmma_thickness_voxels": 1, "cap_axis": "y"},
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "sideways_fall", "displacement": 1.0},
    )

    assert built.metadata["model"]["registration"]["enabled"] is True
    assert built.metadata["model"]["registration"]["method"] == "lightweight_icp"
    assert built.metadata["model"]["load_axis"] == "y"
    nonzero = np.abs(built.boundary_conditions.fixed_values) > 1.0e-12
    assert np.all(built.boundary_conditions.fixed_values[nonzero] > 0.0)
    assert built.element_sets["femoral_head_cap"] > 0
    assert built.element_sets["greater_trochanter_cap"] > 0
    active = np.argwhere(built.postprocess_mask)
    assert int(active[:, 0].min()) <= 3
    assert int(built.postprocess_mask.shape[0] - active[:, 0].max()) <= 4


def test_spine_model_can_use_lightweight_icp_registration(tmp_path: Path):
    density = np.zeros((8, 7, 6), dtype=np.float32)
    mask = np.zeros_like(density, dtype=np.uint8)
    density[2:6, 2:5, 2:4] = 800.0
    mask[2:6, 2:5, 2:4] = 2
    mask[3:5, 3:4, 3:5] = 1
    np.save(tmp_path / "density.npy", density)
    np.save(tmp_path / "mask.npy", mask)
    reference_points = surface_points_from_mask(
        mask == 2,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    np.savez(tmp_path / "reference_points.npz", points=reference_points)

    built = build_model(
        {
            "type": "spine_compression",
            "density_image": "density.npy",
            "mask_image": "mask.npy",
            "labels": {"body": 2, "process": 1},
            "registration": {
                "enabled": True,
                "reference_points": "reference_points.npz",
                "max_points": 2000,
                "iterations": 5,
            },
            "geometry": {"pmma_thickness_mm": 2, "axis": "z"},
            "outputs": {"manifest": "model/model.json"},
        },
        base_dir=tmp_path,
        material_config={
            "density": {"equation": "linear", "slope": 10.0, "intercept": 0.0},
            "poisson_ratio": 0.3,
            "pmma": {"E": 2500, "nu": 0.3},
        },
        load_case_config={"type": "spine_compression", "displacement": -0.2},
    )

    assert built.metadata["model"]["registration"]["enabled"] is True
    assert built.element_sets["inferior_disk"] > 0
    manifest = json.loads(built.exported["manifest"].read_text(encoding="utf-8"))
    assert manifest["model"]["registration"]["method"] == "lightweight_icp"
