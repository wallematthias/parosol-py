from __future__ import annotations

import numpy as np
import pytest

from parosol_py.nodesets import nodes_from_labeled_voxels
from parosol_py.workflow_geometry import (
    generate_disk_and_nodeset_geometry,
    resolve_reference_space_editor,
)


def test_resolve_reference_space_editor_maps_reference_plane_to_sample_space():
    editor = {
        "planes": [
            {
                "name": "Top",
                "reference_space": True,
                "center_ras": [0.0, 0.0, 0.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [10.0, 12.0],
            }
        ]
    }
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=float,
    )
    sample = reference + np.array([5.0, -2.0, 3.0], dtype=float)

    resolved = resolve_reference_space_editor(
        editor,
        reference_points=reference,
        sample_points=sample,
        iterations=10,
        tolerance=1.0e-6,
    )

    plane = resolved["planes"][0]
    assert plane["reference_space"] is False
    assert plane["resolved_from_reference_space"] is True
    assert np.allclose(plane["center_ras"], [5.0, -2.0, 3.0], atol=1.0e-3)


def test_generate_disk_and_nodeset_geometry_builds_intersect_surface_nodeset():
    mask_xyz = np.zeros((8, 8, 8), dtype=bool)
    mask_xyz[2:6, 2:6, 2:6] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    editor = {
        "planes": [
            {
                "name": "Top",
                "axis": "z",
                "normal": "-",
                "contact": "Bone surface",
                "surface_mode": "intersect",
                "bc_mode": "Displacement",
                "direction": "Plane normal",
                "shape": "anatomy",
                "thickness_mm": 0.0,
                "intrusion_depth_mm": 0.0,
                "use_plane_size": True,
                "center_ras": [3.5, 3.5, 5.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [4.0, 4.0],
            }
        ]
    }

    geometry = generate_disk_and_nodeset_geometry(
        editor,
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_labels={"top": 201},
        nodeset_names={"Top": "top"},
    )

    labels = geometry.nodeset_labels_xyz
    assert int(np.count_nonzero(labels == 201)) > 0
    top_nodes = nodes_from_labeled_voxels(
        labels,
        label=201,
        selection="surface_nodes",
        material=material_xyz,
    )
    assert len(top_nodes) > 0


def test_generate_disk_and_nodeset_geometry_builds_projected_cap_and_face_nodeset():
    mask_xyz = np.zeros((8, 8, 8), dtype=bool)
    mask_xyz[2:6, 2:6, 2:6] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    editor = {
        "planes": [
            {
                "name": "Support disk",
                "axis": "z",
                "normal": "-",
                "contact": "Material disks",
                "surface_mode": "project_bounded",
                "bc_mode": "Displacement",
                "direction": "Plane normal",
                "shape": "anatomy",
                "thickness_mm": 2.0,
                "intrusion_depth_mm": 1.0,
                "use_plane_size": True,
                "disk": {"E": 2500.0, "nu": 0.3},
                "center_ras": [3.5, 3.5, 7.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [4.0, 4.0],
            }
        ]
    }

    geometry = generate_disk_and_nodeset_geometry(
        editor,
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_labels={"support_disk": 202},
        nodeset_names={"Support disk": "support_disk"},
        disk_labels={"Support disk": 22},
    )

    assert int(np.count_nonzero(geometry.disk_labels_xyz == 22)) > 0
    assert int(np.count_nonzero(geometry.nodeset_labels_xyz == 202)) > 0
    assert "support_disk" in geometry.node_sets
    assert len(geometry.node_sets["support_disk"]) > 0


def test_projected_anatomy_disk_follows_local_surface_height():
    mask_xyz = np.zeros((8, 8, 8), dtype=bool)
    mask_xyz[2, 3:5, 2:6] = True
    mask_xyz[5, 3:5, 2:4] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    editor = {
        "planes": [
            {
                "name": "Support disk",
                "contact": "Material disks",
                "surface_mode": "project_bounded",
                "shape": "anatomy",
                "thickness_mm": 2.0,
                "intrusion_depth_mm": 0.0,
                "center_ras": [3.5, 3.5, 7.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [6.0, 4.0],
            }
        ]
    }

    geometry = generate_disk_and_nodeset_geometry(
        editor,
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_labels={"support_disk": 202},
        nodeset_names={"Support disk": "support_disk"},
        disk_labels={"Support disk": 22},
    )

    disk = geometry.disk_labels_xyz == 22
    high_column_z = np.argwhere(disk[2, 3:5, :])[:, 1]
    low_column_z = np.argwhere(disk[5, 3:5, :])[:, 1]
    assert high_column_z.size > 0
    assert low_column_z.size > 0
    assert int(low_column_z.min()) < int(high_column_z.min())
    nodeset = geometry.nodeset_labels_xyz == 202
    assert np.count_nonzero(nodeset[2, 3:5, :]) > 0
    assert np.count_nonzero(nodeset[5, 3:5, :]) > 0


def test_projected_anatomy_disk_has_flat_load_facing_nodeset_on_uneven_bone():
    mask_xyz = np.zeros((8, 8, 10), dtype=bool)
    mask_xyz[2:4, 3:5, 5:7] = True
    mask_xyz[5:7, 3:5, 3:5] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    editor = {
        "planes": [
            {
                "name": "Superior disk",
                "contact": "Material disks",
                "surface_mode": "project_bounded",
                "shape": "anatomy",
                "thickness_mm": 3.0,
                "intrusion_depth_mm": 1.0,
                "center_ras": [3.5, 3.5, 9.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [8.0, 4.0],
            }
        ]
    }

    geometry = generate_disk_and_nodeset_geometry(
        editor,
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_labels={"superior_disk": 201},
        nodeset_names={"Superior disk": "superior_disk"},
        disk_labels={"Superior disk": 22},
    )

    disk = geometry.disk_labels_xyz == 22
    assert np.count_nonzero(disk[2:4, 3:5, :]) > 0
    assert np.count_nonzero(disk[5:7, 3:5, :]) > 0

    nodeset = np.argwhere(geometry.nodeset_labels_xyz == 201)
    assert nodeset.size > 0
    assert np.unique(nodeset[:, 2]).tolist() == [8]


def test_material_disk_outer_face_nodes_select_only_load_facing_node_plane():
    mask_xyz = np.zeros((8, 8, 10), dtype=bool)
    mask_xyz[2:4, 3:5, 5:7] = True
    mask_xyz[5:7, 3:5, 3:5] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    editor = {
        "planes": [
            {
                "name": "Superior disk",
                "contact": "Material disks",
                "surface_mode": "project_bounded",
                "shape": "anatomy",
                "thickness_mm": 3.0,
                "intrusion_depth_mm": 1.0,
                "center_ras": [3.5, 3.5, 9.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [8.0, 4.0],
            }
        ]
    }

    geometry = generate_disk_and_nodeset_geometry(
        editor,
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_specs={"superior_disk": {"selection": "outer_face_nodes"}},
        nodeset_labels={"superior_disk": 201},
        nodeset_names={"Superior disk": "superior_disk"},
        disk_labels={"Superior disk": 22},
    )

    nodes = np.asarray(geometry.node_sets["superior_disk"], dtype=int)

    assert np.unique(np.argwhere(geometry.nodeset_labels_xyz == 201)[:, 2]).tolist() == [8]
    assert np.unique(nodes[:, 2]).tolist() == [9]


def test_projected_material_disk_never_labels_bone_voxels():
    mask_xyz = np.zeros((7, 7, 7), dtype=bool)
    mask_xyz[2:5, 2:5, 2:5] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    editor = {
        "planes": [
            {
                "name": "Superior disk",
                "contact": "Material disks",
                "surface_mode": "project_bounded",
                "shape": "anatomy",
                "thickness_mm": 2.0,
                "intrusion_depth_mm": 2.0,
                "center_ras": [3.0, 3.0, 6.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [5.0, 5.0],
            }
        ]
    }

    geometry = generate_disk_and_nodeset_geometry(
        editor,
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_labels={"superior_disk": 201},
        nodeset_names={"Superior disk": "superior_disk"},
        disk_labels={"Superior disk": 22},
    )

    disk = geometry.disk_labels_xyz == 22
    assert np.count_nonzero(disk) > 0
    assert np.count_nonzero(disk & mask_xyz) == 0


def test_larger_intrusion_wraps_more_anatomy_columns_without_entering_bone():
    mask_xyz = np.zeros((9, 9, 9), dtype=bool)
    mask_xyz[3:6, 3:6, 4:6] = True
    mask_xyz[1:3, 3:6, 1:3] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0

    def build(intrusion_depth_mm: float):
        editor = {
            "planes": [
                {
                    "name": "Support disk",
                    "contact": "Material disks",
                    "surface_mode": "project_bounded",
                    "shape": "anatomy",
                    "thickness_mm": 2.0,
                    "intrusion_depth_mm": intrusion_depth_mm,
                    "center_ras": [4.0, 4.0, 8.0],
                    "normal_ras": [0.0, 0.0, -1.0],
                    "u_axis_ras": [1.0, 0.0, 0.0],
                    "v_axis_ras": [0.0, 1.0, 0.0],
                    "size_mm": [6.0, 6.0],
                }
            ]
        }
        return generate_disk_and_nodeset_geometry(
            editor,
            mask_xyz=mask_xyz,
            material_xyz=material_xyz,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            nodeset_labels={"support_disk": 202},
            nodeset_names={"Support disk": "support_disk"},
            disk_labels={"Support disk": 22},
        ).disk_labels_xyz == 22

    shallow = build(0.0)
    wrapped = build(3.0)
    shallow_columns = {(int(x), int(y)) for x, y, _z in np.argwhere(shallow)}
    wrapped_columns = {(int(x), int(y)) for x, y, _z in np.argwhere(wrapped)}

    assert np.count_nonzero(wrapped) > np.count_nonzero(shallow)
    assert len(wrapped_columns) > len(shallow_columns)
    assert np.count_nonzero(shallow & mask_xyz) == 0
    assert np.count_nonzero(wrapped & mask_xyz) == 0


def test_legacy_protrusion_depth_is_not_geometry_input():
    mask_xyz = np.zeros((9, 9, 9), dtype=bool)
    mask_xyz[3:6, 3:6, 4:6] = True
    mask_xyz[1:3, 3:6, 1:3] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0

    def build(extra: dict[str, object]):
        plane = {
            "name": "Support disk",
            "contact": "Material disks",
            "surface_mode": "project_bounded",
            "shape": "anatomy",
            "thickness_mm": 2.0,
            "center_ras": [4.0, 4.0, 8.0],
            "normal_ras": [0.0, 0.0, -1.0],
            "u_axis_ras": [1.0, 0.0, 0.0],
            "v_axis_ras": [0.0, 1.0, 0.0],
            "size_mm": [6.0, 6.0],
        }
        plane.update(extra)
        editor = {"planes": [plane]}
        return generate_disk_and_nodeset_geometry(
            editor,
            mask_xyz=mask_xyz,
            material_xyz=material_xyz,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            nodeset_labels={"support_disk": 202},
            nodeset_names={"Support disk": "support_disk"},
            disk_labels={"Support disk": 22},
        ).disk_labels_xyz == 22

    default = build({})
    intrusion = build({"intrusion_depth_mm": 0.0})
    legacy_protrusion = build({"protrusion_depth_mm": 0.0})

    assert not np.array_equal(default, intrusion)
    assert np.array_equal(legacy_protrusion, default)
    assert np.count_nonzero(legacy_protrusion & mask_xyz) == 0


@pytest.mark.parametrize("invalid_depth", [-1.0, np.nan, np.inf, "invalid"])
def test_invalid_intrusion_depth_falls_back_to_default(invalid_depth):
    mask_xyz = np.zeros((9, 9, 9), dtype=bool)
    mask_xyz[3:6, 3:6, 4:6] = True
    mask_xyz[1:3, 3:6, 1:3] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0

    def build(extra_plane_values: dict):
        plane = {
            "name": "Support disk",
            "contact": "Material disks",
            "surface_mode": "project_bounded",
            "shape": "anatomy",
            "thickness_mm": 2.0,
            "center_ras": [4.0, 4.0, 8.0],
            "normal_ras": [0.0, 0.0, -1.0],
            "u_axis_ras": [1.0, 0.0, 0.0],
            "v_axis_ras": [0.0, 1.0, 0.0],
            "size_mm": [6.0, 6.0],
        }
        plane.update(extra_plane_values)
        return generate_disk_and_nodeset_geometry(
            {"planes": [plane]},
            mask_xyz=mask_xyz,
            material_xyz=material_xyz,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            nodeset_labels={"support_disk": 202},
            nodeset_names={"Support disk": "support_disk"},
            disk_labels={"Support disk": 22},
        ).disk_labels_xyz == 22

    default = build({})
    invalid = build({"intrusion_depth_mm": invalid_depth})

    assert np.array_equal(invalid, default)
    assert np.count_nonzero(invalid & mask_xyz) == 0


def test_axis_aligned_projection_matches_general_projected_surface():
    mask_xyz = np.zeros((8, 8, 8), dtype=bool)
    mask_xyz[2:6, 2:6, 3:6] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    base_plane = {
        "name": "Top",
        "contact": "Bone surface",
        "surface_mode": "project_bounded",
        "shape": "anatomy",
        "thickness_mm": 0.0,
        "intrusion_depth_mm": 0.0,
        "center_ras": [3.5, 3.5, 7.0],
        "normal_ras": [0.0, 0.0, -1.0],
        "u_axis_ras": [1.0, 0.0, 0.0],
        "v_axis_ras": [0.0, 1.0, 0.0],
        "size_mm": [5.0, 5.0],
    }

    geometry = generate_disk_and_nodeset_geometry(
        {"planes": [base_plane]},
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_labels={"top": 201},
        nodeset_names={"Top": "top"},
    )

    nodes = np.argwhere(geometry.nodeset_labels_xyz == 201)
    assert nodes.size > 0
    assert np.unique(nodes[:, 2]).tolist() == [5]


def test_projected_anatomy_disk_ignores_surfaces_beyond_search_depth():
    mask_xyz = np.zeros((8, 8, 12), dtype=bool)
    mask_xyz[2:4, 3:5, 8:10] = True
    mask_xyz[5:7, 3:5, 2:4] = True
    material_xyz = mask_xyz.astype(np.float32) * 1000.0
    editor = {
        "planes": [
            {
                "name": "Support disk",
                "contact": "Material disks",
                "surface_mode": "project_bounded",
                "shape": "anatomy",
                "thickness_mm": 2.0,
                "intrusion_depth_mm": 1.0,
                "center_ras": [3.5, 3.5, 11.0],
                "normal_ras": [0.0, 0.0, -1.0],
                "u_axis_ras": [1.0, 0.0, 0.0],
                "v_axis_ras": [0.0, 1.0, 0.0],
                "size_mm": [8.0, 4.0],
            }
        ]
    }

    geometry = generate_disk_and_nodeset_geometry(
        editor,
        mask_xyz=mask_xyz,
        material_xyz=material_xyz,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        nodeset_labels={"support_disk": 202},
        nodeset_names={"Support disk": "support_disk"},
        disk_labels={"Support disk": 22},
    )

    disk = geometry.disk_labels_xyz == 22
    assert np.count_nonzero(disk[2:4, 3:5, :]) > 0
    assert np.count_nonzero(disk[5:7, 3:5, :]) == 0
