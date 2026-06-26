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
                "protrusion_depth_mm": 0.0,
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
                "protrusion_depth_mm": 1.0,
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
                "protrusion_depth_mm": 0.0,
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
    assert int(low_column_z.max()) < int(high_column_z.max())
    nodeset = geometry.nodeset_labels_xyz == 202
    assert np.count_nonzero(nodeset[2, 3:5, :]) > 0
    assert np.count_nonzero(nodeset[5, 3:5, :]) > 0


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
                "protrusion_depth_mm": 1.0,
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
