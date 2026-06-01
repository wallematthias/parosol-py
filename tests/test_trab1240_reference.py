import json
import os
from pathlib import Path

import numpy as np
import pytest

from parosol_py.config import run_case_config
from parosol_py.runner import packaged_executable

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "trab1240"


def test_trab1240_fixture_is_connected_component_filtered():
    reference = _load_reference()

    with np.load(FIXTURE_ROOT / "trab1240_labels.npz") as data:
        labels = np.asarray(data["labels"])
        spacing = tuple(float(v) for v in np.asarray(data["spacing_xyz"]).reshape(-1))

    assert labels.shape == (126, 108, 108)
    assert labels.dtype == np.uint8
    assert int(np.count_nonzero(labels)) == reference["source"]["filtered_active_voxels"]
    assert spacing == pytest.approx(tuple(reference["source"]["spacing_xyz"]))
    assert reference["source"]["raw_active_voxels"] > int(np.count_nonzero(labels))


def test_trab1240_reference_sed_fixture_matches_label_grid():
    reference = _load_reference()["reference"]

    with np.load(FIXTURE_ROOT / "trab1240_labels.npz") as labels_data:
        labels = np.asarray(labels_data["labels"])
    with np.load(FIXTURE_ROOT / "reference_sed.npz") as sed_data:
        sed = np.asarray(sed_data["sed_zyx"])

    assert sed.shape == labels.shape
    assert sed.dtype == np.float32
    assert int(np.count_nonzero(sed)) == reference["elements"]
    assert float(np.mean(sed[labels > 0])) == pytest.approx(
        reference["sed_mean"],
        rel=1e-6,
    )


@pytest.mark.skipif(
    os.environ.get("PAROSOL_RUN_REFERENCE_TESTS") != "1",
    reason="set PAROSOL_RUN_REFERENCE_TESTS=1 to run the trab1240 solver reference",
)
def test_trab1240_axial_z_matches_reference(tmp_path: Path):
    executable = packaged_executable()
    if not executable.exists():
        pytest.skip(f"ParOSol executable is not packaged at {executable}")

    reference_doc = _load_reference()
    reference = reference_doc["reference"]
    z_spacing = reference_doc["source"]["spacing_xyz"][2]
    config_path = tmp_path / "trab1240_axial_z.json"
    config_path.write_text(
        json.dumps(
            {
                "case": {"name": "trab1240_axial_z", "work_dir": str(tmp_path / "run")},
                "input": {
                    "image": str(FIXTURE_ROOT / "trab1240_labels.npz"),
                    "image_type": "material_labels",
                    "spacing": [z_spacing, z_spacing, z_spacing],
                },
                "materials": {"file": str(FIXTURE_ROOT / "material_table.yaml")},
                "load_case": {"type": "axial", "axis": "z", "strain": -0.01},
                "solver": {
                    "outputs": ["sed"],
                    "convergence_tolerance": 1e-6,
                    "level": 6,
                    "mpi_processes": int(os.environ.get("PAROSOL_REFERENCE_MPI", "6")),
                },
                "output": {
                    "summary": str(tmp_path / "summary.json"),
                    "export_fields": False,
                },
                "failure": {
                    "criterion": "pistoia",
                    "critical_strain": reference["critical_strain"],
                    "critical_volume_percent": reference["critical_volume_percent"],
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_case_config(config_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert summary["fields"]["sed"]["count"] == reference["elements"]
    assert summary["mechanics"]["top_node_count"] == reference["top_node_count"]
    assert summary["mechanics"]["applied_displacement"]["z"] == pytest.approx(
        reference["applied_displacement_z"],
        rel=5e-5,
    )
    assert summary["mechanics"]["reaction_force"]["z"] == pytest.approx(
        reference["reaction_force_z"],
        rel=5e-4,
    )
    assert summary["mechanics"]["stiffness"]["z"] == pytest.approx(
        reference["stiffness_z"],
        rel=5e-4,
    )
    assert summary["failure"]["ees_at_critical_volume"] == pytest.approx(
        reference["ees_at_critical_volume"],
        rel=5e-4,
    )
    assert summary["failure"]["failure_load"]["z"] == pytest.approx(
        reference["failure_load_z"],
        rel=5e-4,
    )

    reference_sed = _reference_sed_active_values()
    parosol_sed = np.asarray(result.fields["sed"], dtype=np.float64).reshape(-1)
    assert parosol_sed.shape == reference_sed.shape

    difference = parosol_sed - reference_sed
    relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(reference_sed))
    assert relative_l2 < 2e-3
    assert float(np.mean(parosol_sed)) == pytest.approx(
        float(np.mean(reference_sed)),
        rel=5e-4,
    )
    assert float(np.percentile(parosol_sed, 95)) == pytest.approx(
        float(np.percentile(reference_sed, 95)),
        rel=5e-4,
    )
    assert float(np.percentile(parosol_sed, 98)) == pytest.approx(
        float(np.percentile(reference_sed, 98)),
        rel=5e-4,
    )
    assert float(np.max(parosol_sed)) == pytest.approx(
        float(np.max(reference_sed)),
        rel=2e-3,
    )


def _load_reference() -> dict:
    return json.loads((FIXTURE_ROOT / "reference.json").read_text(encoding="utf-8"))


def _reference_sed_active_values() -> np.ndarray:
    with np.load(FIXTURE_ROOT / "trab1240_labels.npz") as labels_data:
        labels_zyx = np.asarray(labels_data["labels"])
    with np.load(FIXTURE_ROOT / "reference_sed.npz") as sed_data:
        sed_zyx = np.asarray(sed_data["sed_zyx"], dtype=np.float64)

    labels_xyz = np.transpose(labels_zyx, (2, 1, 0))
    sed_xyz = np.transpose(sed_zyx, (2, 1, 0))
    active_coords = sorted(
        (tuple(int(v) for v in coord) for coord in np.argwhere(labels_xyz > 0)),
        key=lambda coord: _morton_key(*coord),
    )
    return np.asarray([sed_xyz[coord] for coord in active_coords], dtype=np.float64)


def _morton_key(x: int, y: int, z: int) -> int:
    key = 0
    bit_index = 0
    limit = max(x, y, z)
    while (1 << bit_index) <= limit:
        key |= ((x >> bit_index) & 1) << (3 * bit_index)
        key |= ((y >> bit_index) & 1) << (3 * bit_index + 1)
        key |= ((z >> bit_index) & 1) << (3 * bit_index + 2)
        bit_index += 1
    return key
