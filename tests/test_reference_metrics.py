from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from parosol_py.reference_metrics import (
    build_core_metric_lock,
    write_core_metric_lock,
)


def test_core_metric_lock_extracts_reported_metrics_and_load_history(tmp_path: Path):
    xtreme_result = tmp_path / "xtreme_result.json"
    xtreme_result.write_text(
        json.dumps(
            {
                "load_case": {
                    "type": "constrained_axial",
                    "axis": "z",
                    "strain": -0.01,
                },
                "mechanics": {
                    "applied_displacement": {"x": None, "y": None, "z": -0.05},
                    "generalized_load": {
                        "name": "force",
                        "component": "z",
                        "units": "N",
                        "value": -100.0,
                    },
                    "generalized_stiffness": {
                        "name": "stiffness",
                        "units": "N/mm",
                        "value": 2000.0,
                    },
                    "reaction_force": {"x": 0.0, "y": 0.0, "z": -100.0},
                    "reference_length_mm": 5.0,
                    "top_node_count": 11,
                    "bottom_node_count": 10,
                },
                "failure": {
                    "criterion": "pistoia",
                    "critical_strain": 0.007,
                    "critical_volume_percent": 2.0,
                    "ees_at_critical_volume": 0.014,
                    "factor": 0.5,
                    "failure_generalized_load": {
                        "name": "force",
                        "units": "N",
                        "value": -50.0,
                    },
                    "failure_load": {"x": None, "y": None, "z": -50.0},
                    "status": "computed",
                },
            }
        ),
        encoding="utf-8",
    )
    load_history_result = tmp_path / "load_history_result.json"
    load_history_result.write_text(
        json.dumps(
            {
                "batch": {"name": "load_history_3", "case_count": 3},
                "postprocess": {
                    "load_history": {
                        "status": "computed",
                        "method": "nnls",
                        "cases": ["compression_z", "shear_zx", "bending_x"],
                        "details": {
                            "load_amplitudes": [4.0, 3.0, 12.0],
                            "input_load_amplitudes": [40.0, 30.0, 24.0],
                            "scaling_factors": [0.1, 0.1, 0.5],
                            "mean": 0.01,
                            "std": 0.002,
                            "residual": 1.5,
                        },
                        "failure": {
                            "factor": 2.5,
                            "ees_at_critical_volume": 0.0028,
                            "critical_strain": 0.007,
                            "critical_volume_percent": 2.0,
                            "status": "computed",
                        },
                        "results": {
                            "estimated_loads": [
                                {
                                    "case": "sample_compression_z",
                                    "load_type": "force",
                                    "units": "N",
                                    "value": -4.0,
                                    "vector": {"x": 0.0, "y": 0.0, "z": -4.0},
                                },
                                {
                                    "case": "sample_shear_zx",
                                    "load_type": "force",
                                    "units": "N",
                                    "value": -3.0,
                                    "vector": {"x": -3.0, "y": 0.0, "z": 0.0},
                                },
                                {
                                    "case": "sample_bending_x",
                                    "load_type": "moment",
                                    "units": "N*mm",
                                    "value": 12.0,
                                    "vector": {"x": 12.0, "y": 0.0, "z": 0.0},
                                },
                            ],
                            "failure_loads": [
                                {
                                    "case": "sample_compression_z",
                                    "load_type": "force",
                                    "units": "N",
                                    "value": -10.0,
                                    "vector": {"x": 0.0, "y": 0.0, "z": -10.0},
                                }
                            ],
                        },
                        "final_rerun": {
                            "status": "computed",
                            "case": {
                                "results": {
                                    "generalized_load": {"value": -125.0, "units": "N"},
                                    "generalized_stiffness": {
                                        "value": 2500.0,
                                        "units": "N/mm",
                                    },
                                    "failure_generalized_load": {
                                        "value": -75.0,
                                        "units": "N",
                                    },
                                    "pistoia_factor": 0.6,
                                }
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "output_root": str(tmp_path),
                "runs": [
                    {
                        "engine": "parosol-py",
                        "fixture": "xtremect_tibia_mini",
                        "profile": "XtremeCTI",
                        "status": "passed",
                        "artifacts": {"result_json": str(xtreme_result)},
                    },
                    {
                        "engine": "ogo",
                        "fixture": "vertebra_l4_mini",
                        "profile": "spine-compression",
                        "status": "passed",
                        "artifacts": {
                            "metrics": {
                                "applied_displacement": "-0.2924",
                                "reaction_force_N": "-6812.0",
                                "stiffness_N_per_mm": "23296.853625171",
                                "characteristic_length_mm": "43.0",
                            }
                        },
                    },
                    {
                        "engine": "parosol-py",
                        "fixture": "xtremect_tibia_mini",
                        "profile": "load_history_3",
                        "status": "passed",
                        "artifacts": {"result_json": str(load_history_result)},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    lock = build_core_metric_lock(index_path)

    records = {
        (record["engine"], record["fixture"], record["profile"]): record
        for record in lock["records"]
    }
    xtreme = records[("parosol-py", "xtremect_tibia_mini", "XtremeCTI")]
    assert xtreme["metric_family"] == "xtremect_pistoia"
    assert xtreme["stiffness_n_per_mm"] == 2000.0
    assert xtreme["pistoia_factor"] == 0.5
    assert xtreme["pistoia_failure_load_n"] == -50.0
    assert xtreme["target_deformation_percent"] == pytest.approx(1.0)

    spine = records[("ogo", "vertebra_l4_mini", "spine-compression")]
    assert spine["metric_family"] == "reference_deformation"
    assert spine["target_deformation_percent"] == pytest.approx(0.68)
    assert spine["reaction_force_n"] == -6812.0
    assert spine["stiffness_n_per_mm"] == pytest.approx(23296.853625171)

    load_history = records[("parosol-py", "xtremect_tibia_mini", "load_history_3")]
    assert load_history["metric_family"] == "load_history"
    assert load_history["estimated_total_force_vector_n"] == {
        "x": -3.0,
        "y": 0.0,
        "z": -4.0,
    }
    assert load_history["estimated_total_force_magnitude_n"] == pytest.approx(5.0)
    assert load_history["estimated_total_moment_vector_nmm"] == {
        "x": 12.0,
        "y": 0.0,
        "z": 0.0,
    }
    assert load_history["estimated_total_moment_magnitude_nmm"] == pytest.approx(12.0)
    assert load_history["load_history_pistoia_factor"] == 2.5
    assert load_history["final_rerun"]["status"] == "computed"
    assert load_history["final_rerun"]["stiffness_n_per_mm"] == 2500.0
    assert load_history["final_rerun"]["pistoia_failure_load_n"] == -75.0

    json_path, csv_path = write_core_metric_lock(index_path, output_dir=tmp_path)

    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["records"] == lock["records"]
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["profile"] for row in rows} == {
        "XtremeCTI",
        "spine-compression",
        "load_history_3",
    }
    assert "estimated_total_force_magnitude_n" in rows[0]
