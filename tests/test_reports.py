import json
from pathlib import Path

import numpy as np
import pytest

from parosol_py.reports import (
    compact_summary_dict,
    field_statistics,
    parse_legacy_analysis_file,
    parse_pistoia_file,
    solve_summary_dict,
    write_results_csv,
    write_summary_json,
)
from parosol_py.api import SolveResult, SolveSummary
from parosol_py.runner import RunSummary

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "reference"


def test_field_statistics_are_json_friendly():
    stats = field_statistics(np.array([1.0, 2.0, 3.0]))

    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["percentiles"]["p50"] == pytest.approx(2.0)


def test_parse_legacy_outputs_to_compact_json(tmp_path: Path):
    analysis = parse_legacy_analysis_file(FIXTURE_ROOT / "SAMPLE_HOM_LS_analysis.txt")
    pistoia = parse_pistoia_file(FIXTURE_ROOT / "SAMPLE_HOM_LS_pistoia.txt")

    assert analysis["model_input"]["number_of_elements"] == 5205150
    assert analysis["strain_energy_density"]["all"]["average"] == pytest.approx(0.3292)
    assert pistoia["factor"] == pytest.approx(0.62004)
    assert pistoia["failure_load"]["fz"] == pytest.approx(-4741.0)

    out = write_summary_json(
        tmp_path / "summary.json", {"analysis": analysis, "pistoia": pistoia}
    )
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["pistoia"]["axial_stiffness"]["z"] == pytest.approx(74985.0)


def test_solve_summary_includes_solution_quality_checks(tmp_path: Path):
    result = SolveResult(
        input_file=tmp_path / "input.h5",
        command=["parosol"],
        fields={},
        summary=SolveSummary(
            dimensions_xyz=(1, 1, 1),
            spacing=(1, 1, 1),
            origin=(0, 0, 0),
            run=RunSummary(iterations=12, relative_residual=1e-5),
        ),
    )

    summary = solve_summary_dict(
        result,
        extra={"quality": {"checks": {"max_relative_residual": 1e-6}}},
    )

    assert summary["quality"]["status"] == "warning"
    assert summary["quality"]["issues"] == ["relative_residual"]


def test_solve_summary_warns_when_sed_is_zero_for_nonzero_solution(tmp_path: Path):
    result = SolveResult(
        input_file=tmp_path / "input.h5",
        command=["parosol"],
        fields={
            "sed": np.zeros((4,), dtype=float),
            "displacements": np.array([[0.0, 0.0, -0.1]], dtype=float),
        },
        summary=SolveSummary(
            dimensions_xyz=(1, 1, 1),
            spacing=(1, 1, 1),
            origin=(0, 0, 0),
            run=RunSummary(iterations=4, relative_residual=1e-5),
        ),
    )

    summary = solve_summary_dict(result)

    assert summary["quality"]["status"] == "warning"
    assert "zero_sed_with_nonzero_solution" in summary["quality"]["issues"]


def test_compact_summary_uses_model_stiffness_as_primary_result():
    summary = {
        "mechanics": {
            "generalized_load": {"name": "force", "value": -8.0, "units": "N"},
            "generalized_stiffness": {
                "name": "stiffness",
                "value": 40.0,
                "units": "N/mm",
            },
            "interface_stiffness": {
                "name": "interface_stiffness",
                "value": 80.0,
                "units": "N/mm",
            },
            "reaction_force": {"x": None, "y": None, "z": -8.0},
            "stiffness": {"x": None, "y": None, "z": 40.0},
        },
        "failure": {
            "failure_load": {"x": None, "y": None, "z": -4.0},
            "failure_generalized_load": {
                "name": "force",
                "value": -4.0,
                "units": "N",
            },
        },
    }

    compact = compact_summary_dict(summary)

    assert compact["results"]["stiffness"]["value"] == pytest.approx(40.0)
    assert compact["results"]["generalized_stiffness"]["value"] == pytest.approx(40.0)
    assert compact["results"]["interface_stiffness"]["value"] == pytest.approx(80.0)
    assert compact["results"]["stiffness_by_axis"]["z"] == pytest.approx(40.0)


def test_compact_summary_includes_nonlinear_diagnostics():
    compact = compact_summary_dict(
        {
            "nonlinear": {
                "material": "nonlinear density",
                "preset": "hip_nonlinear",
                "site": "femoral_neck",
                "convergence_tolerance": 1.0e-4,
                "maximum_plastic_iterations": 150,
                "plastic_iterations": 4,
                "yielded_last": 27,
                "plastic_convergence_last": 5.0e-7,
                "internal_detail": "omitted",
            }
        }
    )

    assert compact["nonlinear"] == {
        "material": "nonlinear density",
        "preset": "hip_nonlinear",
        "site": "femoral_neck",
        "convergence_tolerance": pytest.approx(1.0e-4),
        "maximum_plastic_iterations": 150,
        "plastic_iterations": 4,
        "yielded_last": 27,
        "plastic_convergence_last": pytest.approx(5.0e-7),
    }


def test_solve_summary_merges_nonlinear_metadata_with_solver_diagnostics(tmp_path: Path):
    result = SolveResult(
        input_file=tmp_path / "input.h5",
        command=["parosol"],
        fields={},
        summary=SolveSummary(dimensions_xyz=(1, 1, 1), spacing=(1, 1, 1), origin=(0, 0, 0)),
        diagnostics={
            "nonlinear": {
                "plastic_iterations": 6,
                "yielded_last": 12,
                "plastic_convergence_last": 8.0e-5,
            }
        },
    )

    summary = solve_summary_dict(
        result,
        extra={
            "nonlinear": {
                "material": "nonlinear density",
                "preset": "spine_nonlinear",
                "convergence_tolerance": 1.0e-4,
            }
        },
    )

    assert summary["nonlinear"]["plastic_iterations"] == 6
    assert summary["nonlinear"]["yielded_last"] == 12
    assert summary["nonlinear"]["plastic_convergence_last"] == pytest.approx(8.0e-5)
    assert summary["nonlinear"]["material"] == "nonlinear density"
    assert summary["nonlinear"]["preset"] == "spine_nonlinear"
    assert summary["nonlinear"]["convergence_tolerance"] == pytest.approx(1.0e-4)


def test_write_results_csv_exports_compact_single_row(tmp_path: Path):
    summary = {
        "case": {"name": "sample"},
        "load_case": {"type": "constrained_axial", "axis": "z"},
        "mechanics": {
            "load_direction": "z",
            "reaction_force": {"x": 1.0, "y": 2.0, "z": -10.0},
            "stiffness": {"x": None, "y": None, "z": 50.0},
            "generalized_load": {"name": "force", "value": -10.0, "units": "N"},
            "generalized_stiffness": {
                "name": "stiffness",
                "value": 50.0,
                "units": "N/mm",
            },
        },
        "failure": {
            "criterion": "pistoia",
            "factor": 0.5,
            "critical_strain": 0.007,
            "critical_volume_percent": 2.0,
            "ees_at_critical_volume": 0.01,
            "failure_load": {"x": None, "y": None, "z": -5.0},
            "failure_generalized_load": {"name": "force", "value": -5.0, "units": "N"},
        },
        "solver": {"iterations": 20, "relative_residual": 1.0e-6, "runtime_seconds": 3.5},
    }

    path = write_results_csv(tmp_path / "results.csv", summary)
    text = path.read_text(encoding="utf-8")

    assert "case_name,load_case_type,load_axis,load_direction" in text
    assert "sample,constrained_axial,z,z,1.0,2.0,-10.0" in text
    assert "pistoia" in text


def test_write_results_csv_includes_model_crop_warnings(tmp_path: Path):
    summary = {
        "case": {"name": "hip_sample"},
        "load_case": {"type": "sideways_fall", "axis": "y"},
        "mechanics": {},
        "failure": {},
        "solver": {},
        "model": {
            "type": "hip-sideways-fall-left",
            "shaft_standardization": {
                "cut_mode": "proportional_length",
                "retain_multiplier": 1.35,
                "warnings": [
                    "requested retained length exceeds occupied length along the cut axis; crop collapses to full extent"
                ],
            },
        },
    }

    path = write_results_csv(tmp_path / "results.csv", summary)
    text = path.read_text(encoding="utf-8")

    assert "model_warning_count" in text
    assert "model_warnings" in text
    assert "crop collapses to full extent" in text
