from pathlib import Path

import pytest

from parosol_py.reference_validation import (
    ReferenceCase,
    compare_pistoia_summary,
    discover_reference_cases,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "reference"


def test_discover_reference_cases_finds_local_reference_set():
    cases = discover_reference_cases(FIXTURE_ROOT)

    names = {case.name for case in cases}
    assert "SAMPLE_HOM_LS" in names


def test_compare_pistoia_summary_reports_relative_errors():
    case = ReferenceCase(
        name="sample",
        aim_path=Path("sample.AIM"),
        analysis_path=Path("sample_analysis.txt"),
        pistoia_path=Path("sample_pistoia.txt"),
        critical_volume_percent=2.0,
        critical_strain=0.007,
    )
    parosol = {
        "failure": {
            "factor": 0.5,
            "ees_at_critical_volume": 0.014,
            "failure_load": {"z": -100.0},
        },
        "mechanics": {"stiffness": {"z": 1000.0}, "reaction_force": {"z": -200.0}},
    }
    reference = {
        "factor": 0.4,
        "ees_at_critical_volume": 0.01,
        "failure_load": {"fz": -80.0},
        "axial_stiffness": {"z": 800.0},
        "reaction_force_node_set_1": {"fz": -160.0},
    }

    comparison = compare_pistoia_summary(case, parosol, reference)

    assert comparison["factor"]["absolute_error"] == pytest.approx(0.1)
    assert comparison["failure_load_z"]["relative_error"] == pytest.approx(0.25)
