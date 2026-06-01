import json
from pathlib import Path

import numpy as np
import pytest

from parosol_py.reports import (
    field_statistics,
    parse_legacy_analysis_file,
    parse_pistoia_file,
    write_summary_json,
)

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
