from __future__ import annotations

from parosol_py.parity import compare_metric, summarize_metric_comparisons


def test_compare_metric_reports_absolute_and_percent_error():
    comparison = compare_metric("stiffness_N_per_mm", observed=1505.58, expected=1671.01)

    assert comparison["name"] == "stiffness_N_per_mm"
    assert comparison["observed"] == 1505.58
    assert comparison["expected"] == 1671.01
    assert comparison["absolute_error"] == abs(1505.58 - 1671.01)
    assert comparison["relative_error_percent"] == abs(1505.58 - 1671.01) / 1671.01 * 100.0


def test_summarize_metric_comparisons_flags_tolerance_failures():
    summary = summarize_metric_comparisons(
        [
            compare_metric("force", observed=90.0, expected=100.0, tolerance_percent=15.0),
            compare_metric("stiffness", observed=80.0, expected=100.0, tolerance_percent=5.0),
        ]
    )

    assert summary["status"] == "failed"
    assert summary["failed_metrics"] == ["stiffness"]
