from __future__ import annotations

from parosol_py.parity import (
    compare_case_metrics,
    compare_metric,
    default_expected_metrics,
    reference_bundle_assets,
    summarize_metric_comparisons,
)


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


def test_compare_case_metrics_uses_versioned_hip_defaults():
    expected = default_expected_metrics("hip_10001")

    summary = compare_case_metrics(
        "hip_10001",
        observed_metrics=expected,
        tolerance_percent=None,
    )

    assert summary["status"] == "passed"
    assert summary["profile"] == "hip-sideways-fall-left"
    assert {item["name"] for item in summary["comparisons"]} == {
        "reaction_force_N",
        "stiffness_N_per_mm",
    }


def test_reference_bundle_assets_validates_case_files(tmp_path):
    for relative in (
        "hip-sub-RETRO2_10001/input/density.nii.gz",
        "hip-sub-RETRO2_10001/input/segmentation.nii.gz",
        "references/LT_FEMUR_SIDEWAYS_FALL_REF.vtk",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")

    assets = reference_bundle_assets("hip_10001", root=tmp_path)

    assert assets["profile"] == "hip-sideways-fall-left"
    assert assets["assets"]["density_image"].endswith("density.nii.gz")
