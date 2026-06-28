from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

DEFAULT_REFERENCE_BUNDLE = Path(
    "/Users/matthias.walle/Downloads/n88_ogo_reference_test_bundle_20260625"
)

PARITY_CASE_SPECS: dict[str, dict[str, Any]] = {
    "hip_10001": {
        "profile": "hip-sideways-fall-left",
        "density_image": "hip-sub-RETRO2_10001/input/density.nii.gz",
        "segmentation_image": "hip-sub-RETRO2_10001/input/segmentation.nii.gz",
        "reference_points": "references/LT_FEMUR_SIDEWAYS_FALL_REF.vtk",
        "expected_metrics": {
            "reaction_force_N": 6283.0,
            "stiffness_N_per_mm": 1671.0106382978724,
        },
        "tolerance_percent": 12.0,
    },
    "spine_10001": {
        "profile": "spine-compression",
        "density_image": "spine-sub-001/input/density.nii.gz",
        "segmentation_image": "spine-sub-001/input/segmentation.nii.gz",
        "reference_points": "references/L4_BODY_SPINE_COMPRESSION_REF.vtk",
        "expected_metrics": {},
        "tolerance_percent": 10.0,
    },
}


def compare_metric(
    name: str,
    *,
    observed: float,
    expected: float,
    tolerance_percent: float = 10.0,
) -> dict[str, Any]:
    absolute_error = abs(float(observed) - float(expected))
    relative = None if float(expected) == 0.0 else absolute_error / abs(float(expected)) * 100.0
    passed = relative is not None and relative <= float(tolerance_percent)
    return {
        "name": name,
        "observed": float(observed),
        "expected": float(expected),
        "absolute_error": absolute_error,
        "relative_error_percent": relative,
        "tolerance_percent": float(tolerance_percent),
        "status": "passed" if passed else "failed",
    }


def parity_case_spec(case: str) -> dict[str, Any]:
    try:
        return dict(PARITY_CASE_SPECS[str(case)])
    except KeyError as exc:
        expected = ", ".join(sorted(PARITY_CASE_SPECS))
        raise ValueError(f"unknown parity case {case!r}; expected one of: {expected}") from exc


def reference_bundle_assets(
    case: str,
    *,
    root: str | Path = DEFAULT_REFERENCE_BUNDLE,
) -> dict[str, Any]:
    spec = parity_case_spec(case)
    root_path = Path(root).expanduser().resolve()
    assets = {
        "density_image": root_path / spec["density_image"],
        "segmentation_image": root_path / spec["segmentation_image"],
        "reference_points": root_path / spec["reference_points"],
    }
    missing = [name for name, path in assets.items() if not path.is_file()]
    if missing:
        details = ", ".join(f"{name}={assets[name]}" for name in missing)
        raise FileNotFoundError(f"missing parity reference asset(s): {details}")
    return {
        "root": str(root_path),
        "case": case,
        "profile": spec["profile"],
        "assets": {name: str(path) for name, path in assets.items()},
    }


def default_expected_metrics(case: str) -> dict[str, float]:
    spec = parity_case_spec(case)
    return {
        str(name): float(value)
        for name, value in dict(spec.get("expected_metrics", {})).items()
    }


def default_tolerance_percent(case: str) -> float:
    return float(parity_case_spec(case).get("tolerance_percent", 10.0))


def compare_case_metrics(
    case: str,
    *,
    observed_metrics: dict[str, float],
    expected_metrics: dict[str, float] | None = None,
    tolerance_percent: float | None = None,
) -> dict[str, Any]:
    expected = default_expected_metrics(case) if expected_metrics is None else expected_metrics
    tolerance = default_tolerance_percent(case) if tolerance_percent is None else tolerance_percent
    missing = [name for name in expected if name not in observed_metrics]
    if missing:
        raise ValueError(f"missing observed metric(s) for {case}: {', '.join(missing)}")
    comparisons = [
        compare_metric(
            name,
            observed=float(observed_metrics[name]),
            expected=float(expected_value),
            tolerance_percent=float(tolerance),
        )
        for name, expected_value in expected.items()
    ]
    summary = summarize_metric_comparisons(comparisons)
    summary["case"] = case
    summary["profile"] = parity_case_spec(case)["profile"]
    return summary


def summarize_metric_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [item["name"] for item in comparisons if item.get("status") != "passed"]
    return {
        "status": "passed" if not failed else "failed",
        "failed_metrics": failed,
        "comparisons": comparisons,
    }


def read_first_results_csv_row(path: str | Path) -> dict[str, str]:
    with Path(path).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return dict(row)
    raise ValueError(f"results CSV has no data rows: {path}")


def write_parity_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output
