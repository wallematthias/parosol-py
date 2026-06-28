from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json


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
