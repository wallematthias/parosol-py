from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fea_reference"
LOCK_JSON = FIXTURE_DIR / "core_metric_lock.json"
LOCK_CSV = FIXTURE_DIR / "core_metric_lock.csv"


def test_real_world_core_metric_lock_is_complete_and_discoverable():
    lock = _load_lock()

    assert LOCK_CSV.exists()
    assert lock["schema_version"] == 1
    assert lock["description"].startswith("Locked real-world core metrics")

    records = _records_by_key(lock)
    assert set(records) == {
        ("parosol-py", "xtremect_tibia_mini", "XtremeCTI"),
        ("parosol-py", "xtremect_tibia_mini", "XtremeCTII"),
        ("parosol-py", "xtremect_tibia_mini", "load_history_3"),
        ("parosol-py", "xtremect_tibia_mini", "load_history_6"),
        ("parosol-py", "vertebra_l4_mini", "spine-compression"),
        ("parosol-py", "femur_left_mini", "hip-sideways-fall-left"),
        ("parosol-py", "femur_left_mini", "hip-compression-manual"),
        ("ogo", "vertebra_l4_mini", "spine-compression"),
        ("ogo", "femur_left_mini", "hip-sideways-fall-left"),
        ("faim", "xtremect_tibia_mini", "XtremeCTI"),
        ("faim", "xtremect_tibia_mini", "XtremeCTII"),
        ("faim", "xtremect_tibia_mini", "load_history_3"),
    }

    csv_keys = {
        (row["engine"], row["fixture"], row["profile"]) for row in _load_csv_rows()
    }
    assert csv_keys == set(records)
    assert _load_csv_rows() == _flattened_csv_rows(lock["records"])


def test_real_world_canonical_metric_values_are_locked():
    records = _records_by_key(_load_lock())

    assert records[
        ("parosol-py", "xtremect_tibia_mini", "XtremeCTI")
    ]["stiffness_n_per_mm"] == pytest.approx(12832.691760088526)
    assert records[
        ("parosol-py", "xtremect_tibia_mini", "XtremeCTI")
    ]["pistoia_failure_load_n"] == pytest.approx(-328.0471078629065)
    assert records[
        ("parosol-py", "xtremect_tibia_mini", "XtremeCTII")
    ]["stiffness_n_per_mm"] == pytest.approx(16438.755080140352)
    assert records[
        ("parosol-py", "xtremect_tibia_mini", "XtremeCTII")
    ]["pistoia_failure_load_n"] == pytest.approx(-420.2242073163756)

    assert records[
        ("parosol-py", "vertebra_l4_mini", "spine-compression")
    ]["stiffness_n_per_mm"] == pytest.approx(23358.63915762964)
    assert records[
        ("parosol-py", "femur_left_mini", "hip-sideways-fall-left")
    ]["stiffness_n_per_mm"] == pytest.approx(1614.9917086520325)
    assert records[
        ("parosol-py", "femur_left_mini", "hip-compression-manual")
    ]["stiffness_n_per_mm"] == pytest.approx(1633.8266426868377)
    assert records[
        ("parosol-py", "femur_left_mini", "hip-compression-manual")
    ]["applied_displacement_mm"] == pytest.approx(-1.0)
    assert records[
        ("parosol-py", "vertebra_l4_mini", "spine-compression")
    ]["target_deformation_percent"] == pytest.approx(0.6800000057664028)
    assert records[
        ("parosol-py", "femur_left_mini", "hip-sideways-fall-left")
    ]["target_deformation_percent"] == pytest.approx(3.9999999783255835)

    load_history = records[("parosol-py", "xtremect_tibia_mini", "load_history_3")]
    assert load_history["load_history_input_load_amplitudes"] == pytest.approx(
        [798.4077347978715, 141.09941825856413, 135.69820602395055]
    )
    assert load_history["estimated_total_force_magnitude_n"] == pytest.approx(
        142.21410823547805
    )


def test_real_world_reference_engine_deltas_stay_small():
    records = _records_by_key(_load_lock())

    _assert_relative_delta(
        records[("parosol-py", "vertebra_l4_mini", "spine-compression")],
        records[("ogo", "vertebra_l4_mini", "spine-compression")],
        "stiffness_n_per_mm",
        max_relative_delta=0.003,
    )
    _assert_relative_delta(
        records[("parosol-py", "femur_left_mini", "hip-sideways-fall-left")],
        records[("ogo", "femur_left_mini", "hip-sideways-fall-left")],
        "stiffness_n_per_mm",
        max_relative_delta=0.001,
    )
    _assert_relative_delta(
        records[("parosol-py", "xtremect_tibia_mini", "XtremeCTI")],
        records[("faim", "xtremect_tibia_mini", "XtremeCTI")],
        "stiffness_n_per_mm",
        max_relative_delta=0.001,
    )
    _assert_relative_delta(
        records[("parosol-py", "xtremect_tibia_mini", "XtremeCTII")],
        records[("faim", "xtremect_tibia_mini", "XtremeCTII")],
        "stiffness_n_per_mm",
        max_relative_delta=0.001,
    )

    parosol_loads = records[
        ("parosol-py", "xtremect_tibia_mini", "load_history_3")
    ]["load_history_input_load_amplitudes"]
    faim_loads = records[
        ("faim", "xtremect_tibia_mini", "load_history_3")
    ]["load_history_input_load_amplitudes"]
    for parosol_value, reference_value in zip(parosol_loads, faim_loads, strict=True):
        assert abs(parosol_value - reference_value) / abs(reference_value) <= 0.001


def _load_lock() -> dict[str, Any]:
    return json.loads(LOCK_JSON.read_text(encoding="utf-8"))


def _load_csv_rows() -> list[dict[str, str]]:
    with LOCK_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _records_by_key(lock: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (record["engine"], record["fixture"], record["profile"]): record
        for record in lock["records"]
    }


def _assert_relative_delta(
    actual: dict[str, Any],
    reference: dict[str, Any],
    field: str,
    *,
    max_relative_delta: float,
) -> None:
    actual_value = float(actual[field])
    reference_value = float(reference[field])
    relative_delta = abs(actual_value - reference_value) / abs(reference_value)
    assert relative_delta <= max_relative_delta


def _flattened_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = [_flatten(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    return [
        {field: _csv_cell(row.get(field)) for field in fieldnames}
        for row in rows
    ]


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            row.update(_flatten(value, name))
        else:
            row[name] = value
    return row


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, sort_keys=True)
    return str(value)
