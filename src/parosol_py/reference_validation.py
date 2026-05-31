from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reports import parse_pistoia_file


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    aim_path: Path
    analysis_path: Path
    pistoia_path: Path
    critical_volume_percent: float | None
    critical_strain: float | None


def discover_reference_cases(root: str | Path) -> list[ReferenceCase]:
    root_path = Path(root).expanduser().resolve()
    cases: list[ReferenceCase] = []
    for aim_path in sorted(root_path.glob("*.AIM")):
        name = aim_path.stem
        analysis_path = root_path / f"{name}_analysis.txt"
        pistoia_path = root_path / f"{name}_pistoia.txt"
        if not analysis_path.exists() or not pistoia_path.exists():
            continue
        pistoia = parse_pistoia_file(pistoia_path)
        cases.append(
            ReferenceCase(
                name=name,
                aim_path=aim_path,
                analysis_path=analysis_path,
                pistoia_path=pistoia_path,
                critical_volume_percent=pistoia.get("critical_volume_percent"),
                critical_strain=pistoia.get("critical_ees"),
            )
        )
    return cases


def compare_pistoia_summary(
    case: ReferenceCase,
    parosol_summary: dict[str, Any],
    reference_pistoia: dict[str, Any],
) -> dict[str, Any]:
    del case
    pairs = {
        "factor": (
            _at(parosol_summary, "failure", "factor"),
            reference_pistoia.get("factor"),
        ),
        "ees_at_critical_volume": (
            _at(parosol_summary, "failure", "ees_at_critical_volume"),
            reference_pistoia.get("ees_at_critical_volume"),
        ),
        "failure_load_z": (
            _at(parosol_summary, "failure", "failure_load", "z"),
            _at(reference_pistoia, "failure_load", "fz"),
        ),
        "stiffness_z": (
            _at(parosol_summary, "mechanics", "stiffness", "z"),
            _at(reference_pistoia, "axial_stiffness", "z"),
        ),
        "reaction_force_z": (
            _at(parosol_summary, "mechanics", "reaction_force", "z"),
            _at(reference_pistoia, "reaction_force_node_set_1", "fz"),
        ),
    }
    return {
        name: _comparison(parosol_value, reference_value)
        for name, (parosol_value, reference_value) in pairs.items()
    }


def _at(data: dict[str, Any], *keys: str):
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _comparison(parosol_value, reference_value) -> dict[str, float | None]:
    if parosol_value is None or reference_value is None:
        return {
            "parosol": parosol_value,
            "reference": reference_value,
            "absolute_error": None,
            "relative_error": None,
        }
    parosol = float(parosol_value)
    reference = float(reference_value)
    absolute_error = abs(parosol - reference)
    return {
        "parosol": parosol,
        "reference": reference,
        "absolute_error": absolute_error,
        "relative_error": None if reference == 0.0 else absolute_error / abs(reference),
    }
