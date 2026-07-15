from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reference_geometry import ImageGridMetadata
from .reference_visuals import write_visual_report


@dataclass(frozen=True, slots=True)
class ReferenceComparisonBundle:
    reference_json: Path
    replay_json: Path
    equivalence_json: Path
    visual_report: Path
    png_paths: tuple[Path, ...]
    equivalence: dict[str, Any]


def write_reference_comparison_bundle(
    output_dir: str | Path,
    *,
    fixture_name: str,
    grid: ImageGridMetadata,
    reference_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    anatomy_zyx: np.ndarray,
    reference_labels_zyx: np.ndarray,
    replay_labels_zyx: np.ndarray,
    tolerances: dict[str, float] | None = None,
    scalar_reference_zyx: np.ndarray | None = None,
    scalar_replay_zyx: np.ndarray | None = None,
) -> ReferenceComparisonBundle:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    reference = _document(
        fixture_name=fixture_name,
        role="reference",
        grid=grid,
        summary=reference_summary,
    )
    replay = _document(
        fixture_name=fixture_name,
        role="replay",
        grid=grid,
        summary=replay_summary,
    )
    equivalence = build_equivalence_report(
        reference_summary=reference_summary,
        replay_summary=replay_summary,
        reference_labels_zyx=reference_labels_zyx,
        replay_labels_zyx=replay_labels_zyx,
        tolerances=tolerances,
    )
    visual = write_visual_report(
        out,
        fixture_name=fixture_name,
        grid=grid,
        anatomy_zyx=anatomy_zyx,
        reference_labels_zyx=reference_labels_zyx,
        replay_labels_zyx=replay_labels_zyx,
        scalar_reference_zyx=scalar_reference_zyx,
        scalar_replay_zyx=scalar_replay_zyx,
    )

    reference_path = _write_json(out / "reference.json", reference)
    replay_path = _write_json(out / "replay.json", replay)
    equivalence_path = _write_json(out / "equivalence.json", equivalence)
    return ReferenceComparisonBundle(
        reference_json=reference_path,
        replay_json=replay_path,
        equivalence_json=equivalence_path,
        visual_report=visual.html_path,
        png_paths=visual.png_paths,
        equivalence=equivalence,
    )


def build_equivalence_report(
    *,
    reference_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    reference_labels_zyx: np.ndarray,
    replay_labels_zyx: np.ndarray,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    tolerance_values = {
        "label_dice_min": 0.995,
        "node_count_delta_max": 0.0,
    }
    if tolerances:
        tolerance_values.update({str(key): float(value) for key, value in tolerances.items()})
    overlap = label_overlap(reference_labels_zyx, replay_labels_zyx)
    reference_nodes = _nested(reference_summary, "mechanics", "top_node_count")
    replay_nodes = _nested(replay_summary, "mechanics", "top_node_count")
    node_delta = _absolute_delta(reference_nodes, replay_nodes)
    checks = {
        "label_dice": overlap["label_dice"] >= tolerance_values["label_dice_min"],
        "top_node_count": (
            node_delta is None
            or node_delta <= tolerance_values["node_count_delta_max"]
        ),
    }
    return {
        "tolerances": tolerance_values,
        "label_overlap": overlap,
        "label_dice": overlap["label_dice"],
        "node_counts": {
            "reference_top": reference_nodes,
            "replay_top": replay_nodes,
            "absolute_delta": node_delta,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def label_overlap(reference_labels_zyx: np.ndarray, replay_labels_zyx: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference_labels_zyx)
    replay = np.asarray(replay_labels_zyx)
    if reference.shape != replay.shape:
        raise ValueError("label arrays must have matching shapes")
    reference_active = reference != 0
    replay_active = replay != 0
    matching_active = reference_active & replay_active & (reference == replay)
    reference_count = int(np.count_nonzero(reference_active))
    replay_count = int(np.count_nonzero(replay_active))
    denominator = reference_count + replay_count
    dice = 1.0 if denominator == 0 else 2.0 * int(np.count_nonzero(matching_active)) / denominator
    return {
        "shape_zyx": list(reference.shape),
        "reference_nonzero_voxels": reference_count,
        "replay_nonzero_voxels": replay_count,
        "matching_label_voxels": int(np.count_nonzero(matching_active)),
        "different_voxels": int(np.count_nonzero(reference != replay)),
        "label_dice": float(dice),
    }


def _document(
    *,
    fixture_name: str,
    role: str,
    grid: ImageGridMetadata,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "fixture": fixture_name,
        "role": role,
        "grid": grid.to_dict(),
        "summary": summary,
        "transform_chain": summary.get("transform_chain", []),
    }


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _absolute_delta(reference: Any, replay: Any) -> float | None:
    if reference is None or replay is None:
        return None
    return abs(float(replay) - float(reference))
