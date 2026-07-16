from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reference_geometry import ImageGridMetadata


EXPECTED_REFERENCE_FIXTURES = (
    "xtremect_tibia_mini",
    "vertebra_l4_mini",
    "femur_left_mini",
)


@dataclass(frozen=True, slots=True)
class ReferenceFixture:
    name: str
    root: Path
    anatomy: str
    workflows: tuple[str, ...]
    arrays: dict[str, Path]
    grid: ImageGridMetadata
    labels: dict[str, Any]
    provenance: dict[str, Any]
    transform_chain: tuple[dict[str, Any], ...]


def load_reference_fixture(
    name_or_root: str | Path,
    *,
    fixture_root: str | Path | None = None,
) -> ReferenceFixture:
    root = Path(name_or_root)
    if fixture_root is not None and not root.exists():
        root = Path(fixture_root) / str(name_or_root)
    root = root.expanduser().resolve()
    metadata_path = root / "fixture.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = {
        str(key): (root / str(value)).resolve()
        for key, value in dict(data.get("arrays", {})).items()
    }
    return ReferenceFixture(
        name=str(data["name"]),
        root=root,
        anatomy=str(data["anatomy"]),
        workflows=tuple(str(value) for value in data.get("workflows", ())),
        arrays=arrays,
        grid=ImageGridMetadata.from_mapping(data["grid"]),
        labels=dict(data.get("labels", {})),
        provenance=dict(data.get("provenance", {})),
        transform_chain=tuple(dict(value) for value in data.get("transform_chain", ())),
    )


def load_fixture_array(fixture: ReferenceFixture, key: str) -> np.ndarray:
    array_path = fixture.arrays[str(key)]
    with np.load(array_path) as data:
        preferred = f"{key}_zyx"
        if preferred in data:
            return np.asarray(data[preferred])
        if key in data:
            return np.asarray(data[key])
        return np.asarray(data[data.files[0]])


def fixture_array_xyz(fixture: ReferenceFixture, key: str) -> np.ndarray:
    array = load_fixture_array(fixture, key)
    if fixture.grid.array_order.strip().lower() != "zyx":
        raise ValueError("fixture_array_xyz currently requires zyx fixture arrays")
    return np.ascontiguousarray(np.transpose(array, (2, 1, 0)))


def validate_reference_fixture(fixture: ReferenceFixture) -> list[str]:
    issues: list[str] = []
    if fixture.name not in EXPECTED_REFERENCE_FIXTURES:
        issues.append(f"unexpected fixture name {fixture.name}")
    if fixture.grid.coordinate_system != "RAS":
        issues.append("fixture grid must use RAS coordinates")
    if fixture.grid.units != "mm":
        issues.append("fixture grid units must be mm")
    if fixture.grid.array_order != "zyx":
        issues.append("fixture arrays must be stored in zyx order")
    if not fixture.transform_chain:
        issues.append("fixture must record a transform chain")
    for transform in fixture.transform_chain:
        matrix = np.asarray(transform.get("matrix"), dtype=float)
        if matrix.shape != (4, 4):
            issues.append(f"transform {transform.get('name')} must have a 4x4 matrix")
    for key, path in fixture.arrays.items():
        if not path.exists():
            issues.append(f"missing array {key}: {path}")
            continue
        try:
            array = load_fixture_array(fixture, key)
        except Exception as exc:  # pragma: no cover - defensive metadata check
            issues.append(f"could not load array {key}: {exc}")
            continue
        if array.shape != fixture.grid.shape_zyx:
            issues.append(
                f"array {key} shape {array.shape} does not match {fixture.grid.shape_zyx}"
            )
    labels = fixture.arrays.get("labels")
    if labels is not None and labels.exists():
        label_array = load_fixture_array(fixture, "labels")
        nonzero = int(np.count_nonzero(label_array))
        expected = fixture.provenance.get("nonzero_label_voxels")
        if expected is not None and int(expected) != nonzero:
            issues.append(
                f"label nonzero voxel count {nonzero} does not match provenance {expected}"
            )
    banned = "par" + "ity"
    if banned in fixture.root.as_posix().lower():
        issues.append("fixture path uses banned reference wording")
    return issues
