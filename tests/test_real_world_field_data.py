from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FIELD_DATA_DIR = Path(__file__).parent / "fixtures" / "fea_reference" / "field_data"
MANIFEST_PATH = FIELD_DATA_DIR / "manifest.json"


def test_real_world_field_data_manifest_is_discoverable():
    manifest = _load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["description"].startswith("Rediscoverable real-world field outputs")
    assert "entries" in manifest
    assert _manifest_text(manifest).find("/Users/") == -1


def test_real_world_field_data_files_exist_and_match_hashes():
    manifest = _load_manifest()

    for entry in manifest["entries"]:
        for field_name in ("material", "sed"):
            field = entry["fields"][field_name]
            path = FIELD_DATA_DIR / field["path"]
            assert path.exists(), path
            assert path.name == f"{field_name}.nii.gz"
            assert path.stat().st_size == field["bytes"]
            assert _sha256(path) == field["sha256"]


def test_real_world_field_data_has_reference_engine_pairs():
    manifest = _load_manifest()
    available = {
        (entry["fixture"], entry["profile"], entry["engine"], entry["case"])
        for entry in manifest["entries"]
    }

    assert (
        "vertebra_l4_mini",
        "spine-compression",
        "parosol-py",
        "vertebra_l4_mini_spine_compression",
    ) in available
    assert (
        "vertebra_l4_mini",
        "spine-compression",
        "ogo",
        "vertebra_l4_mini_spine_compression",
    ) in available
    assert (
        "femur_left_mini",
        "hip-sideways-fall-left",
        "parosol-py",
        "femur_left_mini_hip_sideways_fall",
    ) in available
    assert (
        "femur_left_mini",
        "hip-sideways-fall-left",
        "ogo",
        "femur_left_mini_hip_sideways_fall",
    ) in available
    assert (
        "xtremect_tibia_mini",
        "load_history_3",
        "parosol-py",
        "compression_z",
    ) in available
    assert (
        "xtremect_tibia_mini",
        "load_history_3",
        "faim",
        "compression_z",
    ) in available


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
