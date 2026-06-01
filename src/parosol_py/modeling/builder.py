from __future__ import annotations

from pathlib import Path
from typing import Any

from .femur import build_proximal_femur_model
from .spine import build_spine_compression_model
from .types import BuiltModel


def build_model(
    model_config: dict[str, Any],
    *,
    base_dir: str | Path,
    material_config: dict[str, Any] | None = None,
    load_case_config: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
) -> BuiltModel:
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a table/object")
    kind = str(model_config.get("type", "direct_voxel")).strip().lower()
    base = Path(base_dir).expanduser().resolve()
    materials = {} if material_config is None else material_config
    if kind in {"spine_compression", "vertebra", "vertebra_compression"}:
        return build_spine_compression_model(
            model_config,
            base_dir=base,
            material_config=materials,
            load_case_config=load_case_config,
            preprocessing_config=preprocessing_config,
        )
    if kind in {
        "proximal_femur",
        "proximal_femur_sideways_fall",
        "femur",
        "sideways_fall",
    }:
        return build_proximal_femur_model(
            model_config,
            base_dir=base,
            material_config=materials,
            load_case_config=load_case_config,
            preprocessing_config=preprocessing_config,
        )
    raise NotImplementedError(
        "model.type must be spine_compression/vertebra or proximal_femur"
    )
