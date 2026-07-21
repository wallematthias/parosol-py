from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import BuiltModel
from .workflow_replay import build_workflow_replay_model


def build_model(
    model_config: dict[str, Any],
    *,
    base_dir: str | Path,
    material_config: dict[str, Any] | None = None,
    load_case_config: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    custom_preprocessing_config: Any | None = None,
    nodeset_config: dict[str, Any] | None = None,
) -> BuiltModel:
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a table/object")
    base = Path(base_dir).expanduser().resolve()
    materials = {} if material_config is None else material_config
    replay_cfg = model_config.get("workflow_replay", {})
    if isinstance(replay_cfg, dict) and replay_cfg.get("enabled", False):
        return build_workflow_replay_model(
            model_config,
            base_dir=base,
            material_config=materials,
            load_case_config=load_case_config,
            preprocessing_config=preprocessing_config,
            custom_preprocessing_config=custom_preprocessing_config,
            nodeset_config=nodeset_config,
        )
    raise NotImplementedError(
        "modeling requires model.workflow_replay.enabled=true; reusable recipes "
        "must be supplied as .parosol-workflow/.parosol-profile assets"
    )
