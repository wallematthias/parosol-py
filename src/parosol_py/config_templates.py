from __future__ import annotations

from importlib import resources

from .workflow_registry import available_profiles, builtin_profile_path
from .workflow_template import load_workflow_template

_ROOT = resources.files("parosol_py") / "config_templates"


def read_config_template(name: str = "default") -> str:
    builtin = builtin_profile_path(name)
    if builtin is not None:
        template, _ = load_workflow_template(builtin)
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read workflow templates") from exc
        return yaml.safe_dump(template, sort_keys=False)
    path = _template_path(name)
    return path.read_text(encoding="utf-8")


def available_config_profiles() -> tuple[str, ...]:
    return available_profiles()


def _template_path(name: str):
    token = name.strip().lower().removesuffix(".yaml")
    if token == "default":
        return _ROOT / "default.yaml"
    raise ValueError(f"unknown config template/profile: {name}")
