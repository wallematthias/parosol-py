from __future__ import annotations

from importlib import resources

_ROOT = resources.files("parosol_py") / "config_templates"


def read_config_template(name: str = "default") -> str:
    path = _template_path(name)
    return path.read_text(encoding="utf-8")


def available_config_profiles() -> tuple[str, ...]:
    profiles_dir = _ROOT / "profiles"
    return tuple(
        sorted(path.name.removesuffix(".yaml") for path in profiles_dir.iterdir())
    )


def _template_path(name: str):
    token = name.strip().lower().removesuffix(".yaml")
    if token == "default":
        return _ROOT / "default.yaml"
    path = _ROOT / "profiles" / f"{token}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown config template/profile: {name}")
    return path
