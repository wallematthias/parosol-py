from __future__ import annotations

from importlib import resources

from .workflow_template import available_builtin_workflows, builtin_workflow_path, load_workflow_template

_ROOT = resources.files("parosol_py") / "config_templates"
_PROFILE_DISPLAY_NAMES = {
    "xtremecti": "XtremeCTI",
    "xtremectii": "XtremeCTII",
}


def read_config_template(name: str = "default") -> str:
    builtin = builtin_workflow_path(name)
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
    profiles_dir = _ROOT / "profiles"
    yaml_profiles = {
        _PROFILE_DISPLAY_NAMES.get(
            path.name.removesuffix(".yaml"), path.name.removesuffix(".yaml")
        )
        for path in profiles_dir.iterdir()
    }
    return tuple(sorted(yaml_profiles | set(available_builtin_workflows())))


def _template_path(name: str):
    token = name.strip().lower().removesuffix(".yaml")
    if token == "default":
        return _ROOT / "default.yaml"
    path = _ROOT / "profiles" / f"{token}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown config template/profile: {name}")
    return path
