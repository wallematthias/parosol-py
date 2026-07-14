"""Registry helpers for packaged workflow/profile recipes.

The CLI accepts both "profile" and "workflow" terminology while built-in
recipes live in the same package directory. Keep this module as the single
lookup boundary so shortcut, batch, contract, and baseline code all discover
the same public recipe set.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

WORKFLOW_BUNDLE_SUFFIX = ".parosol-workflow"
PROFILE_BUNDLE_SUFFIX = ".parosol-profile"
RECIPE_SUFFIXES = (WORKFLOW_BUNDLE_SUFFIX, PROFILE_BUNDLE_SUFFIX)

_BUILTIN_RECIPES = resources.files("parosol_py") / "workflows"


def builtin_profile_path(name: str) -> Path | None:
    token = name.strip()
    if not token:
        return None
    for suffix in RECIPE_SUFFIXES:
        candidate = _BUILTIN_RECIPES / f"{token}{suffix}"
        if candidate.is_file():
            return Path(candidate)
    return None


def available_profiles() -> tuple[str, ...]:
    if not _BUILTIN_RECIPES.is_dir():
        return ()
    names: list[str] = []
    for path in _BUILTIN_RECIPES.iterdir():
        if not path.is_file():
            continue
        for suffix in RECIPE_SUFFIXES:
            if path.name.endswith(suffix):
                names.append(path.name.removesuffix(suffix))
                break
    return tuple(sorted(names))


def builtin_workflow_path(name: str) -> Path | None:
    return builtin_profile_path(name)


def available_builtin_workflows() -> tuple[str, ...]:
    return available_profiles()
