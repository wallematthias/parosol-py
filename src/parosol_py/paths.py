from __future__ import annotations

import re
from pathlib import Path

_AIM_VERSION_SUFFIX = re.compile(r"(?i)(\.aim);[0-9]+$")


def format_name(path: str | Path) -> str:
    """Return a filename normalized for image-format detection."""

    return _AIM_VERSION_SUFFIX.sub(r"\1", Path(path).name)


def suffix_text(path: str | Path) -> str:
    return "".join(Path(format_name(path)).suffixes).lower()


def image_stem(path: str | Path) -> str:
    name = format_name(path)
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".aim", ".npy", ".npz"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem
