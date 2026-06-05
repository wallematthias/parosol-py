from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import run_case_config
from .paths import suffix_text

BUNDLE_FORMAT = "parosol-py-bundle"
BUNDLE_VERSION = 1
BUNDLE_CONFIG = "parosol_case.yaml"
BUNDLE_MANIFEST = "manifest.json"
STAGING_DIR = ".parosol_bundle"


def create_bundle(config_path: str | Path, output_path: str | Path) -> Path:
    """Create a portable `.parosol` bundle from a case YAML."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to create ParOSol bundles") from exc

    source_config_path = Path(config_path).expanduser().resolve()
    out = Path(output_path).expanduser().resolve()
    if suffix_text(out) != ".parosol":
        out = out.with_suffix(".parosol")
    out.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"case config must be a mapping: {source_config_path}")

    base_dir = source_config_path.parent
    files: dict[Path, str] = {}
    used_names: set[str] = set()

    portable = _portable_config(config)
    _collect_and_rewrite_paths(
        portable,
        base_dir=base_dir,
        files=files,
        used_names=used_names,
    )
    native_h5 = _existing_native_input(portable, base_dir=base_dir)
    if native_h5 is not None:
        files[native_h5] = _unique_archive_name("native/parosol_input.h5", used_names)

    case_name = str(_section(portable, "case").get("name", source_config_path.stem))
    manifest = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": {"name": case_name},
        "config": BUNDLE_CONFIG,
        "files": [
            {"path": archive_name, "source": str(path)}
            for path, archive_name in sorted(files.items(), key=lambda item: item[1])
        ],
    }

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(BUNDLE_MANIFEST, json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr(BUNDLE_CONFIG, yaml.safe_dump(portable, sort_keys=False))
        archive.writestr("scripts/run.sh", _run_script())
        archive.writestr("scripts/run_slurm.sh", _slurm_script(case_name))
        archive.writestr("README.txt", _bundle_readme(case_name))
        for source, archive_name in files.items():
            archive.write(source, archive_name)
    return out


def inspect_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Return manifest and file inventory for a `.parosol` bundle."""
    bundle = Path(bundle_path).expanduser().resolve()
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        manifest = _read_manifest(archive)
    return {"path": str(bundle), "manifest": manifest, "files": names}


def run_bundle(
    bundle_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
):
    """Extract a `.parosol` bundle, run it, and write full ParOSol-py outputs."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to run ParOSol bundles") from exc

    bundle = Path(bundle_path).expanduser().resolve()
    if output_dir is None:
        out = bundle.with_suffix("")
    else:
        out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    stage = out / STAGING_DIR
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    with zipfile.ZipFile(bundle) as archive:
        manifest = _read_manifest(archive)
        archive.extractall(stage)

    config_path = stage / str(manifest.get("config", BUNDLE_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"bundle config must be a mapping: {config_path}")

    _absolutize_input_paths(config, base_dir=stage)
    _prepare_runtime_outputs(config, output_dir=out, bundle_path=bundle)
    runtime_config = out / BUNDLE_CONFIG
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return run_case_config(runtime_config, dry_run=dry_run, work_dir=out)


def is_bundle_path(path: str | Path) -> bool:
    return suffix_text(Path(path)) == ".parosol"


def _portable_config(config: dict[str, Any]) -> dict[str, Any]:
    import copy

    portable = copy.deepcopy(config)
    case_cfg = _section(portable, "case")
    case_cfg["work_dir"] = "."
    output_cfg = _section(portable, "output")
    output_cfg["result"] = "result.json"
    output_cfg["summary"] = "result.json"
    output_cfg["run_summary"] = "summary.json"
    output_cfg["fields_dir"] = "fields"
    if output_cfg.get("visualization", True) is not False:
        output_cfg["visualization"] = "overview.png"
    execution = portable.setdefault("execution", {})
    if isinstance(execution, dict):
        execution["interface"] = "bundle"
    return portable


def _collect_and_rewrite_paths(
    config: dict[str, Any],
    *,
    base_dir: Path,
    files: dict[Path, str],
    used_names: set[str],
) -> None:
    input_cfg = config.get("input", {})
    if isinstance(input_cfg, dict):
        for key in ("image", "mask", "segmentation", "active_mask", "outer_contour"):
            _rewrite_file_value(input_cfg, key, base_dir=base_dir, files=files, used_names=used_names)

    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict):
        for key in ("density_image", "mask_image", "reference_points"):
            _rewrite_file_value(model_cfg, key, base_dir=base_dir, files=files, used_names=used_names)

    nodesets = config.get("nodesets", {})
    if isinstance(nodesets, dict):
        _rewrite_nested_file_values(nodesets, base_dir=base_dir, files=files, used_names=used_names)


def _rewrite_nested_file_values(
    value: dict[str, Any],
    *,
    base_dir: Path,
    files: dict[Path, str],
    used_names: set[str],
) -> None:
    for key, item in list(value.items()):
        if isinstance(item, dict):
            _rewrite_nested_file_values(item, base_dir=base_dir, files=files, used_names=used_names)
        elif key in {"image", "mask"}:
            _rewrite_file_value(value, key, base_dir=base_dir, files=files, used_names=used_names)


def _rewrite_file_value(
    section: dict[str, Any],
    key: str,
    *,
    base_dir: Path,
    files: dict[Path, str],
    used_names: set[str],
) -> None:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        return
    source = _resolve_existing(value, base_dir=base_dir)
    archive_name = files.get(source)
    if archive_name is None:
        archive_name = _unique_archive_name(f"inputs/{source.name}", used_names)
        files[source] = archive_name
    section[key] = archive_name


def _resolve_existing(value: str, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"bundle input file does not exist: {path}")
    return path


def _unique_archive_name(preferred: str, used_names: set[str]) -> str:
    path = PurePosixPath(preferred)
    candidate = str(path)
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    stem = path.stem
    suffix = "".join(path.suffixes)
    parent = path.parent
    index = 2
    while True:
        candidate = str(parent / f"{stem}_{index}{suffix}")
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _existing_native_input(config: dict[str, Any], *, base_dir: Path) -> Path | None:
    case_cfg = config.get("case", {})
    if isinstance(case_cfg, dict):
        work_dir = case_cfg.get("work_dir")
        if isinstance(work_dir, str):
            candidate = Path(work_dir)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            native = candidate / "parosol_input.h5"
            if native.exists():
                return native.resolve()
    native = base_dir / "parosol_input.h5"
    return native.resolve() if native.exists() else None


def _read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    if BUNDLE_MANIFEST not in archive.namelist():
        raise ValueError(f"bundle is missing {BUNDLE_MANIFEST}")
    manifest = json.loads(archive.read(BUNDLE_MANIFEST))
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format: {manifest.get('format')!r}")
    return manifest


def _absolutize_input_paths(config: dict[str, Any], *, base_dir: Path) -> None:
    for section_name in ("input", "model", "nodesets"):
        section = config.get(section_name)
        if isinstance(section, dict):
            _absolutize_paths_in_mapping(section, base_dir=base_dir)


def _absolutize_paths_in_mapping(value: dict[str, Any], *, base_dir: Path) -> None:
    for key, item in list(value.items()):
        if isinstance(item, dict):
            _absolutize_paths_in_mapping(item, base_dir=base_dir)
        elif key in {
            "image",
            "mask",
            "segmentation",
            "active_mask",
            "outer_contour",
            "density_image",
            "mask_image",
            "reference_points",
        } and isinstance(item, str) and item:
            path = Path(item).expanduser()
            if not path.is_absolute():
                value[key] = str((base_dir / path).resolve())


def _prepare_runtime_outputs(
    config: dict[str, Any],
    *,
    output_dir: Path,
    bundle_path: Path,
) -> None:
    case_cfg = _section(config, "case")
    case_cfg["work_dir"] = str(output_dir)
    output_cfg = _section(config, "output")
    output_cfg["result"] = str(output_dir / "result.json")
    output_cfg["summary"] = output_cfg["result"]
    output_cfg["run_summary"] = str(output_dir / "summary.json")
    output_cfg["fields_dir"] = str(output_dir / "fields")
    if output_cfg.get("visualization", True) is not False:
        output_cfg["visualization"] = str(output_dir / "overview.png")
    execution = config.setdefault("execution", {})
    if isinstance(execution, dict):
        execution.update(
            {
                "interface": "bundle",
                "bundle": str(bundle_path),
                "output_dir": str(output_dir),
            }
        )


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.setdefault(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    return section


def _run_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
bundle="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd).parosol}"
out="${2:-parosol_results}"
parosol run "$bundle" --output "$out"
"""


def _slurm_script(case_name: str) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=parosol_{case_name}
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G

set -euo pipefail
module purge || true
parosol run "${{1:?bundle .parosol file required}}" --output "${{2:-parosol_results}}"
"""


def _bundle_readme(case_name: str) -> str:
    return f"""ParOSol portable run bundle: {case_name}

Run on a workstation or cluster login node with:

  parosol run {case_name}.parosol --output {case_name}_results

The run command unpacks this bundle, launches the native ParOSol solver, and
writes ParOSol-py postprocessing outputs including result.json, summary.json,
fields/sed.nii.gz, and solver logs.
"""
