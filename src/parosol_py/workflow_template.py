from __future__ import annotations

import copy
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .modeling.io import resolve_path
from .paths import suffix_text
from .workflow_registry import (
    WORKFLOW_BUNDLE_SUFFIX,
)


WORKFLOW_FILENAMES = ("workflow.yaml", "workflow.yml", "parosol_slicer_case.yaml")
WORKFLOW_BUNDLE_FORMAT = "parosol-py-workflow"
WORKFLOW_MANIFEST = "manifest.json"


def load_workflow_template(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a reusable ParOSol workflow template folder or workflow file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read workflow templates") from exc

    template_path = Path(path).expanduser().resolve()
    if _is_workflow_bundle(template_path):
        source_path = template_path
        workflow_root = _extract_workflow_bundle(template_path)
        workflow_path = _workflow_path(workflow_root)
    else:
        source_path = template_path
        workflow_path = _workflow_path(template_path)
    loaded = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"workflow template must be a mapping: {workflow_path}")
    return _resolve_template_paths(copy.deepcopy(loaded), workflow_path.parent), source_path


def create_workflow_bundle(source: str | Path, output_path: str | Path) -> Path:
    """Pack a workflow folder/file and its reference files into one archive."""
    source_path = Path(source).expanduser().resolve()
    workflow_path = _workflow_path(source_path)
    base_dir = workflow_path.parent
    out = Path(output_path).expanduser().resolve()
    if not _is_workflow_bundle_name(out):
        out = out.with_suffix(WORKFLOW_BUNDLE_SUFFIX)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = [
        path
        for path in sorted(base_dir.rglob("*"))
        if path.is_file() and path.resolve() != out
    ]
    manifest = {
        "format": WORKFLOW_BUNDLE_FORMAT,
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow_path.relative_to(base_dir).as_posix(),
        "files": [path.relative_to(base_dir).as_posix() for path in files],
    }
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(WORKFLOW_MANIFEST, json.dumps(manifest, indent=2, sort_keys=True))
        for path in files:
            archive.write(path, path.relative_to(base_dir).as_posix())
    return out


def apply_workflow_template(
    template: dict[str, Any],
    *,
    image_path: str | Path,
    mask_path: str | Path | None,
    output_dir: str | Path,
    case_name: str,
    profile: str,
    command: str,
    template_path: str | Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Specialize a workflow template for a new input image and output folder."""
    config = copy.deepcopy(template)
    image = Path(image_path).expanduser().resolve()
    mask = Path(mask_path).expanduser().resolve() if mask_path else None
    out = Path(output_dir).expanduser().resolve()

    case_cfg = _section(config, "case")
    case_cfg["name"] = case_name
    is_batch_workflow = isinstance(config.get("batch"), dict)
    case_cfg["work_dir"] = str(out / case_name) if is_batch_workflow else str(out)

    input_cfg = _section(config, "input")
    input_cfg["image"] = str(image)
    if _supports_image_metadata(image):
        input_cfg.setdefault("spacing", "auto")
        input_cfg.setdefault("origin", "auto")
    else:
        input_cfg.pop("spacing", None)
        input_cfg.pop("origin", None)
    if mask is not None:
        input_cfg["mask"] = str(mask)
    else:
        input_cfg.pop("mask", None)

    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model_cfg["density_image"] = str(image)
        if mask is not None:
            model_cfg["mask_image"] = str(mask)
        else:
            model_cfg.pop("mask_image", None)
        workflow_template_cfg = config.get("workflow_template", {})
        reference_cfg = (
            workflow_template_cfg.get("reference", {})
            if isinstance(workflow_template_cfg, dict)
            else {}
        )
        if (
            isinstance(reference_cfg, dict)
            and reference_cfg.get("disk_labels")
            and reference_cfg.get("nodesets")
        ):
            replay_cfg = model_cfg.setdefault("workflow_replay", {})
            replay_cfg["enabled"] = True
            replay_cfg["disk_labels"] = str(
                resolve_path(reference_cfg["disk_labels"], base_dir=Path(template_path).expanduser().resolve().parent)
            )
            replay_cfg["nodesets"] = str(
                resolve_path(reference_cfg["nodesets"], base_dir=Path(template_path).expanduser().resolve().parent)
            )
            registration_cfg = model_cfg.get("registration", {})
            if isinstance(registration_cfg, dict) and registration_cfg.get("reference_points"):
                replay_cfg["reference_points"] = registration_cfg["reference_points"]
                reference_dir = (
                    Path(str(registration_cfg["reference_points"])).resolve().parent
                )
                for filename in (
                    "slicer_reference_points.npy",
                    "slicer_reference_points.npz",
                ):
                    sibling_editor_reference = reference_dir / filename
                    if sibling_editor_reference.is_file():
                        replay_cfg["editor_reference_points"] = str(
                            sibling_editor_reference
                        )
                        break
            target = registration_cfg.get("target") if isinstance(registration_cfg, dict) else None
            if target:
                replay_cfg["registration_target"] = target
        model_outputs = model_cfg.setdefault("outputs", {})
        if not isinstance(model_outputs, dict):
            raise ValueError("model.outputs must be a mapping in workflow template")
        model_dir = out / "model"
        model_outputs["material_image"] = str(model_dir / "material.nii.gz")
        model_outputs["nodeset_image"] = str(model_dir / "nodesets.nii.gz")
        model_outputs["manifest"] = str(model_dir / "model.json")
        model_outputs["qc_image"] = str(model_dir / "qc.png")

    output_cfg = _section(config, "output")
    output_cfg["result"] = str(out / case_name / "result.json" if is_batch_workflow else out / "result.json")
    output_cfg["summary"] = output_cfg["result"]
    output_cfg["run_summary"] = str(out / case_name / "summary.json" if is_batch_workflow else out / "summary.json")
    output_cfg.setdefault("fields", ["sed"])
    output_cfg["fields_dir"] = str(out / case_name / "fields" if is_batch_workflow else out / "fields")
    output_cfg["visualization"] = str(out / case_name / "overview.png" if is_batch_workflow else out / "overview.png")

    if is_batch_workflow:
        batch_cfg = _section(config, "batch")
        batch_cfg["work_dir"] = str(out)
        batch_cfg["summary"] = str(out / "result.json")

    config["execution"] = {
        "interface": "shortcut-template",
        "command": command,
        "profile": profile,
        "template": str(Path(template_path).expanduser().resolve()),
        "image": str(image),
        "mask": None if mask is None else str(mask),
        "output_dir": str(out),
        "dry_run": bool(dry_run),
    }
    return config


def _workflow_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"workflow template does not exist: {path}")
    for name in WORKFLOW_FILENAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    expected = ", ".join(WORKFLOW_FILENAMES)
    raise ValueError(f"workflow template folder must contain one of: {expected}")


def _is_workflow_bundle(path: Path) -> bool:
    return path.is_file() and _is_workflow_bundle_name(path)


def _is_workflow_bundle_name(path: Path) -> bool:
    return path.name.lower().endswith(WORKFLOW_BUNDLE_SUFFIX)


def _extract_workflow_bundle(path: Path) -> Path:
    stage = Path(tempfile.mkdtemp(prefix="parosol_workflow_"))
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if WORKFLOW_MANIFEST in names:
            manifest = json.loads(archive.read(WORKFLOW_MANIFEST))
            if manifest.get("format") != WORKFLOW_BUNDLE_FORMAT:
                raise ValueError(f"unsupported workflow bundle format: {manifest.get('format')!r}")
        for member in archive.infolist():
            target = (stage / member.filename).resolve()
            if not str(target).startswith(str(stage.resolve())):
                raise ValueError(f"unsafe workflow bundle member: {member.filename}")
            archive.extract(member, stage)
    return stage


def _resolve_template_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    for section_name in ("input", "nodesets"):
        section = config.get(section_name)
        if isinstance(section, dict):
            _resolve_paths_in_mapping(section, base_dir)
    model = config.get("model")
    if isinstance(model, dict):
        _resolve_paths_in_mapping(model, base_dir)
    workflow_template = config.get("workflow_template")
    if isinstance(workflow_template, dict):
        _resolve_paths_in_mapping(workflow_template, base_dir)
    return config


def _resolve_paths_in_mapping(value: dict[str, Any], base_dir: Path) -> None:
    for key, item in list(value.items()):
        if isinstance(item, dict):
            _resolve_paths_in_mapping(item, base_dir)
        elif key in {
            "image",
            "mask",
            "density_image",
            "mask_image",
            "reference_points",
            "editor_reference_points",
            "disk_labels",
            "nodesets",
        } and isinstance(item, str) and item:
            path = Path(item).expanduser()
            if not path.is_absolute():
                value[key] = str((base_dir / path).resolve())


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.setdefault(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping in workflow template")
    return value


def _supports_image_metadata(path: Path) -> bool:
    suffixes = suffix_text(path)
    return suffixes.endswith((".aim", ".mha", ".mhd", ".nii", ".nii.gz", ".npz"))
