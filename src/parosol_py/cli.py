from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .batch import run_batch_config
from .bundle import create_bundle, inspect_bundle, is_bundle_path, run_bundle
from .config import run_case_config
from .config_templates import available_config_profiles, read_config_template
from .load_history import estimate_load_history_from_files
from .paths import image_stem, suffix_text
from .reports import parse_legacy_analysis_file, parse_pistoia_file, write_summary_json
from .workflow_template import (
    apply_workflow_template,
    builtin_workflow_path,
    load_workflow_template,
)

_COMMANDS = {
    "run",
    "batch",
    "bundle",
    "load-history",
    "summarize-legacy",
    "config-template",
}


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = (
        _build_shortcut_parser()
        if tokens and not tokens[0].startswith("-") and tokens[0] not in _COMMANDS
        else _build_parser()
    )
    args = parser.parse_args(tokens)
    args._argv = tokens
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"parosol: error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parosol",
        description=(
            "Run workflow/profile-driven ParOSol cases. Shortcut form: "
            "parosol IMAGE --profile PROFILE [--mask MASK] [--output OUT]."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run a ParOSol case config or portable .parosol bundle"
    )
    run_parser.add_argument(
        "config", help="Path to a .yaml, .toml, .json case config, or .parosol bundle"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write inputs/summary without launching ParOSol",
    )
    run_parser.add_argument("--work-dir", help="Override the configured work directory")
    run_parser.add_argument(
        "-o",
        "--output",
        help="Output directory for .parosol bundle runs",
    )
    run_parser.set_defaults(func=_run)

    batch_parser = subparsers.add_parser("batch", help="Run a ParOSol batch config")
    batch_parser.add_argument(
        "target", help="Path to a batch config or an input image folder"
    )
    batch_parser.add_argument(
        "--profile",
        choices=(*available_config_profiles(), "interactive_custom"),
        help="Built-in workflow/profile recipe to apply when TARGET is a folder",
    )
    batch_parser.add_argument(
        "--template",
        help=(
            "Reusable workflow/profile template folder/file or .parosol-workflow to apply when TARGET is a folder. "
            "Use with --profile interactive_custom for Slicer-authored workflows."
        ),
    )
    batch_parser.add_argument(
        "-o",
        "--output",
        help="Output directory for folder batches. Defaults to TARGET/parosol_batch.",
    )
    batch_parser.add_argument(
        "--pattern",
        action="append",
        help="Input glob for folder batches. Can be repeated. Defaults to image files.",
    )
    batch_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover input files in folder batch mode.",
    )
    batch_parser.add_argument(
        "--mask-dir",
        help="Folder containing masks for model profiles in folder batch mode.",
    )
    batch_parser.add_argument(
        "--mask-pattern",
        default="{stem}_SEG.nii.gz",
        help="Mask filename pattern for folder batches; supports {stem} and {name}.",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write inputs/summaries without launching ParOSol",
    )
    batch_parser.add_argument(
        "--work-dir", help="Override the configured batch directory"
    )
    batch_parser.set_defaults(func=_batch)

    bundle_parser = subparsers.add_parser(
        "bundle", help="Create and inspect portable .parosol bundles"
    )
    bundle_subparsers = bundle_parser.add_subparsers(
        dest="bundle_command", required=True
    )
    bundle_create_parser = bundle_subparsers.add_parser(
        "create", help="Create a portable .parosol bundle from a case config"
    )
    bundle_create_parser.add_argument("config", help="Path to a case config")
    bundle_create_parser.add_argument(
        "-o", "--output", required=True, help="Output .parosol bundle path"
    )
    bundle_create_parser.set_defaults(func=_bundle_create)

    bundle_inspect_parser = bundle_subparsers.add_parser(
        "inspect", help="Inspect a portable .parosol bundle"
    )
    bundle_inspect_parser.add_argument("bundle", help="Path to a .parosol bundle")
    bundle_inspect_parser.set_defaults(func=_bundle_inspect)

    history_parser = subparsers.add_parser(
        "load-history",
        help="Estimate load history from solved SED fields",
    )
    history_parser.add_argument("load_cases", nargs="+", help="SED image/array paths")
    history_parser.add_argument(
        "--bone-mask", required=True, help="Bone mask image/array"
    )
    history_parser.add_argument(
        "-o", "--output", help="Output load-history image/array"
    )
    history_parser.add_argument("--summary", required=True, help="Output summary JSON")
    history_parser.add_argument("--target-average", type=float, default=0.02)
    history_parser.add_argument("--cutoff-percentile", type=float, default=95.0)
    history_parser.set_defaults(func=_load_history)

    summary_parser = subparsers.add_parser(
        "summarize-legacy",
        help="Convert old legacy solver analysis/Pistoia text outputs to compact JSON",
    )
    summary_parser.add_argument("--analysis", help="Path to old *_analysis.txt")
    summary_parser.add_argument("--pistoia", help="Path to old *_pistoia.txt")
    summary_parser.add_argument(
        "-o", "--output", required=True, help="Output JSON path"
    )
    summary_parser.set_defaults(func=_summarize_legacy)

    template_parser = subparsers.add_parser(
        "config-template",
        help="Print the commented default case config and optional workflow/profile recipe",
    )
    template_parser.add_argument(
        "--profile",
        choices=available_config_profiles(),
        help="Append a built-in workflow/profile recipe",
    )
    template_parser.set_defaults(func=_config_template)
    return parser


def _build_shortcut_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parosol",
        description="Run one image with a ParOSol workflow/profile recipe",
    )
    parser.add_argument("image", help="Input material/density image")
    parser.add_argument(
        "--profile",
        required=True,
        choices=(*available_config_profiles(), "interactive_custom"),
        help="Built-in workflow/profile recipe to apply",
    )
    parser.add_argument(
        "--mask",
        help="Optional segmentation/nodeset mask. Workflow recipes use this as the model mask.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory. Defaults to '<input>_parosol' next to the input image.",
    )
    parser.add_argument("--name", help="Case name. Defaults to the input file stem.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write inputs/summary without launching ParOSol",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        help="Model side override for custom templates that support it.",
    )
    parser.add_argument(
        "--reference-points",
        help="Reference point cloud for lightweight ICP registration.",
    )
    parser.add_argument(
        "--template",
        help=(
            "Reusable workflow/profile template folder/file or .parosol-workflow. "
            "SlicerParOSol workflows contain workflow.yaml plus optional reference files."
        ),
    )
    parser.set_defaults(func=_shortcut)
    return parser


def _run(args: argparse.Namespace) -> int:
    if is_bundle_path(args.config):
        result = run_bundle(
            args.config,
            output_dir=args.output or args.work_dir,
            dry_run=bool(args.dry_run),
        )
    else:
        if args.output:
            raise ValueError("--output is only supported when running a .parosol bundle")
        result = run_case_config(
            args.config, dry_run=True if args.dry_run else None, work_dir=args.work_dir
        )
    print(f"input: {result.input_file}")
    if result.exported:
        for name, path in sorted(result.exported.items()):
            print(f"{name}: {path}")
    return 0


def _bundle_create(args: argparse.Namespace) -> int:
    created = create_bundle(args.config, args.output)
    print(created)
    return 0


def _bundle_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_bundle(args.bundle), indent=2, sort_keys=True))
    return 0


def _batch(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser()
    if target.is_dir():
        summary = _batch_folder(args, target.resolve())
    else:
        if args.profile or args.template:
            raise ValueError(
                "--profile/--template are only used when batch TARGET is a folder"
            )
        summary = run_batch_config(
            target,
            dry_run=True if args.dry_run else None,
            work_dir=args.work_dir,
        )
    print(summary["batch"]["summary"])
    return 0


def _batch_folder(args: argparse.Namespace, input_dir: Path) -> dict[str, Any]:
    if not args.profile and not args.template:
        raise ValueError("folder batch mode requires --profile or --template")
    output_root = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_dir / "parosol_batch"
    )
    images = _discover_batch_images(
        input_dir,
        patterns=args.pattern,
        recursive=bool(args.recursive),
    )
    if not images:
        raise ValueError(f"no input images found in {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    case_summaries: list[dict[str, Any]] = []
    for image_path in images:
        case_name = _case_stem(image_path)
        case_dir = output_root / case_name
        mask_path = _batch_mask_path(args, image_path) if args.mask_dir else None
        case_args = argparse.Namespace(
            image=str(image_path),
            profile=args.profile or "interactive_custom",
            template=args.template,
            mask=None if mask_path is None else str(mask_path),
            output=str(case_dir),
            name=case_name,
            dry_run=args.dry_run,
            side=None,
            reference_points=None,
            _argv=[
                "batch",
                str(input_dir),
                "--profile",
                args.profile or "interactive_custom",
                *(["--template", str(args.template)] if args.template else []),
                "--output",
                str(output_root),
            ],
        )
        config = _shortcut_config(case_args)
        config["execution"]["interface"] = "batch-folder"
        config["execution"]["batch_input_dir"] = str(input_dir)
        config["execution"]["batch_output_dir"] = str(output_root)
        case_dir.mkdir(parents=True, exist_ok=True)
        config_path = case_dir / "parosol_case.yaml"
        config["execution"]["generated_config"] = str(config_path)
        _write_yaml(config_path, config)
        run_case_config(
            config_path,
            dry_run=True if args.dry_run else None,
            work_dir=case_dir,
        )
        summary_path = case_dir / "result.json"
        case_summaries.append(_folder_case_summary(summary_path))

    summary_path = output_root / "result.json"
    summary = {
        "batch": {
            "name": input_dir.name,
            "mode": "folder",
            "profile": args.profile or "interactive_custom",
            "template": (
                None
                if not args.template
                else str(Path(args.template).expanduser().resolve())
            ),
            "case_count": len(case_summaries),
            "input_dir": str(input_dir),
            "output_dir": str(output_root),
            "summary": str(summary_path),
            "patterns": args.pattern or ["<supported image files>"],
            "recursive": bool(args.recursive),
        },
        "cases": case_summaries,
    }
    write_summary_json(summary_path, summary)
    return summary


def _load_history(args: argparse.Namespace) -> int:
    estimate_load_history_from_files(
        args.load_cases,
        bone_mask_path=args.bone_mask,
        output_path=args.output,
        summary_path=args.summary,
        target_average=args.target_average,
        cutoff_percentile=args.cutoff_percentile,
    )
    print(args.summary)
    return 0


def _summarize_legacy(args: argparse.Namespace) -> int:
    summary = {"reference": {}}
    if args.analysis:
        summary["reference"]["analysis"] = parse_legacy_analysis_file(
            Path(args.analysis)
        )
    if args.pistoia:
        summary["reference"]["pistoia"] = parse_pistoia_file(Path(args.pistoia))
    write_summary_json(args.output, summary)
    print(args.output)
    return 0


def _config_template(args: argparse.Namespace) -> int:
    print(read_config_template("default"))
    if args.profile:
        print("\n# --- profile override ---\n")
        print(read_config_template(args.profile))
    return 0


def _shortcut(args: argparse.Namespace) -> int:
    config = _shortcut_config(args)
    output_dir = Path(config["execution"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if "batch" in config:
        config_path = output_dir / "parosol_batch.yaml"
        config["execution"]["generated_config"] = str(config_path)
        _write_yaml(config_path, config)
        summary = run_batch_config(
            config_path,
            dry_run=True if args.dry_run else None,
            work_dir=output_dir,
        )
        print(f"config: {config_path}")
        print(f"summary: {summary['batch']['summary']}")
        return 0

    config_path = output_dir / "parosol_case.yaml"
    config["execution"]["generated_config"] = str(config_path)
    _write_yaml(config_path, config)

    result = run_case_config(
        config_path,
        dry_run=True if args.dry_run else None,
        work_dir=output_dir,
    )
    print(f"config: {config_path}")
    print(f"input: {result.input_file}")
    if result.exported:
        for name, path in sorted(result.exported.items()):
            print(f"{name}: {path}")
    return 0


def _shortcut_config(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image).expanduser().resolve()
    mask_path = Path(args.mask).expanduser().resolve() if args.mask else None
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else image_path.parent / f"{_case_stem(image_path)}_parosol"
    )
    case_name = args.name or _case_stem(image_path)
    if getattr(args, "template", None):
        template, workflow_path = load_workflow_template(args.template)
        return apply_workflow_template(
            template,
            image_path=image_path,
            mask_path=mask_path,
            output_dir=output_dir,
            case_name=case_name,
            profile=args.profile,
            command=" ".join(["parosol", *getattr(args, "_argv", [])]),
            template_path=workflow_path,
            dry_run=bool(args.dry_run),
        )
    builtin_workflow = builtin_workflow_path(args.profile)
    if builtin_workflow is not None:
        template, workflow_path = load_workflow_template(builtin_workflow)
        config = apply_workflow_template(
            template,
            image_path=image_path,
            mask_path=mask_path,
            output_dir=output_dir,
            case_name=case_name,
            profile=args.profile,
            command=" ".join(["parosol", *getattr(args, "_argv", [])]),
            template_path=workflow_path,
            dry_run=bool(args.dry_run),
        )
        config["execution"]["interface"] = "shortcut"
        return config
    if args.profile == "interactive_custom":
        raise ValueError("--profile interactive_custom requires --template")

    raise ValueError(f"unknown built-in workflow/profile recipe: {args.profile}")


def _discover_batch_images(
    input_dir: Path,
    *,
    patterns: list[str] | None,
    recursive: bool,
) -> list[Path]:
    candidates: list[Path] = []
    if patterns:
        for pattern in patterns:
            matches = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
            candidates.extend(path for path in matches if path.is_file())
    else:
        iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
        candidates.extend(
            path
            for path in iterator
            if path.is_file() and _is_supported_input_image(path)
        )
    return sorted(set(candidates), key=lambda path: str(path))


def _batch_mask_path(args: argparse.Namespace, image_path: Path) -> Path:
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    mask_name = args.mask_pattern.format(
        stem=_case_stem(image_path),
        name=image_path.name,
    )
    mask_path = mask_dir / mask_name
    if not mask_path.exists():
        raise ValueError(f"mask not found for {image_path.name}: {mask_path}")
    return mask_path


def _folder_case_summary(summary_path: Path) -> dict[str, Any]:
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mechanics = summary.get("mechanics", {})
    failure = summary.get("failure", {})
    return {
        "case": summary.get("case", {}),
        "image": summary.get("execution", {}).get("image"),
        "mask": summary.get("execution", {}).get("mask"),
        "summary": str(summary_path),
        "load_case": summary.get("load_case", {}),
        "generalized_load": mechanics.get("generalized_load"),
        "generalized_stiffness": mechanics.get("generalized_stiffness"),
        "failure_generalized_load": failure.get("failure_generalized_load"),
        "failure": {
            "factor": failure.get("factor"),
            "status": failure.get("status"),
        },
    }


def _write_yaml(path: Path, config: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write generated configs") from exc
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _dict_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.setdefault(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping in profile config")
    return value


def _case_stem(path: Path) -> str:
    return image_stem(path)


def _supports_image_metadata(path: Path) -> bool:
    suffixes = suffix_text(path)
    return suffixes.endswith((".aim", ".mha", ".mhd", ".nii", ".nii.gz", ".npz"))


def _is_supported_input_image(path: Path) -> bool:
    suffixes = suffix_text(path)
    return suffixes.endswith(
        (".aim", ".mha", ".mhd", ".nii", ".nii.gz", ".npy", ".npz")
    )


if __name__ == "__main__":
    raise SystemExit(main())
