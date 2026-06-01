from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .batch import run_batch_config
from .config import run_case_config
from .config_templates import available_config_profiles, read_config_template
from .load_history import estimate_load_history_from_files
from .reports import parse_legacy_analysis_file, parse_pistoia_file, write_summary_json

_COMMANDS = {"run", "batch", "load-history", "summarize-legacy", "config-template"}


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
            "Run profile-driven ParOSol cases. Shortcut form: "
            "parosol IMAGE --profile PROFILE [--mask MASK] [--output OUT]."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a ParOSol case config")
    run_parser.add_argument(
        "config", help="Path to a .yaml, .toml, or .json case config"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write inputs/summary without launching ParOSol",
    )
    run_parser.add_argument("--work-dir", help="Override the configured work directory")
    run_parser.set_defaults(func=_run)

    batch_parser = subparsers.add_parser("batch", help="Run a ParOSol batch config")
    batch_parser.add_argument(
        "config", help="Path to a .yaml, .toml, or .json batch config"
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
        help="Print the commented default config and optional profile override",
    )
    template_parser.add_argument(
        "--profile",
        choices=available_config_profiles(),
        help="Append a user-facing profile override snippet",
    )
    template_parser.set_defaults(func=_config_template)
    return parser


def _build_shortcut_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parosol",
        description="Run one image with a ParOSol profile",
    )
    parser.add_argument("image", help="Input material/density image")
    parser.add_argument(
        "--profile",
        required=True,
        choices=available_config_profiles(),
        help="Built-in profile to apply",
    )
    parser.add_argument(
        "--mask",
        help="Optional segmentation/nodeset mask. Model profiles use this as the model mask.",
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
        help="Model side override for profiles that support it, such as proximal femur.",
    )
    parser.add_argument(
        "--reference-points",
        help="Reference point cloud for lightweight ICP registration.",
    )
    parser.set_defaults(func=_shortcut)
    return parser


def _run(args: argparse.Namespace) -> int:
    result = run_case_config(
        args.config, dry_run=True if args.dry_run else None, work_dir=args.work_dir
    )
    print(f"input: {result.input_file}")
    if result.exported:
        for name, path in sorted(result.exported.items()):
            print(f"{name}: {path}")
    return 0


def _batch(args: argparse.Namespace) -> int:
    summary = run_batch_config(
        args.config,
        dry_run=True if args.dry_run else None,
        work_dir=args.work_dir,
    )
    print(summary["batch"]["summary"])
    return 0


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
    output_dir = Path(config["case"]["work_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    config = _load_profile(args.profile)
    image_path = Path(args.image).expanduser().resolve()
    mask_path = Path(args.mask).expanduser().resolve() if args.mask else None
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else image_path.parent / f"{_case_stem(image_path)}_parosol"
    )
    case_name = args.name or _case_stem(image_path)

    case_cfg = _dict_section(config, "case")
    case_cfg["name"] = case_name
    case_cfg["work_dir"] = str(output_dir)
    output_cfg = _dict_section(config, "output")
    output_cfg["summary"] = str(output_dir / "summary.json")
    output_cfg.setdefault("fields", ["sed"])
    output_cfg["fields_dir"] = str(output_dir / "fields")
    output_cfg["visualization"] = str(output_dir / "overview.png")

    if "model" in config:
        model_cfg = _dict_section(config, "model")
        model_cfg["density_image"] = str(image_path)
        if mask_path is None:
            raise ValueError(f"profile {args.profile!r} requires --mask")
        model_cfg["mask_image"] = str(mask_path)
        if args.side:
            model_cfg["side"] = args.side
        if args.reference_points:
            registration_cfg = _dict_section(model_cfg, "registration")
            registration_cfg["enabled"] = True
            registration_cfg.setdefault("method", "lightweight_icp")
            registration_cfg["reference_points"] = str(
                Path(args.reference_points).expanduser().resolve()
            )
        model_outputs = _dict_section(model_cfg, "outputs")
        model_dir = output_dir / "model"
        model_outputs["material_image"] = str(model_dir / "material.nii.gz")
        model_outputs["nodeset_image"] = str(model_dir / "nodesets.nii.gz")
        model_outputs["manifest"] = str(model_dir / "model.json")
        model_outputs["qc_image"] = str(model_dir / "qc.png")
    else:
        input_cfg = _dict_section(config, "input")
        input_cfg["image"] = str(image_path)
        if _supports_image_metadata(image_path):
            input_cfg.setdefault("spacing", "auto")
            input_cfg.setdefault("origin", "auto")
        if mask_path is not None:
            input_cfg["mask"] = str(mask_path)

    config["execution"] = {
        "interface": "shortcut",
        "command": " ".join(["parosol", *getattr(args, "_argv", [])]),
        "profile": args.profile,
        "image": str(image_path),
        "mask": None if mask_path is None else str(mask_path),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
    }
    return config


def _load_profile(profile: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to run built-in profiles") from exc
    loaded = yaml.safe_load(read_config_template(profile))
    return {} if loaded is None else loaded


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
    name = path.name
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".aim", ".AIM", ".npy", ".npz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _supports_image_metadata(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return suffixes.endswith((".aim", ".mha", ".mhd", ".nii", ".nii.gz", ".npz"))


if __name__ == "__main__":
    raise SystemExit(main())
