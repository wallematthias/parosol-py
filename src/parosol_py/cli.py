from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import run_case_config
from .reports import parse_faim_analysis_file, parse_pistoia_file, write_summary_json


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"parosol: error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parosol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a ParOSol case config")
    run_parser.add_argument("config", help="Path to a .yaml, .toml, or .json case config")
    run_parser.add_argument("--dry-run", action="store_true", help="Write inputs/summary without launching ParOSol")
    run_parser.add_argument("--work-dir", help="Override the configured work directory")
    run_parser.set_defaults(func=_run)

    summary_parser = subparsers.add_parser(
        "summarize-faim",
        help="Convert old FAIM analysis/Pistoia text outputs to compact JSON",
    )
    summary_parser.add_argument("--analysis", help="Path to old *_analysis.txt")
    summary_parser.add_argument("--pistoia", help="Path to old *_pistoia.txt")
    summary_parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    summary_parser.set_defaults(func=_summarize_faim)
    return parser


def _run(args: argparse.Namespace) -> int:
    result = run_case_config(args.config, dry_run=True if args.dry_run else None, work_dir=args.work_dir)
    print(f"input: {result.input_file}")
    if result.exported:
        for name, path in sorted(result.exported.items()):
            print(f"{name}: {path}")
    return 0


def _summarize_faim(args: argparse.Namespace) -> int:
    summary = {"faim": {}}
    if args.analysis:
        summary["faim"]["analysis"] = parse_faim_analysis_file(Path(args.analysis))
    if args.pistoia:
        summary["faim"]["pistoia"] = parse_pistoia_file(Path(args.pistoia))
    write_summary_json(args.output, summary)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
