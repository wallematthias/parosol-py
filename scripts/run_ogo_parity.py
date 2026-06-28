from __future__ import annotations

import argparse
from pathlib import Path

from parosol_py.parity import (
    PARITY_CASE_SPECS,
    compare_case_metrics,
    compare_metric,
    reference_bundle_assets,
    summarize_metric_comparisons,
    write_parity_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare compact ParOSol and Ogo/n88 parity metrics.")
    parser.add_argument("--case", required=True, choices=tuple(sorted(PARITY_CASE_SPECS)))
    parser.add_argument("--reference-bundle", help="Ogo/n88 reference bundle root to validate and record")
    parser.add_argument(
        "--check-assets-only",
        action="store_true",
        help="Only validate reference-bundle assets and write the summary",
    )
    parser.add_argument("--observed-force", type=float)
    parser.add_argument("--expected-force", type=float)
    parser.add_argument("--observed-stiffness", type=float)
    parser.add_argument("--expected-stiffness", type=float)
    parser.add_argument(
        "--use-case-defaults",
        action="store_true",
        help="Use versioned expected metrics and tolerance for the selected case",
    )
    parser.add_argument("--tolerance-percent", type=float)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    if args.check_assets_only and not args.reference_bundle:
        raise SystemExit("--check-assets-only requires --reference-bundle")

    reference_assets = None
    if args.reference_bundle:
        reference_assets = reference_bundle_assets(args.case, root=args.reference_bundle)

    if args.check_assets_only:
        summary = {
            "case": args.case,
            "status": "passed",
            "failed_metrics": [],
            "comparisons": [],
        }
    elif args.use_case_defaults:
        _require_observed_metrics(args)
        summary = compare_case_metrics(
            args.case,
            observed_metrics={
                "reaction_force_N": args.observed_force,
                "stiffness_N_per_mm": args.observed_stiffness,
            },
            tolerance_percent=args.tolerance_percent,
        )
    else:
        _require_explicit_metrics(args)
        summary = summarize_metric_comparisons(
            [
                compare_metric(
                    "reaction_force_N",
                    observed=args.observed_force,
                    expected=args.expected_force,
                    tolerance_percent=_explicit_tolerance(args),
                ),
                compare_metric(
                    "stiffness_N_per_mm",
                    observed=args.observed_stiffness,
                    expected=args.expected_stiffness,
                    tolerance_percent=_explicit_tolerance(args),
                ),
            ]
        )
        summary["case"] = args.case
    if reference_assets is not None:
        summary["reference_bundle"] = reference_assets
    write_parity_summary(Path(args.summary), summary)
    print(summary["status"])
    return 0 if summary["status"] == "passed" else 1


def _require_observed_metrics(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("observed_force", "observed_stiffness")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"missing required observed metric(s): {', '.join(missing)}")


def _require_explicit_metrics(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "observed_force",
            "expected_force",
            "observed_stiffness",
            "expected_stiffness",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"missing required metric(s): {', '.join(missing)}")


def _explicit_tolerance(args: argparse.Namespace) -> float:
    return 10.0 if args.tolerance_percent is None else float(args.tolerance_percent)


if __name__ == "__main__":
    raise SystemExit(main())
