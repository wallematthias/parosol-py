from __future__ import annotations

import argparse
from pathlib import Path

from parosol_py.parity import compare_metric, summarize_metric_comparisons, write_parity_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare compact ParOSol and Ogo/n88 parity metrics.")
    parser.add_argument("--case", required=True, choices=("hip_10001", "spine_10001"))
    parser.add_argument("--observed-force", required=True, type=float)
    parser.add_argument("--expected-force", required=True, type=float)
    parser.add_argument("--observed-stiffness", required=True, type=float)
    parser.add_argument("--expected-stiffness", required=True, type=float)
    parser.add_argument("--tolerance-percent", type=float, default=10.0)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    summary = summarize_metric_comparisons(
        [
            compare_metric(
                "reaction_force_N",
                observed=args.observed_force,
                expected=args.expected_force,
                tolerance_percent=args.tolerance_percent,
            ),
            compare_metric(
                "stiffness_N_per_mm",
                observed=args.observed_stiffness,
                expected=args.expected_stiffness,
                tolerance_percent=args.tolerance_percent,
            ),
        ]
    )
    summary["case"] = args.case
    write_parity_summary(Path(args.summary), summary)
    print(summary["status"])
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
