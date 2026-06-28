from __future__ import annotations

import argparse
import json
from pathlib import Path

from parosol_py.workflow_baseline import build_builtin_workflow_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a compact built-in workflow baseline JSON file.")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_builtin_workflow_baseline(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
