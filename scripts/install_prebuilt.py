#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install the matching parosol-py wheel from a private GitHub Release. "
            "Requires GitHub CLI authentication for private repositories."
        )
    )
    parser.add_argument(
        "--repo",
        default="wallematthias/parosol-py",
        help="GitHub repository containing release wheel artifacts.",
    )
    parser.add_argument(
        "--tag",
        help="Release tag to install. Defaults to the latest release.",
    )
    parser.add_argument(
        "--package",
        default="parosol-py",
        help="Package requirement to install from the downloaded wheel directory.",
    )
    parser.add_argument(
        "--keep-wheels",
        type=Path,
        help="Directory where downloaded wheels should be kept for reuse.",
    )
    args = parser.parse_args(argv)

    if shutil.which("gh") is None:
        raise SystemExit(
            "GitHub CLI is required for private release downloads. "
            "Install it from https://cli.github.com/ and run 'gh auth login'."
        )

    with _wheel_dir(args.keep_wheels) as wheel_dir:
        command = [
            "gh",
            "release",
            "download",
            "--repo",
            args.repo,
            "--pattern",
            "*.whl",
            "--dir",
            str(wheel_dir),
        ]
        if args.tag:
            command.insert(3, args.tag)
        _run(command)
        wheels = sorted(wheel_dir.glob("*.whl"))
        if not wheels:
            raise SystemExit(f"No wheels were downloaded from {args.repo}.")
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(wheel_dir),
                args.package,
            ]
        )
    return 0


class _wheel_dir:
    def __init__(self, keep_wheels: Path | None):
        self.keep_wheels = keep_wheels
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        if self.keep_wheels is not None:
            self.keep_wheels.expanduser().mkdir(parents=True, exist_ok=True)
            return self.keep_wheels.expanduser().resolve()
        self._tmp = tempfile.TemporaryDirectory(prefix="parosol-wheels-")
        return Path(self._tmp.name)

    def __exit__(self, *exc_info) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
