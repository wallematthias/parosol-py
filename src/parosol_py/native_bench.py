from __future__ import annotations

from pathlib import Path


def build_case_matrix(output_root: Path) -> list[dict[str, object]]:
    return [
        {
            "name": "mpi1_no_fields",
            "work_dir": output_root / "mpi1_no_fields",
            "mpi": 1,
            "outputs": [],
        },
        {
            "name": "mpi1_sed",
            "work_dir": output_root / "mpi1_sed",
            "mpi": 1,
            "outputs": ["sed"],
        },
        {
            "name": "mpi4_no_fields",
            "work_dir": output_root / "mpi4_no_fields",
            "mpi": 4,
            "outputs": [],
        },
        {
            "name": "mpi4_sed",
            "work_dir": output_root / "mpi4_sed",
            "mpi": 4,
            "outputs": ["sed"],
        },
    ]
