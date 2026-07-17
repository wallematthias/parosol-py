from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

OUTPUT_DATASETS = {
    "sed": "SED",
    "von_mises": "VonMises",
    "effective_strain": "EFF",
    "deviatoric_strain": "e_dev",
    "volumetric_strain": "e_vol",
}
DEFAULT_OUTPUTS = ("sed", "effective_strain", "von_mises")
TENSOR_AXES = ("xx", "yy", "zz", "xy", "yz", "xz")


def read_solution_fields(
    path: str | Path, *, outputs: tuple[str, ...] = DEFAULT_OUTPUTS
) -> dict[str, Any]:
    requested = tuple(output.strip().lower() for output in outputs)
    fields: dict[str, Any] = {}

    with h5py.File(Path(path), "r") as h5:
        if "Solution" not in h5:
            raise ValueError("ParOSol output does not contain /Solution")

        solution = h5["Solution"]
        for output in requested:
            if output in OUTPUT_DATASETS:
                fields[output] = _read_dataset(
                    solution, output, OUTPUT_DATASETS[output]
                )
            elif output == "strain":
                fields[output] = _read_tensor(solution, prefix="e_")
            elif output == "stress":
                fields[output] = _read_tensor(solution, prefix="s_")
            elif output == "plastic_strain":
                fields[output] = _read_dataset(solution, output, "PlasticStrain")
            elif output in {"forces", "force"}:
                fields["forces"] = _read_dataset(solution, output, "force")
            elif output in {"displacements", "disp"}:
                fields["displacements"] = _read_dataset(solution, output, "disp")
            else:
                raise ValueError(f"Unsupported output '{output}'")

    return fields


def _read_tensor(solution: h5py.Group, *, prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for axis in TENSOR_AXES:
        name = f"{prefix}{axis}"
        if name not in solution:
            raise ValueError(
                f"Requested tensor component not found in /Solution/{name}"
            )
        out[axis] = np.asarray(solution[name][...])
    return out


def _read_dataset(solution: h5py.Group, output: str, dataset: str) -> np.ndarray:
    if dataset not in solution:
        raise ValueError(
            f"Requested output '{output}' not found in /Solution/{dataset}"
        )
    return np.asarray(solution[dataset][...])
