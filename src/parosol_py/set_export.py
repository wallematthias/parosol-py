from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_node_sets(
    node_sets: dict[str, list[tuple[int, int, int]]],
    *,
    directory: str | Path,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    formats: tuple[str, ...] = ("json",),
) -> dict[str, Path]:
    out_dir = Path(directory).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    tokens = {fmt.strip().lower() for fmt in formats}
    if "json" in tokens:
        path = out_dir / "node_sets.json"
        path.write_text(
            json.dumps(
                {
                    name: [list(coord) for coord in coords]
                    for name, coords in sorted(node_sets.items())
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written["node_sets_json"] = path
    if "vtk" in tokens:
        for name, coords in sorted(node_sets.items()):
            path = out_dir / f"{name}_nodes.vtk"
            _write_points_vtk(path, coords, spacing=spacing, origin=origin)
            written[f"{name}_nodes_vtk"] = path
    return written


def write_element_sets(
    material_xyz,
    *,
    directory: str | Path,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    formats: tuple[str, ...] = ("json",),
) -> dict[str, Path]:
    out_dir = Path(directory).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    material = np.asarray(material_xyz)
    labels = sorted(int(value) for value in np.unique(material) if value != 0)
    element_sets = {
        str(label): np.argwhere(material == label).astype(int).tolist()
        for label in labels
    }
    written: dict[str, Path] = {}
    tokens = {fmt.strip().lower() for fmt in formats}
    if "json" in tokens:
        path = out_dir / "element_sets.json"
        path.write_text(
            json.dumps(element_sets, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written["element_sets_json"] = path
    if "vtk" in tokens:
        for label, coords in element_sets.items():
            path = out_dir / f"material_{label}_elements.vtk"
            centers = [
                (int(coord[0]) + 0.5, int(coord[1]) + 0.5, int(coord[2]) + 0.5)
                for coord in coords
            ]
            _write_points_vtk(path, centers, spacing=spacing, origin=origin)
            written[f"material_{label}_elements_vtk"] = path
    return written


def _write_points_vtk(
    path: Path,
    coords,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> None:
    points = [
        _physical_point(coord, spacing=spacing, origin=origin) for coord in coords
    ]
    lines = [
        "# vtk DataFile Version 3.0",
        "parosol-py point set",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(f"{x:g} {y:g} {z:g}" for x, y, z in points)
    lines.extend(
        [
            f"VERTICES {len(points)} {len(points) * 2}",
            *[f"1 {index}" for index in range(len(points))],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _physical_point(
    coord,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        float(origin[index]) + float(coord[index]) * float(spacing[index])
        for index in range(3)
    )
