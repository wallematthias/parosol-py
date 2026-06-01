from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary_conditions import AXIS_TO_INDEX

Node = tuple[int, int, int]


@dataclass(frozen=True)
class SurfaceSelection:
    mode: str = "intersection"
    depth: int | str | None = None

    @classmethod
    def from_value(cls, value) -> "SurfaceSelection":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(mode=value)
        return cls(
            mode=str(value.get("mode", value.get("type", "intersection"))),
            depth=value.get("depth", value.get("maximum_depth")),
        )


def top_bottom_surface_nodes(
    material_xyz,
    *,
    axis: str,
    selection: SurfaceSelection | str | dict | None = None,
) -> dict[str, list[Node]]:
    selected = SurfaceSelection.from_value(selection)
    mode = selected.mode.strip().lower()
    if mode in {"intersection", "boundary"}:
        return _intersection_surface_nodes(material_xyz, axis=axis)
    if mode in {"visible", "uneven", "smart", "auto"}:
        return _visible_surface_nodes(
            material_xyz,
            axis=axis,
            depth=_surface_depth(material_xyz, axis=axis, depth=selected.depth),
        )
    raise ValueError(
        "surface mode must be intersection, visible, uneven, smart, or auto"
    )


def _intersection_surface_nodes(material_xyz, *, axis: str) -> dict[str, list[Node]]:
    axis_index = AXIS_TO_INDEX[axis]
    dims = tuple(int(v) for v in np.asarray(material_xyz).shape)
    occupied = np.asarray(material_xyz) > 0
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    out = {"bottom": set(), "top": set()}
    for label, element_axis_value, node_axis_value in (
        ("bottom", 0, 0),
        ("top", dims[axis_index] - 1, dims[axis_index]),
    ):
        surface = np.take(occupied, indices=element_axis_value, axis=axis_index)
        for lateral_index in np.argwhere(surface):
            base = [0, 0, 0]
            base[axis_index] = node_axis_value
            base[lateral_axes[0]] = int(lateral_index[0])
            base[lateral_axes[1]] = int(lateral_index[1])
            _add_face_nodes(out[label], base, lateral_axes)
    return {name: sorted(coords) for name, coords in out.items()}


def _visible_surface_nodes(
    material_xyz, *, axis: str, depth: int
) -> dict[str, list[Node]]:
    axis_index = AXIS_TO_INDEX[axis]
    occupied = np.asarray(material_xyz) > 0
    dims = tuple(int(v) for v in occupied.shape)
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    out = {"bottom": set(), "top": set()}
    lateral_shape = tuple(dims[idx] for idx in lateral_axes)

    for lateral_index in np.ndindex(lateral_shape):
        slicer = [slice(None), slice(None), slice(None)]
        slicer[lateral_axes[0]] = lateral_index[0]
        slicer[lateral_axes[1]] = lateral_index[1]
        column = occupied[tuple(slicer)]
        active = np.flatnonzero(column)
        if active.size == 0:
            continue
        bottom_limit = min(int(active[0]) + depth - 1, int(active[-1]))
        top_limit = max(int(active[-1]) - depth + 1, int(active[0]))
        for element_axis_value in range(int(active[0]), bottom_limit + 1):
            _add_surface_voxel_nodes(
                out["bottom"],
                axis_index=axis_index,
                lateral_axes=lateral_axes,
                lateral_index=lateral_index,
                element_axis_value=element_axis_value,
                node_axis_value=element_axis_value,
            )
        for element_axis_value in range(top_limit, int(active[-1]) + 1):
            _add_surface_voxel_nodes(
                out["top"],
                axis_index=axis_index,
                lateral_axes=lateral_axes,
                lateral_index=lateral_index,
                element_axis_value=element_axis_value,
                node_axis_value=element_axis_value + 1,
            )
    return {name: sorted(coords) for name, coords in out.items()}


def _surface_depth(material_xyz, *, axis: str, depth: int | str | None) -> int:
    if depth is None:
        return 1
    if isinstance(depth, str):
        token = depth.strip().lower()
        if token == "auto":
            thicknesses = _column_thicknesses(material_xyz, axis=axis)
            if thicknesses.size == 0:
                return 1
            return max(
                1, min(5, int(round(float(np.percentile(thicknesses, 10)) * 0.1)))
            )
        return int(token)
    return int(depth)


def _column_thicknesses(material_xyz, *, axis: str) -> np.ndarray:
    axis_index = AXIS_TO_INDEX[axis]
    occupied = np.asarray(material_xyz) > 0
    dims = tuple(int(v) for v in occupied.shape)
    lateral_axes = [idx for idx in range(3) if idx != axis_index]
    lateral_shape = tuple(dims[idx] for idx in lateral_axes)
    thicknesses: list[int] = []
    for lateral_index in np.ndindex(lateral_shape):
        slicer = [slice(None), slice(None), slice(None)]
        slicer[lateral_axes[0]] = lateral_index[0]
        slicer[lateral_axes[1]] = lateral_index[1]
        active = np.flatnonzero(occupied[tuple(slicer)])
        if active.size:
            thicknesses.append(int(active[-1] - active[0] + 1))
    return np.asarray(thicknesses, dtype=np.int64)


def _add_surface_voxel_nodes(
    nodes: set[Node],
    *,
    axis_index: int,
    lateral_axes: list[int],
    lateral_index: tuple[int, int],
    element_axis_value: int,
    node_axis_value: int,
) -> None:
    base = [0, 0, 0]
    base[axis_index] = int(node_axis_value)
    base[lateral_axes[0]] = int(lateral_index[0])
    base[lateral_axes[1]] = int(lateral_index[1])
    _add_face_nodes(nodes, base, lateral_axes)


def _add_face_nodes(nodes: set[Node], base: list[int], lateral_axes: list[int]) -> None:
    for du in (0, 1):
        for dv in (0, 1):
            node = base.copy()
            node[lateral_axes[0]] += du
            node[lateral_axes[1]] += dv
            nodes.add(tuple(node))
