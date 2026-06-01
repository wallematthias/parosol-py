from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core import BoundaryConditionSet
from .field_export import NativeFieldMapper


def write_case_overview(
    material_xyz,
    *,
    output_path: str | Path,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    field_xyz=None,
    field_name: str = "SED",
    field_mask_xyz=None,
    material_labels_xyz=None,
    boundary_conditions: BoundaryConditionSet | None = None,
    title: str | None = None,
) -> Path:
    """Write a compact 2D material/result/boundary-condition overview PNG."""
    material = np.asarray(material_xyz)
    if material.ndim != 3:
        raise ValueError(f"material_xyz must be 3D, got shape {material.shape}")
    if field_xyz is not None:
        field = np.asarray(field_xyz)
        if field.shape != material.shape:
            raise ValueError(
                f"field_xyz shape {field.shape} does not match material {material.shape}"
            )
        if field_mask_xyz is not None:
            field_mask = np.asarray(field_mask_xyz, dtype=bool)
            if field_mask.shape != material.shape:
                raise ValueError(
                    f"field_mask_xyz shape {field_mask.shape} does not match material {material.shape}"
                )
        else:
            field_mask = material > 0
        field_context = (material > 0) & ~field_mask
        field = np.where(field_mask, field, np.nan)
    else:
        field = None
        field_context = None
    if material_labels_xyz is not None:
        material_labels = np.asarray(material_labels_xyz)
        if material_labels.shape != material.shape:
            raise ValueError(
                f"material_labels_xyz shape {material_labels.shape} does not match material {material.shape}"
            )
    else:
        material_labels = None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    field_cmap = plt.get_cmap("jet").copy()
    field_cmap.set_bad(color=(0.0, 0.0, 0.0, 0.0))
    context_cmap = plt.matplotlib.colors.ListedColormap(
        [(0.0, 0.0, 0.0, 0.0), (0.45, 0.50, 0.58, 1.0)]
    )

    slices = _mid_slices(material, spacing=spacing, origin=origin)
    label_slices = (
        _mid_slices(material_labels, spacing=spacing, origin=origin)
        if material_labels is not None
        else {}
    )
    field_slices = (
        _mid_slices(field, spacing=spacing, origin=origin) if field is not None else {}
    )
    context_slices = (
        _mid_slices(field_context, spacing=spacing, origin=origin)
        if field_context is not None and np.any(field_context)
        else {}
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8))
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(title or "FEA overview", fontsize=15, fontweight="semibold")

    material_vmin, material_vmax = _finite_range(material)
    field_vmin, field_vmax = (
        _finite_range(field, robust_nonzero=True) if field is not None else (0.0, 1.0)
    )
    material_images = []
    field_images = []
    for col, view in enumerate(("axial", "sagittal", "coronal")):
        view_slice = slices[view]
        ax = axes[0, col]
        ax.set_facecolor("#f1f5f9")
        image = ax.imshow(
            view_slice.image,
            origin="lower",
            cmap="bone",
            vmin=material_vmin,
            vmax=material_vmax,
            interpolation="nearest",
            extent=view_slice.extent,
        )
        material_images.append(image)
        if material_labels is not None:
            _overlay_material_labels(ax, label_slices[view].image)
        _style_axis(
            ax,
            f"Material: {view_slice.label}",
            view_slice.xlabel,
            view_slice.ylabel,
        )
        if boundary_conditions is not None:
            _overlay_boundary_conditions(
                ax,
                boundary_conditions,
                view=view,
                slice_axis=view_slice.axis,
                slice_index=view_slice.index,
                spacing=spacing,
                origin=origin,
            )

        ax = axes[1, col]
        ax.set_facecolor("#f1f5f9")
        if field is None:
            image = ax.imshow(
                (slices[view].image > 0).astype(float),
                origin="lower",
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                extent=slices[view].extent,
                alpha=0.65,
            )
            ax.text(
                0.5,
                0.5,
                f"{field_name} not available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#334155",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.92,
                },
            )
        else:
            image = ax.imshow(
                field_slices[view].image,
                origin="lower",
                cmap=field_cmap,
                vmin=field_vmin,
                vmax=field_vmax,
                interpolation="nearest",
                extent=field_slices[view].extent,
            )
            field_images.append(image)
            if context_slices:
                ax.imshow(
                    context_slices[view].image.astype(float),
                    origin="lower",
                    cmap=context_cmap,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                    extent=context_slices[view].extent,
                    alpha=0.82,
                    zorder=3,
                )
        _style_axis(
            ax,
            f"{field_name}: {view_slice.label}",
            view_slice.xlabel,
            view_slice.ylabel,
        )
        if boundary_conditions is not None:
            _overlay_boundary_conditions(
                ax,
                boundary_conditions,
                view=view,
                slice_axis=view_slice.axis,
                slice_index=view_slice.index,
                spacing=spacing,
                origin=origin,
            )

    if material_images:
        material_cax = fig.add_axes((0.925, 0.58, 0.014, 0.26))
        fig.colorbar(
            material_images[0],
            cax=material_cax,
            label="Material stiffness",
        )
    if field_images:
        field_cax = fig.add_axes((0.925, 0.22, 0.014, 0.26))
        fig.colorbar(
            field_images[0],
            cax=field_cax,
            label=field_name,
        )
    if boundary_conditions is not None:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#64748b",
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=6,
                label="fixed",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="#f97316",
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=7,
                label="prescribed displacement",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="#0891b2",
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=7,
                label="applied force",
            ),
        ]
        if context_slices:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="none",
                    markerfacecolor="#94a3b8",
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    markersize=7,
                    label="PMMA/context",
                )
            )
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=min(4, len(handles)),
            frameon=False,
        )
    fig.subplots_adjust(
        left=0.055,
        right=0.89,
        top=0.9,
        bottom=0.13,
        wspace=0.24,
        hspace=0.32,
    )
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def dense_scalar_field(material_xyz, values) -> np.ndarray | None:
    """Map native dense or active element values onto the material grid for display."""
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        return None
    material = np.asarray(material_xyz)
    expected = (material.size, int(np.count_nonzero(material > 0)))
    if array.size not in expected:
        return None
    return NativeFieldMapper(material).scalar_to_dense(array)


class _Slice:
    def __init__(
        self,
        image,
        extent,
        label: str,
        xlabel: str,
        ylabel: str,
        axis: int,
        index: int,
    ) -> None:
        self.image = image
        self.extent = extent
        self.label = label
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.axis = axis
        self.index = index


def _mid_slices(
    array: np.ndarray | None,
    *,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, _Slice]:
    if array is None:
        return {}
    nx, ny, nz = (int(v) for v in array.shape)
    x_mid, y_mid, z_mid = nx // 2, ny // 2, nz // 2
    x0, x1 = _axis_extent(nx, spacing[0], origin[0])
    y0, y1 = _axis_extent(ny, spacing[1], origin[1])
    z0, z1 = _axis_extent(nz, spacing[2], origin[2])
    return {
        "axial": _Slice(
            array[:, :, z_mid].T,
            (x0, x1, y0, y1),
            f"z={z_mid}",
            "x (mm)",
            "y (mm)",
            2,
            z_mid,
        ),
        "sagittal": _Slice(
            array[x_mid, :, :].T,
            (y0, y1, z0, z1),
            f"x={x_mid}",
            "y (mm)",
            "z (mm)",
            0,
            x_mid,
        ),
        "coronal": _Slice(
            array[:, y_mid, :].T,
            (x0, x1, z0, z1),
            f"y={y_mid}",
            "x (mm)",
            "z (mm)",
            1,
            y_mid,
        ),
    }


def _axis_extent(size: int, spacing: float, origin: float) -> tuple[float, float]:
    half = float(spacing) * 0.5
    return float(origin) - half, float(origin) + (int(size) - 0.5) * float(spacing)


def _style_axis(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="semibold", color="#0f172a")
    ax.set_xlabel(xlabel, fontsize=8, color="#475569")
    ax.set_ylabel(ylabel, fontsize=8, color="#475569")
    ax.tick_params(labelsize=7, colors="#475569")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.set_aspect("equal")


def _overlay_material_labels(ax: Any, labels: np.ndarray) -> None:
    import matplotlib.colors as mcolors

    values = np.asarray(labels)
    active = np.ma.masked_where(values <= 0, values)
    if active.count() == 0:
        return
    cmap = mcolors.ListedColormap(
        [
            "#22c55e",  # body/bone
            "#14b8a6",  # process/second bone label
            "#f97316",  # inferior/contact cap
            "#ef4444",  # superior/contact cap
            "#8b5cf6",
            "#0ea5e9",
        ]
    )
    unique = [int(v) for v in np.unique(values) if int(v) > 0]
    remapped = np.zeros(values.shape, dtype=np.float32)
    for idx, label in enumerate(unique, start=1):
        remapped[values == label] = float(idx)
    masked = np.ma.masked_where(remapped <= 0, remapped)
    ax.imshow(
        masked,
        origin="lower",
        cmap=cmap,
        vmin=1,
        vmax=max(1, len(unique)),
        interpolation="nearest",
        extent=ax.images[0].get_extent(),
        alpha=0.32,
        zorder=2,
    )
    for label in unique:
        if label < 10:
            continue
        ax.contour(
            values == label,
            levels=[0.5],
            colors=["#ffffff"],
            linewidths=0.7,
            origin="lower",
            extent=ax.images[0].get_extent(),
            zorder=3,
        )


def _overlay_boundary_conditions(
    ax: Any,
    boundary_conditions: BoundaryConditionSet,
    *,
    view: str,
    slice_axis: int,
    slice_index: int,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> None:
    fixed_coords = np.asarray(boundary_conditions.fixed_coordinates)
    fixed_values = np.asarray(boundary_conditions.fixed_values)
    loaded_coords = np.asarray(boundary_conditions.loaded_coordinates)
    loaded_values = np.asarray(boundary_conditions.loaded_values)
    fixed_node_mask, prescribed_node_mask = _fixed_and_prescribed_node_masks(
        fixed_coords, fixed_values
    )
    fixed_slice = _slice_mask(fixed_coords, axis=slice_axis, index=slice_index)
    loaded_slice = _slice_mask(loaded_coords, axis=slice_axis, index=slice_index)
    true_fixed = fixed_slice & fixed_node_mask
    prescribed = fixed_slice & prescribed_node_mask
    loaded_coords = loaded_coords[loaded_slice]
    loaded_values = loaded_values[loaded_slice]
    _render_bc_group(
        ax,
        _unique_node_coordinates(fixed_coords[true_fixed]),
        values=None,
        view=view,
        spacing=spacing,
        origin=origin,
        color="#64748b",
        marker="o",
        alpha=0.72,
    )
    _draw_bc_symbols(
        ax,
        fixed_coords[prescribed],
        fixed_values[prescribed],
        view=view,
        spacing=spacing,
        origin=origin,
        color="#f97316",
    )
    _draw_bc_symbols(
        ax,
        loaded_coords,
        loaded_values,
        view=view,
        spacing=spacing,
        origin=origin,
        color="#0891b2",
    )


def _fixed_and_prescribed_node_masks(
    coords: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if coords.size == 0:
        empty = np.zeros((0,), dtype=bool)
        return empty, empty
    zero = np.isclose(values, 0.0) | np.isclose(values, 1e-16)
    nonzero_nodes = {
        tuple(int(v) for v in coord[:3]) for coord in coords[~zero]
    }
    prescribed = np.asarray(
        [tuple(int(v) for v in coord[:3]) in nonzero_nodes for coord in coords],
        dtype=bool,
    )
    fixed = zero & ~prescribed
    return fixed, prescribed


def _unique_node_coordinates(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return coords.reshape((0, 4))
    nodes = np.unique(coords[:, :3], axis=0)
    return np.column_stack([nodes, np.zeros(nodes.shape[0], dtype=coords.dtype)])


def _slice_mask(coords: np.ndarray, *, axis: int, index: int) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0,), dtype=bool)
    node_axis = coords[:, axis].astype(float) - 0.5
    return np.abs(node_axis - float(index)) <= 0.5


def _render_bc_group(
    ax: Any,
    coords: np.ndarray,
    *,
    values: np.ndarray | None,
    view: str,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    color: str,
    marker: str,
    alpha: float,
) -> None:
    if coords.size == 0:
        return
    if coords.shape[0] <= 600:
        if values is None:
            _scatter_bc(
                ax,
                coords,
                view=view,
                spacing=spacing,
                origin=origin,
                color=color,
                marker=marker,
                alpha=alpha,
            )
        else:
            _draw_bc_symbols(
                ax,
                coords,
                values,
                view=view,
                spacing=spacing,
                origin=origin,
                color=color,
            )
        return
    _density_overlay(
        ax,
        coords,
        view=view,
        spacing=spacing,
        origin=origin,
        color=color,
        alpha=max(alpha * 2.5, 0.45),
    )
    if values is not None:
        _draw_representative_symbols(
            ax,
            coords,
            values,
            view=view,
            spacing=spacing,
            origin=origin,
            color=color,
        )


def _scatter_bc(
    ax: Any,
    coords: np.ndarray,
    *,
    view: str,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    color: str,
    marker: str,
    alpha: float,
) -> None:
    if coords.size == 0:
        return
    u, v = _project_points(coords, view=view, spacing=spacing, origin=origin)
    ax.scatter(
        u,
        v,
        s=16,
        c=color,
        marker=marker,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.35,
        clip_on=False,
        zorder=4,
    )


def _draw_bc_symbols(
    ax: Any,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    view: str,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    color: str,
) -> None:
    if coords.size == 0:
        return
    if coords.shape[0] > 600:
        _density_overlay(
            ax,
            coords,
            view=view,
            spacing=spacing,
            origin=origin,
            color=color,
            alpha=0.65,
        )
        _draw_representative_symbols(
            ax,
            coords,
            values,
            view=view,
            spacing=spacing,
            origin=origin,
            color=color,
        )
        return
    u, v = _project_points(coords, view=view, spacing=spacing, origin=origin)
    du, dv, out_of_plane = _project_vectors(coords[:, 3], values, view=view)
    markers = np.asarray(
        [
            _bc_marker(float(x), float(y), bool(out), float(value))
            for x, y, out, value in zip(du, dv, out_of_plane, values, strict=True)
        ]
    )
    for marker in sorted(set(str(value) for value in markers)):
        mask = markers == marker
        ax.scatter(
            u[mask],
            v[mask],
            s=26,
            c=color,
            marker=marker,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
            clip_on=False,
            zorder=4,
        )


def _density_overlay(
    ax: Any,
    coords: np.ndarray,
    *,
    view: str,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    color: str,
    alpha: float,
) -> None:
    u, v = _project_points(coords, view=view, spacing=spacing, origin=origin)
    extent = ax.images[0].get_extent() if ax.images else None
    if extent is None:
        return
    x_min, x_max, y_min, y_max = (float(value) for value in extent)
    x_bins = max(8, min(96, int(round(x_max - x_min + 1))))
    y_bins = max(8, min(96, int(round(y_max - y_min + 1))))
    hist, _, _ = np.histogram2d(
        v,
        u,
        bins=(y_bins, x_bins),
        range=((y_min, y_max), (x_min, x_max)),
    )
    if not np.any(hist):
        return
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    ax.contour(
        x_centers,
        y_centers,
        hist,
        levels=[0.5],
        colors=[color],
        linewidths=0.85,
        alpha=alpha,
        zorder=3,
    )


def _draw_representative_symbols(
    ax: Any,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    view: str,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    color: str,
) -> None:
    if coords.size == 0:
        return
    dofs = coords[:, 3].astype(int)
    points = []
    for dof in sorted(set(int(v) for v in dofs)):
        mask = dofs == dof
        if not np.any(mask):
            continue
        u, v = _project_points(coords[mask], view=view, spacing=spacing, origin=origin)
        du, dv, out_of_plane = _project_vectors(
            coords[mask, 3], values[mask], view=view
        )
        if np.all(out_of_plane):
            marker = _bc_marker(0.0, 0.0, True, float(np.nanmedian(values[mask])))
            points.append(
                (
                    float(np.nanmedian(u)),
                    float(np.nanmedian(v)),
                    marker,
                )
            )
            continue
        in_plane = ~out_of_plane
        du_med = float(np.nanmedian(du[in_plane]))
        dv_med = float(np.nanmedian(dv[in_plane]))
        value_med = float(np.nanmedian(values[mask][in_plane]))
        points.append(
            (
                float(np.nanmedian(u[in_plane])),
                float(np.nanmedian(v[in_plane])),
                _bc_marker(du_med, dv_med, False, value_med),
            )
        )
    if not points:
        u, v = _project_points(coords, view=view, spacing=spacing, origin=origin)
        ax.scatter(
            [float(np.nanmedian(u))],
            [float(np.nanmedian(v))],
            s=36,
            c=color,
            marker="^",
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
        return
    for marker in sorted(set(point[2] for point in points)):
        group = np.asarray([point[:2] for point in points if point[2] == marker])
        ax.scatter(
            group[:, 0],
            group[:, 1],
            s=46,
            c=color,
            marker=marker,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )


def _bc_marker(du: float, dv: float, out_of_plane: bool, value: float) -> str:
    if out_of_plane:
        return "^" if value >= 0 else "v"
    if abs(du) >= abs(dv):
        return ">" if du >= 0 else "<"
    return "^" if dv >= 0 else "v"


def _project_points(
    coords: np.ndarray,
    *,
    view: str,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(origin, dtype=float) + (
        coords[:, :3].astype(float) - 0.5
    ) * np.asarray(spacing, dtype=float)
    if view == "axial":
        return xyz[:, 0], xyz[:, 1]
    if view == "sagittal":
        return xyz[:, 1], xyz[:, 2]
    if view == "coronal":
        return xyz[:, 0], xyz[:, 2]
    raise ValueError(f"unknown view {view!r}")


def _project_vectors(
    dofs: np.ndarray,
    values: np.ndarray,
    *,
    view: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.zeros((len(values), 3), dtype=float)
    vectors[np.arange(len(values)), dofs.astype(int)] = np.asarray(values, dtype=float)
    if view == "axial":
        return vectors[:, 0], vectors[:, 1], dofs.astype(int) == 2
    if view == "sagittal":
        return vectors[:, 1], vectors[:, 2], dofs.astype(int) == 0
    if view == "coronal":
        return vectors[:, 0], vectors[:, 2], dofs.astype(int) == 1
    raise ValueError(f"unknown view {view!r}")


def _display_vector_scale(du: np.ndarray, dv: np.ndarray) -> float:
    magnitude = np.hypot(du, dv)
    finite = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
    if finite.size == 0:
        return 1.0
    return 0.55 / float(np.nanpercentile(finite, 95))


def _panel_vector_scale(ax: Any, du: np.ndarray, dv: np.ndarray) -> float:
    magnitude = np.hypot(du, dv)
    finite = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
    if finite.size == 0:
        return 1.0
    extent = ax.images[0].get_extent() if ax.images else None
    if extent is None:
        target_length = 1.0
    else:
        x_min, x_max, y_min, y_max = (float(value) for value in extent)
        target_length = max(1.0, min(abs(x_max - x_min), abs(y_max - y_min)) * 0.09)
    return target_length / float(np.nanpercentile(finite, 95))


def _finite_range(
    array: np.ndarray | None, *, robust_nonzero: bool = False
) -> tuple[float, float]:
    if array is None:
        return 0.0, 1.0
    values = np.asarray(array, dtype=float)
    finite = values[np.isfinite(values)]
    if robust_nonzero:
        nonzero = finite[np.abs(finite) > 0]
        if nonzero.size:
            finite = nonzero
    if finite.size == 0:
        return 0.0, 1.0
    if robust_nonzero and finite.size >= 20:
        vmin, vmax = (float(v) for v in np.nanpercentile(finite, (1.0, 99.0)))
    else:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if np.isclose(vmin, vmax):
        return vmin - 0.5, vmax + 0.5
    return vmin, vmax
