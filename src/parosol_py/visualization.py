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
    boundary_conditions: BoundaryConditionSet | None = None,
    title: str | None = None,
) -> Path:
    """Write a compact material/result/boundary-condition overview PNG."""
    material = np.asarray(material_xyz)
    if material.ndim != 3:
        raise ValueError(f"material_xyz must be 3D, got shape {material.shape}")
    if field_xyz is not None:
        field = np.asarray(field_xyz)
        if field.shape != material.shape:
            raise ValueError(
                f"field_xyz shape {field.shape} does not match material {material.shape}"
            )
        field = np.where(material > 0, field, np.nan)
    else:
        field = None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    field_cmap = plt.get_cmap("jet").copy()
    field_cmap.set_bad(color="#f8fafc")

    slices = _mid_slices(material, spacing=spacing, origin=origin)
    field_slices = (
        _mid_slices(field, spacing=spacing, origin=origin) if field is not None else {}
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
        _style_axis(
            ax,
            f"{field_name}: {view_slice.label}",
            view_slice.xlabel,
            view_slice.ylabel,
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
                markersize=6,
                label="fixed",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#f97316",
                markersize=6,
                label="prescribed displacement",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#0891b2",
                markersize=6,
                label="applied force",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=3,
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
    fixed_slice = _slice_mask(fixed_coords, axis=slice_axis, index=slice_index)
    loaded_slice = _slice_mask(loaded_coords, axis=slice_axis, index=slice_index)
    fixed_coords = fixed_coords[fixed_slice]
    fixed_values = fixed_values[fixed_slice]
    loaded_coords = loaded_coords[loaded_slice]
    loaded_values = loaded_values[loaded_slice]
    zero_mask = np.isclose(fixed_values, 0.0) | np.isclose(fixed_values, 1e-16)
    _render_bc_group(
        ax,
        fixed_coords[zero_mask],
        values=None,
        view=view,
        spacing=spacing,
        origin=origin,
        color="#64748b",
        marker="s",
        alpha=0.18,
    )
    _draw_vectors(
        ax,
        fixed_coords[~zero_mask],
        fixed_values[~zero_mask],
        view=view,
        spacing=spacing,
        origin=origin,
        color="#f97316",
    )
    if loaded_coords.size:
        _draw_vectors(
            ax,
            loaded_coords,
            loaded_values,
            view=view,
            spacing=spacing,
            origin=origin,
            color="#0891b2",
        )


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
        _draw_representative_vectors(
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
    ax.scatter(u, v, s=8, c=color, marker=marker, alpha=alpha, linewidths=0.0)


def _draw_vectors(
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
        _draw_representative_vectors(
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
    in_plane = ~out_of_plane
    if np.any(in_plane):
        scale = _panel_vector_scale(ax, du[in_plane], dv[in_plane])
        ax.quiver(
            u[in_plane],
            v[in_plane],
            du[in_plane] * scale,
            dv[in_plane] * scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.008,
            color=color,
            alpha=0.86,
            zorder=4,
            clip_on=False,
        )
    if np.any(out_of_plane):
        ax.scatter(
            u[out_of_plane],
            v[out_of_plane],
            s=12,
            c=color,
            marker="o",
            alpha=0.55,
            edgecolors="white",
            linewidths=0.35,
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


def _draw_representative_vectors(
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
    vectors = []
    for dof in sorted(set(int(v) for v in dofs)):
        mask = dofs == dof
        if not np.any(mask):
            continue
        u, v = _project_points(coords[mask], view=view, spacing=spacing, origin=origin)
        du, dv, out_of_plane = _project_vectors(
            coords[mask, 3], values[mask], view=view
        )
        if np.all(out_of_plane):
            continue
        in_plane = ~out_of_plane
        vectors.append(
            (
                float(np.nanmedian(u[in_plane])),
                float(np.nanmedian(v[in_plane])),
                float(np.nanmedian(du[in_plane])),
                float(np.nanmedian(dv[in_plane])),
            )
        )
    if not vectors:
        u, v = _project_points(coords, view=view, spacing=spacing, origin=origin)
        ax.scatter(
            [float(np.nanmedian(u))],
            [float(np.nanmedian(v))],
            s=36,
            c=color,
            marker="o",
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
        return
    vector_array = np.asarray(vectors, dtype=float)
    scale = _panel_vector_scale(ax, vector_array[:, 2], vector_array[:, 3])
    ax.quiver(
        vector_array[:, 0],
        vector_array[:, 1],
        vector_array[:, 2] * scale,
        vector_array[:, 3] * scale,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.008,
        color=color,
        alpha=0.9,
        zorder=4,
    )


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
