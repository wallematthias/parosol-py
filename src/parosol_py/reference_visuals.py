from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reference_geometry import ImageGridMetadata


@dataclass(frozen=True, slots=True)
class VisualReport:
    html_path: Path
    png_paths: tuple[Path, ...]


def write_visual_report(
    output_dir: str | Path,
    *,
    fixture_name: str,
    grid: ImageGridMetadata,
    anatomy_zyx: np.ndarray,
    reference_labels_zyx: np.ndarray,
    replay_labels_zyx: np.ndarray,
    scalar_reference_zyx: np.ndarray | None = None,
    scalar_replay_zyx: np.ndarray | None = None,
) -> VisualReport:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    anatomy = np.asarray(anatomy_zyx)
    reference = np.asarray(reference_labels_zyx)
    replay = np.asarray(replay_labels_zyx)
    if anatomy.shape != reference.shape or reference.shape != replay.shape:
        raise ValueError("anatomy and label arrays must have matching shapes")

    z_index = _representative_slice(reference, replay)
    png_paths = [
        _write_side_by_side_png(
            out / "labels_side_by_side.png",
            anatomy[z_index],
            reference[z_index],
            replay[z_index],
            title=f"{fixture_name} labels z={z_index}",
        ),
        _write_difference_png(
            out / "label_difference.png",
            reference[z_index],
            replay[z_index],
            title=f"{fixture_name} label differences z={z_index}",
        ),
    ]
    if scalar_reference_zyx is not None and scalar_replay_zyx is not None:
        scalar_reference = np.asarray(scalar_reference_zyx)
        scalar_replay = np.asarray(scalar_replay_zyx)
        if scalar_reference.shape != reference.shape or scalar_replay.shape != reference.shape:
            raise ValueError("scalar arrays must match label array shape")
        png_paths.append(
            _write_scalar_difference_png(
                out / "scalar_difference.png",
                scalar_reference[z_index],
                scalar_replay[z_index],
                title=f"{fixture_name} scalar differences z={z_index}",
            )
        )
    plt.close("all")

    html_path = out / "visual_report.html"
    image_tags = "\n".join(
        f'<figure><img src="{path.name}" alt="{path.stem}"></figure>'
        for path in png_paths
    )
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '<meta charset="utf-8">',
                f"<title>{fixture_name} reference comparison</title>",
                "<style>body{font-family:sans-serif;margin:24px}"
                "img{max-width:100%;border:1px solid #bbb}"
                "figure{margin:0 0 20px}</style>",
                "</head>",
                "<body>",
                f"<h1>{fixture_name}</h1>",
                "<p>RAS-mm visual comparison report.</p>",
                f"<p>Grid shape z/y/x: {grid.shape_zyx}; spacing x/y/z: {grid.spacing_xyz}; origin RAS: {grid.origin_ras}</p>",
                image_tags,
                "</body>",
                "</html>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return VisualReport(html_path=html_path, png_paths=tuple(png_paths))


def _representative_slice(reference: np.ndarray, replay: np.ndarray) -> int:
    active = np.argwhere((np.asarray(reference) != 0) | (np.asarray(replay) != 0))
    if active.size == 0:
        return int(reference.shape[0] // 2)
    return int(np.median(active[:, 0]))


def _write_side_by_side_png(
    path: Path,
    anatomy_slice: np.ndarray,
    reference_slice: np.ndarray,
    replay_slice: np.ndarray,
    *,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, labels, name in zip(
        axes,
        (reference_slice, replay_slice),
        ("reference", "replay"),
        strict=True,
    ):
        axis.imshow(_normalized(anatomy_slice), cmap="gray", interpolation="nearest")
        overlay = np.ma.masked_where(labels == 0, labels)
        axis.imshow(overlay, cmap="autumn", alpha=0.55, interpolation="nearest")
        axis.set_title(name)
        axis.set_axis_off()
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _write_difference_png(
    path: Path,
    reference_slice: np.ndarray,
    replay_slice: np.ndarray,
    *,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    difference = np.asarray(reference_slice) != np.asarray(replay_slice)
    fig, axis = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    axis.imshow(difference, cmap="magma", interpolation="nearest")
    axis.set_title(title)
    axis.set_axis_off()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _write_scalar_difference_png(
    path: Path,
    reference_slice: np.ndarray,
    replay_slice: np.ndarray,
    *,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    difference = np.asarray(replay_slice, dtype=float) - np.asarray(reference_slice, dtype=float)
    limit = float(np.nanmax(np.abs(difference))) if np.size(difference) else 0.0
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    fig, axis = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    image = axis.imshow(
        difference,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_axis_off()
    fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _normalized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=float)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        return np.zeros(array.shape, dtype=float)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)
