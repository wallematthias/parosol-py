from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True, slots=True)
class FilledSegmentationSplit:
    full_zyx: np.ndarray
    trabecular_zyx: np.ndarray
    cortical_zyx: np.ndarray
    erosion_iterations: int


def split_filled_segmentation_compartments(
    segmentation_zyx: np.ndarray,
    *,
    erosion_iterations: int = 2,
) -> FilledSegmentationSplit:
    segmentation = np.asarray(segmentation_zyx) > 0
    if segmentation.ndim != 3:
        raise ValueError("segmentation_zyx must be a 3D array")
    iterations = int(erosion_iterations)
    if iterations < 0:
        raise ValueError("erosion_iterations must be non-negative")

    full = np.zeros_like(segmentation, dtype=bool)
    trabecular = np.zeros_like(segmentation, dtype=bool)
    structure = ndi.generate_binary_structure(2, 1)

    for z_index in range(segmentation.shape[0]):
        filled_slice = ndi.binary_fill_holes(segmentation[z_index])
        full[z_index] = filled_slice
        if iterations == 0:
            trabecular[z_index] = filled_slice
        else:
            trabecular[z_index] = ndi.binary_erosion(
                filled_slice,
                structure=structure,
                iterations=iterations,
                border_value=0,
            )

    cortical = full & ~trabecular
    return FilledSegmentationSplit(
        full_zyx=np.ascontiguousarray(full),
        trabecular_zyx=np.ascontiguousarray(trabecular),
        cortical_zyx=np.ascontiguousarray(cortical),
        erosion_iterations=iterations,
    )
