"""Pure NumPy global shape descriptors for categorical wafer maps."""

from __future__ import annotations

import math

import numpy as np


def _validated_map(wafer: np.ndarray) -> np.ndarray:
    array = np.asarray(wafer)
    if array.ndim != 2 or min(array.shape) < 4:
        raise ValueError("wafer map must be a two-dimensional array of at least 4x4")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("wafer map must contain finite numeric values")
    rounded = np.rint(array)
    if not np.allclose(array, rounded) or not set(np.unique(rounded)).issubset({0, 1, 2}):
        raise ValueError("wafer map values must be 0, 1, or 2")
    return rounded.astype(np.uint8)


def _component_sizes(mask: np.ndarray) -> list[int]:
    """Return 8-connected component sizes for a boolean mask."""
    visited = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    height, width = mask.shape
    for row, column in np.argwhere(mask):
        row = int(row)
        column = int(column)
        if visited[row, column]:
            continue
        visited[row, column] = True
        stack = [(row, column)]
        size = 0
        while stack:
            current_row, current_column = stack.pop()
            size += 1
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    neighbor_row = current_row + row_delta
                    neighbor_column = current_column + column_delta
                    if not (0 <= neighbor_row < height and 0 <= neighbor_column < width):
                        continue
                    if mask[neighbor_row, neighbor_column] and not visited[neighbor_row, neighbor_column]:
                        visited[neighbor_row, neighbor_column] = True
                        stack.append((neighbor_row, neighbor_column))
        sizes.append(size)
    return sizes


def wafer_shape_features(wafer: np.ndarray) -> dict[str, float | int]:
    """Measure global defect area, connectedness, radial position, and spread.

    Values 1 and 2 are active die and value 2 is defective. Components use
    8-neighbor connectivity. Center and edge bands are normalized by the
    farthest active die from the active-die centroid.
    """
    array = _validated_map(wafer)
    active = array > 0
    defect = array == 2
    active_count = int(active.sum())
    defect_count = int(defect.sum())
    if active_count == 0:
        raise ValueError("wafer map must contain at least one active die")

    active_coordinates = np.argwhere(active).astype(float)
    active_center = active_coordinates.mean(axis=0)
    distances = np.linalg.norm(
        np.indices(array.shape).transpose(1, 2, 0) - active_center,
        axis=2,
    )
    active_radius = float(distances[active].max())
    normalized_radius = distances / max(active_radius, np.finfo(float).eps)

    padded = np.pad(active, 1, constant_values=False)
    interior = active & padded[:-2, 1:-1] & padded[2:, 1:-1]
    interior &= padded[1:-1, :-2] & padded[1:-1, 2:]
    boundary = active & ~interior
    boundary_count = int(boundary.sum())

    component_sizes = _component_sizes(defect)
    largest_component = max(component_sizes, default=0)
    active_rows, active_columns = np.where(active)
    active_bbox_area = int(
        (active_rows.max() - active_rows.min() + 1)
        * (active_columns.max() - active_columns.min() + 1)
    )

    if defect_count:
        defect_radii = normalized_radius[defect]
        defect_coordinates = np.argwhere(defect).astype(float)
        defect_center = defect_coordinates.mean(axis=0)
        centroid_offset = float(
            np.linalg.norm(defect_center - active_center)
            / max(active_radius, np.finfo(float).eps)
        )
        defect_rows, defect_columns = np.where(defect)
        defect_bbox_area = int(
            (defect_rows.max() - defect_rows.min() + 1)
            * (defect_columns.max() - defect_columns.min() + 1)
        )
        radial_counts, _ = np.histogram(
            defect_radii, bins=np.array([0.0, 0.25, 0.5, 0.75, 1.0000001])
        )
        probabilities = radial_counts[radial_counts > 0] / defect_count
        radial_entropy = float(
            -(probabilities * np.log(probabilities)).sum() / math.log(4)
        )
        center_fraction = float((defect_radii <= 0.35).mean())
        edge_fraction = float((defect_radii >= 0.70).mean())
        mean_radius = float(defect_radii.mean())
    else:
        defect_bbox_area = 0
        centroid_offset = 0.0
        radial_entropy = 0.0
        center_fraction = 0.0
        edge_fraction = 0.0
        mean_radius = 0.0

    return {
        "active_die_count": active_count,
        "defect_die_count": defect_count,
        "defect_ratio": defect_count / active_count,
        "component_count_8": len(component_sizes),
        "largest_component_fraction": largest_component / max(defect_count, 1),
        "mean_normalized_radius": mean_radius,
        "center_defect_fraction": center_fraction,
        "edge_defect_fraction": edge_fraction,
        "boundary_defect_coverage": int((defect & boundary).sum()) / max(boundary_count, 1),
        "defect_bbox_fraction": defect_bbox_area / active_bbox_area,
        "defect_centroid_offset": centroid_offset,
        "radial_entropy_4bin": radial_entropy,
    }
