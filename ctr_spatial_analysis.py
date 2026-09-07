"""Physical radial--vertical aggregation for protocol-v1 CTR analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_propagation, label as connected_components


def _points(values: np.ndarray, name: str = "points_mm") -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{name} must have non-empty shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _identifiers(values: np.ndarray | None, row_count: int) -> np.ndarray:
    if values is None:
        return np.arange(row_count, dtype=np.int64)
    identifiers = np.asarray(values)
    if identifiers.shape != (row_count,):
        raise ValueError("canonical_sample_ids must have shape (N,)")
    return identifiers


@dataclass(frozen=True)
class AxisymmetricGrid:
    """A radial/Z grid whose cells represent swept annular physical volumes."""

    radial_edges_mm: np.ndarray
    z_edges_mm: np.ndarray
    cell_volume_mm3: np.ndarray

    def __post_init__(self) -> None:
        radial = np.asarray(self.radial_edges_mm, dtype=float)
        z_values = np.asarray(self.z_edges_mm, dtype=float)
        volume = np.asarray(self.cell_volume_mm3, dtype=float)
        if radial.ndim != 1 or len(radial) < 2 or not np.all(np.diff(radial) > 0.0):
            raise ValueError("radial edges must be strictly increasing")
        if radial[0] < 0.0:
            raise ValueError("radial edges cannot be negative")
        if z_values.ndim != 1 or len(z_values) < 2 or not np.all(np.diff(z_values) > 0.0):
            raise ValueError("Z edges must be strictly increasing")
        expected_shape = (len(radial) - 1, len(z_values) - 1)
        if volume.shape != expected_shape or np.any(volume <= 0.0):
            raise ValueError("cell volumes must be positive and match grid shape")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.radial_edges_mm) - 1, len(self.z_edges_mm) - 1)

    @property
    def radial_centres_mm(self) -> np.ndarray:
        return 0.5 * (self.radial_edges_mm[:-1] + self.radial_edges_mm[1:])

    @property
    def z_centres_mm(self) -> np.ndarray:
        return 0.5 * (self.z_edges_mm[:-1] + self.z_edges_mm[1:])


def _aligned_edges(
    minimum: float,
    maximum: float,
    step: float,
    *,
    lower_bound: float | None = None,
) -> np.ndarray:
    lower = np.floor(minimum / step) * step
    if lower_bound is not None:
        lower = max(float(lower_bound), lower)
    upper = np.ceil(maximum / step) * step
    if np.isclose(maximum, upper, rtol=0.0, atol=1e-12):
        upper += step
    if upper <= lower:
        upper = lower + step
    return np.arange(lower, upper + 0.5 * step, step, dtype=float)


def build_axisymmetric_grid(
    point_sets_mm: Iterable[np.ndarray],
    *,
    radial_bin_mm: float = 10.0,
    z_bin_mm: float = 10.0,
) -> AxisymmetricGrid:
    """Build one common grid covering all supplied point sets."""
    radial_step = float(radial_bin_mm)
    z_step = float(z_bin_mm)
    if radial_step <= 0.0 or z_step <= 0.0:
        raise ValueError("radial and Z cell sizes must be positive")
    sets = [_points(values) for values in point_sets_mm]
    if not sets:
        raise ValueError("at least one point set is required")
    radius = np.concatenate([np.hypot(values[:, 0], values[:, 1]) for values in sets])
    z_values = np.concatenate([values[:, 2] for values in sets])
    radial_edges = _aligned_edges(
        0.0,
        float(np.max(radius)),
        radial_step,
        lower_bound=0.0,
    )
    z_edges = _aligned_edges(
        float(np.min(z_values)),
        float(np.max(z_values)),
        z_step,
    )
    annular_area = np.pi * (
        radial_edges[1:] ** 2 - radial_edges[:-1] ** 2
    )
    height = np.diff(z_edges)
    cell_volume = annular_area[:, None] * height[None, :]
    return AxisymmetricGrid(radial_edges, z_edges, cell_volume)


def _grid_indices(
    points_mm: np.ndarray,
    grid: AxisymmetricGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = _points(points_mm)
    radius = np.hypot(points[:, 0], points[:, 1])
    radial_index = np.searchsorted(grid.radial_edges_mm, radius, side="right") - 1
    z_index = np.searchsorted(grid.z_edges_mm, points[:, 2], side="right") - 1
    inside = (
        (radial_index >= 0)
        & (radial_index < grid.shape[0])
        & (z_index >= 0)
        & (z_index < grid.shape[1])
    )
    return radial_index, z_index, inside


def axisymmetric_cell_indices(
    points_mm: np.ndarray,
    grid: AxisymmetricGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Return global radial/Z cell indices and an inside-grid mask."""
    radial_index, z_index, inside = _grid_indices(points_mm, grid)
    return np.column_stack((radial_index, z_index)).astype(np.int64), inside


def _canonical_representative_rows(
    points_mm: np.ndarray,
    identifiers: np.ndarray,
    values: np.ndarray | None = None,
) -> np.ndarray:
    """Globally reduce render copies to one row per canonical sample.

    Repeated display copies may differ in X/Y azimuth, but must have the same
    radius, Z coordinate and scalar result. This check is deliberately applied
    before binning so floating-point boundary jitter cannot count one source in
    two cells.
    """
    points = _points(points_mm)
    ids = _identifiers(identifiers, len(points))
    radius_z = np.column_stack(
        (np.hypot(points[:, 0], points[:, 1]), points[:, 2])
    )
    scalar = None if values is None else np.asarray(values, dtype=float).reshape(-1)
    if scalar is not None and scalar.shape != (len(points),):
        raise ValueError("values must contain one scalar per point")
    _, first, inverse, counts = np.unique(
        ids,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    if len(first) == len(ids):
        return np.arange(len(ids), dtype=np.int64)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    for representative_row, start, count in zip(first, starts, counts):
        rows = order[start : start + count]
        if not np.allclose(
            radius_z[rows],
            radius_z[representative_row],
            rtol=1e-7,
            atol=1e-6,
        ):
            raise ValueError(
                "one canonical sample maps to conflicting radial/Z positions"
            )
        if scalar is not None and not np.allclose(
            scalar[rows],
            scalar[representative_row],
            rtol=1e-6,
            atol=1e-9,
            equal_nan=True,
        ):
            raise ValueError(
                "one canonical sample has conflicting repeated scalar values"
            )
    return np.sort(first.astype(np.int64))


def enclosed_internal_void_mask(occupied: np.ndarray) -> np.ndarray:
    """Return empty cells disconnected from the exterior of an r/Z section.

    The Z boundaries and maximum-radius boundary connect to exterior space. The
    r=0 axis is not an exterior boundary, so an axial cavity enclosed radially
    and vertically can be identified correctly.
    """
    mask = np.asarray(occupied, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("occupied must be a non-empty 2D mask")
    # Add exterior padding at max radius and both Z faces, but deliberately not
    # at r=0. This allows an axially centred, radially and vertically capped
    # cavity to remain classified as enclosed.
    padded = np.pad(mask, ((0, 1), (1, 1)), mode="constant", constant_values=False)
    empty = ~padded
    exterior_seed = np.zeros_like(empty)
    exterior_seed[-1, :] = empty[-1, :]
    exterior_seed[:, 0] = empty[:, 0]
    exterior_seed[:, -1] = empty[:, -1]
    exterior = binary_propagation(exterior_seed, mask=empty)
    void = empty & ~exterior
    return void[:-1, 1:-1]


@dataclass(frozen=True)
class AxisymmetricWorkspaceMap:
    grid: AxisymmetricGrid
    sample_count: np.ndarray
    occupied: np.ndarray
    internal_void: np.ndarray
    minimum_occupancy_samples: int
    independent_sample_count: int

    @property
    def occupied_volume_mm3(self) -> float:
        return float(np.sum(self.grid.cell_volume_mm3[self.occupied]))

    @property
    def internal_void_volume_mm3(self) -> float:
        return float(np.sum(self.grid.cell_volume_mm3[self.internal_void]))

    @property
    def occupied_or_internal_volume_mm3(self) -> float:
        return self.occupied_volume_mm3 + self.internal_void_volume_mm3

    @property
    def internal_void_component_count(self) -> int:
        return int(connected_components(self.internal_void)[1])

    @property
    def largest_internal_void_volume_mm3(self) -> float:
        labels, count = connected_components(self.internal_void)
        if count == 0:
            return 0.0
        return float(
            max(
                np.sum(self.grid.cell_volume_mm3[labels == component])
                for component in range(1, count + 1)
            )
        )

    @property
    def occupied_component_count(self) -> int:
        return int(connected_components(self.occupied)[1])


def aggregate_axisymmetric_workspace(
    points_mm: np.ndarray,
    *,
    canonical_sample_ids: np.ndarray | None = None,
    grid: AxisymmetricGrid | None = None,
    radial_bin_mm: float = 10.0,
    z_bin_mm: float = 10.0,
    minimum_occupancy_samples: int = 1,
) -> AxisymmetricWorkspaceMap:
    """Estimate sampled occupied swept volume without filling internal voids."""
    points = _points(points_mm)
    identifiers = _identifiers(canonical_sample_ids, len(points))
    representative_rows = _canonical_representative_rows(points, identifiers)
    points = points[representative_rows]
    identifiers = identifiers[representative_rows]
    minimum = int(minimum_occupancy_samples)
    if minimum < 1:
        raise ValueError("minimum_occupancy_samples must be positive")
    selected_grid = grid or build_axisymmetric_grid(
        [points],
        radial_bin_mm=radial_bin_mm,
        z_bin_mm=z_bin_mm,
    )
    radial_index, z_index, inside = _grid_indices(points, selected_grid)
    counts = np.zeros(selected_grid.shape, dtype=np.int32)
    np.add.at(counts, (radial_index[inside], z_index[inside]), 1)
    occupied = counts >= minimum
    internal_void = enclosed_internal_void_mask(occupied)
    return AxisymmetricWorkspaceMap(
        grid=selected_grid,
        sample_count=counts,
        occupied=occupied,
        internal_void=internal_void,
        minimum_occupancy_samples=minimum,
        independent_sample_count=len(np.unique(identifiers)),
    )


@dataclass(frozen=True)
class WorkspaceCoverageComparison:
    reference: AxisymmetricWorkspaceMap
    candidate: AxisymmetricWorkspaceMap
    shared_volume_mm3: float
    lost_reference_volume_mm3: float
    gained_candidate_volume_mm3: float
    reference_coverage_fraction: float
    reference_volume_mm3: float
    candidate_volume_mm3: float
    volume_weighted_jaccard: float


def compare_axisymmetric_workspaces(
    reference_points_mm: np.ndarray,
    candidate_points_mm: np.ndarray,
    *,
    reference_sample_ids: np.ndarray | None = None,
    candidate_sample_ids: np.ndarray | None = None,
    radial_bin_mm: float = 10.0,
    z_bin_mm: float = 10.0,
    minimum_occupancy_samples: int = 1,
) -> WorkspaceCoverageComparison:
    """Compare both designs on one grid while freezing baseline occupied cells."""
    reference_points = _points(reference_points_mm, "reference_points_mm")
    candidate_points = _points(candidate_points_mm, "candidate_points_mm")
    grid = build_axisymmetric_grid(
        [reference_points, candidate_points],
        radial_bin_mm=radial_bin_mm,
        z_bin_mm=z_bin_mm,
    )
    reference = aggregate_axisymmetric_workspace(
        reference_points,
        canonical_sample_ids=reference_sample_ids,
        grid=grid,
        minimum_occupancy_samples=minimum_occupancy_samples,
    )
    candidate = aggregate_axisymmetric_workspace(
        candidate_points,
        canonical_sample_ids=candidate_sample_ids,
        grid=grid,
        minimum_occupancy_samples=minimum_occupancy_samples,
    )
    volume = grid.cell_volume_mm3
    shared = float(np.sum(volume[reference.occupied & candidate.occupied]))
    lost = float(np.sum(volume[reference.occupied & ~candidate.occupied]))
    gained = float(np.sum(volume[~reference.occupied & candidate.occupied]))
    reference_volume = reference.occupied_volume_mm3
    candidate_volume = candidate.occupied_volume_mm3
    coverage = shared / reference_volume if reference_volume > 0.0 else float("nan")
    union = shared + lost + gained
    return WorkspaceCoverageComparison(
        reference=reference,
        candidate=candidate,
        shared_volume_mm3=shared,
        lost_reference_volume_mm3=lost,
        gained_candidate_volume_mm3=gained,
        reference_coverage_fraction=float(coverage),
        reference_volume_mm3=reference_volume,
        candidate_volume_mm3=candidate_volume,
        volume_weighted_jaccard=(shared / union if union > 0.0 else float("nan")),
    )


@dataclass(frozen=True)
class AxisymmetricScalarMap:
    """Unique-canonical scalar statistics on physical swept-volume cells."""

    grid: AxisymmetricGrid
    maximum: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    lower_p10: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    iqr: np.ndarray
    sample_count: np.ndarray
    minimum_median_samples: int
    minimum_lower_tail_samples: int


@dataclass(frozen=True)
class AxisymmetricResidualMap:
    """Numerical IK residual and failure statistics on radial/Z cells."""

    grid: AxisymmetricGrid
    maximum_residual_mm: np.ndarray
    median_residual_mm: np.ndarray
    p95_residual_mm: np.ndarray
    failure_fraction: np.ndarray
    sample_count: np.ndarray
    evaluation_threshold_mm: float
    minimum_median_samples: int
    minimum_failure_or_p95_samples: int


def aggregate_axisymmetric_scalar(
    points_mm: np.ndarray,
    values: np.ndarray,
    *,
    canonical_sample_ids: np.ndarray | None = None,
    grid: AxisymmetricGrid | None = None,
    radial_bin_mm: float = 10.0,
    z_bin_mm: float = 10.0,
    minimum_median_samples: int = 30,
    minimum_lower_tail_samples: int = 50,
) -> AxisymmetricScalarMap:
    """Aggregate a configuration metric before any angular display sweep."""
    points = _points(points_mm)
    scalar = np.asarray(values, dtype=float).reshape(-1)
    if scalar.shape != (len(points),) or np.any(np.isinf(scalar)):
        raise ValueError("values must contain one finite-or-NaN scalar per point")
    identifiers = _identifiers(canonical_sample_ids, len(points))
    representative_rows = _canonical_representative_rows(
        points,
        identifiers,
        scalar,
    )
    points = points[representative_rows]
    scalar = scalar[representative_rows]
    identifiers = identifiers[representative_rows]
    median_minimum = int(minimum_median_samples)
    tail_minimum = int(minimum_lower_tail_samples)
    if median_minimum < 1 or tail_minimum < median_minimum:
        raise ValueError("lower-tail support must be at least the median support")
    selected_grid = grid or build_axisymmetric_grid(
        [points],
        radial_bin_mm=radial_bin_mm,
        z_bin_mm=z_bin_mm,
    )
    shape = selected_grid.shape
    maximum = np.full(shape, np.nan, dtype=np.float32)
    mean = np.full(shape, np.nan, dtype=np.float32)
    median = np.full(shape, np.nan, dtype=np.float32)
    lower_p10 = np.full(shape, np.nan, dtype=np.float32)
    q25 = np.full(shape, np.nan, dtype=np.float32)
    q75 = np.full(shape, np.nan, dtype=np.float32)
    iqr = np.full(shape, np.nan, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.int32)
    radial_index, z_index, inside = _grid_indices(points, selected_grid)
    inside_rows = np.flatnonzero(inside)
    linear = radial_index[inside] * shape[1] + z_index[inside]
    order = np.argsort(linear, kind="stable")
    sorted_linear = linear[order]
    unique_linear, starts, group_counts = np.unique(
        sorted_linear,
        return_index=True,
        return_counts=True,
    )
    for linear_value, start, group_count in zip(
        unique_linear, starts, group_counts
    ):
        radial_value, z_value = divmod(int(linear_value), shape[1])
        rows = inside_rows[order[start : start + group_count]]
        independent = scalar[rows]
        independent = independent[np.isfinite(independent)]
        count = len(independent)
        if count == 0:
            continue
        counts[radial_value, z_value] = count
        maximum[radial_value, z_value] = np.max(independent)
        if count >= median_minimum:
            mean[radial_value, z_value] = np.mean(independent)
            median[radial_value, z_value] = np.median(independent)
            q25[radial_value, z_value] = np.quantile(independent, 0.25)
            q75[radial_value, z_value] = np.quantile(independent, 0.75)
            iqr[radial_value, z_value] = (
                q75[radial_value, z_value] - q25[radial_value, z_value]
            )
        if count >= tail_minimum:
            lower_p10[radial_value, z_value] = np.quantile(independent, 0.10)
    return AxisymmetricScalarMap(
        grid=selected_grid,
        maximum=maximum,
        mean=mean,
        median=median,
        lower_p10=lower_p10,
        q25=q25,
        q75=q75,
        iqr=iqr,
        sample_count=counts,
        minimum_median_samples=median_minimum,
        minimum_lower_tail_samples=tail_minimum,
    )


def aggregate_axisymmetric_residual(
    targets_mm: np.ndarray,
    residual_mm: np.ndarray,
    *,
    canonical_target_ids: np.ndarray | None = None,
    grid: AxisymmetricGrid | None = None,
    radial_bin_mm: float = 10.0,
    z_bin_mm: float = 10.0,
    evaluation_threshold_mm: float = 0.5,
    minimum_median_samples: int = 30,
    minimum_failure_or_p95_samples: int = 100,
) -> AxisymmetricResidualMap:
    """Aggregate continuous residuals and threshold failures independently."""
    points = _points(targets_mm, "targets_mm")
    residual = np.asarray(residual_mm, dtype=float).reshape(-1)
    if residual.shape != (len(points),) or not np.all(np.isfinite(residual)):
        raise ValueError("residual_mm must contain one finite value per target")
    if np.any(residual < 0.0):
        raise ValueError("IK residuals cannot be negative")
    identifiers = _identifiers(canonical_target_ids, len(points))
    representative_rows = _canonical_representative_rows(
        points,
        identifiers,
        residual,
    )
    points = points[representative_rows]
    residual = residual[representative_rows]
    median_minimum = int(minimum_median_samples)
    tail_minimum = int(minimum_failure_or_p95_samples)
    threshold = float(evaluation_threshold_mm)
    if median_minimum < 1 or tail_minimum < median_minimum:
        raise ValueError("failure/p95 support must be at least median support")
    if threshold <= 0.0 or not np.isfinite(threshold):
        raise ValueError("evaluation threshold must be finite and positive")
    selected_grid = grid or build_axisymmetric_grid(
        [points],
        radial_bin_mm=radial_bin_mm,
        z_bin_mm=z_bin_mm,
    )
    shape = selected_grid.shape
    maximum = np.full(shape, np.nan, dtype=np.float32)
    median = np.full(shape, np.nan, dtype=np.float32)
    p95 = np.full(shape, np.nan, dtype=np.float32)
    failure = np.full(shape, np.nan, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.int32)
    radial_index, z_index, inside = _grid_indices(points, selected_grid)
    rows_inside = np.flatnonzero(inside)
    linear = radial_index[inside] * shape[1] + z_index[inside]
    order = np.argsort(linear, kind="stable")
    sorted_linear = linear[order]
    unique_linear, starts, group_counts = np.unique(
        sorted_linear,
        return_index=True,
        return_counts=True,
    )
    for linear_value, start, group_count in zip(
        unique_linear, starts, group_counts
    ):
        radial_value, z_value = divmod(int(linear_value), shape[1])
        rows = rows_inside[order[start : start + group_count]]
        values = residual[rows]
        count = len(values)
        counts[radial_value, z_value] = count
        maximum[radial_value, z_value] = np.max(values)
        if count >= median_minimum:
            median[radial_value, z_value] = np.median(values)
        if count >= tail_minimum:
            p95[radial_value, z_value] = np.quantile(values, 0.95)
            failure[radial_value, z_value] = np.mean(values > threshold)
    return AxisymmetricResidualMap(
        grid=selected_grid,
        maximum_residual_mm=maximum,
        median_residual_mm=median,
        p95_residual_mm=p95,
        failure_fraction=failure,
        sample_count=counts,
        evaluation_threshold_mm=threshold,
        minimum_median_samples=median_minimum,
        minimum_failure_or_p95_samples=tail_minimum,
    )


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    data = np.asarray(values, dtype=float).reshape(-1)
    mass = np.asarray(weights, dtype=float).reshape(-1)
    if data.shape != mass.shape or not len(data):
        raise ValueError("values and weights must be non-empty and have equal shape")
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(mass)):
        raise ValueError("values and weights must be finite")
    if np.any(mass < 0.0) or not np.sum(mass) > 0.0:
        raise ValueError("weights must be non-negative with positive total")
    probability = float(quantile)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    order = np.argsort(data)
    sorted_values = data[order]
    cumulative = np.cumsum(mass[order])
    threshold = probability * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(data) - 1)
    return float(sorted_values[index])


def summarise_axisymmetric_scalar(
    field: AxisymmetricScalarMap,
    *,
    task_workspace: AxisymmetricWorkspaceMap | None = None,
) -> dict[str, float]:
    """Summarise local medians using exact swept physical-volume weights."""
    if task_workspace is not None and (
        task_workspace.grid.shape != field.grid.shape
        or not np.array_equal(
            task_workspace.grid.radial_edges_mm,
            field.grid.radial_edges_mm,
        )
        or not np.array_equal(task_workspace.grid.z_edges_mm, field.grid.z_edges_mm)
    ):
        raise ValueError("task workspace and scalar field must use the same grid")
    valid = np.isfinite(field.median)
    task_mask = (
        field.sample_count > 0
        if task_workspace is None
        else np.asarray(task_workspace.occupied, dtype=bool)
    )
    valid &= task_mask
    if not np.any(valid):
        return {
            "volume_weighted_mean": float("nan"),
            "volume_weighted_median": float("nan"),
            "volume_weighted_p10": float("nan"),
            "volume_weighted_std": float("nan"),
            "reportable_volume_fraction": 0.0,
        }
    values = np.asarray(field.median[valid], dtype=float)
    weights = np.asarray(field.grid.cell_volume_mm3[valid], dtype=float)
    weighted_mean = float(np.average(values, weights=weights))
    weighted_variance = float(np.average((values - weighted_mean) ** 2, weights=weights))
    task_volume = float(np.sum(field.grid.cell_volume_mm3[task_mask]))
    reportable_volume = float(np.sum(weights))
    return {
        "volume_weighted_mean": weighted_mean,
        "volume_weighted_median": weighted_quantile(values, weights, 0.50),
        "volume_weighted_p10": weighted_quantile(values, weights, 0.10),
        "volume_weighted_std": float(np.sqrt(weighted_variance)),
        "reportable_volume_fraction": (
            reportable_volume / task_volume if task_volume > 0.0 else 0.0
        ),
    }


@dataclass(frozen=True)
class StratifiedSpatialSelection:
    row_indices: np.ndarray
    canonical_sample_ids: np.ndarray
    radial_cell_index: np.ndarray
    z_cell_index: np.ndarray


def stratified_spatial_selection(
    points_mm: np.ndarray,
    *,
    canonical_sample_ids: np.ndarray,
    target_count: int,
    seed: int,
    grid: AxisymmetricGrid | None = None,
    radial_bin_mm: float = 10.0,
    z_bin_mm: float = 10.0,
) -> StratifiedSpatialSelection:
    """Compatibility row selector for exploratory uses.

    Formal Stage-2 IK targets use ``ctr_task_space`` so the task region is
    frozen, discovery/candidate data are independent, spatial duplicates are
    removed and physical-volume target weights are retained.
    """
    points = _points(points_mm)
    identifiers = _identifiers(canonical_sample_ids, len(points))
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("target-source canonical IDs must be unique")
    requested = int(target_count)
    if requested < 1 or requested > len(points):
        raise ValueError("target_count must lie within the number of unique sources")
    selected_grid = grid or build_axisymmetric_grid(
        [points],
        radial_bin_mm=radial_bin_mm,
        z_bin_mm=z_bin_mm,
    )
    radial_index, z_index, inside = _grid_indices(points, selected_grid)
    rows_inside = np.flatnonzero(inside)
    if requested > len(rows_inside):
        raise ValueError("the requested target count exceeds sources inside the grid")
    groups: dict[tuple[int, int], list[int]] = {}
    for row in rows_inside:
        groups.setdefault((int(radial_index[row]), int(z_index[row])), []).append(int(row))
    rng = np.random.default_rng(int(seed))
    queues: dict[tuple[int, int], np.ndarray] = {}
    cursors: dict[tuple[int, int], int] = {}
    for cell, rows in groups.items():
        values = np.asarray(rows, dtype=np.int64)
        rng.shuffle(values)
        queues[cell] = values
        cursors[cell] = 0
    active = list(queues)
    chosen: list[int] = []
    while len(chosen) < requested and active:
        rng.shuffle(active)
        next_active = []
        for cell in active:
            cursor = cursors[cell]
            queue = queues[cell]
            if cursor < len(queue) and len(chosen) < requested:
                chosen.append(int(queue[cursor]))
                cursor += 1
                cursors[cell] = cursor
            if cursor < len(queue):
                next_active.append(cell)
        active = next_active
    if len(chosen) != requested:
        raise RuntimeError("unable to create the requested stratified target set")
    indices = np.asarray(chosen, dtype=np.int64)
    return StratifiedSpatialSelection(
        row_indices=indices,
        canonical_sample_ids=identifiers[indices].copy(),
        radial_cell_index=radial_index[indices].astype(np.int32),
        z_cell_index=z_index[indices].astype(np.int32),
    )
