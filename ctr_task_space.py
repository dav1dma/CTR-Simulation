"""Frozen baseline task regions and spatially stratified canonical IK targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from ctr_spatial_analysis import (
    AxisymmetricGrid,
    AxisymmetricWorkspaceMap,
    aggregate_axisymmetric_workspace,
    axisymmetric_cell_indices,
)


def _sha256_parts(metadata: dict, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenTaskRegion:
    """The occupied baseline inner-tip cells used for every design."""

    task_region_id: str
    grid: AxisymmetricGrid
    target_tube: int
    occupied: np.ndarray
    discovery_count_per_cell: np.ndarray
    cell_volume_mm3: np.ndarray
    baseline_parameter_sha256: str
    source_dataset_id: str
    minimum_occupancy_samples: int
    occupancy_rule: str = "independent-canonical-sample-count"

    @property
    def occupied_cell_count(self) -> int:
        return int(np.count_nonzero(self.occupied))

    @property
    def occupied_volume_mm3(self) -> float:
        return float(np.sum(self.cell_volume_mm3[self.occupied]))


def freeze_baseline_task_region(
    points_mm: np.ndarray,
    canonical_sample_ids: np.ndarray,
    *,
    target_tube: int,
    baseline_parameter_sha256: str,
    source_dataset_id: str,
    grid: AxisymmetricGrid | None = None,
    radial_bin_mm: float = 10.0,
    z_bin_mm: float = 10.0,
    minimum_occupancy_samples: int = 1,
) -> FrozenTaskRegion:
    """Freeze empirically occupied baseline cells without filling voids."""
    tube = int(target_tube)
    if tube not in (0, 1, 2):
        raise ValueError("target_tube must be 0, 1, or 2")
    parameter_hash = str(baseline_parameter_sha256).strip().lower()
    dataset_id = str(source_dataset_id).strip()
    if len(parameter_hash) != 64 or any(
        character not in "0123456789abcdef" for character in parameter_hash
    ):
        raise ValueError("baseline_parameter_sha256 must be a SHA-256 hex digest")
    if not dataset_id:
        raise ValueError("source_dataset_id is required")
    workspace = aggregate_axisymmetric_workspace(
        points_mm,
        canonical_sample_ids=canonical_sample_ids,
        grid=grid,
        radial_bin_mm=radial_bin_mm,
        z_bin_mm=z_bin_mm,
        minimum_occupancy_samples=minimum_occupancy_samples,
    )
    metadata = {
        "schema": "ctr-frozen-task-region-v1",
        "target_tube": tube,
        "baseline_parameter_sha256": parameter_hash,
        "source_dataset_id": dataset_id,
        "minimum_occupancy_samples": int(minimum_occupancy_samples),
        "occupancy_rule": "independent-canonical-sample-count",
    }
    region_hash = _sha256_parts(
        metadata,
        np.asarray(workspace.grid.radial_edges_mm, dtype="<f8"),
        np.asarray(workspace.grid.z_edges_mm, dtype="<f8"),
        np.asarray(workspace.occupied, dtype=np.uint8),
    )
    return FrozenTaskRegion(
        task_region_id=f"ctr-task-region-v1-{region_hash[:16]}",
        grid=workspace.grid,
        target_tube=tube,
        occupied=workspace.occupied.copy(),
        discovery_count_per_cell=workspace.sample_count.copy(),
        cell_volume_mm3=workspace.grid.cell_volume_mm3.copy(),
        baseline_parameter_sha256=parameter_hash,
        source_dataset_id=dataset_id,
        minimum_occupancy_samples=int(minimum_occupancy_samples),
    )


@dataclass(frozen=True)
class StratifiedCanonicalTargetSet:
    """Known-reachable targets distributed approximately equally by task cell."""

    target_set_id: str
    target_ids: np.ndarray
    targets_mm: np.ndarray
    source_sample_ids: np.ndarray
    source_deployment_m: np.ndarray
    source_rotation_rad: np.ndarray
    radial_cell_index: np.ndarray
    z_cell_index: np.ndarray
    target_weight_mm3: np.ndarray
    targets_per_cell: np.ndarray
    task_region_id: str
    split: str
    seed: int
    source_dataset_id: str
    selection_method: str = (
        "round-robin-cells-nearest-unused-volume-uniform-proposals"
    )

    @property
    def independent_target_count(self) -> int:
        return len(self.target_ids)

    @property
    def represented_volume_mm3(self) -> float:
        return float(np.sum(self.target_weight_mm3))


def _canonicalise_states(
    points_mm: np.ndarray,
    rotations_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_mm, dtype=float)
    rotations = np.asarray(rotations_rad, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or rotations.shape != points.shape:
        raise ValueError("points_mm and rotations_rad must have shape (N, 3)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(rotations)):
        raise ValueError("candidate positions and rotations must be finite")
    radius = np.hypot(points[:, 0], points[:, 1])
    azimuth = np.where(radius > 1e-12, np.arctan2(points[:, 1], points[:, 0]), 0.0)
    canonical_points = np.column_stack((radius, np.zeros(len(points)), points[:, 2]))
    canonical_rotations = (
        rotations + azimuth[:, None] + np.pi
    ) % (2.0 * np.pi) - np.pi
    return canonical_points, canonical_rotations


def _cell_seed(seed: int, task_region_id: str, cell: tuple[int, int]) -> int:
    payload = f"{int(seed)}|{task_region_id}|{cell[0]}|{cell[1]}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _proposal_order_for_cell(
    candidate_rows: np.ndarray,
    canonical_points_mm: np.ndarray,
    source_ids: np.ndarray,
    grid: AxisymmetricGrid,
    cell: tuple[int, int],
    *,
    seed: int,
    task_region_id: str,
) -> np.ndarray:
    """Match volume-uniform proposals to nearest unused reachable candidates."""
    rows = np.asarray(candidate_rows, dtype=np.int64)
    radial_index, z_index = cell
    r0, r1 = grid.radial_edges_mm[radial_index : radial_index + 2]
    z0, z1 = grid.z_edges_mm[z_index : z_index + 2]
    radius = canonical_points_mm[rows, 0]
    z_values = canonical_points_mm[rows, 2]
    normalised = np.column_stack(
        (
            (radius**2 - r0**2) / max(r1**2 - r0**2, 1e-15),
            (z_values - z0) / max(z1 - z0, 1e-15),
        )
    )
    rng = np.random.default_rng(_cell_seed(seed, task_region_id, cell))
    available = np.arange(len(rows), dtype=np.int64)
    ordered: list[int] = []
    for _ in range(len(rows)):
        proposal = rng.random(2)
        distance = np.sum((normalised[available] - proposal) ** 2, axis=1)
        # The source ID provides a deterministic tie-break independent of row order.
        ordering = np.lexsort((source_ids[rows[available]].astype(str), distance))
        selected_position = int(ordering[0])
        selected_local = int(available[selected_position])
        ordered.append(int(rows[selected_local]))
        available = np.delete(available, selected_position)
    return np.asarray(ordered, dtype=np.int64)


def select_stratified_canonical_targets(
    task_region: FrozenTaskRegion,
    candidate_points_mm: np.ndarray,
    candidate_deployment_m: np.ndarray,
    candidate_rotation_rad: np.ndarray,
    candidate_sample_ids: np.ndarray,
    *,
    target_count: int,
    split: str,
    seed: int,
    source_dataset_id: str,
    spatial_deduplication_mm: float = 1e-6,
) -> StratifiedCanonicalTargetSet:
    """Select actual baseline FK points without configuration-density weighting.

    The source actuator states prove that every selected point is reachable by
    the ideal baseline model. They are provenance only and must not be used as
    IK initial states; formal IK still begins fully retracted at zero rotation.
    """
    points = np.asarray(candidate_points_mm, dtype=float)
    deployment = np.asarray(candidate_deployment_m, dtype=float)
    rotation = np.asarray(candidate_rotation_rad, dtype=float)
    source_ids = np.asarray(candidate_sample_ids)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("candidate_points_mm must have shape (N, 3)")
    if deployment.shape != points.shape or rotation.shape != points.shape:
        raise ValueError("candidate actuator arrays must have shape (N, 3)")
    if source_ids.shape != (len(points),) or len(np.unique(source_ids)) != len(points):
        raise ValueError("candidate sample IDs must be unique with shape (N,)")
    if not np.all(np.isfinite(deployment)):
        raise ValueError("candidate deployment states must be finite")
    source_name = str(source_dataset_id).strip()
    split_name = str(split).strip().lower()
    if not source_name or not split_name:
        raise ValueError("source_dataset_id and split are required")
    if source_name == task_region.source_dataset_id:
        raise ValueError(
            "task discovery and target-candidate data must be independent datasets"
        )
    requested = int(target_count)
    if requested < 1:
        raise ValueError("target_count must be positive")
    tolerance = float(spatial_deduplication_mm)
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("spatial_deduplication_mm must be finite and positive")

    canonical_points, canonical_rotation = _canonicalise_states(points, rotation)
    cell_indices, inside = axisymmetric_cell_indices(
        canonical_points,
        task_region.grid,
    )
    valid_rows = np.flatnonzero(inside)
    valid_rows = valid_rows[
        task_region.occupied[
            cell_indices[valid_rows, 0],
            cell_indices[valid_rows, 1],
        ]
    ]
    if not len(valid_rows):
        raise ValueError("no candidate target lies in the frozen task region")

    # Remove effectively identical spatial targets independently of source-row
    # ordering. Keep the lexically smallest source ID for each coordinate.
    sorted_rows = valid_rows[np.argsort(source_ids[valid_rows].astype(str))]
    coordinate_keys = np.rint(
        canonical_points[sorted_rows][:, (0, 2)] / tolerance
    ).astype(np.int64)
    _, first_coordinate = np.unique(coordinate_keys, axis=0, return_index=True)
    valid_rows = sorted_rows[np.sort(first_coordinate)]
    if requested > len(valid_rows):
        raise ValueError(
            "target_count exceeds the number of unique reachable candidates "
            "inside the frozen task region"
        )

    groups: dict[tuple[int, int], np.ndarray] = {}
    for radial_value, z_value in np.unique(cell_indices[valid_rows], axis=0):
        cell = (int(radial_value), int(z_value))
        rows = valid_rows[
            (cell_indices[valid_rows, 0] == cell[0])
            & (cell_indices[valid_rows, 1] == cell[1])
        ]
        groups[cell] = _proposal_order_for_cell(
            rows,
            canonical_points,
            source_ids,
            task_region.grid,
            cell,
            seed=int(seed),
            task_region_id=task_region.task_region_id,
        )

    # A stable hash ordering avoids always favouring low-radius/low-Z cells when
    # a requested prefix ends part-way through a round.
    cells = sorted(
        groups,
        key=lambda cell: _cell_seed(int(seed), task_region.task_region_id, cell),
    )
    cursors = {cell: 0 for cell in cells}
    active = cells.copy()
    chosen: list[int] = []
    while active and len(chosen) < requested:
        next_active: list[tuple[int, int]] = []
        for cell in active:
            cursor = cursors[cell]
            queue = groups[cell]
            if cursor < len(queue) and len(chosen) < requested:
                chosen.append(int(queue[cursor]))
                cursor += 1
                cursors[cell] = cursor
            if cursor < len(queue):
                next_active.append(cell)
        active = next_active
    if len(chosen) != requested:
        raise RuntimeError("unable to allocate the requested stratified targets")

    rows = np.asarray(chosen, dtype=np.int64)
    selected_cells = cell_indices[rows]
    counts = np.zeros(task_region.grid.shape, dtype=np.int32)
    np.add.at(counts, (selected_cells[:, 0], selected_cells[:, 1]), 1)
    weights = np.asarray(
        [
            task_region.cell_volume_mm3[radial_value, z_value]
            / counts[radial_value, z_value]
            for radial_value, z_value in selected_cells
        ],
        dtype=float,
    )
    namespace = hashlib.sha256(
        (
            f"ctr-stratified-targets-v1|{task_region.task_region_id}|"
            f"{split_name}|{int(seed)}|{source_name}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    target_ids = np.asarray(
        [f"{namespace}:{index:010d}" for index in range(requested)],
        dtype="U23",
    )
    metadata = {
        "schema": "ctr-stratified-target-set-v1",
        "task_region_id": task_region.task_region_id,
        "split": split_name,
        "seed": int(seed),
        "source_dataset_id": source_name,
        "selection_method": (
            "round-robin-cells-nearest-unused-volume-uniform-proposals"
        ),
    }
    target_hash = _sha256_parts(
        metadata,
        np.asarray(target_ids, dtype="S23"),
        np.asarray(canonical_points[rows], dtype="<f8"),
        np.asarray(source_ids[rows].astype(str), dtype="S64"),
        np.asarray(selected_cells, dtype="<i8"),
    )
    return StratifiedCanonicalTargetSet(
        target_set_id=f"ctr-target-set-v1-{target_hash[:16]}",
        target_ids=target_ids,
        targets_mm=canonical_points[rows].astype(np.float32),
        source_sample_ids=source_ids[rows].copy(),
        source_deployment_m=deployment[rows].astype(np.float32),
        source_rotation_rad=canonical_rotation[rows].astype(np.float32),
        radial_cell_index=selected_cells[:, 0].astype(np.int32),
        z_cell_index=selected_cells[:, 1].astype(np.int32),
        target_weight_mm3=weights,
        targets_per_cell=counts,
        task_region_id=task_region.task_region_id,
        split=split_name,
        seed=int(seed),
        source_dataset_id=source_name,
    )
