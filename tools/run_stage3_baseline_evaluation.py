"""Run the protocol-v1 Stage-3 baseline convergence and figure evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_protocol import (  # noqa: E402
    baseline_parameter_hash,
    build_run_manifest,
    load_analysis_protocol,
    write_run_manifest,
)
from ctr_design_analysis import (  # noqa: E402
    IKErrorMap,
    baseline_design,
    calculate_ik_error_map,
    position_jacobians_all_endpoints,
    positional_isotropy,
    summarise_ik_errors,
)
from ctr_inverse_kinematics import ConstrainedTipIK  # noqa: E402
from ctr_sampling import CanonicalConfigurationBank, canonical_configuration_bank  # noqa: E402
from ctr_spatial_analysis import (  # noqa: E402
    AxisymmetricGrid,
    AxisymmetricResidualMap,
    AxisymmetricScalarMap,
    AxisymmetricWorkspaceMap,
    aggregate_axisymmetric_residual,
    aggregate_axisymmetric_scalar,
    aggregate_axisymmetric_workspace,
    build_axisymmetric_grid,
    compare_axisymmetric_workspaces,
    summarise_axisymmetric_scalar,
)
from ctr_task_space import (  # noqa: E402
    FrozenTaskRegion,
    StratifiedCanonicalTargetSet,
    freeze_baseline_task_region,
    select_stratified_canonical_targets,
)


TUBE_NAMES = ("inner", "middle", "outer")
_WORKER_SOLVER: ConstrainedTipIK | None = None
_WORKER_TRANSLATION_STEP_MM = 0.1
_WORKER_ROTATION_STEP_RAD = 0.001
_IK_PARAMETERS: dict | None = None
_IK_TARGET_TUBE = 0
_IK_SOLVER_TOLERANCE_MM = 0.01
_IK_MAX_ITERATIONS = 40


def _initialise_fk_worker(
    parameters: dict,
    translation_step_mm: float,
    rotation_step_rad: float,
) -> None:
    global _WORKER_SOLVER, _WORKER_TRANSLATION_STEP_MM, _WORKER_ROTATION_STEP_RAD
    _WORKER_SOLVER = ConstrainedTipIK(parameters, model_points_per_section=2)
    _WORKER_TRANSLATION_STEP_MM = float(translation_step_mm)
    _WORKER_ROTATION_STEP_RAD = float(rotation_step_rad)


def _jacobian_chunk(payload):
    start, deployment, rotation = payload
    if _WORKER_SOLVER is None:
        raise RuntimeError("forward-model worker was not initialised")
    count = len(deployment)
    positions = np.empty((count, 3, 3), dtype=np.float32)
    isotropy = np.full((count, 3), np.nan, dtype=np.float32)
    singular_values = np.full((count, 3, 3), np.nan, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    for index, (deployment_row, rotation_row) in enumerate(zip(deployment, rotation)):
        result = position_jacobians_all_endpoints(
            _WORKER_SOLVER,
            deployment_row,
            rotation_row,
            translation_step_mm=_WORKER_TRANSLATION_STEP_MM,
            rotation_step_rad=_WORKER_ROTATION_STEP_RAD,
        )
        positions[index] = result.base_endpoints_mm
        valid[index] = result.valid
        if result.valid:
            for endpoint in range(3):
                score, values = positional_isotropy(
                    result.scaled_jacobians[endpoint]
                )
                isotropy[index, endpoint] = score
                singular_values[index, endpoint] = values
    return start, positions, isotropy, singular_values, valid


def _forward_chunk(payload):
    start, deployment, rotation = payload
    if _WORKER_SOLVER is None:
        raise RuntimeError("forward-model worker was not initialised")
    positions = np.empty((len(deployment), 3, 3), dtype=np.float32)
    for index, (deployment_row, rotation_row) in enumerate(zip(deployment, rotation)):
        positions[index] = _WORKER_SOLVER.forward_endpoints_mm(
            deployment_row,
            rotation_row,
        )
    return start, positions


def _initialise_ik_worker(
    parameters: dict,
    target_tube: int,
    solver_tolerance_mm: float,
    max_iterations: int,
) -> None:
    global _IK_PARAMETERS, _IK_TARGET_TUBE, _IK_SOLVER_TOLERANCE_MM, _IK_MAX_ITERATIONS
    _IK_PARAMETERS = parameters
    _IK_TARGET_TUBE = int(target_tube)
    _IK_SOLVER_TOLERANCE_MM = float(solver_tolerance_mm)
    _IK_MAX_ITERATIONS = int(max_iterations)


def _ik_chunk(payload):
    start, targets, target_ids, weights = payload
    if _IK_PARAMETERS is None:
        raise RuntimeError("IK worker was not initialised")
    result = calculate_ik_error_map(
        _IK_PARAMETERS,
        targets,
        target_tube=_IK_TARGET_TUBE,
        solver_tolerance_mm=_IK_SOLVER_TOLERANCE_MM,
        max_iterations=_IK_MAX_ITERATIONS,
        canonical_sample_ids=target_ids,
        target_weight_mm3=weights,
    )
    return start, result


def _chunks(
    deployment: np.ndarray,
    rotation: np.ndarray,
    chunk_size: int,
):
    for start in range(0, len(deployment), chunk_size):
        stop = min(start + chunk_size, len(deployment))
        yield start, deployment[start:stop], rotation[start:stop]


def _evaluate_configuration_bank(
    parameters: dict,
    bank: CanonicalConfigurationBank,
    total_lengths_m: np.ndarray,
    *,
    workers: int,
    chunk_size: int,
    translation_step_mm: float,
    rotation_step_rad: float,
    progress_label: str,
) -> dict[str, np.ndarray]:
    deployment = bank.decode_deployments(total_lengths_m)
    rotation = bank.rotation_rad
    count = bank.independent_count
    positions = np.empty((count, 3, 3), dtype=np.float32)
    isotropy = np.full((count, 3), np.nan, dtype=np.float32)
    singular_values = np.full((count, 3, 3), np.nan, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    completed = 0
    reported = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise_fk_worker,
        initargs=(parameters, translation_step_mm, rotation_step_rad),
    ) as executor:
        futures = {
            executor.submit(_jacobian_chunk, payload): len(payload[1])
            for payload in _chunks(deployment, rotation, chunk_size)
        }
        for future in as_completed(futures):
            start, chunk_positions, chunk_isotropy, chunk_singular, chunk_valid = (
                future.result()
            )
            stop = start + len(chunk_positions)
            positions[start:stop] = chunk_positions
            isotropy[start:stop] = chunk_isotropy
            singular_values[start:stop] = chunk_singular
            valid[start:stop] = chunk_valid
            completed += len(chunk_positions)
            if completed - reported >= 5000 or completed == count:
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-9)
                print(
                    f"{progress_label}: {completed:,}/{count:,} configurations "
                    f"({rate:,.0f}/s)",
                    flush=True,
                )
                reported = completed
    return {
        "sample_ids": bank.sample_ids.copy(),
        "deployment_m": deployment.astype(np.float32),
        "rotation_rad": rotation.astype(np.float32),
        "endpoint_positions_mm": positions,
        "isotropy": isotropy,
        "singular_values": singular_values,
        "jacobian_valid": valid,
    }


def _forward_configuration_bank(
    parameters: dict,
    bank: CanonicalConfigurationBank,
    total_lengths_m: np.ndarray,
    *,
    workers: int,
    chunk_size: int,
    progress_label: str,
) -> dict[str, np.ndarray]:
    deployment = bank.decode_deployments(total_lengths_m)
    rotation = bank.rotation_rad
    positions = np.empty((bank.independent_count, 3, 3), dtype=np.float32)
    completed = 0
    reported = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise_fk_worker,
        initargs=(parameters, 0.1, 0.001),
    ) as executor:
        futures = {
            executor.submit(_forward_chunk, payload): len(payload[1])
            for payload in _chunks(deployment, rotation, chunk_size)
        }
        for future in as_completed(futures):
            start, chunk_positions = future.result()
            positions[start : start + len(chunk_positions)] = chunk_positions
            completed += len(chunk_positions)
            if completed - reported >= 10000 or completed == bank.independent_count:
                elapsed = time.perf_counter() - started
                print(
                    f"{progress_label}: {completed:,}/{bank.independent_count:,} "
                    f"configurations ({completed / max(elapsed, 1e-9):,.0f}/s)",
                    flush=True,
                )
                reported = completed
    return {
        "sample_ids": bank.sample_ids.copy(),
        "deployment_m": deployment.astype(np.float32),
        "rotation_rad": rotation.astype(np.float32),
        "endpoint_positions_mm": positions,
    }


def _parallel_ik(
    parameters: dict,
    targets: StratifiedCanonicalTargetSet,
    *,
    target_tube: int,
    solver_tolerance_mm: float,
    max_iterations: int,
    workers: int,
    chunk_size: int,
    progress_label: str,
) -> IKErrorMap:
    count = targets.independent_target_count
    achieved = np.empty((count, 3), dtype=np.float32)
    residual = np.empty(count, dtype=np.float32)
    reached = np.empty(count, dtype=bool)
    iterations = np.empty(count, dtype=np.int16)
    evaluations = np.empty(count, dtype=np.int32)
    solve_time = np.empty(count, dtype=np.float32)
    termination = np.empty(count, dtype="U32")
    initial_error = np.empty(count, dtype=np.float32)
    completed = 0
    reported = 0
    started = time.perf_counter()
    payloads = []
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        payloads.append(
            (
                start,
                targets.targets_mm[start:stop],
                targets.target_ids[start:stop],
                targets.target_weight_mm3[start:stop],
            )
        )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise_ik_worker,
        initargs=(
            parameters,
            target_tube,
            solver_tolerance_mm,
            max_iterations,
        ),
    ) as executor:
        futures = {
            executor.submit(_ik_chunk, payload): len(payload[1])
            for payload in payloads
        }
        for future in as_completed(futures):
            start, values = future.result()
            stop = start + len(values.targets_mm)
            achieved[start:stop] = values.achieved_mm
            residual[start:stop] = values.residual_mm
            reached[start:stop] = values.reached
            iterations[start:stop] = values.iterations
            evaluations[start:stop] = values.evaluations
            solve_time[start:stop] = values.solve_time_ms
            termination[start:stop] = values.termination_reason
            initial_error[start:stop] = values.initial_error_mm
            completed += len(values.targets_mm)
            if completed - reported >= 2000 or completed == count:
                elapsed = time.perf_counter() - started
                print(
                    f"{progress_label}: {completed:,}/{count:,} targets "
                    f"({completed / max(elapsed, 1e-9):,.0f}/s)",
                    flush=True,
                )
                reported = completed
    return IKErrorMap(
        targets_mm=targets.targets_mm.copy(),
        achieved_mm=achieved,
        residual_mm=residual,
        reached=reached,
        iterations=iterations,
        evaluations=evaluations,
        solve_time_ms=solve_time,
        canonical_sample_ids=targets.target_ids.copy(),
        termination_reason=termination,
        solver_tolerance_mm=float(solver_tolerance_mm),
        initial_error_mm=initial_error,
        starts_attempted=np.ones(count, dtype=np.int16),
        converged_start_count=reached.astype(np.int16),
        target_weight_mm3=targets.target_weight_mm3.copy(),
    )


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_radial_map(
    grid: AxisymmetricGrid,
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    colour_label: str,
    cmap: str,
    limits: tuple[float, float] | None = None,
    logarithmic: bool = False,
) -> None:
    data = np.asarray(values, dtype=float)
    figure, axis = plt.subplots(figsize=(8.0, 6.2), constrained_layout=True)
    axis.set_xlabel("Radial distance r (mm)")
    axis.set_ylabel("Z position (mm)")
    axis.set_title(title)
    if not np.any(np.isfinite(data)):
        axis.text(
            0.5,
            0.5,
            "No cells meet the predeclared independent-sample support threshold",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f1f4f8", "edgecolor": "#64748b"},
        )
        figure.savefig(path, dpi=220)
        plt.close(figure)
        return
    masked = np.ma.masked_invalid(data.T)
    kwargs = {"cmap": cmap, "shading": "flat"}
    if logarithmic:
        positive = data[np.isfinite(data) & (data > 0.0)]
        if len(positive):
            kwargs["norm"] = LogNorm(vmin=max(1.0, float(np.min(positive))), vmax=float(np.max(positive)))
    elif limits is not None:
        kwargs["vmin"], kwargs["vmax"] = limits
    mesh = axis.pcolormesh(
        grid.radial_edges_mm,
        grid.z_edges_mm,
        masked,
        **kwargs,
    )
    figure.colorbar(mesh, ax=axis, label=colour_label)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_workspace_sweep(
    positions_mm: np.ndarray,
    isotropy: np.ndarray,
    path: Path,
    *,
    tube_name: str,
    independent_count: int,
) -> None:
    count = min(2500, len(positions_mm))
    rows = np.linspace(0, len(positions_mm) - 1, count, dtype=int)
    radius = np.hypot(positions_mm[rows, 0], positions_mm[rows, 1])
    z_values = positions_mm[rows, 2]
    scores = isotropy[rows]
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    x = (np.cos(angles)[:, None] * radius[None, :]).reshape(-1)
    y = (np.sin(angles)[:, None] * radius[None, :]).reshape(-1)
    z = np.tile(z_values, len(angles))
    colour = np.tile(scores, len(angles))
    figure = plt.figure(figsize=(8.4, 7.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    artist = axis.scatter(
        x,
        y,
        z,
        c=colour,
        s=1.0,
        alpha=0.18,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(
        f"{tube_name.title()}-tip workspace\n"
        f"render-only 360° sweep; independent N={independent_count:,}"
    )
    figure.colorbar(artist, ax=axis, shrink=0.72, label="Positional isotropy")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_convergence(
    rows: list[dict],
    path: Path,
    *,
    tube_name: str,
    primary_cell_size_mm: float,
) -> None:
    selected = [row for row in rows if row["endpoint"] == tube_name]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for cell_size in sorted({row["cell_size_mm"] for row in selected}):
        group = sorted(
            [row for row in selected if row["cell_size_mm"] == cell_size],
            key=lambda row: row["independent_configurations"],
        )
        axes[0].plot(
            [row["independent_configurations"] for row in group],
            [row["occupied_volume_cm3"] for row in group],
            marker="o",
            label=f"{cell_size:g} mm cells",
        )
    primary = sorted(
        [row for row in selected if row["cell_size_mm"] == primary_cell_size_mm],
        key=lambda row: row["independent_configurations"],
    )
    axes[1].plot(
        [row["independent_configurations"] for row in primary],
        [row["volume_weighted_mean"] for row in primary],
        marker="o",
        label="Weighted mean cell median",
    )
    axes[1].plot(
        [row["independent_configurations"] for row in primary],
        [row["volume_weighted_p10"] for row in primary],
        marker="o",
        label="Weighted spatial p10",
    )
    axes[0].set_ylabel("Occupied-cell volume (cm³)")
    axes[1].set_ylabel("Positional isotropy")
    for axis in axes:
        axis.set_xlabel("Independent configurations")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"{tube_name.title()}-tip baseline convergence")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _save_task_region(path: Path, region: FrozenTaskRegion) -> None:
    _save_npz(
        path,
        task_region_id=np.asarray(region.task_region_id),
        radial_edges_mm=region.grid.radial_edges_mm,
        z_edges_mm=region.grid.z_edges_mm,
        occupied=region.occupied,
        discovery_count_per_cell=region.discovery_count_per_cell,
        cell_volume_mm3=region.cell_volume_mm3,
        target_tube=np.asarray(region.target_tube),
        baseline_parameter_sha256=np.asarray(region.baseline_parameter_sha256),
        source_dataset_id=np.asarray(region.source_dataset_id),
    )


def _save_target_set(path: Path, targets: StratifiedCanonicalTargetSet) -> None:
    _save_npz(
        path,
        target_set_id=np.asarray(targets.target_set_id),
        target_ids=targets.target_ids,
        targets_mm=targets.targets_mm,
        source_sample_ids=targets.source_sample_ids,
        source_deployment_m=targets.source_deployment_m,
        source_rotation_rad=targets.source_rotation_rad,
        radial_cell_index=targets.radial_cell_index,
        z_cell_index=targets.z_cell_index,
        target_weight_mm3=targets.target_weight_mm3,
        targets_per_cell=targets.targets_per_cell,
        task_region_id=np.asarray(targets.task_region_id),
        split=np.asarray(targets.split),
        seed=np.asarray(targets.seed),
        source_dataset_id=np.asarray(targets.source_dataset_id),
    )


def _save_ik(path: Path, errors: IKErrorMap) -> None:
    _save_npz(
        path,
        target_ids=errors.canonical_sample_ids,
        targets_mm=errors.targets_mm,
        achieved_mm=errors.achieved_mm,
        residual_mm=errors.residual_mm,
        solver_converged=errors.solver_converged,
        iterations=errors.iterations,
        evaluations=errors.evaluations,
        solve_time_ms=errors.solve_time_ms,
        termination_reason=errors.termination_reason,
        solver_tolerance_mm=np.asarray(errors.solver_tolerance_mm),
        initial_error_mm=errors.initial_error_mm,
        target_weight_mm3=errors.target_weight_mm3,
    )


def _convergence_analysis(
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    counts: list[int],
    cell_sizes_mm: list[float],
    primary_cell_size_mm: float,
    minimum_median_samples: int,
    minimum_lower_tail_samples: int,
) -> tuple[
    list[dict],
    dict[str, AxisymmetricWorkspaceMap],
    dict[str, AxisymmetricScalarMap],
    dict[str, dict],
]:
    records: list[dict] = []
    final_workspace: dict[str, AxisymmetricWorkspaceMap] = {}
    final_scalar: dict[str, AxisymmetricScalarMap] = {}
    validation_metrics: dict[str, dict] = {}
    for endpoint, tube_name in enumerate(TUBE_NAMES):
        full_points = training["endpoint_positions_mm"][:, endpoint]
        full_values = training["isotropy"][:, endpoint]
        ids = training["sample_ids"]
        for cell_size in cell_sizes_mm:
            grid = build_axisymmetric_grid(
                [full_points],
                radial_bin_mm=cell_size,
                z_bin_mm=cell_size,
            )
            maps = []
            scalar_maps = []
            for count in counts:
                workspace = aggregate_axisymmetric_workspace(
                    full_points[:count],
                    canonical_sample_ids=ids[:count],
                    grid=grid,
                )
                scalar = aggregate_axisymmetric_scalar(
                    full_points[:count],
                    full_values[:count],
                    canonical_sample_ids=ids[:count],
                    grid=grid,
                    minimum_median_samples=minimum_median_samples,
                    minimum_lower_tail_samples=minimum_lower_tail_samples,
                )
                summary = summarise_axisymmetric_scalar(
                    scalar,
                    task_workspace=workspace,
                )
                maps.append(workspace)
                scalar_maps.append(scalar)
                records.append(
                    {
                        "endpoint": tube_name,
                        "independent_configurations": count,
                        "cell_size_mm": float(cell_size),
                        "occupied_cells": int(np.count_nonzero(workspace.occupied)),
                        "occupied_volume_cm3": workspace.occupied_volume_mm3 / 1000.0,
                        "enclosed_unsampled_volume_cm3": workspace.internal_void_volume_mm3 / 1000.0,
                        "enclosed_unsampled_components": workspace.internal_void_component_count,
                        "median_supported_cells": int(np.count_nonzero(np.isfinite(scalar.median))),
                        **summary,
                    }
                )
            full_map = maps[-1]
            full_scalar_map = scalar_maps[-1]
            full_record = records[-1]
            for index, count in enumerate(counts):
                volume = grid.cell_volume_mm3
                shared = float(np.sum(volume[maps[index].occupied & full_map.occupied]))
                coverage = shared / max(full_map.occupied_volume_mm3, 1e-15)
                record = next(
                    row
                    for row in records
                    if row["endpoint"] == tube_name
                    and row["cell_size_mm"] == float(cell_size)
                    and row["independent_configurations"] == count
                )
                record["coverage_of_final_workspace"] = coverage
                record["relative_volume_difference_from_final"] = abs(
                    record["occupied_volume_cm3"] - full_record["occupied_volume_cm3"]
                ) / max(full_record["occupied_volume_cm3"], 1e-15)
            if np.isclose(cell_size, primary_cell_size_mm):
                final_workspace[tube_name] = full_map
                final_scalar[tube_name] = full_scalar_map

        reference = final_workspace[tube_name]
        validation_points = validation["endpoint_positions_mm"][:, endpoint]
        validation_values = validation["isotropy"][:, endpoint]
        validation_ids = validation["sample_ids"]
        validation_workspace = aggregate_axisymmetric_workspace(
            validation_points,
            canonical_sample_ids=validation_ids,
            grid=reference.grid,
        )
        validation_scalar = aggregate_axisymmetric_scalar(
            validation_points,
            validation_values,
            canonical_sample_ids=validation_ids,
            grid=reference.grid,
            minimum_median_samples=minimum_median_samples,
            minimum_lower_tail_samples=minimum_lower_tail_samples,
        )
        comparison = compare_axisymmetric_workspaces(
            full_points,
            validation_points,
            reference_sample_ids=ids,
            candidate_sample_ids=validation_ids,
            radial_bin_mm=primary_cell_size_mm,
            z_bin_mm=primary_cell_size_mm,
        )
        validation_metrics[tube_name] = {
            "training_occupied_volume_cm3": reference.occupied_volume_mm3 / 1000.0,
            "validation_occupied_volume_cm3_on_training_grid": validation_workspace.occupied_volume_mm3 / 1000.0,
            "validation_coverage_of_training_workspace": comparison.reference_coverage_fraction,
            "volume_weighted_jaccard": comparison.volume_weighted_jaccard,
            **{
                f"validation_{key}": value
                for key, value in summarise_axisymmetric_scalar(
                    validation_scalar,
                    task_workspace=reference,
                ).items()
            },
        }
    return records, final_workspace, final_scalar, validation_metrics


def _gate_results(
    convergence_rows: list[dict],
    validation_metrics: dict[str, dict],
    *,
    primary_cell_size_mm: float,
    workspace_relative_limit: float,
    validation_coverage_minimum: float,
    isotropy_absolute_limit: float,
    training_valid_fraction: float,
    validation_valid_fraction: float,
) -> dict:
    results = {}
    for tube_name in TUBE_NAMES:
        def row(count):
            return next(
                item
                for item in convergence_rows
                if item["endpoint"] == tube_name
                and item["independent_configurations"] == count
                and item["cell_size_mm"] == primary_cell_size_mm
            )

        fifty = row(50000)
        final = row(100000)
        relative_volume = abs(
            fifty["occupied_volume_cm3"] - final["occupied_volume_cm3"]
        ) / max(final["occupied_volume_cm3"], 1e-15)
        isotropy_changes = {
            key: abs(fifty[key] - final[key])
            for key in (
                "volume_weighted_mean",
                "volume_weighted_median",
                "volume_weighted_p10",
            )
        }
        endpoint_gates = {
            "workspace_50k_to_100k_relative_change": relative_volume,
            "workspace_relative_limit": workspace_relative_limit,
            "workspace_converged": relative_volume <= workspace_relative_limit,
            "validation_coverage": validation_metrics[tube_name][
                "validation_coverage_of_training_workspace"
            ],
            "validation_coverage_minimum": validation_coverage_minimum,
            "validation_coverage_passed": validation_metrics[tube_name][
                "validation_coverage_of_training_workspace"
            ]
            >= validation_coverage_minimum,
            "isotropy_50k_to_100k_absolute_changes": isotropy_changes,
            "isotropy_absolute_limit": isotropy_absolute_limit,
            "isotropy_converged": all(
                value <= isotropy_absolute_limit
                for value in isotropy_changes.values()
            ),
        }
        endpoint_gates["all_endpoint_gates_passed"] = all(
            endpoint_gates[key]
            for key in (
                "workspace_converged",
                "validation_coverage_passed",
                "isotropy_converged",
            )
        )
        results[tube_name] = endpoint_gates
    results["jacobian_valid_fraction_training"] = training_valid_fraction
    results["jacobian_valid_fraction_validation"] = validation_valid_fraction
    results["primary_inner_endpoint_passed"] = bool(
        results["inner"]["all_endpoint_gates_passed"]
        and training_valid_fraction >= 0.999
        and validation_valid_fraction >= 0.999
    )
    results["optimisation_may_begin"] = False
    results["optimisation_gate_note"] = (
        "This baseline run cannot enable optimisation automatically; methods and "
        "results must be reviewed before a new protocol version enables it."
    )
    return results


def _plot_summary(
    workspace_maps: dict[str, AxisymmetricWorkspaceMap],
    scalar_maps: dict[str, AxisymmetricScalarMap],
    ik_metrics: dict[str, dict],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    names = list(TUBE_NAMES)
    axes[0].bar(names, [workspace_maps[name].occupied_volume_mm3 / 1000.0 for name in names])
    axes[0].set_ylabel("Occupied-cell volume (cm³)")
    axes[0].set_title("Baseline workspace")
    axes[1].bar(
        names,
        [summarise_axisymmetric_scalar(scalar_maps[name])["volume_weighted_mean"] for name in names],
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Positional isotropy")
    axes[1].set_title("Weighted mean cell median")
    axes[2].bar(names, [ik_metrics[name]["ik_within_evaluation_threshold_rate"] for name in names])
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Physical-volume fraction")
    axes[2].set_title("IK residual ≤ 0.5 mm")
    figure.suptitle("Supervisor nominal baseline: endpoint comparison")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--ik-chunk-size", type=int, default=200)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.chunk_size < 1 or arguments.ik_chunk_size < 1:
        raise ValueError("worker and chunk settings must be positive")

    protocol = load_analysis_protocol()
    sampling = protocol.section("sampling")
    spatial = protocol.section("spatial")
    ik_settings = protocol.section("ik")
    jacobian = protocol.section("jacobian")
    design = baseline_design()
    parameters = design.to_parameters()
    lengths_m = design.total_length_mm * 1e-3
    commit_result = subprocess.run(
        ("git", "rev-parse", "--short=8", "HEAD"),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = commit_result.stdout.strip() or "nogit"
    output_root = (
        PROJECT_ROOT / protocol.section("versioning")["formal_output_root"]
        if arguments.output_root is None
        else arguments.output_root.resolve()
    )
    run_name = (
        f"baseline_{datetime.now().strftime('%Y%m%d')}_"
        f"{protocol.sha256[:8]}_{commit[:8]}"
    )
    output = output_root / run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty formal run: {output}")
    data_directory = output / "data"
    figure_directory = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        protocol,
        run_kind="stage-3-full-baseline-evaluation",
        effective_run_settings={
            "workers": arguments.workers,
            "configuration_chunk_size": arguments.chunk_size,
            "ik_chunk_size": arguments.ik_chunk_size,
            "output_directory": str(output.relative_to(PROJECT_ROOT)),
        },
        project_root=PROJECT_ROOT,
    )
    manifest_path = output / "run_manifest.json"
    write_run_manifest(manifest_path, manifest)

    def progress(stage: str) -> None:
        manifest["progress_stage"] = stage
        write_run_manifest(manifest_path, manifest)
        print(f"\n[{stage}]", flush=True)

    try:
        progress("training configuration bank and all-endpoint Jacobians")
        training_bank = canonical_configuration_bank(
            int(sampling["baseline_training_configurations"]),
            seed=int(sampling["training_seed"]),
            split="stage3-baseline-training",
        )
        training = _evaluate_configuration_bank(
            parameters,
            training_bank,
            lengths_m,
            workers=arguments.workers,
            chunk_size=arguments.chunk_size,
            translation_step_mm=float(jacobian["translation_step_mm"]),
            rotation_step_rad=float(jacobian["rotation_step_rad"]),
            progress_label="training FK/Jacobian",
        )
        _save_npz(data_directory / "baseline_training_100000.npz", **training)

        progress("independent validation configuration bank")
        validation_bank = canonical_configuration_bank(
            int(sampling["baseline_validation_configurations"]),
            seed=int(sampling["validation_seed"]),
            split="stage3-baseline-validation",
        )
        validation = _evaluate_configuration_bank(
            parameters,
            validation_bank,
            lengths_m,
            workers=arguments.workers,
            chunk_size=arguments.chunk_size,
            translation_step_mm=float(jacobian["translation_step_mm"]),
            rotation_step_rad=float(jacobian["rotation_step_rad"]),
            progress_label="validation FK/Jacobian",
        )
        _save_npz(data_directory / "baseline_validation_25000.npz", **validation)

        progress("workspace and isotropy convergence")
        convergence_rows, workspace_maps, scalar_maps, validation_metrics = (
            _convergence_analysis(
                training,
                validation,
                counts=list(map(int, sampling["baseline_convergence_counts"])),
                cell_sizes_mm=list(
                    map(float, spatial["cell_size_convergence_candidates_mm"])
                ),
                primary_cell_size_mm=float(spatial["primary_cell_size_mm"]),
                minimum_median_samples=int(spatial["minimum_samples_median"]),
                minimum_lower_tail_samples=int(
                    spatial["minimum_samples_lower_tail"]
                ),
            )
        )
        _write_csv(output / "workspace_isotropy_convergence.csv", convergence_rows)
        _write_json(output / "workspace_validation_metrics.json", validation_metrics)

        for endpoint, tube_name in enumerate(TUBE_NAMES):
            workspace = workspace_maps[tube_name]
            scalar = scalar_maps[tube_name]
            _save_npz(
                data_directory / f"{tube_name}_workspace_isotropy_10mm.npz",
                radial_edges_mm=workspace.grid.radial_edges_mm,
                z_edges_mm=workspace.grid.z_edges_mm,
                cell_volume_mm3=workspace.grid.cell_volume_mm3,
                workspace_sample_count=workspace.sample_count,
                occupied=workspace.occupied,
                enclosed_unsampled=workspace.internal_void,
                isotropy_maximum=scalar.maximum,
                isotropy_mean=scalar.mean,
                isotropy_median=scalar.median,
                isotropy_p10=scalar.lower_p10,
                isotropy_q25=scalar.q25,
                isotropy_q75=scalar.q75,
                isotropy_iqr=scalar.iqr,
                isotropy_sample_count=scalar.sample_count,
            )
            _plot_convergence(
                convergence_rows,
                figure_directory / f"{tube_name}_convergence.png",
                tube_name=tube_name,
                primary_cell_size_mm=float(spatial["primary_cell_size_mm"]),
            )
            _plot_radial_map(
                workspace.grid,
                np.where(workspace.occupied, workspace.sample_count, np.nan),
                figure_directory / f"{tube_name}_workspace_occupied_counts.png",
                title=f"{tube_name.title()}-tip occupied workspace (10 mm cells)",
                colour_label="Independent configurations per occupied cell",
                cmap="viridis",
                logarithmic=True,
            )
            for label, values, title in (
                ("maximum", scalar.maximum, "Maximum positional isotropy"),
                ("median", scalar.median, "Median positional isotropy"),
                ("p10", scalar.lower_p10, "Configuration-level p10 isotropy"),
            ):
                _plot_radial_map(
                    scalar.grid,
                    values,
                    figure_directory / f"{tube_name}_isotropy_{label}.png",
                    title=f"{tube_name.title()} tip: {title}",
                    colour_label="Positional isotropy",
                    cmap="viridis",
                    limits=(0.0, 1.0),
                )
            _plot_radial_map(
                scalar.grid,
                np.where(scalar.sample_count > 0, scalar.sample_count, np.nan),
                figure_directory / f"{tube_name}_isotropy_support.png",
                title=f"{tube_name.title()}-tip isotropy support",
                colour_label="Independent valid Jacobians per cell",
                cmap="plasma",
                logarithmic=True,
            )
            _plot_workspace_sweep(
                training["endpoint_positions_mm"][:, endpoint],
                training["isotropy"][:, endpoint],
                figure_directory / f"{tube_name}_workspace_3d_display.png",
                tube_name=tube_name,
                independent_count=len(training["sample_ids"]),
            )

        progress("independent IK target candidate banks")
        training_target_bank = canonical_configuration_bank(
            int(sampling["ik_target_candidate_configurations"]),
            seed=int(sampling["ik_target_training_seed"]),
            split="stage3-ik-target-training-candidates",
        )
        training_candidates = _forward_configuration_bank(
            parameters,
            training_target_bank,
            lengths_m,
            workers=arguments.workers,
            chunk_size=arguments.chunk_size,
            progress_label="training target candidates",
        )
        validation_target_bank = canonical_configuration_bank(
            int(sampling["ik_target_candidate_configurations"]),
            seed=int(sampling["ik_target_validation_seed"]),
            split="stage3-ik-target-validation-candidates",
        )
        validation_candidates = _forward_configuration_bank(
            parameters,
            validation_target_bank,
            lengths_m,
            workers=arguments.workers,
            chunk_size=arguments.chunk_size,
            progress_label="validation target candidates",
        )

        regions: dict[str, FrozenTaskRegion] = {}
        validation_targets: dict[str, StratifiedCanonicalTargetSet] = {}
        for endpoint, tube_name in enumerate(TUBE_NAMES):
            region = freeze_baseline_task_region(
                training["endpoint_positions_mm"][:, endpoint],
                training["sample_ids"],
                target_tube=endpoint,
                baseline_parameter_sha256=baseline_parameter_hash(),
                source_dataset_id="stage3-baseline-training",
                grid=workspace_maps[tube_name].grid,
            )
            regions[tube_name] = region
            _save_task_region(data_directory / f"{tube_name}_task_region.npz", region)
            validation_count = (
                int(sampling["ik_primary_validation_targets"])
                if endpoint == 0
                else int(sampling["ik_secondary_validation_targets"])
            )
            targets = select_stratified_canonical_targets(
                region,
                validation_candidates["endpoint_positions_mm"][:, endpoint],
                validation_candidates["deployment_m"],
                validation_candidates["rotation_rad"],
                validation_candidates["sample_ids"],
                target_count=validation_count,
                split="validation",
                seed=int(sampling["ik_target_validation_seed"]),
                source_dataset_id="stage3-ik-target-validation-candidates",
            )
            validation_targets[tube_name] = targets
            _save_target_set(
                data_directory / f"{tube_name}_ik_validation_targets.npz",
                targets,
            )
            if endpoint == 0:
                training_targets = select_stratified_canonical_targets(
                    region,
                    training_candidates["endpoint_positions_mm"][:, endpoint],
                    training_candidates["deployment_m"],
                    training_candidates["rotation_rad"],
                    training_candidates["sample_ids"],
                    target_count=int(sampling["ik_primary_training_targets"]),
                    split="training",
                    seed=int(sampling["ik_target_training_seed"]),
                    source_dataset_id="stage3-ik-target-training-candidates",
                )
                _save_target_set(
                    data_directory / "inner_ik_training_targets.npz",
                    training_targets,
                )

        progress("fixed-retraction IK validation")
        ik_metrics: dict[str, dict] = {}
        for endpoint, tube_name in enumerate(TUBE_NAMES):
            errors = _parallel_ik(
                parameters,
                validation_targets[tube_name],
                target_tube=endpoint,
                solver_tolerance_mm=float(ik_settings["solver_tolerance_mm"]),
                max_iterations=int(ik_settings["max_iterations"]),
                workers=arguments.workers,
                chunk_size=arguments.ik_chunk_size,
                progress_label=f"{tube_name} IK",
            )
            _save_ik(data_directory / f"{tube_name}_ik_validation_results.npz", errors)
            metrics = summarise_ik_errors(
                errors,
                evaluation_threshold_mm=float(ik_settings["evaluation_threshold_mm"]),
            )
            metrics.pop("ik_success_rate", None)
            target_set = validation_targets[tube_name]
            metrics.update(
                {
                    "independent_target_count": target_set.independent_target_count,
                    "represented_target_cells": int(np.count_nonzero(target_set.targets_per_cell)),
                    "task_region_cells": regions[tube_name].occupied_cell_count,
                }
            )
            ik_metrics[tube_name] = metrics
            residual_map = aggregate_axisymmetric_residual(
                errors.targets_mm,
                errors.residual_mm,
                canonical_target_ids=errors.canonical_sample_ids,
                grid=regions[tube_name].grid,
                evaluation_threshold_mm=float(ik_settings["evaluation_threshold_mm"]),
                minimum_median_samples=int(spatial["minimum_samples_median"]),
                minimum_failure_or_p95_samples=int(
                    spatial["minimum_samples_failure_or_p95"]
                ),
            )
            _save_npz(
                data_directory / f"{tube_name}_ik_residual_10mm.npz",
                radial_edges_mm=residual_map.grid.radial_edges_mm,
                z_edges_mm=residual_map.grid.z_edges_mm,
                maximum_residual_mm=residual_map.maximum_residual_mm,
                median_residual_mm=residual_map.median_residual_mm,
                p95_residual_mm=residual_map.p95_residual_mm,
                failure_fraction=residual_map.failure_fraction,
                sample_count=residual_map.sample_count,
                evaluation_threshold_mm=np.asarray(residual_map.evaluation_threshold_mm),
            )
            finite_residual = errors.residual_mm[np.isfinite(errors.residual_mm)]
            residual_upper = max(0.5, float(np.quantile(finite_residual, 0.95)))
            for label, values, colour_label, limits, cmap in (
                (
                    "median",
                    residual_map.median_residual_mm,
                    "Median numerical IK residual (mm)",
                    (0.0, residual_upper),
                    "magma",
                ),
                (
                    "p95",
                    residual_map.p95_residual_mm,
                    "95th-percentile numerical IK residual (mm)",
                    (0.0, residual_upper),
                    "magma",
                ),
                (
                    "failure_fraction",
                    residual_map.failure_fraction,
                    "Fraction above 0.5 mm",
                    (0.0, 1.0),
                    "inferno",
                ),
            ):
                _plot_radial_map(
                    residual_map.grid,
                    values,
                    figure_directory / f"{tube_name}_ik_{label}.png",
                    title=f"{tube_name.title()}-tip IK {label.replace('_', ' ')}",
                    colour_label=colour_label,
                    cmap=cmap,
                    limits=limits,
                )
            _plot_radial_map(
                residual_map.grid,
                np.where(residual_map.sample_count > 0, residual_map.sample_count, np.nan),
                figure_directory / f"{tube_name}_ik_target_support.png",
                title=f"{tube_name.title()}-tip IK target support",
                colour_label="Independent targets per cell",
                cmap="plasma",
                logarithmic=True,
            )

        _write_json(output / "ik_validation_metrics.json", ik_metrics)
        training_valid_fraction = float(np.mean(training["jacobian_valid"]))
        validation_valid_fraction = float(np.mean(validation["jacobian_valid"]))
        gates = _gate_results(
            convergence_rows,
            validation_metrics,
            primary_cell_size_mm=float(spatial["primary_cell_size_mm"]),
            workspace_relative_limit=float(spatial["workspace_relative_change_limit"]),
            validation_coverage_minimum=float(spatial["validation_coverage_minimum"]),
            isotropy_absolute_limit=float(spatial["spatial_isotropy_absolute_change_limit"]),
            training_valid_fraction=training_valid_fraction,
            validation_valid_fraction=validation_valid_fraction,
        )
        _write_json(output / "validation_gates.json", gates)
        _plot_summary(
            workspace_maps,
            scalar_maps,
            ik_metrics,
            figure_directory / "baseline_endpoint_summary.png",
        )

        manifest["run_status"] = "completed"
        manifest["progress_stage"] = "completed"
        manifest["validation_gates"] = gates
        manifest["formal_result_note"] = (
            "Stage-3 baseline completed. Optimisation remains disabled pending "
            "human review and a new protocol version."
        )
        write_run_manifest(manifest_path, manifest)
        print(f"\nCompleted Stage-3 baseline: {output}", flush=True)
    except Exception as error:
        manifest["run_status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        write_run_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
