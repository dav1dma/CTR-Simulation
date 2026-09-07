"""Run the legacy fixed-start, multi-start and symmetry diagnostic study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_design_analysis import (  # noqa: E402
    IKErrorMap,
    aggregate_ik_residual_voxels,
    baseline_design,
    canonical_configuration_samples,
    calculate_ik_error_map,
    calculate_multistart_ik_error_map,
    expand_first_quadrant_points,
    fold_configurations_to_first_quadrant,
    independent_baseline_target_set,
    summarise_ik_errors,
)
from ctr_inverse_kinematics import ConstrainedTipIK  # noqa: E402
from ctr_workspace_map import radial_workspace_envelope  # noqa: E402
from tools.run_design_study import equal_3d_axes  # noqa: E402


def write_records(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def forward_points(
    parameters: dict,
    deployment_m: np.ndarray,
    rotation_rad: np.ndarray,
    target_tube: int,
) -> np.ndarray:
    solver = ConstrainedTipIK(parameters, model_points_per_section=2)
    points = np.empty((len(deployment_m), 3), dtype=np.float32)
    for index in range(len(points)):
        points[index] = solver.forward_endpoint_mm(
            deployment_m[index], rotation_rad[index], target_tube
        )
    return points


def _solve_batch(
    parameters: dict,
    targets: np.ndarray,
    target_tube: int,
    mode: str,
    seeds: np.ndarray,
) -> IKErrorMap:
    if mode == "fixed":
        return calculate_ik_error_map(
            parameters,
            targets,
            target_tube=target_tube,
            align_retracted_rotation=False,
        )
    if mode == "multistart":
        return calculate_multistart_ik_error_map(
            parameters,
            targets,
            target_tube=target_tube,
            common_rotation_seeds_rad=seeds,
        )
    raise ValueError(f"Unknown IK mode: {mode}")


def solve_parallel(
    parameters: dict,
    targets: np.ndarray,
    *,
    target_tube: int,
    mode: str,
    seeds: np.ndarray,
    workers: int,
    canonical_sample_ids: np.ndarray | None = None,
) -> IKErrorMap:
    batch_count = max(workers * 4, 1)
    indexed_batches = [
        (index, batch)
        for index, batch in enumerate(np.array_split(targets, batch_count))
        if len(batch)
    ]
    results: dict[int, IKErrorMap] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _solve_batch,
                parameters,
                batch,
                target_tube,
                mode,
                seeds,
            ): index
            for index, batch in indexed_batches
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            print(
                f"{mode} IK batch {completed}/{len(indexed_batches)} complete",
                flush=True,
            )
    ordered = [results[index] for index, _ in indexed_batches]
    fields = (
        "targets_mm",
        "achieved_mm",
        "residual_mm",
        "reached",
        "iterations",
        "evaluations",
        "solve_time_ms",
    )
    identifiers = (
        np.arange(len(targets), dtype=np.int64)
        if canonical_sample_ids is None
        else np.asarray(canonical_sample_ids)
    )
    if identifiers.shape != (len(targets),):
        raise ValueError("canonical_sample_ids must have shape (N,)")
    return IKErrorMap(
        **{
            field: np.concatenate([getattr(result, field) for result in ordered], axis=0)
            for field in fields
        },
        canonical_sample_ids=identifiers.copy(),
    )


def save_ik_csv(path: Path, errors: IKErrorMap, quadrants: np.ndarray) -> None:
    write_records(
        path,
        [
            {
                "target_index": index,
                "base_index": index % (len(errors.targets_mm) // 4),
                "canonical_target_id": str(errors.canonical_sample_ids[index]),
                "quadrant": int(quadrants[index] + 1),
                "target_x_mm": float(target[0]),
                "target_y_mm": float(target[1]),
                "target_z_mm": float(target[2]),
                "achieved_x_mm": float(achieved[0]),
                "achieved_y_mm": float(achieved[1]),
                "achieved_z_mm": float(achieved[2]),
                "numerical_ik_residual_mm": float(residual),
                "reached_0_15_mm_tolerance": bool(reached),
                "iterations": int(iterations),
                "evaluations": int(evaluations),
                "solve_time_ms": float(solve_time),
            }
            for index, (
                target,
                achieved,
                residual,
                reached,
                iterations,
                evaluations,
                solve_time,
            ) in enumerate(
                zip(
                    errors.targets_mm,
                    errors.achieved_mm,
                    errors.residual_mm,
                    errors.reached,
                    errors.iterations,
                    errors.evaluations,
                    errors.solve_time_ms,
                )
            )
        ],
    )


def residual_limits(errors: IKErrorMap) -> tuple[float, float]:
    positive = np.asarray(errors.residual_mm, dtype=float)
    positive = positive[positive > 0.0]
    lower = max(float(np.quantile(positive, 0.01)), 1e-4) if len(positive) else 1e-4
    upper = max(float(np.max(positive)), lower * 10.0) if len(positive) else 1.0
    return lower, upper


def plot_log_residual(errors: IKErrorMap, path: Path, title: str) -> None:
    lower, upper = residual_limits(errors)
    values = np.clip(errors.residual_mm, lower, upper)
    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        errors.targets_mm[:, 0],
        errors.targets_mm[:, 1],
        errors.targets_mm[:, 2],
        c=values,
        cmap="magma",
        norm=LogNorm(vmin=lower, vmax=upper),
        s=5,
        alpha=0.58,
        linewidths=0.0,
    )
    figure.colorbar(
        scatter,
        ax=axis,
        pad=0.08,
        shrink=0.72,
        label="Numerical IK residual (mm, logarithmic scale)",
    )
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(title)
    equal_3d_axes(axis, errors.targets_mm)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_failure_map(errors: IKErrorMap, path: Path, title: str) -> None:
    failed = ~np.asarray(errors.reached, dtype=bool)
    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        errors.targets_mm[~failed, 0],
        errors.targets_mm[~failed, 1],
        errors.targets_mm[~failed, 2],
        color="0.72",
        s=2,
        alpha=0.06,
        linewidths=0.0,
        label="Reached",
    )
    axis.scatter(
        errors.targets_mm[failed, 0],
        errors.targets_mm[failed, 1],
        errors.targets_mm[failed, 2],
        color="#d62728",
        s=9,
        alpha=0.78,
        linewidths=0.0,
        label="Residual > 0.15 mm",
    )
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(title)
    axis.legend(loc="upper left")
    equal_3d_axes(axis, errors.targets_mm)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_log_median_slices(
    errors: IKErrorMap,
    path: Path,
    title: str,
    *,
    voxel_size_mm: float,
    minimum_median_samples: int,
) -> None:
    voxels = aggregate_ik_residual_voxels(
        errors,
        voxel_size_mm=voxel_size_mm,
        minimum_median_samples=minimum_median_samples,
    )
    centres = np.asarray(voxels.centres_mm, dtype=float)
    values = np.asarray(voxels.median_residual_mm, dtype=float)
    valid = np.isfinite(values)
    positive = values[valid & (values > 0.0)]
    lower = max(float(np.quantile(positive, 0.01)), 1e-4)
    upper = max(float(np.max(positive)), lower * 10.0)
    weighted_z = np.repeat(centres[valid, 2], np.maximum(voxels.sample_count[valid], 1))
    requested_z = np.quantile(weighted_z, [0.20, 0.40, 0.60, 0.80])
    available_z = np.unique(centres[valid, 2])
    selected_z = np.unique(
        [available_z[np.argmin(np.abs(available_z - value))] for value in requested_z]
    )
    size = float(voxel_size_mm)
    x_values = np.arange(np.min(centres[:, 0]), np.max(centres[:, 0]) + size * 0.5, size)
    y_values = np.arange(np.min(centres[:, 1]), np.max(centres[:, 1]) + size * 0.5, size)
    figure, axes = plt.subplots(1, len(selected_z), figsize=(5.0 * len(selected_z), 5.2))
    figure.subplots_adjust(left=0.055, right=0.91, bottom=0.12, top=0.82, wspace=0.28)
    axes = np.atleast_1d(axes)
    norm = LogNorm(vmin=lower, vmax=upper)
    image = None
    for axis, z_value in zip(axes, selected_z):
        grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
        layer = valid & np.isclose(centres[:, 2], z_value)
        for point, value in zip(centres[layer], values[layer]):
            x_index = int(round((point[0] - x_values[0]) / size))
            y_index = int(round((point[1] - y_values[0]) / size))
            grid[y_index, x_index] = max(value, lower)
        image = axis.imshow(
            grid,
            origin="lower",
            extent=(
                x_values[0] - size / 2,
                x_values[-1] + size / 2,
                y_values[0] - size / 2,
                y_values[-1] + size / 2,
            ),
            cmap="magma",
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(f"Z = {z_value:.0f} mm")
        axis.set_xlabel("X (mm)")
        axis.set_ylabel("Y (mm)")
        axis.grid(alpha=0.15)
    figure.colorbar(
        image,
        ax=axes.tolist(),
        shrink=0.78,
        label="Median numerical IK residual (mm, logarithmic scale)",
    )
    figure.suptitle(title, y=0.965)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_ik_comparison(
    fixed: IKErrorMap,
    multistart: IKErrorMap,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for errors, label, colour in (
        (fixed, "Fixed zero-rotation start", "#1f77b4"),
        (multistart, "Eight-seed multi-start", "#ff7f0e"),
    ):
        values = np.maximum(np.sort(errors.residual_mm.astype(float)), 1e-5)
        cumulative = np.arange(1, len(values) + 1) / len(values)
        axes[0].plot(values, cumulative, label=label, color=colour, linewidth=2.0)
    axes[0].axvline(0.15, color="0.25", linestyle="--", linewidth=1.2, label="0.15 mm tolerance")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Numerical IK residual (mm)")
    axes[0].set_ylabel("Fraction of targets")
    axes[0].set_title("Residual distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.2, which="both")

    success = 100.0 * np.asarray([np.mean(fixed.reached), np.mean(multistart.reached)])
    axes[1].bar([0, 1], success, color=["#1f77b4", "#ff7f0e"], width=0.58)
    axes[1].set_xticks([0, 1], ["Fixed start", "Multi-start"])
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_ylabel("Targets within 0.15 mm (%)")
    axes[1].set_title("IK success rate")
    axes[1].grid(axis="y", alpha=0.2)

    radial = np.linalg.norm(fixed.targets_mm[:, :2], axis=1)
    edges = np.arange(0.0, np.max(radial) + 25.0, 25.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for errors, label, colour in (
        (fixed, "Fixed start", "#1f77b4"),
        (multistart, "Multi-start", "#ff7f0e"),
    ):
        medians = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = (radial >= lower) & (radial < upper)
            medians.append(np.median(errors.residual_mm[selected]) if np.any(selected) else np.nan)
        axes[2].plot(centres, medians, marker="o", label=label, color=colour)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Radial distance from Z-axis (mm)")
    axes[2].set_ylabel("Median numerical IK residual (mm)")
    axes[2].set_title("Residual versus radial distance")
    axes[2].legend()
    axes[2].grid(alpha=0.2, which="both")
    figure.suptitle("Fixed-Start and Multi-Start Inverse-Kinematics Comparison")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_workspace_validation(
    symmetry_points: np.ndarray,
    random_points: np.ndarray,
    path: Path,
) -> None:
    all_points = np.vstack((symmetry_points, random_points))
    figure = plt.figure(figsize=(13, 6.2), constrained_layout=True)
    for index, (points, title) in enumerate(
        (
            (symmetry_points, "15,000 independent configurations\nexpanded to 60,000 positions"),
            (random_points, "15,000 independent full-space\nMonte Carlo configurations"),
        ),
        start=1,
    ):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=2.5, alpha=0.16, color="#2386bd", linewidths=0.0,
        )
        axis.set_xlabel("X (mm)")
        axis.set_ylabel("Y (mm)")
        axis.set_zlabel("Z (mm)")
        axis.set_title(title)
        equal_3d_axes(axis, all_points)
    figure.suptitle("Symmetry-Expanded Workspace and Independent Validation Sample")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_radial_validation(
    symmetry_points: np.ndarray,
    random_points: np.ndarray,
    path: Path,
    *,
    bin_mm: float,
) -> dict[str, float]:
    symmetry_rz = np.column_stack(
        (np.linalg.norm(symmetry_points[:, :2], axis=1), symmetry_points[:, 2])
    )
    random_rz = np.column_stack(
        (np.linalg.norm(random_points[:, :2], axis=1), random_points[:, 2])
    )
    radial_edges = np.arange(0.0, max(np.max(symmetry_rz[:, 0]), np.max(random_rz[:, 0])) + bin_mm, bin_mm)
    z_min = min(np.min(symmetry_rz[:, 1]), np.min(random_rz[:, 1]))
    z_max = max(np.max(symmetry_rz[:, 1]), np.max(random_rz[:, 1]))
    z_edges = np.arange(np.floor(z_min / bin_mm) * bin_mm, z_max + bin_mm, bin_mm)
    histograms = []
    for values in (symmetry_rz, random_rz):
        histogram, _, _ = np.histogram2d(
            values[:, 0], values[:, 1], bins=(radial_edges, z_edges)
        )
        histograms.append(histogram / np.sum(histogram))
    difference = np.abs(histograms[0] - histograms[1])
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    extent = (z_edges[0], z_edges[-1], radial_edges[0], radial_edges[-1])
    maximum = max(np.max(histograms[0]), np.max(histograms[1]))
    for axis, values, title in (
        (axes[0], histograms[0], "Symmetry-expanded"),
        (axes[1], histograms[1], "Independent full-space"),
    ):
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=maximum,
        )
        axis.set_title(title)
        axis.set_xlabel("Z (mm)")
        axis.set_ylabel("Radial distance (mm)")
    figure.colorbar(image, ax=axes[:2].tolist(), label="Normalised sample density")
    diff_image = axes[2].imshow(
        difference,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
    )
    axes[2].set_title("Absolute density difference")
    axes[2].set_xlabel("Z (mm)")
    axes[2].set_ylabel("Radial distance (mm)")
    figure.colorbar(diff_image, ax=axes[2], label="Absolute probability difference")
    figure.suptitle("Radial–Vertical Workspace Validation")
    figure.savefig(path, dpi=220)
    plt.close(figure)

    symmetry_sections = max(8, min(72, len(symmetry_points) // 4))
    random_sections = max(8, min(72, len(random_points) // 4))
    profile_z_a, profile_r_a = radial_workspace_envelope(
        symmetry_points, height_sections=symmetry_sections
    )
    profile_z_b, profile_r_b = radial_workspace_envelope(
        random_points, height_sections=random_sections
    )
    common_z = np.linspace(max(profile_z_a[0], profile_z_b[0]), min(profile_z_a[-1], profile_z_b[-1]), 100)
    envelope_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.interp(common_z, profile_z_a, profile_r_a)
                    - np.interp(common_z, profile_z_b, profile_r_b)
                ) ** 2
            )
        )
    )
    return {
        "radial_wasserstein_distance_mm": float(
            wasserstein_distance(symmetry_rz[:, 0], random_rz[:, 0])
        ),
        "z_wasserstein_distance_mm": float(
            wasserstein_distance(symmetry_rz[:, 1], random_rz[:, 1])
        ),
        "radial_z_jensen_shannon_distance": float(
            jensenshannon(histograms[0].ravel(), histograms[1].ravel())
        ),
        "radial_envelope_rmse_mm": envelope_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-configurations", type=int, default=15_000)
    parser.add_argument("--validation-configurations", type=int, default=15_000)
    parser.add_argument("--multistart-seeds", type=int, default=8)
    parser.add_argument("--voxel-size-mm", type=float, default=15.0)
    parser.add_argument("--minimum-median-samples", type=int, default=5)
    parser.add_argument("--target-tube", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "hybrid_ik_15000",
    )
    args = parser.parse_args()
    if args.base_configurations < 8 or args.validation_configurations < 8:
        raise ValueError("configuration counts must be at least 8")
    if args.multistart_seeds < 2 or args.multistart_seeds % 4:
        raise ValueError("multistart-seeds must be a positive multiple of four")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    design = baseline_design()
    parameters = design.to_parameters()
    lengths_m = design.total_length_mm * 1e-3
    print("Generating render-expanded and validation workspaces...", flush=True)
    _, deployment, rotation = canonical_configuration_samples(
        lengths_m,
        args.base_configurations,
        seed=args.seed + 1,
        split="training-hybrid-workspace-diagnostic",
    )
    base_points = forward_points(parameters, deployment, rotation, args.target_tube)
    folded_points, _, _ = fold_configurations_to_first_quadrant(
        base_points, rotation
    )
    symmetry_points, _ = expand_first_quadrant_points(folded_points)
    _, validation_deployment, validation_rotation = canonical_configuration_samples(
        lengths_m,
        args.validation_configurations,
        seed=args.seed + 2_000,
        split="validation-hybrid-workspace-diagnostic",
    )
    validation_points = forward_points(
        parameters,
        validation_deployment,
        validation_rotation,
        args.target_tube,
    )
    plot_workspace_validation(
        symmetry_points,
        validation_points,
        output / "workspace_sampling_validation.png",
    )
    validation_metrics = plot_radial_validation(
        symmetry_points,
        validation_points,
        output / "workspace_radial_z_validation.png",
        bin_mm=args.voxel_size_mm,
    )

    print("Generating independently reachable IK targets...", flush=True)
    base_target_set = independent_baseline_target_set(
        args.base_configurations,
        target_tube=args.target_tube,
        seed=args.seed + 1_000,
        split="training-hybrid-ik-diagnostic",
    )
    folded_targets, _, _ = fold_configurations_to_first_quadrant(
        base_target_set.targets_mm,
        np.zeros((len(base_target_set.targets_mm), 3), dtype=float),
    )
    targets, quadrants = expand_first_quadrant_points(folded_targets)
    expanded_target_ids = np.tile(base_target_set.canonical_sample_ids, 4)
    seeds = np.linspace(-np.pi, np.pi, args.multistart_seeds, endpoint=False)

    print(
        f"Solving {len(targets):,} rotated repeated-measure targets from one "
        "fixed actuator configuration...",
        flush=True,
    )
    fixed = solve_parallel(
        parameters,
        targets,
        target_tube=args.target_tube,
        mode="fixed",
        seeds=seeds,
        workers=args.workers,
        canonical_sample_ids=expanded_target_ids,
    )
    save_ik_csv(output / "fixed_start_ik_results.csv", fixed, quadrants)
    plot_log_residual(
        fixed,
        output / "fixed_start_ik_residual_log.png",
        "Fixed Zero-Rotation Start: Numerical IK Residual",
    )
    plot_failure_map(
        fixed,
        output / "fixed_start_ik_failures.png",
        "Fixed Zero-Rotation Start: Targets Outside 0.15 mm Tolerance",
    )
    plot_log_median_slices(
        fixed,
        output / "fixed_start_ik_median_xy_slices_log.png",
        "Fixed Zero-Rotation Start: Median Voxel Residual",
        voxel_size_mm=args.voxel_size_mm,
        minimum_median_samples=args.minimum_median_samples,
    )

    print(
        f"Solving {len(targets):,} targets with {args.multistart_seeds} fixed rotational seeds...",
        flush=True,
    )
    multistart = solve_parallel(
        parameters,
        targets,
        target_tube=args.target_tube,
        mode="multistart",
        seeds=seeds,
        workers=args.workers,
        canonical_sample_ids=expanded_target_ids,
    )
    save_ik_csv(output / "multistart_ik_results.csv", multistart, quadrants)
    plot_log_residual(
        multistart,
        output / "multistart_ik_residual_log.png",
        f"{args.multistart_seeds}-Seed Multi-Start: Best Numerical IK Residual",
    )
    plot_failure_map(
        multistart,
        output / "multistart_ik_failures.png",
        f"{args.multistart_seeds}-Seed Multi-Start: Targets Outside 0.15 mm Tolerance",
    )
    plot_log_median_slices(
        multistart,
        output / "multistart_ik_median_xy_slices_log.png",
        f"{args.multistart_seeds}-Seed Multi-Start: Median Voxel Residual",
        voxel_size_mm=args.voxel_size_mm,
        minimum_median_samples=args.minimum_median_samples,
    )
    plot_ik_comparison(
        fixed,
        multistart,
        output / "fixed_vs_multistart_ik_comparison.png",
    )

    summary = {
        "independent_symmetry_reduced_configurations": args.base_configurations,
        "symmetry_expanded_display_positions": int(len(symmetry_points)),
        "display_positions_are_independent_configurations": False,
        "independent_fullspace_validation_configurations": args.validation_configurations,
        "canonical_target_count": int(len(base_target_set.targets_mm)),
        "rotated_repeated_measure_solves_per_method": int(len(targets)),
        "rotated_solves_are_independent_targets": False,
        "fixed_start": {
            "deployment_mm": [0.0, 0.0, 0.0],
            "rotation_deg": [0.0, 0.0, 0.0],
            "metrics": summarise_ik_errors(fixed),
        },
        "multistart": {
            "deployment_mm_for_every_seed": [0.0, 0.0, 0.0],
            "common_rotation_seeds_deg": np.degrees(seeds).tolist(),
            "metrics": summarise_ik_errors(multistart),
        },
        "workspace_validation": validation_metrics,
        "voxel_size_mm": args.voxel_size_mm,
        "minimum_median_samples": args.minimum_median_samples,
        "notes": (
            "IK residuals are numerical convergence residuals against targets "
            "generated by the same ideal forward model, not physical robot error."
        ),
    }
    (output / "hybrid_study_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Saved hybrid study to {output}", flush=True)


if __name__ == "__main__":
    main()
