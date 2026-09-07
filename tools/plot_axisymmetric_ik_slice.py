"""Generate one symmetry-reduced XY inverse-kinematics validation slice."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_design_analysis import baseline_design, summarise_ik_errors  # noqa: E402
from tools.run_hybrid_ik_study import solve_parallel  # noqa: E402


def read_quadrant_targets(
    source: Path,
    *,
    centre_z_mm: float,
    thickness_mm: float,
) -> np.ndarray:
    lower = centre_z_mm - thickness_mm / 2.0
    upper = centre_z_mm + thickness_mm / 2.0
    targets = []
    with source.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["quadrant"]) != 1:
                continue
            z_value = float(row["target_z_mm"])
            if lower <= z_value < upper:
                x_value = float(row["target_x_mm"])
                y_value = float(row["target_y_mm"])
                radius = float(np.hypot(x_value, y_value))
                targets.append((radius, 0.0, z_value))
    values = np.asarray(targets, dtype=float)
    if len(values) < 20:
        raise ValueError("The selected Z layer contains too few canonical targets")
    return values


def write_canonical_results(
    path: Path,
    targets: np.ndarray,
    errors,
    *,
    success_threshold_mm: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "canonical_radius_mm",
                "canonical_z_mm",
                "numerical_ik_residual_mm",
                "within_evaluation_threshold",
                "evaluation_threshold_mm",
            )
        )
        for target, residual in zip(targets, errors.residual_mm):
            writer.writerow(
                (
                    target[0],
                    target[2],
                    residual,
                    bool(residual <= success_threshold_mm),
                    success_threshold_mm,
                )
            )


def canonical_radial_slice_grid(
    radii_mm: np.ndarray,
    residual_mm: np.ndarray,
    *,
    voxel_size_mm: float,
    minimum_independent_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Map canonical radial statistics to XY cells before display sweeping.

    Each input row is one independent canonical target. Angular rendering does
    not enter this calculation, so changing graphical sweep resolution cannot
    change statistical support or cell masking.
    """
    radii = np.asarray(radii_mm, dtype=float).reshape(-1)
    residual = np.asarray(residual_mm, dtype=float).reshape(-1)
    if radii.shape != residual.shape or not len(radii):
        raise ValueError("radii and residuals must be non-empty and have equal length")
    if not np.all(np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("canonical radii must be finite and non-negative")
    if not np.all(np.isfinite(residual)) or np.any(residual < 0.0):
        raise ValueError("residuals must be finite and non-negative")
    if voxel_size_mm <= 0.0 or minimum_independent_samples < 1:
        raise ValueError("cell size and minimum independent count must be positive")

    cell = float(voxel_size_mm)
    limit = max(cell, np.ceil(np.max(radii) / cell) * cell)
    radial_edges = np.arange(0.0, limit + cell * 1.01, cell)
    radial_index = np.minimum(
        np.floor(radii / cell).astype(int),
        len(radial_edges) - 2,
    )
    radial_median = np.full(len(radial_edges) - 1, np.nan, dtype=float)
    radial_count = np.zeros(len(radial_edges) - 1, dtype=np.int32)
    for index in np.unique(radial_index):
        values = residual[radial_index == index]
        radial_count[index] = len(values)
        if len(values) >= minimum_independent_samples:
            radial_median[index] = np.median(values)

    edges = np.arange(-limit, limit + cell * 1.01, cell)
    centres = 0.5 * (edges[:-1] + edges[1:])
    x_grid, y_grid = np.meshgrid(centres, centres, indexing="ij")
    grid_radius = np.hypot(x_grid, y_grid)
    grid_index = np.floor(grid_radius / cell).astype(int)
    inside = grid_index < len(radial_median)
    median_grid = np.full(grid_index.shape, np.nan, dtype=float)
    count_grid = np.zeros(grid_index.shape, dtype=np.int32)
    median_grid[inside] = radial_median[grid_index[inside]]
    count_grid[inside] = radial_count[grid_index[inside]]
    median_grid[count_grid < minimum_independent_samples] = np.nan
    return (
        median_grid,
        count_grid,
        edges,
        edges,
        int(np.sum(radial_count > 0)),
    )


def plot_swept_slice(
    radii_mm: np.ndarray,
    residual_mm: np.ndarray,
    output: Path,
    *,
    centre_z_mm: float,
    thickness_mm: float,
    angular_samples: int,
    voxel_size_mm: float,
    success_threshold_mm: float,
    success_rate: float,
) -> dict[str, float | int]:
    median, count, x_edges, y_edges, radial_bin_count = canonical_radial_slice_grid(
        radii_mm,
        residual_mm,
        voxel_size_mm=voxel_size_mm,
        minimum_independent_samples=5,
    )
    limit = float(x_edges[-1])
    positive = median[np.isfinite(median) & (median > 0.0)]
    if not len(positive):
        raise ValueError(
            "no radial cell has five independent canonical targets with positive residual"
        )
    lower = max(float(np.quantile(positive, 0.01)), 1e-4)
    upper = max(float(np.max(positive)), lower * 10.0)
    displayed = np.where(np.isfinite(median), np.maximum(median, lower), np.nan)

    figure, axis = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    image = axis.pcolormesh(
        x_edges,
        y_edges,
        displayed.T,
        cmap="magma",
        norm=LogNorm(vmin=lower, vmax=upper),
        shading="flat",
    )
    figure.colorbar(
        image,
        ax=axis,
        label="Median numerical IK residual (mm, logarithmic scale)",
    )
    axis.axhline(0.0, color="0.45", linewidth=0.55, alpha=0.45)
    axis.axvline(0.0, color="0.45", linewidth=0.55, alpha=0.45)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_title(
        "Symmetry-Reduced IK Residual After Continuous Z-Axis Sweep\n"
        f"Z = {centre_z_mm:.0f} ± {thickness_mm / 2.0:.1f} mm"
    )
    axis.text(
        0.02,
        0.02,
        (
            f"{len(radii_mm):,} independent canonical targets\n"
            f"{angular_samples} visualisation angles; {voxel_size_mm:g} mm cells\n"
            f"{success_rate * 100.0:.1f}% within {success_threshold_mm:g} mm"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.86},
    )
    figure.savefig(output, dpi=240)
    plt.close(figure)

    return {
        "canonical_target_count": int(len(radii_mm)),
        "angular_visualisation_samples": int(angular_samples),
        "display_copy_count": int(len(radii_mm) * angular_samples),
        "display_copies_are_independent": False,
        "canonical_radial_bins_with_data": int(radial_bin_count),
        "occupied_voxels": int(np.sum(np.isfinite(median))),
        "minimum_displayed_residual_mm": lower,
        "maximum_displayed_residual_mm": upper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "hybrid_ik_15000"
        / "fixed_start_ik_results.csv",
    )
    parser.add_argument("--centre-z-mm", type=float, default=170.0)
    parser.add_argument("--thickness-mm", type=float, default=15.0)
    parser.add_argument("--angular-samples", type=int, default=360)
    parser.add_argument("--voxel-size-mm", type=float, default=15.0)
    parser.add_argument("--success-threshold-mm", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "axisymmetric_ik_slice"
        / "axisymmetric_ik_xy_slice_z170.png",
    )
    args = parser.parse_args()
    if args.angular_samples < 12:
        raise ValueError("angular-samples must be at least 12")
    if (
        args.thickness_mm <= 0.0
        or args.voxel_size_mm <= 0.0
        or args.success_threshold_mm <= 0.0
    ):
        raise ValueError("slice thickness, cell size, and threshold must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    canonical_targets = read_quadrant_targets(
        args.source.resolve(),
        centre_z_mm=args.centre_z_mm,
        thickness_mm=args.thickness_mm,
    )
    parameters = baseline_design().to_parameters()
    errors = solve_parallel(
        parameters,
        canonical_targets,
        target_tube=0,
        mode="fixed",
        seeds=np.empty(0),
        workers=args.workers,
    )
    write_canonical_results(
        output.with_name(f"{output.stem}_canonical_results.csv"),
        canonical_targets,
        errors,
        success_threshold_mm=args.success_threshold_mm,
    )
    success_rate = float(np.mean(errors.residual_mm <= args.success_threshold_mm))
    plot_metadata = plot_swept_slice(
        canonical_targets[:, 0],
        errors.residual_mm,
        output,
        centre_z_mm=args.centre_z_mm,
        thickness_mm=args.thickness_mm,
        angular_samples=args.angular_samples,
        voxel_size_mm=args.voxel_size_mm,
        success_threshold_mm=args.success_threshold_mm,
        success_rate=success_rate,
    )
    ik_metrics = summarise_ik_errors(errors)
    solver_stopping_rate = ik_metrics.pop("ik_success_rate")
    summary = {
        "method": (
            "Targets transformed to the canonical (radius, 0, Z) plane, solved "
            "from zero deployment and zero rotation, then revolved about Z for "
            "visualisation. Revolved positions are not independent configurations."
        ),
        "centre_z_mm": args.centre_z_mm,
        "thickness_mm": args.thickness_mm,
        "voxel_size_mm": args.voxel_size_mm,
        "evaluation_success_threshold_mm": args.success_threshold_mm,
        "evaluation_success_rate": success_rate,
        "solver_stopping_tolerance_mm": 0.15,
        "solver_stopping_rate": solver_stopping_rate,
        "ik_metrics": ik_metrics,
        **plot_metadata,
    }
    output.with_name(f"{output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Saved validation slice to {output}", flush=True)


if __name__ == "__main__":
    main()
