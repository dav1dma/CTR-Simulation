"""Run the legacy exploratory CTR workspace, dexterity and design workflow."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_protocol import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    build_run_manifest,
    load_analysis_protocol,
    write_run_manifest,
)
from ctr_design_analysis import (  # noqa: E402
    DesignEvaluation,
    DesignVariable,
    OptimisationResult,
    VoxelDexterityMap,
    VoxelResidualMap,
    aggregate_cylindrical_dexterity,
    aggregate_dexterity_voxels,
    baseline_design,
    design_variables,
    evaluate_design,
    independent_baseline_targets,
    optimise_design,
    rank_sensitivity_variables,
    sensitivity_study,
)


PROFILES = {
    "quick": {
        "workspace": 256,
        "dexterity": 2_000,
        "voxel_size_mm": 25.0,
        "minimum_median_samples": 3,
        "ik": 24,
        "sensitivity_workspace": 96,
        "sensitivity_dexterity": 16,
        "sensitivity_ik": 8,
        "optimisation_workspace": 96,
        "optimisation_dexterity": 16,
        "optimisation_ik": 8,
        "optimisation_iterations": 1,
        "optimisation_population": 3,
        "optimisation_variables": 4,
    },
    "standard": {
        "workspace": 1_200,
        "dexterity": 10_000,
        "voxel_size_mm": 20.0,
        "minimum_median_samples": 4,
        "ik": 100,
        "sensitivity_workspace": 400,
        "sensitivity_dexterity": 80,
        "sensitivity_ik": 24,
        "optimisation_workspace": 250,
        "optimisation_dexterity": 50,
        "optimisation_ik": 16,
        "optimisation_iterations": 5,
        "optimisation_population": 5,
        "optimisation_variables": 6,
    },
    "publication": {
        "workspace": 12_000,
        "dexterity": 50_000,
        "voxel_size_mm": 15.0,
        "minimum_median_samples": 5,
        "ik": 500,
        "sensitivity_workspace": 2_000,
        "sensitivity_dexterity": 400,
        "sensitivity_ik": 100,
        "optimisation_workspace": 600,
        "optimisation_dexterity": 120,
        "optimisation_ik": 50,
        "optimisation_iterations": 15,
        "optimisation_population": 8,
        "optimisation_variables": 8,
    },
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_records(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def evaluation_summary(evaluation: DesignEvaluation) -> dict:
    return {
        "design": evaluation.design.as_dict(),
        "metrics": evaluation.metrics,
    }


def optimisation_summary(result: OptimisationResult) -> dict:
    return {
        "variables": list(result.variables),
        "values": result.values,
        "objective": result.objective,
        "success": result.success,
        "message": result.message,
        "function_evaluations": result.function_evaluations,
        **evaluation_summary(result.evaluation),
    }


def equal_3d_axes(axis, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float)
    centre = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
    radius = max(float(np.max(np.ptp(points, axis=0))) * 0.52, 1.0)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plot_3d_heatmap(
    points: np.ndarray,
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    colour_label: str,
    cmap: str,
    limits: tuple[float, float] | None = None,
) -> None:
    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    options = {}
    if limits is not None:
        options = {"vmin": limits[0], "vmax": limits[1]}
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=values,
        cmap=cmap,
        s=10,
        alpha=0.78,
        linewidths=0.0,
        **options,
    )
    figure.colorbar(scatter, ax=axis, pad=0.08, shrink=0.72, label=colour_label)
    axis.scatter([0.0], [0.0], [0.0], marker="x", color="black", s=55)
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(title)
    equal_3d_axes(axis, points)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_filled_voxels(
    voxels: VoxelDexterityMap | VoxelResidualMap,
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    colour_label: str,
    cmap: str,
    logarithmic: bool = False,
    limits: tuple[float, float] | None = None,
) -> None:
    centres = np.asarray(voxels.centres_mm, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return
    size = voxels.voxel_size_mm
    indices = np.floor(centres / size).astype(int)
    minimum = np.min(indices, axis=0)
    maximum = np.max(indices, axis=0)
    shape = tuple((maximum - minimum + 1).tolist())
    occupied = np.zeros(shape, dtype=bool)
    colours = np.zeros((*shape, 4), dtype=float)
    colour_map = plt.colormaps[cmap]
    finite_values = values[valid]
    if logarithmic:
        lower = max(float(np.min(finite_values)), 1.0)
        upper = max(float(np.max(finite_values)), lower * 1.001)
        norm = LogNorm(vmin=lower, vmax=upper)
    else:
        lower, upper = limits if limits is not None else (0.0, 1.0)
        norm = Normalize(vmin=float(lower), vmax=float(upper))
    for index, value, is_valid in zip(indices, values, valid):
        if not is_valid:
            continue
        local = tuple((index - minimum).tolist())
        occupied[local] = True
        colours[local] = colour_map(norm(value))
    edges = [
        np.arange(minimum[axis], maximum[axis] + 2, dtype=float) * size
        for axis in range(3)
    ]
    x_edges, y_edges, z_edges = np.meshgrid(*edges, indexing="ij")
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.voxels(
        x_edges,
        y_edges,
        z_edges,
        occupied,
        facecolors=colours,
        edgecolor=(0.12, 0.12, 0.12, 0.10),
        linewidth=0.12,
    )
    figure.colorbar(
        ScalarMappable(norm=norm, cmap=colour_map),
        ax=axis,
        pad=0.08,
        shrink=0.72,
        label=colour_label,
    )
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(title)
    equal_3d_axes(axis, centres)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_xy_slices(
    voxels: VoxelDexterityMap | VoxelResidualMap,
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    colour_label: str = r"Isotropy $\sigma_{min}/\sigma_{max}$",
    cmap: str = "viridis",
    limits: tuple[float, float] = (0.0, 1.0),
) -> None:
    centres = np.asarray(voxels.centres_mm, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    available_z = np.unique(centres[valid, 2])
    if not len(available_z):
        return
    # Use occupancy-weighted interior quantiles instead of the nearly empty
    # extreme Z layers, making each cross-section representative of the data.
    weighted_z = np.repeat(
        centres[valid, 2],
        np.maximum(voxels.sample_count[valid], 1),
    )
    requested_z = np.quantile(weighted_z, [0.20, 0.40, 0.60, 0.80])
    selected_z = np.unique(
        [available_z[np.argmin(np.abs(available_z - value))] for value in requested_z]
    )
    columns = len(selected_z)
    figure, axes = plt.subplots(1, columns, figsize=(5.0 * columns, 5.2))
    # Reserve a fixed title band.  Matplotlib's constrained layout can place
    # sparse image axes too close to the suptitle, especially for the median
    # map where many edge cells are intentionally masked.
    figure.subplots_adjust(
        left=0.055,
        right=0.91,
        bottom=0.12,
        top=0.82,
        wspace=0.28,
    )
    axes = np.atleast_1d(axes)
    size = voxels.voxel_size_mm
    x_values = np.arange(np.min(centres[:, 0]), np.max(centres[:, 0]) + size * 0.5, size)
    y_values = np.arange(np.min(centres[:, 1]), np.max(centres[:, 1]) + size * 0.5, size)
    last_image = None
    for axis, z_value in zip(axes, selected_z):
        grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
        layer = valid & np.isclose(centres[:, 2], z_value)
        for point, value in zip(centres[layer], values[layer]):
            x_index = int(round((point[0] - x_values[0]) / size))
            y_index = int(round((point[1] - y_values[0]) / size))
            grid[y_index, x_index] = value
        last_image = axis.imshow(
            grid,
            origin="lower",
            extent=(
                x_values[0] - size / 2,
                x_values[-1] + size / 2,
                y_values[0] - size / 2,
                y_values[-1] + size / 2,
            ),
            cmap=cmap,
            vmin=float(limits[0]),
            vmax=float(limits[1]),
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(f"Z = {z_value:.0f} mm")
        axis.set_xlabel("X (mm)")
        axis.set_ylabel("Y (mm)")
        axis.grid(alpha=0.15)
    if last_image is not None:
        figure.colorbar(
            last_image,
            ax=axes.tolist(),
            shrink=0.78,
            label=colour_label,
        )
    figure.suptitle(title, y=0.965)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_cylindrical_dexterity(dexterity, output: Path, settings: dict) -> None:
    bin_size = float(settings["voxel_size_mm"])
    cylindrical = aggregate_cylindrical_dexterity(
        dexterity,
        radial_bin_mm=bin_size,
        z_bin_mm=bin_size,
        minimum_median_samples=int(settings["minimum_median_samples"]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    extent = (
        cylindrical.z_centres_mm[0] - bin_size / 2,
        cylindrical.z_centres_mm[-1] + bin_size / 2,
        cylindrical.radial_centres_mm[0] - bin_size / 2,
        cylindrical.radial_centres_mm[-1] + bin_size / 2,
    )
    fields = (
        (cylindrical.maximum_isotropy, "Maximum isotropy", "viridis", 0.0, 1.0),
        (cylindrical.median_isotropy, "Median isotropy", "viridis", 0.0, 1.0),
        (
            np.where(cylindrical.sample_count > 0, cylindrical.sample_count, np.nan),
            "Configurations per bin",
            "plasma",
            None,
            None,
        ),
    )
    for axis, (field, title, cmap, lower, upper) in zip(axes, fields):
        image_options = {}
        if lower is not None:
            image_options.update(vmin=lower, vmax=upper)
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            **image_options,
        )
        figure.colorbar(image, ax=axis, shrink=0.82)
        axis.set_title(title)
        axis.set_xlabel("Z (mm)")
        axis.set_ylabel("Radial distance (mm)")
    figure.suptitle("Baseline CTR Radial–Vertical Dexterity and Sampling")
    figure.savefig(output / "baseline_dexterity_radial_z.png", dpi=220)
    plt.close(figure)


def save_baseline_outputs(
    output: Path,
    evaluation: DesignEvaluation,
    settings: dict,
) -> None:
    write_json(output / "baseline_summary.json", evaluation_summary(evaluation))
    if evaluation.dexterity is not None:
        dexterity = evaluation.dexterity
        dexterity_ids = (
            np.arange(len(dexterity.positions_mm), dtype=np.int64)
            if dexterity.canonical_sample_ids is None
            else np.asarray(dexterity.canonical_sample_ids)
        )
        records = [
            {
                "canonical_sample_id": str(identifier),
                "x_mm": float(point[0]),
                "y_mm": float(point[1]),
                "z_mm": float(point[2]),
                "positional_isotropy": float(score),
                "sigma_max": float(singular[0]),
                "sigma_middle": float(singular[1]),
                "sigma_min": float(singular[2]),
            }
            for identifier, point, score, singular in zip(
                dexterity_ids,
                dexterity.positions_mm,
                dexterity.isotropy,
                dexterity.singular_values,
            )
        ]
        write_records(output / "baseline_dexterity.csv", records)
        plot_3d_heatmap(
            dexterity.positions_mm,
            dexterity.isotropy,
            output / "baseline_configuration_dexterity.png",
            title="Baseline CTR Configuration Dexterity",
            colour_label=r"Isotropy $\sigma_{min}/\sigma_{max}$",
            cmap="viridis",
            limits=(0.0, 1.0),
        )
        voxels = aggregate_dexterity_voxels(
            dexterity,
            voxel_size_mm=float(settings["voxel_size_mm"]),
            minimum_median_samples=int(settings["minimum_median_samples"]),
        )
        write_records(
            output / "baseline_dexterity_voxels.csv",
            [
                {
                    "voxel_x_mm": float(point[0]),
                    "voxel_y_mm": float(point[1]),
                    "voxel_z_mm": float(point[2]),
                    "maximum_positional_isotropy": float(maximum),
                    "median_positional_isotropy": float(median),
                    "configuration_count": int(count),
                    "voxel_size_mm": voxels.voxel_size_mm,
                    "minimum_median_samples": voxels.minimum_median_samples,
                }
                for point, maximum, median, count in zip(
                    voxels.centres_mm,
                    voxels.maximum_isotropy,
                    voxels.median_isotropy,
                    voxels.sample_count,
                )
            ],
        )
        for aggregation, values, description in (
            ("maximum", voxels.maximum_isotropy, "Best available configuration"),
            (
                "median",
                voxels.median_isotropy,
                f"Typical configuration; at least {voxels.minimum_median_samples} samples per voxel",
            ),
        ):
            plot_filled_voxels(
                voxels,
                values,
                output / f"baseline_dexterity_{aggregation}_heatmap.png",
                title=f"Baseline CTR {aggregation.title()} Voxel Dexterity\n{description}",
                colour_label=r"Isotropy $\sigma_{min}/\sigma_{max}$",
                cmap="viridis",
            )
            plot_xy_slices(
                voxels,
                values,
                output / f"baseline_dexterity_{aggregation}_xy_slices.png",
                title=f"Baseline {aggregation.title()} Dexterity Cross-Sections",
            )
        plot_filled_voxels(
            voxels,
            voxels.sample_count,
            output / "baseline_dexterity_voxel_occupancy.png",
            title="Baseline CTR Dexterity-Sample Occupancy",
            colour_label="Configurations per occupied voxel",
            cmap="plasma",
            logarithmic=True,
        )
        plot_cylindrical_dexterity(dexterity, output, settings)
    if evaluation.ik_errors is not None:
        errors = evaluation.ik_errors
        error_ids = (
            np.arange(len(errors.targets_mm), dtype=np.int64)
            if errors.canonical_sample_ids is None
            else np.asarray(errors.canonical_sample_ids)
        )
        records = [
            {
                "canonical_target_id": str(identifier),
                "target_x_mm": float(target[0]),
                "target_y_mm": float(target[1]),
                "target_z_mm": float(target[2]),
                "achieved_x_mm": float(achieved[0]),
                "achieved_y_mm": float(achieved[1]),
                "achieved_z_mm": float(achieved[2]),
                "numerical_ik_residual_mm": float(residual),
                "reached_0_15_mm_tolerance": bool(reached),
                "iterations": int(iterations),
                "evaluations": int(function_evaluations),
                "solve_time_ms": float(solve_time),
            }
            for (
                identifier,
                target,
                achieved,
                residual,
                reached,
                iterations,
                function_evaluations,
                solve_time,
            ) in zip(
                error_ids,
                errors.targets_mm,
                errors.achieved_mm,
                errors.residual_mm,
                errors.reached,
                errors.iterations,
                errors.evaluations,
                errors.solve_time_ms,
            )
        ]
        write_records(output / "baseline_ik_residual.csv", records)
        upper = max(float(np.quantile(errors.residual_mm, 0.95)), 0.15)
        plot_3d_heatmap(
            errors.targets_mm,
            errors.residual_mm,
            output / "baseline_ik_residual_heatmap.png",
            title="Local IK Residual from Fixed Fully Retracted Configuration",
            colour_label="Numerical IK residual (mm)",
            cmap="magma",
            limits=(0.0, upper),
        )


def plot_sensitivity(
    records: list[dict],
    variables: tuple[DesignVariable, ...],
    output: Path,
) -> None:
    metric_groups = (
        (
            "sensitivity_workspace.png",
            (("workspace_volume_cm3", "Workspace volume (cm³)"),),
            "Workspace sensitivity",
        ),
        (
            "sensitivity_dexterity.png",
            (
                ("dexterity_mean", "Mean positional isotropy"),
                ("dexterity_p10", "10th-percentile isotropy"),
                ("dexterity_cv", "Dexterity coefficient of variation"),
            ),
            "Dexterity sensitivity",
        ),
        (
            "sensitivity_ik_residual.png",
            (
                ("ik_rmse_mm", "IK RMSE (mm)"),
                ("ik_p95_error_mm", "95th-percentile residual (mm)"),
                ("ik_success_rate", "IK success rate"),
            ),
            "Local IK sensitivity from a fixed fully retracted configuration",
        ),
    )
    for filename, metrics, title in metric_groups:
        figure, axes = plt.subplots(
            1, len(metrics), figsize=(6.0 * len(metrics), 5.0), constrained_layout=True
        )
        axes = np.atleast_1d(axes)
        for axis, (metric, ylabel) in zip(axes, metrics):
            for variable in variables:
                rows = [row for row in records if row["parameter"] == variable.name]
                if len(rows) < 2:
                    continue
                rows.sort(key=lambda row: float(row["relative_change"]))
                axis.plot(
                    [100.0 * float(row["relative_change"]) for row in rows],
                    [float(row[metric]) for row in rows],
                    marker="o",
                    linewidth=1.2,
                    markersize=3.5,
                    label=variable.label,
                )
            axis.axvline(0.0, color="0.5", linewidth=0.8)
            axis.set_xlabel("Parameter change from baseline (%)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22)
        axes[0].legend(fontsize=7, ncol=2)
        figure.suptitle(title)
        figure.savefig(output / filename, dpi=220)
        plt.close(figure)


def plot_comparison(
    evaluations: list[tuple[str, DesignEvaluation]],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(6.3 * len(evaluations), 11), constrained_layout=True)
    all_points = np.vstack([evaluation.workspace_tips_mm for _name, evaluation in evaluations])
    for column, (name, evaluation) in enumerate(evaluations):
        workspace_axis = figure.add_subplot(2, len(evaluations), column + 1, projection="3d")
        points = evaluation.workspace_tips_mm
        workspace_axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.18, color="#2386c8"
        )
        workspace_axis.set_title(
            f"{name}\nVolume {evaluation.metrics['workspace_volume_cm3']:.1f} cm³"
        )
        workspace_axis.set_xlabel("X (mm)")
        workspace_axis.set_ylabel("Y (mm)")
        workspace_axis.set_zlabel("Z (mm)")
        equal_3d_axes(workspace_axis, all_points)

        dexterity_axis = figure.add_subplot(
            2, len(evaluations), len(evaluations) + column + 1, projection="3d"
        )
        dexterity = evaluation.dexterity
        if dexterity is not None:
            scatter = dexterity_axis.scatter(
                dexterity.positions_mm[:, 0],
                dexterity.positions_mm[:, 1],
                dexterity.positions_mm[:, 2],
                c=dexterity.isotropy,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=7,
                alpha=0.75,
                linewidths=0.0,
            )
            figure.colorbar(scatter, ax=dexterity_axis, shrink=0.55, pad=0.08)
            dexterity_axis.set_title(
                f"Mean {evaluation.metrics['dexterity_mean']:.3f}; "
                f"CV {evaluation.metrics['dexterity_cv']:.3f}"
            )
            equal_3d_axes(dexterity_axis, all_points)
        dexterity_axis.set_xlabel("X (mm)")
        dexterity_axis.set_ylabel("Y (mm)")
        dexterity_axis.set_zlabel("Z (mm)")
    figure.suptitle("Baseline and Exploratory CTR Optimisation Candidates")
    figure.savefig(output / "exploratory_design_comparison.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="versioned methodology configuration recorded in the run manifest",
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--target-tube", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-optimization", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="generate workspace/dexterity/IK baseline outputs without sensitivity or optimisation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "design_analysis",
    )
    args = parser.parse_args()
    protocol = load_analysis_protocol(args.protocol)
    settings = PROFILES[args.profile]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = baseline_design()
    variables = design_variables(baseline)
    targets = independent_baseline_targets(
        settings["ik"], target_tube=args.target_tube, seed=args.seed + 1000
    )

    protocol_deviations = [
        "legacy exploratory runner: it does not use the Stage-2 occupied-volume and spatially weighted objectives",
        "legacy profile cell sizes and minimum counts override the Stage-2 protocol settings",
        "configuration-derived IK targets remain in use instead of the implemented frozen-region stratified targets",
    ]
    if args.target_tube != protocol.section("scope")["primary_endpoint_index"]:
        protocol_deviations.append(
            "a secondary endpoint was selected instead of the primary inner endpoint"
        )
    study_manifest = build_run_manifest(
        protocol,
        run_kind="legacy-exploratory-design-study",
        effective_run_settings={
            "profile": args.profile,
            "settings": settings,
            "target_tube": args.target_tube,
            "seed": args.seed,
            "baseline_only": args.baseline_only,
            "skip_optimization": args.skip_optimization,
        },
        protocol_deviations=protocol_deviations,
        project_root=PROJECT_ROOT,
    )
    study_manifest.update(
        {
            "profile": args.profile,
            "settings": settings,
            "target_tube": args.target_tube,
            "seed": args.seed,
            "definitions": {
                "dexterity": "legacy positional isotropy of a design-dependent normalised-state Jacobian",
                "ik_error": "legacy local numerical IK residual from zero deployment and zero rotation",
                "workspace": "legacy axisymmetric 99.5%-radial-quantile envelope",
                "bounds": "exploratory only; not certified manufacturing limits",
            },
            "variables": [asdict(variable) for variable in variables],
        }
    )
    manifest_path = output / "study_manifest.json"
    write_run_manifest(manifest_path, study_manifest)

    def complete_manifest(completion: str) -> None:
        study_manifest["run_status"] = "completed"
        study_manifest["completion"] = completion
        write_run_manifest(manifest_path, study_manifest)

    print(f"Running {args.profile} CTR design study in {output}")
    print("IK error means numerical residual from a fully retracted initial state.")
    baseline_evaluation = evaluate_design(
        baseline,
        workspace_sample_count=settings["workspace"],
        dexterity_sample_count=settings["dexterity"],
        ik_targets_mm=targets,
        target_tube=args.target_tube,
        seed=args.seed,
    )
    save_baseline_outputs(output, baseline_evaluation, settings)
    print("Baseline maps complete.")

    if args.baseline_only:
        complete_manifest("legacy baseline-only outputs generated")
        print("Baseline-only study complete.")
        return

    sensitivity_targets = targets[: settings["sensitivity_ik"]]

    def sensitivity_evaluator(candidate):
        return evaluate_design(
            candidate,
            workspace_sample_count=settings["sensitivity_workspace"],
            dexterity_sample_count=settings["sensitivity_dexterity"],
            ik_targets_mm=sensitivity_targets,
            target_tube=args.target_tube,
            seed=args.seed + 10,
        )

    sensitivity_records = sensitivity_study(
        baseline, variables, sensitivity_evaluator
    )
    for record in sensitivity_records:
        record["dexterity_uniformity_objective"] = (
            -float(record["dexterity_mean"]) + 0.35 * float(record["dexterity_cv"])
        )
    write_records(output / "parameter_sensitivity.csv", sensitivity_records)
    plot_sensitivity(sensitivity_records, variables, output)
    print(f"Sensitivity complete ({len(sensitivity_records)} design evaluations).")

    if args.skip_optimization:
        complete_manifest("legacy baseline and sensitivity outputs generated")
        print("Optimisation skipped by request.")
        return

    variable_lookup = {variable.name: variable for variable in variables}
    variable_limit = settings["optimisation_variables"]
    dexterity_names = rank_sensitivity_variables(
        sensitivity_records, "dexterity_uniformity_objective"
    )[:variable_limit]
    error_names = rank_sensitivity_variables(
        sensitivity_records, "ik_rmse_mm"
    )[:variable_limit]
    optimisation_targets = targets[: settings["optimisation_ik"]]

    def optimisation_evaluator(candidate):
        return evaluate_design(
            candidate,
            workspace_sample_count=settings["optimisation_workspace"],
            dexterity_sample_count=settings["optimisation_dexterity"],
            ik_targets_mm=optimisation_targets,
            target_tube=args.target_tube,
            seed=args.seed + 20,
        )

    optimisation_baseline = optimisation_evaluator(baseline)
    reference_volume = optimisation_baseline.metrics["workspace_volume_mm3"]
    reference_rmse = max(optimisation_baseline.metrics["ik_rmse_mm"], 0.15)

    def volume_penalty(evaluation: DesignEvaluation) -> float:
        ratio = evaluation.metrics["workspace_volume_mm3"] / reference_volume
        return 2.0 * max(0.0, 0.80 - ratio)

    def dexterity_objective(evaluation: DesignEvaluation) -> float:
        metrics = evaluation.metrics
        return (
            -metrics["dexterity_mean"]
            + 0.35 * metrics["dexterity_cv"]
            + volume_penalty(evaluation)
        )

    def error_objective(evaluation: DesignEvaluation) -> float:
        metrics = evaluation.metrics
        return (
            metrics["ik_rmse_mm"] / reference_rmse
            + 2.0 * (1.0 - metrics["ik_success_rate"])
            + volume_penalty(evaluation)
        )

    dexterity_result = optimise_design(
        baseline,
        [variable_lookup[name] for name in dexterity_names],
        optimisation_evaluator,
        dexterity_objective,
        seed=args.seed + 30,
        max_iterations=settings["optimisation_iterations"],
        population_size=settings["optimisation_population"],
    )
    error_result = optimise_design(
        baseline,
        [variable_lookup[name] for name in error_names],
        optimisation_evaluator,
        error_objective,
        seed=args.seed + 40,
        max_iterations=settings["optimisation_iterations"],
        population_size=settings["optimisation_population"],
    )
    write_json(output / "dexterity_optimisation.json", optimisation_summary(dexterity_result))
    write_json(output / "error_optimisation.json", optimisation_summary(error_result))

    # Re-evaluate final candidates at the baseline map resolution so the
    # comparison uses identical target sets, seeds and plot scales.
    dexterity_final = evaluate_design(
        dexterity_result.design,
        workspace_sample_count=settings["workspace"],
        dexterity_sample_count=settings["dexterity"],
        ik_targets_mm=targets,
        target_tube=args.target_tube,
        seed=args.seed,
    )
    error_final = evaluate_design(
        error_result.design,
        workspace_sample_count=settings["workspace"],
        dexterity_sample_count=settings["dexterity"],
        ik_targets_mm=targets,
        target_tube=args.target_tube,
        seed=args.seed,
    )
    write_json(output / "dexterity_optimised_final.json", evaluation_summary(dexterity_final))
    write_json(output / "error_optimised_final.json", evaluation_summary(error_final))
    plot_comparison(
        [
            ("Baseline", baseline_evaluation),
            (
                "Dexterity optimum" if dexterity_result.success else "Dexterity-objective candidate",
                dexterity_final,
            ),
            (
                "IK residual optimum" if error_result.success else "IK-residual-objective candidate",
                error_final,
            ),
        ],
        output,
    )
    complete_manifest("legacy exploratory optimisation outputs generated")
    print("Exploratory optimisation and matched comparison complete.")


if __name__ == "__main__":
    main()
