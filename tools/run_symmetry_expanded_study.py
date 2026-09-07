"""Run the legacy quadrant-reduced CTR rotational diagnostic."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_design_analysis import (  # noqa: E402
    DesignEvaluation,
    DexterityMap,
    IKErrorMap,
    aggregate_ik_residual_voxels,
    baseline_design,
    canonical_configuration_samples,
    calculate_dexterity_map,
    calculate_ik_error_map,
    expand_first_quadrant_points,
    fold_configurations_to_first_quadrant,
    independent_baseline_target_set,
    position_jacobian,
    positional_isotropy,
    rotate_about_z,
    summarise_dexterity,
    summarise_ik_errors,
    workspace_size_metrics,
)
from ctr_inverse_kinematics import ConstrainedTipIK, wrap_angles  # noqa: E402
from tools.run_design_study import (  # noqa: E402
    equal_3d_axes,
    plot_filled_voxels,
    plot_xy_slices,
    save_baseline_outputs,
)


def write_records(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _solve_ik_batch(parameters: dict, targets: np.ndarray, target_tube: int) -> IKErrorMap:
    return calculate_ik_error_map(parameters, targets, target_tube=target_tube)


def solve_ik_parallel(
    parameters: dict,
    targets: np.ndarray,
    *,
    target_tube: int,
    workers: int,
    canonical_sample_ids: np.ndarray | None = None,
) -> IKErrorMap:
    """Solve independent target batches and restore their original ordering."""
    targets = np.asarray(targets, dtype=float)
    batch_count = max(workers * 4, 1)
    indexed_batches = [
        (index, batch)
        for index, batch in enumerate(np.array_split(targets, batch_count))
        if len(batch)
    ]
    results: dict[int, IKErrorMap] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_solve_ik_batch, parameters, batch, target_tube): index
            for index, batch in indexed_batches
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            print(
                f"IK batch {completed}/{len(indexed_batches)} complete",
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


def symmetry_expand_dexterity(
    base: DexterityMap,
) -> tuple[DexterityMap, DexterityMap, np.ndarray]:
    base_ids = (
        np.arange(len(base.positions_mm), dtype=np.int64)
        if base.canonical_sample_ids is None
        else np.asarray(base.canonical_sample_ids)
    )
    folded_positions, folded_rotations, source_quadrants = (
        fold_configurations_to_first_quadrant(
            base.positions_mm,
            base.rotation_rad,
        )
    )
    quadrant_one = DexterityMap(
        positions_mm=folded_positions,
        isotropy=base.isotropy.copy(),
        singular_values=base.singular_values.copy(),
        deployment_m=base.deployment_m.copy(),
        rotation_rad=folded_rotations,
        canonical_sample_ids=base_ids.copy(),
    )
    expanded_positions, expanded_quadrants = expand_first_quadrant_points(
        folded_positions
    )
    expanded_rotations = np.vstack(
        [
            wrap_angles(folded_rotations - quadrant * 0.5 * np.pi)
            for quadrant in range(4)
        ]
    ).astype(np.float32)
    expanded = DexterityMap(
        positions_mm=expanded_positions,
        isotropy=np.tile(base.isotropy, 4),
        singular_values=np.tile(base.singular_values, (4, 1)),
        deployment_m=np.tile(base.deployment_m, (4, 1)),
        rotation_rad=expanded_rotations,
        canonical_sample_ids=np.tile(base_ids, 4),
    )
    return quadrant_one, expanded, np.column_stack(
        (np.tile(source_quadrants, 4), expanded_quadrants)
    )


def validate_symmetry(
    parameters: dict,
    deployment_m: np.ndarray,
    rotation_rad: np.ndarray,
    *,
    target_tube: int,
    sample_count: int = 32,
) -> dict[str, float]:
    solver = ConstrainedTipIK(parameters, model_points_per_section=2)
    position_errors = []
    isotropy_errors = []
    for deployment, rotation in zip(
        deployment_m[:sample_count], rotation_rad[:sample_count]
    ):
        base_tip = solver.forward_endpoint_mm(deployment, rotation, target_tube)
        base_score, _ = positional_isotropy(
            position_jacobian(
                solver, deployment, rotation, target_tube=target_tube
            )
        )
        for quadrant in (1, 2, 3):
            angle = quadrant * 0.5 * np.pi
            rotated_state = wrap_angles(rotation - angle)
            actual_tip = solver.forward_endpoint_mm(
                deployment, rotated_state, target_tube
            )
            expected_tip = rotate_about_z(base_tip, angle)
            score, _ = positional_isotropy(
                position_jacobian(
                    solver,
                    deployment,
                    rotated_state,
                    target_tube=target_tube,
                )
            )
            position_errors.append(float(np.linalg.norm(actual_tip - expected_tip)))
            isotropy_errors.append(abs(float(score - base_score)))
    return {
        "validation_configurations": int(min(sample_count, len(deployment_m))),
        "maximum_rotated_tip_disagreement_mm": float(np.max(position_errors)),
        "maximum_rotated_isotropy_disagreement": float(np.max(isotropy_errors)),
    }


def plot_workspace(points: np.ndarray, path: Path, title: str) -> None:
    values = np.asarray(points, dtype=float)
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        values[:, 0], values[:, 1], values[:, 2],
        s=3.0, alpha=0.24, color="#2386bd", linewidths=0.0,
    )
    axis.scatter([0.0], [0.0], [0.0], marker="x", color="black", s=55)
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(title)
    equal_3d_axes(axis, values)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_quadrant_xy(points: np.ndarray, path: Path, sample_count: int) -> None:
    values = np.asarray(points, dtype=float)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.scatter(values[:, 0], values[:, 1], s=2.0, alpha=0.18, color="#2386bd")
    axis.axhline(0.0, color="0.45", linewidth=1.0)
    axis.axvline(0.0, color="0.45", linewidth=1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_title(
        f"{sample_count:,} Independent Base Configurations Folded to Quadrant 1"
    )
    axis.grid(alpha=0.16)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_ik_quadrant_comparison(
    residual_mm: np.ndarray,
    reached: np.ndarray,
    quadrants: np.ndarray,
    path: Path,
) -> None:
    residual = np.asarray(residual_mm, dtype=float)
    success = np.asarray(reached, dtype=bool)
    quadrant_ids = np.asarray(quadrants, dtype=int)
    means = []
    medians = []
    success_rates = []
    for quadrant in range(4):
        selected = quadrant_ids == quadrant
        means.append(float(np.mean(residual[selected])))
        medians.append(float(np.median(residual[selected])))
        success_rates.append(float(np.mean(success[selected])))
    x = np.arange(4)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, means, width, label="Mean residual")
    axes[0].bar(x + width / 2, medians, width, label="Median residual")
    axes[0].set_ylabel("Numerical IK residual (mm)")
    axes[0].set_xticks(x, [f"Q{value}" for value in range(1, 5)])
    axes[0].set_title("Residual by target quadrant")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.18)
    axes[1].bar(x, 100.0 * np.asarray(success_rates), width=0.55)
    axes[1].set_ylabel("Targets within 0.15 mm (%)")
    axes[1].set_xticks(x, [f"Q{value}" for value in range(1, 5)])
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_title("Local IK success by target quadrant")
    axes[1].grid(axis="y", alpha=0.18)
    figure.suptitle("Fixed-Start Numerical IK Directional Behaviour")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_ik_voxel_outputs(
    output: Path,
    errors: IKErrorMap,
    *,
    voxel_size_mm: float,
    minimum_median_samples: int,
) -> dict[str, float | int]:
    voxels = aggregate_ik_residual_voxels(
        errors,
        voxel_size_mm=voxel_size_mm,
        minimum_median_samples=minimum_median_samples,
    )
    write_records(
        output / "baseline_ik_residual_voxels.csv",
        [
            {
                "voxel_x_mm": float(centre[0]),
                "voxel_y_mm": float(centre[1]),
                "voxel_z_mm": float(centre[2]),
                "maximum_numerical_ik_residual_mm": float(maximum),
                "median_numerical_ik_residual_mm": float(median),
                "target_count": int(count),
            }
            for centre, maximum, median, count in zip(
                voxels.centres_mm,
                voxels.maximum_residual_mm,
                voxels.median_residual_mm,
                voxels.sample_count,
            )
        ],
    )
    limits = (0.0, float(np.max(errors.residual_mm)))
    for aggregation, values, description in (
        (
            "maximum",
            voxels.maximum_residual_mm,
            "Worst independently solved target in each voxel",
        ),
        (
            "median",
            voxels.median_residual_mm,
            f"Typical independently solved target; at least {minimum_median_samples} targets",
        ),
    ):
        plot_filled_voxels(
            voxels,
            values,
            output / f"baseline_ik_residual_{aggregation}_voxel.png",
            title=f"{aggregation.title()} Local IK Voxel Residual\n{description}",
            colour_label="Numerical IK residual (mm)",
            cmap="magma",
            limits=limits,
        )
        plot_xy_slices(
            voxels,
            values,
            output / f"baseline_ik_residual_{aggregation}_xy_slices.png",
            title=f"{aggregation.title()} Local IK Residual Cross-Sections",
            colour_label="Numerical IK residual (mm)",
            cmap="magma",
            limits=limits,
        )
    plot_filled_voxels(
        voxels,
        voxels.sample_count,
        output / "baseline_ik_residual_voxel_occupancy.png",
        title="Independently Solved IK Target Occupancy",
        colour_label="Targets per occupied voxel",
        cmap="plasma",
        logarithmic=True,
    )
    valid_median = np.isfinite(voxels.median_residual_mm)
    return {
        "ik_occupied_voxels": int(len(voxels.sample_count)),
        "ik_median_reported_voxels": int(np.sum(valid_median)),
        "ik_mean_targets_per_occupied_voxel": float(np.mean(voxels.sample_count)),
        "ik_maximum_targets_per_voxel": int(np.max(voxels.sample_count)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-configurations", type=int, default=15_000)
    parser.add_argument("--voxel-size-mm", type=float, default=15.0)
    parser.add_argument("--minimum-median-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tube", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--skip-ik", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "symmetry_expanded_15000",
    )
    args = parser.parse_args()
    if args.base_configurations < 8:
        raise ValueError("base-configurations must be at least 8")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    design = baseline_design()
    parameters = design.to_parameters()
    lengths_m = design.total_length_mm * 1e-3
    print(
        f"Calculating {args.base_configurations:,} independent configurations...",
        flush=True,
    )
    configuration_bank, deployment, rotation = canonical_configuration_samples(
        lengths_m,
        args.base_configurations,
        seed=args.seed + 1,
        split="training-symmetry-diagnostic",
    )
    base_dexterity = calculate_dexterity_map(
        parameters,
        deployment,
        rotation,
        target_tube=args.target_tube,
        canonical_sample_ids=configuration_bank.sample_ids,
    )
    quadrant_one, expanded_dexterity, _symmetry_labels = symmetry_expand_dexterity(
        base_dexterity
    )
    print(
        f"Expanded to {len(expanded_dexterity.positions_mm):,} render-only positions.",
        flush=True,
    )

    symmetry_validation = validate_symmetry(
        parameters,
        deployment,
        rotation,
        target_tube=args.target_tube,
    )
    plot_workspace(
        quadrant_one.positions_mm,
        output / f"quadrant1_workspace_{args.base_configurations}.png",
        f"Quadrant-1 Workspace — {args.base_configurations:,} Independent Configurations",
    )
    plot_quadrant_xy(
        quadrant_one.positions_mm,
        output / f"quadrant1_xy_{args.base_configurations}.png",
        args.base_configurations,
    )
    plot_workspace(
        expanded_dexterity.positions_mm,
        output / f"symmetry_expanded_workspace_{len(expanded_dexterity.positions_mm)}.png",
        f"Symmetry-Expanded Workspace — {len(expanded_dexterity.positions_mm):,} Display Positions",
    )
    write_records(
        output / "quadrant1_configurations.csv",
        [
            {
                "base_index": index,
                "canonical_sample_id": str(quadrant_one.canonical_sample_ids[index]),
                "x_mm": float(point[0]),
                "y_mm": float(point[1]),
                "z_mm": float(point[2]),
                "positional_isotropy": float(isotropy),
                "inner_deployment_mm": float(deployed[0] * 1000.0),
                "middle_deployment_mm": float(deployed[1] * 1000.0),
                "outer_deployment_mm": float(deployed[2] * 1000.0),
                "inner_rotation_deg": float(np.degrees(rotated[0])),
                "middle_rotation_deg": float(np.degrees(rotated[1])),
                "outer_rotation_deg": float(np.degrees(rotated[2])),
            }
            for index, (point, isotropy, deployed, rotated) in enumerate(
                zip(
                    quadrant_one.positions_mm,
                    quadrant_one.isotropy,
                    quadrant_one.deployment_m,
                    quadrant_one.rotation_rad,
                )
            )
        ],
    )

    ik_errors = None
    if not args.skip_ik:
        print(
            f"Generating {args.base_configurations:,} independent quadrant-1 IK targets...",
            flush=True,
        )
        base_target_set = independent_baseline_target_set(
            args.base_configurations,
            target_tube=args.target_tube,
            seed=args.seed + 1000,
            split="training-symmetry-ik-diagnostic",
        )
        folded_targets, _, _ = fold_configurations_to_first_quadrant(
            base_target_set.targets_mm,
            np.zeros((len(base_target_set.targets_mm), 3), dtype=float),
        )
        expanded_targets, target_quadrants = expand_first_quadrant_points(
            folded_targets
        )
        expanded_target_ids = np.tile(
            base_target_set.canonical_sample_ids,
            4,
        )
        print(
            f"Solving {len(expanded_targets):,} rotated repeated-measure targets "
            f"with {args.workers} workers (not statistically independent)...",
            flush=True,
        )
        ik_errors = solve_ik_parallel(
            parameters,
            expanded_targets,
            target_tube=args.target_tube,
            workers=args.workers,
            canonical_sample_ids=expanded_target_ids,
        )
        plot_ik_quadrant_comparison(
            ik_errors.residual_mm,
            ik_errors.reached,
            target_quadrants,
            output / "baseline_ik_residual_quadrant_comparison.png",
        )
        write_records(
            output / "ik_target_quadrants.csv",
            [
                {
                    "target_index": index,
                    "base_index": index % args.base_configurations,
                    "canonical_target_id": str(expanded_target_ids[index]),
                    "quadrant": int(target_quadrants[index] + 1),
                    "target_x_mm": float(target[0]),
                    "target_y_mm": float(target[1]),
                    "target_z_mm": float(target[2]),
                    "numerical_ik_residual_mm": float(residual),
                    "reached_0_15_mm_tolerance": bool(reached),
                }
                for index, (target, residual, reached) in enumerate(
                    zip(ik_errors.targets_mm, ik_errors.residual_mm, ik_errors.reached)
                )
            ],
        )

    workspace_volume, workspace_area = workspace_size_metrics(
        expanded_dexterity.positions_mm
    )
    metrics = {
        "workspace_volume_mm3": workspace_volume,
        "workspace_volume_cm3": workspace_volume / 1000.0,
        "workspace_surface_area_mm2": workspace_area,
        "workspace_surface_area_cm2": workspace_area / 100.0,
        **summarise_dexterity(expanded_dexterity),
        **summarise_ik_errors(ik_errors),
    }
    evaluation = DesignEvaluation(
        design=design,
        metrics=metrics,
        workspace_tips_mm=expanded_dexterity.positions_mm,
        dexterity=expanded_dexterity,
        ik_errors=ik_errors,
    )
    settings = {
        "voxel_size_mm": args.voxel_size_mm,
        "minimum_median_samples": args.minimum_median_samples,
    }
    save_baseline_outputs(output, evaluation, settings)
    ik_voxel_summary = (
        save_ik_voxel_outputs(
            output,
            ik_errors,
            voxel_size_mm=args.voxel_size_mm,
            minimum_median_samples=args.minimum_median_samples,
        )
        if ik_errors is not None
        else {}
    )

    summary = {
        "method": (
            f"{args.base_configurations:,} independent configurations folded into "
            "quadrant 1 and expanded by proper rotations of 0, 90, 180 and 270 degrees"
        ),
        "independent_base_configurations": int(args.base_configurations),
        "symmetry_expanded_display_positions": int(
            len(expanded_dexterity.positions_mm)
        ),
        "display_positions_are_independent_configurations": False,
        "canonical_ik_targets": int(
            len(np.unique(ik_errors.canonical_sample_ids))
            if ik_errors is not None
            else 0
        ),
        "rotated_ik_solves": int(
            len(ik_errors.targets_mm) if ik_errors is not None else 0
        ),
        "rotated_ik_solves_are_independent_targets": False,
        "ik_initialisation": (
            "one fixed actuator configuration: all tubes fully retracted "
            "with zero rotation"
        ),
        "voxel_size_mm": float(args.voxel_size_mm),
        "minimum_median_samples": int(args.minimum_median_samples),
        "symmetry_validation": symmetry_validation,
        "metrics": metrics,
        **ik_voxel_summary,
    }
    (output / "symmetry_study_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Saved symmetry-expanded study to {output}", flush=True)


if __name__ == "__main__":
    main()
