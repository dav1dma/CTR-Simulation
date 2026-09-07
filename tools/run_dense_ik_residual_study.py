"""Generate dense point-cloud and voxel plots of local numerical IK residual."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_design_analysis import (  # noqa: E402
    aggregate_ik_residual_voxels,
    baseline_design,
    calculate_ik_error_map,
    independent_baseline_targets,
    summarise_ik_errors,
)
from tools.run_design_study import (  # noqa: E402
    plot_3d_heatmap,
    plot_filled_voxels,
    plot_xy_slices,
)


def write_records(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--voxel-size-mm", type=float, default=20.0)
    parser.add_argument("--minimum-median-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--target-tube", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "ik_residual_dense_10000",
    )
    args = parser.parse_args()
    if args.samples < 8:
        raise ValueError("samples must be at least 8")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    design = baseline_design()
    parameters = design.to_parameters()
    print(f"Generating {args.samples:,} independent reachable targets...")
    targets = independent_baseline_targets(
        args.samples,
        target_tube=args.target_tube,
        seed=args.seed,
    )
    print("Solving every target from the fully retracted initial state...")
    errors = calculate_ik_error_map(
        parameters,
        targets,
        target_tube=args.target_tube,
    )
    voxels = aggregate_ik_residual_voxels(
        errors,
        voxel_size_mm=args.voxel_size_mm,
        minimum_median_samples=args.minimum_median_samples,
    )

    residual_records = [
        {
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
        for target, achieved, residual, reached, iterations, evaluations, solve_time in zip(
            errors.targets_mm,
            errors.achieved_mm,
            errors.residual_mm,
            errors.reached,
            errors.iterations,
            errors.evaluations,
            errors.solve_time_ms,
        )
    ]
    write_records(output / "ik_residual_dense.csv", residual_records)

    voxel_records = [
        {
            "voxel_x_mm": float(centre[0]),
            "voxel_y_mm": float(centre[1]),
            "voxel_z_mm": float(centre[2]),
            "maximum_numerical_ik_residual_mm": float(maximum),
            "median_numerical_ik_residual_mm": float(median),
            "target_count": int(count),
            "voxel_size_mm": voxels.voxel_size_mm,
            "minimum_median_samples": voxels.minimum_median_samples,
        }
        for centre, maximum, median, count in zip(
            voxels.centres_mm,
            voxels.maximum_residual_mm,
            voxels.median_residual_mm,
            voxels.sample_count,
        )
    ]
    write_records(output / "ik_residual_voxels.csv", voxel_records)

    maximum_residual = float(np.max(errors.residual_mm))
    colour_limits = (0.0, maximum_residual)
    plot_3d_heatmap(
        errors.targets_mm,
        errors.residual_mm,
        output / "ik_residual_point_cloud.png",
        title=(
            "Local IK Residual from a Fixed Fully Retracted Configuration — "
            f"{args.samples:,} Targets"
        ),
        colour_label="Numerical IK residual (mm)",
        cmap="magma",
        limits=colour_limits,
    )
    for aggregation, values, description in (
        (
            "maximum",
            voxels.maximum_residual_mm,
            "Worst residual observed in each occupied voxel",
        ),
        (
            "median",
            voxels.median_residual_mm,
            f"Typical residual; at least {voxels.minimum_median_samples} targets per voxel",
        ),
    ):
        plot_filled_voxels(
            voxels,
            values,
            output / f"ik_residual_{aggregation}_voxel.png",
            title=f"{aggregation.title()} Local IK Voxel Residual\n{description}",
            colour_label="Numerical IK residual (mm)",
            cmap="magma",
            limits=colour_limits,
        )
        plot_xy_slices(
            voxels,
            values,
            output / f"ik_residual_{aggregation}_xy_slices.png",
            title=f"{aggregation.title()} Local IK Residual Cross-Sections",
            colour_label="Numerical IK residual (mm)",
            cmap="magma",
            limits=colour_limits,
        )
    plot_filled_voxels(
        voxels,
        voxels.sample_count,
        output / "ik_residual_voxel_occupancy.png",
        title="IK Target-Sample Occupancy",
        colour_label="Targets per occupied voxel",
        cmap="plasma",
        logarithmic=True,
    )

    valid_median = np.isfinite(voxels.median_residual_mm)
    counts = np.asarray(voxels.sample_count, dtype=int)
    summary = {
        "definition": (
            "Numerical endpoint residual after local IK from the fully retracted "
            "state; this is not measured physical model error."
        ),
        "samples": int(args.samples),
        "seed": int(args.seed),
        "target_tube": int(args.target_tube),
        "voxel_size_mm": float(args.voxel_size_mm),
        "minimum_median_samples": int(args.minimum_median_samples),
        "occupied_voxels": int(len(counts)),
        "median_reported_voxels": int(np.sum(valid_median)),
        "single_sample_voxel_fraction": float(np.mean(counts == 1)),
        "mean_targets_per_occupied_voxel": float(np.mean(counts)),
        "maximum_targets_per_voxel": int(np.max(counts)),
        "metrics": summarise_ik_errors(errors),
    }
    (output / "ik_residual_dense_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved dense IK residual study to {output}")


if __name__ == "__main__":
    main()
