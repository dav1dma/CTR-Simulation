"""Reaggregate and redraw an existing symmetry-expanded CTR study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_design_analysis import (  # noqa: E402
    DexterityMap,
    IKErrorMap,
    aggregate_dexterity_voxels,
)
from tools.run_design_study import (  # noqa: E402
    plot_cylindrical_dexterity,
    plot_filled_voxels,
    plot_xy_slices,
    write_records,
)
from tools.run_symmetry_expanded_study import (  # noqa: E402
    plot_ik_quadrant_comparison,
    plot_workspace,
    save_ik_voxel_outputs,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        default=PROJECT_ROOT / "results" / "symmetry_expanded_15000",
    )
    parser.add_argument("--voxel-size-mm", type=float, default=15.0)
    parser.add_argument("--minimum-median-samples", type=int, default=5)
    args = parser.parse_args()
    output = args.study.resolve()

    dexterity_rows = read_csv(output / "baseline_dexterity.csv")
    positions = np.asarray(
        [[float(row[key]) for key in ("x_mm", "y_mm", "z_mm")] for row in dexterity_rows]
    )
    dexterity = DexterityMap(
        positions_mm=positions,
        isotropy=np.asarray(
            [float(row["positional_isotropy"]) for row in dexterity_rows]
        ),
        singular_values=np.asarray(
            [
                [float(row[key]) for key in ("sigma_max", "sigma_middle", "sigma_min")]
                for row in dexterity_rows
            ]
        ),
        deployment_m=np.empty((len(positions), 3)),
        rotation_rad=np.empty((len(positions), 3)),
    )
    settings = {
        "voxel_size_mm": args.voxel_size_mm,
        "minimum_median_samples": args.minimum_median_samples,
    }
    voxels = aggregate_dexterity_voxels(
        dexterity,
        voxel_size_mm=args.voxel_size_mm,
        minimum_median_samples=args.minimum_median_samples,
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
            f"Typical configuration: at least {args.minimum_median_samples} samples per voxel",
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

    quadrant_rows = read_csv(output / "quadrant1_configurations.csv")
    quadrant_points = np.asarray(
        [[float(row[key]) for key in ("x_mm", "y_mm", "z_mm")] for row in quadrant_rows]
    )
    plot_workspace(
        quadrant_points,
        output / f"quadrant1_workspace_{len(quadrant_points)}.png",
        f"Quadrant-1 Workspace — {len(quadrant_points):,} Independent Configurations",
    )
    plot_workspace(
        positions,
        output / f"symmetry_expanded_workspace_{len(positions)}.png",
        f"Symmetry-Expanded Workspace — {len(positions):,} Spatial Configurations",
    )

    ik_rows = read_csv(output / "baseline_ik_residual.csv")
    errors = IKErrorMap(
        targets_mm=np.asarray(
            [[float(row[f"target_{axis}_mm"]) for axis in "xyz"] for row in ik_rows]
        ),
        achieved_mm=np.asarray(
            [[float(row[f"achieved_{axis}_mm"]) for axis in "xyz"] for row in ik_rows]
        ),
        residual_mm=np.asarray(
            [float(row["numerical_ik_residual_mm"]) for row in ik_rows]
        ),
        reached=np.asarray(
            [row["reached_0_15_mm_tolerance"].lower() == "true" for row in ik_rows]
        ),
        iterations=np.asarray([int(row["iterations"]) for row in ik_rows]),
        evaluations=np.asarray([int(row["evaluations"]) for row in ik_rows]),
        solve_time_ms=np.asarray([float(row["solve_time_ms"]) for row in ik_rows]),
    )
    ik_summary = save_ik_voxel_outputs(
        output,
        errors,
        voxel_size_mm=args.voxel_size_mm,
        minimum_median_samples=args.minimum_median_samples,
    )
    quadrant_rows = read_csv(output / "ik_target_quadrants.csv")
    plot_ik_quadrant_comparison(
        np.asarray([float(row["numerical_ik_residual_mm"]) for row in quadrant_rows]),
        np.asarray(
            [row["reached_0_15_mm_tolerance"].lower() == "true" for row in quadrant_rows]
        ),
        np.asarray([int(row["quadrant"]) - 1 for row in quadrant_rows]),
        output / "baseline_ik_residual_quadrant_comparison.png",
    )

    summary_path = output / "symmetry_study_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["minimum_median_samples"] = int(args.minimum_median_samples)
    summary.update(ik_summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Replotted {output} with median threshold {args.minimum_median_samples}.")


if __name__ == "__main__":
    main()
