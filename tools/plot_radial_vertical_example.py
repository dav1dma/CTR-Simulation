"""Plot a high-Z, low-radial-distance CTR configuration from a design study."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CTR_superPosKin_fun_sectioned import superPosKin  # noqa: E402
from ctr_design_analysis import baseline_design, shared_configuration_samples  # noqa: E402


TUBE_NAMES = ("Inner", "Middle", "Outer")
TUBE_COLOURS = ("#0d52e3", "#0da647", "#f26b0a")


def polyline_interval(points: np.ndarray, start_mm: float, end_mm: float) -> np.ndarray:
    distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    start = float(np.clip(start_mm, 0.0, distance[-1]))
    end = float(np.clip(end_mm, start, distance[-1]))
    targets = np.linspace(start, end, 60)
    return np.column_stack(
        [np.interp(targets, distance, points[:, axis]) for axis in range(3)]
    )


def backbone_mm(parameters: dict, deployment: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = superPosKin(
        parameters,
        {"ul": deployment.tolist(), "uphi": rotation.tolist()},
        {"n_p": 30, "isPlot": False},
    )
    parts = []
    for index, section in enumerate(result[3]):
        part = np.column_stack([np.asarray(section[axis]) for axis in range(3)])
        parts.append(part if index == 0 else part[1:])
    return np.vstack(parts) * 1000.0


def equal_3d_axes(axis, points: np.ndarray) -> None:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    centre = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.58, 20.0)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(max(0.0, centre[2] - radius), centre[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.15))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        default=PROJECT_ROOT / "results" / "design_analysis_standard",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radial-limit-mm", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    csv_path = args.study / "baseline_dexterity.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    positions = np.asarray(
        [[float(row[key]) for key in ("x_mm", "y_mm", "z_mm")] for row in rows]
    )
    isotropy = np.asarray([float(row["positional_isotropy"]) for row in rows])
    radial = np.linalg.norm(positions[:, :2], axis=1)

    design = baseline_design()
    deployment, rotation = shared_configuration_samples(
        design.total_length_mm * 1e-3,
        len(positions),
        args.seed + 1,
    )

    # Keep every tube visibly deployed so the example explains the complete
    # robot rather than selecting the trivial fully straight inner-only state.
    minimum_deployment_m = np.asarray([0.20, 0.10, 0.05])
    candidates = (radial <= args.radial_limit_mm) & np.all(
        deployment >= minimum_deployment_m, axis=1
    )
    if not np.any(candidates):
        raise RuntimeError("No sampled configuration satisfies the selection criteria.")
    selected = np.flatnonzero(candidates)[np.argmax(positions[candidates, 2])]

    points = backbone_mm(
        design.to_parameters(), deployment[selected], rotation[selected]
    )
    deployed_mm = deployment[selected] * 1000.0
    segments = (
        polyline_interval(points, deployed_mm[1], deployed_mm[0]),
        polyline_interval(points, deployed_mm[2], deployed_mm[1]),
        polyline_interval(points, 0.0, deployed_mm[2]),
    )

    figure = plt.figure(figsize=(14.5, 6.4), constrained_layout=True)
    shape_axis = figure.add_subplot(1, 2, 1, projection="3d")
    map_axis = figure.add_subplot(1, 2, 2)

    shape_axis.plot(
        [0.0, 0.0], [0.0, 0.0], [0.0, max(positions[selected, 2], 1.0)],
        linestyle="--", color="0.55", linewidth=1.5, label="Vertical reference",
    )
    for tube in (2, 1, 0):
        segment = segments[tube]
        shape_axis.plot(
            segment[:, 0], segment[:, 1], segment[:, 2],
            color=TUBE_COLOURS[tube], linewidth=5.0 - tube,
            label=f"{TUBE_NAMES[tube]} tube",
        )
    tip = positions[selected]
    shape_axis.scatter(*tip, color="#d7191c", edgecolor="black", s=65, zorder=10)
    shape_axis.scatter(0.0, 0.0, 0.0, marker="x", color="black", s=55)
    shape_axis.set_xlabel("X (mm)")
    shape_axis.set_ylabel("Y (mm)")
    shape_axis.set_zlabel("Z (mm)")
    shape_axis.set_title("Representative tube configuration")
    shape_axis.view_init(elev=22.0, azim=-55.0)
    equal_3d_axes(shape_axis, np.vstack((points, [[0.0, 0.0, 0.0]])))
    shape_axis.legend(loc="upper left", fontsize=8)

    cloud = map_axis.scatter(
        positions[:, 2], radial, c=isotropy, cmap="viridis",
        vmin=0.0, vmax=1.0, s=8, alpha=0.28, linewidths=0.0,
    )
    map_axis.scatter(
        tip[2], radial[selected], marker="*", s=230,
        color="#d7191c", edgecolor="black", linewidth=0.8,
        label="Plotted configuration", zorder=5,
    )
    map_axis.axhline(args.radial_limit_mm, color="0.5", linestyle="--", linewidth=1.0)
    map_axis.set_xlabel("Z (mm)")
    map_axis.set_ylabel(r"Radial distance $r=\sqrt{x^2+y^2}$ (mm)")
    map_axis.set_title("Location in the radial–vertical workspace")
    map_axis.grid(alpha=0.18)
    map_axis.legend(loc="upper left")
    figure.colorbar(cloud, ax=map_axis, label=r"Positional isotropy $\sigma_{min}/\sigma_{max}$")

    figure.suptitle(
        "High-Z, Low-Radial-Distance CTR Example\n"
        f"Tip = ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}) mm, "
        f"r = {radial[selected]:.1f} mm, isotropy = {isotropy[selected]:.3f}"
    )
    output = args.output or args.study / "high_z_low_radial_configuration.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)

    print(f"Saved {output.resolve()}")
    print(f"Sample index: {selected}")
    print(f"Tip XYZ (mm): {np.round(tip, 3).tolist()}")
    print(f"Radial distance (mm): {radial[selected]:.3f}")
    print(f"Positional isotropy: {isotropy[selected]:.6f}")
    print(f"Deployment (mm): {np.round(deployed_mm, 3).tolist()}")
    print(f"Rotation (deg): {np.round(np.degrees(rotation[selected]), 3).tolist()}")


if __name__ == "__main__":
    main()
