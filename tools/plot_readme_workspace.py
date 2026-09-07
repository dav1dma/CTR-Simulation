"""Reproduce the README figure from the endpoint cache shipped with the viewer."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ctr_workspace_map import load_endpoint_workspace_maps


def main() -> None:
    workspace = load_endpoint_workspace_maps()
    count = workspace.tips_mm.shape[1]
    colors = ("#167b9b", "#d68a20", "#bd5264")
    names = ("Inner endpoint", "Middle endpoint", "Outer endpoint")
    fig = plt.figure(figsize=(15, 6.8), facecolor="white")
    fig.suptitle("Concentric-tube robot · sampled endpoint workspaces",
                 x=0.055, y=0.96, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.055, 0.89,
             f"{count:,} shared actuator configurations · current section-aware model · coordinates in mm",
             fontsize=11, color="#526170")
    # Identical limits and aspect ratios allow direct comparison of endpoint reach.
    radial_limit = np.ceil(np.abs(workspace.tips_mm[:, :, :2]).max() / 50) * 50
    z_min = min(0, np.floor(workspace.tips_mm[:, :, 2].min() / 50) * 50)
    z_max = np.ceil(workspace.tips_mm[:, :, 2].max() / 50) * 50
    for tube, (name, color) in enumerate(zip(names, colors)):
        ax = fig.add_subplot(1, 3, tube + 1, projection="3d")
        points = workspace.tips_mm[tube]
        ax.scatter(*points.T, s=2, alpha=0.45, color=color,
                   edgecolors="none", depthshade=False, rasterized=True)
        ax.scatter([0], [0], [0], marker="+", s=65, color="#172b3a", linewidths=1.6)
        ax.set(title=name, xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
               xlim=(-radial_limit, radial_limit), ylim=(-radial_limit, radial_limit),
               zlim=(z_min, z_max))
        ax.set_box_aspect((2 * radial_limit, 2 * radial_limit, z_max - z_min))
        ax.view_init(elev=24, azim=-58)
        ax.set_xticks([-radial_limit, 0, radial_limit])
        ax.set_yticks([-radial_limit, 0, radial_limit])
        ax.tick_params(labelsize=8)
        ax.title.set_color(color)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"].update(color="#dce3e8", linewidth=0.6)
    fig.subplots_adjust(left=0.025, right=0.97, bottom=0.16, top=0.8, wspace=0.14)
    fig.text(0.055, 0.075, "Same scale in all panels. + marks the plate origin (Z = 0).",
             fontsize=10, color="#334958")
    fig.text(0.055, 0.035,
             "Viewer cache samples illustrate reach; they do not certify continuous reachability or represent formal analysis sample counts.",
             fontsize=9, color="#526170")
    output = PROJECT_ROOT / "docs/images/ctr_workspace_sectioned.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
