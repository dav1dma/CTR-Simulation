"""Reproduce the README's CTR tube and reachable-workspace illustration."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.animation import PillowWriter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ctr_workspace_map import load_endpoint_workspace_maps, radial_workspace_envelope
from tube_parameters import build_supervisor_ctr_parameters
from tools.plot_radial_vertical_example import backbone_mm, polyline_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", action="store_true", help="Also render a looping camera orbit GIF.")
    args = parser.parse_args()
    workspace = load_endpoint_workspace_maps()
    samples = workspace.tips_mm[0]
    profile_z, profile_radius = radial_workspace_envelope(samples)
    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    field_x = profile_radius[:, None] * np.cos(angles)
    field_y = profile_radius[:, None] * np.sin(angles)
    field_z = np.broadcast_to(profile_z[:, None], field_x.shape)
    deployment_mm = np.array([270.0, 150.0, 75.0])
    rotation_deg = np.array([0.0, 65.0, -20.0])
    backbone = backbone_mm(build_supervisor_ctr_parameters(),
                           deployment_mm * 1e-3, np.deg2rad(rotation_deg))
    # The outermost tube present at each arclength is the visible tube surface.
    segments = (
        polyline_interval(backbone, deployment_mm[1], deployment_mm[0]),
        polyline_interval(backbone, deployment_mm[2], deployment_mm[1]),
        polyline_interval(backbone, 0.0, deployment_mm[2]),
    )
    colors = ("#2166d1", "#22966c", "#ed8936")
    fig = plt.figure(figsize=(11, 9), facecolor="#f8fafc")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#f8fafc")
    # Use the same smooth axisymmetric envelope as the interactive viewer.
    ax.plot_surface(field_x, field_y, field_z, color="#559ab7", alpha=0.20,
                    rstride=1, cstride=1, linewidth=0, edgecolor="none",
                    antialiased=True, shade=False, rasterized=True, zorder=1)
    # Emphasise the robot against the field; widths are illustrative, not to scale.
    for tube in (2, 1, 0):
        segment = segments[tube]
        width = (4.5, 7.0, 10.0)[tube]
        ax.plot(*segment.T, color="white", linewidth=width + 2,
                solid_capstyle="round", zorder=3)
        ax.plot(*segment.T, color=colors[tube], linewidth=width,
                solid_capstyle="round", zorder=4)
    ax.scatter(*backbone[-1], s=55, color=colors[0], edgecolor="white",
               linewidth=1.5, depthshade=False, zorder=5)
    ax.scatter(0, 0, 0, s=95, marker="+", color="#23354a", linewidth=2, zorder=5)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
           xlim=(-300, 300), ylim=(-300, 300), zlim=(0, 350))
    ax.set_box_aspect((600, 600, 350))
    ax.view_init(elev=23, azim=-58)
    ax.set_xticks([-300, -150, 0, 150, 300])
    ax.set_yticks([-300, -150, 0, 150, 300])
    ax.tick_params(labelsize=9, colors="#536579")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"].update(color="#dce5ed", linewidth=0.6)
    fig.text(0.065, 0.95, "CTR tubes & reachable workspace", fontsize=23,
             fontweight="bold", color="#1c3046")
    fig.text(0.065, 0.91, "One simulated robot configuration inside the inner-tip workspace field",
             fontsize=12, color="#536579")
    handles = [Line2D([0], [0], color=color, linewidth=5, label=name)
               for color, name in zip(colors, ("Inner tube", "Middle tube", "Outer tube"))]
    handles.append(Patch(facecolor="#559ab7", alpha=0.30, edgecolor="none",
                         label="Smooth workspace field"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.89),
               frameon=False, ncol=2, fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.94, bottom=0.12, top=0.84)
    fig.text(0.065, 0.073, "Plate origin: +   |   Tube thickness exaggerated for visibility",
             fontsize=10, color="#536579")
    fig.text(0.065, 0.04, "Envelope estimated from 12,000 model samples; not a guarantee of reach at every interior point.",
             fontsize=9, color="#536579")
    output = PROJECT_ROOT / "docs/images/ctr_workspace_sectioned.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    print(f"Saved {output}", flush=True)
    if args.gif:
        animation_path = output.with_suffix(".gif")
        writer = PillowWriter(fps=12)
        with writer.saving(fig, str(animation_path), dpi=85):
            for frame in range(72):
                ax.view_init(elev=23, azim=-58 + frame * 5)
                writer.grab_frame(facecolor=fig.get_facecolor())
                if frame % 18 == 0:
                    print(f"Rendered frame {frame + 1}/72", flush=True)
        print(f"Saved {animation_path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
