"""Generate cached blue/red/grey reachability zones for the live viewer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_workspace_map import (  # noqa: E402
    DEFAULT_REACHABILITY_ZONES_PATH,
    DEFAULT_SEED,
    DEFAULT_ZONE_SAMPLE_COUNT,
    generate_reachability_zones,
    generate_workspace_map,
    save_reachability_zones,
)
from tube_parameters import build_supervisor_ctr_parameters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_ZONE_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--voxel-mm", type=float, default=8.0)
    parser.add_argument("--reachable-mm", type=float, default=10.0)
    parser.add_argument("--uncertain-mm", type=float, default=18.0)
    parser.add_argument("--smoothing", type=float, default=1.15)
    parser.add_argument(
        "--cutaway",
        action="store_true",
        help="remove the +X/+Y quadrant for an internal diagnostic view",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REACHABILITY_ZONES_PATH,
    )
    arguments = parser.parse_args()

    started = time.perf_counter()
    print(
        f"Sampling {arguments.samples:,} valid tube configurations...",
        flush=True,
    )
    workspace = generate_workspace_map(
        build_supervisor_ctr_parameters(),
        sample_count=arguments.samples,
        seed=arguments.seed,
    )
    print("Classifying and meshing the voxel zones...", flush=True)
    zones = generate_reachability_zones(
        workspace,
        voxel_size_mm=arguments.voxel_mm,
        reachable_distance_mm=arguments.reachable_mm,
        uncertain_distance_mm=arguments.uncertain_mm,
        smoothing_sigma_voxels=arguments.smoothing,
        cutaway_quadrant=arguments.cutaway,
    )
    save_reachability_zones(zones, arguments.output)
    elapsed = time.perf_counter() - started
    total = (
        zones.reachable_voxels
        + zones.uncertain_voxels
        + zones.inaccessible_voxels
    )
    print(f"Saved: {arguments.output}")
    print(
        "Visible classified voxels: "
        f"blue {zones.reachable_voxels:,}, "
        f"grey {zones.uncertain_voxels:,}, "
        f"red {zones.inaccessible_voxels:,} "
        f"(total {total:,})"
    )
    print(f"Completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
