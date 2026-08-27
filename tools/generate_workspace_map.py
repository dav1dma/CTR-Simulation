"""Regenerate the compact workspace map used by the VisPy tip viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_workspace_map import (  # noqa: E402
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_WORKSPACE_MAP_PATH,
    generate_workspace_map,
    save_workspace_map,
)
from tube_parameters import build_supervisor_ctr_parameters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_WORKSPACE_MAP_PATH,
    )
    arguments = parser.parse_args()

    print(f"Generating {arguments.samples:,} reachable-tip samples...")
    workspace = generate_workspace_map(
        build_supervisor_ctr_parameters(),
        sample_count=arguments.samples,
        seed=arguments.seed,
    )
    save_workspace_map(workspace, arguments.output)
    limits_min = workspace.tips_mm.min(axis=0)
    limits_max = workspace.tips_mm.max(axis=0)
    print(f"Saved: {arguments.output}")
    print(
        "Sampled XYZ limits (mm): "
        f"X {limits_min[0]:.1f}..{limits_max[0]:.1f}, "
        f"Y {limits_min[1]:.1f}..{limits_max[1]:.1f}, "
        f"Z {limits_min[2]:.1f}..{limits_max[2]:.1f}"
    )


if __name__ == "__main__":
    main()
