#!/usr/bin/env python3
"""Generate cached inner/middle/outer endpoint workspace samples."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_workspace_map import (  # noqa: E402
    DEFAULT_ENDPOINT_WORKSPACE_PATH,
    generate_endpoint_workspace_maps,
    load_workspace_map,
    save_endpoint_workspace_maps,
)
from tube_parameters import build_supervisor_ctr_parameters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate endpoint-specific CTR workspace samples."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ENDPOINT_WORKSPACE_PATH,
    )
    arguments = parser.parse_args()

    source = load_workspace_map()
    started = time.perf_counter()
    endpoint_maps = generate_endpoint_workspace_maps(
        build_supervisor_ctr_parameters(),
        source,
    )
    save_endpoint_workspace_maps(endpoint_maps, arguments.output)
    elapsed = time.perf_counter() - started
    print(
        f"Saved {endpoint_maps.tips_mm.shape[1]:,} states for three endpoints "
        f"to {arguments.output} in {elapsed:.1f} s."
    )


if __name__ == "__main__":
    main()
