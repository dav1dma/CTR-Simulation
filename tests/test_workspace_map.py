"""Executable checks for the sampled reachable-tip workspace map."""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_workspace_map import (  # noqa: E402
    DEFAULT_ENDPOINT_WORKSPACE_PATH,
    DEFAULT_REACHABILITY_ZONES_PATH,
    DEFAULT_WORKSPACE_MAP_PATH,
    constrain_point_to_workspace_surface,
    generate_endpoint_workspace_maps,
    generate_reachability_zones,
    generate_workspace_map,
    load_endpoint_workspace_maps,
    load_reachability_zones,
    load_workspace_map,
    radial_workspace_envelope,
    sample_valid_configurations,
    workspace_surface_of_revolution,
)
from tube_parameters import (  # noqa: E402
    build_supervisor_ctr_parameters,
    total_tube_lengths,
)


def run_checks() -> None:
    parameters = build_supervisor_ctr_parameters()
    lengths = total_tube_lengths(parameters)
    deployment, rotation = sample_valid_configurations(lengths, 64, seed=7)

    assert deployment.shape == (64, 3)
    assert rotation.shape == (64, 3)
    assert np.all(deployment >= 0.0)
    assert np.all(deployment <= lengths + 1e-12)
    assert np.all(deployment[:, 2] <= deployment[:, 1])
    assert np.all(deployment[:, 1] <= deployment[:, 0])
    assert np.any(deployment < lengths * 0.5)
    assert np.any(np.isclose(deployment, lengths, atol=1e-12))
    assert np.all(rotation >= -np.pi) and np.all(rotation <= np.pi)

    small_map = generate_workspace_map(parameters, sample_count=16, seed=8)
    assert small_map.tips_mm.shape == (16, 3)
    assert np.all(np.isfinite(small_map.tips_mm))

    stored_map = load_workspace_map(DEFAULT_WORKSPACE_MAP_PATH)
    assert len(stored_map.tips_mm) >= 10_000
    assert np.ptp(stored_map.tips_mm[:, 0]) > 400.0
    assert np.ptp(stored_map.tips_mm[:, 1]) > 400.0
    assert np.ptp(stored_map.tips_mm[:, 2]) > 250.0

    profile_z, profile_radius = radial_workspace_envelope(
        stored_map.tips_mm,
        height_sections=24,
    )
    assert profile_z.shape == profile_radius.shape == (26,)
    assert np.all(np.diff(profile_z) > 0.0)
    assert profile_radius[0] == 0.0 and profile_radius[-1] == 0.0
    assert np.max(profile_radius) > 250.0

    surface = workspace_surface_of_revolution(
        stored_map.tips_mm,
        height_sections=24,
        azimuth_sections=32,
    )
    assert surface.vertices_mm.shape == (24 * 32 + 2, 3)
    assert surface.faces.shape == (2 * 24 * 32, 3)
    assert np.all(surface.faces < len(surface.vertices_mm))
    assert np.allclose(surface.vertices_mm[[0, -1], :2], 0.0)

    inside = np.asarray([0.0, 0.0, np.mean(surface.profile_z_mm)])
    unchanged, limited = constrain_point_to_workspace_surface(
        inside,
        surface.profile_z_mm,
        surface.profile_radius_mm,
    )
    assert not limited and np.allclose(unchanged, inside)
    outside = np.asarray([1000.0, 0.0, np.mean(surface.profile_z_mm)])
    projected, limited = constrain_point_to_workspace_surface(
        outside,
        surface.profile_z_mm,
        surface.profile_radius_mm,
    )
    assert limited
    allowed = np.interp(
        projected[2],
        surface.profile_z_mm,
        surface.profile_radius_mm,
    )
    assert np.hypot(projected[0], projected[1]) <= allowed + 1e-5

    small_endpoints = generate_endpoint_workspace_maps(parameters, small_map)
    assert small_endpoints.tips_mm.shape == (3, 16, 3)
    assert np.all(np.isfinite(small_endpoints.tips_mm))

    stored_endpoints = load_endpoint_workspace_maps(DEFAULT_ENDPOINT_WORKSPACE_PATH)
    assert stored_endpoints.tips_mm.shape[0] == 3
    assert stored_endpoints.tips_mm.shape[1] >= 10_000
    assert np.max(np.linalg.norm(stored_endpoints.tips_mm[2], axis=1)) <= (
        lengths[2] * 1000.0 + 1e-3
    )

    coarse_zones = generate_reachability_zones(
        stored_map,
        voxel_size_mm=20.0,
        reachable_distance_mm=24.0,
        uncertain_distance_mm=40.0,
    )
    assert coarse_zones.reachable_voxels > 0
    assert coarse_zones.uncertain_voxels > 0
    assert coarse_zones.inaccessible_voxels > 0

    stored_zones = load_reachability_zones(DEFAULT_REACHABILITY_ZONES_PATH)
    assert stored_zones.sample_count >= 80_000
    assert stored_zones.voxel_size_mm == 8.0
    assert not stored_zones.cutaway_quadrant
    for mesh in (
        stored_zones.reachable,
        stored_zones.uncertain,
        stored_zones.inaccessible,
    ):
        assert len(mesh.vertices_mm) > 0
        assert len(mesh.faces) > 0
        assert np.all(mesh.faces < len(mesh.vertices_mm))

    print(
        "Workspace-map, endpoint-envelope, and diagnostic-zone checks passed "
        f"({stored_endpoints.tips_mm.shape[1]:,} endpoint states; "
        f"{stored_zones.sample_count:,} zone samples)."
    )


if __name__ == "__main__":
    run_checks()
    constrain_point_to_workspace_surface,
    generate_endpoint_workspace_maps,
    load_endpoint_workspace_maps,
