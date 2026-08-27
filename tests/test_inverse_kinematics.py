"""Executable checks for constrained CTR tip-position inverse kinematics."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_inverse_kinematics import ConstrainedTipIK
from tube_parameters import build_supervisor_ctr_parameters


def assert_valid_deployment(solver: ConstrainedTipIK, deployment: np.ndarray) -> None:
    solver.validate_deployment(deployment)
    assert deployment[2] <= deployment[1] <= deployment[0]


def run_checks() -> None:
    solver = ConstrainedTipIK(build_supervisor_ctr_parameters())
    deployment = np.array([0.120, 0.070, 0.040])
    rotation = np.zeros(3)

    encoded = solver.encode_deployment(deployment)
    assert np.allclose(solver.decode_deployment(encoded), deployment)

    initial_tip = solver.forward_tip_mm(deployment, rotation)
    endpoints = solver.forward_endpoints_mm(deployment, rotation)
    assert endpoints.shape == (3, 3)
    assert np.allclose(endpoints[0], initial_tip)
    for tube in range(3):
        stationary_endpoint = solver.solve(
            endpoints[tube],
            deployment,
            rotation,
            target_tube=tube,
        )
        assert stationary_endpoint.reached
        assert stationary_endpoint.target_tube == tube
        assert np.allclose(
            stationary_endpoint.achieved_tip_mm,
            endpoints[tube],
        )

    stationary = solver.solve(initial_tip, deployment, rotation)
    assert stationary.reached
    assert stationary.position_error_mm < 1e-9
    assert np.allclose(stationary.deployment_m, deployment)
    assert np.allclose(stationary.rotation_rad, rotation)

    for displacement in np.identity(3):
        result = solver.solve(initial_tip + displacement, deployment, rotation)
        assert result.reached, (displacement, result.position_error_mm)
        assert result.position_error_mm <= solver.tolerance_mm
        assert_valid_deployment(solver, result.deployment_m)

    known_deployment = np.array([0.124, 0.073, 0.041])
    known_rotation = np.deg2rad([0.0, 2.0, -2.0])
    known_tip = solver.forward_tip_mm(known_deployment, known_rotation)
    recovered = solver.solve(known_tip, deployment, rotation)
    assert recovered.reached, recovered.position_error_mm
    assert_valid_deployment(solver, recovered.deployment_m)

    locked = solver.solve(
        initial_tip + np.asarray([1.0, 0.0, 0.0]),
        deployment,
        rotation,
        target_tube=0,
        locked_targets_mm={2: endpoints[2]},
    )
    assert locked.reached, locked.endpoint_errors_mm
    assert locked.endpoint_errors_mm[0] <= solver.tolerance_mm
    assert locked.endpoint_errors_mm[2] <= 2.0

    unreachable = solver.solve(
        initial_tip + np.array([1000.0, 0.0, 0.0]), deployment, rotation
    )
    assert not unreachable.reached
    assert_valid_deployment(solver, unreachable.deployment_m)

    # Follow a small three-dimensional closed path using each accepted state as
    # the next warm start, matching the interactive resolved-target workflow.
    path_deployment = deployment.copy()
    path_rotation = rotation.copy()
    path_results = []
    for angle in np.linspace(0.0, 2.0 * np.pi, 81)[1:]:
        target = initial_tip + np.array(
            [
                3.0 * np.cos(angle),
                3.0 * np.sin(angle),
                1.5 * np.sin(2.0 * angle),
            ]
        )
        path_result = solver.solve(target, path_deployment, path_rotation)
        assert path_result.reached, (angle, path_result.position_error_mm)
        assert_valid_deployment(solver, path_result.deployment_m)
        path_deployment = path_result.deployment_m
        path_rotation = path_result.rotation_rad
        path_results.append(path_result)

    maximum_path_error = max(item.position_error_mm for item in path_results)
    assert maximum_path_error <= solver.tolerance_mm

    print(
        "Inverse-kinematics checks passed; "
        f"1 mm target solve: {result.solve_time_ms:.2f} ms; "
        f"80-step path maximum error: {maximum_path_error:.3f} mm"
    )


if __name__ == "__main__":
    run_checks()
