"""Executable checks for confirmed-target planning and smooth execution."""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_inverse_kinematics import ConstrainedTipIK  # noqa: E402
from ctr_motion_planner import (  # noqa: E402
    DIRECT_STRATEGY,
    RETRACT_STRATEGY,
    TipTargetPlanner,
    build_motion_route,
    interpolate_actuators,
    motion_duration_s,
    route_state_at_time,
    sample_motion_route,
    sample_planned_tip_path,
    shortest_angle_delta,
)
from ctr_workspace_map import (  # noqa: E402
    load_endpoint_workspace_maps,
    load_workspace_map,
)
from tube_parameters import build_supervisor_ctr_parameters  # noqa: E402


def run_checks() -> None:
    parameters = build_supervisor_ctr_parameters()
    local_solver = ConstrainedTipIK(parameters)
    restart_solver = ConstrainedTipIK(
        parameters,
        max_iterations=12,
        damping_mm=1.0,
        max_normalised_step=0.10,
    )
    planner = TipTargetPlanner(
        local_solver,
        restart_solver,
        load_workspace_map(),
    )

    deployment = np.asarray([0.120, 0.070, 0.040])
    rotation = np.zeros(3)
    original_deployment = deployment.copy()
    original_rotation = rotation.copy()
    initial_tip = local_solver.forward_tip_mm(deployment, rotation)

    endpoint_maps = load_endpoint_workspace_maps()
    outer_planner = TipTargetPlanner(
        local_solver,
        restart_solver,
        endpoint_maps.for_tube(2),
        target_tube=2,
    )
    outer_target = local_solver.forward_endpoint_mm(deployment, rotation, 2)
    outer_attempt = outer_planner.plan(outer_target, deployment, rotation)
    assert outer_attempt.plan is not None
    assert outer_attempt.plan.target_tube == 2
    outer_route = build_motion_route(outer_attempt.plan, DIRECT_STRATEGY)
    outer_path = sample_motion_route(
        restart_solver,
        outer_route,
        samples_per_phase=4,
    )
    assert np.allclose(outer_path[-1], outer_target, atol=1e-6)

    local_attempt = planner.plan(
        initial_tip + np.asarray([5.0, 3.0, 2.0]),
        deployment,
        rotation,
    )
    assert local_attempt.plan is not None
    assert local_attempt.plan.solution_source == "LOCAL IK"
    assert local_attempt.alternatives
    assert local_attempt.alternatives[0] is local_attempt.plan
    assert len(local_attempt.alternatives) >= 2
    assert len(local_attempt.alternatives) <= 6
    for alternative in local_attempt.alternatives:
        assert alternative.target_tube == 0
        assert alternative.position_error_mm <= restart_solver.tolerance_mm

    # This target stalls on the initial local branch but is reached from a
    # nearby valid configuration in the sampled workspace.
    restart_target = initial_tip + np.asarray([-80.0, -60.0, 0.0])
    restart_attempt = planner.plan(restart_target, deployment, rotation)
    assert restart_attempt.plan is not None, restart_attempt.message
    assert restart_attempt.plan.solution_source == "WORKSPACE RESTART"
    assert restart_attempt.plan.position_error_mm <= restart_solver.tolerance_mm

    # Planning must never move the live actuator arrays.
    assert np.allclose(deployment, original_deployment)
    assert np.allclose(rotation, original_rotation)

    plan = restart_attempt.plan
    start_deployment, start_rotation = interpolate_actuators(plan, 0.0)
    goal_deployment, goal_rotation = interpolate_actuators(plan, 1.0)
    assert np.allclose(start_deployment, plan.start_deployment_m)
    assert np.allclose(start_rotation, plan.start_rotation_rad)
    assert np.allclose(goal_deployment, plan.goal_deployment_m)
    assert np.allclose(goal_rotation, plan.goal_rotation_rad)

    for progress in np.linspace(0.0, 1.0, 21):
        intermediate_deployment, _rotation = interpolate_actuators(plan, progress)
        restart_solver.validate_deployment(intermediate_deployment)

    path = sample_planned_tip_path(restart_solver, plan, sample_count=16)
    assert path.shape == (16, 3)
    assert np.linalg.norm(path[-1] - restart_target) <= restart_solver.tolerance_mm
    assert 0.4 <= motion_duration_s(plan) <= 8.0

    direct_route = build_motion_route(plan, DIRECT_STRATEGY)
    assert len(direct_route.phases) == 1
    direct_goal = route_state_at_time(direct_route, direct_route.total_duration_s)
    assert np.allclose(direct_goal[0], plan.goal_deployment_m)
    assert np.allclose(direct_goal[1], plan.goal_rotation_rad)

    retract_route = build_motion_route(plan, RETRACT_STRATEGY)
    assert len(retract_route.phases) == 3
    assert np.allclose(retract_route.phases[0].goal_deployment_m, 0.0)
    assert np.allclose(retract_route.phases[1].start_deployment_m, 0.0)
    assert np.allclose(retract_route.phases[1].goal_deployment_m, 0.0)
    assert np.allclose(
        retract_route.phases[-1].goal_deployment_m,
        plan.goal_deployment_m,
    )
    for elapsed in np.linspace(0.0, retract_route.total_duration_s, 61):
        route_deployment, _route_rotation, progress, phase_index, phase_progress = (
            route_state_at_time(retract_route, elapsed)
        )
        restart_solver.validate_deployment(route_deployment)
        assert 0.0 <= progress <= 1.0
        assert 0 <= phase_index < 3
        assert 0.0 <= phase_progress <= 1.0

    retract_path = sample_motion_route(
        restart_solver,
        retract_route,
        samples_per_phase=16,
    )
    assert retract_path.shape == (46, 3)
    assert np.min(np.linalg.norm(retract_path, axis=1)) < 1e-9
    assert np.linalg.norm(retract_path[-1] - restart_target) <= (
        restart_solver.tolerance_mm
    )

    wrapped_delta = np.rad2deg(
        shortest_angle_delta(
            np.deg2rad([170.0, 0.0, 0.0]),
            np.deg2rad([-170.0, 0.0, 0.0]),
        )
    )
    assert np.allclose(wrapped_delta, [20.0, 0.0, 0.0])

    unreachable = planner.plan(
        initial_tip + np.asarray([1000.0, 0.0, 0.0]),
        deployment,
        rotation,
    )
    assert unreachable.plan is None

    print(
        "Confirmed-target motion-planner checks passed; "
        f"restart solve {restart_attempt.solve_time_ms:.1f} ms; "
        "direct and retract routes valid."
    )


if __name__ == "__main__":
    run_checks()
