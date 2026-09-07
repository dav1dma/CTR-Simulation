"""Stage-2 checks for Jacobians, spatial aggregation and IK evaluation."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_design_analysis import (  # noqa: E402
    BASELINE_JACOBIAN_ACTUATOR_SCALE,
    IKErrorMap,
    baseline_design,
    canonical_configuration_samples,
    position_jacobian,
    position_jacobians_all_endpoints,
    positional_isotropy,
    rotate_about_z,
    summarise_ik_errors,
    within_evaluation_threshold,
)
from ctr_inverse_kinematics import (  # noqa: E402
    IK_TERMINATION_REASONS,
    ConstrainedTipIK,
)
from ctr_spatial_analysis import (  # noqa: E402
    AxisymmetricGrid,
    aggregate_axisymmetric_scalar,
    aggregate_axisymmetric_residual,
    aggregate_axisymmetric_workspace,
    compare_axisymmetric_workspaces,
    summarise_axisymmetric_scalar,
)
from ctr_task_space import (  # noqa: E402
    freeze_baseline_task_region,
    select_stratified_canonical_targets,
)


def _grid(radial_cells: int = 3, z_cells: int = 3) -> AxisymmetricGrid:
    radial = np.arange(radial_cells + 1, dtype=float) * 10.0
    z_values = np.arange(z_cells + 1, dtype=float) * 10.0
    volume = np.pi * (
        radial[1:] ** 2 - radial[:-1] ** 2
    )[:, None] * np.diff(z_values)[None, :]
    return AxisymmetricGrid(radial, z_values, volume)


def _point(cell: tuple[int, int], azimuth: float = 0.0) -> np.ndarray:
    radius = (cell[0] + 0.5) * 10.0
    z_value = (cell[1] + 0.5) * 10.0
    return np.asarray(
        [radius * np.cos(azimuth), radius * np.sin(azimuth), z_value],
        dtype=float,
    )


def _expect_value_error(callable_object) -> None:
    try:
        callable_object()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def run_checks() -> None:
    design = baseline_design()
    parameters = design.to_parameters()
    bank, deployment, rotation = canonical_configuration_samples(
        design.total_length_mm * 1e-3,
        16,
        seed=1042,
        split="stage2-jacobian-test",
    )
    solver = ConstrainedTipIK(parameters, model_points_per_section=2)

    # Physical derivatives are central, all-endpoint and scaled once by the
    # immutable baseline reference rather than each candidate's own lengths.
    result = position_jacobians_all_endpoints(
        solver,
        deployment[7],
        rotation[7],
    )
    assert result.valid
    assert result.raw_jacobians.shape == result.scaled_jacobians.shape == (3, 3, 6)
    assert np.array_equal(result.actuator_scale, BASELINE_JACOBIAN_ACTUATOR_SCALE)
    assert np.allclose(
        result.scaled_jacobians,
        result.raw_jacobians * BASELINE_JACOBIAN_ACTUATOR_SCALE[None, None, :],
    )
    assert np.allclose(
        position_jacobian(solver, deployment[7], rotation[7], target_tube=1),
        result.scaled_jacobians[1],
    )
    boundary = position_jacobians_all_endpoints(
        solver,
        np.zeros(3),
        np.zeros(3),
    )
    assert not boundary.valid and boundary.reason == "constraint-boundary"
    assert np.all(np.isnan(boundary.scaled_jacobians))

    refined = position_jacobians_all_endpoints(
        solver,
        deployment[7],
        rotation[7],
        translation_step_mm=0.03,
        rotation_step_rad=3e-4,
    )
    relative_change = np.linalg.norm(
        result.scaled_jacobians - refined.scaled_jacobians
    ) / np.linalg.norm(refined.scaled_jacobians)
    assert relative_change < 0.005
    for endpoint in range(3):
        coarse_isotropy = positional_isotropy(result.scaled_jacobians[endpoint])[0]
        fine_isotropy = positional_isotropy(refined.scaled_jacobians[endpoint])[0]
        assert abs(coarse_isotropy - fine_isotropy) < 0.002

    arbitrary_angle = 0.731
    rotated_state = (rotation[7] - arbitrary_angle + np.pi) % (2.0 * np.pi) - np.pi
    rotated = position_jacobians_all_endpoints(
        solver,
        deployment[7],
        rotated_state,
    )
    for endpoint in range(3):
        expected = rotate_about_z(
            result.scaled_jacobians[endpoint].T,
            arbitrary_angle,
        ).T
        assert np.allclose(rotated.scaled_jacobians[endpoint], expected, atol=1e-4)
        assert np.isclose(
            positional_isotropy(rotated.scaled_jacobians[endpoint])[0],
            positional_isotropy(result.scaled_jacobians[endpoint])[0],
            atol=1e-8,
        )

    # Exact swept-annular volumes, internal voids and unique canonical counts.
    grid = _grid()
    assert np.isclose(grid.cell_volume_mm3[0, 0], 1000.0 * np.pi)
    assert np.isclose(np.sum(grid.cell_volume_mm3), np.pi * 30.0**2 * 30.0)
    ring_cells = [
        (radial_index, z_index)
        for radial_index in range(3)
        for z_index in range(3)
        if (radial_index, z_index) != (1, 1)
    ]
    ring_points = np.asarray([_point(cell) for cell in ring_cells])
    ring_ids = np.asarray([f"ring-{index}" for index in range(len(ring_points))])
    ring = aggregate_axisymmetric_workspace(
        ring_points,
        canonical_sample_ids=ring_ids,
        grid=grid,
    )
    assert ring.internal_void[1, 1]
    assert ring.internal_void_component_count == 1
    assert np.isclose(ring.internal_void_volume_mm3, 3000.0 * np.pi)
    opened_cells = [cell for cell in ring_cells if cell != (2, 1)]
    opened = aggregate_axisymmetric_workspace(
        np.asarray([_point(cell) for cell in opened_cells]),
        canonical_sample_ids=np.asarray(
            [f"open-{index}" for index in range(len(opened_cells))]
        ),
        grid=grid,
    )
    assert not np.any(opened.internal_void)
    axial_cavity_cells = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]
    axial = aggregate_axisymmetric_workspace(
        np.asarray([_point(cell) for cell in axial_cavity_cells]),
        canonical_sample_ids=np.asarray(
            [f"axis-{index}" for index in range(len(axial_cavity_cells))]
        ),
        grid=grid,
    )
    assert axial.internal_void[0, 1]

    copy_angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    copied_points = np.vstack(
        [np.asarray([_point(cell, angle) for cell in ring_cells]) for angle in copy_angles]
    )
    copied_ids = np.tile(ring_ids, len(copy_angles))
    copied = aggregate_axisymmetric_workspace(
        copied_points,
        canonical_sample_ids=copied_ids,
        grid=grid,
    )
    assert np.array_equal(copied.sample_count, ring.sample_count)
    assert np.array_equal(copied.occupied, ring.occupied)
    conflicting_points = np.asarray([_point((0, 0)), _point((1, 0))])
    _expect_value_error(
        lambda: aggregate_axisymmetric_workspace(
            conflicting_points,
            canonical_sample_ids=np.asarray(["same", "same"]),
            grid=grid,
        )
    )

    comparison = compare_axisymmetric_workspaces(
        np.asarray([_point((0, 0)), _point((1, 0))]),
        np.asarray([_point((1, 0)), _point((2, 0))]),
        reference_sample_ids=np.asarray(["r0", "r1"]),
        candidate_sample_ids=np.asarray(["c1", "c2"]),
        radial_bin_mm=10.0,
        z_bin_mm=10.0,
    )
    assert np.isclose(comparison.reference_coverage_fraction, 0.75)
    assert np.isclose(comparison.volume_weighted_jaccard, 1.0 / 3.0)

    scalar_points = np.asarray([_point((0, 0)), _point((1, 0))])
    scalar_map = aggregate_axisymmetric_scalar(
        scalar_points,
        np.asarray([1.0, 3.0]),
        canonical_sample_ids=np.asarray(["s0", "s1"]),
        grid=grid,
        minimum_median_samples=1,
        minimum_lower_tail_samples=1,
    )
    scalar_summary = summarise_axisymmetric_scalar(scalar_map)
    assert np.isclose(scalar_summary["volume_weighted_mean"], 2.5)
    assert np.isclose(scalar_summary["volume_weighted_median"], 3.0)

    fifty_points = np.asarray(
        [[5.0 + 1e-3 * index, 0.0, 5.0] for index in range(50)]
    )
    fifty_values = np.linspace(0.0, 1.0, 50)
    fifty_ids = np.asarray([f"m-{index}" for index in range(50)])
    support_29 = aggregate_axisymmetric_scalar(
        fifty_points[:29],
        fifty_values[:29],
        canonical_sample_ids=fifty_ids[:29],
        grid=grid,
        minimum_median_samples=30,
        minimum_lower_tail_samples=50,
    )
    assert np.isnan(support_29.median[0, 0])
    support_30 = aggregate_axisymmetric_scalar(
        fifty_points[:30],
        fifty_values[:30],
        canonical_sample_ids=fifty_ids[:30],
        grid=grid,
        minimum_median_samples=30,
        minimum_lower_tail_samples=50,
    )
    assert np.isfinite(support_30.median[0, 0])
    assert np.isnan(support_30.lower_p10[0, 0])
    support_50 = aggregate_axisymmetric_scalar(
        fifty_points,
        fifty_values,
        canonical_sample_ids=fifty_ids,
        grid=grid,
        minimum_median_samples=30,
        minimum_lower_tail_samples=50,
    )
    assert np.isfinite(support_50.lower_p10[0, 0])

    one_hundred_points = np.asarray(
        [[5.0 + 1e-3 * index, 0.0, 5.0] for index in range(100)]
    )
    one_hundred_residuals = np.linspace(0.0, 1.0, 100)
    one_hundred_ids = np.asarray([f"ik-{index}" for index in range(100)])
    residual_99 = aggregate_axisymmetric_residual(
        one_hundred_points[:99],
        one_hundred_residuals[:99],
        canonical_target_ids=one_hundred_ids[:99],
        grid=grid,
        minimum_median_samples=30,
        minimum_failure_or_p95_samples=100,
    )
    assert np.isfinite(residual_99.median_residual_mm[0, 0])
    assert np.isnan(residual_99.p95_residual_mm[0, 0])
    assert np.isnan(residual_99.failure_fraction[0, 0])
    residual_100 = aggregate_axisymmetric_residual(
        one_hundred_points,
        one_hundred_residuals,
        canonical_target_ids=one_hundred_ids,
        grid=grid,
        minimum_median_samples=30,
        minimum_failure_or_p95_samples=100,
    )
    assert np.isfinite(residual_100.p95_residual_mm[0, 0])
    assert np.isclose(residual_100.failure_fraction[0, 0], 0.5)

    # Task discovery and target candidates are independent; selected targets
    # are canonical, prefix-stable and equally allocated until exhaustion.
    discovery_cells = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    discovery_points = np.asarray([_point(cell) for cell in discovery_cells])
    task_region = freeze_baseline_task_region(
        discovery_points,
        np.asarray([f"discovery-{i}" for i in range(len(discovery_points))]),
        target_tube=0,
        baseline_parameter_sha256="a" * 64,
        source_dataset_id="workspace-discovery-training",
        grid=grid,
    )
    candidate_points = []
    for cell in discovery_cells:
        for offset in (-1.0, 0.0, 1.0):
            base_point = _point(cell, azimuth=0.35)
            base_point[2] += offset
            candidate_points.append(base_point)
    candidate_points = np.asarray(candidate_points)
    candidate_count = len(candidate_points)
    candidate_deployment = np.column_stack(
        (
            np.linspace(0.10, 0.20, candidate_count),
            np.linspace(0.06, 0.10, candidate_count),
            np.linspace(0.02, 0.04, candidate_count),
        )
    )
    candidate_rotation = np.column_stack(
        (
            np.linspace(-1.0, 1.0, candidate_count),
            np.linspace(0.5, -0.5, candidate_count),
            np.zeros(candidate_count),
        )
    )
    candidate_ids = np.asarray([f"candidate-{i:03d}" for i in range(candidate_count)])
    target_8 = select_stratified_canonical_targets(
        task_region,
        candidate_points,
        candidate_deployment,
        candidate_rotation,
        candidate_ids,
        target_count=8,
        split="training",
        seed=77,
        source_dataset_id="ik-candidates-training",
    )
    target_12 = select_stratified_canonical_targets(
        task_region,
        candidate_points,
        candidate_deployment,
        candidate_rotation,
        candidate_ids,
        target_count=12,
        split="training",
        seed=77,
        source_dataset_id="ik-candidates-training",
    )
    assert np.array_equal(target_8.target_ids, target_12.target_ids[:8])
    assert np.array_equal(target_8.source_sample_ids, target_12.source_sample_ids[:8])
    assert np.all(target_12.targets_mm[:, 0] >= 0.0)
    assert np.allclose(target_12.targets_mm[:, 1], 0.0)
    selected_counts = target_12.targets_per_cell[target_12.targets_per_cell > 0]
    assert np.ptp(selected_counts) <= 1
    represented_cells = target_12.targets_per_cell > 0
    assert np.isclose(
        target_12.represented_volume_mm3,
        np.sum(task_region.cell_volume_mm3[represented_cells]),
    )

    # Solver stopping convergence and the 0.5 mm task-accuracy assessment are
    # separate classifications. Live `.reached` remains a compatibility alias.
    stationary = solver.solve(np.zeros(3), np.zeros(3), np.zeros(3))
    assert stationary.solver_converged and stationary.reached
    assert stationary.termination_reason == "initial-tolerance-met"
    assert stationary.iterations == 0
    assert solver.tolerance_mm == solver.solver_tolerance_mm == 0.15

    one_step_solver = ConstrainedTipIK(
        parameters,
        solver_tolerance_mm=0.001,
        max_iterations=1,
        damping_mm=1.0,
        max_normalised_step=0.1,
    )
    reachable_target = one_step_solver.forward_endpoint_mm(
        deployment[7],
        rotation[7],
        0,
    )
    one_step = one_step_solver.solve(
        reachable_target,
        np.zeros(3),
        np.zeros(3),
    )
    assert one_step.termination_reason == "max-iterations"
    assert not one_step.solver_converged

    constant_solver = ConstrainedTipIK(parameters, max_iterations=3)
    constant_solver.forward_endpoints_mm = lambda *_args: np.zeros((3, 3))
    stalled = constant_solver.solve(
        np.asarray([10.0, 0.0, 10.0]),
        np.zeros(3),
        np.zeros(3),
    )
    assert not stalled.solver_converged
    assert stalled.termination_reason == "no-improving-step"

    failed_solver = ConstrainedTipIK(parameters, max_iterations=3)
    with patch(
        "ctr_inverse_kinematics.np.linalg.solve",
        side_effect=np.linalg.LinAlgError,
    ):
        failed = failed_solver.solve(
            np.asarray([10.0, 0.0, 10.0]),
            np.zeros(3),
            np.zeros(3),
        )
    assert failed.termination_reason == "linear-solve-failed"
    assert failed.termination_reason in IK_TERMINATION_REASONS

    synthetic_errors = IKErrorMap(
        targets_mm=np.zeros((2, 3)),
        achieved_mm=np.zeros((2, 3)),
        residual_mm=np.asarray([0.2, 0.8]),
        reached=np.asarray([False, False]),
        iterations=np.asarray([2, 2]),
        evaluations=np.asarray([10, 10]),
        solve_time_ms=np.asarray([1.0, 1.0]),
        canonical_sample_ids=np.asarray(["target-a", "target-b"]),
        termination_reason=np.asarray(["max-iterations", "no-improving-step"]),
        solver_tolerance_mm=0.01,
        initial_error_mm=np.asarray([5.0, 5.0]),
        target_weight_mm3=np.asarray([1.0, 3.0]),
    )
    assert np.array_equal(
        within_evaluation_threshold(synthetic_errors, 0.5),
        [True, False],
    )
    summary = summarise_ik_errors(
        synthetic_errors,
        evaluation_threshold_mm=0.5,
    )
    assert summary["ik_solver_convergence_rate"] == 0.0
    assert summary["ik_within_evaluation_threshold_rate"] == 0.25
    assert summary["ik_evaluation_threshold_mm"] == 0.5
    assert summary["ik_weighted_target_volume_mm3"] == 4.0
    termination_total = sum(
        value
        for key, value in summary.items()
        if key.startswith("ik_termination_count_")
    )
    assert termination_total == 2.0

    print("Stage-2 analysis checks passed.")


if __name__ == "__main__":
    run_checks()
