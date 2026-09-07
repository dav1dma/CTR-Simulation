"""Executable checks for the reproducible CTR design-analysis pipeline."""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctr_design_analysis import (  # noqa: E402
    DesignEvaluation,
    aggregate_cylindrical_dexterity,
    aggregate_dexterity_voxels,
    aggregate_ik_residual_voxels,
    baseline_design,
    calculate_dexterity_map,
    calculate_ik_error_map,
    calculate_multistart_ik_error_map,
    design_value,
    design_variables,
    evaluate_design,
    expand_first_quadrant_points,
    fold_configurations_to_first_quadrant,
    independent_baseline_targets,
    optimise_design,
    position_jacobian,
    positional_isotropy,
    rotate_about_z,
    sensitivity_study,
    shared_configuration_samples,
    with_design_value,
    workspace_size_metrics,
)
from ctr_inverse_kinematics import ConstrainedTipIK  # noqa: E402


def run_checks() -> None:
    design = baseline_design()
    parameters = design.to_parameters()
    assert np.allclose(design.total_length_mm, [350.0, 170.0, 80.0])
    assert np.allclose(parameters["kappa_0"], [0.0, 19.12, 14.04])

    deployment, rotation = shared_configuration_samples(
        design.total_length_mm * 1e-3, 32, seed=5
    )
    assert deployment.shape == rotation.shape == (32, 3)
    assert np.all(deployment[:, 2] <= deployment[:, 1])
    assert np.all(deployment[:, 1] <= deployment[:, 0])
    assert np.all(deployment <= design.total_length_mm * 1e-3 + 1e-12)

    sample_points = np.asarray([[2.0, -3.0, 4.0], [-5.0, -1.0, 6.0]])
    quarter_turned = rotate_about_z(sample_points, np.pi / 2.0)
    assert np.allclose(quarter_turned[:, :2], [[3.0, 2.0], [1.0, -5.0]])
    folded_points, folded_rotation, original_quadrants = (
        fold_configurations_to_first_quadrant(sample_points, rotation[:2])
    )
    assert np.all(folded_points[:, :2] >= -1e-5)
    assert folded_rotation.shape == (2, 3)
    assert original_quadrants.shape == (2,)
    expanded_points, expanded_quadrants = expand_first_quadrant_points(folded_points)
    assert expanded_points.shape == (8, 3)
    assert np.array_equal(np.bincount(expanded_quadrants), [2, 2, 2, 2])

    solver = ConstrainedTipIK(parameters)
    jacobian = position_jacobian(solver, deployment[12], rotation[12])
    score, singular_values = positional_isotropy(jacobian)
    assert jacobian.shape == (3, 6)
    assert singular_values.shape == (3,)
    assert 0.0 <= score <= 1.0
    quarter_turn = np.pi / 2.0
    rotated_state = rotation[12] - quarter_turn
    rotated_tip = solver.forward_endpoint_mm(deployment[12], rotated_state, 0)
    expected_tip = rotate_about_z(
        solver.forward_endpoint_mm(deployment[12], rotation[12], 0),
        quarter_turn,
    )
    rotated_score, _ = positional_isotropy(
        position_jacobian(solver, deployment[12], rotated_state)
    )
    assert np.allclose(rotated_tip, expected_tip, atol=1e-8)
    assert np.isclose(rotated_score, score, atol=1e-9)

    dexterity = calculate_dexterity_map(
        parameters, deployment[8:16], rotation[8:16]
    )
    assert dexterity.positions_mm.shape == (8, 3)
    assert dexterity.isotropy.shape == (8,)
    assert np.all((dexterity.isotropy >= 0.0) & (dexterity.isotropy <= 1.0))
    voxels = aggregate_dexterity_voxels(
        dexterity, voxel_size_mm=25.0, minimum_median_samples=2
    )
    assert len(voxels.centres_mm) <= len(dexterity.positions_mm)
    assert np.sum(voxels.sample_count) == len(dexterity.positions_mm)
    reported_median = np.isfinite(voxels.median_isotropy)
    assert np.all(
        voxels.maximum_isotropy[reported_median]
        >= voxels.median_isotropy[reported_median]
    )
    assert np.all(voxels.sample_count[reported_median] >= 2)
    cylindrical = aggregate_cylindrical_dexterity(
        dexterity,
        radial_bin_mm=25.0,
        z_bin_mm=25.0,
        minimum_median_samples=2,
    )
    assert np.sum(cylindrical.sample_count) == len(dexterity.positions_mm)
    assert cylindrical.maximum_isotropy.shape == cylindrical.sample_count.shape

    targets = independent_baseline_targets(8, seed=17)
    errors = calculate_ik_error_map(parameters, targets)
    assert errors.targets_mm.shape == errors.achieved_mm.shape == (8, 3)
    assert errors.residual_mm.shape == errors.reached.shape == (8,)
    assert np.all(errors.residual_mm >= 0.0)
    # Targets are still solved locally from full retraction rather than using
    # cached workspace restarts.
    assert np.any(errors.reached) and np.any(~errors.reached)
    azimuths = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    axisymmetric_targets = np.column_stack(
        (140.0 * np.cos(azimuths), 140.0 * np.sin(azimuths), np.full(8, 160.0))
    )
    fixed_axisymmetric_errors = calculate_ik_error_map(
        parameters, axisymmetric_targets
    )
    assert np.ptp(fixed_axisymmetric_errors.residual_mm) > 1.0
    multistart_axisymmetric_errors = calculate_multistart_ik_error_map(
        parameters, axisymmetric_targets
    )
    assert np.ptp(multistart_axisymmetric_errors.residual_mm) < 1e-5
    assert np.all(multistart_axisymmetric_errors.reached)
    residual_voxels = aggregate_ik_residual_voxels(
        errors, voxel_size_mm=25.0, minimum_median_samples=2
    )
    assert np.sum(residual_voxels.sample_count) == len(errors.targets_mm)
    assert np.all(residual_voxels.maximum_residual_mm >= 0.0)
    reported_residual_median = np.isfinite(residual_voxels.median_residual_mm)
    assert np.all(residual_voxels.sample_count[reported_residual_median] >= 2)

    synthetic_workspace = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 20.0],
            [-20.0, 0.0, 20.0],
            [0.0, 20.0, 40.0],
            [0.0, -20.0, 40.0],
            [10.0, 10.0, 60.0],
            [-10.0, -10.0, 60.0],
            [0.0, 0.0, 80.0],
        ]
    )
    volume, area = workspace_size_metrics(synthetic_workspace)
    assert volume > 0.0 and area > 0.0

    evaluation = evaluate_design(
        design,
        workspace_sample_count=32,
        dexterity_sample_count=8,
        ik_targets_mm=targets,
        seed=23,
    )
    assert evaluation.metrics["workspace_volume_mm3"] > 0.0
    assert 0.0 <= evaluation.metrics["dexterity_mean"] <= 1.0
    assert 0.0 <= evaluation.metrics["ik_success_rate"] <= 1.0

    variable = next(
        item
        for item in design_variables(design)
        if item.name == "middle_precurvature_per_m"
    )
    changed = with_design_value(design, variable.name, variable.baseline * 1.05)
    assert np.isclose(
        design_value(changed, variable.name), variable.baseline * 1.05
    )

    def lightweight_evaluator(candidate):
        value = design_value(candidate, variable.name)
        metrics = {"synthetic": (value - variable.baseline * 0.95) ** 2}
        return DesignEvaluation(candidate, metrics, np.zeros((1, 3)), None, None)

    sensitivity = sensitivity_study(
        design,
        [variable],
        lightweight_evaluator,
        relative_levels=(-0.05, 0.0, 0.05),
    )
    assert len(sensitivity) == 3
    result = optimise_design(
        design,
        [variable],
        lightweight_evaluator,
        lambda item: item.metrics["synthetic"],
        max_iterations=1,
        population_size=3,
        seed=31,
    )
    assert variable.lower <= result.values[0] <= variable.upper
    assert np.isfinite(result.objective)

    print("CTR design-analysis checks passed.")


if __name__ == "__main__":
    run_checks()
