"""Checks for the versioned CTR methodology and canonical sampling."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_protocol import (  # noqa: E402
    baseline_model_snapshot,
    baseline_parameter_hash,
    build_run_manifest,
    canonical_json_bytes,
    load_analysis_protocol,
    sha256_mapping,
    validate_analysis_protocol,
    write_run_manifest,
)
from ctr_design_analysis import (  # noqa: E402
    DexterityMap,
    aggregate_cylindrical_dexterity,
    baseline_design,
    canonicalise_tip_states_about_z,
    position_jacobian,
    positional_isotropy,
    revolve_canonical_points_for_display,
    rotate_about_z,
)
from ctr_inverse_kinematics import ConstrainedTipIK  # noqa: E402
from ctr_sampling import (  # noqa: E402
    boundary_deployment_fractions,
    canonical_configuration_bank,
    independent_configuration_banks,
)
from tools.plot_axisymmetric_ik_slice import canonical_radial_slice_grid  # noqa: E402


def run_checks() -> None:
    protocol = load_analysis_protocol()
    reloaded = load_analysis_protocol()
    assert protocol.sha256 == reloaded.sha256
    assert canonical_json_bytes(protocol.snapshot) == canonical_json_bytes(
        reloaded.snapshot
    )

    scope = protocol.section("scope")
    baseline = protocol.section("baseline")
    sampling = protocol.section("sampling")
    symmetry = protocol.section("symmetry")
    spatial = protocol.section("spatial")
    ik = protocol.section("ik")
    jacobian = protocol.section("jacobian")
    optimisation = protocol.section("optimisation")
    assert scope["primary_endpoint"] == "inner"
    assert scope["secondary_endpoints"] == ["middle", "outer"]
    assert baseline["youngs_modulus_gpa"] == [75.0, 75.0, 75.0]
    assert baseline["youngs_modulus_uncertainty_ranges_gpa"] == [
        [40.0, 75.0],
        [60.0, 83.0],
        [60.0, 83.0],
    ]
    assert baseline["precurvature_per_m"] == [0.0, 19.12, 14.04]
    assert "pending-supervisor-confirmation" in baseline["precurvature_status"]
    assert sampling["independent_dimensions"] == 5
    assert sampling["streams"] == {
        "deployment_fractions": 0,
        "relative_rotations": 1,
    }
    assert sampling["boundary_cases_are_statistical_samples"] is False
    assert symmetry["rotated_copies_are_independent"] is False
    assert symmetry["statistical_aggregation_before_sweep"] is True
    assert spatial["provisional_cell_size_mm"] == 10.0
    assert ik["evaluation_threshold_mm"] == 0.5
    assert ik["solver_tolerance_mm"] == 0.01
    assert ik["evaluation_threshold_mm"] == 0.5
    assert ik["max_iterations"] == 40
    assert jacobian["actuator_scale"] == [
        350.0,
        170.0,
        80.0,
        np.pi,
        np.pi,
        np.pi,
    ]
    assert jacobian["translation_step_mm"] == 0.1
    assert jacobian["rotation_step_rad"] == 0.001
    assert optimisation["enabled"] is False
    assert optimisation["youngs_modulus_is_design_variable"] is False
    assert optimisation["orientation_is_objective"] is False
    assert optimisation["force_or_loaded_stiffness_is_objective"] is False
    assert baseline_parameter_hash() == baseline["expected_model_parameter_sha256"]
    assert baseline_model_snapshot()["youngs_modulus_gpa"] == [75.0] * 3

    # Semantic hashing ignores dictionary insertion order and TOML formatting.
    reordered = dict(reversed(list(protocol.snapshot.items())))
    assert sha256_mapping(reordered) == protocol.sha256

    malformed = protocol.snapshot
    malformed["baseline"]["precurvature_per_m"][1] = 19.13
    try:
        validate_analysis_protocol(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("a baseline parameter mismatch must be rejected")
    malformed = protocol.snapshot
    malformed["sampling"]["validation_seed"] = malformed["sampling"]["training_seed"]
    try:
        validate_analysis_protocol(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping train/validation seeds must be rejected")
    malformed = protocol.snapshot
    malformed["schema_version"] = 99
    try:
        validate_analysis_protocol(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("an unsupported schema must be rejected")

    manifest = build_run_manifest(
        protocol,
        run_kind="stage-1-test",
        command=["test_analysis_protocol.py"],
        project_root=PROJECT_ROOT,
    )
    assert manifest["protocol"]["sha256"] == protocol.sha256
    assert manifest["baseline_parameters"]["sha256"] == baseline_parameter_hash()
    assert "git_commit" in manifest["code"]
    assert "ctr_sampling.py" in manifest["code"]["source_file_sha256"]
    assert "ctr_spatial_analysis.py" in manifest["code"]["source_file_sha256"]
    assert "ctr_task_space.py" in manifest["code"]["source_file_sha256"]
    assert "numpy" in manifest["runtime"]
    with TemporaryDirectory() as directory:
        target = Path(directory) / "manifest.json"
        write_run_manifest(target, manifest)
        stored = json.loads(target.read_text(encoding="utf-8"))
        assert stored["protocol"]["sha256"] == protocol.sha256

    # Canonical samples are exactly prefix-stable and do not contain injected
    # boundary corners.
    small = canonical_configuration_bank(32, seed=42, split="training")
    large = canonical_configuration_bank(128, seed=42, split="training")
    repeated = canonical_configuration_bank(32, seed=42, split="training")
    assert np.array_equal(small.sample_ids, large.sample_ids[:32])
    assert np.array_equal(small.sample_ids, repeated.sample_ids)
    assert np.array_equal(
        small.deployment_fractions,
        large.deployment_fractions[:32],
    )
    assert np.array_equal(
        small.relative_rotation_rad,
        large.relative_rotation_rad[:32],
    )
    assert not any(
        np.array_equal(row, corner)
        for row in small.deployment_fractions
        for corner in boundary_deployment_fractions()
    )

    training, validation = independent_configuration_banks(
        training_count=64,
        validation_count=64,
        training_seed=42,
        validation_seed=1042,
    )
    assert np.intersect1d(training.sample_ids, validation.sample_ids).size == 0
    train_states = {
        tuple(row)
        for row in np.column_stack(
            (training.deployment_fractions, training.relative_rotation_rad)
        )
    }
    validation_states = {
        tuple(row)
        for row in np.column_stack(
            (validation.deployment_fractions, validation.relative_rotation_rad)
        )
    }
    assert train_states.isdisjoint(validation_states)
    design = baseline_design()
    deployment = training.decode_deployments(design.total_length_mm * 1e-3)
    longer_lengths = design.total_length_mm * np.asarray([1.05, 1.03, 1.01]) * 1e-3
    longer_deployment = training.decode_deployments(longer_lengths)
    assert np.all(deployment[:, 2] <= deployment[:, 1])
    assert np.all(deployment[:, 1] <= deployment[:, 0])
    assert not np.array_equal(deployment, longer_deployment)
    assert np.all(training.rotation_rad[:, 2] == 0.0)
    assert np.allclose(
        training.rotation_rad[:, :2] - training.rotation_rad[:, 2, None],
        training.relative_rotation_rad,
    )

    # Display copies preserve source IDs and never increase independent N.
    canonical_points = np.column_stack(
        (
            np.linspace(10.0, 30.0, 17),
            np.zeros(17),
            np.linspace(50.0, 90.0, 17),
        )
    )
    identifiers = training.sample_ids[:17]
    expansion_12 = revolve_canonical_points_for_display(
        canonical_points,
        identifiers,
        np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False),
    )
    expansion_37 = revolve_canonical_points_for_display(
        canonical_points,
        identifiers,
        np.linspace(0.0, 2.0 * np.pi, 37, endpoint=False),
    )
    assert expansion_12.rendered_count == 17 * 12
    assert expansion_37.rendered_count == 17 * 37
    assert expansion_12.independent_count == expansion_37.independent_count == 17
    assert np.array_equal(expansion_12.source_sample_ids[:17], identifiers)

    radial_grid, radial_counts, _, _, populated_bins = canonical_radial_slice_grid(
        np.repeat([5.0, 15.0, 25.0], 10),
        np.linspace(0.1, 0.9, 30),
        voxel_size_mm=10.0,
        minimum_independent_samples=5,
    )
    assert populated_bins == 3
    assert np.nanmax(radial_counts) == 10
    assert np.any(np.isfinite(radial_grid))

    scores = np.linspace(0.1, 0.9, 17, dtype=float)
    for expansion, angle_count in ((expansion_12, 12), (expansion_37, 37)):
        repeated_scores = np.tile(scores, angle_count)
        rows = len(expansion.points_mm)
        expanded_map = DexterityMap(
            positions_mm=expansion.points_mm,
            isotropy=repeated_scores,
            singular_values=np.ones((rows, 3)),
            deployment_m=np.zeros((rows, 3)),
            rotation_rad=np.zeros((rows, 3)),
            canonical_sample_ids=expansion.source_sample_ids,
        )
        cylindrical = aggregate_cylindrical_dexterity(
            expanded_map,
            radial_bin_mm=50.0,
            z_bin_mm=100.0,
            minimum_median_samples=1,
        )
        assert np.sum(cylindrical.sample_count) == 17
        assert np.nanmax(cylindrical.sample_count) == 17

    # Validate continuous rotational equivariance at arbitrary angles and all
    # endpoints.  The canonical actuator state must reproduce (r, 0, z).
    parameters = design.to_parameters()
    solver = ConstrainedTipIK(parameters, model_points_per_section=2)
    test_deployment = deployment[11]
    test_rotation = training.rotation_rad[11]
    arbitrary_angles = (0.137, -0.691, 1.234, 2.718)
    for endpoint in (0, 1, 2):
        base_tip = solver.forward_endpoint_mm(
            test_deployment,
            test_rotation,
            endpoint,
        )
        base_jacobian = position_jacobian(
            solver,
            test_deployment,
            test_rotation,
            target_tube=endpoint,
        )
        base_isotropy, _ = positional_isotropy(base_jacobian)
        for angle in arbitrary_angles:
            rotated_state = (test_rotation - angle + np.pi) % (2.0 * np.pi) - np.pi
            actual_tip = solver.forward_endpoint_mm(
                test_deployment,
                rotated_state,
                endpoint,
            )
            actual_jacobian = position_jacobian(
                solver,
                test_deployment,
                rotated_state,
                target_tube=endpoint,
            )
            expected_tip = rotate_about_z(base_tip, angle)
            expected_jacobian = rotate_about_z(base_jacobian.T, angle).T
            rotated_isotropy, _ = positional_isotropy(actual_jacobian)
            assert np.allclose(actual_tip, expected_tip, atol=1e-8)
            assert np.allclose(actual_jacobian, expected_jacobian, atol=1e-5)
            assert np.isclose(rotated_isotropy, base_isotropy, atol=1e-9)

        canonical = canonicalise_tip_states_about_z(
            base_tip.reshape(1, 3),
            test_rotation.reshape(1, 3),
            training.sample_ids[11:12],
        )
        reproduced = solver.forward_endpoint_mm(
            test_deployment,
            canonical.rotation_rad[0],
            endpoint,
        )
        assert canonical.positions_mm[0, 0] >= 0.0
        assert abs(float(canonical.positions_mm[0, 1])) <= 1e-12
        assert np.allclose(reproduced, canonical.positions_mm[0], atol=1e-5)
        base_relative = (test_rotation[:2] - test_rotation[2] + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        canonical_relative = (
            canonical.rotation_rad[0, :2]
            - canonical.rotation_rad[0, 2]
            + np.pi
        ) % (2.0 * np.pi) - np.pi
        assert np.allclose(base_relative, canonical_relative, atol=1e-6)

    print("Versioned analysis protocol and canonical-sampling checks passed.")


if __name__ == "__main__":
    run_checks()
