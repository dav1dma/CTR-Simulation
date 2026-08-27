"""Executable checks for Cartesian controller and keyboard mappings."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interactive_ctr_tip_control import (
    FREE,
    HARD_LOCK,
    JOINT_MODE,
    SOFT_LOCK,
    TIP_MODE,
    VisPyCTRTipControlViewer,
    cartesian_velocity_mm_s,
    conditional_workspace_masks,
    controller_target_velocity_mm_s,
    front_plate_box_mesh,
    keyboard_target_step_mm,
    polyline_endpoint_poses,
    rear_tube_geometry,
    sample_local_conditional_workspace,
)
from ctr_inverse_kinematics import ConstrainedTipIK
from tube_parameters import build_supervisor_ctr_parameters


def run_checks() -> None:
    speed = 15.0

    assert np.allclose(cartesian_velocity_mm_s(1.0, 0.0, 0.0, speed), [15, 0, 0])
    assert np.allclose(cartesian_velocity_mm_s(0.0, -1.0, 0.0, speed), [0, 15, 0])
    assert np.allclose(cartesian_velocity_mm_s(0.0, 0.0, 1.0, speed), [0, 0, 15])
    assert np.allclose(cartesian_velocity_mm_s(0.0, 0.0, -1.0, speed), [0, 0, -15])

    diagonal = cartesian_velocity_mm_s(1.0, -1.0, 0.0, speed)
    assert diagonal[0] > 0.0 and diagonal[1] > 0.0
    assert np.isclose(np.linalg.norm(diagonal), speed)

    combined_xyz = cartesian_velocity_mm_s(1.0, -1.0, 1.0, speed)
    assert np.all(combined_xyz > 0.0)
    assert np.isclose(np.linalg.norm(combined_xyz), speed)

    assert keyboard_target_step_mm(set()) == 1.0
    assert keyboard_target_step_mm({"Shift"}) == 0.1
    assert keyboard_target_step_mm({"Control"}) == 5.0

    tangent = np.asarray([0.0, 0.0, 1.0])
    analogue = controller_target_velocity_mm_s(
        1.0, 0.0, 0, 0, 0.0, tangent, precision=False
    )
    assert np.allclose(analogue, [40.0, 0.0, 0.0])
    tangent_extension = controller_target_velocity_mm_s(
        0.0, 0.0, 0, 0, 1.0, tangent, precision=False
    )
    assert np.allclose(tangent_extension, [0.0, 0.0, 40.0])
    dpad_xy = controller_target_velocity_mm_s(
        0.0, 0.0, -1, 1, 0.0, tangent, precision=False
    )
    assert np.allclose(dpad_xy, [-10.0, 10.0, 0.0])
    square_dpad_xz = controller_target_velocity_mm_s(
        0.0, 0.0, -1, 1, 0.0, tangent, precision=True
    )
    assert np.allclose(square_dpad_xz, [-2.0, 0.0, 2.0])

    straight_backbone = np.column_stack(
        (np.zeros(5), np.zeros(5), np.linspace(0.0, 120.0, 5))
    )
    endpoints, tangents = polyline_endpoint_poses(
        straight_backbone,
        np.asarray([0.120, 0.070, 0.040]),
    )
    assert np.allclose(endpoints[:, 2], [120.0, 70.0, 40.0])
    assert np.allclose(tangents, [0.0, 0.0, 1.0])

    endpoint_samples = np.zeros((3, 4, 3), dtype=float)
    endpoint_samples[2, :, 0] = [0.0, 4.0, 8.0, 20.0]
    strict, relaxed, active = conditional_workspace_masks(
        endpoint_samples,
        selected_tube=0,
        lock_modes=[FREE, FREE, SOFT_LOCK],
        lock_targets_mm={2: np.zeros(3)},
    )
    assert active == 1
    assert np.array_equal(strict, [True, True, False, False])
    assert np.array_equal(relaxed, [False, False, True, False])

    hard, hard_relaxed, active = conditional_workspace_masks(
        endpoint_samples,
        selected_tube=0,
        lock_modes=[FREE, FREE, HARD_LOCK],
        lock_targets_mm={2: np.zeros(3)},
    )
    assert active == 1
    assert np.array_equal(hard, [True, False, False, False])
    assert not np.any(hard_relaxed)

    solver = ConstrainedTipIK(build_supervisor_ctr_parameters())
    deployment = np.asarray([0.120, 0.070, 0.040])
    rotation = np.zeros(3)
    actual_endpoints = solver.forward_endpoints_mm(deployment, rotation)
    local_samples = sample_local_conditional_workspace(
        solver,
        deployment,
        rotation,
        selected_tube=0,
        lock_modes=[FREE, FREE, HARD_LOCK],
        lock_targets_mm={2: actual_endpoints[2]},
        sample_count=24,
    )
    local_strict, _local_relaxed, active = conditional_workspace_masks(
        local_samples,
        selected_tube=0,
        lock_modes=[FREE, FREE, HARD_LOCK],
        lock_targets_mm={2: actual_endpoints[2]},
    )
    assert active == 1
    assert np.any(local_strict)

    tip_key = VisPyCTRTipControlViewer.control_key_text(
        SimpleNamespace(control_mode=TIP_MODE)
    )
    joint_key = VisPyCTRTipControlViewer.control_key_text(
        SimpleNamespace(control_mode=JOINT_MODE)
    )
    assert "CONTROLS — TIP MODE ONLY" in tip_key
    assert "Angle decrease / increase" not in tip_key
    assert "CONTROLS — JOINT MODE ONLY" in joint_key
    assert "Cycle FREE / SOFT / HARD" not in joint_key

    total_lengths = np.asarray([0.350, 0.170, 0.080])
    exposed = np.asarray([0.120, 0.070, 0.040])
    rear_geometry = rear_tube_geometry(total_lengths, exposed, np.zeros(3))
    expected_hidden_mm = (total_lengths - exposed) * 1000.0
    for tube, (centre, stripe) in enumerate(rear_geometry):
        assert centre.shape == stripe.shape == (2, 3)
        assert np.isclose(centre[0, 2], -expected_hidden_mm[tube])
        assert np.allclose(centre[-1], [0.0, 0.0, 0.0])

    fully_retracted = rear_tube_geometry(
        total_lengths,
        np.zeros(3),
        np.zeros(3),
    )
    for tube, (centre, _stripe) in enumerate(fully_retracted):
        assert np.isclose(centre[0, 2], -total_lengths[tube] * 1000.0)

    plate_vertices, plate_faces = front_plate_box_mesh()
    assert plate_vertices.shape == (8, 3)
    assert plate_faces.shape == (12, 3)
    assert np.min(plate_vertices[:, 2]) == 0.0
    assert np.max(plate_vertices[:, 2]) > 0.0

    print("Tip-control mapping checks passed.")


if __name__ == "__main__":
    run_checks()
