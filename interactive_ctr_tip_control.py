"""Tip-position control for the VisPy concentric-tube robot simulator.

This is a separate application from ``interactive_ctr_vispy.py``. L3 or Tab
switches between joint and Cartesian target control. The target workflow can
preview either direct motion or a retract/reorient/advance route, while tube
lengths behind the Z=0 front-plate reference remain visible at negative Z.
"""

from __future__ import annotations

import time

import numpy as np
import pygame
from scipy.spatial import cKDTree
from vispy import app, scene

import interactive_ctr_vispy as joint_viewer
from ctr_inverse_kinematics import ConstrainedTipIK
from ctr_motion_planner import (
    DIRECT_STRATEGY,
    RETRACT_STRATEGY,
    MotionPlan,
    MotionRoute,
    TipTargetPlanner,
    build_motion_route,
    route_state_at_time,
    sample_motion_route,
)
from ctr_workspace_map import (
    constrain_point_to_workspace_surface,
    load_endpoint_workspace_maps,
    workspace_surface_of_revolution,
)


BUTTON_L3 = 7
LEFT_STICK_X, LEFT_STICK_Y = 0, 1

JOINT_MODE = "joint"
TIP_MODE = "tip"
AXIS_NAMES = ("X", "Y", "Z")

TIP_SPEED_MM_S = 40.0
TIP_PRECISION_SPEED_MM_S = 5.0
DPAD_TARGET_SPEED_MM_S = 10.0
DPAD_PRECISION_SPEED_MM_S = 2.0
TARGET_UPDATE_HZ = 30.0
OPTIONS_EXPORT_HOLD_S = 0.8
MAX_ALTERNATIVE_SOLUTIONS = 6

FREE = "FREE"
SOFT_LOCK = "SOFT LOCK"
HARD_LOCK = "HARD LOCK"
LOCK_MODES = (FREE, SOFT_LOCK, HARD_LOCK)
SOFT_LOCK_TOLERANCE_MM = 5.0
SOFT_LOCK_RELAXED_TOLERANCE_MM = 15.0
HARD_LOCK_TOLERANCE_MM = 1.5
SOFT_LOCK_WEIGHT = 0.45
HARD_LOCK_WEIGHT = 1.0
CONDITIONAL_DISPLAY_LIMIT = 2500
CONDITIONAL_PROJECTION_THRESHOLD_MM = 2.0

TARGET_SELECT = "SELECT TARGET"
TARGET_PREVIEW = "PREVIEW READY"
TARGET_EXECUTING = "EXECUTING"
TARGET_REACHED = "TARGET REACHED"
TARGET_NO_SOLUTION = "NO SOLUTION"

TARGET_MARKER_SIZE = 8.0
TARGET_SELECT_COLOUR = (0.95, 0.08, 0.06, 0.95)
TARGET_PREVIEW_COLOUR = (0.05, 0.72, 0.24, 1.0)
TARGET_EXECUTING_COLOUR = (0.58, 0.08, 0.78, 1.0)
TARGET_NO_SOLUTION_COLOUR = (0.42, 0.44, 0.48, 1.0)
TARGET_EDGE_COLOUR = (0.3, 0.03, 0.02, 1.0)
TARGET_LINE_COLOUR = (0.75, 0.08, 0.08, 0.65)
LIMIT_LINE_COLOUR = (0.9, 0.45, 0.05, 0.8)
PLANNED_PATH_COLOUR = (0.58, 0.08, 0.78, 0.9)
DIRECT_PATH_COLOUR = (0.0, 0.55, 0.82, 0.9)
SMOOTH_WORKSPACE_COLOUR = (0.04, 0.42, 0.95, 0.10)
CONDITIONAL_FEASIBLE_COLOUR = (0.05, 0.72, 0.24, 0.30)
CONDITIONAL_RELAXED_COLOUR = (0.95, 0.58, 0.05, 0.24)
FRONT_PLATE_SIZE_MM = 190.0
FRONT_PLATE_THICKNESS_MM = 4.0
FRONT_PLATE_COLOUR = (0.24, 0.28, 0.34, 0.26)


def rear_tube_geometry(
    total_lengths_m: np.ndarray,
    exposed_lengths_m: np.ndarray,
    rotations_rad: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return centre and rotation-stripe lines behind the Z=0 plate plane."""
    total = np.asarray(total_lengths_m, dtype=float).reshape(3)
    exposed = np.asarray(exposed_lengths_m, dtype=float).reshape(3)
    rotations = np.asarray(rotations_rad, dtype=float).reshape(3)
    hidden_mm = np.maximum(total - exposed, 0.0) * 1000.0
    geometry = []
    for tube in range(3):
        centre = np.asarray(
            [[0.0, 0.0, -hidden_mm[tube]], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        offset = joint_viewer.TUBE_STRIPE_OFFSETS_MM[tube]
        stripe_x = offset * np.cos(rotations[tube])
        stripe_y = offset * np.sin(rotations[tube])
        stripe = np.asarray(
            [
                [stripe_x, stripe_y, -hidden_mm[tube]],
                [stripe_x, stripe_y, 0.0],
            ],
            dtype=np.float32,
        )
        geometry.append((centre, stripe))
    return tuple(geometry)


def front_plate_box_mesh(
    size_mm: float = FRONT_PLATE_SIZE_MM,
    thickness_mm: float = FRONT_PLATE_THICKNESS_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a thin square plate whose back surface is the Z=0 reference."""
    half = float(size_mm) * 0.5
    depth = float(thickness_mm)
    vertices = np.asarray(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
            [-half, -half, depth],
            [half, -half, depth],
            [half, half, depth],
            [-half, half, depth],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def cartesian_velocity_mm_s(
    left_x: float,
    left_y: float,
    z_command: float,
    speed_mm_s: float,
) -> np.ndarray:
    """Convert controller input to a bounded global XYZ tip velocity.

    Pygame reports stick-up as negative Y, so it is inverted here.  The full
    three-dimensional direction is normalised when necessary, which prevents
    diagonal movement from being faster than single-axis movement.
    """
    direction = np.asarray(
        [left_x, -left_y, np.clip(z_command, -1.0, 1.0)], dtype=float
    )
    magnitude = float(np.linalg.norm(direction))
    if magnitude > 1.0:
        direction /= magnitude
    return direction * float(speed_mm_s)


def polyline_endpoint_poses(
    backbone_mm: np.ndarray,
    deployment_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each tube endpoint and local tangent along a backbone."""
    points = np.asarray(backbone_mm, dtype=float)
    deployment = np.asarray(deployment_m, dtype=float).reshape(3)
    if len(points) < 2 or np.max(np.abs(deployment)) <= 1e-12:
        return np.zeros((3, 3), dtype=float), np.tile((0.0, 0.0, 1.0), (3, 1))
    segment = np.diff(points, axis=0)
    segment_length = np.linalg.norm(segment, axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment_length)))
    endpoints = np.empty((3, 3), dtype=float)
    tangents = np.empty((3, 3), dtype=float)
    for tube, requested_arc in enumerate(deployment * 1000.0):
        distance = float(np.clip(requested_arc, 0.0, arc[-1]))
        endpoints[tube] = [
            np.interp(distance, arc, points[:, axis]) for axis in range(3)
        ]
        segment_index = min(
            int(np.searchsorted(arc, distance, side="right") - 1),
            len(segment) - 1,
        )
        segment_index = max(segment_index, 0)
        while segment_index > 0 and segment_length[segment_index] <= 1e-9:
            segment_index -= 1
        tangent = segment[segment_index]
        tangent_norm = float(np.linalg.norm(tangent))
        tangents[tube] = (
            tangent / tangent_norm
            if tangent_norm > 1e-9
            else np.asarray([0.0, 0.0, 1.0])
        )
    return endpoints, tangents


def controller_target_velocity_mm_s(
    left_x: float,
    left_y: float,
    dpad_x: int,
    dpad_y: int,
    trigger_command: float,
    tangent: np.ndarray,
    *,
    precision: bool,
) -> np.ndarray:
    """Combine analogue XY, fine D-pad XYZ and tangent extension control."""
    analogue_speed = TIP_PRECISION_SPEED_MM_S if precision else TIP_SPEED_MM_S
    dpad_speed = DPAD_PRECISION_SPEED_MM_S if precision else DPAD_TARGET_SPEED_MM_S
    velocity = cartesian_velocity_mm_s(left_x, left_y, 0.0, analogue_speed)
    if precision:
        velocity += np.asarray([dpad_x, 0.0, dpad_y], dtype=float) * dpad_speed
    else:
        velocity += np.asarray([dpad_x, dpad_y, 0.0], dtype=float) * dpad_speed
    tangent_vector = np.asarray(tangent, dtype=float).reshape(3)
    tangent_norm = float(np.linalg.norm(tangent_vector))
    if tangent_norm > 1e-9:
        velocity += (
            np.clip(trigger_command, -1.0, 1.0)
            * analogue_speed
            * tangent_vector
            / tangent_norm
        )
    return velocity


def conditional_workspace_masks(
    endpoint_tips_mm: np.ndarray,
    selected_tube: int,
    lock_modes: list[str] | tuple[str, ...],
    lock_targets_mm: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Classify cached states as strict-feasible or soft-lock relaxed."""
    endpoints = np.asarray(endpoint_tips_mm, dtype=float)
    if endpoints.ndim != 3 or endpoints.shape[0] != 3 or endpoints.shape[2] != 3:
        raise ValueError("endpoint_tips_mm must have shape (3, samples, 3)")
    if int(selected_tube) not in (0, 1, 2):
        raise ValueError("selected_tube must be 0, 1, or 2")
    if len(lock_modes) != 3 or any(mode not in LOCK_MODES for mode in lock_modes):
        raise ValueError("lock_modes must contain three valid modes")
    strict = np.ones(endpoints.shape[1], dtype=bool)
    relaxed = np.ones(endpoints.shape[1], dtype=bool)
    active_constraints = 0
    for tube, target in lock_targets_mm.items():
        tube_index = int(tube)
        if tube_index not in (0, 1, 2):
            raise ValueError("lock target tube must be 0, 1, or 2")
        if tube_index == int(selected_tube) or lock_modes[tube_index] == FREE:
            continue
        target_vector = np.asarray(target, dtype=float).reshape(3)
        errors = np.linalg.norm(endpoints[tube_index] - target_vector, axis=1)
        active_constraints += 1
        if lock_modes[tube_index] == SOFT_LOCK:
            strict &= errors <= SOFT_LOCK_TOLERANCE_MM
            relaxed &= errors <= SOFT_LOCK_RELAXED_TOLERANCE_MM
        else:
            within = errors <= HARD_LOCK_TOLERANCE_MM
            strict &= within
            relaxed &= within
    return strict, relaxed & ~strict, active_constraints


def sample_local_conditional_workspace(
    solver: ConstrainedTipIK,
    deployment_m: np.ndarray,
    rotation_rad: np.ndarray,
    selected_tube: int,
    lock_modes: list[str] | tuple[str, ...],
    lock_targets_mm: dict[int, np.ndarray],
    *,
    sample_count: int = 180,
    seed: int = 73,
) -> np.ndarray:
    """Sample the local actuator null space of the active endpoint locks."""
    active_tubes = [
        tube
        for tube in sorted(lock_targets_mm)
        if tube != int(selected_tube) and lock_modes[tube] != FREE
    ]
    if not active_tubes or sample_count < 1:
        return np.empty((3, 0, 3), dtype=np.float32)

    deployment = solver.validate_deployment(deployment_m)
    rotation = np.asarray(rotation_rad, dtype=float).reshape(3)
    state = np.concatenate((solver.encode_deployment(deployment), rotation / np.pi))
    current = solver.forward_endpoints_mm(deployment, rotation)
    base_task = current[active_tubes].reshape(-1)
    jacobian = np.empty((len(base_task), 6), dtype=float)
    finite_step = 1e-4
    for column in range(6):
        step = finite_step
        if column < 3 and state[column] + step > 1.0:
            step = -step
        perturbed = state.copy()
        perturbed[column] += step
        perturbed[:3] = np.clip(perturbed[:3], 0.0, 1.0)
        trial_deployment = solver.decode_deployment(perturbed[:3])
        trial_rotation = np.pi * perturbed[3:]
        trial = solver.forward_endpoints_mm(trial_deployment, trial_rotation)
        jacobian[:, column] = (trial[active_tubes].reshape(-1) - base_task) / step

    _u, singular_values, vh = np.linalg.svd(jacobian, full_matrices=True)
    threshold = (
        max(jacobian.shape) * np.finfo(float).eps * singular_values[0]
        if len(singular_values) and singular_values[0] > 0.0
        else 1e-10
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    null_basis = vh[rank:].T
    samples = [current]
    if null_basis.shape[1] == 0:
        return np.asarray(samples, dtype=np.float32).transpose(1, 0, 2)

    has_hard_lock = any(lock_modes[tube] == HARD_LOCK for tube in active_tubes)
    maximum_radius = 0.05 if has_hard_lock else 0.10
    rng = np.random.default_rng(seed + 17 * int(selected_tube))
    for _index in range(sample_count - 1):
        direction = rng.normal(size=null_basis.shape[1])
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        direction /= norm
        radius = maximum_radius * float(rng.random() ** (1.0 / null_basis.shape[1]))
        candidate = state + null_basis @ (direction * radius)
        candidate[:3] = np.clip(candidate[:3], 0.0, 1.0)
        candidate_deployment = solver.decode_deployment(candidate[:3])
        candidate_rotation = np.pi * candidate[3:]
        samples.append(
            solver.forward_endpoints_mm(candidate_deployment, candidate_rotation)
        )
    return np.asarray(samples, dtype=np.float32).transpose(1, 0, 2)


def keyboard_target_step_mm(modifier_names: set[str]) -> float:
    """Return the Cartesian keyboard step selected by modifier keys."""
    names = {name.lower() for name in modifier_names}
    if "control" in names or "ctrl" in names:
        return 5.0
    if "shift" in names:
        return 0.1
    return 1.0


class VisPyCTRTipControlViewer(joint_viewer.VisPyCTRViewer):
    """Joint-space viewer extended with constrained Cartesian tip control."""

    def __init__(self, *, mode="hardware", tubes="original", workspace_samples=12000) -> None:
        # These are initialised before the base class because it builds the
        # sidebar using the overridden sidebar_text method.
        self.control_mode = JOINT_MODE
        self.tip_stage = TARGET_SELECT
        self.keyboard_target_axis = 0
        self.target_tip_mm = np.zeros(3, dtype=float)
        self.tracking_error_mm = 0.0
        self.last_solve_error_mm = 0.0
        self.last_ik_ms = 0.0
        self.ik_status = "JOINT CONTROL"
        self.left_x = 0.0
        self.left_y = 0.0
        self.target_command_accumulator = np.zeros(3, dtype=float)
        self.sync_target_pending = False
        self.target_marker = None
        self.target_error_line = None
        self.endpoint_markers = None
        self.locked_target_markers = None
        self.projected_target_marker = None
        self.conditional_feasible_visual = None
        self.conditional_relaxed_visual = None
        self.planned_path_visual = None
        self.preview_tube_lines = []
        self.alternative_preview_lines = []
        self.rear_tube_lines = []
        self.rear_rotation_stripes = []
        self.front_plate_visual = None
        self.motion_plan: MotionPlan | None = None
        self.motion_plans: list[MotionPlan] = []
        self.motion_route: MotionRoute | None = None
        self.solution_index = 0
        self.motion_strategy = DIRECT_STRATEGY
        self.execution_started = 0.0
        self.execution_duration = 0.0
        self.execution_progress = 0.0
        self.execution_phase_index = 0
        self.execution_phase_progress = 0.0
        self.previous_strategy_dpad_y = 0
        self.previous_solution_dpad_x = 0
        self.plan_solution_source = "NONE"
        self.target_sample_distance_mm = 0.0
        self.workspace_surface_visual = None
        self.workspace_surfaces = []
        self.workspace_sample_count = 0
        self.workspace_limit_hit = False
        self.endpoint_positions_mm = np.zeros((3, 3), dtype=float)
        self.endpoint_tangents = np.tile((0.0, 0.0, 1.0), (3, 1))
        self.current_backbone_mm = np.zeros((2, 3), dtype=np.float32)
        self.locked_endpoint_targets: dict[int, np.ndarray] = {}
        self.endpoint_lock_modes = [FREE, FREE, FREE]
        self.conditional_feasible_count = 0
        self.conditional_relaxed_count = 0
        self.conditional_active_lock_count = 0
        self.conditional_tree: cKDTree | None = None
        self.conditional_projection_is_relaxed = False
        self.conditional_projection_mm: np.ndarray | None = None
        self.conditional_projection_distance_mm = 0.0
        self.options_hold_start: float | None = None
        self.options_hold_fired = False

        super().__init__(mode=mode, tubes=tubes, workspace_samples=workspace_samples)

        self.canvas.title = "PS5 CTR Simulator — Tip Position Control"
        self.controller.previous[BUTTON_L3] = False
        self.ik_solver = ConstrainedTipIK(self.parameters, deployment_limits=self.limits)
        self.planning_solver = ConstrainedTipIK(
            self.parameters,
            deployment_limits=self.limits,
            max_iterations=12,
            damping_mm=1.0,
            max_normalised_step=0.10,
        )
        self.joint_workspace_visual.visible = False
        self.workspace_sample_count = self.endpoint_workspace_maps.tips_mm.shape[1]
        self.target_planners = [
            TipTargetPlanner(
                self.ik_solver,
                self.planning_solver,
                self.endpoint_workspace_maps.for_tube(tube),
                target_tube=tube,
            )
            for tube in range(3)
        ]
        self.workspace_surfaces = [
            workspace_surface_of_revolution(
                self.endpoint_workspace_maps.tips_mm[tube]
            )
            for tube in range(3)
        ]
        self.target_planner = self.target_planners[self.selected_tube]
        self.target_tip_mm[:] = self.endpoint_positions_mm[self.selected_tube]
        self.target_sample_distance_mm = (
            self.target_planner.nearest_sample_distance_mm(self.target_tip_mm)
        )
        self.last_target_update = time.perf_counter()

        self.target_error_line = scene.visuals.Line(
            pos=np.asarray([self.tip_mm, self.target_tip_mm], dtype=np.float32),
            color=TARGET_LINE_COLOUR,
            width=1.5,
            connect="strip",
            method="gl",
            antialias=True,
            parent=self.view.scene,
        )
        self.target_marker = scene.visuals.Markers(parent=self.view.scene)
        self.target_marker.set_data(
            np.asarray([self.target_tip_mm], dtype=np.float32),
            symbol="disc",
            face_color=TARGET_SELECT_COLOUR,
            edge_color=TARGET_EDGE_COLOUR,
            edge_width=1.5,
            size=TARGET_MARKER_SIZE,
        )
        self.endpoint_markers = scene.visuals.Markers(parent=self.view.scene)
        self.locked_target_markers = scene.visuals.Markers(parent=self.view.scene)
        self.projected_target_marker = scene.visuals.Markers(parent=self.view.scene)
        self.projected_target_marker.visible = False
        self.conditional_feasible_visual = scene.visuals.Markers(
            parent=self.view.scene
        )
        self.conditional_feasible_visual.visible = False
        self.conditional_relaxed_visual = scene.visuals.Markers(
            parent=self.view.scene
        )
        self.conditional_relaxed_visual.visible = False
        self.planned_path_visual = scene.visuals.Line(
            pos=np.asarray([self.tip_mm, self.target_tip_mm], dtype=np.float32),
            color=PLANNED_PATH_COLOUR,
            width=2.5,
            connect="strip",
            method="gl",
            antialias=True,
            parent=self.view.scene,
        )
        self.planned_path_visual.visible = False
        for tube in (2, 1, 0):
            colour = (*joint_viewer.TUBE_COLOURS[tube][:3], 0.28)
            line = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=colour,
                width=joint_viewer.TUBE_LINE_WIDTHS[tube] + 1.5,
                connect="strip",
                method="gl",
                antialias=True,
                parent=self.view.scene,
            )
            line.visible = False
            self.preview_tube_lines.append((tube, line))
        for solution_slot in range(MAX_ALTERNATIVE_SOLUTIONS - 1):
            for tube in (2, 1, 0):
                colour = (*joint_viewer.TUBE_COLOURS[tube][:3], 0.10)
                line = scene.visuals.Line(
                    pos=np.zeros((2, 3), dtype=np.float32),
                    color=colour,
                    width=max(1.0, joint_viewer.TUBE_LINE_WIDTHS[tube] - 1.0),
                    connect="strip",
                    method="gl",
                    antialias=True,
                    parent=self.view.scene,
                )
                line.order = -1
                line.visible = False
                self.alternative_preview_lines.append(
                    (solution_slot, tube, line)
                )

        plate_vertices, plate_faces = front_plate_box_mesh()
        plate_vertices[:, 2] -= FRONT_PLATE_THICKNESS_MM
        self.front_plate_visual = scene.visuals.Mesh(
            vertices=plate_vertices,
            faces=plate_faces,
            color=FRONT_PLATE_COLOUR,
            shading=None,
            parent=self.view.scene,
        )
        self.front_plate_visual.set_gl_state(
            "translucent",
            depth_test=True,
            depth_mask=False,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        for tube in (2, 1, 0):
            line = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=joint_viewer.TUBE_COLOURS[tube],
                width=joint_viewer.TUBE_LINE_WIDTHS[tube],
                connect="strip",
                method="gl",
                antialias=True,
                parent=self.view.scene,
            )
            self.rear_tube_lines.append((tube, line))
            stripe = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=joint_viewer.TUBE_STRIPE_COLOUR,
                width=joint_viewer.TUBE_STRIPE_WIDTH,
                connect="strip",
                method="gl",
                antialias=True,
                parent=self.view.scene,
            )
            self.rear_rotation_stripes.append((tube, stripe))

        selected_surface = self.workspace_surfaces[self.selected_tube]
        self.workspace_surface_visual = scene.visuals.Mesh(
            vertices=selected_surface.vertices_mm,
            faces=selected_surface.faces,
            color=SMOOTH_WORKSPACE_COLOUR,
            shading="smooth",
            parent=self.view.scene,
        )
        self.workspace_surface_visual.set_gl_state(
            "translucent",
            depth_test=True,
            depth_mask=False,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            cull_face="back",
        )
        self.workspace_surface_visual.order = -12

        # Frame the complete sampled workspace on first launch. Mouse drag and
        # wheel control remain available for closer inspection.
        self.view.camera.center = (0.0, 0.0, 55.0 if mode == "hardware" else 80.0)
        self.view.camera.distance = 400.0 if mode == "hardware" else 900.0
        self.sidebar.label.font_size = 6.5
        self.update_rear_tube_visuals()
        self.update_endpoint_visuals()
        self.update_conditional_workspace()
        self.update_target_visual()
        self.update_status()

    def calculate_backbone(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keep a defined origin tip when all tubes are fully retracted."""
        if np.max(np.abs(self.deployment)) <= 1e-10:
            collapsed = np.zeros((2, 3), dtype=np.float32)
            result = (
                collapsed,
                np.zeros(3, dtype=float),
                np.asarray([0.0, 0.0, 1.0]),
            )
        else:
            result = super().calculate_backbone()
        self.current_backbone_mm = result[0]
        self.endpoint_positions_mm, self.endpoint_tangents = polyline_endpoint_poses(
            result[0],
            self.deployment,
        )
        return result

    @property
    def selected_endpoint_mm(self) -> np.ndarray:
        return self.endpoint_positions_mm[self.selected_tube]

    @property
    def selected_endpoint_tangent(self) -> np.ndarray:
        return self.endpoint_tangents[self.selected_tube]

    def update_endpoint_visuals(self) -> None:
        if self.endpoint_markers is None or self.locked_target_markers is None:
            return
        colours = np.asarray(joint_viewer.TUBE_COLOURS, dtype=np.float32)
        sizes = np.full(3, 7.0, dtype=np.float32)
        sizes[self.selected_tube] = 11.0
        self.endpoint_markers.set_data(
            np.asarray(self.endpoint_positions_mm, dtype=np.float32),
            symbol="disc",
            face_color=colours,
            edge_color=(0.05, 0.05, 0.05, 1.0),
            edge_width=1.0,
            size=sizes,
        )
        self.endpoint_markers.visible = self.guides_visible
        if self.locked_endpoint_targets:
            locked_tubes = sorted(self.locked_endpoint_targets)
            locked_points = np.asarray(
                [self.locked_endpoint_targets[tube] for tube in locked_tubes],
                dtype=np.float32,
            )
            locked_colours = np.asarray(
                [
                    CONDITIONAL_RELAXED_COLOUR
                    if self.endpoint_lock_modes[tube] == SOFT_LOCK
                    else (0.58, 0.08, 0.78, 1.0)
                    for tube in locked_tubes
                ],
                dtype=np.float32,
            )
            self.locked_target_markers.set_data(
                locked_points,
                symbol="cross",
                face_color=locked_colours,
                edge_color=locked_colours,
                size=10.0,
            )
            self.locked_target_markers.visible = self.guides_visible
        else:
            self.locked_target_markers.visible = False

    def active_lock_parameters(
        self,
    ) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float]]:
        """Return constraints on endpoints other than the actively moved one."""
        targets: dict[int, np.ndarray] = {}
        weights: dict[int, float] = {}
        tolerances: dict[int, float] = {}
        for tube, target in self.locked_endpoint_targets.items():
            if tube == self.selected_tube:
                continue
            mode = self.endpoint_lock_modes[tube]
            if mode == FREE:
                continue
            targets[tube] = target
            if mode == SOFT_LOCK:
                weights[tube] = SOFT_LOCK_WEIGHT
                tolerances[tube] = SOFT_LOCK_TOLERANCE_MM
            else:
                weights[tube] = HARD_LOCK_WEIGHT
                tolerances[tube] = HARD_LOCK_TOLERANCE_MM
        return targets, weights, tolerances

    def cycle_selected_lock_mode(self) -> None:
        if self.tip_stage in {TARGET_PREVIEW, TARGET_EXECUTING}:
            self.last_action = "Cancel or finish the current plan before changing lock"
            self.update_status()
            return
        tube = self.selected_tube
        current = self.endpoint_lock_modes[tube]
        following = LOCK_MODES[(LOCK_MODES.index(current) + 1) % len(LOCK_MODES)]
        self.endpoint_lock_modes[tube] = following
        if following == FREE:
            self.locked_endpoint_targets.pop(tube, None)
        elif current == FREE or tube not in self.locked_endpoint_targets:
            self.locked_endpoint_targets[tube] = self.selected_endpoint_mm.copy()
        self.clear_motion_plan()
        self.tip_stage = TARGET_SELECT
        self.last_action = (
            f"{joint_viewer.TUBE_NAMES[tube]} endpoint set to {following}"
        )
        self.update_endpoint_visuals()
        self.update_conditional_workspace()
        self.update_target_visual()
        self.update_status()

    def update_options_action(self, now: float) -> None:
        """Use a tap for lock mode and a hold for export in tip mode."""
        if self.controller.down(joint_viewer.BUTTON_OPTIONS):
            if self.options_hold_start is None:
                self.options_hold_start = float(now)
                self.options_hold_fired = False
            elif (
                not self.options_hold_fired
                and now - self.options_hold_start >= OPTIONS_EXPORT_HOLD_S
            ):
                self.export_snapshot()
                self.options_hold_fired = True
            return
        if self.options_hold_start is not None and not self.options_hold_fired:
            if self.control_mode == TIP_MODE:
                self.cycle_selected_lock_mode()
            else:
                self.export_snapshot()
        self.options_hold_start = None
        self.options_hold_fired = False

    @staticmethod
    def _display_subset(points: np.ndarray) -> np.ndarray:
        if len(points) <= CONDITIONAL_DISPLAY_LIMIT:
            return np.asarray(points, dtype=np.float32)
        indices = np.linspace(
            0,
            len(points) - 1,
            CONDITIONAL_DISPLAY_LIMIT,
            dtype=int,
        )
        return np.asarray(points[indices], dtype=np.float32)

    def update_conditional_workspace(self) -> None:
        """Show sampled positions compatible with all non-selected locks."""
        if self.conditional_feasible_visual is None:
            return
        strict, relaxed_only_mask, active_constraints = conditional_workspace_masks(
            self.endpoint_workspace_maps.tips_mm,
            self.selected_tube,
            self.endpoint_lock_modes,
            self.locked_endpoint_targets,
        )

        selected_samples = self.endpoint_workspace_maps.tips_mm[self.selected_tube]
        feasible_points = selected_samples[strict] if active_constraints else selected_samples
        relaxed_only = (
            selected_samples[relaxed_only_mask]
            if active_constraints
            else np.empty((0, 3), dtype=np.float32)
        )
        if active_constraints:
            local_endpoints = sample_local_conditional_workspace(
                self.ik_solver,
                self.deployment,
                self.rotation,
                self.selected_tube,
                self.endpoint_lock_modes,
                self.locked_endpoint_targets,
            )
            if local_endpoints.shape[1]:
                local_strict, local_relaxed, _active = conditional_workspace_masks(
                    local_endpoints,
                    self.selected_tube,
                    self.endpoint_lock_modes,
                    self.locked_endpoint_targets,
                )
                feasible_points = np.vstack(
                    (feasible_points, local_endpoints[self.selected_tube, local_strict])
                )
                relaxed_only = np.vstack(
                    (relaxed_only, local_endpoints[self.selected_tube, local_relaxed])
                )
        self.conditional_feasible_count = len(feasible_points)
        self.conditional_relaxed_count = len(relaxed_only)
        self.conditional_active_lock_count = active_constraints

        if active_constraints and len(feasible_points):
            displayed = self._display_subset(feasible_points)
            self.conditional_feasible_visual.set_data(
                displayed,
                symbol="disc",
                face_color=CONDITIONAL_FEASIBLE_COLOUR,
                edge_width=0.0,
                size=3.0,
            )
            self.conditional_feasible_visual.visible = self.guides_visible
            self.conditional_tree = cKDTree(np.asarray(feasible_points, dtype=float))
            self.conditional_projection_is_relaxed = False
        else:
            self.conditional_feasible_visual.visible = False
            self.conditional_tree = None
            self.conditional_projection_is_relaxed = False

        if active_constraints and len(relaxed_only):
            displayed = self._display_subset(relaxed_only)
            self.conditional_relaxed_visual.set_data(
                displayed,
                symbol="disc",
                face_color=CONDITIONAL_RELAXED_COLOUR,
                edge_width=0.0,
                size=3.0,
            )
            self.conditional_relaxed_visual.visible = self.guides_visible
            if self.conditional_tree is None:
                self.conditional_tree = cKDTree(
                    np.asarray(relaxed_only, dtype=float)
                )
                self.conditional_projection_is_relaxed = True
        else:
            self.conditional_relaxed_visual.visible = False
        self.update_target_projection()
        self.canvas.update()

    def update_target_projection(self) -> None:
        if self.projected_target_marker is None:
            return
        if self.conditional_tree is None:
            self.conditional_projection_mm = None
            self.conditional_projection_distance_mm = 0.0
            self.projected_target_marker.visible = False
            return
        distance, index = self.conditional_tree.query(self.target_tip_mm, k=1)
        projection = np.asarray(self.conditional_tree.data[int(index)], dtype=float)
        self.conditional_projection_mm = projection
        self.conditional_projection_distance_mm = float(distance)
        visible = (
            self.guides_visible
            and distance > CONDITIONAL_PROJECTION_THRESHOLD_MM
        )
        colour = (
            CONDITIONAL_RELAXED_COLOUR
            if self.conditional_projection_is_relaxed
            else CONDITIONAL_FEASIBLE_COLOUR
        )
        edge_colour = (
            (0.45, 0.22, 0.02, 1.0)
            if self.conditional_projection_is_relaxed
            else (0.04, 0.25, 0.08, 1.0)
        )
        self.projected_target_marker.set_data(
            np.asarray([projection], dtype=np.float32),
            symbol="diamond",
            face_color=colour,
            edge_color=edge_colour,
            edge_width=1.0,
            size=9.0,
        )
        self.projected_target_marker.visible = visible

    def select_tip_endpoint(self, index: int) -> None:
        """Cycle the controlled endpoint and its matching workspace/IK map."""
        if self.tip_stage == TARGET_EXECUTING:
            self.last_action = "Stop the current motion before changing endpoint"
            return
        self.clear_motion_plan()
        self.selected_tube = int(index) % len(joint_viewer.TUBE_NAMES)
        self.target_planner = self.target_planners[self.selected_tube]
        surface = self.workspace_surfaces[self.selected_tube]
        if self.workspace_surface_visual is not None:
            self.workspace_surface_visual.set_data(
                vertices=surface.vertices_mm,
                faces=surface.faces,
                color=SMOOTH_WORKSPACE_COLOUR,
            )
        self.tip_stage = TARGET_SELECT
        self.sync_target_to_tip("AWAITING TARGET")
        self.last_action = (
            f"{joint_viewer.TUBE_NAMES[self.selected_tube]} endpoint selected"
        )
        self.update_endpoint_visuals()
        self.update_conditional_workspace()
        self.update_status()

    def update_rear_tube_visuals(self) -> None:
        """Draw the portions parked behind the Z=0 front-plate reference."""
        geometry = rear_tube_geometry(
            self.total_lengths,
            self.deployment,
            self.rotation,
        )
        for tube, line in self.rear_tube_lines:
            line.set_data(pos=geometry[tube][0])
        for tube, stripe in self.rear_rotation_stripes:
            stripe.set_data(pos=geometry[tube][1])

    def read_left_stick(self) -> tuple[float, float]:
        if not self.controller.connected:
            return 0.0, 0.0
        joystick = self.controller.joystick
        if joystick.get_numaxes() <= LEFT_STICK_Y:
            return 0.0, 0.0
        return joint_viewer.apply_stick_deadzone(
            joystick.get_axis(LEFT_STICK_X),
            joystick.get_axis(LEFT_STICK_Y),
        )

    def sync_target_to_tip(self, status: str | None = None) -> None:
        self.target_tip_mm[:] = self.selected_endpoint_mm
        self.tracking_error_mm = 0.0
        self.last_solve_error_mm = 0.0
        self.workspace_limit_hit = False
        planner = getattr(self, "target_planner", None)
        if planner is not None:
            self.target_sample_distance_mm = planner.nearest_sample_distance_mm(
                self.target_tip_mm
            )
        if status is not None:
            self.ik_status = status
        self.update_target_visual()

    def target_marker_colour(self) -> tuple[float, float, float, float]:
        if self.tip_stage == TARGET_PREVIEW or self.tip_stage == TARGET_REACHED:
            return TARGET_PREVIEW_COLOUR
        if self.tip_stage == TARGET_EXECUTING:
            return TARGET_EXECUTING_COLOUR
        if self.tip_stage == TARGET_NO_SOLUTION:
            return TARGET_NO_SOLUTION_COLOUR
        return TARGET_SELECT_COLOUR

    def target_zone_label(self) -> str:
        return "AT SMOOTH LIMIT" if self.workspace_limit_hit else "INSIDE SMOOTH LIMIT"

    def update_target_visual(self) -> None:
        if self.target_marker is None or self.target_error_line is None:
            return
        visible = self.control_mode == TIP_MODE
        self.target_marker.visible = visible
        self.target_error_line.visible = visible
        self.target_marker.set_data(
            np.asarray([self.target_tip_mm], dtype=np.float32),
            symbol="disc",
            face_color=self.target_marker_colour(),
            edge_color=TARGET_EDGE_COLOUR,
            edge_width=1.5,
            size=TARGET_MARKER_SIZE,
        )
        line_colour = (
            LIMIT_LINE_COLOUR
            if self.tip_stage == TARGET_NO_SOLUTION
            else TARGET_LINE_COLOUR
        )
        self.target_error_line.set_data(
            pos=np.asarray(
                [self.selected_endpoint_mm, self.target_tip_mm],
                dtype=np.float32,
            ),
            color=line_colour,
        )
        self.update_target_projection()
        self.canvas.update()

    def clear_motion_plan(self) -> None:
        self.motion_plan = None
        self.motion_plans = []
        self.motion_route = None
        self.solution_index = 0
        self.plan_solution_source = "NONE"
        self.execution_started = 0.0
        self.execution_duration = 0.0
        self.execution_progress = 0.0
        self.execution_phase_index = 0
        self.execution_phase_progress = 0.0
        if self.planned_path_visual is not None:
            self.planned_path_visual.visible = False
        for _tube, line in self.preview_tube_lines:
            line.visible = False
        for _slot, _tube, line in self.alternative_preview_lines:
            line.visible = False

    def set_target_tip(self, target_tip_mm: np.ndarray, action: str) -> None:
        if self.tip_stage not in {
            TARGET_SELECT,
            TARGET_NO_SOLUTION,
            TARGET_REACHED,
        }:
            self.last_action = "Cancel the current preview before editing the target"
            return
        candidate = np.asarray(target_tip_mm, dtype=float).reshape(3)
        if not np.all(np.isfinite(candidate)):
            return
        surface = self.workspace_surfaces[self.selected_tube]
        candidate, limited = constrain_point_to_workspace_surface(
            candidate,
            surface.profile_z_mm,
            surface.profile_radius_mm,
        )
        self.clear_motion_plan()
        self.tip_stage = TARGET_SELECT
        self.target_tip_mm[:] = candidate
        self.tracking_error_mm = float(
            np.linalg.norm(candidate - self.selected_endpoint_mm)
        )
        self.workspace_limit_hit = limited
        self.target_sample_distance_mm = (
            self.target_planner.nearest_sample_distance_mm(candidate)
        )
        self.ik_status = "AWAITING CONFIRMATION"
        self.last_action = (
            f"{action}; workspace boundary reached" if limited else action
        )
        self.update_target_visual()

    def configuration_backbone_mm(
        self,
        deployment_m: np.ndarray,
        rotation_rad: np.ndarray,
    ) -> np.ndarray:
        deployment = self.limits.validate(np.asarray(deployment_m, dtype=float).reshape(3))
        if np.max(np.abs(deployment)) <= 1e-10:
            return np.zeros((2, 3), dtype=np.float32)
        result = joint_viewer.superPosKin(
            self.parameters,
            {
                "ul": deployment.tolist(),
                "uphi": np.asarray(rotation_rad, dtype=float).tolist(),
            },
            self.sim_parameters,
        )
        parts = []
        for index, section in enumerate(result[3]):
            points = np.column_stack(
                [np.asarray(section[axis]) for axis in range(3)]
            )
            if index:
                points = points[1:]
            if len(points):
                parts.append(points)
        if not parts:
            raise RuntimeError("Forward model returned no preview backbone")
        return np.vstack(parts) * 1000.0

    def show_motion_plan(self, plan: MotionPlan) -> None:
        self.motion_route = build_motion_route(plan, self.motion_strategy,
            deployment_limits=self.limits)
        backbone = self.configuration_backbone_mm(
            plan.goal_deployment_m,
            plan.goal_rotation_rad,
        )
        segments = joint_viewer.visible_tube_segments(
            backbone,
            plan.goal_deployment_m,
        )
        for tube, line in self.preview_tube_lines:
            line.set_data(pos=segments[tube])
            line.visible = True
        planned_path = sample_motion_route(
            self.planning_solver,
            self.motion_route,
            samples_per_phase=24,
        )
        path_colour = (
            DIRECT_PATH_COLOUR
            if self.motion_strategy == DIRECT_STRATEGY
            else PLANNED_PATH_COLOUR
        )
        self.planned_path_visual.set_data(pos=planned_path, color=path_colour)
        self.planned_path_visual.visible = True
        self.update_alternative_previews()
        self.canvas.update()

    def update_alternative_previews(self) -> None:
        alternatives = [
            plan
            for index, plan in enumerate(self.motion_plans)
            if index != self.solution_index
        ]
        geometry: dict[tuple[int, int], np.ndarray] = {}
        for slot, plan in enumerate(alternatives[: MAX_ALTERNATIVE_SOLUTIONS - 1]):
            backbone = self.configuration_backbone_mm(
                plan.goal_deployment_m,
                plan.goal_rotation_rad,
            )
            segments = joint_viewer.visible_tube_segments(
                backbone,
                plan.goal_deployment_m,
            )
            for tube in range(3):
                geometry[(slot, tube)] = segments[tube]
        for slot, tube, line in self.alternative_preview_lines:
            points = geometry.get((slot, tube))
            if points is None:
                line.visible = False
            else:
                line.set_data(pos=points)
                line.visible = True

    def select_alternative_solution(self, direction: int) -> None:
        if self.tip_stage != TARGET_PREVIEW or len(self.motion_plans) < 2:
            return
        self.solution_index = (
            self.solution_index + int(np.sign(direction))
        ) % len(self.motion_plans)
        self.motion_plan = self.motion_plans[self.solution_index]
        self.plan_solution_source = self.motion_plan.solution_source
        self.show_motion_plan(self.motion_plan)
        self.last_action = (
            f"IK solution {self.solution_index + 1} of {len(self.motion_plans)} selected"
        )
        self.update_status()

    def select_motion_strategy(self, strategy: str) -> None:
        if strategy not in {DIRECT_STRATEGY, RETRACT_STRATEGY}:
            return
        if self.tip_stage == TARGET_EXECUTING:
            self.last_action = "Stop the current motion before changing route"
            return
        self.motion_strategy = strategy
        if self.tip_stage == TARGET_PREVIEW and self.motion_plan is not None:
            self.show_motion_plan(self.motion_plan)
            self.last_action = f"{strategy} preview selected"
        else:
            self.last_action = f"{strategy} route selected"
        self.update_status()

    def toggle_motion_strategy(self) -> None:
        strategy = (
            RETRACT_STRATEGY
            if self.motion_strategy == DIRECT_STRATEGY
            else DIRECT_STRATEGY
        )
        self.select_motion_strategy(strategy)

    def toggle_control_mode(self) -> None:
        if self.robot_dirty:
            self.update_robot()
        if self.tip_stage == TARGET_EXECUTING:
            self.cancel_tip_action("Motion stopped before changing control mode")
        self.control_mode = TIP_MODE if self.control_mode == JOINT_MODE else JOINT_MODE
        self.target_command_accumulator[:] = 0.0
        self.clear_motion_plan()
        self.tip_stage = TARGET_SELECT
        if self.control_mode == TIP_MODE:
            self.sync_target_to_tip("AWAITING TARGET")
            self.last_action = "Target selection enabled"
        else:
            self.sync_target_to_tip("JOINT CONTROL")
            self.last_action = "Joint control enabled"
        self.update_target_visual()
        self.update_status()

    def toggle_guides(self) -> None:
        super().toggle_guides()
        self.joint_workspace_visual.visible = False
        if self.workspace_surface_visual is not None:
            self.workspace_surface_visual.visible = self.guides_visible
        self.update_endpoint_visuals()
        self.update_conditional_workspace()
        state = "shown" if self.guides_visible else "hidden"
        self.last_action = f"Guides and smooth workspace {state}"
        self.canvas.update()

    def reset(self) -> None:
        self.clear_motion_plan()
        self.tip_stage = TARGET_SELECT
        self.locked_endpoint_targets.clear()
        self.endpoint_lock_modes[:] = [FREE, FREE, FREE]
        super().reset()
        self.sync_target_pending = True
        self.update_conditional_workspace()

    def undo(self) -> None:
        self.clear_motion_plan()
        self.tip_stage = TARGET_SELECT
        super().undo()
        self.sync_target_pending = True

    def update_robot(self) -> None:
        super().update_robot()
        self.update_rear_tube_visuals()
        self.update_endpoint_visuals()
        if self.control_mode == JOINT_MODE or self.sync_target_pending:
            status = "JOINT CONTROL" if self.control_mode == JOINT_MODE else "AWAITING TARGET"
            self.sync_target_to_tip(status)
            self.sync_target_pending = False
        else:
            self.tracking_error_mm = float(
                np.linalg.norm(self.target_tip_mm - self.selected_endpoint_mm)
            )
            self.update_target_visual()

    def calculate_target_plan(self) -> None:
        self.ik_status = "SOLVING TARGET"
        self.last_action = "Calculating target configuration"
        self.update_status()
        lock_targets, lock_weights, lock_tolerances = self.active_lock_parameters()
        attempt = self.target_planner.plan(
            self.target_tip_mm,
            self.deployment,
            self.rotation,
            locked_targets_mm=lock_targets,
            locked_weights=lock_weights,
            locked_tolerances_mm=lock_tolerances,
            maximum_alternatives=MAX_ALTERNATIVE_SOLUTIONS,
        )
        self.last_ik_ms = attempt.solve_time_ms
        self.last_solve_error_mm = attempt.best_error_mm
        self.target_sample_distance_mm = attempt.nearest_sample_mm
        if attempt.plan is None:
            self.clear_motion_plan()
            self.tip_stage = TARGET_NO_SOLUTION
            self.ik_status = "NO SOLUTION FOUND"
            self.last_action = attempt.message
            self.update_target_visual()
            self.update_status()
            return

        self.motion_plans = list(attempt.alternatives or (attempt.plan,))
        self.solution_index = 0
        self.motion_plan = self.motion_plans[0]
        self.plan_solution_source = attempt.plan.solution_source
        self.tip_stage = TARGET_PREVIEW
        self.ik_status = "SOLUTION READY"
        self.last_action = (
            f"Preview ready with {len(self.motion_plans)} IK solution(s); "
            "confirm again to execute"
        )
        self.show_motion_plan(self.motion_plan)
        self.update_target_visual()
        self.update_status()

    def begin_execution(self, now: float | None = None) -> None:
        if (
            self.motion_plan is None
            or self.motion_route is None
            or self.tip_stage != TARGET_PREVIEW
        ):
            return
        self.remember_undo_state()
        self.execution_started = time.perf_counter() if now is None else float(now)
        self.execution_duration = self.motion_route.total_duration_s
        self.execution_progress = 0.0
        self.execution_phase_index = 0
        self.execution_phase_progress = 0.0
        self.tip_stage = TARGET_EXECUTING
        self.ik_status = "SIMULATED MOTION IN PROGRESS"
        self.last_action = f"Executing {self.motion_strategy} route"
        self.update_target_visual()
        self.update_status()

    def update_execution(self, now: float) -> None:
        if (
            self.tip_stage != TARGET_EXECUTING
            or self.motion_plan is None
            or self.motion_route is None
        ):
            return
        elapsed = max(0.0, float(now) - self.execution_started)
        (
            deployment,
            rotation,
            self.execution_progress,
            self.execution_phase_index,
            self.execution_phase_progress,
        ) = route_state_at_time(
            self.motion_route,
            elapsed,
        )
        self.deployment[:] = self.limits.validate(deployment)
        self.rotation[:] = rotation
        self.robot_dirty = True
        if self.execution_progress >= 1.0:
            self.deployment[:] = self.limits.validate(self.motion_plan.goal_deployment_m)
            self.rotation[:] = self.motion_plan.goal_rotation_rad
            self.tip_stage = TARGET_REACHED
            self.ik_status = "TARGET REACHED"
            selected_mode = self.endpoint_lock_modes[self.selected_tube]
            if selected_mode == FREE:
                self.locked_endpoint_targets.pop(self.selected_tube, None)
                lock_action = "remains free"
            else:
                self.locked_endpoint_targets[self.selected_tube] = (
                    self.target_tip_mm.copy()
                )
                lock_action = f"updated {selected_mode.lower()} position"
            self.last_action = (
                f"{joint_viewer.TUBE_NAMES[self.selected_tube]} endpoint reached; "
                f"{lock_action}"
            )
            for _tube, line in self.preview_tube_lines:
                line.visible = False
            for _slot, _tube, line in self.alternative_preview_lines:
                line.visible = False
            self.update_endpoint_visuals()
            self.update_conditional_workspace()
            self.update_target_visual()

    def cancel_tip_action(self, action: str = "Target cancelled") -> None:
        if self.robot_dirty:
            self.update_robot()
        self.clear_motion_plan()
        self.tip_stage = TARGET_SELECT
        self.target_command_accumulator[:] = 0.0
        self.sync_target_to_tip("AWAITING TARGET")
        self.last_action = action
        self.update_status()

    def confirm_tip_action(self, now: float | None = None) -> None:
        if self.tip_stage in {TARGET_SELECT, TARGET_NO_SOLUTION}:
            self.calculate_target_plan()
        elif self.tip_stage == TARGET_PREVIEW:
            self.begin_execution(now)
        elif self.tip_stage == TARGET_REACHED:
            self.tip_stage = TARGET_SELECT
            self.ik_status = "AWAITING TARGET"
            self.last_action = "Move the red target to choose the next point"
            self.update_target_visual()
            self.update_status()

    def update_tip_command(self, now: float, dt: float) -> None:
        self.left_x, self.left_y = self.read_left_stick()
        trigger_command = self.inputs.r2 - self.inputs.l2
        velocity = controller_target_velocity_mm_s(
            self.left_x,
            self.left_y,
            self.inputs.dpad_x,
            self.inputs.dpad_y,
            trigger_command,
            self.selected_endpoint_tangent,
            precision=self.inputs.precision,
        )
        if self.tip_stage not in {
            TARGET_SELECT,
            TARGET_NO_SOLUTION,
            TARGET_REACHED,
        }:
            self.target_command_accumulator[:] = 0.0
            return
        self.target_command_accumulator += velocity * dt
        if (
            np.linalg.norm(self.target_command_accumulator) > 1e-9
            and now - self.last_target_update >= 1.0 / TARGET_UPDATE_HZ
        ):
            self.set_target_tip(
                self.target_tip_mm + self.target_command_accumulator,
                "Target marker moved",
            )
            self.target_command_accumulator[:] = 0.0
            self.last_target_update = now

    def modifier_names(self, event) -> set[str]:
        names = set()
        for modifier in getattr(event, "modifiers", ()):
            name = getattr(modifier, "name", None)
            names.add((name if name is not None else str(modifier)).lower())
        return names

    def adjust_selected_target(self, direction: float, event) -> None:
        step = keyboard_target_step_mm(self.modifier_names(event))
        candidate = self.target_tip_mm.copy()
        candidate[self.keyboard_target_axis] += float(direction) * step
        self.set_target_tip(candidate, "Target coordinate adjusted")
        self.update_status()

    def pose_record(self, label: str) -> dict[str, float | str]:
        record = super().pose_record(label)
        record.update(
            {
                "control_mode": self.control_mode,
                "controlled_endpoint": joint_viewer.TUBE_NAMES[self.selected_tube],
                "target_x_mm": float(self.target_tip_mm[0]),
                "target_y_mm": float(self.target_tip_mm[1]),
                "target_z_mm": float(self.target_tip_mm[2]),
                "tracking_error_mm": float(self.tracking_error_mm),
                "ik_status": self.ik_status,
                "ik_solve_time_ms": float(self.last_ik_ms),
                "tip_workflow_stage": self.tip_stage,
                "plan_solution_source": self.plan_solution_source,
                "motion_strategy": self.motion_strategy,
                "execution_progress": float(self.execution_progress),
                "locked_endpoint_count": len(self.locked_endpoint_targets),
                "inner_lock_mode": self.endpoint_lock_modes[0],
                "middle_lock_mode": self.endpoint_lock_modes[1],
                "outer_lock_mode": self.endpoint_lock_modes[2],
                "ik_solution_index": (
                    self.solution_index + 1 if self.motion_plans else 0
                ),
                "ik_solution_count": len(self.motion_plans),
                "conditional_feasible_samples": self.conditional_feasible_count,
                "conditional_relaxed_samples": self.conditional_relaxed_count,
                "conditional_projection_distance_mm": float(
                    self.conditional_projection_distance_mm
                ),
            }
        )
        return record

    def sidebar_text(self) -> str:
        mm = self.deployment * 1000.0
        rear_mm = (self.total_lengths - self.deployment) * 1000.0
        deg = np.rad2deg(self.rotation)
        state = "CONNECTED" if self.controller.connected else "DISCONNECTED"
        markers = [
            ">" if index == self.selected_tube else " "
            for index in range(3)
        ]
        axis_markers = [
            ">" if self.control_mode == TIP_MODE and index == self.keyboard_target_axis else " "
            for index in range(3)
        ]
        mode = "TIP IK" if self.control_mode == TIP_MODE else "JOINT"
        precision = "PRECISION" if self.inputs.precision else "NORMAL"
        workflow = self.tip_stage if self.control_mode == TIP_MODE else "NOT ACTIVE"
        plate_state = (
            "FULLY RETRACTED"
            if np.max(np.abs(self.deployment)) <= 1e-10
            else "TUBES EXPOSED"
        )
        phase_label = "NONE"
        route_duration = 0.0
        if self.motion_route is not None:
            route_duration = self.motion_route.total_duration_s
            phase_index = min(
                self.execution_phase_index,
                len(self.motion_route.phases) - 1,
            )
            phase_label = self.motion_route.phases[phase_index].label
        selected_position = self.selected_endpoint_mm
        selected_direction = self.selected_endpoint_tangent
        selected_tilt, selected_azimuth = joint_viewer.orientation_from_vertical(
            selected_direction
        )
        endpoint_rows = "\n".join(
            f"{markers[tube]} {joint_viewer.TUBE_NAMES[tube]:6s}  "
            f"{self.endpoint_positions_mm[tube, 0]:7.1f}  "
            f"{self.endpoint_positions_mm[tube, 1]:7.1f}  "
            f"{self.endpoint_positions_mm[tube, 2]:7.1f} mm"
            for tube in range(3)
        )
        lock_rows = []
        for tube, name in enumerate(joint_viewer.TUBE_NAMES):
            mode_label = self.endpoint_lock_modes[tube]
            target = self.locked_endpoint_targets.get(tube)
            if (
                self.tip_stage == TARGET_PREVIEW
                and self.motion_plan is not None
                and np.isfinite(self.motion_plan.endpoint_errors_mm[tube])
            ):
                error_label = f"{self.motion_plan.endpoint_errors_mm[tube]:7.2f}"
            elif target is None:
                error_label = "   --   "
            else:
                error_label = (
                    f"{np.linalg.norm(self.endpoint_positions_mm[tube] - target):7.2f}"
                )
            lock_rows.append(f"{name:6s} {mode_label:9s} {error_label} mm")
        lock_lines = "\n".join(lock_rows)
        solution_count = len(self.motion_plans)
        solution_number = self.solution_index + 1 if solution_count else 0
        text = f"""{self.profile.description(self.deployment)}
F6: change limits | F7: change tubes (resets state)

CONTROL: {mode} ({precision}) | PS5: {state}
ACTIVE ENDPOINT: {joint_viewer.TUBE_NAMES[self.selected_tube]}
WORKFLOW: {workflow}
LAST: {self.last_action}

             EXPOSURE mm   ROTATION deg
{markers[0]} INNER     {mm[0]:7.1f}       {deg[0]:7.1f}
{markers[1]} MIDDLE    {mm[1]:7.1f}       {deg[1]:7.1f}
{markers[2]} OUTER     {mm[2]:7.1f}       {deg[2]:7.1f}
Exposure is material length beyond the front plate.

MODEL ENDPOINTS          X        Y        Z
{endpoint_rows}
TARGET (mm): {self.target_tip_mm[0]:.1f} / {self.target_tip_mm[1]:.1f} / {self.target_tip_mm[2]:.1f}
Numerical residual: {self.tracking_error_mm:.3f} mm
IK: {self.ik_status}
Solution {solution_number}/{solution_count} | Nearest sample: {self.target_sample_distance_mm:.2f} mm

ENDPOINT LOCKS          ERROR
{lock_lines}
Soft {SOFT_LOCK_TOLERANCE_MM:.1f} mm | Hard {HARD_LOCK_TOLERANCE_MM:.1f} mm
Route: {self.motion_strategy}
Phase: {phase_label} | Progress: {100*self.execution_progress:.0f}%
Simulation only; paths are not collision checked.

WORKSPACE: {self.workspace_sample_count:,} valid sampled states
Blue surface: approximate selected-endpoint envelope.
Enclosed points are not guaranteed reachable.
Z=0: front surface of front plate (exit reference).
Behind-plate lines are schematic, not solved shapes.
Inner blue | Middle green | Outer orange

CONTROLS
Tab / L3: joint or Cartesian target control
1/2/3: select endpoint | X/Y/Z: target axis
W/S: exposure (joint); Left/Right: target (Cartesian)
A/D: rotation (joint) | Enter / Cross: plan, confirm
L1/R1: select endpoint | Options: endpoint lock
D-pad in preview: route / alternative solution
R: feasible reset | Circle: undo
Triangle: guides | R3: sidebar | Mouse: orbit / zoom
Cross in joint mode: save waypoint
Options hold: export image and state | Q: quit
"""
        return text + (f"\n\nERROR\n{self.error_text}" if self.error_text else "")

    def control_key_text(self) -> str:
        common = """Right stick   Change view orientation
Square (hold) Precision movement
R3            Hide / show UI
Triangle      Hide / show guides + workspace
Touchpad hold Reset robot
Mouse drag    Change view orientation
Mouse wheel   Zoom in / out"""
        if self.control_mode == TIP_MODE:
            return f"""CONTROLS — TIP MODE ONLY

CONTROLLER
L3            Switch to joint control
Left stick    Move red target X / Y
L1 / R1       Previous / next tube endpoint
L2 / R2       Retract / extend along endpoint direction
D-pad         Fine target X / Y
Square+D-pad  Fine target X / Z
D-pad Up/Down Choose route while previewing
D-pad L/R     Cycle IK solutions while previewing
Options tap   Cycle FREE / SOFT / HARD
Options hold  Export screenshot and data
Cross         Plan target / confirm motion
Circle        Cancel target / stop motion
{common}

KEYBOARD
Tab           Switch to joint control
1 / 2 / 3     Inner / Middle / Outer endpoint
X / Y / Z     Select target coordinate
Left / Right  Adjust target / cycle preview solution
Enter         Plan target / confirm motion
Escape        Cancel target / stop motion
M             Toggle motion route
L             Cycle FREE / SOFT / HARD
E             Export screenshot and data
Shift         0.1 mm step
Control       5 mm step
R             Reset robot
Q             Quit"""
        return f"""CONTROLS — JOINT MODE ONLY

CONTROLLER
L3            Switch to tip IK control
L1 / R1       Previous / next tube
D-pad Up/Down Angle decrease / increase
D-pad Left/Right Extension decrease / increase
Left stick    Unused
Cross         Save waypoint
Circle        Undo adjustment
Options       Export screenshot and data
{common}

KEYBOARD
Tab           Switch to tip IK control
1 / 2 / 3     Inner / Middle / Outer
W / S         Increase / decrease extension
A / D         Decrease / increase angle
R             Reset robot
Q             Quit"""

    def tick(self, _event=None) -> None:
        now = time.perf_counter()
        dt = float(np.clip(now - self.last_tick, 0.0, 0.05))
        self.last_tick = now
        try:
            self.controller.process_events(now)
            if self.controller.connected:
                toggle_pressed = self.controller.pressed(BUTTON_L3)
                l1_pressed = self.controller.pressed(joint_viewer.BUTTON_L1)
                r1_pressed = self.controller.pressed(joint_viewer.BUTTON_R1)

                if toggle_pressed:
                    self.toggle_control_mode()
                if self.control_mode == JOINT_MODE:
                    if l1_pressed:
                        self.select(self.selected_tube - 1)
                    if r1_pressed:
                        self.select(self.selected_tube + 1)
                else:
                    if l1_pressed:
                        self.select_tip_endpoint(self.selected_tube - 1)
                    if r1_pressed:
                        self.select_tip_endpoint(self.selected_tube + 1)

                if self.controller.pressed(joint_viewer.BUTTON_CROSS):
                    if self.control_mode == TIP_MODE:
                        self.confirm_tip_action(now)
                    else:
                        self.save_waypoint()
                if self.controller.pressed(joint_viewer.BUTTON_CIRCLE):
                    if self.control_mode == TIP_MODE:
                        self.cancel_tip_action()
                    else:
                        self.undo()
                if self.controller.pressed(joint_viewer.BUTTON_TRIANGLE):
                    self.toggle_guides()
                if self.controller.pressed(joint_viewer.BUTTON_R3):
                    self.toggle_ui()
                self.update_options_action(now)
                self.update_touchpad_reset(now)
                self.inputs = self.controller.read_inputs()

                if self.control_mode == JOINT_MODE:
                    self.previous_strategy_dpad_y = 0
                    self.previous_solution_dpad_x = 0
                    dpad = (self.inputs.dpad_x, self.inputs.dpad_y)
                    if dpad != (0, 0) and dpad != self.previous_dpad:
                        self.remember_undo_state()
                    self.previous_dpad = dpad
                    rotation_speed = (
                        joint_viewer.PRECISION_ROTATION_SPEED_DEG_S
                        if self.inputs.precision
                        else joint_viewer.DPAD_ROTATION_SPEED_DEG_S
                    )
                    translation_speed = (
                        joint_viewer.PRECISION_TRANSLATION_SPEED_MM_S
                        if self.inputs.precision
                        else joint_viewer.DPAD_TRANSLATION_SPEED_MM_S
                    )
                    if self.inputs.dpad_y:
                        self.rotate(self.inputs.dpad_y * rotation_speed * dt)
                    if self.inputs.dpad_x:
                        self.move(self.inputs.dpad_x * translation_speed * dt)
                    self.left_x = 0.0
                    self.left_y = 0.0
                    self.target_command_accumulator[:] = 0.0
                else:
                    self.previous_dpad = (0, 0)
                    strategy_dpad_y = int(self.inputs.dpad_y)
                    solution_dpad_x = int(self.inputs.dpad_x)
                    if (
                        self.tip_stage == TARGET_PREVIEW
                        and strategy_dpad_y
                        and not self.previous_strategy_dpad_y
                    ):
                        self.select_motion_strategy(
                            DIRECT_STRATEGY
                            if strategy_dpad_y > 0
                            else RETRACT_STRATEGY
                        )
                    self.previous_strategy_dpad_y = strategy_dpad_y
                    if (
                        self.tip_stage == TARGET_PREVIEW
                        and solution_dpad_x
                        and not self.previous_solution_dpad_x
                    ):
                        self.select_alternative_solution(solution_dpad_x)
                    self.previous_solution_dpad_x = solution_dpad_x
                    if self.tip_stage != TARGET_PREVIEW:
                        self.update_tip_command(now, dt)

                if self.inputs.right_x or self.inputs.right_y:
                    camera = self.view.camera
                    camera.azimuth += (
                        self.inputs.right_x
                        * joint_viewer.CAMERA_ORBIT_SPEED_DEG_S
                        * dt
                    )
                    camera.elevation = float(
                        np.clip(
                            camera.elevation
                            - self.inputs.right_y
                            * joint_viewer.CAMERA_ORBIT_SPEED_DEG_S
                            * dt,
                            -85.0,
                            85.0,
                        )
                    )
                    self.canvas.update()

            else:
                self.inputs = joint_viewer.InputSnapshot()
                self.previous_dpad = (0, 0)
                self.left_x = 0.0
                self.left_y = 0.0
                self.target_command_accumulator[:] = 0.0
                self.previous_strategy_dpad_y = 0
                self.previous_solution_dpad_x = 0
                self.options_hold_start = None
                self.options_hold_fired = False

            self.update_execution(now)

            if (
                self.robot_dirty
                and now - self.last_model_update >= 1.0 / joint_viewer.MODEL_UPDATE_HZ
            ):
                self.update_robot()
                self.last_model_update = now
            if now - self.last_status_update >= 1.0 / joint_viewer.STATUS_UPDATE_HZ:
                self.update_status()
                self.last_status_update = now
            self.error_text = ""
        except pygame.error as exc:
            self.controller.disconnect()
            self.inputs = joint_viewer.InputSnapshot()
            self.record_error(exc, now)
        except Exception as exc:
            self.record_error(exc, now)
        finally:
            self.update_rates(now)

    def on_key_press(self, event) -> None:
        key = event.key.name.lower() if event.key is not None else ""
        if self.profile_key(key):
            return
        if key == "q":
            self.canvas.close()
            return
        if key == "r":
            self.reset_with_undo()
            return
        if key == "tab":
            self.toggle_control_mode()
            return

        if self.control_mode == TIP_MODE:
            if key in {"enter", "return", "space"}:
                self.confirm_tip_action()
            elif key == "escape":
                self.cancel_tip_action()
            elif key == "m":
                self.toggle_motion_strategy()
            elif key == "l":
                self.cycle_selected_lock_mode()
            elif key == "e":
                self.export_snapshot()
            elif key in {"1", "2", "3"}:
                self.select_tip_endpoint(int(key) - 1)
            elif key in {"x", "y", "z"}:
                self.keyboard_target_axis = AXIS_NAMES.index(key.upper())
                self.last_action = f"Target {key.upper()} selected"
                self.update_status()
            elif key == "left":
                if self.tip_stage == TARGET_PREVIEW:
                    self.select_alternative_solution(-1)
                else:
                    self.adjust_selected_target(-1.0, event)
            elif key == "right":
                if self.tip_stage == TARGET_PREVIEW:
                    self.select_alternative_solution(1)
                else:
                    self.adjust_selected_target(1.0, event)
            return

        actions = {
            "1": lambda: self.select(0),
            "2": lambda: self.select(1),
            "3": lambda: self.select(2),
            "w": lambda: self.keyboard_move(
                joint_viewer.KEYBOARD_TRANSLATION_STEP_MM
            ),
            "s": lambda: self.keyboard_move(
                -joint_viewer.KEYBOARD_TRANSLATION_STEP_MM
            ),
            "a": lambda: self.keyboard_rotate(
                -joint_viewer.KEYBOARD_ROTATION_STEP_DEG
            ),
            "d": lambda: self.keyboard_rotate(
                joint_viewer.KEYBOARD_ROTATION_STEP_DEG
            ),
        }
        action = actions.get(key)
        if action is not None:
            action()

    def run(self) -> None:
        print(
            "\nVISPY PS5 CTR TIP-CONTROL SIMULATOR\n"
            "L1/R1 endpoint | Options lock mode | move red target | "
            "Cross/Enter plan and confirm | preview D-pad selects route/solution\n"
        )
        self.canvas.show()
        self.timer.start()
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.on_close()
            print("VisPy CTR tip-control simulator closed.")


def main() -> None:
    args = joint_viewer.viewer_arguments()
    viewer = VisPyCTRTipControlViewer(mode=args.mode, tubes=args.tubes)
    if args.waypoints:
        viewer.load_waypoints(args.waypoints)
    viewer.run()


if __name__ == "__main__":
    main()
