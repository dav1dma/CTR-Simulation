"""Target planning and smooth actuator interpolation for the CTR simulator."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ctr_inverse_kinematics import ConstrainedTipIK, IKResult, wrap_angles
from ctr_workspace_map import WorkspaceMap


DIRECT_STRATEGY = "DIRECT"
RETRACT_STRATEGY = "RETRACT / REORIENT / ADVANCE"


@dataclass(frozen=True)
class MotionPlan:
    """A confirmed target and the actuator configuration proposed for it."""

    target_tip_mm: np.ndarray
    start_deployment_m: np.ndarray
    start_rotation_rad: np.ndarray
    goal_deployment_m: np.ndarray
    goal_rotation_rad: np.ndarray
    achieved_tip_mm: np.ndarray
    position_error_mm: float
    solve_time_ms: float
    solution_source: str
    nearest_sample_mm: float
    target_tube: int
    locked_targets_mm: dict[int, np.ndarray]
    endpoint_errors_mm: np.ndarray


@dataclass(frozen=True)
class PlanAttempt:
    """The outcome of planning without changing the displayed robot state."""

    plan: MotionPlan | None
    solve_time_ms: float
    nearest_sample_mm: float
    best_error_mm: float
    message: str
    alternatives: tuple[MotionPlan, ...] = ()


@dataclass(frozen=True)
class MotionPhase:
    """One continuous actuator segment in a confirmed route."""

    label: str
    start_deployment_m: np.ndarray
    start_rotation_rad: np.ndarray
    goal_deployment_m: np.ndarray
    goal_rotation_rad: np.ndarray
    duration_s: float


@dataclass(frozen=True)
class MotionRoute:
    """An ordered route from the current configuration to an IK solution."""

    strategy: str
    phases: tuple[MotionPhase, ...]
    total_duration_s: float
    target_tube: int


def shortest_angle_delta(
    start_rotation_rad: np.ndarray,
    goal_rotation_rad: np.ndarray,
) -> np.ndarray:
    """Return the signed shortest rotation from start to goal for each tube."""
    start = np.asarray(start_rotation_rad, dtype=float).reshape(3)
    goal = np.asarray(goal_rotation_rad, dtype=float).reshape(3)
    return wrap_angles(goal - start)


def smoothstep(progress: float) -> float:
    """Ease a normalised movement so it starts and stops without a velocity jump."""
    value = float(np.clip(progress, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def interpolate_actuators(
    plan: MotionPlan,
    progress: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a planned actuator movement using shortest-angle rotation."""
    blend = smoothstep(progress)
    deployment = plan.start_deployment_m + blend * (
        plan.goal_deployment_m - plan.start_deployment_m
    )
    rotation = wrap_angles(
        plan.start_rotation_rad
        + blend
        * shortest_angle_delta(plan.start_rotation_rad, plan.goal_rotation_rad)
    )
    return deployment, rotation


def interpolate_phase(
    phase: MotionPhase,
    progress: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one route phase with the same easing as a direct plan."""
    blend = smoothstep(progress)
    deployment = phase.start_deployment_m + blend * (
        phase.goal_deployment_m - phase.start_deployment_m
    )
    rotation = wrap_angles(
        phase.start_rotation_rad
        + blend
        * shortest_angle_delta(phase.start_rotation_rad, phase.goal_rotation_rad)
    )
    return deployment, rotation


def actuator_motion_duration_s(
    start_deployment_m: np.ndarray,
    start_rotation_rad: np.ndarray,
    goal_deployment_m: np.ndarray,
    goal_rotation_rad: np.ndarray,
    *,
    translation_speed_mm_s: float = 25.0,
    rotation_speed_deg_s: float = 60.0,
    minimum_s: float = 0.4,
    maximum_s: float = 8.0,
) -> float:
    """Choose a duration for one simultaneous translation/rotation segment."""
    if translation_speed_mm_s <= 0.0 or rotation_speed_deg_s <= 0.0:
        raise ValueError("actuator speeds must be positive")
    translation_mm = float(
        np.max(
            np.abs(
                np.asarray(goal_deployment_m, dtype=float)
                - np.asarray(start_deployment_m, dtype=float)
            )
        )
        * 1000.0
    )
    rotation_deg = float(
        np.max(
            np.abs(
                np.rad2deg(
                    shortest_angle_delta(start_rotation_rad, goal_rotation_rad)
                )
            )
        )
    )
    required = max(
        translation_mm / translation_speed_mm_s,
        rotation_deg / rotation_speed_deg_s,
    )
    return float(np.clip(required, minimum_s, maximum_s))


def motion_duration_s(
    plan: MotionPlan,
    *,
    translation_speed_mm_s: float = 25.0,
    rotation_speed_deg_s: float = 60.0,
    minimum_s: float = 0.4,
    maximum_s: float = 8.0,
) -> float:
    """Choose a duration that respects the configured simulated actuator rates."""
    return actuator_motion_duration_s(
        plan.start_deployment_m,
        plan.start_rotation_rad,
        plan.goal_deployment_m,
        plan.goal_rotation_rad,
        translation_speed_mm_s=translation_speed_mm_s,
        rotation_speed_deg_s=rotation_speed_deg_s,
        minimum_s=minimum_s,
        maximum_s=maximum_s,
    )


def build_motion_route(
    plan: MotionPlan,
    strategy: str,
    *,
    retracted_deployment_m: np.ndarray | None = None,
) -> MotionRoute:
    """Build either a direct route or a full retract/reorient/advance route.

    The retracted route is a free-space simulation. It does not imply that the
    path is safe until a sheath, anatomy mesh, and collision model are added.
    """
    if strategy not in {DIRECT_STRATEGY, RETRACT_STRATEGY}:
        raise ValueError(f"Unknown motion strategy: {strategy}")
    if retracted_deployment_m is None:
        retracted = np.zeros(3, dtype=float)
    else:
        retracted = np.asarray(retracted_deployment_m, dtype=float).reshape(3)
    if np.any(retracted < 0.0) or not (
        retracted[2] <= retracted[1] <= retracted[0]
    ):
        raise ValueError("retracted deployment must satisfy outer <= middle <= inner")

    if strategy == DIRECT_STRATEGY:
        phases = (
            MotionPhase(
                label="DIRECT TO TARGET",
                start_deployment_m=plan.start_deployment_m.copy(),
                start_rotation_rad=plan.start_rotation_rad.copy(),
                goal_deployment_m=plan.goal_deployment_m.copy(),
                goal_rotation_rad=plan.goal_rotation_rad.copy(),
                duration_s=motion_duration_s(plan),
            ),
        )
    else:
        phases = (
            MotionPhase(
                label="RETRACT BEHIND PLATE",
                start_deployment_m=plan.start_deployment_m.copy(),
                start_rotation_rad=plan.start_rotation_rad.copy(),
                goal_deployment_m=retracted.copy(),
                goal_rotation_rad=plan.start_rotation_rad.copy(),
                duration_s=actuator_motion_duration_s(
                    plan.start_deployment_m,
                    plan.start_rotation_rad,
                    retracted,
                    plan.start_rotation_rad,
                ),
            ),
            MotionPhase(
                label="REORIENT WHILE RETRACTED",
                start_deployment_m=retracted.copy(),
                start_rotation_rad=plan.start_rotation_rad.copy(),
                goal_deployment_m=retracted.copy(),
                goal_rotation_rad=plan.goal_rotation_rad.copy(),
                duration_s=actuator_motion_duration_s(
                    retracted,
                    plan.start_rotation_rad,
                    retracted,
                    plan.goal_rotation_rad,
                ),
            ),
            MotionPhase(
                label="ADVANCE TO TARGET",
                start_deployment_m=retracted.copy(),
                start_rotation_rad=plan.goal_rotation_rad.copy(),
                goal_deployment_m=plan.goal_deployment_m.copy(),
                goal_rotation_rad=plan.goal_rotation_rad.copy(),
                duration_s=actuator_motion_duration_s(
                    retracted,
                    plan.goal_rotation_rad,
                    plan.goal_deployment_m,
                    plan.goal_rotation_rad,
                ),
            ),
        )
    return MotionRoute(
        strategy=strategy,
        phases=phases,
        total_duration_s=float(sum(phase.duration_s for phase in phases)),
        target_tube=plan.target_tube,
    )


def route_state_at_time(
    route: MotionRoute,
    elapsed_s: float,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """Return actuator state, total progress, phase index and phase progress."""
    elapsed = float(np.clip(elapsed_s, 0.0, route.total_duration_s))
    passed = 0.0
    for index, phase in enumerate(route.phases):
        phase_end = passed + phase.duration_s
        if elapsed <= phase_end or index == len(route.phases) - 1:
            phase_progress = float(
                np.clip((elapsed - passed) / max(phase.duration_s, 1e-9), 0.0, 1.0)
            )
            deployment, rotation = interpolate_phase(phase, phase_progress)
            overall = elapsed / max(route.total_duration_s, 1e-9)
            return deployment, rotation, float(overall), index, phase_progress
        passed = phase_end
    raise RuntimeError("motion route contains no phases")


class TipTargetPlanner:
    """Plan a tip target using local IK first and sampled restarts second.

    The sampled restarts make target confirmation more capable than the live
    resolved-rate controller while keeping all potentially discontinuous
    configuration changes behind an explicit preview and confirmation step.
    """

    def __init__(
        self,
        local_solver: ConstrainedTipIK,
        restart_solver: ConstrainedTipIK,
        workspace: WorkspaceMap,
        *,
        target_tube: int = 0,
        restart_count: int = 24,
        maximum_seed_distance_mm: float = 45.0,
    ) -> None:
        self.local_solver = local_solver
        self.restart_solver = restart_solver
        self.workspace = workspace
        self.target_tube = int(target_tube)
        self.restart_count = int(restart_count)
        self.maximum_seed_distance_mm = float(maximum_seed_distance_mm)
        if self.target_tube not in (0, 1, 2):
            raise ValueError("target_tube must be 0, 1, or 2")
        if self.restart_count < 1:
            raise ValueError("restart_count must be positive")
        if self.maximum_seed_distance_mm <= 0.0:
            raise ValueError("maximum_seed_distance_mm must be positive")
        self._tree = cKDTree(np.asarray(workspace.tips_mm, dtype=float))

    def nearest_sample_distance_mm(self, target_tip_mm: np.ndarray) -> float:
        target = np.asarray(target_tip_mm, dtype=float).reshape(3)
        return float(self._tree.query(target, k=1)[0])

    def _configuration_cost(
        self,
        result: IKResult,
        current_deployment_m: np.ndarray,
        current_rotation_rad: np.ndarray,
    ) -> float:
        translation_cost = np.linalg.norm(
            (result.deployment_m - current_deployment_m)
            / self.local_solver.total_lengths
        )
        rotation_cost = np.linalg.norm(
            shortest_angle_delta(current_rotation_rad, result.rotation_rad) / np.pi
        )
        return float(translation_cost + 0.35 * rotation_cost)

    def plan(
        self,
        target_tip_mm: np.ndarray,
        current_deployment_m: np.ndarray,
        current_rotation_rad: np.ndarray,
        *,
        locked_targets_mm: dict[int, np.ndarray] | None = None,
        locked_weights: dict[int, float] | None = None,
        locked_tolerances_mm: dict[int, float] | None = None,
        maximum_alternatives: int = 6,
    ) -> PlanAttempt:
        """Calculate a target configuration without mutating viewer state."""
        if maximum_alternatives < 1:
            raise ValueError("maximum_alternatives must be positive")
        target = np.asarray(target_tip_mm, dtype=float).reshape(3)
        current_deployment = np.asarray(current_deployment_m, dtype=float).reshape(3)
        current_rotation = np.asarray(current_rotation_rad, dtype=float).reshape(3)
        started = time.perf_counter()

        local = self.local_solver.solve(
            target,
            current_deployment,
            current_rotation,
            target_tube=self.target_tube,
            locked_targets_mm=locked_targets_mm,
            locked_weights=locked_weights,
            locked_tolerances_mm=locked_tolerances_mm,
        )
        nearest_distance, nearest_index = self._tree.query(
            target,
            k=min(self.restart_count, len(self.workspace.tips_mm)),
        )
        distances = np.atleast_1d(nearest_distance).astype(float)
        indices = np.atleast_1d(nearest_index).astype(int)
        nearest = float(distances[0])
        best_error = float(local.position_error_mm)

        if not local.reached and nearest > self.maximum_seed_distance_mm:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return PlanAttempt(
                plan=None,
                solve_time_ms=elapsed_ms,
                nearest_sample_mm=nearest,
                best_error_mm=best_error,
                message="No sampled configuration near this target",
            )

        reached: list[tuple[IKResult, str]] = []
        if local.reached:
            reached.append((local, "LOCAL IK"))
        for index in indices:
            candidate = self.restart_solver.solve(
                target,
                self.workspace.deployment_m[index],
                self.workspace.rotation_rad[index],
                target_tube=self.target_tube,
                locked_targets_mm=locked_targets_mm,
                locked_weights=locked_weights,
                locked_tolerances_mm=locked_tolerances_mm,
            )
            best_error = min(best_error, candidate.position_error_mm)
            if candidate.reached:
                reached.append((candidate, "WORKSPACE RESTART"))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not reached:
            return PlanAttempt(
                plan=None,
                solve_time_ms=elapsed_ms,
                nearest_sample_mm=nearest,
                best_error_mm=best_error,
                message="No IK solution found at the current search resolution",
            )

        diverse = self._diverse_solutions(
            reached,
            current_deployment,
            current_rotation,
            maximum_alternatives,
        )
        plans = tuple(
            self._motion_plan(
                target,
                current_deployment,
                current_rotation,
                result,
                elapsed_ms,
                source,
                nearest,
                locked_targets_mm,
            )
            for result, source in diverse
        )
        return PlanAttempt(
            plan=plans[0],
            solve_time_ms=elapsed_ms,
            nearest_sample_mm=nearest,
            best_error_mm=min(result.position_error_mm for result, _source in reached),
            message=f"{len(plans)} diverse IK solution(s) ready",
            alternatives=plans,
        )

    def _diverse_solutions(
        self,
        reached: list[tuple[IKResult, str]],
        current_deployment_m: np.ndarray,
        current_rotation_rad: np.ndarray,
        maximum_alternatives: int,
        *,
        separation: float = 0.16,
    ) -> list[tuple[IKResult, str]]:
        """Keep low-cost actuator solutions that are meaningfully different."""
        ordered = sorted(
            reached,
            key=lambda item: self._configuration_cost(
                item[0],
                current_deployment_m,
                current_rotation_rad,
            ),
        )
        local_index = next(
            (
                index
                for index, (_result, source) in enumerate(ordered)
                if source == "LOCAL IK"
            ),
            None,
        )
        if local_index is not None:
            ordered.insert(0, ordered.pop(local_index))

        selected: list[tuple[IKResult, str]] = []
        for candidate in ordered:
            if all(
                self._solution_distance(candidate[0], existing[0]) >= separation
                for existing in selected
            ):
                selected.append(candidate)
            if len(selected) >= maximum_alternatives:
                break
        return selected or [ordered[0]]

    def _solution_distance(self, first: IKResult, second: IKResult) -> float:
        translation = np.linalg.norm(
            (first.deployment_m - second.deployment_m)
            / self.local_solver.total_lengths
        )
        rotation = np.linalg.norm(
            shortest_angle_delta(first.rotation_rad, second.rotation_rad) / np.pi
        )
        return float(translation + 0.35 * rotation)

    @staticmethod
    def _motion_plan(
        target_tip_mm: np.ndarray,
        current_deployment_m: np.ndarray,
        current_rotation_rad: np.ndarray,
        result: IKResult,
        solve_time_ms: float,
        source: str,
        nearest_sample_mm: float,
        locked_targets_mm: dict[int, np.ndarray] | None,
    ) -> MotionPlan:
        return MotionPlan(
            target_tip_mm=np.asarray(target_tip_mm, dtype=float).copy(),
            start_deployment_m=np.asarray(current_deployment_m, dtype=float).copy(),
            start_rotation_rad=np.asarray(current_rotation_rad, dtype=float).copy(),
            goal_deployment_m=result.deployment_m.copy(),
            goal_rotation_rad=result.rotation_rad.copy(),
            achieved_tip_mm=result.achieved_tip_mm.copy(),
            position_error_mm=float(result.position_error_mm),
            solve_time_ms=float(solve_time_ms),
            solution_source=source,
            nearest_sample_mm=float(nearest_sample_mm),
            target_tube=result.target_tube,
            locked_targets_mm={
                int(tube): np.asarray(position, dtype=float).copy()
                for tube, position in (locked_targets_mm or {}).items()
                if int(tube) != result.target_tube
            },
            endpoint_errors_mm=result.endpoint_errors_mm.copy(),
        )


def sample_planned_tip_path(
    solver: ConstrainedTipIK,
    plan: MotionPlan,
    *,
    sample_count: int = 32,
) -> np.ndarray:
    """Sample the tip path produced by simultaneous actuator interpolation."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    path = np.empty((sample_count, 3), dtype=np.float32)
    for index, progress in enumerate(np.linspace(0.0, 1.0, sample_count)):
        deployment, rotation = interpolate_actuators(plan, float(progress))
        path[index] = solver.forward_endpoint_mm(
            deployment,
            rotation,
            plan.target_tube,
        )
    return path


def tip_for_actuator_state(
    solver: ConstrainedTipIK,
    deployment_m: np.ndarray,
    rotation_rad: np.ndarray,
    target_tube: int = 0,
) -> np.ndarray:
    """Return the exposed tip, using the plate origin at full retraction."""
    deployment = np.asarray(deployment_m, dtype=float).reshape(3)
    if deployment[int(target_tube)] <= 1e-10:
        return np.zeros(3, dtype=float)
    return solver.forward_endpoint_mm(deployment, rotation_rad, target_tube)


def sample_motion_route(
    solver: ConstrainedTipIK,
    route: MotionRoute,
    *,
    samples_per_phase: int = 20,
) -> np.ndarray:
    """Sample the complete tip trace of a direct or multi-stage route."""
    if samples_per_phase < 2:
        raise ValueError("samples_per_phase must be at least two")
    points: list[np.ndarray] = []
    for phase_index, phase in enumerate(route.phases):
        progress_values = np.linspace(0.0, 1.0, samples_per_phase)
        if phase_index:
            progress_values = progress_values[1:]
        for progress in progress_values:
            deployment, rotation = interpolate_phase(phase, float(progress))
            points.append(
                tip_for_actuator_state(
                    solver,
                    deployment,
                    rotation,
                    route.target_tube,
                )
            )
    return np.asarray(points, dtype=np.float32)
