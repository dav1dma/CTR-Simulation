"""Constrained numerical inverse kinematics for CTR tip-position control."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from CTR_superPosKin_fun_sectioned import superPosKin
from ctr_operating_profile import DeploymentLimits


IK_TERMINATION_REASONS = frozenset(
    {
        "initial-tolerance-met",
        "tolerance-met",
        "max-iterations",
        "no-improving-step",
        "nonfinite-step",
        "linear-solve-failed",
    }
)


@dataclass(frozen=True)
class IKResult:
    """One inverse-kinematics solution and its tracking information."""

    deployment_m: np.ndarray
    rotation_rad: np.ndarray
    achieved_tip_mm: np.ndarray
    target_tip_mm: np.ndarray
    position_error_mm: float
    solver_converged: bool
    iterations: int
    evaluations: int
    solve_time_ms: float
    message: str
    target_tube: int
    endpoint_positions_mm: np.ndarray
    endpoint_errors_mm: np.ndarray
    termination_reason: str
    solver_tolerance_mm: float
    primary_tolerance_met: bool
    locks_satisfied: bool
    initial_position_error_mm: float

    @property
    def reached(self) -> bool:
        """Compatibility alias used by the interactive motion planner."""
        return self.solver_converged


def wrap_angles(angles_rad: np.ndarray) -> np.ndarray:
    """Wrap angles to the displayed -pi .. +pi range."""
    angles = np.asarray(angles_rad, dtype=float)
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


class ConstrainedTipIK:
    """Damped least-squares IK with nested tube-extension constraints.

    The three deployment variables are encoded as fractions that always decode
    to ``outer <= middle <= inner`` and never exceed the physical tube lengths.
    Rotation variables are normalised by pi so translation and rotation changes
    have comparable numerical scale. The damped pseudoinverse selects a local,
    minimum-change solution for the redundant six-input/three-output problem.
    """

    def __init__(
        self,
        parameters: dict,
        *,
        deployment_limits: DeploymentLimits | None = None,
        model_points_per_section: int = 2,
        damping_mm: float = 2.0,
        tolerance_mm: float | None = None,
        solver_tolerance_mm: float | None = None,
        max_iterations: int = 4,
        finite_difference_step: float = 1e-4,
        max_normalised_step: float = 0.06,
    ) -> None:
        self.parameters = parameters
        self.sim_parameters = {
            "n_p": int(model_points_per_section),
            "isPlot": False,
        }
        self.total_lengths = np.asarray(
            [sum(lengths) for lengths in parameters["l_t"]], dtype=float
        )
        if self.total_lengths.shape != (3,):
            raise ValueError("Tip IK currently requires exactly three tubes.")
        if not (
            self.total_lengths[0] >= self.total_lengths[1]
            and self.total_lengths[1] >= self.total_lengths[2]
            and np.all(self.total_lengths > 0.0)
        ):
            raise ValueError(
                "Tube lengths must be positive and ordered inner >= middle >= outer."
            )
        self.deployment_limits = deployment_limits or DeploymentLimits(
            (0.0, 0.0, 0.0), tuple(self.total_lengths)
        )
        if np.any(np.asarray(self.deployment_limits.upper_m) > self.total_lengths + 1e-12):
            raise ValueError("Deployment limit exceeds tube length")
        if tolerance_mm is not None and solver_tolerance_mm is not None:
            if not np.isclose(
                float(tolerance_mm),
                float(solver_tolerance_mm),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    "tolerance_mm and solver_tolerance_mm cannot disagree"
                )
        selected_tolerance = (
            0.15
            if tolerance_mm is None and solver_tolerance_mm is None
            else float(
                solver_tolerance_mm
                if solver_tolerance_mm is not None
                else tolerance_mm
            )
        )
        self.damping_mm = float(damping_mm)
        self.solver_tolerance_mm = selected_tolerance
        self.max_iterations = int(max_iterations)
        self.finite_difference_step = float(finite_difference_step)
        self.max_normalised_step = float(max_normalised_step)
        if self.damping_mm <= 0.0 or not np.isfinite(self.damping_mm):
            raise ValueError("damping_mm must be finite and positive")
        if (
            self.solver_tolerance_mm <= 0.0
            or not np.isfinite(self.solver_tolerance_mm)
        ):
            raise ValueError("solver tolerance must be finite and positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if (
            self.finite_difference_step <= 0.0
            or not np.isfinite(self.finite_difference_step)
        ):
            raise ValueError("finite_difference_step must be finite and positive")
        if (
            self.max_normalised_step <= 0.0
            or not np.isfinite(self.max_normalised_step)
        ):
            raise ValueError("max_normalised_step must be finite and positive")

    @property
    def tolerance_mm(self) -> float:
        """Legacy name for the numerical solver stopping tolerance."""
        return self.solver_tolerance_mm

    @tolerance_mm.setter
    def tolerance_mm(self, value: float) -> None:
        number = float(value)
        if number <= 0.0 or not np.isfinite(number):
            raise ValueError("tolerance_mm must be finite and positive")
        self.solver_tolerance_mm = number

    @staticmethod
    def _vector(values: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain three finite values.")
        return vector

    def validate_deployment(self, deployment_m: np.ndarray) -> np.ndarray:
        return self.deployment_limits.validate(self._vector(deployment_m, "deployment_m"))

    def encode_deployment(self, deployment_m: np.ndarray) -> np.ndarray:
        """Encode a valid state under the active bounds."""
        return self.deployment_limits.encode(self.validate_deployment(deployment_m))

    def decode_deployment(self, fractions: np.ndarray) -> np.ndarray:
        """Decode fractions into valid nested exposures under the active bounds."""
        return self.deployment_limits.decode(self._vector(fractions, "deployment fractions"))

    def forward_tip_mm(
        self, deployment_m: np.ndarray, rotation_rad: np.ndarray
    ) -> np.ndarray:
        """Return the distal (inner-tube) endpoint for compatibility."""
        return self.forward_endpoints_mm(deployment_m, rotation_rad)[0]

    def forward_endpoints_mm(
        self, deployment_m: np.ndarray, rotation_rad: np.ndarray
    ) -> np.ndarray:
        """Return inner, middle and outer endpoint positions in millimetres."""
        deployment = self.validate_deployment(deployment_m)
        rotation = self._vector(rotation_rad, "rotation_rad")
        if np.max(np.abs(deployment)) <= 1e-12:
            return np.zeros((3, 3), dtype=float)
        result = superPosKin(
            self.parameters,
            {"ul": deployment.tolist(), "uphi": rotation.tolist()},
            self.sim_parameters,
        )
        local_arc_lengths, sections = result[2], result[3]
        arc_parts: list[np.ndarray] = []
        point_parts: list[np.ndarray] = []
        offset = 0.0
        for section_index, (section_s, section) in enumerate(
            zip(local_arc_lengths, sections)
        ):
            local = np.asarray(section_s, dtype=float)
            points = np.column_stack(
                [np.asarray(section[axis], dtype=float) for axis in range(3)]
            )
            if section_index:
                local = local[1:]
                points = points[1:]
            if len(local):
                arc_parts.append(offset + local)
                point_parts.append(points)
                offset += float(section_s[-1])
        if not arc_parts:
            return np.zeros((3, 3), dtype=float)
        arc = np.concatenate(arc_parts)
        points_m = np.vstack(point_parts)
        endpoints_m = np.column_stack(
            [
                np.interp(deployment, arc, points_m[:, axis])
                for axis in range(3)
            ]
        )
        return endpoints_m * 1000.0

    def forward_endpoint_mm(
        self,
        deployment_m: np.ndarray,
        rotation_rad: np.ndarray,
        tube: int,
    ) -> np.ndarray:
        """Return one tube endpoint using INNER=0, MIDDLE=1, OUTER=2."""
        tube_index = int(tube)
        if tube_index not in (0, 1, 2):
            raise ValueError("tube must be 0 (inner), 1 (middle), or 2 (outer)")
        return self.forward_endpoints_mm(deployment_m, rotation_rad)[tube_index]

    def _decode_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        deployment = self.decode_deployment(state[:3])
        rotation = np.pi * np.asarray(state[3:], dtype=float)
        return deployment, rotation

    def solve(
        self,
        target_tip_mm: np.ndarray,
        current_deployment_m: np.ndarray,
        current_rotation_rad: np.ndarray,
        *,
        target_tube: int = 0,
        locked_targets_mm: dict[int, np.ndarray] | None = None,
        locked_weight: float = 0.65,
        locked_tolerance_mm: float = 2.0,
        locked_weights: dict[int, float] | None = None,
        locked_tolerances_mm: dict[int, float] | None = None,
    ) -> IKResult:
        """Move locally toward a selected Cartesian endpoint target.

        Small interactive target changes normally converge in one or two
        iterations. An unreachable target returns the closest improving state
        found and sets ``reached`` to ``False``. Other tube endpoints may be
        supplied as soft locks; this supports sequential shape control without
        pretending that nine Cartesian constraints fit six actuators exactly.
        """
        target = self._vector(target_tip_mm, "target_tip_mm")
        primary_tube = int(target_tube)
        if primary_tube not in (0, 1, 2):
            raise ValueError("target_tube must be 0, 1, or 2")
        if not 0.0 < locked_weight <= 1.0:
            raise ValueError("locked_weight must be in the range (0, 1]")
        if locked_tolerance_mm <= 0.0:
            raise ValueError("locked_tolerance_mm must be positive")

        locks: dict[int, np.ndarray] = {}
        for tube, locked_target in (locked_targets_mm or {}).items():
            tube_index = int(tube)
            if tube_index not in (0, 1, 2):
                raise ValueError("locked endpoint tube must be 0, 1, or 2")
            if tube_index != primary_tube:
                locks[tube_index] = self._vector(
                    locked_target,
                    f"locked target for tube {tube_index}",
                )
        task_tubes = (primary_tube, *sorted(locks))
        task_targets = np.vstack((target, *(locks[tube] for tube in sorted(locks))))
        lock_weights = {
            tube: float((locked_weights or {}).get(tube, locked_weight))
            for tube in locks
        }
        lock_tolerances = {
            tube: float(
                (locked_tolerances_mm or {}).get(tube, locked_tolerance_mm)
            )
            for tube in locks
        }
        if any(not 0.0 < value <= 1.0 for value in lock_weights.values()):
            raise ValueError("each locked endpoint weight must be in (0, 1]")
        if any(value <= 0.0 for value in lock_tolerances.values()):
            raise ValueError("each locked endpoint tolerance must be positive")
        task_weights = np.asarray(
            [1.0, *(lock_weights[tube] for tube in sorted(locks))],
            dtype=float,
        )
        deployment = self.validate_deployment(current_deployment_m)
        rotation = self._vector(current_rotation_rad, "current_rotation_rad")
        state = np.concatenate((self.encode_deployment(deployment), rotation / np.pi))

        started = time.perf_counter()
        endpoints = self.forward_endpoints_mm(deployment, rotation)
        evaluations = 1
        iterations = 0
        initial_position_error = float(
            np.linalg.norm(target - endpoints[primary_tube])
        )
        termination_reason = "max-iterations"

        for iteration in range(self.max_iterations):
            endpoint_errors = task_targets - endpoints[list(task_tubes)]
            primary_error = float(np.linalg.norm(endpoint_errors[0]))
            locked_errors = np.linalg.norm(endpoint_errors[1:], axis=1)
            locks_satisfied = bool(
                not len(locked_errors)
                or all(
                    error <= lock_tolerances[tube]
                    for tube, error in zip(sorted(locks), locked_errors)
                )
            )
            if primary_error <= self.solver_tolerance_mm and locks_satisfied:
                termination_reason = (
                    "initial-tolerance-met" if iteration == 0 else "tolerance-met"
                )
                break
            weighted_error = (endpoint_errors * task_weights[:, None]).reshape(-1)
            error_norm = float(np.linalg.norm(weighted_error))

            jacobian = np.empty((3 * len(task_tubes), 6), dtype=float)
            for column in range(6):
                step = self.finite_difference_step
                if column < 3 and state[column] + step > 1.0:
                    step = -step
                perturbed = state.copy()
                perturbed[column] += step
                perturbed[:3] = np.clip(perturbed[:3], 0.0, 1.0)
                trial_deployment, trial_rotation = self._decode_state(perturbed)
                trial_endpoints = self.forward_endpoints_mm(
                    trial_deployment,
                    trial_rotation,
                )
                evaluations += 1
                task_change = (
                    trial_endpoints[list(task_tubes)]
                    - endpoints[list(task_tubes)]
                )
                jacobian[:, column] = (
                    task_change * task_weights[:, None]
                ).reshape(-1) / step

            normal_matrix = (
                jacobian.T @ jacobian
                + self.damping_mm**2 * np.identity(6)
            )
            try:
                state_change = np.linalg.solve(
                    normal_matrix,
                    jacobian.T @ weighted_error,
                )
            except np.linalg.LinAlgError:
                termination_reason = "linear-solve-failed"
                iterations = iteration + 1
                break
            state_change = np.clip(
                state_change,
                -self.max_normalised_step,
                self.max_normalised_step,
            )
            if not np.all(np.isfinite(state_change)):
                termination_reason = "nonfinite-step"
                iterations = iteration + 1
                break

            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = state + scale * state_change
                candidate[:3] = np.clip(candidate[:3], 0.0, 1.0)
                candidate_deployment, candidate_rotation = self._decode_state(candidate)
                candidate_endpoints = self.forward_endpoints_mm(
                    candidate_deployment, candidate_rotation
                )
                evaluations += 1
                candidate_error = (
                    task_targets - candidate_endpoints[list(task_tubes)]
                ) * task_weights[:, None]
                if np.linalg.norm(candidate_error) < error_norm - 1e-9:
                    state = candidate
                    endpoints = candidate_endpoints
                    accepted = True
                    break

            iterations = iteration + 1
            if not accepted:
                termination_reason = "no-improving-step"
                break
        else:
            termination_reason = "max-iterations"

        solved_deployment, solved_rotation = self._decode_state(state)
        solved_rotation = wrap_angles(solved_rotation)
        # Re-evaluate after wrapping so all reported coordinates describe the
        # returned actuator state exactly.
        endpoints = self.forward_endpoints_mm(solved_deployment, solved_rotation)
        evaluations += 1
        endpoint_error_values = np.full(3, np.nan, dtype=float)
        endpoint_error_values[primary_tube] = np.linalg.norm(
            target - endpoints[primary_tube]
        )
        for tube, locked_target in locks.items():
            endpoint_error_values[tube] = np.linalg.norm(
                locked_target - endpoints[tube]
            )
        position_error = float(endpoint_error_values[primary_tube])
        locks_satisfied = all(
            endpoint_error_values[tube] <= lock_tolerances[tube] for tube in locks
        )
        primary_tolerance_met = position_error <= self.solver_tolerance_mm
        solver_converged = primary_tolerance_met and locks_satisfied
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        message = (
            "Target reached with endpoint locks preserved"
            if solver_converged and locks
            else "Target reached"
            if solver_converged
            else f"Nearest constrained position ({termination_reason})"
        )
        if termination_reason not in IK_TERMINATION_REASONS:
            raise RuntimeError(f"unknown IK termination reason: {termination_reason}")
        return IKResult(
            deployment_m=solved_deployment,
            rotation_rad=solved_rotation,
            achieved_tip_mm=endpoints[primary_tube].copy(),
            target_tip_mm=target.copy(),
            position_error_mm=position_error,
            solver_converged=solver_converged,
            iterations=iterations,
            evaluations=evaluations,
            solve_time_ms=elapsed_ms,
            message=message,
            target_tube=primary_tube,
            endpoint_positions_mm=endpoints.copy(),
            endpoint_errors_mm=endpoint_error_values,
            termination_reason=termination_reason,
            solver_tolerance_mm=self.solver_tolerance_mm,
            primary_tolerance_met=primary_tolerance_met,
            locks_satisfied=bool(locks_satisfied),
            initial_position_error_mm=initial_position_error,
        )
