"""Reproducible design-analysis methods for the CTR simulation.

The quantities in this module are predictions of the ideal kinematic model.
In particular, ``ik_*`` values are numerical inverse-kinematics residuals, not
errors measured from a physical robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import differential_evolution

from ctr_sampling import (
    CanonicalConfigurationBank,
    SymmetryDisplayExpansion,
    canonical_configuration_samples as _canonical_configuration_samples,
    decode_nested_deployments,
    revolve_points_for_display,
)
from ctr_inverse_kinematics import ConstrainedTipIK
from ctr_spatial_analysis import weighted_quantile
from ctr_workspace_map import radial_workspace_envelope
from tube_parameters import TUBE_DATA, build_supervisor_ctr_parameters


TUBE_KEYS = ("inner", "middle", "outer")
BASELINE_JACOBIAN_ACTUATOR_SCALE = np.asarray(
    [350.0, 170.0, 80.0, np.pi, np.pi, np.pi],
    dtype=float,
)


@dataclass(frozen=True)
class TubeDesign:
    """Physical inputs used by the current three-tube kinematic model."""

    total_length_mm: np.ndarray
    curved_length_mm: np.ndarray
    od_mm: np.ndarray
    id_mm: np.ndarray
    precurvature_per_m: np.ndarray
    youngs_modulus_gpa: np.ndarray

    def validated(self, minimum_clearance_mm: float = 0.01) -> "TubeDesign":
        arrays = (
            self.total_length_mm,
            self.curved_length_mm,
            self.od_mm,
            self.id_mm,
            self.precurvature_per_m,
            self.youngs_modulus_gpa,
        )
        if any(np.asarray(values).shape != (3,) for values in arrays):
            raise ValueError("Every TubeDesign field must contain three values.")
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("TubeDesign values must be finite.")
        if np.any(self.total_length_mm <= 0.0):
            raise ValueError("Tube lengths must be positive.")
        if not (
            self.total_length_mm[0]
            >= self.total_length_mm[1]
            >= self.total_length_mm[2]
        ):
            raise ValueError("Lengths must satisfy inner >= middle >= outer.")
        if np.any(self.curved_length_mm < 0.0) or np.any(
            self.curved_length_mm > self.total_length_mm
        ):
            raise ValueError("Curved lengths must lie within total lengths.")
        if np.any(self.id_mm < 0.0) or np.any(self.od_mm <= self.id_mm):
            raise ValueError("Every tube must have a positive wall thickness.")
        if self.id_mm[1] - self.od_mm[0] < minimum_clearance_mm:
            raise ValueError("Inner-to-middle clearance is too small.")
        if self.id_mm[2] - self.od_mm[1] < minimum_clearance_mm:
            raise ValueError("Middle-to-outer clearance is too small.")
        if np.any(self.precurvature_per_m < 0.0):
            raise ValueError("Pre-curvature cannot be negative in this study.")
        if np.any(self.youngs_modulus_gpa <= 0.0):
            raise ValueError("Young's modulus must be positive.")
        return self

    def to_parameters(self) -> dict:
        """Convert millimetre/GPa design data to the forward-model format."""
        self.validated()
        straight_mm = self.total_length_mm - self.curved_length_mm
        radii_m = [
            [self.id_mm[index] * 0.5e-3, self.od_mm[index] * 0.5e-3]
            for index in range(3)
        ]
        youngs_pa = self.youngs_modulus_gpa * 1e9
        return {
            "n_t": 3,
            "l_t": [
                [straight_mm[index] * 1e-3, self.curved_length_mm[index] * 1e-3]
                for index in range(3)
            ],
            "E": youngs_pa.tolist(),
            # Retained for compatibility. The current forward model does not
            # use G, so no shear-stiffness conclusion should be drawn from it.
            "G": (youngs_pa / 3.0).tolist(),
            "kappa_0": self.precurvature_per_m.tolist(),
            "r": radii_m,
        }

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "total_length_mm": self.total_length_mm.tolist(),
            "curved_length_mm": self.curved_length_mm.tolist(),
            "od_mm": self.od_mm.tolist(),
            "id_mm": self.id_mm.tolist(),
            "precurvature_per_m": self.precurvature_per_m.tolist(),
            "youngs_modulus_gpa": self.youngs_modulus_gpa.tolist(),
        }


@dataclass(frozen=True)
class DesignVariable:
    name: str
    label: str
    unit: str
    lower: float
    baseline: float
    upper: float


@dataclass(frozen=True)
class DexterityMap:
    positions_mm: np.ndarray
    isotropy: np.ndarray
    singular_values: np.ndarray
    deployment_m: np.ndarray
    rotation_rad: np.ndarray
    canonical_sample_ids: np.ndarray | None = None
    jacobian_valid: np.ndarray | None = None
    jacobian_actuator_scale: np.ndarray | None = None


@dataclass(frozen=True)
class PhysicalJacobianResult:
    """All endpoint Jacobians in physical and fixed scaled coordinates.

    ``raw_jacobians`` differentiates endpoint millimetres with respect to
    deployment millimetres and tube radians. ``scaled_jacobians`` then applies
    one immutable baseline-derived input scale so candidate designs are
    compared in the same actuator coordinate system.
    """

    base_endpoints_mm: np.ndarray
    raw_jacobians: np.ndarray
    scaled_jacobians: np.ndarray
    actuator_scale: np.ndarray
    effective_steps: np.ndarray
    valid: bool
    reason: str


@dataclass(frozen=True)
class VoxelDexterityMap:
    """Dexterity aggregated over configurations reaching the same XYZ voxel."""

    centres_mm: np.ndarray
    maximum_isotropy: np.ndarray
    median_isotropy: np.ndarray
    sample_count: np.ndarray
    voxel_size_mm: float
    minimum_median_samples: int


@dataclass(frozen=True)
class CylindricalDexterityMap:
    """Dexterity aggregated in radial-distance versus Z-height bins."""

    radial_centres_mm: np.ndarray
    z_centres_mm: np.ndarray
    maximum_isotropy: np.ndarray
    median_isotropy: np.ndarray
    sample_count: np.ndarray
    radial_bin_mm: float
    z_bin_mm: float
    minimum_median_samples: int


@dataclass(frozen=True)
class IKErrorMap:
    targets_mm: np.ndarray
    achieved_mm: np.ndarray
    residual_mm: np.ndarray
    reached: np.ndarray
    iterations: np.ndarray
    evaluations: np.ndarray
    solve_time_ms: np.ndarray
    canonical_sample_ids: np.ndarray | None = None
    termination_reason: np.ndarray | None = None
    solver_tolerance_mm: float = 0.15
    initial_error_mm: np.ndarray | None = None
    starts_attempted: np.ndarray | None = None
    converged_start_count: np.ndarray | None = None
    target_weight_mm3: np.ndarray | None = None

    @property
    def solver_converged(self) -> np.ndarray:
        """Explicit name for the legacy ``reached`` storage field."""
        return np.asarray(self.reached, dtype=bool)


@dataclass(frozen=True)
class CanonicalTargetSet:
    """Reachable legacy targets with retained canonical source provenance."""

    targets_mm: np.ndarray
    canonical_sample_ids: np.ndarray
    source_deployment_m: np.ndarray
    source_rotation_rad: np.ndarray
    split: str
    seed: int


@dataclass(frozen=True)
class CanonicalisedTipStates:
    """Tip states after removing their redundant common Z-axis rotation."""

    positions_mm: np.ndarray
    rotation_rad: np.ndarray
    removed_azimuth_rad: np.ndarray
    canonical_sample_ids: np.ndarray


@dataclass(frozen=True)
class VoxelResidualMap:
    """Numerical IK residuals aggregated over target-position voxels."""

    centres_mm: np.ndarray
    maximum_residual_mm: np.ndarray
    median_residual_mm: np.ndarray
    sample_count: np.ndarray
    voxel_size_mm: float
    minimum_median_samples: int


@dataclass(frozen=True)
class DesignEvaluation:
    design: TubeDesign
    metrics: dict[str, float]
    workspace_tips_mm: np.ndarray
    dexterity: DexterityMap | None
    ik_errors: IKErrorMap | None


@dataclass(frozen=True)
class OptimisationResult:
    design: TubeDesign
    variables: tuple[str, ...]
    values: np.ndarray
    objective: float
    evaluation: DesignEvaluation
    success: bool
    message: str
    function_evaluations: int


def baseline_design() -> TubeDesign:
    """Return the supervisor-provided baseline in explicit physical units."""
    parameters = build_supervisor_ctr_parameters()
    return TubeDesign(
        total_length_mm=np.asarray(
            [TUBE_DATA[key]["total_length_mm"] for key in TUBE_KEYS], dtype=float
        ),
        curved_length_mm=np.asarray(
            [TUBE_DATA[key]["curved_length_mm"] for key in TUBE_KEYS], dtype=float
        ),
        od_mm=np.asarray([TUBE_DATA[key]["od_mm"] for key in TUBE_KEYS], dtype=float),
        id_mm=np.asarray([TUBE_DATA[key]["id_mm"] for key in TUBE_KEYS], dtype=float),
        precurvature_per_m=np.asarray(parameters["kappa_0"], dtype=float),
        youngs_modulus_gpa=np.asarray(parameters["E"], dtype=float) / 1e9,
    ).validated()


def design_variables(
    design: TubeDesign | None = None,
    *,
    dimension_span: float = 0.10,
    curvature_span: float = 0.20,
) -> tuple[DesignVariable, ...]:
    """Return exploratory geometry/curvature bounds, not manufacturing limits.

    Young's modulus is intentionally excluded: protocol v1 fixes all tubes at
    75 GPa and reserves the supplied ranges for later uncertainty analysis.
    """
    base = baseline_design() if design is None else design.validated()
    variables: list[DesignVariable] = []

    def relative(name: str, label: str, unit: str, value: float, span: float) -> None:
        variables.append(
            DesignVariable(name, label, unit, value * (1.0 - span), value, value * (1.0 + span))
        )

    for index, tube in enumerate(TUBE_KEYS):
        relative(
            f"{tube}_total_length_mm",
            f"{tube.title()} total length",
            "mm",
            float(base.total_length_mm[index]),
            dimension_span,
        )
    for index, tube in ((1, "middle"), (2, "outer")):
        relative(
            f"{tube}_curved_length_mm",
            f"{tube.title()} curved length",
            "mm",
            float(base.curved_length_mm[index]),
            dimension_span,
        )
    for index, tube in enumerate(TUBE_KEYS):
        relative(
            f"{tube}_od_mm",
            f"{tube.title()} outer diameter",
            "mm",
            float(base.od_mm[index]),
            dimension_span,
        )
    for index, tube in ((1, "middle"), (2, "outer")):
        relative(
            f"{tube}_id_mm",
            f"{tube.title()} inner diameter",
            "mm",
            float(base.id_mm[index]),
            dimension_span,
        )
        relative(
            f"{tube}_precurvature_per_m",
            f"{tube.title()} pre-curvature",
            "1/m",
            float(base.precurvature_per_m[index]),
            curvature_span,
        )
    return tuple(variables)


def _field_and_index(variable_name: str) -> tuple[str, int]:
    for index, tube in enumerate(TUBE_KEYS):
        prefix = f"{tube}_"
        if variable_name.startswith(prefix):
            return variable_name[len(prefix) :], index
    raise KeyError(f"Unknown design variable: {variable_name}")


def design_value(design: TubeDesign, variable_name: str) -> float:
    field, index = _field_and_index(variable_name)
    field_map = {
        "total_length_mm": design.total_length_mm,
        "curved_length_mm": design.curved_length_mm,
        "od_mm": design.od_mm,
        "id_mm": design.id_mm,
        "precurvature_per_m": design.precurvature_per_m,
        "youngs_modulus_gpa": design.youngs_modulus_gpa,
    }
    if field not in field_map:
        raise KeyError(f"Unknown design field: {field}")
    return float(field_map[field][index])


def with_design_value(
    design: TubeDesign,
    variable_name: str,
    value: float,
    *,
    validate: bool = True,
) -> TubeDesign:
    field, index = _field_and_index(variable_name)
    values = {
        "total_length_mm": design.total_length_mm.copy(),
        "curved_length_mm": design.curved_length_mm.copy(),
        "od_mm": design.od_mm.copy(),
        "id_mm": design.id_mm.copy(),
        "precurvature_per_m": design.precurvature_per_m.copy(),
        "youngs_modulus_gpa": design.youngs_modulus_gpa.copy(),
    }
    if field not in values:
        raise KeyError(f"Unknown design field: {field}")
    values[field][index] = float(value)
    candidate = TubeDesign(**values)
    return candidate.validated() if validate else candidate


def _decode_fraction_samples(
    fractions: np.ndarray,
    total_lengths_m: np.ndarray,
) -> np.ndarray:
    return decode_nested_deployments(fractions, total_lengths_m)


def canonical_configuration_samples(
    total_lengths_m: np.ndarray,
    sample_count: int,
    *,
    seed: int,
    split: str,
) -> tuple[CanonicalConfigurationBank, np.ndarray, np.ndarray]:
    """Return protocol-v1 five-DOF samples and decoded actuator arrays."""
    return _canonical_configuration_samples(
        total_lengths_m,
        sample_count,
        seed=seed,
        split=split,
    )


def shared_configuration_samples(
    total_lengths_m: np.ndarray,
    sample_count: int,
    seed: int,
    *,
    split: str = "legacy-training",
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around protocol-v1 canonical sampling.

    New formal code should retain the returned configuration bank and its
    canonical IDs by calling :func:`canonical_configuration_samples` directly.
    """
    _, deployment, rotation = canonical_configuration_samples(
        total_lengths_m,
        sample_count,
        seed=seed,
        split=split,
    )
    return deployment, rotation


def rotate_about_z(points: np.ndarray, angles_rad: np.ndarray | float) -> np.ndarray:
    """Rotate one or more XYZ points about the robot's Z axis."""
    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, 3)
        squeeze = True
    else:
        squeeze = False
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (3,) or (N, 3)")
    angles = np.broadcast_to(np.asarray(angles_rad, dtype=float), (len(values),))
    cosine = np.cos(angles)
    sine = np.sin(angles)
    rotated = values.copy()
    rotated[:, 0] = cosine * values[:, 0] - sine * values[:, 1]
    rotated[:, 1] = sine * values[:, 0] + cosine * values[:, 1]
    return rotated[0] if squeeze else rotated


def canonicalise_tip_states_about_z(
    positions_mm: np.ndarray,
    rotation_rad: np.ndarray,
    canonical_sample_ids: np.ndarray | None = None,
) -> CanonicalisedTipStates:
    """Move every tip exactly onto the positive-X/Z canonical half-plane.

    The forward-model actuator convention means subtracting a common actuator
    angle rotates the physical shape by the positive of that angle.  Removing a
    tip azimuth from Cartesian space therefore adds that azimuth to all three
    tube rotations.  Relative tube rotations are unchanged.
    """
    positions = np.asarray(positions_mm, dtype=float)
    rotations = np.asarray(rotation_rad, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_mm must have shape (N, 3)")
    if rotations.shape != positions.shape:
        raise ValueError("rotation_rad must have shape (N, 3)")
    identifiers = _canonical_ids_or_rows(canonical_sample_ids, len(positions))
    azimuth = np.arctan2(positions[:, 1], positions[:, 0])
    radius = np.hypot(positions[:, 0], positions[:, 1])
    azimuth = np.where(radius > 1e-12, azimuth, 0.0)
    canonical_positions = rotate_about_z(positions, -azimuth)
    canonical_positions[:, 0] = radius
    canonical_positions[:, 1] = 0.0
    canonical_rotations = (
        rotations + azimuth[:, None] + np.pi
    ) % (2.0 * np.pi) - np.pi
    return CanonicalisedTipStates(
        positions_mm=canonical_positions.astype(np.float32),
        rotation_rad=canonical_rotations.astype(np.float32),
        removed_azimuth_rad=azimuth.astype(np.float32),
        canonical_sample_ids=identifiers.copy(),
    )


def revolve_canonical_points_for_display(
    canonical_points_mm: np.ndarray,
    canonical_sample_ids: np.ndarray,
    angles_rad: np.ndarray,
) -> SymmetryDisplayExpansion:
    """Create render-only angular copies while preserving source provenance."""
    return revolve_points_for_display(
        canonical_points_mm,
        canonical_sample_ids,
        angles_rad,
    )


def fold_configurations_to_first_quadrant(
    positions_mm: np.ndarray,
    rotation_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy display helper that moves every tip into quadrant 1.

    The inherited tube-angle convention rotates the physical shape by the
    negative of a common actuator-angle offset.  Applying the opposite offset
    to every tube preserves relative rotations and the ideal CTR shape.
    """
    positions = np.asarray(positions_mm, dtype=float)
    rotations = np.asarray(rotation_rad, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_mm must have shape (N, 3)")
    if rotations.shape != positions.shape:
        raise ValueError("rotation_rad must have shape (N, 3)")
    azimuth = np.mod(np.arctan2(positions[:, 1], positions[:, 0]), 2.0 * np.pi)
    quadrant = np.floor(azimuth / (0.5 * np.pi)).astype(np.int8)
    offsets = -quadrant.astype(float) * (0.5 * np.pi)
    folded_positions = rotate_about_z(positions, offsets)
    folded_rotations = (rotations - offsets[:, None] + np.pi) % (2.0 * np.pi) - np.pi
    tolerance = 2e-5
    folded_positions[np.abs(folded_positions) < tolerance] = 0.0
    if np.any(folded_positions[:, :2] < -tolerance):
        raise RuntimeError("Quadrant folding produced a negative X or Y coordinate")
    return folded_positions.astype(np.float32), folded_rotations.astype(np.float32), quadrant


def expand_first_quadrant_points(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Legacy display helper for four rotated copies.

    This function returns no provenance and must not be used for protocol-v1
    statistical counts. Use :func:`revolve_canonical_points_for_display` with
    canonical sample IDs for formal work.
    """
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_mm must have shape (N, 3)")
    quadrants = np.repeat(np.arange(4, dtype=np.int8), len(points))
    expanded = np.vstack(
        [rotate_about_z(points, quadrant * 0.5 * np.pi) for quadrant in range(4)]
    )
    expanded[np.abs(expanded) < 2e-5] = 0.0
    return expanded.astype(np.float32), quadrants


def _canonical_ids_or_rows(
    canonical_sample_ids: np.ndarray | None,
    row_count: int,
) -> np.ndarray:
    if canonical_sample_ids is None:
        return np.arange(row_count, dtype=np.int64)
    identifiers = np.asarray(canonical_sample_ids)
    if identifiers.shape != (row_count,):
        raise ValueError("canonical_sample_ids must have shape (N,)")
    return identifiers


def _independent_values_in_group(
    row_indices: np.ndarray,
    canonical_sample_ids: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Return one statistically independent value per canonical source ID."""
    selected_ids = canonical_sample_ids[row_indices]
    _, first, inverse = np.unique(
        selected_ids,
        return_index=True,
        return_inverse=True,
    )
    independent = values[row_indices[first]]
    for identifier_index, representative in enumerate(independent):
        repeated = values[row_indices[inverse == identifier_index]]
        if not np.allclose(repeated, representative, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "one canonical sample has conflicting repeated values in a spatial cell"
            )
    return independent


def position_jacobians_all_endpoints(
    solver: ConstrainedTipIK,
    deployment_m: np.ndarray,
    rotation_rad: np.ndarray,
    *,
    translation_step_mm: float = 0.1,
    rotation_step_rad: float = 1e-3,
    actuator_scale: np.ndarray = BASELINE_JACOBIAN_ACTUATOR_SCALE,
    minimum_two_sided_step_mm: float = 1e-8,
) -> PhysicalJacobianResult:
    """Calculate constraint-safe, central physical Jacobians for all tips.

    The physical actuator vector is ``[u0_mm, u1_mm, u2_mm, a0_rad,
    a1_rad, a2_rad]``. A six-dimensional Jacobian is only returned where a
    two-sided neighbourhood exists inside the nested deployment constraints;
    exact boundary states are deliberately marked invalid.
    """
    deployment = solver.validate_deployment(deployment_m)
    rotation = np.asarray(rotation_rad, dtype=float).reshape(3)
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_rad must contain three finite values")
    translation_step = float(translation_step_mm)
    rotation_step = float(rotation_step_rad)
    if translation_step <= 0.0 or rotation_step <= 0.0:
        raise ValueError("finite-difference steps must be positive")
    scale = np.asarray(actuator_scale, dtype=float).reshape(-1)
    if scale.shape != (6,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("actuator_scale must contain six finite positive values")

    base_endpoints = solver.forward_endpoints_mm(deployment, rotation)
    raw = np.full((3, 3, 6), np.nan, dtype=float)
    effective_steps = np.full(6, np.nan, dtype=float)
    lengths = solver.total_lengths
    inner, middle, outer = deployment
    deployment_margins_mm = 1000.0 * np.asarray(
        [
            min(inner - middle, lengths[0] - inner),
            min(middle - outer, inner - middle, lengths[1] - middle),
            min(outer, middle - outer, lengths[2] - outer),
        ],
        dtype=float,
    )
    for column, margin_mm in enumerate(deployment_margins_mm):
        step_mm = min(translation_step, 0.45 * max(0.0, margin_mm))
        if step_mm < float(minimum_two_sided_step_mm):
            return PhysicalJacobianResult(
                base_endpoints_mm=base_endpoints,
                raw_jacobians=raw,
                scaled_jacobians=raw.copy(),
                actuator_scale=scale.copy(),
                effective_steps=effective_steps,
                valid=False,
                reason="constraint-boundary",
            )
        plus_deployment = deployment.copy()
        minus_deployment = deployment.copy()
        plus_deployment[column] += step_mm * 1e-3
        minus_deployment[column] -= step_mm * 1e-3
        plus = solver.forward_endpoints_mm(plus_deployment, rotation)
        minus = solver.forward_endpoints_mm(minus_deployment, rotation)
        raw[:, :, column] = (plus - minus) / (2.0 * step_mm)
        effective_steps[column] = step_mm

    for rotation_column in range(3):
        plus_rotation = rotation.copy()
        minus_rotation = rotation.copy()
        plus_rotation[rotation_column] += rotation_step
        minus_rotation[rotation_column] -= rotation_step
        plus = solver.forward_endpoints_mm(deployment, plus_rotation)
        minus = solver.forward_endpoints_mm(deployment, minus_rotation)
        raw[:, :, rotation_column + 3] = (
            plus - minus
        ) / (2.0 * rotation_step)
        effective_steps[rotation_column + 3] = rotation_step

    scaled = raw * scale[None, None, :]
    return PhysicalJacobianResult(
        base_endpoints_mm=base_endpoints,
        raw_jacobians=raw,
        scaled_jacobians=scaled,
        actuator_scale=scale.copy(),
        effective_steps=effective_steps,
        valid=True,
        reason="valid-interior-state",
    )


def position_jacobian(
    solver: ConstrainedTipIK,
    deployment_m: np.ndarray,
    rotation_rad: np.ndarray,
    *,
    target_tube: int = 0,
    translation_step_mm: float = 0.1,
    rotation_step_rad: float = 1e-3,
    actuator_scale: np.ndarray = BASELINE_JACOBIAN_ACTUATOR_SCALE,
    scaled: bool = True,
) -> np.ndarray:
    """Return one endpoint's baseline-range-normalised positional Jacobian."""
    tube = int(target_tube)
    if tube not in (0, 1, 2):
        raise ValueError("target_tube must be 0, 1, or 2")
    result = position_jacobians_all_endpoints(
        solver,
        deployment_m,
        rotation_rad,
        translation_step_mm=translation_step_mm,
        rotation_step_rad=rotation_step_rad,
        actuator_scale=actuator_scale,
    )
    selected = result.scaled_jacobians if scaled else result.raw_jacobians
    return selected[tube].copy()


def positional_isotropy(jacobian: np.ndarray) -> tuple[float, np.ndarray]:
    """Return sigma_min/sigma_max and all three task-space singular values."""
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.shape != (3, 6):
        raise ValueError("a positional Jacobian must have shape (3, 6)")
    if not np.all(np.isfinite(matrix)):
        return float("nan"), np.full(3, np.nan, dtype=float)
    singular_values = np.linalg.svd(
        matrix, compute_uv=False
    )
    if singular_values.shape != (3,) or singular_values[0] <= 1e-12:
        return 0.0, np.zeros(3, dtype=float)
    return float(singular_values[-1] / singular_values[0]), singular_values


def calculate_dexterity_map(
    parameters: dict,
    deployment_m: np.ndarray,
    rotation_rad: np.ndarray,
    *,
    target_tube: int = 0,
    model_points_per_section: int = 2,
    canonical_sample_ids: np.ndarray | None = None,
    translation_step_mm: float = 0.1,
    rotation_step_rad: float = 1e-3,
    actuator_scale: np.ndarray = BASELINE_JACOBIAN_ACTUATOR_SCALE,
) -> DexterityMap:
    deployment = np.asarray(deployment_m, dtype=float)
    rotation = np.asarray(rotation_rad, dtype=float)
    if deployment.ndim != 2 or deployment.shape[1] != 3 or rotation.shape != deployment.shape:
        raise ValueError("deployment_m and rotation_rad must both have shape (N, 3)")
    identifiers = _canonical_ids_or_rows(canonical_sample_ids, len(deployment))
    solver = ConstrainedTipIK(
        parameters, model_points_per_section=model_points_per_section
    )
    positions = np.empty_like(deployment, dtype=np.float32)
    isotropy = np.empty(len(deployment), dtype=np.float32)
    singular_values = np.empty_like(deployment, dtype=np.float32)
    jacobian_valid = np.empty(len(deployment), dtype=bool)
    for index in range(len(deployment)):
        jacobians = position_jacobians_all_endpoints(
            solver,
            deployment[index],
            rotation[index],
            translation_step_mm=translation_step_mm,
            rotation_step_rad=rotation_step_rad,
            actuator_scale=actuator_scale,
        )
        positions[index] = jacobians.base_endpoints_mm[target_tube]
        jacobian_valid[index] = jacobians.valid
        if jacobians.valid:
            score, values = positional_isotropy(
                jacobians.scaled_jacobians[target_tube]
            )
            isotropy[index] = score
            singular_values[index] = values
        else:
            isotropy[index] = np.nan
            singular_values[index] = np.nan
    return DexterityMap(
        positions_mm=positions,
        isotropy=isotropy,
        singular_values=singular_values,
        deployment_m=deployment.astype(np.float32),
        rotation_rad=rotation.astype(np.float32),
        canonical_sample_ids=identifiers.copy(),
        jacobian_valid=jacobian_valid,
        jacobian_actuator_scale=np.asarray(actuator_scale, dtype=float).copy(),
    )


def aggregate_dexterity_voxels(
    dexterity: DexterityMap,
    *,
    voxel_size_mm: float = 10.0,
    minimum_median_samples: int = 3,
) -> VoxelDexterityMap:
    """Aggregate redundant configurations into spatial dexterity summaries."""
    if voxel_size_mm <= 0.0:
        raise ValueError("voxel_size_mm must be positive")
    if minimum_median_samples < 1:
        raise ValueError("minimum_median_samples must be positive")
    points = np.asarray(dexterity.positions_mm, dtype=float)
    scores = np.asarray(dexterity.isotropy, dtype=float)
    identifiers = _canonical_ids_or_rows(
        dexterity.canonical_sample_ids,
        len(points),
    )
    if points.ndim != 2 or points.shape[1] != 3 or scores.shape != (len(points),):
        raise ValueError("dexterity map positions and scores are inconsistent")
    voxel_index = np.floor(points / float(voxel_size_mm)).astype(np.int64)
    unique, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    centres = (unique.astype(float) + 0.5) * float(voxel_size_mm)
    maximum = np.empty(len(unique), dtype=np.float32)
    median = np.empty(len(unique), dtype=np.float32)
    counts = np.empty(len(unique), dtype=np.int32)
    for index in range(len(unique)):
        row_indices = np.flatnonzero(inverse == index)
        values = _independent_values_in_group(row_indices, identifiers, scores)
        maximum[index] = np.max(values)
        counts[index] = len(values)
        median[index] = (
            np.median(values)
            if counts[index] >= minimum_median_samples
            else np.nan
        )
    return VoxelDexterityMap(
        centres_mm=centres.astype(np.float32),
        maximum_isotropy=maximum,
        median_isotropy=median,
        sample_count=counts,
        voxel_size_mm=float(voxel_size_mm),
        minimum_median_samples=int(minimum_median_samples),
    )


def aggregate_ik_residual_voxels(
    errors: IKErrorMap,
    *,
    voxel_size_mm: float = 10.0,
    minimum_median_samples: int = 3,
) -> VoxelResidualMap:
    """Aggregate numerical IK residuals by requested target-position voxel."""
    if voxel_size_mm <= 0.0:
        raise ValueError("voxel_size_mm must be positive")
    if minimum_median_samples < 1:
        raise ValueError("minimum_median_samples must be positive")
    points = np.asarray(errors.targets_mm, dtype=float)
    residual = np.asarray(errors.residual_mm, dtype=float)
    identifiers = _canonical_ids_or_rows(
        errors.canonical_sample_ids,
        len(points),
    )
    if points.ndim != 2 or points.shape[1] != 3 or residual.shape != (len(points),):
        raise ValueError("IK targets and residuals are inconsistent")
    voxel_index = np.floor(points / float(voxel_size_mm)).astype(np.int64)
    unique, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    centres = (unique.astype(float) + 0.5) * float(voxel_size_mm)
    maximum = np.empty(len(unique), dtype=np.float32)
    median = np.empty(len(unique), dtype=np.float32)
    counts = np.empty(len(unique), dtype=np.int32)
    for index in range(len(unique)):
        row_indices = np.flatnonzero(inverse == index)
        values = _independent_values_in_group(row_indices, identifiers, residual)
        counts[index] = len(values)
        maximum[index] = np.max(values)
        median[index] = (
            np.median(values)
            if counts[index] >= minimum_median_samples
            else np.nan
        )
    return VoxelResidualMap(
        centres_mm=centres.astype(np.float32),
        maximum_residual_mm=maximum,
        median_residual_mm=median,
        sample_count=counts,
        voxel_size_mm=float(voxel_size_mm),
        minimum_median_samples=int(minimum_median_samples),
    )


def aggregate_cylindrical_dexterity(
    dexterity: DexterityMap,
    *,
    radial_bin_mm: float = 10.0,
    z_bin_mm: float = 10.0,
    minimum_median_samples: int = 3,
) -> CylindricalDexterityMap:
    """Aggregate all azimuths into an occupancy-aware radius/Z heat map."""
    if radial_bin_mm <= 0.0 or z_bin_mm <= 0.0:
        raise ValueError("cylindrical bin sizes must be positive")
    if minimum_median_samples < 1:
        raise ValueError("minimum_median_samples must be positive")
    points = np.asarray(dexterity.positions_mm, dtype=float)
    scores = np.asarray(dexterity.isotropy, dtype=float)
    identifiers = _canonical_ids_or_rows(
        dexterity.canonical_sample_ids,
        len(points),
    )
    radius = np.hypot(points[:, 0], points[:, 1])
    radial_index = np.floor(radius / float(radial_bin_mm)).astype(np.int64)
    z_index = np.floor(points[:, 2] / float(z_bin_mm)).astype(np.int64)
    radial_min, radial_max = int(np.min(radial_index)), int(np.max(radial_index))
    z_min, z_max = int(np.min(z_index)), int(np.max(z_index))
    shape = (radial_max - radial_min + 1, z_max - z_min + 1)
    maximum = np.full(shape, np.nan, dtype=np.float32)
    median = np.full(shape, np.nan, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.int32)
    for radial_value, z_value in np.unique(
        np.column_stack((radial_index, z_index)), axis=0
    ):
        selected = (radial_index == radial_value) & (z_index == z_value)
        row_indices = np.flatnonzero(selected)
        values = _independent_values_in_group(row_indices, identifiers, scores)
        row = int(radial_value - radial_min)
        column = int(z_value - z_min)
        counts[row, column] = len(values)
        maximum[row, column] = np.max(values)
        if len(values) >= minimum_median_samples:
            median[row, column] = np.median(values)
    radial_centres = (
        np.arange(radial_min, radial_max + 1, dtype=float) + 0.5
    ) * float(radial_bin_mm)
    z_centres = (
        np.arange(z_min, z_max + 1, dtype=float) + 0.5
    ) * float(z_bin_mm)
    return CylindricalDexterityMap(
        radial_centres_mm=radial_centres.astype(np.float32),
        z_centres_mm=z_centres.astype(np.float32),
        maximum_isotropy=maximum,
        median_isotropy=median,
        sample_count=counts,
        radial_bin_mm=float(radial_bin_mm),
        z_bin_mm=float(z_bin_mm),
        minimum_median_samples=int(minimum_median_samples),
    )


def _resolve_solver_tolerance(
    solver_tolerance_mm: float | None,
    legacy_tolerance_mm: float | None,
) -> float:
    """Resolve the explicit Stage-2 name while retaining legacy callers."""
    if solver_tolerance_mm is not None and legacy_tolerance_mm is not None:
        if not np.isclose(
            float(solver_tolerance_mm),
            float(legacy_tolerance_mm),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "solver_tolerance_mm and tolerance_mm cannot disagree"
            )
    value = (
        0.15
        if solver_tolerance_mm is None and legacy_tolerance_mm is None
        else float(
            solver_tolerance_mm
            if solver_tolerance_mm is not None
            else legacy_tolerance_mm
        )
    )
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError("solver tolerance must be finite and positive")
    return value


def _validated_target_weights(
    values: np.ndarray | None,
    target_count: int,
) -> np.ndarray | None:
    if values is None:
        return None
    weights = np.asarray(values, dtype=float).reshape(-1)
    if weights.shape != (target_count,):
        raise ValueError("target_weight_mm3 must have shape (N,)")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("target weights must be finite and positive")
    return weights.copy()


def calculate_ik_error_map(
    parameters: dict,
    targets_mm: np.ndarray,
    *,
    target_tube: int = 0,
    solver_tolerance_mm: float | None = None,
    tolerance_mm: float | None = None,
    max_iterations: int = 12,
    align_retracted_rotation: bool = False,
    canonical_sample_ids: np.ndarray | None = None,
    target_weight_mm3: np.ndarray | None = None,
) -> IKErrorMap:
    """Solve targets locally from a fully retracted actuator state.

    With every tube fully retracted, a common tube rotation does not change the
    Cartesian tip position. It does, however, select the bending plane that
    appears when the tubes begin to deploy. Aligning that physically equivalent
    retracted state with each target avoids privileging the global XZ plane and
    makes the numerical test rotationally equivariant. The default instead
    uses one fixed zero-rotation actuator configuration for every
    target. ``align_retracted_rotation=True`` is retained for controlled
    diagnostic comparisons, but it must not be described as a single fixed
    actuator configuration because it changes the common starting rotation for
    every target.
    """
    targets = np.asarray(targets_mm, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.all(np.isfinite(targets)):
        raise ValueError("targets_mm must have shape (N, 3) and be finite")
    identifiers = _canonical_ids_or_rows(canonical_sample_ids, len(targets))
    target_weights = _validated_target_weights(target_weight_mm3, len(targets))
    selected_tolerance = _resolve_solver_tolerance(
        solver_tolerance_mm,
        tolerance_mm,
    )
    solver = ConstrainedTipIK(
        parameters,
        solver_tolerance_mm=selected_tolerance,
        max_iterations=max_iterations,
        damping_mm=1.0,
        max_normalised_step=0.10,
    )
    achieved = np.empty_like(targets, dtype=np.float32)
    residual = np.empty(len(targets), dtype=np.float32)
    reached = np.empty(len(targets), dtype=bool)
    iterations = np.empty(len(targets), dtype=np.int16)
    evaluations = np.empty(len(targets), dtype=np.int16)
    solve_time = np.empty(len(targets), dtype=np.float32)
    termination_reason = np.empty(len(targets), dtype="U32")
    initial_error = np.empty(len(targets), dtype=np.float32)
    zero_deployment = np.zeros(3, dtype=float)
    for index, target in enumerate(targets):
        initial_rotation = np.zeros(3, dtype=float)
        if align_retracted_rotation:
            target_azimuth = np.arctan2(target[1], target[0])
            # This actuator convention rotates the physical shape by the
            # negative of a common tube-angle offset.
            initial_rotation.fill(-target_azimuth)
        result = solver.solve(
            target,
            zero_deployment,
            initial_rotation,
            target_tube=target_tube,
        )
        achieved[index] = result.achieved_tip_mm
        residual[index] = result.position_error_mm
        reached[index] = result.reached
        iterations[index] = result.iterations
        evaluations[index] = result.evaluations
        solve_time[index] = result.solve_time_ms
        termination_reason[index] = result.termination_reason
        initial_error[index] = result.initial_position_error_mm
    return IKErrorMap(
        targets_mm=targets.astype(np.float32),
        achieved_mm=achieved,
        residual_mm=residual,
        reached=reached,
        iterations=iterations,
        evaluations=evaluations,
        solve_time_ms=solve_time,
        canonical_sample_ids=identifiers.copy(),
        termination_reason=termination_reason,
        solver_tolerance_mm=selected_tolerance,
        initial_error_mm=initial_error,
        starts_attempted=np.ones(len(targets), dtype=np.int16),
        converged_start_count=reached.astype(np.int16),
        target_weight_mm3=target_weights,
    )


def calculate_multistart_ik_error_map(
    parameters: dict,
    targets_mm: np.ndarray,
    *,
    target_tube: int = 0,
    common_rotation_seeds_rad: np.ndarray | None = None,
    solver_tolerance_mm: float | None = None,
    tolerance_mm: float | None = None,
    max_iterations: int = 12,
    canonical_sample_ids: np.ndarray | None = None,
    target_weight_mm3: np.ndarray | None = None,
) -> IKErrorMap:
    """Return the best local IK result from a fixed bank of retracted seeds.

    Every seed has zero tube deployment and a common rotation applied to all
    three tubes. The seed bank is identical for every target. Reported solve
    time and function evaluations include all attempted seeds, while the
    achieved position and iteration count belong to the minimum-residual seed.
    """
    targets = np.asarray(targets_mm, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.all(np.isfinite(targets)):
        raise ValueError("targets_mm must have shape (N, 3) and be finite")
    identifiers = _canonical_ids_or_rows(canonical_sample_ids, len(targets))
    target_weights = _validated_target_weights(target_weight_mm3, len(targets))
    selected_tolerance = _resolve_solver_tolerance(
        solver_tolerance_mm,
        tolerance_mm,
    )
    if common_rotation_seeds_rad is None:
        common_rotation_seeds_rad = np.linspace(
            -np.pi, np.pi, 8, endpoint=False, dtype=float
        )
    seeds = np.asarray(common_rotation_seeds_rad, dtype=float).reshape(-1)
    if not len(seeds) or not np.all(np.isfinite(seeds)):
        raise ValueError("common_rotation_seeds_rad must contain finite values")
    solver = ConstrainedTipIK(
        parameters,
        solver_tolerance_mm=selected_tolerance,
        max_iterations=max_iterations,
        damping_mm=1.0,
        max_normalised_step=0.10,
    )
    achieved = np.empty_like(targets, dtype=np.float32)
    residual = np.empty(len(targets), dtype=np.float32)
    reached = np.empty(len(targets), dtype=bool)
    iterations = np.empty(len(targets), dtype=np.int16)
    evaluations = np.empty(len(targets), dtype=np.int32)
    solve_time = np.empty(len(targets), dtype=np.float32)
    termination_reason = np.empty(len(targets), dtype="U32")
    initial_error = np.empty(len(targets), dtype=np.float32)
    starts_attempted = np.full(len(targets), len(seeds), dtype=np.int16)
    converged_start_count = np.empty(len(targets), dtype=np.int16)
    zero_deployment = np.zeros(3, dtype=float)
    for index, target in enumerate(targets):
        best = None
        total_evaluations = 0
        total_time_ms = 0.0
        converged_count = 0
        for seed in seeds:
            result = solver.solve(
                target,
                zero_deployment,
                np.full(3, seed, dtype=float),
                target_tube=target_tube,
            )
            total_evaluations += result.evaluations
            total_time_ms += result.solve_time_ms
            converged_count += int(result.solver_converged)
            if best is None or result.position_error_mm < best.position_error_mm:
                best = result
        achieved[index] = best.achieved_tip_mm
        residual[index] = best.position_error_mm
        reached[index] = best.reached
        iterations[index] = best.iterations
        evaluations[index] = total_evaluations
        solve_time[index] = total_time_ms
        termination_reason[index] = best.termination_reason
        initial_error[index] = best.initial_position_error_mm
        converged_start_count[index] = converged_count
    return IKErrorMap(
        targets_mm=targets.astype(np.float32),
        achieved_mm=achieved,
        residual_mm=residual,
        reached=reached,
        iterations=iterations,
        evaluations=evaluations,
        solve_time_ms=solve_time,
        canonical_sample_ids=identifiers.copy(),
        termination_reason=termination_reason,
        solver_tolerance_mm=selected_tolerance,
        initial_error_mm=initial_error,
        starts_attempted=starts_attempted,
        converged_start_count=converged_start_count,
        target_weight_mm3=target_weights,
    )


def workspace_size_metrics(tips_mm: np.ndarray) -> tuple[float, float]:
    """Return axisymmetric envelope volume (mm^3) and area (mm^2)."""
    tips = np.asarray(tips_mm, dtype=float)
    height_sections = max(8, min(72, len(tips) // 4))
    profile_z, profile_radius = radial_workspace_envelope(
        tips,
        height_sections=height_sections,
        boundary_quantile=0.995,
        smoothing_passes=3,
    )
    volume = float(np.trapezoid(np.pi * profile_radius**2, profile_z))
    dz = np.diff(profile_z)
    dr = np.diff(profile_radius)
    slant = np.hypot(dz, dr)
    area = float(
        np.sum(np.pi * (profile_radius[:-1] + profile_radius[1:]) * slant)
    )
    return volume, area


def _workspace_tips(
    parameters: dict,
    deployment_m: np.ndarray,
    rotation_rad: np.ndarray,
    target_tube: int,
) -> np.ndarray:
    solver = ConstrainedTipIK(parameters, model_points_per_section=2)
    tips = np.empty((len(deployment_m), 3), dtype=np.float32)
    for index in range(len(tips)):
        tips[index] = solver.forward_endpoint_mm(
            deployment_m[index], rotation_rad[index], target_tube
        )
    return tips


def summarise_dexterity(dexterity: DexterityMap | None) -> dict[str, float]:
    if dexterity is None or not len(dexterity.isotropy):
        return {
            "dexterity_mean": float("nan"),
            "dexterity_median": float("nan"),
            "dexterity_p10": float("nan"),
            "dexterity_std": float("nan"),
            "dexterity_cv": float("nan"),
        }
    values = np.asarray(dexterity.isotropy, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "dexterity_mean": float("nan"),
            "dexterity_median": float("nan"),
            "dexterity_p10": float("nan"),
            "dexterity_std": float("nan"),
            "dexterity_cv": float("nan"),
        }
    mean = float(np.mean(values))
    return {
        "dexterity_mean": mean,
        "dexterity_median": float(np.median(values)),
        "dexterity_p10": float(np.quantile(values, 0.10)),
        "dexterity_std": float(np.std(values)),
        "dexterity_cv": float(np.std(values) / mean) if mean > 1e-12 else float("inf"),
    }


def within_evaluation_threshold(
    errors: IKErrorMap,
    threshold_mm: float = 0.5,
) -> np.ndarray:
    """Classify task accuracy without changing or rerunning the IK solver."""
    threshold = float(threshold_mm)
    if threshold <= 0.0 or not np.isfinite(threshold):
        raise ValueError("evaluation threshold must be finite and positive")
    return np.asarray(errors.residual_mm, dtype=float) <= threshold


def _independent_ik_rows(errors: IKErrorMap) -> np.ndarray:
    residual = np.asarray(errors.residual_mm, dtype=float).reshape(-1)
    identifiers = _canonical_ids_or_rows(errors.canonical_sample_ids, len(residual))
    _, first, inverse, counts = np.unique(
        identifiers,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    if len(first) == len(identifiers):
        return np.arange(len(identifiers), dtype=np.int64)
    reached = np.asarray(errors.solver_converged, dtype=bool)
    reasons = (
        None
        if errors.termination_reason is None
        else np.asarray(errors.termination_reason).astype(str)
    )
    weights = (
        None
        if errors.target_weight_mm3 is None
        else np.asarray(errors.target_weight_mm3, dtype=float)
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    for representative_row, start, count in zip(first, starts, counts):
        rows = order[start : start + count]
        if not np.allclose(
            residual[rows], residual[representative_row], rtol=1e-6, atol=1e-8
        ):
            raise ValueError(
                "one canonical target has conflicting repeated IK residuals"
            )
        if np.any(reached[rows] != reached[representative_row]):
            raise ValueError(
                "one canonical target has conflicting solver-convergence flags"
            )
        if reasons is not None and np.any(reasons[rows] != reasons[representative_row]):
            raise ValueError(
                "one canonical target has conflicting IK termination reasons"
            )
        if weights is not None and not np.allclose(
            weights[rows],
            weights[representative_row],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "one canonical target has conflicting repeated physical weights"
            )
    return np.sort(first)


def summarise_ik_errors(
    errors: IKErrorMap | None,
    *,
    evaluation_threshold_mm: float = 0.5,
) -> dict[str, float]:
    threshold = float(evaluation_threshold_mm)
    if threshold <= 0.0 or not np.isfinite(threshold):
        raise ValueError("evaluation threshold must be finite and positive")
    if errors is None or not len(errors.residual_mm):
        return {
            "ik_mean_error_mm": float("nan"),
            "ik_rmse_mm": float("nan"),
            "ik_median_error_mm": float("nan"),
            "ik_p95_error_mm": float("nan"),
            "ik_max_error_mm": float("nan"),
            "ik_solver_convergence_rate": float("nan"),
            "ik_evaluation_threshold_mm": threshold,
            "ik_within_evaluation_threshold_rate": float("nan"),
            "ik_weighted_target_volume_mm3": float("nan"),
            # Deprecated compatibility field: this remains solver convergence.
            "ik_success_rate": float("nan"),
            "ik_mean_solve_time_ms": float("nan"),
        }
    rows = _independent_ik_rows(errors)
    residual = np.asarray(errors.residual_mm, dtype=float)[rows]
    solver_converged = np.asarray(errors.solver_converged, dtype=bool)[rows]
    weights = (
        np.ones(len(rows), dtype=float)
        if errors.target_weight_mm3 is None
        else np.asarray(errors.target_weight_mm3, dtype=float)[rows]
    )
    summary = {
        "ik_mean_error_mm": float(np.average(residual, weights=weights)),
        "ik_rmse_mm": float(np.sqrt(np.average(residual**2, weights=weights))),
        "ik_median_error_mm": weighted_quantile(residual, weights, 0.50),
        "ik_p95_error_mm": weighted_quantile(residual, weights, 0.95),
        "ik_max_error_mm": float(np.max(residual)),
        "ik_solver_convergence_rate": float(
            np.average(solver_converged.astype(float), weights=weights)
        ),
        "ik_evaluation_threshold_mm": threshold,
        "ik_within_evaluation_threshold_rate": float(
            np.average((residual <= threshold).astype(float), weights=weights)
        ),
        "ik_weighted_target_volume_mm3": (
            float(np.sum(weights))
            if errors.target_weight_mm3 is not None
            else float("nan")
        ),
        # Deprecated compatibility alias used only by legacy exploratory tools.
        "ik_success_rate": float(
            np.average(solver_converged.astype(float), weights=weights)
        ),
        "ik_mean_solve_time_ms": float(
            np.mean(np.asarray(errors.solve_time_ms, dtype=float)[rows])
        ),
    }
    if errors.termination_reason is not None:
        reasons = np.asarray(errors.termination_reason).astype(str)[rows]
        for reason in sorted(set(reasons.tolist())):
            key = "ik_termination_count_" + reason.replace("-", "_")
            summary[key] = float(np.count_nonzero(reasons == reason))
    return summary


def evaluate_design(
    design: TubeDesign,
    *,
    workspace_sample_count: int,
    dexterity_sample_count: int,
    ik_targets_mm: np.ndarray | None,
    target_tube: int = 0,
    seed: int = 42,
) -> DesignEvaluation:
    """Evaluate one design with deterministic, comparable actuator samples."""
    design = design.validated()
    parameters = design.to_parameters()
    lengths_m = design.total_length_mm * 1e-3
    _, deployment, rotation = canonical_configuration_samples(
        lengths_m,
        workspace_sample_count,
        seed=seed,
        split="training-workspace",
    )
    workspace_tips = _workspace_tips(parameters, deployment, rotation, target_tube)
    volume_mm3, area_mm2 = workspace_size_metrics(workspace_tips)

    dexterity = None
    if dexterity_sample_count > 0:
        dexterity_bank, dex_deployment, dex_rotation = canonical_configuration_samples(
            lengths_m,
            dexterity_sample_count,
            seed=seed + 1,
            split="training-dexterity",
        )
        dexterity = calculate_dexterity_map(
            parameters,
            dex_deployment,
            dex_rotation,
            target_tube=target_tube,
            canonical_sample_ids=dexterity_bank.sample_ids,
        )
    ik_errors = None
    if ik_targets_mm is not None and len(ik_targets_mm):
        ik_errors = calculate_ik_error_map(
            parameters,
            ik_targets_mm,
            target_tube=target_tube,
        )
    metrics = {
        "workspace_volume_mm3": volume_mm3,
        "workspace_volume_cm3": volume_mm3 / 1000.0,
        "workspace_surface_area_mm2": area_mm2,
        "workspace_surface_area_cm2": area_mm2 / 100.0,
        **summarise_dexterity(dexterity),
        **summarise_ik_errors(ik_errors),
    }
    return DesignEvaluation(design, metrics, workspace_tips, dexterity, ik_errors)


def independent_baseline_targets(
    sample_count: int,
    *,
    target_tube: int = 0,
    seed: int = 1042,
) -> np.ndarray:
    """Compatibility wrapper returning configuration-derived target positions."""
    return independent_baseline_target_set(
        sample_count,
        target_tube=target_tube,
        seed=seed,
    ).targets_mm


def independent_baseline_target_set(
    sample_count: int,
    *,
    target_tube: int = 0,
    seed: int = 1042,
    split: str = "training-legacy-ik-target-source",
) -> CanonicalTargetSet:
    """Generate reachable legacy targets with canonical source IDs.

    Stage 2 will replace this configuration-derived target distribution with a
    spatially stratified fixed task-space target set. Retaining IDs now prevents
    rotated display or repeated-solve copies from inflating independent counts.
    """
    design = baseline_design()
    parameters = design.to_parameters()
    bank, deployment, rotation = canonical_configuration_samples(
        design.total_length_mm * 1e-3,
        sample_count,
        seed=seed,
        split=split,
    )
    return CanonicalTargetSet(
        targets_mm=_workspace_tips(parameters, deployment, rotation, target_tube),
        canonical_sample_ids=bank.sample_ids.copy(),
        source_deployment_m=deployment.astype(np.float32),
        source_rotation_rad=rotation.astype(np.float32),
        split=bank.split,
        seed=bank.seed,
    )


def sensitivity_study(
    design: TubeDesign,
    variables: Iterable[DesignVariable],
    evaluator: Callable[[TubeDesign], DesignEvaluation],
    *,
    relative_levels: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10),
) -> list[dict[str, float | str]]:
    """Perform a bounded one-at-a-time sensitivity study."""
    records: list[dict[str, float | str]] = []
    for variable in variables:
        used_values: set[float] = set()
        for relative in relative_levels:
            requested = variable.baseline * (1.0 + relative)
            value = float(np.clip(requested, variable.lower, variable.upper))
            rounded = round(value, 12)
            if rounded in used_values:
                continue
            used_values.add(rounded)
            try:
                candidate = with_design_value(design, variable.name, value)
            except ValueError:
                continue
            evaluation = evaluator(candidate)
            actual_relative = (
                value / variable.baseline - 1.0
                if abs(variable.baseline) > 1e-12
                else float("nan")
            )
            records.append(
                {
                    "parameter": variable.name,
                    "label": variable.label,
                    "unit": variable.unit,
                    "value": value,
                    "relative_change": actual_relative,
                    **evaluation.metrics,
                }
            )
    return records


def rank_sensitivity_variables(
    records: list[dict[str, float | str]],
    metric: str,
) -> list[str]:
    """Rank parameters by their finite range in a sensitivity output metric."""
    groups: dict[str, list[float]] = {}
    for record in records:
        value = float(record[metric])
        if np.isfinite(value):
            groups.setdefault(str(record["parameter"]), []).append(value)
    return sorted(
        groups,
        key=lambda name: np.ptp(groups[name]) if len(groups[name]) > 1 else 0.0,
        reverse=True,
    )


def optimise_design(
    baseline: TubeDesign,
    variables: Iterable[DesignVariable],
    evaluator: Callable[[TubeDesign], DesignEvaluation],
    objective_from_evaluation: Callable[[DesignEvaluation], float],
    *,
    seed: int = 42,
    max_iterations: int = 8,
    population_size: int = 6,
) -> OptimisationResult:
    """Run bounded differential evolution with invalid-design penalties."""
    selected = tuple(variables)
    if not selected:
        raise ValueError("At least one design variable is required")
    bounds = [(item.lower, item.upper) for item in selected]
    best_evaluation: DesignEvaluation | None = None
    best_objective = float("inf")

    def objective(values: np.ndarray) -> float:
        nonlocal best_evaluation, best_objective
        candidate = baseline
        try:
            for variable, value in zip(selected, values):
                candidate = with_design_value(
                    candidate, variable.name, float(value), validate=False
                )
            candidate = candidate.validated()
            evaluation = evaluator(candidate)
            score = float(objective_from_evaluation(evaluation))
            if not np.isfinite(score):
                return 1e12
            if score < best_objective:
                best_objective = score
                best_evaluation = evaluation
            return score
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e12

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=int(max_iterations),
        popsize=int(population_size),
        polish=False,
        updating="immediate",
        workers=1,
    )
    if best_evaluation is None:
        raise RuntimeError("Optimisation found no valid candidate design")
    return OptimisationResult(
        design=best_evaluation.design,
        variables=tuple(item.name for item in selected),
        values=np.asarray(
            [design_value(best_evaluation.design, item.name) for item in selected]
        ),
        objective=best_objective,
        evaluation=best_evaluation,
        success=bool(result.success),
        message=str(result.message),
        function_evaluations=int(result.nfev),
    )
