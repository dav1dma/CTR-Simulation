"""Versioned configuration sampling and rotational-display bookkeeping.

The formal design-analysis sampler removes the redundant common tube rotation
from the independent configuration space.  It samples three nested deployment
fractions and two relative rotations, with the outer-tube rotation fixed as the
SO(2) gauge.  A common rotation is applied only when a canonical result is
rendered around the robot's Z axis.

Viewer sampling remains available separately because its nearest-neighbour
cache needs points distributed through the complete rendered XYZ workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


CANONICAL_SAMPLER_VERSION = "ctr-canonical-pcg64-v1"
FULL_STATE_SAMPLER_VERSION = "ctr-full-state-pcg64-v1"

_DEPLOYMENT_STREAM = 0
_ROTATION_STREAM = 1


def _validate_lengths(total_lengths_m: np.ndarray) -> np.ndarray:
    lengths = np.asarray(total_lengths_m, dtype=float).reshape(-1)
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)):
        raise ValueError("total_lengths_m must contain three finite values")
    if np.any(lengths <= 0.0):
        raise ValueError("tube lengths must be positive")
    if not lengths[0] >= lengths[1] >= lengths[2]:
        raise ValueError("tube lengths must be ordered inner >= middle >= outer")
    return lengths


def decode_nested_deployments(
    deployment_fractions: np.ndarray,
    total_lengths_m: np.ndarray,
) -> np.ndarray:
    """Map three unit-cube fractions to valid nested tube deployments."""
    fractions = np.asarray(deployment_fractions, dtype=float)
    if fractions.ndim != 2 or fractions.shape[1] != 3:
        raise ValueError("deployment_fractions must have shape (N, 3)")
    if not np.all(np.isfinite(fractions)) or np.any(
        (fractions < 0.0) | (fractions > 1.0)
    ):
        raise ValueError("deployment fractions must be finite and lie in [0, 1]")
    lengths = _validate_lengths(total_lengths_m)
    outer = fractions[:, 2] * lengths[2]
    middle = outer + fractions[:, 1] * (lengths[1] - outer)
    inner = middle + fractions[:, 0] * (lengths[0] - middle)
    return np.column_stack((inner, middle, outer))


def _rng_stream(seed: int, stream: int) -> np.random.Generator:
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be non-negative")
    sequence = np.random.SeedSequence([value, int(stream)])
    return np.random.Generator(np.random.PCG64(sequence))


def _sample_namespace(method: str, split: str, seed: int) -> str:
    payload = f"{method}|{split}|{int(seed)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def canonical_sample_ids(
    sample_count: int,
    *,
    split: str,
    seed: int,
    method: str = CANONICAL_SAMPLER_VERSION,
) -> np.ndarray:
    """Return stable, split-namespaced IDs whose prefixes never change."""
    count = int(sample_count)
    if count < 1:
        raise ValueError("sample_count must be positive")
    split_name = str(split).strip().lower()
    if not split_name or any(character.isspace() for character in split_name):
        raise ValueError("split must be a non-empty identifier without whitespace")
    namespace = _sample_namespace(method, split_name, seed)
    return np.asarray(
        [f"{namespace}:{index:010d}" for index in range(count)],
        dtype="U23",
    )


@dataclass(frozen=True)
class CanonicalConfigurationBank:
    """Independent configurations after quotienting common Z-axis rotation."""

    sample_ids: np.ndarray
    split: str
    seed: int
    deployment_fractions: np.ndarray
    relative_rotation_rad: np.ndarray
    sampler_version: str = CANONICAL_SAMPLER_VERSION

    def __post_init__(self) -> None:
        count = len(self.sample_ids)
        if self.deployment_fractions.shape != (count, 3):
            raise ValueError("deployment_fractions must have shape (N, 3)")
        if self.relative_rotation_rad.shape != (count, 2):
            raise ValueError("relative_rotation_rad must have shape (N, 2)")
        if len(np.unique(self.sample_ids)) != count:
            raise ValueError("canonical sample IDs must be unique")

    @property
    def independent_count(self) -> int:
        return len(self.sample_ids)

    @property
    def rotation_rad(self) -> np.ndarray:
        """Return [inner, middle, outer] angles with outer fixed as the gauge."""
        return np.column_stack(
            (
                self.relative_rotation_rad,
                np.zeros(self.independent_count, dtype=float),
            )
        )

    def decode_deployments(self, total_lengths_m: np.ndarray) -> np.ndarray:
        return decode_nested_deployments(
            self.deployment_fractions,
            total_lengths_m,
        )

    def prefix(self, sample_count: int) -> "CanonicalConfigurationBank":
        count = int(sample_count)
        if count < 1 or count > self.independent_count:
            raise ValueError("prefix size must lie within the configuration bank")
        return CanonicalConfigurationBank(
            sample_ids=self.sample_ids[:count].copy(),
            split=self.split,
            seed=self.seed,
            deployment_fractions=self.deployment_fractions[:count].copy(),
            relative_rotation_rad=self.relative_rotation_rad[:count].copy(),
            sampler_version=self.sampler_version,
        )


def canonical_configuration_bank(
    sample_count: int,
    *,
    seed: int,
    split: str,
) -> CanonicalConfigurationBank:
    """Sample prefix-stable independent configurations for formal analysis.

    Boundary corners are deliberately not injected into this statistical bank.
    They are available from :func:`boundary_deployment_fractions` as separate
    deterministic diagnostics.
    """
    count = int(sample_count)
    if count < 1:
        raise ValueError("sample_count must be positive")
    fraction_rng = _rng_stream(seed, _DEPLOYMENT_STREAM)
    rotation_rng = _rng_stream(seed, _ROTATION_STREAM)
    fractions = fraction_rng.random((count, 3))
    relative_rotation = rotation_rng.uniform(-np.pi, np.pi, size=(count, 2))
    return CanonicalConfigurationBank(
        sample_ids=canonical_sample_ids(count, split=split, seed=seed),
        split=str(split).strip().lower(),
        seed=int(seed),
        deployment_fractions=fractions,
        relative_rotation_rad=relative_rotation,
    )


def independent_configuration_banks(
    *,
    training_count: int,
    validation_count: int,
    training_seed: int,
    validation_seed: int,
) -> tuple[CanonicalConfigurationBank, CanonicalConfigurationBank]:
    """Create explicitly disjoint training and validation configuration banks."""
    if int(training_seed) == int(validation_seed):
        raise ValueError("training and validation seeds must differ")
    training = canonical_configuration_bank(
        training_count,
        seed=training_seed,
        split="training",
    )
    validation = canonical_configuration_bank(
        validation_count,
        seed=validation_seed,
        split="validation",
    )
    if np.intersect1d(training.sample_ids, validation.sample_ids).size:
        raise RuntimeError("training and validation sample IDs overlap")
    return training, validation


def boundary_deployment_fractions() -> np.ndarray:
    """Return the eight unit-cube corners as non-statistical diagnostics."""
    return np.asarray(
        [
            [inner, middle, outer]
            for inner in (0.0, 1.0)
            for middle in (0.0, 1.0)
            for outer in (0.0, 1.0)
        ],
        dtype=float,
    )


def canonical_configuration_samples(
    total_lengths_m: np.ndarray,
    sample_count: int,
    *,
    seed: int,
    split: str,
) -> tuple[CanonicalConfigurationBank, np.ndarray, np.ndarray]:
    """Return a canonical bank and its decoded deployment/rotation arrays."""
    bank = canonical_configuration_bank(sample_count, seed=seed, split=split)
    return bank, bank.decode_deployments(total_lengths_m), bank.rotation_rad


def full_configuration_samples(
    total_lengths_m: np.ndarray,
    sample_count: int,
    *,
    seed: int,
    include_boundary_cases: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample all six actuator inputs for viewer or legacy full-XYZ uses.

    This is not the protocol-v1 statistical sampler because the common tube
    rotation is redundant under the assumed continuous Z-axis symmetry.
    """
    count = int(sample_count)
    if count < 1:
        raise ValueError("sample_count must be positive")
    if include_boundary_cases and count < 8:
        raise ValueError("at least eight rows are required for boundary cases")
    fraction_rng = _rng_stream(seed, _DEPLOYMENT_STREAM)
    rotation_rng = _rng_stream(seed, _ROTATION_STREAM)
    fractions = fraction_rng.random((count, 3))
    if include_boundary_cases:
        fractions[:8] = boundary_deployment_fractions()
    deployment = decode_nested_deployments(fractions, total_lengths_m)
    rotation = rotation_rng.uniform(-np.pi, np.pi, size=(count, 3))
    return deployment, rotation


@dataclass(frozen=True)
class SymmetryDisplayExpansion:
    """Rendered copies that retain their independent canonical provenance."""

    points_mm: np.ndarray
    source_sample_ids: np.ndarray
    display_angles_rad: np.ndarray
    display_copy_index: np.ndarray

    @property
    def rendered_count(self) -> int:
        return len(self.points_mm)

    @property
    def independent_count(self) -> int:
        return len(np.unique(self.source_sample_ids))


def revolve_points_for_display(
    canonical_points_mm: np.ndarray,
    canonical_sample_ids_values: np.ndarray,
    angles_rad: np.ndarray,
) -> SymmetryDisplayExpansion:
    """Revolve canonical points without creating new statistical samples."""
    points = np.asarray(canonical_points_mm, dtype=float)
    identifiers = np.asarray(canonical_sample_ids_values)
    angles = np.asarray(angles_rad, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("canonical_points_mm must have shape (N, 3)")
    if identifiers.shape != (len(points),):
        raise ValueError("canonical sample IDs must have shape (N,)")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("canonical sample IDs must be unique before display expansion")
    if not len(angles) or not np.all(np.isfinite(angles)):
        raise ValueError("angles_rad must contain at least one finite angle")
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    x = points[:, 0][None, :]
    y = points[:, 1][None, :]
    rendered = np.empty((len(angles), len(points), 3), dtype=float)
    rendered[:, :, 0] = cosine * x - sine * y
    rendered[:, :, 1] = sine * x + cosine * y
    rendered[:, :, 2] = points[:, 2][None, :]
    return SymmetryDisplayExpansion(
        points_mm=rendered.reshape(-1, 3).astype(np.float32),
        source_sample_ids=np.tile(identifiers, len(angles)),
        display_angles_rad=np.repeat(angles, len(points)),
        display_copy_index=np.repeat(
            np.arange(len(angles), dtype=np.int32),
            len(points),
        ),
    )
