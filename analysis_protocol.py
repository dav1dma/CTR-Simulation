"""Load, validate and record the versioned CTR analysis methodology."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Mapping

import numpy as np

from tube_parameters import TUBE_DATA, build_supervisor_ctr_parameters


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "config" / "analysis_protocol_v1.toml"
SUPPORTED_SCHEMA_VERSION = 1

_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "protocol_id",
    "protocol_version",
    "methodology_stage",
    "status",
    "scope",
    "baseline",
    "task_space",
    "sampling",
    "symmetry",
    "spatial",
    "ik",
    "jacobian",
    "endpoints",
    "optimisation",
    "versioning",
}
_OPTIONAL_TOP_LEVEL = {"extension"}


def canonical_json_bytes(value: Any) -> bytes:
    """Serialise methodological data deterministically for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def baseline_model_snapshot() -> dict[str, list[float]]:
    """Return only numerical values that affect the current forward model."""
    parameters = build_supervisor_ctr_parameters()
    keys = ("inner", "middle", "outer")
    return {
        "total_length_mm": [
            float(TUBE_DATA[key]["total_length_mm"]) for key in keys
        ],
        "curved_length_mm": [
            float(TUBE_DATA[key]["curved_length_mm"]) for key in keys
        ],
        "od_mm": [float(TUBE_DATA[key]["od_mm"]) for key in keys],
        "id_mm": [float(TUBE_DATA[key]["id_mm"]) for key in keys],
        "precurvature_per_m": [float(value) for value in parameters["kappa_0"]],
        "youngs_modulus_gpa": [float(value) / 1e9 for value in parameters["E"]],
    }


def baseline_parameter_hash() -> str:
    return sha256_mapping(baseline_model_snapshot())


def _require_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"protocol section [{name}] is required")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not number > 0.0 or not number < float("inf"):
        raise ValueError(f"{label} must be finite and positive")
    return number


def validate_analysis_protocol(data: Mapping[str, Any]) -> None:
    """Reject incomplete or internally inconsistent Stage-3 protocols."""
    value = dict(data)
    missing = _REQUIRED_TOP_LEVEL - set(value)
    unknown = set(value) - _REQUIRED_TOP_LEVEL - _OPTIONAL_TOP_LEVEL
    if missing:
        raise ValueError(f"protocol is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"protocol has unknown fields: {sorted(unknown)}")
    schema = value["schema_version"]
    if isinstance(schema, bool) or schema != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported analysis protocol schema: {schema!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )
    methodology_stage = value["methodology_stage"]
    if methodology_stage not in (3, 4):
        raise ValueError("this configuration must describe methodology Stage 3 or 4")
    if methodology_stage == 3 and "extension" in value:
        raise ValueError("Stage-3 protocols cannot contain a Stage-4 extension")
    if methodology_stage == 4 and "extension" not in value:
        raise ValueError("Stage-4 protocols require an [extension] section")

    scope = _require_table(value, "scope")
    if scope.get("primary_endpoint_index") != 0 or scope.get("primary_endpoint") != "inner":
        raise ValueError("the agreed primary endpoint is the distal inner-tube tip")
    if scope.get("secondary_endpoints") != ["middle", "outer"]:
        raise ValueError("middle and outer endpoints must remain secondary outputs")
    excluded = set(scope.get("excluded_quantities", []))
    required_exclusions = {
        "tip_orientation_objective",
        "force_capability",
        "loaded_stiffness_capability",
    }
    if not required_exclusions.issubset(excluded):
        raise ValueError("orientation, force and loaded stiffness must be out of scope")

    baseline = _require_table(value, "baseline")
    snapshot = baseline_model_snapshot()
    comparisons = {
        "total_length_mm": snapshot["total_length_mm"],
        "curved_length_mm": snapshot["curved_length_mm"],
        "outer_diameter_mm": snapshot["od_mm"],
        "inner_diameter_mm": snapshot["id_mm"],
        "youngs_modulus_gpa": snapshot["youngs_modulus_gpa"],
        "precurvature_per_m": snapshot["precurvature_per_m"],
    }
    for field, expected in comparisons.items():
        actual = baseline.get(field)
        if actual != expected:
            raise ValueError(
                f"protocol baseline {field} does not match tube_parameters.py: "
                f"{actual!r} != {expected!r}"
            )
    expected_hash = baseline.get("expected_model_parameter_sha256")
    if expected_hash != baseline_parameter_hash():
        raise ValueError(
            "protocol baseline parameter hash does not match the live model parameters"
        )
    if baseline.get("youngs_modulus_role") != "fixed-nominal-uncertainty-only":
        raise ValueError("Young's modulus must be fixed nominal uncertainty only")
    modulus_ranges = [
        list(map(float, TUBE_DATA[key]["E_range_GPa"]))
        for key in ("inner", "middle", "outer")
    ]
    if baseline.get("youngs_modulus_uncertainty_ranges_gpa") != modulus_ranges:
        raise ValueError("protocol modulus uncertainty ranges do not match tube metadata")
    if (
        baseline.get("precurvature_interpretation")
        != "reported-numeric-values-treated-as-per-metre"
    ):
        raise ValueError("the provisional pre-curvature interpretation must be explicit")

    task_space = _require_table(value, "task_space")
    if task_space.get("definition") != "baseline-inner-tip-occupied-workspace":
        raise ValueError("the provisional task space must be the baseline inner-tip region")
    if (
        task_space.get("target_selection")
        != "spatially-stratified-known-reachable-canonical-points"
    ):
        raise ValueError("formal IK targets must be spatially stratified")
    if (
        task_space.get("target_weighting")
        != "swept-cell-volume-divided-by-targets-in-cell"
    ):
        raise ValueError("formal IK metrics must use swept-volume target weights")
    if task_space.get("task_discovery_and_target_candidates_are_independent") is not True:
        raise ValueError("task discovery and target candidates must be independent")
    if task_space.get("target_ik_initialisation_uses_source_state") is not False:
        raise ValueError("target-generating states cannot initialise formal IK")

    sampling = _require_table(value, "sampling")
    if sampling.get("sampler_version") != "ctr-canonical-pcg64-v1":
        raise ValueError("unsupported canonical sampler version")
    if sampling.get("independent_dimensions") != 5:
        raise ValueError("formal symmetry-reduced sampling must contain five DOF")
    if sampling.get("training_seed") == sampling.get("validation_seed"):
        raise ValueError("training and validation seeds must be different")
    if sampling.get("boundary_cases_are_statistical_samples") is not False:
        raise ValueError("boundary cases cannot be counted as statistical samples")
    if sampling.get("streams") != {
        "deployment_fractions": 0,
        "relative_rotations": 1,
    }:
        raise ValueError("canonical sampler stream identifiers must remain fixed")
    counts = sampling.get("final_convergence_counts")
    if not isinstance(counts, list) or not counts or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in counts
    ):
        raise ValueError("final convergence counts must be positive integers")
    if counts != sorted(set(counts)):
        raise ValueError("final convergence counts must be unique and increasing")
    baseline_counts = sampling.get("baseline_convergence_counts")
    expected_counts = (
        [15000, 25000, 50000, 100000]
        if methodology_stage == 3
        else [50000, 100000, 150000, 200000]
    )
    if baseline_counts != expected_counts:
        raise ValueError(
            f"Stage-{methodology_stage} convergence counts must remain predeclared"
        )
    if sampling.get("baseline_training_configurations") != baseline_counts[-1]:
        raise ValueError("training bank must cover the largest convergence prefix")
    for field in (
        "baseline_validation_configurations",
        "ik_target_candidate_configurations",
        "ik_primary_training_targets",
        "ik_primary_validation_targets",
        "ik_secondary_validation_targets",
    ):
        count = sampling.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{field} must be a positive integer")
    sampling_seeds = {
        sampling.get("training_seed"),
        sampling.get("validation_seed"),
        sampling.get("ik_target_training_seed"),
        sampling.get("ik_target_validation_seed"),
    }
    if len(sampling_seeds) != 4:
        raise ValueError("all formal sampling roles must use distinct seeds")

    if methodology_stage == 4:
        extension = _require_table(value, "extension")
        extension_type = extension.get("extension_type")
        common_extension = {
            "primary_endpoint_only": True,
            "ik_spatial_support_cell_size_mm": 10.0,
            "ik_spatial_support_minimum_independent_targets": 100,
            "ik_spatial_reportable_volume_minimum": 0.9,
            "post_hoc_objective_or_threshold_change": False,
        }
        for field, expected in common_extension.items():
            if extension.get(field) != expected:
                raise ValueError(f"unexpected Stage-4 extension setting: {field}")
        if sampling.get("baseline_training_configurations") != 200000:
            raise ValueError("Stage-4 evidence must retain 200,000 training samples")
        if sampling.get("baseline_validation_configurations") != 50000:
            raise ValueError("Stage-4 evidence must retain 50,000 validation samples")
        if extension_type == "controlled-primary-endpoint-convergence":
            expected = {
                "reuse_verified_stage3_prefix": True,
                "configuration_comparison_counts": [150000, 200000],
            }
            source_fields = (
                "source_stage3_run",
                "source_protocol_sha256",
                "source_training_sha256",
                "source_validation_sha256",
            )
            digest_fields = source_fields[1:]
            candidate_count = 300000
            target_count = 75000
        elif extension_type == "controlled-primary-endpoint-ik-support-completion":
            expected = {
                "reuse_verified_stage4_task_region": True,
                "previous_supported_task_volume_fraction": 0.8660898760330579,
            }
            source_fields = (
                "source_stage4_run",
                "source_run_manifest_sha256",
                "source_protocol_sha256",
                "source_task_region_sha256",
                "source_workspace_metrics_sha256",
                "source_validation_gates_sha256",
                "source_ik_metrics_sha256",
            )
            digest_fields = source_fields[1:]
            candidate_count = 600000
            target_count = 100000
        else:
            raise ValueError(f"unsupported Stage-4 extension type: {extension_type!r}")
        for field, expected_value in expected.items():
            if extension.get(field) != expected_value:
                raise ValueError(f"unexpected Stage-4 extension setting: {field}")
        if sampling.get("ik_target_candidate_configurations") != candidate_count:
            raise ValueError(
                f"{extension_type} requires {candidate_count:,} IK candidates"
            )
        if sampling.get("ik_primary_validation_targets") != target_count:
            raise ValueError(
                f"{extension_type} requires {target_count:,} inner-tip targets"
            )
        for field in source_fields:
            text = extension.get(field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Stage-4 extension requires {field}")
        for field in digest_fields:
            digest = extension[field]
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")

    symmetry = _require_table(value, "symmetry")
    if symmetry.get("axis") != "z" or symmetry.get("period_degrees") != 360.0:
        raise ValueError("the agreed symmetry is a continuous 360-degree Z rotation")
    if symmetry.get("rotated_copies_are_independent") is not False:
        raise ValueError("rotated display copies cannot be independent samples")
    if symmetry.get("statistical_aggregation_before_sweep") is not True:
        raise ValueError("statistics must be aggregated before the display sweep")

    spatial = _require_table(value, "spatial")
    _finite_positive(spatial.get("provisional_cell_size_mm"), "cell size")
    if spatial.get("cell_size_convergence_candidates_mm") != [5.0, 10.0, 15.0]:
        raise ValueError("Stage-1 cell-size convergence candidates must be 5/10/15 mm")
    for field in (
        "minimum_samples_median",
        "minimum_samples_lower_tail",
        "minimum_samples_failure_or_p95",
    ):
        count = spatial.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{field} must be a positive integer")
    expected_spatial = {
        "coordinate_system": "axisymmetric-r-z",
        "cell_intervals": "left-closed-right-open",
        "occupancy_minimum_independent_samples": 1,
        "occupied_volume_method": "sum-full-swept-annular-cells",
        "metric_smoothing": False,
        "void_connectivity": 4,
        "axis_is_exterior_boundary": False,
        "local_isotropy_representative": (
            "median-over-independent-configurations"
        ),
        "spatial_weighting": "exact-swept-annular-volume",
        "weighted_quantile_method": "inverted-cdf",
    }
    for field, expected in expected_spatial.items():
        if spatial.get(field) != expected:
            raise ValueError(f"unexpected Stage-2 spatial setting: {field}")
    if spatial.get("radial_origin_mm") != 0.0 or spatial.get("z_origin_mm") != 0.0:
        raise ValueError("the radial/Z grid must remain globally anchored at zero")
    if spatial.get("primary_cell_size_mm") != spatial.get("provisional_cell_size_mm"):
        raise ValueError("the predeclared primary cell size must remain 10 mm")
    _finite_positive(
        spatial.get("workspace_relative_change_limit"),
        "workspace convergence limit",
    )
    coverage_minimum = _finite_positive(
        spatial.get("validation_coverage_minimum"),
        "validation coverage minimum",
    )
    if coverage_minimum > 1.0:
        raise ValueError("validation coverage minimum cannot exceed one")
    _finite_positive(
        spatial.get("spatial_isotropy_absolute_change_limit"),
        "spatial isotropy convergence limit",
    )
    if (
        spatial.get("numerical_gate_interpretation")
        != "analysis-stability-not-clinical-performance"
    ):
        raise ValueError("Stage-3 gates cannot be presented as clinical limits")
    if spatial.get("cell_size_sensitivity_required") is not True:
        raise ValueError("5/10/15 mm cell-size sensitivity remains required")

    ik = _require_table(value, "ik")
    evaluation_threshold = _finite_positive(
        ik.get("evaluation_threshold_mm"), "IK evaluation threshold"
    )
    tolerance_candidates = ik.get("solver_tolerance_candidates_mm")
    if not isinstance(tolerance_candidates, list) or not tolerance_candidates:
        raise ValueError("solver tolerance candidates are required")
    if any(
        _finite_positive(item, "solver tolerance candidate") >= evaluation_threshold
        for item in tolerance_candidates
    ):
        raise ValueError("solver tolerance candidates must be below the 0.5 mm threshold")
    selected_solver_tolerance = _finite_positive(
        ik.get("solver_tolerance_mm"), "selected solver tolerance"
    )
    if selected_solver_tolerance not in tolerance_candidates:
        raise ValueError("selected solver tolerance must be one tested candidate")
    if selected_solver_tolerance >= evaluation_threshold:
        raise ValueError("solver tolerance must remain below evaluation threshold")
    if ik.get("solver_tolerance_status") != "provisional-stage-2-pilot-selection":
        raise ValueError("the Stage-2 solver-tolerance status must remain explicit")
    for field in (
        "max_iterations",
        "pilot_target_count",
        "pilot_task_discovery_configurations",
        "pilot_target_candidate_configurations",
    ):
        count = ik.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in (
        "damping_mm",
        "maximum_normalised_step",
        "finite_difference_step_normalised",
    ):
        _finite_positive(ik.get(field), field)
    if ik.get("pilot_revalidation_required_at_full_scale") is not True:
        raise ValueError("the provisional IK pilot must be revalidated at full scale")

    jacobian = _require_table(value, "jacobian")
    expected_scale = [350.0, 170.0, 80.0, float(np.pi), float(np.pi), float(np.pi)]
    actual_scale = jacobian.get("actuator_scale")
    if not isinstance(actual_scale, list) or not np.allclose(
        actual_scale,
        expected_scale,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Jacobian scale must remain fixed to the baseline reference")
    if jacobian.get("normalisation_status") != "fixed-baseline-range-scale":
        raise ValueError("Jacobian normalisation must be fixed across designs")
    if jacobian.get("actuator_velocity_interpretation") is not False:
        raise ValueError("no actuator-speed interpretation is justified")
    if (
        jacobian.get("finite_difference_scheme")
        != "three-point-central-adaptive-inside-constraints"
    ):
        raise ValueError("formal Jacobians must use adaptive central differences")
    translation_candidates = jacobian.get("translation_step_candidates_mm")
    rotation_candidates = jacobian.get("rotation_step_candidates_rad")
    if jacobian.get("translation_step_mm") not in translation_candidates:
        raise ValueError("selected translation step was not convergence-tested")
    if jacobian.get("rotation_step_rad") not in rotation_candidates:
        raise ValueError("selected rotation step was not convergence-tested")
    if (
        jacobian.get("boundary_policy")
        != "mask-no-six-dimensional-two-sided-neighbourhood"
    ):
        raise ValueError("deployment-boundary Jacobians must remain masked")

    optimisation = _require_table(value, "optimisation")
    if optimisation.get("enabled") is not False:
        raise ValueError("optimisation must remain disabled until baseline validation")
    if optimisation.get("youngs_modulus_is_design_variable") is not False:
        raise ValueError("Young's modulus cannot be an optimisation variable")
    if optimisation.get("orientation_is_objective") is not False:
        raise ValueError("tip orientation is outside the agreed optimisation scope")
    if optimisation.get("force_or_loaded_stiffness_is_objective") is not False:
        raise ValueError("force and loaded stiffness are outside the agreed scope")


@dataclass(frozen=True)
class AnalysisProtocol:
    path: Path
    _snapshot: dict[str, Any]
    sha256: str

    @property
    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._snapshot)

    def section(self, name: str) -> dict[str, Any]:
        value = self._snapshot.get(name)
        if not isinstance(value, dict):
            raise KeyError(name)
        return deepcopy(value)


def load_analysis_protocol(
    path: Path | str = DEFAULT_PROTOCOL_PATH,
) -> AnalysisProtocol:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    validate_analysis_protocol(data)
    return AnalysisProtocol(source, deepcopy(data), sha256_mapping(data))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_code_provenance(
    project_root: Path | str = PROJECT_ROOT,
    source_paths: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Capture Git state plus hashes for tracked and untracked analysis sources."""
    root = Path(project_root).resolve()
    if source_paths is None:
        source_paths = (
            "analysis_protocol.py",
            "optimization_protocol.py",
            "ctr_sampling.py",
            "ctr_design_analysis.py",
            "ctr_spatial_analysis.py",
            "ctr_task_space.py",
            "ctr_inverse_kinematics.py",
            "ctr_workspace_map.py",
            "CTR_superPosKin_fun_sectioned.py",
            "tube_parameters.py",
            "tools/run_design_study.py",
            "tools/run_stage2_method_validation.py",
            "tools/run_stage3_baseline_evaluation.py",
            "tools/run_stage4_convergence_extension.py",
            "tools/run_stage4_1_ik_support_extension.py",
            "tools/run_stage5a_protocol_preflight.py",
            "tools/run_symmetry_expanded_study.py",
            "tools/run_hybrid_ik_study.py",
            "tools/plot_axisymmetric_ik_slice.py",
        )

    def git(*arguments: str) -> str | None:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--short")
    hashes = {
        relative: _file_sha256(root / relative)
        for relative in source_paths
        if (root / relative).is_file()
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
        "source_file_sha256": hashes,
    }


def runtime_provenance() -> dict[str, str]:
    packages = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        **packages,
    }


def build_run_manifest(
    protocol: AnalysisProtocol,
    *,
    run_kind: str,
    run_status: str = "started",
    effective_run_settings: Mapping[str, Any] | None = None,
    protocol_deviations: list[str] | None = None,
    command: list[str] | None = None,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build a self-contained manifest before any expensive evaluation runs."""
    settings = dict(effective_run_settings or {})
    deviations = list(protocol_deviations or [])
    return {
        "manifest_schema_version": 1,
        "run_kind": str(run_kind),
        "run_status": str(run_status),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv if command is None else command),
        "protocol": {
            "path": os.path.relpath(protocol.path, Path(project_root).resolve()),
            "sha256": protocol.sha256,
            "snapshot": protocol.snapshot,
        },
        "baseline_parameters": {
            "parameter_set_id": protocol.section("baseline")["parameter_set_id"],
            "sha256": baseline_parameter_hash(),
            "snapshot": baseline_model_snapshot(),
        },
        "code": capture_code_provenance(project_root),
        "runtime": runtime_provenance(),
        "effective_run_settings": settings,
        "protocol_deviations": deviations,
        "protocol_compliant": not deviations,
    }


def write_run_manifest(path: Path | str, manifest: Mapping[str, Any]) -> None:
    """Atomically write a manifest, rejecting NaN and infinite values."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(manifest),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    temporary.replace(destination)
