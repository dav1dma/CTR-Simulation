"""Load, validate and audit the frozen Stage-5A optimisation methodology."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any, Mapping

import numpy as np

from analysis_protocol import (
    baseline_model_snapshot,
    baseline_parameter_hash,
    capture_code_provenance,
    runtime_provenance,
    sha256_mapping,
)
from ctr_design_analysis import TubeDesign, baseline_design


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OPTIMISATION_PROTOCOL_PATH = (
    PROJECT_ROOT / "config" / "optimization_protocol_stage5a.toml"
)

_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "protocol_id",
    "protocol_version",
    "methodology_stage",
    "status",
    "scope",
    "baseline_evidence",
    "task_space",
    "fixed_parameters",
    "design_variables",
    "constraints",
    "objectives",
    "sampling",
    "stage5b_pilot",
    "stage5c_search",
    "stage5d_validation",
    "reporting",
    "unresolved_decisions",
    "versioning",
}

_VARIABLES = (
    ("inner_total_length_mm", "total_length_mm", 0, 315.0, 350.0, 385.0),
    ("middle_total_length_mm", "total_length_mm", 1, 153.0, 170.0, 187.0),
    ("outer_total_length_mm", "total_length_mm", 2, 72.0, 80.0, 88.0),
    ("middle_curved_length_mm", "curved_length_mm", 1, 81.0, 90.0, 99.0),
    ("outer_curved_length_mm", "curved_length_mm", 2, 58.5, 65.0, 71.5),
    ("middle_precurvature_per_m", "precurvature_per_m", 1, 17.208, 19.12, 21.032),
    ("outer_precurvature_per_m", "precurvature_per_m", 2, 12.636, 14.04, 15.444),
)

_OBJECTIVES = (
    ("common_task_region_coverage", "maximize"),
    ("mean_positional_isotropy", "maximize"),
    ("spatial_isotropy_iqr", "minimize"),
    ("ik_failure_fraction_0_5mm", "minimize"),
    ("ik_p95_residual_mm", "minimize"),
)


def _table(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = value.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"optimisation protocol section [{name}] is required")
    return section


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_optimization_protocol(data: Mapping[str, Any]) -> None:
    """Reject changes to the frozen Stage-5A methodological decisions."""
    value = dict(data)
    missing = _REQUIRED_TOP_LEVEL - set(value)
    unknown = set(value) - _REQUIRED_TOP_LEVEL
    if missing:
        raise ValueError(f"optimisation protocol is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"optimisation protocol has unknown fields: {sorted(unknown)}")
    if value["schema_version"] != 1 or value["methodology_stage"] != "5A":
        raise ValueError("this file must describe Stage 5A schema version 1")
    if value["status"] != "frozen-for-local-computational-pilot-not-manufacturing-certified":
        raise ValueError("the Stage-5A status must retain its manufacturing limitation")

    scope = _table(value, "scope")
    if scope.get("primary_endpoint") != "inner":
        raise ValueError("the inner tip must remain the primary endpoint")
    if scope.get("secondary_endpoints") != ["middle", "outer"]:
        raise ValueError("middle and outer endpoints must remain secondary")
    required_exclusions = {
        "tip-orientation",
        "force-capability",
        "loaded-stiffness-capability",
        "collision-contact-and-buckling",
        "clinical-safety-or-success",
    }
    if not required_exclusions.issubset(set(scope.get("excluded_quantities", []))):
        raise ValueError("the agreed out-of-scope quantities must remain explicit")

    evidence = _table(value, "baseline_evidence")
    if evidence.get("parameter_sha256") != baseline_parameter_hash():
        raise ValueError("Stage-5A baseline parameter hash does not match the live model")
    for field in (
        "stage4_protocol_sha256",
        "stage4_manifest_sha256",
        "stage4_gates_sha256",
        "task_region_sha256",
        "training_configurations_sha256",
        "validation_configurations_sha256",
        "stage4_1_protocol_sha256",
        "stage4_1_manifest_sha256",
        "stage4_1_gates_sha256",
        "reserved_validation_targets_sha256",
        "reserved_validation_results_sha256",
    ):
        _sha256_text(evidence.get(field), field)

    task = _table(value, "task_space")
    expected_task = {
        "definition": "frozen-stage4-baseline-inner-tip-occupied-cells",
        "coordinate_system": "axisymmetric-r-z",
        "cell_size_mm": 10.0,
        "cell_volume_weighting": "exact-swept-annular-volume",
        "uncovered_or_under_supported_isotropy_value": 0.0,
        "candidate_workspace_outside_task_region_role": "reported-secondary-not-optimised",
        "clinical_interpretation": False,
    }
    for field, expected in expected_task.items():
        if task.get(field) != expected:
            raise ValueError(f"unexpected Stage-5A task-space setting: {field}")

    fixed = _table(value, "fixed_parameters")
    snapshot = baseline_model_snapshot()
    if fixed.get("youngs_modulus_gpa") != [75.0, 75.0, 75.0]:
        raise ValueError("Young's modulus must remain fixed at 75 GPa")
    if fixed.get("od_mm") != snapshot["od_mm"] or fixed.get("id_mm") != snapshot["id_mm"]:
        raise ValueError("OD and ID must remain at the supervisor nominal values")
    if fixed.get("inner_curved_length_mm") != 0.0 or fixed.get("inner_precurvature_per_m") != 0.0:
        raise ValueError("the supervisor nominal inner wire must remain straight")
    if fixed.get("diameter_role") != "fixed-no-manufacturing-catalogue-or-discrete-stock-set":
        raise ValueError("the reason diameters are fixed must remain explicit")

    variables = value.get("design_variables")
    if not isinstance(variables, list) or len(variables) != len(_VARIABLES):
        raise ValueError("Stage 5A requires exactly seven local design variables")
    for actual, expected in zip(variables, _VARIABLES):
        name, field, index, lower, baseline, upper = expected
        comparison = {
            "name": name,
            "field": field,
            "tube_index": index,
            "lower": lower,
            "baseline": baseline,
            "upper": upper,
            "bound_basis": "local-plus-minus-10-percent-computational-neighbourhood",
        }
        for key, expected_value in comparison.items():
            if actual.get(key) != expected_value:
                raise ValueError(f"unexpected design-variable setting: {name}.{key}")
        if not lower < baseline < upper:
            raise ValueError(f"invalid ordered bounds for {name}")

    constraints = _table(value, "constraints")
    if constraints.get("minimum_common_task_region_coverage") != 0.95:
        raise ValueError("candidate common-region coverage must remain at least 95%")
    if constraints.get("minimum_jacobian_valid_fraction") != 0.999:
        raise ValueError("candidate Jacobian validity must remain at least 99.9%")
    if constraints.get("feasibility_handling") != "constraint-domination-no-finite-penalty-scalarisation":
        raise ValueError("formal multi-objective constraints cannot use an arbitrary penalty")

    objectives = value.get("objectives")
    if not isinstance(objectives, list) or [
        (item.get("name"), item.get("direction")) for item in objectives
    ] != list(_OBJECTIVES):
        raise ValueError("the five ordered Stage-5A objectives must remain frozen")

    sampling = _table(value, "sampling")
    seeds = [
        sampling.get("configuration_training_seed"),
        sampling.get("configuration_validation_seed"),
        sampling.get("ik_training_seed"),
        sampling.get("ik_validation_seed"),
    ]
    if seeds != [42, 1042, 2042, 3042] or len(set(seeds)) != 4:
        raise ValueError("formal training and validation seeds must remain disjoint")
    if sampling.get("rotated_copies_are_independent") is not False:
        raise ValueError("rotated copies cannot be counted as independent")
    if sampling.get("validation_is_never_used_for-search-selection-or-tuning") is not True:
        raise ValueError("validation must remain sealed during search")

    pilot = _table(value, "stage5b_pilot")
    for field, expected in {
        "design_count_including_baseline": 33,
        "low_fidelity_configurations": 15000,
        "low_fidelity_ik_targets": 2000,
        "calibration_design_count": 8,
        "medium_fidelity_configurations": 50000,
        "medium_fidelity_ik_targets": 5000,
    }.items():
        if pilot.get(field) != expected:
            raise ValueError(f"unexpected Stage-5B pilot budget: {field}")
    if pilot.get("minimum_objective_rank_correlation") != 0.8:
        raise ValueError("the pilot rank-stability gate must remain 0.8")
    if pilot.get("full_search_authorized") is not False:
        raise ValueError("Stage 5A cannot authorise the full optimisation")

    search = _table(value, "stage5c_search")
    if search.get("planned_algorithm") != "nsga-ii":
        raise ValueError("the planned formal search must remain NSGA-II")
    if search.get("constraint_handling") != "deb-feasibility-first-constraint-domination":
        raise ValueError("NSGA-II must use feasibility-first constraint domination")
    for field in (
        "population_size",
        "generations",
        "search_configurations_per_design",
        "search_ik_targets_per_design",
        "pareto_reevaluation_maximum_designs",
        "pareto_reevaluation_configurations",
        "pareto_reevaluation_ik_targets",
        "shortlist_maximum_designs_excluding_baseline",
        "shortlist_training_configurations",
        "shortlist_training_ik_targets",
    ):
        _positive_integer(search.get(field), field)
    if not str(search.get("execution_status", "")).startswith("blocked-"):
        raise ValueError("Stage 5C must remain blocked until the pilot passes")

    validation = _table(value, "stage5d_validation")
    if validation.get("validation_configurations") != 50000:
        raise ValueError("Stage 5D must retain 50,000 configuration validations")
    if validation.get("validation_ik_targets") != 100000:
        raise ValueError("Stage 5D must retain 100,000 sealed IK targets")
    if validation.get("selection_must_be_frozen_before_validation") is not True:
        raise ValueError("shortlist selection must precede validation")
    if validation.get("no_post_validation_retuning") is not True:
        raise ValueError("post-validation retuning is forbidden")

    reporting = _table(value, "reporting")
    if reporting.get("single_weighted_winner_without_application_weights") is not False:
        raise ValueError("no unsupported scalar-weighted winner may be reported")
    if reporting.get("pareto_front_required") is not True:
        raise ValueError("the nondominated Pareto front must be reported")

    unresolved = _table(value, "unresolved_decisions")
    for field in (
        "manufacturing_bounds_required_for_global_or_manufacturable_optimum",
        "discrete_available_tube_sizes_not_supplied",
        "anatomical_or_clinical_task_region_not_supplied",
        "precurvature_units_pending_supervisor_confirmation",
    ):
        if unresolved.get(field) is not True:
            raise ValueError(f"unresolved limitation must remain explicit: {field}")


@dataclass(frozen=True)
class OptimizationProtocol:
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


def load_optimization_protocol(
    path: Path | str = DEFAULT_OPTIMISATION_PROTOCOL_PATH,
) -> OptimizationProtocol:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    validate_optimization_protocol(data)
    return OptimizationProtocol(source, deepcopy(data), sha256_mapping(data))


def design_from_protocol_values(
    protocol: OptimizationProtocol,
    values: Mapping[str, float] | None = None,
) -> TubeDesign:
    """Construct one candidate while retaining every fixed Stage-5A quantity."""
    requested = dict(values or {})
    variable_rows = protocol.snapshot["design_variables"]
    allowed = {row["name"] for row in variable_rows}
    unknown = set(requested) - allowed
    if unknown:
        raise KeyError(f"unknown Stage-5A design values: {sorted(unknown)}")
    base = baseline_design()
    arrays = {
        "total_length_mm": base.total_length_mm.copy(),
        "curved_length_mm": base.curved_length_mm.copy(),
        "od_mm": base.od_mm.copy(),
        "id_mm": base.id_mm.copy(),
        "precurvature_per_m": base.precurvature_per_m.copy(),
        "youngs_modulus_gpa": base.youngs_modulus_gpa.copy(),
    }
    for row in variable_rows:
        value = float(requested.get(row["name"], row["baseline"]))
        if value < float(row["lower"]) or value > float(row["upper"]):
            raise ValueError(f"{row['name']} lies outside the frozen local bounds")
        arrays[row["field"]][int(row["tube_index"])] = value
    return TubeDesign(**arrays).validated()


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_optimization_evidence(
    protocol: OptimizationProtocol,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify the baseline evidence and seal the validation data before Stage 5B."""
    root = Path(project_root).resolve()
    evidence = protocol.section("baseline_evidence")
    stage4 = root / evidence["stage4_run"]
    stage4_1 = root / evidence["stage4_1_run"]
    paths = {
        "stage4_manifest": stage4 / "run_manifest.json",
        "stage4_gates": stage4 / "validation_gates.json",
        "task_region": stage4 / "data" / "inner_task_region.npz",
        "training_configurations": stage4 / "data" / "baseline_training_200000.npz",
        "validation_configurations": stage4 / "data" / "baseline_validation_50000.npz",
        "stage4_1_manifest": stage4_1 / "run_manifest.json",
        "stage4_1_gates": stage4_1 / "validation_gates.json",
        "reserved_validation_targets": stage4_1 / "data" / "inner_ik_validation_targets.npz",
        "reserved_validation_results": stage4_1 / "data" / "inner_ik_validation_results.npz",
    }
    hashes = {
        "stage4_manifest": evidence["stage4_manifest_sha256"],
        "stage4_gates": evidence["stage4_gates_sha256"],
        "task_region": evidence["task_region_sha256"],
        "training_configurations": evidence["training_configurations_sha256"],
        "validation_configurations": evidence["validation_configurations_sha256"],
        "stage4_1_manifest": evidence["stage4_1_manifest_sha256"],
        "stage4_1_gates": evidence["stage4_1_gates_sha256"],
        "reserved_validation_targets": evidence["reserved_validation_targets_sha256"],
        "reserved_validation_results": evidence["reserved_validation_results_sha256"],
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required Stage-5A evidence is missing: {path}")
        if _file_sha256(path) != hashes[name]:
            raise ValueError(f"Stage-5A evidence hash mismatch: {name}")

    stage4_manifest = json.loads(paths["stage4_manifest"].read_text(encoding="utf-8"))
    stage4_1_manifest = json.loads(paths["stage4_1_manifest"].read_text(encoding="utf-8"))
    if stage4_manifest.get("run_status") != "completed" or stage4_1_manifest.get("run_status") != "completed":
        raise ValueError("both baseline evidence runs must be completed")
    if stage4_manifest["protocol"]["sha256"] != evidence["stage4_protocol_sha256"]:
        raise ValueError("Stage-4 protocol hash mismatch")
    if stage4_1_manifest["protocol"]["sha256"] != evidence["stage4_1_protocol_sha256"]:
        raise ValueError("Stage-4.1 protocol hash mismatch")
    if stage4_manifest["baseline_parameters"]["sha256"] != baseline_parameter_hash():
        raise ValueError("Stage-4 baseline parameter mismatch")
    if stage4_1_manifest["baseline_parameters"]["sha256"] != baseline_parameter_hash():
        raise ValueError("Stage-4.1 baseline parameter mismatch")

    stage4_gates = json.loads(paths["stage4_gates"].read_text(encoding="utf-8"))
    stage4_1_gates = json.loads(paths["stage4_1_gates"].read_text(encoding="utf-8"))
    convergence_checks = (
        stage4_gates["workspace_converged"],
        stage4_gates["validation_coverage_passed"],
        stage4_gates["isotropy_converged"],
        stage4_gates["jacobian_validity_passed"],
    )
    if not all(convergence_checks):
        raise ValueError("Stage-4 workspace/isotropy evidence gates did not all pass")
    if stage4_1_gates.get("ik_spatial_support_passed") is not True:
        raise ValueError("Stage-4.1 IK spatial-support gate did not pass")

    with np.load(paths["training_configurations"], allow_pickle=False) as data:
        training_ids = data["sample_ids"]
        training_count = len(training_ids)
    with np.load(paths["validation_configurations"], allow_pickle=False) as data:
        validation_ids = data["sample_ids"]
        validation_count = len(validation_ids)
    if training_count != 200000 or validation_count != 50000:
        raise ValueError("sealed configuration datasets have unexpected counts")
    if np.intersect1d(training_ids, validation_ids).size:
        raise ValueError("training and validation configuration IDs overlap")

    with np.load(paths["reserved_validation_targets"], allow_pickle=False) as data:
        target_ids = data["target_ids"]
        target_seed = int(np.asarray(data["seed"]).reshape(()))
        target_split = str(np.asarray(data["split"]).reshape(()).item())
        task_region_id = str(np.asarray(data["task_region_id"]).reshape(()).item())
    with np.load(paths["reserved_validation_results"], allow_pickle=False) as data:
        result_ids = data["target_ids"]
    if len(target_ids) != 100000 or len(np.unique(target_ids)) != len(target_ids):
        raise ValueError("the sealed validation target set must contain 100,000 unique IDs")
    if target_seed != 3042 or target_split != "validation":
        raise ValueError("the sealed validation target split/seed is not the agreed one")
    if not np.array_equal(target_ids, result_ids):
        raise ValueError("sealed target and baseline-result IDs do not match")

    return {
        "baseline_parameter_sha256": baseline_parameter_hash(),
        "stage4_convergence_gates_passed": True,
        "stage4_1_ik_support_gate_passed": True,
        "stage4_1_supported_task_volume_fraction": float(
            stage4_1_gates["ik_p95_supported_task_volume_fraction"]
        ),
        "training_configuration_count": training_count,
        "validation_configuration_count": validation_count,
        "training_validation_configuration_ids_disjoint": True,
        "reserved_validation_target_count": len(target_ids),
        "reserved_validation_target_seed": target_seed,
        "reserved_validation_target_split": target_split,
        "frozen_task_region_id": task_region_id,
        "stage5b_pilot_may_begin": True,
        "stage5c_full_optimisation_may_begin": False,
    }


def build_optimization_manifest(
    protocol: OptimizationProtocol,
    *,
    evidence_audit: Mapping[str, Any],
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return {
        "manifest_schema_version": 1,
        "run_kind": "stage-5a-optimisation-protocol-preflight",
        "run_status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": os.path.relpath(protocol.path, root),
            "sha256": protocol.sha256,
            "snapshot": protocol.snapshot,
        },
        "baseline_parameters": {
            "sha256": baseline_parameter_hash(),
            "snapshot": baseline_model_snapshot(),
        },
        "evidence_audit": dict(evidence_audit),
        "code": capture_code_provenance(root),
        "runtime": runtime_provenance(),
    }


def write_optimization_manifest(path: Path | str, value: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
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
