"""Reproduce the lightweight numerical checks used to freeze Stage-2 settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_protocol import baseline_parameter_hash, load_analysis_protocol  # noqa: E402
from ctr_design_analysis import (  # noqa: E402
    baseline_design,
    calculate_ik_error_map,
    position_jacobians_all_endpoints,
    positional_isotropy,
    summarise_ik_errors,
    with_design_value,
)
from ctr_inverse_kinematics import ConstrainedTipIK  # noqa: E402
from ctr_sampling import canonical_configuration_bank  # noqa: E402
from ctr_task_space import (  # noqa: E402
    freeze_baseline_task_region,
    select_stratified_canonical_targets,
)


def _forward_tips(parameters: dict, deployment: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    solver = ConstrainedTipIK(parameters, model_points_per_section=2)
    return np.asarray(
        [
            solver.forward_endpoint_mm(deployment_row, rotation_row, 0)
            for deployment_row, rotation_row in zip(deployment, rotation)
        ],
        dtype=float,
    )


def _jacobian_pilot(protocol) -> dict:
    settings = protocol.section("jacobian")
    base = baseline_design()
    designs = (
        ("baseline", base),
        (
            "middle-length-plus-5-percent",
            with_design_value(base, "middle_total_length_mm", 178.5),
        ),
        (
            "outer-precurvature-plus-10-percent",
            with_design_value(base, "outer_precurvature_per_m", 15.444),
        ),
    )
    bank = canonical_configuration_bank(
        int(settings["pilot_independent_configurations"]),
        seed=1042,
        split="stage2-jacobian-validation",
    )
    translation_candidates = settings["translation_step_candidates_mm"]
    rotation_candidates = settings["rotation_step_candidates_rad"]
    selected_translation = float(settings["translation_step_mm"])
    selected_rotation = float(settings["rotation_step_rad"])
    translation_index = translation_candidates.index(selected_translation)
    rotation_index = rotation_candidates.index(selected_rotation)
    refined_translation = float(translation_candidates[translation_index + 1])
    refined_rotation = float(rotation_candidates[rotation_index + 1])
    records = []
    for name, design in designs:
        deployment = bank.decode_deployments(design.total_length_mm * 1e-3)
        rotation = bank.rotation_rad
        solver = ConstrainedTipIK(design.to_parameters(), model_points_per_section=2)
        relative_change = []
        isotropy_change = []
        for deployment_row, rotation_row in zip(deployment, rotation):
            selected = position_jacobians_all_endpoints(
                solver,
                deployment_row,
                rotation_row,
                translation_step_mm=selected_translation,
                rotation_step_rad=selected_rotation,
            )
            refined = position_jacobians_all_endpoints(
                solver,
                deployment_row,
                rotation_row,
                translation_step_mm=refined_translation,
                rotation_step_rad=refined_rotation,
            )
            for endpoint in range(3):
                reference = refined.scaled_jacobians[endpoint]
                relative_change.append(
                    np.linalg.norm(
                        selected.scaled_jacobians[endpoint] - reference
                    )
                    / max(np.linalg.norm(reference), 1e-12)
                )
                isotropy_change.append(
                    abs(
                        positional_isotropy(
                            selected.scaled_jacobians[endpoint]
                        )[0]
                        - positional_isotropy(reference)[0]
                    )
                )
        records.append(
            {
                "design": name,
                "p95_relative_jacobian_change": float(
                    np.quantile(relative_change, 0.95)
                ),
                "maximum_relative_jacobian_change": float(np.max(relative_change)),
                "p95_absolute_isotropy_change": float(
                    np.quantile(isotropy_change, 0.95)
                ),
                "maximum_absolute_isotropy_change": float(
                    np.max(isotropy_change)
                ),
            }
        )
    return {
        "selected_translation_step_mm": selected_translation,
        "selected_rotation_step_rad": selected_rotation,
        "refined_translation_step_mm": refined_translation,
        "refined_rotation_step_rad": refined_rotation,
        "independent_configurations": bank.independent_count,
        "endpoints_per_configuration": 3,
        "records": records,
    }


def _ik_pilot(protocol) -> dict:
    sampling = protocol.section("sampling")
    spatial = protocol.section("spatial")
    settings = protocol.section("ik")
    design = baseline_design()
    parameters = design.to_parameters()
    lengths_m = design.total_length_mm * 1e-3

    discovery = canonical_configuration_bank(
        int(settings["pilot_task_discovery_configurations"]),
        seed=int(sampling["training_seed"]),
        split="stage2-pilot-task-discovery",
    )
    discovery_deployment = discovery.decode_deployments(lengths_m)
    discovery_points = _forward_tips(
        parameters,
        discovery_deployment,
        discovery.rotation_rad,
    )
    region = freeze_baseline_task_region(
        discovery_points,
        discovery.sample_ids,
        target_tube=0,
        baseline_parameter_sha256=baseline_parameter_hash(),
        source_dataset_id="stage2-pilot-task-discovery",
        radial_bin_mm=float(spatial["provisional_cell_size_mm"]),
        z_bin_mm=float(spatial["provisional_cell_size_mm"]),
        minimum_occupancy_samples=int(
            spatial["occupancy_minimum_independent_samples"]
        ),
    )

    candidates = canonical_configuration_bank(
        int(settings["pilot_target_candidate_configurations"]),
        seed=int(sampling["validation_seed"]),
        split="stage2-pilot-target-candidates",
    )
    candidate_deployment = candidates.decode_deployments(lengths_m)
    candidate_points = _forward_tips(
        parameters,
        candidate_deployment,
        candidates.rotation_rad,
    )
    targets = select_stratified_canonical_targets(
        region,
        candidate_points,
        candidate_deployment,
        candidates.rotation_rad,
        candidates.sample_ids,
        target_count=int(settings["pilot_target_count"]),
        split="validation",
        seed=int(sampling["validation_seed"]),
        source_dataset_id="stage2-pilot-target-candidates",
    )
    records = []
    for maximum_iterations in (12, 20, 40, 60):
        for tolerance in settings["solver_tolerance_candidates_mm"]:
            errors = calculate_ik_error_map(
                parameters,
                targets.targets_mm,
                solver_tolerance_mm=float(tolerance),
                max_iterations=maximum_iterations,
                canonical_sample_ids=targets.target_ids,
                target_weight_mm3=targets.target_weight_mm3,
            )
            metrics = summarise_ik_errors(
                errors,
                evaluation_threshold_mm=float(settings["evaluation_threshold_mm"]),
            )
            metrics.pop("ik_success_rate", None)
            records.append(
                {
                    "maximum_iterations": maximum_iterations,
                    "solver_tolerance_mm": float(tolerance),
                    **metrics,
                }
            )
    return {
        "task_region_id": region.task_region_id,
        "occupied_task_cells": region.occupied_cell_count,
        "target_set_id": targets.target_set_id,
        "independent_targets": targets.independent_target_count,
        "represented_target_cells": int(np.count_nonzero(targets.targets_per_cell)),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    protocol = load_analysis_protocol()
    report = {
        "report_schema": "ctr-stage2-method-validation-v1",
        "protocol_sha256": protocol.sha256,
        "jacobian": _jacobian_pilot(protocol),
        "ik": _ik_pilot(protocol),
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        destination = arguments.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        destination.write_text(payload, encoding="utf-8")
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
