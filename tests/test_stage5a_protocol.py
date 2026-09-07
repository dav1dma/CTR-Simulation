"""Checks for the frozen Stage-5A optimisation methodology."""

from itertools import product
import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization_protocol import (  # noqa: E402
    audit_optimization_evidence,
    design_from_protocol_values,
    load_optimization_protocol,
    validate_optimization_protocol,
)


def run_checks(*, with_evidence: bool = False) -> None:
    protocol = load_optimization_protocol()
    snapshot = protocol.snapshot
    assert snapshot["methodology_stage"] == "5A"
    assert len(snapshot["design_variables"]) == 7
    assert len(snapshot["objectives"]) == 5
    assert snapshot["fixed_parameters"]["youngs_modulus_gpa"] == [75.0] * 3
    assert snapshot["fixed_parameters"]["od_mm"] == [0.5, 0.7, 0.9]
    assert snapshot["fixed_parameters"]["id_mm"] == [0.0, 0.62, 0.8]
    assert snapshot["stage5b_pilot"]["full_search_authorized"] is False
    assert snapshot["stage5c_search"]["planned_algorithm"] == "nsga-ii"
    assert snapshot["stage5d_validation"]["no_post_validation_retuning"] is True

    variable_names = [row["name"] for row in snapshot["design_variables"]]
    assert not any("od_mm" in name or "id_mm" in name for name in variable_names)
    assert not any("youngs_modulus" in name for name in variable_names)

    baseline = design_from_protocol_values(protocol)
    assert np.array_equal(baseline.total_length_mm, [350.0, 170.0, 80.0])
    assert np.array_equal(baseline.curved_length_mm, [0.0, 90.0, 65.0])
    assert np.array_equal(baseline.precurvature_per_m, [0.0, 19.12, 14.04])

    variables = snapshot["design_variables"]
    checked = 0
    for choices in product((0, 1), repeat=len(variables)):
        values = {
            row["name"]: row["lower"] if choice == 0 else row["upper"]
            for row, choice in zip(variables, choices)
        }
        candidate = design_from_protocol_values(protocol, values)
        candidate.validated()
        checked += 1
    assert checked == 128

    try:
        design_from_protocol_values(protocol, {"inner_total_length_mm": 500.0})
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-bounds Stage-5A candidates must be rejected")

    malformed = snapshot
    malformed["constraints"]["minimum_common_task_region_coverage"] = 0.9
    try:
        validate_optimization_protocol(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("the frozen common-region coverage constraint must be enforced")

    malformed = protocol.snapshot
    malformed["design_variables"][0]["upper"] = 400.0
    try:
        validate_optimization_protocol(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("the local Stage-5A variable bounds must remain frozen")

    if not with_evidence:
        print("Local results audit skipped; pass --with-evidence to audit formal run datasets.")
        return

    evidence = audit_optimization_evidence(protocol, PROJECT_ROOT)
    assert evidence["stage4_convergence_gates_passed"]
    assert evidence["stage4_1_ik_support_gate_passed"]
    assert evidence["training_configuration_count"] == 200000
    assert evidence["validation_configuration_count"] == 50000
    assert evidence["reserved_validation_target_count"] == 100000
    assert evidence["training_validation_configuration_ids_disjoint"]
    assert evidence["stage5b_pilot_may_begin"]
    assert not evidence["stage5c_full_optimisation_may_begin"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-evidence", action="store_true",
                        help="Also audit the formal datasets in the local results directory.")
    run_checks(with_evidence=parser.parse_args().with_evidence)
    print("Stage-5A optimisation-protocol checks passed.")
