"""Checks for the frozen Stage-4.1 inner-tip IK-support completion."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_protocol import load_analysis_protocol, validate_analysis_protocol  # noqa: E402
from tools.run_stage4_1_ik_support_extension import _support_gates  # noqa: E402


def run_checks() -> None:
    protocol = load_analysis_protocol(
        PROJECT_ROOT / "config" / "analysis_protocol_stage4_1_support.toml"
    )
    sampling = protocol.section("sampling")
    extension = protocol.section("extension")
    assert extension["extension_type"] == (
        "controlled-primary-endpoint-ik-support-completion"
    )
    assert sampling["ik_target_candidate_configurations"] == 600000
    assert sampling["ik_primary_validation_targets"] == 100000
    assert extension["ik_spatial_reportable_volume_minimum"] == 0.9
    assert extension["post_hoc_objective_or_threshold_change"] is False

    malformed = protocol.snapshot
    malformed["sampling"]["ik_primary_validation_targets"] = 99999
    try:
        validate_analysis_protocol(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("the frozen Stage-4.1 target count must be enforced")

    residual_map = SimpleNamespace(
        p95_residual_mm=np.asarray([[1.0, 1.0], [1.0, np.nan]]),
        sample_count=np.asarray([[110, 120], [130, 20]], dtype=np.int32),
    )
    task_region = SimpleNamespace(
        cell_volume_mm3=np.asarray([[1.0, 1.0], [8.0, 1.0]]),
        occupied=np.ones((2, 2), dtype=bool),
        occupied_cell_count=4,
    )
    gates = _support_gates(residual_map, task_region, protocol=protocol)
    assert np.isclose(gates["ik_p95_supported_task_volume_fraction"], 10.0 / 11.0)
    assert gates["ik_spatial_support_passed"]
    assert gates["ik_p95_supported_cells"] == 3
    assert gates["ik_minimum_nonzero_cell_support"] == 20
    assert gates["ik_maximum_cell_support"] == 130
    assert gates["optimisation_may_begin"] is False


if __name__ == "__main__":
    run_checks()
    print("Stage-4.1 IK-support checks passed.")
