"""Checks for the frozen Stage-4 convergence extension."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_protocol import load_analysis_protocol, validate_analysis_protocol  # noqa: E402
from ctr_sampling import canonical_configuration_bank  # noqa: E402
from tools.run_stage4_convergence_extension import (  # noqa: E402
    _extension_gates,
    _merge_prefix_and_tail,
    _tail_bank,
    _verify_prefix,
)


def run_checks() -> None:
    protocol = load_analysis_protocol(
        PROJECT_ROOT / "config" / "analysis_protocol_stage4_extension.toml"
    )
    sampling = protocol.section("sampling")
    extension = protocol.section("extension")
    assert protocol.snapshot["methodology_stage"] == 4
    assert sampling["baseline_training_configurations"] == 200000
    assert sampling["baseline_validation_configurations"] == 50000
    assert sampling["ik_target_candidate_configurations"] == 300000
    assert sampling["ik_primary_validation_targets"] == 75000
    assert extension["configuration_comparison_counts"] == [150000, 200000]
    assert extension["post_hoc_objective_or_threshold_change"] is False

    malformed = protocol.snapshot
    malformed["extension"]["ik_spatial_reportable_volume_minimum"] = 0.8
    try:
        validate_analysis_protocol(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("the frozen Stage-4 support threshold must be enforced")

    lengths_m = np.asarray([0.350, 0.170, 0.080])
    bank = canonical_configuration_bank(
        12,
        seed=42,
        split="stage3-baseline-training",
    )
    prefix_bank = bank.prefix(5)
    prefix = {
        "sample_ids": prefix_bank.sample_ids,
        "deployment_m": prefix_bank.decode_deployments(lengths_m).astype(np.float32),
        "rotation_rad": prefix_bank.rotation_rad.astype(np.float32),
        "jacobian_valid": np.ones(5, dtype=bool),
    }
    assert _verify_prefix(prefix, bank, lengths_m) == 5
    tail_bank = _tail_bank(bank, 5)
    tail = {
        "sample_ids": tail_bank.sample_ids,
        "deployment_m": tail_bank.decode_deployments(lengths_m).astype(np.float32),
        "rotation_rad": tail_bank.rotation_rad.astype(np.float32),
        "jacobian_valid": np.ones(7, dtype=bool),
    }
    merged = _merge_prefix_and_tail(prefix, tail)
    assert np.array_equal(merged["sample_ids"], bank.sample_ids)
    assert len(merged["sample_ids"]) == 12

    convergence_rows = []
    for count, volume, mean, median, p10 in (
        (150000, 100.0, 0.50, 0.51, 0.38),
        (200000, 101.0, 0.505, 0.514, 0.387),
    ):
        convergence_rows.append(
            {
                "endpoint": "inner",
                "independent_configurations": count,
                "cell_size_mm": 10.0,
                "occupied_volume_cm3": volume,
                "volume_weighted_mean": mean,
                "volume_weighted_median": median,
                "volume_weighted_p10": p10,
            }
        )
    validation_metrics = {
        "inner": {"validation_coverage_of_training_workspace": 0.97}
    }
    residual_map = SimpleNamespace(
        p95_residual_mm=np.asarray([[1.0, 1.0], [1.0, np.nan]])
    )
    task_region = SimpleNamespace(
        cell_volume_mm3=np.asarray([[1.0, 1.0], [8.0, 1.0]]),
        occupied=np.ones((2, 2), dtype=bool),
        occupied_cell_count=4,
    )
    gates = _extension_gates(
        convergence_rows,
        validation_metrics,
        residual_map,
        task_region,
        protocol=protocol,
        training_valid_fraction=1.0,
        validation_valid_fraction=1.0,
    )
    assert gates["ik_p95_supported_task_volume_fraction"] == 10.0 / 11.0
    assert gates["extension_passed"]


if __name__ == "__main__":
    run_checks()
    print("Stage-4 convergence-extension checks passed.")
