"""Complete Stage-4 inner-tip 10 mm IK-map support without changing thresholds."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_protocol import (  # noqa: E402
    baseline_parameter_hash,
    build_run_manifest,
    load_analysis_protocol,
    write_run_manifest,
)
from ctr_design_analysis import baseline_design, summarise_ik_errors  # noqa: E402
from ctr_sampling import canonical_configuration_bank  # noqa: E402
from ctr_spatial_analysis import (  # noqa: E402
    AxisymmetricGrid,
    aggregate_axisymmetric_residual,
)
from ctr_task_space import (  # noqa: E402
    FrozenTaskRegion,
    select_stratified_canonical_targets,
)
from tools.run_stage3_baseline_evaluation import (  # noqa: E402
    _forward_configuration_bank,
    _parallel_ik,
    _plot_radial_map,
    _save_ik,
    _save_npz,
    _save_target_set,
    _write_json,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


def _load_task_region(path: Path) -> FrozenTaskRegion:
    with np.load(path, allow_pickle=False) as data:
        grid = AxisymmetricGrid(
            radial_edges_mm=data["radial_edges_mm"].copy(),
            z_edges_mm=data["z_edges_mm"].copy(),
            cell_volume_mm3=data["cell_volume_mm3"].copy(),
        )
        return FrozenTaskRegion(
            task_region_id=_scalar_text(data["task_region_id"]),
            grid=grid,
            target_tube=int(np.asarray(data["target_tube"]).reshape(())),
            occupied=data["occupied"].astype(bool),
            discovery_count_per_cell=data["discovery_count_per_cell"].copy(),
            cell_volume_mm3=data["cell_volume_mm3"].copy(),
            baseline_parameter_sha256=_scalar_text(data["baseline_parameter_sha256"]),
            source_dataset_id=_scalar_text(data["source_dataset_id"]),
            minimum_occupancy_samples=1,
        )


def _verify_stage4_source(protocol) -> tuple[Path, FrozenTaskRegion, dict, dict, dict]:
    extension = protocol.section("extension")
    source_run = PROJECT_ROOT / extension["source_stage4_run"]
    paths = {
        "manifest": source_run / "run_manifest.json",
        "task_region": source_run / "data" / "inner_task_region.npz",
        "workspace_metrics": source_run / "workspace_validation_metrics.json",
        "validation_gates": source_run / "validation_gates.json",
        "ik_metrics": source_run / "ik_validation_metrics.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"required Stage-4 source is missing: {path}")
    expected_hashes = {
        "manifest": extension["source_run_manifest_sha256"],
        "task_region": extension["source_task_region_sha256"],
        "workspace_metrics": extension["source_workspace_metrics_sha256"],
        "validation_gates": extension["source_validation_gates_sha256"],
        "ik_metrics": extension["source_ik_metrics_sha256"],
    }
    for name, expected in expected_hashes.items():
        if _file_sha256(paths[name]) != expected:
            raise ValueError(f"the source Stage-4 {name} hash does not match")
    manifest = _load_json(paths["manifest"])
    if manifest.get("run_status") != "completed":
        raise ValueError("the source Stage-4 run is not marked completed")
    if manifest["protocol"]["sha256"] != extension["source_protocol_sha256"]:
        raise ValueError("the source Stage-4 protocol hash does not match")
    if manifest["baseline_parameters"]["sha256"] != baseline_parameter_hash():
        raise ValueError("the source Stage-4 baseline parameters do not match")
    gates = _load_json(paths["validation_gates"])
    previous_support = float(gates["ik_p95_supported_task_volume_fraction"])
    if not np.isclose(
        previous_support,
        float(extension["previous_supported_task_volume_fraction"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("the recorded Stage-4 support fraction does not match")
    task_region = _load_task_region(paths["task_region"])
    if task_region.target_tube != 0:
        raise ValueError("Stage 4.1 requires the frozen inner-tip task region")
    if task_region.baseline_parameter_sha256 != baseline_parameter_hash():
        raise ValueError("the frozen task region uses different baseline parameters")
    return (
        source_run,
        task_region,
        _load_json(paths["workspace_metrics"]),
        gates,
        _load_json(paths["ik_metrics"]),
    )


def _support_gates(residual_map, task_region: FrozenTaskRegion, *, protocol) -> dict:
    extension = protocol.section("extension")
    supported = np.isfinite(residual_map.p95_residual_mm)
    supported_volume = float(np.sum(task_region.cell_volume_mm3[supported]))
    task_volume = float(np.sum(task_region.cell_volume_mm3[task_region.occupied]))
    support_fraction = supported_volume / max(task_volume, 1e-15)
    support_minimum = float(extension["ik_spatial_reportable_volume_minimum"])
    return {
        "previous_ik_p95_supported_task_volume_fraction": float(
            extension["previous_supported_task_volume_fraction"]
        ),
        "ik_p95_supported_task_volume_fraction": support_fraction,
        "ik_spatial_reportable_volume_minimum": support_minimum,
        "ik_p95_supported_cells": int(np.count_nonzero(supported)),
        "ik_task_region_cells": int(task_region.occupied_cell_count),
        "ik_represented_cells": int(np.count_nonzero(residual_map.sample_count)),
        "ik_minimum_targets_for_p95": int(
            extension["ik_spatial_support_minimum_independent_targets"]
        ),
        "ik_minimum_nonzero_cell_support": int(
            np.min(residual_map.sample_count[residual_map.sample_count > 0])
        ),
        "ik_median_nonzero_cell_support": float(
            np.median(residual_map.sample_count[residual_map.sample_count > 0])
        ),
        "ik_maximum_cell_support": int(np.max(residual_map.sample_count)),
        "ik_spatial_support_passed": support_fraction >= support_minimum,
        "optimisation_may_begin": False,
        "optimisation_gate_note": (
            "Stage 4.1 only completes baseline map support. Optimisation requires "
            "review and a separately frozen optimisation protocol."
        ),
    }


def _plot_support_comparison(
    previous_gates: dict,
    current_gates: dict,
    previous_ik: dict,
    current_ik: dict,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), constrained_layout=True)
    labels = ["Stage 4\n75,000", "Stage 4.1\n100,000"]
    coverage = [
        previous_gates["ik_p95_supported_task_volume_fraction"],
        current_gates["ik_p95_supported_task_volume_fraction"],
    ]
    axes[0].bar(labels, coverage, color=["#9ecae1", "#3182bd"])
    axes[0].axhline(
        current_gates["ik_spatial_reportable_volume_minimum"],
        color="#d7301f",
        linestyle="--",
        label="predeclared minimum",
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Supported physical task volume")
    axes[0].set_title("10 mm p95 map support")
    axes[0].legend(fontsize=8)
    axes[1].bar(
        labels,
        [
            previous_ik["ik_within_evaluation_threshold_rate"],
            current_ik["ik_within_evaluation_threshold_rate"],
        ],
        color=["#a1d99b", "#31a354"],
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Physical-volume fraction")
    axes[1].set_title("IK residual ≤ 0.5 mm")
    axes[2].bar(
        labels,
        [previous_ik["ik_p95_error_mm"], current_ik["ik_p95_error_mm"]],
        color=["#fdae6b", "#e6550d"],
    )
    axes[2].set_ylabel("Numerical residual (mm)")
    axes[2].set_title("Global volume-weighted p95")
    figure.suptitle("Stage-4.1 inner-tip IK-support completion")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _write_report(
    path: Path,
    *,
    gates: dict,
    metrics: dict,
    previous_metrics: dict,
    target_count: int,
) -> None:
    outcome = "passed" if gates["ik_spatial_support_passed"] else "did not pass"
    text = f"""# Stage 4.1 inner-tip IK-support report

## Outcome

The controlled support-completion run {outcome} the unchanged 90% physical-volume
coverage requirement for the 10 mm p95 and failure-fraction maps. Optimisation
remains disabled pending methodological review and a separate frozen protocol.

## Fixed methodology

- Frozen Stage-4 inner-tip task region: {gates['ik_task_region_cells']} occupied cells
- Independent target candidates: 600,000
- Independently selected and solved IK targets: {target_count:,}
- Spatial cell size: 10 mm
- Minimum support for p95/failure statistics: n >= 100 per cell
- Solver stopping tolerance: 0.01 mm
- Separate evaluation threshold: 0.5 mm
- Fixed fully retracted, zero-rotation IK initial state

No tube parameter, solver setting, spatial threshold or performance threshold was
changed after Stage 4. The 100,000 targets form a newly selected set from the larger
candidate bank; they are not rotated copies and are not claimed as a simple 25,000
target suffix to the earlier set.

## Spatial support

| Quantity | Stage 4 | Stage 4.1 |
|---|---:|---:|
| Independent IK targets | 75,000 | {target_count:,} |
| Supported task-space volume | {gates['previous_ik_p95_supported_task_volume_fraction']:.2%} | {gates['ik_p95_supported_task_volume_fraction']:.2%} |
| Cells supporting p95/failure maps | 562 | {gates['ik_p95_supported_cells']} |
| Required physical-volume coverage | 90% | 90% |

## Global numerical IK results

| Quantity | Stage 4 | Stage 4.1 |
|---|---:|---:|
| Residual <= 0.5 mm | {previous_metrics['ik_within_evaluation_threshold_rate']:.2%} | {metrics['ik_within_evaluation_threshold_rate']:.2%} |
| Solver reached 0.01 mm | {previous_metrics['ik_solver_convergence_rate']:.2%} | {metrics['ik_solver_convergence_rate']:.2%} |
| Median residual | {previous_metrics['ik_median_error_mm']:.6f} mm | {metrics['ik_median_error_mm']:.6f} mm |
| Mean residual | {previous_metrics['ik_mean_error_mm']:.3f} mm | {metrics['ik_mean_error_mm']:.3f} mm |
| Volume-weighted p95 residual | {previous_metrics['ik_p95_error_mm']:.3f} mm | {metrics['ik_p95_error_mm']:.3f} mm |

These are numerical fixed-start inverse-model residuals, not measured positioning
accuracy, force capability, clinical success or safety limits.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "config" / "analysis_protocol_stage4_1_support.toml",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--ik-chunk-size", type=int, default=200)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.chunk_size < 1 or arguments.ik_chunk_size < 1:
        raise ValueError("worker and chunk settings must be positive")

    protocol = load_analysis_protocol(arguments.protocol)
    extension = protocol.section("extension")
    if extension["extension_type"] != "controlled-primary-endpoint-ik-support-completion":
        raise ValueError("the Stage-4.1 runner requires its frozen support protocol")
    sampling = protocol.section("sampling")
    spatial = protocol.section("spatial")
    ik_settings = protocol.section("ik")
    design = baseline_design()
    parameters = design.to_parameters()
    lengths_m = design.total_length_mm * 1e-3

    commit_result = subprocess.run(
        ("git", "rev-parse", "--short=8", "HEAD"),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = commit_result.stdout.strip() or "nogit"
    output_root = (
        PROJECT_ROOT / protocol.section("versioning")["formal_output_root"]
        if arguments.output_root is None
        else arguments.output_root.resolve()
    )
    output = output_root / (
        f"baseline_ik_support_{datetime.now().strftime('%Y%m%d')}_"
        f"{protocol.sha256[:8]}_{commit[:8]}"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty formal run: {output}")
    data_directory = output / "data"
    figure_directory = output / "figures"
    data_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        protocol,
        run_kind="stage-4.1-inner-tip-ik-support-completion",
        effective_run_settings={
            "workers": arguments.workers,
            "configuration_chunk_size": arguments.chunk_size,
            "ik_chunk_size": arguments.ik_chunk_size,
            "output_directory": str(output.relative_to(PROJECT_ROOT)),
        },
        project_root=PROJECT_ROOT,
    )
    manifest_path = output / "run_manifest.json"
    write_run_manifest(manifest_path, manifest)

    def progress(stage: str) -> None:
        manifest["progress_stage"] = stage
        write_run_manifest(manifest_path, manifest)
        print(f"\n[{stage}]", flush=True)

    try:
        progress("verify frozen Stage-4 task region and evidence")
        source_run, task_region, workspace_metrics, previous_gates, previous_ik_values = (
            _verify_stage4_source(protocol)
        )
        previous_ik = previous_ik_values["inner"]
        manifest["verified_source_run"] = str(source_run.relative_to(PROJECT_ROOT))

        progress("generate 600,000 independent IK target candidates")
        candidate_bank = canonical_configuration_bank(
            int(sampling["ik_target_candidate_configurations"]),
            seed=int(sampling["ik_target_validation_seed"]),
            split="stage4-1-ik-target-validation-candidates",
        )
        candidates = _forward_configuration_bank(
            parameters,
            candidate_bank,
            lengths_m,
            workers=arguments.workers,
            chunk_size=arguments.chunk_size,
            progress_label="Stage-4.1 inner target candidates",
        )
        targets = select_stratified_canonical_targets(
            task_region,
            candidates["endpoint_positions_mm"][:, 0],
            candidates["deployment_m"],
            candidates["rotation_rad"],
            candidates["sample_ids"],
            target_count=int(sampling["ik_primary_validation_targets"]),
            split="validation",
            seed=int(sampling["ik_target_validation_seed"]),
            source_dataset_id="stage4-1-ik-target-validation-candidates-600000",
        )
        _save_target_set(data_directory / "inner_ik_validation_targets.npz", targets)

        progress("solve 100,000 fixed-retraction inner-tip IK targets")
        errors = _parallel_ik(
            parameters,
            targets,
            target_tube=0,
            solver_tolerance_mm=float(ik_settings["solver_tolerance_mm"]),
            max_iterations=int(ik_settings["max_iterations"]),
            workers=arguments.workers,
            chunk_size=arguments.ik_chunk_size,
            progress_label="Stage-4.1 inner IK",
        )
        _save_ik(data_directory / "inner_ik_validation_results.npz", errors)
        metrics = summarise_ik_errors(
            errors,
            evaluation_threshold_mm=float(ik_settings["evaluation_threshold_mm"]),
        )
        metrics.pop("ik_success_rate", None)
        metrics.update(
            {
                "independent_target_count": targets.independent_target_count,
                "represented_target_cells": int(np.count_nonzero(targets.targets_per_cell)),
                "task_region_cells": task_region.occupied_cell_count,
            }
        )
        _write_json(output / "ik_validation_metrics.json", {"inner": metrics})

        residual_map = aggregate_axisymmetric_residual(
            errors.targets_mm,
            errors.residual_mm,
            canonical_target_ids=errors.canonical_sample_ids,
            grid=task_region.grid,
            evaluation_threshold_mm=float(ik_settings["evaluation_threshold_mm"]),
            minimum_median_samples=int(spatial["minimum_samples_median"]),
            minimum_failure_or_p95_samples=int(spatial["minimum_samples_failure_or_p95"]),
        )
        _save_npz(
            data_directory / "inner_ik_residual_10mm.npz",
            radial_edges_mm=residual_map.grid.radial_edges_mm,
            z_edges_mm=residual_map.grid.z_edges_mm,
            maximum_residual_mm=residual_map.maximum_residual_mm,
            median_residual_mm=residual_map.median_residual_mm,
            p95_residual_mm=residual_map.p95_residual_mm,
            failure_fraction=residual_map.failure_fraction,
            sample_count=residual_map.sample_count,
            evaluation_threshold_mm=np.asarray(residual_map.evaluation_threshold_mm),
        )
        finite_residual = errors.residual_mm[np.isfinite(errors.residual_mm)]
        residual_upper = max(0.5, float(np.quantile(finite_residual, 0.95)))
        for label, values, colour_label, limits, cmap in (
            ("median", residual_map.median_residual_mm, "Median numerical IK residual (mm)", (0.0, residual_upper), "magma"),
            ("p95", residual_map.p95_residual_mm, "95th-percentile numerical IK residual (mm)", (0.0, residual_upper), "magma"),
            ("failure_fraction", residual_map.failure_fraction, "Fraction above 0.5 mm", (0.0, 1.0), "inferno"),
        ):
            _plot_radial_map(
                residual_map.grid,
                values,
                figure_directory / f"inner_ik_{label}.png",
                title=f"Inner-tip IK {label.replace('_', ' ')}",
                colour_label=colour_label,
                cmap=cmap,
                limits=limits,
            )
        _plot_radial_map(
            residual_map.grid,
            np.where(residual_map.sample_count > 0, residual_map.sample_count, np.nan),
            figure_directory / "inner_ik_target_support.png",
            title="Inner-tip IK target support (Stage 4.1)",
            colour_label="Independent targets per cell",
            cmap="plasma",
            logarithmic=True,
        )

        gates = _support_gates(residual_map, task_region, protocol=protocol)
        _write_json(output / "validation_gates.json", gates)
        _write_json(output / "source_workspace_metrics.json", workspace_metrics)
        _plot_support_comparison(
            previous_gates,
            gates,
            previous_ik,
            metrics,
            figure_directory / "stage4_1_support_comparison.png",
        )
        _write_report(
            output / "STAGE4_1_SUPPORT_REPORT.md",
            gates=gates,
            metrics=metrics,
            previous_metrics=previous_ik,
            target_count=targets.independent_target_count,
        )

        manifest["run_status"] = "completed"
        manifest["progress_stage"] = "completed"
        manifest["validation_gates"] = gates
        manifest["formal_result_note"] = (
            "Stage-4.1 inner-tip IK-support completion finished. Optimisation "
            "remains disabled pending review and a separately frozen protocol."
        )
        write_run_manifest(manifest_path, manifest)
        print(f"\nCompleted Stage-4.1 support run: {output}", flush=True)
    except Exception as error:
        manifest["run_status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        write_run_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
