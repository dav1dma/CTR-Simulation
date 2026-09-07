"""Run the frozen Stage-4 inner-tip convergence and IK-support extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

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
from ctr_sampling import (  # noqa: E402
    CanonicalConfigurationBank,
    canonical_configuration_bank,
)
from ctr_spatial_analysis import (  # noqa: E402
    aggregate_axisymmetric_residual,
    summarise_axisymmetric_scalar,
)
from ctr_task_space import (  # noqa: E402
    freeze_baseline_task_region,
    select_stratified_canonical_targets,
)
from tools.run_stage3_baseline_evaluation import (  # noqa: E402
    _convergence_analysis,
    _evaluate_configuration_bank,
    _forward_configuration_bank,
    _parallel_ik,
    _plot_convergence,
    _plot_radial_map,
    _plot_workspace_sweep,
    _save_ik,
    _save_npz,
    _save_target_set,
    _save_task_region,
    _write_csv,
    _write_json,
)


SOURCE_TRAINING_NAME = "baseline_training_100000.npz"
SOURCE_VALIDATION_NAME = "baseline_validation_25000.npz"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def _tail_bank(bank: CanonicalConfigurationBank, start: int) -> CanonicalConfigurationBank:
    if start < 1 or start >= bank.independent_count:
        raise ValueError("tail start must lie inside the configuration bank")
    return CanonicalConfigurationBank(
        sample_ids=bank.sample_ids[start:].copy(),
        split=bank.split,
        seed=bank.seed,
        deployment_fractions=bank.deployment_fractions[start:].copy(),
        relative_rotation_rad=bank.relative_rotation_rad[start:].copy(),
        sampler_version=bank.sampler_version,
    )


def _verify_prefix(
    source: dict[str, np.ndarray],
    full_bank: CanonicalConfigurationBank,
    total_lengths_m: np.ndarray,
) -> int:
    count = len(source["sample_ids"])
    if count >= full_bank.independent_count:
        raise ValueError("source prefix must be smaller than the extension bank")
    prefix = full_bank.prefix(count)
    if not np.array_equal(source["sample_ids"], prefix.sample_ids):
        raise ValueError("source sample IDs are not the expected canonical prefix")
    if not np.allclose(
        source["deployment_m"],
        prefix.decode_deployments(total_lengths_m),
        rtol=0.0,
        atol=2e-8,
    ):
        raise ValueError("source deployments do not match the canonical prefix")
    if not np.allclose(
        source["rotation_rad"],
        prefix.rotation_rad,
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("source rotations do not match the canonical prefix")
    return count


def _merge_prefix_and_tail(
    prefix: dict[str, np.ndarray],
    tail: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if set(prefix) != set(tail):
        raise ValueError("prefix and extension arrays do not have the same fields")
    merged = {name: np.concatenate((prefix[name], tail[name]), axis=0) for name in prefix}
    if len(np.unique(merged["sample_ids"])) != len(merged["sample_ids"]):
        raise ValueError("merged configuration IDs are not unique")
    return merged


def _verify_stage3_source(protocol) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    extension = protocol.section("extension")
    source_run = PROJECT_ROOT / extension["source_stage3_run"]
    manifest_path = source_run / "run_manifest.json"
    training_path = source_run / "data" / SOURCE_TRAINING_NAME
    validation_path = source_run / "data" / SOURCE_VALIDATION_NAME
    for path in (manifest_path, training_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Stage-3 source is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_status") != "completed":
        raise ValueError("the source Stage-3 run is not marked completed")
    if manifest["protocol"]["sha256"] != extension["source_protocol_sha256"]:
        raise ValueError("the source Stage-3 protocol hash does not match")
    if manifest["baseline_parameters"]["sha256"] != baseline_parameter_hash():
        raise ValueError("the source Stage-3 baseline parameters do not match")
    if _file_sha256(training_path) != extension["source_training_sha256"]:
        raise ValueError("the source Stage-3 training dataset hash does not match")
    if _file_sha256(validation_path) != extension["source_validation_sha256"]:
        raise ValueError("the source Stage-3 validation dataset hash does not match")
    return source_run, _load_npz(training_path), _load_npz(validation_path)


def _extension_gates(
    convergence_rows: list[dict],
    validation_metrics: dict[str, dict],
    residual_map,
    task_region,
    *,
    protocol,
    training_valid_fraction: float,
    validation_valid_fraction: float,
) -> dict:
    spatial = protocol.section("spatial")
    extension = protocol.section("extension")
    primary_cell = float(spatial["primary_cell_size_mm"])
    comparison_counts = list(map(int, extension["configuration_comparison_counts"]))

    def row(count: int) -> dict:
        return next(
            item
            for item in convergence_rows
            if item["endpoint"] == "inner"
            and item["independent_configurations"] == count
            and item["cell_size_mm"] == primary_cell
        )

    previous, final = map(row, comparison_counts)
    relative_volume = abs(
        previous["occupied_volume_cm3"] - final["occupied_volume_cm3"]
    ) / max(final["occupied_volume_cm3"], 1e-15)
    isotropy_changes = {
        key: abs(previous[key] - final[key])
        for key in (
            "volume_weighted_mean",
            "volume_weighted_median",
            "volume_weighted_p10",
        )
    }
    supported = np.isfinite(residual_map.p95_residual_mm)
    supported_volume = float(np.sum(task_region.cell_volume_mm3[supported]))
    task_volume = float(np.sum(task_region.cell_volume_mm3[task_region.occupied]))
    support_fraction = supported_volume / max(task_volume, 1e-15)
    workspace_passed = relative_volume <= float(spatial["workspace_relative_change_limit"])
    coverage = validation_metrics["inner"]["validation_coverage_of_training_workspace"]
    coverage_passed = coverage >= float(spatial["validation_coverage_minimum"])
    isotropy_passed = all(
        change <= float(spatial["spatial_isotropy_absolute_change_limit"])
        for change in isotropy_changes.values()
    )
    jacobian_passed = training_valid_fraction >= 0.999 and validation_valid_fraction >= 0.999
    support_passed = support_fraction >= float(
        extension["ik_spatial_reportable_volume_minimum"]
    )
    extension_passed = all(
        (workspace_passed, coverage_passed, isotropy_passed, jacobian_passed, support_passed)
    )
    return {
        "comparison_counts": comparison_counts,
        "workspace_relative_change": relative_volume,
        "workspace_relative_limit": float(spatial["workspace_relative_change_limit"]),
        "workspace_converged": workspace_passed,
        "validation_coverage": coverage,
        "validation_coverage_minimum": float(spatial["validation_coverage_minimum"]),
        "validation_coverage_passed": coverage_passed,
        "isotropy_absolute_changes": isotropy_changes,
        "isotropy_absolute_limit": float(spatial["spatial_isotropy_absolute_change_limit"]),
        "isotropy_converged": isotropy_passed,
        "training_jacobian_valid_fraction": training_valid_fraction,
        "validation_jacobian_valid_fraction": validation_valid_fraction,
        "jacobian_validity_passed": jacobian_passed,
        "ik_p95_supported_task_volume_fraction": support_fraction,
        "ik_p95_supported_cells": int(np.count_nonzero(supported)),
        "ik_task_region_cells": int(task_region.occupied_cell_count),
        "ik_spatial_reportable_volume_minimum": float(
            extension["ik_spatial_reportable_volume_minimum"]
        ),
        "ik_spatial_support_passed": support_passed,
        "extension_passed": extension_passed,
        "optimisation_may_begin": False,
        "optimisation_gate_note": (
            "Stage 4 only tests whether the baseline evidence is stable and reportable. "
            "Optimisation requires a separately frozen protocol after review."
        ),
    }


def _plot_extension_summary(
    convergence_rows: list[dict],
    residual_map,
    path: Path,
    *,
    primary_cell_size_mm: float,
) -> None:
    rows = sorted(
        [
            row
            for row in convergence_rows
            if row["endpoint"] == "inner"
            and row["cell_size_mm"] == primary_cell_size_mm
        ],
        key=lambda row: row["independent_configurations"],
    )
    counts = [row["independent_configurations"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)
    axes[0].plot(counts, [row["occupied_volume_cm3"] for row in rows], marker="o")
    axes[0].set_ylabel("Occupied-cell volume (cm³)")
    axes[0].set_title("Workspace convergence")
    axes[1].plot(counts, [row["volume_weighted_mean"] for row in rows], marker="o", label="mean")
    axes[1].plot(counts, [row["volume_weighted_median"] for row in rows], marker="o", label="median")
    axes[1].plot(counts, [row["volume_weighted_p10"] for row in rows], marker="o", label="p10")
    axes[1].set_ylabel("Positional isotropy")
    axes[1].set_title("Volume-weighted isotropy")
    axes[1].legend()
    supported = np.isfinite(residual_map.p95_residual_mm)
    axes[2].hist(residual_map.sample_count[supported], bins=20, color="#2c7fb8")
    axes[2].axvline(100, color="#d7301f", linestyle="--", label="minimum n=100")
    axes[2].set_xlabel("Independent targets per supported cell")
    axes[2].set_ylabel("Cells")
    axes[2].set_title("10 mm p95 support")
    axes[2].legend()
    for axis in axes[:2]:
        axis.set_xlabel("Independent configurations")
        axis.grid(alpha=0.25)
    figure.suptitle("Stage-4 inner-tip convergence and IK-support extension")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _write_report(path: Path, *, gates: dict, workspace: dict, ik_metrics: dict, target_count: int, max_support: int) -> None:
    changes = gates["isotropy_absolute_changes"]
    outcome = "passed" if gates["extension_passed"] else "did not pass"
    text = f"""# Stage 4 convergence-extension report

## Outcome

The controlled baseline extension {outcome} its predeclared evidence gates.
Optimisation remains disabled until this result is reviewed and a separate
optimisation protocol is frozen.

## Evidence generated

- 200,000 independent baseline training configurations
- 50,000 independent validation configurations
- {target_count:,} independent, spatially stratified inner-tip IK targets
- 300,000 independent target-candidate configurations
- 10 mm primary axisymmetric cells
- unchanged 0.01 mm IK stopping tolerance and 0.5 mm evaluation threshold

The verified 100,000/25,000 Stage-3 configuration prefixes were reused exactly;
only new canonical suffixes were calculated. Rotated display copies were never
counted as independent samples or targets.

## Convergence gates

| Check | Result | Limit | Pass |
|---|---:|---:|---|
| Workspace change, 150k to 200k | {gates['workspace_relative_change']:.5f} | <= {gates['workspace_relative_limit']:.2f} | {gates['workspace_converged']} |
| Independent-validation coverage | {gates['validation_coverage']:.5f} | >= {gates['validation_coverage_minimum']:.2f} | {gates['validation_coverage_passed']} |
| Isotropy mean absolute change | {changes['volume_weighted_mean']:.5f} | <= {gates['isotropy_absolute_limit']:.2f} | {changes['volume_weighted_mean'] <= gates['isotropy_absolute_limit']} |
| Isotropy median absolute change | {changes['volume_weighted_median']:.5f} | <= {gates['isotropy_absolute_limit']:.2f} | {changes['volume_weighted_median'] <= gates['isotropy_absolute_limit']} |
| Isotropy p10 absolute change | {changes['volume_weighted_p10']:.5f} | <= {gates['isotropy_absolute_limit']:.2f} | {changes['volume_weighted_p10'] <= gates['isotropy_absolute_limit']} |
| Task volume supporting 10 mm p95 map | {gates['ik_p95_supported_task_volume_fraction']:.2%} | >= {gates['ik_spatial_reportable_volume_minimum']:.0%} | {gates['ik_spatial_support_passed']} |

## Extended baseline results

- Inner-tip occupied workspace: {workspace['training_occupied_volume_cm3']:.2f} cm^3
- Validation coverage: {workspace['validation_coverage_of_training_workspace']:.2%}
- Validation volume-weighted mean isotropy: {workspace['validation_volume_weighted_mean']:.4f}
- Validation volume-weighted median isotropy: {workspace['validation_volume_weighted_median']:.4f}
- Validation volume-weighted p10 isotropy: {workspace['validation_volume_weighted_p10']:.4f}
- IK residual <= 0.5 mm: {ik_metrics['ik_within_evaluation_threshold_rate']:.2%}
- IK median residual: {ik_metrics['ik_median_error_mm']:.6f} mm
- IK p95 residual: {ik_metrics['ik_p95_error_mm']:.3f} mm
- Maximum targets in one 10 mm cell: {max_support}

These IK values are fixed-start numerical model residuals, not experimental robot
accuracy or clinical safety measures.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "config" / "analysis_protocol_stage4_extension.toml",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--ik-chunk-size", type=int, default=200)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.chunk_size < 1 or arguments.ik_chunk_size < 1:
        raise ValueError("worker and chunk settings must be positive")

    protocol = load_analysis_protocol(arguments.protocol)
    if protocol.snapshot["methodology_stage"] != 4:
        raise ValueError("the Stage-4 runner requires a Stage-4 protocol")
    sampling = protocol.section("sampling")
    spatial = protocol.section("spatial")
    ik_settings = protocol.section("ik")
    jacobian = protocol.section("jacobian")
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
        f"baseline_extension_{datetime.now().strftime('%Y%m%d')}_"
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
        run_kind="stage-4-controlled-convergence-extension",
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
        progress("verify Stage-3 source prefixes")
        source_run, training_prefix, validation_prefix = _verify_stage3_source(protocol)
        manifest["verified_source_run"] = str(source_run.relative_to(PROJECT_ROOT))

        progress("extend training configurations to 200,000")
        training_bank = canonical_configuration_bank(
            int(sampling["baseline_training_configurations"]),
            seed=int(sampling["training_seed"]),
            split="stage3-baseline-training",
        )
        training_prefix_count = _verify_prefix(training_prefix, training_bank, lengths_m)
        training_tail = _evaluate_configuration_bank(
            parameters,
            _tail_bank(training_bank, training_prefix_count),
            lengths_m,
            workers=arguments.workers,
            chunk_size=arguments.chunk_size,
            translation_step_mm=float(jacobian["translation_step_mm"]),
            rotation_step_rad=float(jacobian["rotation_step_rad"]),
            progress_label="training extension FK/Jacobian",
        )
        training = _merge_prefix_and_tail(training_prefix, training_tail)
        _save_npz(data_directory / "baseline_training_200000.npz", **training)

        progress("extend validation configurations to 50,000")
        validation_bank = canonical_configuration_bank(
            int(sampling["baseline_validation_configurations"]),
            seed=int(sampling["validation_seed"]),
            split="stage3-baseline-validation",
        )
        validation_prefix_count = _verify_prefix(validation_prefix, validation_bank, lengths_m)
        validation_tail = _evaluate_configuration_bank(
            parameters,
            _tail_bank(validation_bank, validation_prefix_count),
            lengths_m,
            workers=arguments.workers,
            chunk_size=arguments.chunk_size,
            translation_step_mm=float(jacobian["translation_step_mm"]),
            rotation_step_rad=float(jacobian["rotation_step_rad"]),
            progress_label="validation extension FK/Jacobian",
        )
        validation = _merge_prefix_and_tail(validation_prefix, validation_tail)
        _save_npz(data_directory / "baseline_validation_50000.npz", **validation)

        progress("extended workspace and isotropy convergence")
        convergence_rows, workspace_maps, scalar_maps, validation_metrics = _convergence_analysis(
            training,
            validation,
            counts=list(map(int, sampling["baseline_convergence_counts"])),
            cell_sizes_mm=list(map(float, spatial["cell_size_convergence_candidates_mm"])),
            primary_cell_size_mm=float(spatial["primary_cell_size_mm"]),
            minimum_median_samples=int(spatial["minimum_samples_median"]),
            minimum_lower_tail_samples=int(spatial["minimum_samples_lower_tail"]),
        )
        _write_csv(output / "workspace_isotropy_convergence.csv", convergence_rows)
        _write_json(output / "workspace_validation_metrics.json", {"inner": validation_metrics["inner"]})
        workspace = workspace_maps["inner"]
        scalar = scalar_maps["inner"]
        _save_npz(
            data_directory / "inner_workspace_isotropy_10mm.npz",
            radial_edges_mm=workspace.grid.radial_edges_mm,
            z_edges_mm=workspace.grid.z_edges_mm,
            cell_volume_mm3=workspace.grid.cell_volume_mm3,
            workspace_sample_count=workspace.sample_count,
            occupied=workspace.occupied,
            enclosed_unsampled=workspace.internal_void,
            isotropy_maximum=scalar.maximum,
            isotropy_mean=scalar.mean,
            isotropy_median=scalar.median,
            isotropy_p10=scalar.lower_p10,
            isotropy_q25=scalar.q25,
            isotropy_q75=scalar.q75,
            isotropy_iqr=scalar.iqr,
            isotropy_sample_count=scalar.sample_count,
        )
        _plot_convergence(
            convergence_rows,
            figure_directory / "inner_convergence_extension.png",
            tube_name="inner",
            primary_cell_size_mm=float(spatial["primary_cell_size_mm"]),
        )
        for label, values, title in (
            ("median", scalar.median, "Median positional isotropy"),
            ("p10", scalar.lower_p10, "Configuration-level p10 isotropy"),
        ):
            _plot_radial_map(
                scalar.grid,
                values,
                figure_directory / f"inner_isotropy_{label}.png",
                title=f"Inner tip: {title}",
                colour_label="Positional isotropy",
                cmap="viridis",
                limits=(0.0, 1.0),
            )
        _plot_workspace_sweep(
            training["endpoint_positions_mm"][:, 0],
            training["isotropy"][:, 0],
            figure_directory / "inner_workspace_3d_display.png",
            tube_name="inner",
            independent_count=len(training["sample_ids"]),
        )

        progress("generate 300,000 independent IK target candidates")
        candidate_bank = canonical_configuration_bank(
            int(sampling["ik_target_candidate_configurations"]),
            seed=int(sampling["ik_target_validation_seed"]),
            split="stage4-ik-target-validation-candidates",
        )
        candidates = _forward_configuration_bank(
            parameters,
            candidate_bank,
            lengths_m,
            workers=arguments.workers,
            chunk_size=arguments.chunk_size,
            progress_label="inner target candidates",
        )
        task_region = freeze_baseline_task_region(
            training["endpoint_positions_mm"][:, 0],
            training["sample_ids"],
            target_tube=0,
            baseline_parameter_sha256=baseline_parameter_hash(),
            source_dataset_id="stage4-baseline-training-200000",
            grid=workspace.grid,
        )
        _save_task_region(data_directory / "inner_task_region.npz", task_region)
        targets = select_stratified_canonical_targets(
            task_region,
            candidates["endpoint_positions_mm"][:, 0],
            candidates["deployment_m"],
            candidates["rotation_rad"],
            candidates["sample_ids"],
            target_count=int(sampling["ik_primary_validation_targets"]),
            split="validation",
            seed=int(sampling["ik_target_validation_seed"]),
            source_dataset_id="stage4-ik-target-validation-candidates-300000",
        )
        _save_target_set(data_directory / "inner_ik_validation_targets.npz", targets)

        progress("solve 75,000 fixed-retraction inner-tip IK targets")
        errors = _parallel_ik(
            parameters,
            targets,
            target_tube=0,
            solver_tolerance_mm=float(ik_settings["solver_tolerance_mm"]),
            max_iterations=int(ik_settings["max_iterations"]),
            workers=arguments.workers,
            chunk_size=arguments.ik_chunk_size,
            progress_label="inner IK extension",
        )
        _save_ik(data_directory / "inner_ik_validation_results.npz", errors)
        ik_metrics = summarise_ik_errors(
            errors,
            evaluation_threshold_mm=float(ik_settings["evaluation_threshold_mm"]),
        )
        ik_metrics.pop("ik_success_rate", None)
        ik_metrics.update(
            {
                "independent_target_count": targets.independent_target_count,
                "represented_target_cells": int(np.count_nonzero(targets.targets_per_cell)),
                "task_region_cells": task_region.occupied_cell_count,
            }
        )
        _write_json(output / "ik_validation_metrics.json", {"inner": ik_metrics})
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
            title="Inner-tip IK target support",
            colour_label="Independent targets per cell",
            cmap="plasma",
            logarithmic=True,
        )

        gates = _extension_gates(
            convergence_rows,
            validation_metrics,
            residual_map,
            task_region,
            protocol=protocol,
            training_valid_fraction=float(np.mean(training["jacobian_valid"])),
            validation_valid_fraction=float(np.mean(validation["jacobian_valid"])),
        )
        _write_json(output / "validation_gates.json", gates)
        _plot_extension_summary(
            convergence_rows,
            residual_map,
            figure_directory / "stage4_extension_summary.png",
            primary_cell_size_mm=float(spatial["primary_cell_size_mm"]),
        )
        _write_report(
            output / "STAGE4_EXTENSION_REPORT.md",
            gates=gates,
            workspace=validation_metrics["inner"],
            ik_metrics=ik_metrics,
            target_count=targets.independent_target_count,
            max_support=int(np.max(residual_map.sample_count)),
        )

        manifest["run_status"] = "completed"
        manifest["progress_stage"] = "completed"
        manifest["validation_gates"] = gates
        manifest["formal_result_note"] = (
            "Stage-4 controlled baseline extension completed. Optimisation remains "
            "disabled pending review and a separately frozen optimisation protocol."
        )
        write_run_manifest(manifest_path, manifest)
        print(f"\nCompleted Stage-4 extension: {output}", flush=True)
    except Exception as error:
        manifest["run_status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        write_run_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
