"""Freeze and audit Stage 5A without evaluating or optimising candidate designs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization_protocol import (  # noqa: E402
    audit_optimization_evidence,
    build_optimization_manifest,
    design_from_protocol_values,
    load_optimization_protocol,
    write_optimization_manifest,
)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_variables(path: Path, variables: list[dict]) -> None:
    fields = ("name", "field", "tube_index", "unit", "lower", "baseline", "upper", "bound_basis")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in variables)


def _corner_preflight(protocol) -> dict:
    variables = protocol.snapshot["design_variables"]
    baseline_values = {row["name"]: float(row["baseline"]) for row in variables}
    baseline_candidate = design_from_protocol_values(protocol, baseline_values)
    corner_count = 0
    for choices in product((0, 1), repeat=len(variables)):
        values = {
            row["name"]: float(row["lower"] if choice == 0 else row["upper"])
            for row, choice in zip(variables, choices)
        }
        design_from_protocol_values(protocol, values)
        corner_count += 1
    return {
        "baseline_design_valid": True,
        "baseline_total_length_mm": baseline_candidate.total_length_mm.tolist(),
        "baseline_curved_length_mm": baseline_candidate.curved_length_mm.tolist(),
        "baseline_precurvature_per_m": baseline_candidate.precurvature_per_m.tolist(),
        "bound_corner_designs_checked": corner_count,
        "bound_corner_designs_valid": corner_count,
        "diameters_fixed": True,
        "youngs_modulus_fixed": True,
    }


def _write_report(path: Path, *, protocol, evidence: dict, preflight: dict) -> None:
    variables = protocol.snapshot["design_variables"]
    objectives = protocol.snapshot["objectives"]
    rows = "\n".join(
        f"| {row['name']} | {row['lower']:g} | {row['baseline']:g} | {row['upper']:g} | {row['unit']} |"
        for row in variables
    )
    objective_rows = "\n".join(
        f"| {row['name']} | {row['direction']} | {row['definition']} |"
        for row in objectives
    )
    text = f"""# Stage 5A optimisation-protocol report

## Outcome

Stage 5A is complete. The local computational design protocol is frozen and its
baseline evidence, reserved validation data and physical design constraints passed
preflight. Stage 5B pilot evaluation may begin. Stage 5C full optimisation remains
blocked until the Stage 5B fidelity and ranking-stability gates pass.

This protocol does not claim globally optimal or immediately manufacturable tube
dimensions. Available stock sizes, manufacturing tolerances and application-specific
geometric limits have not been supplied.

## Scope

The distal inner-tube tip is the only optimisation endpoint. Middle and outer tips
will be reported separately without equal weighting. Tip orientation, force,
loaded stiffness, collision/contact, buckling and clinical safety are outside the
available model and are not objectives.

Young's modulus remains fixed at 75 GPa for all tubes. OD and ID remain fixed at the
supervisor nominal values because treating them as continuously adjustable would be
unjustified without a tubing catalogue or manufacturing process. The straight inner
wire retains zero curved length and zero pre-curvature.

## Local design variables

Every active variable uses the same plus/minus 10% neighbourhood around the baseline.
This supports a fair local sensitivity and design study but does not define physical
manufacturing limits.

| Variable | Lower | Baseline | Upper | Unit |
|---|---:|---:|---:|---|
{rows}

All {preflight['bound_corner_designs_checked']} lower/upper bound corners satisfy the
current model's tube-order, curved-length and fixed-clearance constraints.

## Multi-objective formulation

| Objective | Direction | Definition |
|---|---|---|
{objective_rows}

Candidate designs must retain at least 95% of the frozen baseline task-region volume
and at least 99.9% valid interior Jacobians. Constraint domination is used instead
of an arbitrary weighted penalty. No single weighted winner will be reported because
clinical or application-specific trade-off weights are unavailable. Results will be
presented as a Pareto front with objective extremes and a labelled knee candidate.

Uncovered or under-supported task cells receive zero isotropy in the whole-region
objectives, preventing a design from appearing better by reaching only an easy subset.
Candidate workspace outside the common task region is reported but is not rewarded as
an objective.

## Data separation

- Training configurations: seed 42; prefix-stable canonical samples
- Reserved validation configurations: seed 1042; {evidence['validation_configuration_count']:,} rows
- Training IK targets: seed 2042; generated during Stage 5B from the frozen task region
- Sealed validation IK targets: seed 3042; {evidence['reserved_validation_target_count']:,} rows
- Continuous rotations about Z remain display-only and never increase independent N

The sealed validation configurations and targets cannot be used for sensitivity,
search, ranking, hyperparameter selection or shortlist selection.

## Planned staged evaluation

Stage 5B uses 33 designs including the baseline at 15,000 configurations and 2,000
training IK targets. Eight designs are repeated at 50,000 configurations and 5,000
training targets. Every objective requires Spearman rank correlation of at least 0.8
between fidelities, together with at least 50% top-quartile overlap.

Only after those gates pass may a separately authorised Stage 5C use NSGA-II with
feasibility-first constraint domination. The planned search has population 32 and 20
generations, followed by medium- and high-fidelity reevaluation of the nondominated
set. At most five non-baseline designs may be shortlisted.

Stage 5D then evaluates the frozen shortlist once using 50,000 independent validation
configurations and 100,000 sealed validation IK targets. No post-validation retuning
is permitted.

## Verified baseline evidence

- Stage-4 workspace/isotropy convergence gates: passed
- Stage-4.1 spatial IK-support gate: passed
- Stage-4.1 supported 10 mm p95-map volume: {evidence['stage4_1_supported_task_volume_fraction']:.2%}
- Training and validation configuration IDs: disjoint
- Baseline parameter and evidence-file hashes: matched

## Remaining limitations

- The plus/minus 10% bounds are local computational study bounds, not certified
  manufacturing limits.
- Available discrete Nitinol tube sizes and tolerances are unknown.
- No anatomical task-space region has been provided, so the baseline workspace remains
  the provisional common comparison region.
- The interpretation of 19.12 and 14.04 as inverse metres remains provisional.
- Numerical IK residual is not experimental robot accuracy.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "config" / "optimization_protocol_stage5a.toml",
    )
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    protocol = load_optimization_protocol(arguments.protocol)
    evidence = audit_optimization_evidence(protocol, PROJECT_ROOT)
    preflight = _corner_preflight(protocol)

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
        f"stage5a_{datetime.now().strftime('%Y%m%d')}_"
        f"{protocol.sha256[:8]}_{commit[:8]}"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty formal run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    readiness = {
        **evidence,
        **preflight,
        "design_variable_count": len(protocol.snapshot["design_variables"]),
        "objective_count": len(protocol.snapshot["objectives"]),
        "stage5a_complete": True,
        "stage5b_pilot_may_begin": True,
        "stage5c_full_optimisation_may_begin": False,
        "global_or_manufacturable_optimum_claim_allowed": False,
    }
    _write_json(output / "readiness.json", readiness)
    _write_json(output / "objectives.json", protocol.snapshot["objectives"])
    _write_variables(output / "design_variables.csv", protocol.snapshot["design_variables"])
    _write_report(
        output / "STAGE5A_OPTIMISATION_PROTOCOL_REPORT.md",
        protocol=protocol,
        evidence=evidence,
        preflight=preflight,
    )
    manifest = build_optimization_manifest(
        protocol,
        evidence_audit=evidence,
        project_root=PROJECT_ROOT,
    )
    manifest["preflight"] = preflight
    manifest["readiness"] = {
        "stage5a_complete": True,
        "stage5b_pilot_may_begin": True,
        "stage5c_full_optimisation_may_begin": False,
    }
    write_optimization_manifest(output / "run_manifest.json", manifest)
    print(f"Completed Stage-5A protocol preflight: {output}")


if __name__ == "__main__":
    main()
