# Tools

- `ps5_button_identifier.py` reports controller button numbers.
- `ps5_trigger_identifier.py` reports the L2 and R2 axis values.
- `ps5_controller_test.py` provides a broader controller-input check.
- `workspace_generator.py` calculates a 10,000-configuration workspace and
  writes its CSV and plot to the ignored `results/` directory.
- `generate_workspace_map.py` regenerates the compact 12,000-point map used by
  the workspace analysis tools.
- `generate_endpoint_workspace_maps.py` generates the cached inner, middle, and
  outer endpoint samples used for smooth workspace limits and IK restarts.
- `generate_reachability_zones.py` generates the optional legacy blue/red/grey
  diagnostic map from 80,000 valid configurations.
- `run_design_study.py` retains the legacy exploratory workspace,
  positional-dexterity, fully-retracted local-IK residual, sensitivity and
  optimisation workflow. It now records the Stage-2 protocol and deviations in
  its manifest, but its profiles are not protocol-v1 final evaluations.
- `run_stage2_method_validation.py` reproduces the lightweight Jacobian-step
  and IK-tolerance pilots recorded in the Stage-2 protocol. It prints JSON by
  default and only writes a file when an unused `--output` path is supplied.
- `run_dense_ik_residual_study.py` solves an independently sampled reachable
  target cloud from one fixed fully retracted configuration and produces
  point-cloud, maximum-voxel, median-voxel, cross-section, and occupancy
  residual figures.
- `plot_radial_vertical_example.py` shows a representative high-Z,
  low-radial-distance configuration alongside the sampled workspace.
- `run_symmetry_expanded_study.py` is a legacy quadrant diagnostic. Its rotated
  positions are display or repeated-measure copies, not additional independent
  configurations or statistical targets.
- `run_hybrid_ik_study.py` compares fixed-start and eight-seed multi-start IK,
  uses logarithmic residual and failure plots, and validates symmetry expansion
  against an independent full-space Monte Carlo workspace.
- `plot_axisymmetric_ik_slice.py` performs a legacy quick canonical-plane IK
  check. Protocol-v1 statistics must be aggregated from canonical IDs before
  any 360-degree display sweep.
- `replot_symmetry_results.py` changes voxel size or the minimum median-sample
  threshold without rerunning the expensive IK target solves.

Run tools from the repository root with the project environment, for example:

```bash
./.venv/bin/python tools/ps5_button_identifier.py
./.venv/bin/python tools/run_design_study.py --profile quick
./.venv/bin/python tools/run_stage2_method_validation.py
./.venv/bin/python tools/run_dense_ik_residual_study.py --samples 10000
./.venv/bin/python tools/run_symmetry_expanded_study.py
./.venv/bin/python tools/run_hybrid_ik_study.py
./.venv/bin/python tools/plot_axisymmetric_ik_slice.py
```

## Current formal analysis and documentation tools

- `plot_readme_workspace.py` reproduces the GitHub map from the bundled 12,000-state
  endpoint cache without rerunning an analysis study.
- `run_stage3_baseline_evaluation.py` runs the versioned baseline evaluation.
- `run_stage4_convergence_extension.py` extends primary-endpoint convergence evidence.
- `run_stage4_1_ik_support_extension.py` extends primary-endpoint IK spatial support.
- `run_stage5a_protocol_preflight.py` audits optimisation methodology and local
  evidence; it does not optimise candidate designs.

Use `--help` on an analysis runner to inspect its protocol and output options.
Formal runs can be expensive and their datasets are not shipped in Git. The
Stage-5A preflight requires the archived Stage-4 and Stage-4.1 evidence paths
specified in its protocol.
