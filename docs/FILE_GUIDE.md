# File guide

Start with `launch_simulation.py` and the [simulation guide](SIMULATION.md). Use the [research guide](../research/README.md) for runnable study commands. Most numerical modules below are imported by those entry points; they do not open an application by themselves.

## Shared root modules

The root viewers and provisional profiles are earlier versions retained for dependencies and historical tests. The main launcher uses the isolated `proposed_tradeoff_simulator/` snapshot.

| File | Purpose |
| --- | --- |
| [CTR_superPosKin_fun_sectioned.py](../CTR_superPosKin_fun_sectioned.py) | Geometry-aware extension of the inherited constant-curvature CTR forward model. |
| [analysis_protocol.py](../analysis_protocol.py) | Load, validate and record the versioned CTR analysis methodology. |
| [ctr_design_analysis.py](../ctr_design_analysis.py) | Reproducible design-analysis methods for the CTR simulation. |
| [ctr_inverse_kinematics.py](../ctr_inverse_kinematics.py) | Constrained numerical inverse kinematics for CTR tip-position control. |
| [ctr_motion_planner.py](../ctr_motion_planner.py) | Target planning and smooth actuator interpolation for the CTR simulator. |
| [ctr_operating_profile.py](../ctr_operating_profile.py) | Explicit live-viewer operating limits; historical analysis defaults stay unchanged. |
| [ctr_sampling.py](../ctr_sampling.py) | Versioned configuration sampling and rotational-display bookkeeping. |
| [ctr_spatial_analysis.py](../ctr_spatial_analysis.py) | Physical radial--vertical aggregation for protocol-v1 CTR analysis. |
| [ctr_task_space.py](../ctr_task_space.py) | Frozen baseline task regions and spatially stratified canonical IK targets. |
| [ctr_viewer_workspace.py](../ctr_viewer_workspace.py) | Profile-keyed viewer caches, separate from frozen research datasets. |
| [ctr_workspace_map.py](../ctr_workspace_map.py) | Generate and load sampled reachable-tip caches for the CTR viewer. |
| [interactive_ctr_tip_control.py](../interactive_ctr_tip_control.py) | Tip-position control for the VisPy concentric-tube robot simulator. |
| [interactive_ctr_vispy.py](../interactive_ctr_vispy.py) | Real-time VisPy viewer for the concentric-tube robot. |
| [launch_simulation.py](../launch_simulation.py) | Start the final-report original or proposed CTR in an isolated process. |
| [optimization_protocol.py](../optimization_protocol.py) | Load, validate and audit the frozen Stage-5A optimisation methodology. |
| [tube_parameters.py](../tube_parameters.py) | Supervisor-provided CTR specification and explicit model assumptions. |

## Simulator files

These filenames occur in both simulator folders. The proposed folder contains t027; the measured folder retains the earlier candidate.

| File | Purpose |
| --- | --- |
| [interactive_ctr_tip_control.py](../proposed_tradeoff_simulator/interactive_ctr_tip_control.py) | Main desktop interface with joint and Cartesian endpoint control, previews, locks and execution. |
| [interactive_ctr_vispy.py](../proposed_tradeoff_simulator/interactive_ctr_vispy.py) | 3D rendering, keyboard/controller input and joint controls. |
| [CTR_superPosKin_fun_sectioned.py](../proposed_tradeoff_simulator/CTR_superPosKin_fun_sectioned.py) | Section-aware forward kinematics; returns shape and endpoint positions from tube inputs. |
| [tube_parameters.py](../proposed_tradeoff_simulator/tube_parameters.py) | Original geometry, diameters and material parameters. |
| [optimised_configuration.json](../proposed_tradeoff_simulator/optimised_configuration.json) | Alternative geometry and its provenance; t027 in the proposed folder, earlier candidate in the measured folder. |
| [ctr_operating_profile.py](../proposed_tradeoff_simulator/ctr_operating_profile.py) | Selects tube geometry and 108/108.5 mm gap profile; validates imports and reports carriage positions. |
| [design_constraints.py](../proposed_tradeoff_simulator/design_constraints.py) | Derives feasible exterior tube exposures from material lengths and coupled carriage limits. |
| [coordinated_control.py](../proposed_tradeoff_simulator/coordinated_control.py) | Computes minimum required follower-carriage movement for a requested joint translation. |
| [ctr_inverse_kinematics.py](../proposed_tradeoff_simulator/ctr_inverse_kinematics.py) | Constrained numerical solver for a requested endpoint position. |
| [ctr_motion_planner.py](../proposed_tradeoff_simulator/ctr_motion_planner.py) | Previews alternative solutions and direct or retract/reorient/advance motion routes. |
| [ctr_workspace_map.py](../proposed_tradeoff_simulator/ctr_workspace_map.py) | Samples endpoint positions and constructs approximate display envelopes. |
| [ctr_viewer_workspace.py](../proposed_tradeoff_simulator/ctr_viewer_workspace.py) | Builds or reloads caches keyed by geometry, limits and sampling settings. |
| [actuator_diagram.py](../proposed_tradeoff_simulator/actuator_diagram.py) | Draws the carriage travel and gap schematic in the viewer. |
| [test_measured.py](../proposed_tradeoff_simulator/test_measured.py) | Unit checks for measured limits, controls, IK, routes, cache provenance and CSV rejection. |
| [verify_and_capture.py](../proposed_tradeoff_simulator/verify_and_capture.py) | Proposed-only native rendering and end-to-end movement check; writes a screenshot and verification record. |
| [demo_check.py](../measured_hardware_simulator/demo_check.py) | Measured-viewer native rendering and end-to-end control check. |
| [demo_optimised.py](../measured_hardware_simulator/demo_optimised.py) | Opens a demonstration of the earlier measured-search candidate. |
| [original_source_hashes.json](../measured_hardware_simulator/original_source_hashes.json) | Historical fingerprints of the original source before measured-hardware adaptations. |

The `.command` files are macOS launch shortcuts that expect `.venv` in the root. `Launch Proposed CTR.command` opens t027; `Launch Measured CTR.command` opens the measured original; `Launch Optimised CTR.command` opens the earlier candidate. The root Python launcher is the simplest cross-platform route.

## Research and plotting programs

Final-report entry points are identified in the [research guide](../research/README.md). Earlier `stage*`, broad-hardware, regional and symmetry studies are retained as supporting development work, not additional final-report results. Plot/report scripts read the corresponding saved data paths, usually from the release archives.

| File | Purpose |
| --- | --- |
| [build_comparative_study.py](../tools/build_comparative_study.py) | Summarise frozen independent validation; no design retuning or production edits. |
| [build_original_optimised_difference_figures.py](../tools/build_original_optimised_difference_figures.py) | Descriptive difference/overlay figures; no search, retuning or production changes. |
| [build_tradeoff_dissertation_figures.py](../tools/build_tradeoff_dissertation_figures.py) | Figures from frozen results only: no solver or selection changes. |
| [chapter5_measured_figures.py](../tools/chapter5_measured_figures.py) | Draft-style maps from the new dense measured-hardware study. |
| [chapter5_measured_latex.py](../tools/chapter5_measured_latex.py) | Historical report-authoring helper; not needed to run simulations or numerical studies. |
| [chapter5_measured_study.py](../tools/chapter5_measured_study.py) | Frozen-candidate dense comparison for Chapter 5; separate from optimisation. |
| [chapter5_plain_revision.py](../tools/chapter5_plain_revision.py) | Historical report-authoring helper for prose/figure packaging; not a numerical study. |
| [chapter5_topup_support.py](../tools/chapter5_topup_support.py) | Count-only support top-up; original results archived, no design/solver changes. |
| [check_broad_study_robustness.py](../tools/check_broad_study_robustness.py) | Post-selection sampling and solver robustness; no parameter retuning. |
| [extend_regional_search.py](../tools/extend_regional_search.py) | Further bounded search; frozen regions, unchanged tolerances, two holdout seeds. |
| [final_geometry_comparison.py](../tools/final_geometry_comparison.py) | Isolated, budget-matched geometry-only study. Never installs a design. |
| [generate_endpoint_workspace_maps.py](../tools/generate_endpoint_workspace_maps.py) | Generate cached inner/middle/outer endpoint workspace samples. |
| [generate_reachability_zones.py](../tools/generate_reachability_zones.py) | Generate cached blue/red/grey reachability zones for the live viewer. |
| [generate_workspace_map.py](../tools/generate_workspace_map.py) | Regenerate the compact workspace map used by the VisPy tip viewer. |
| [illustrate_extended_reach.py](../tools/illustrate_extended_reach.py) | Independent illustrative multistart check; does not change study results. |
| [investigate_tradeoff_paths.py](../tools/investigate_tradeoff_paths.py) | Post-selection diagnostics of two failed paths; does not change validation or viewer. |
| [optimise_measured_ctr.py](../tools/optimise_measured_ctr.py) | Reproducible bounded search using measured coupled stops, never old offsets. |
| [plot_axisymmetric_ik_slice.py](../tools/plot_axisymmetric_ik_slice.py) | Generate one symmetry-reduced XY inverse-kinematics validation slice. |
| [plot_radial_vertical_example.py](../tools/plot_radial_vertical_example.py) | Plot a high-Z, low-radial-distance CTR configuration from a design study. |
| [plot_readme_workspace.py](../tools/plot_readme_workspace.py) | Reproduce the README's CTR tube and reachable-workspace illustration. |
| [ps5_button_identifier.py](../tools/ps5_button_identifier.py) | Displays pressed PS5 button numbers for input mapping; requires a connected controller. |
| [ps5_controller_test.py](../tools/ps5_controller_test.py) | Checks joystick detection and live PS5 input events. |
| [ps5_trigger_identifier.py](../tools/ps5_trigger_identifier.py) | Displays trigger/axis values to identify controller inputs; requires a connected controller. |
| [redraw_dexterity_difference.py](../tools/redraw_dexterity_difference.py) | Redraw the existing comparison with one map; preserve original study outputs. |
| [redraw_tube_shapes.py](../tools/redraw_tube_shapes.py) | Illustrate the saved comparison's intrinsic tube geometry, without retuning. |
| [render_extended_reach.py](../tools/render_extended_reach.py) | Render saved verified reach example; sampled envelopes are illustrative. |
| [replot_symmetry_results.py](../tools/replot_symmetry_results.py) | Reaggregate and redraw an existing symmetry-expanded CTR study. |
| [report_broad_hardware_study.py](../tools/report_broad_hardware_study.py) | Generate a concise report and charts from the frozen broad study. |
| [report_final_geometry_comparison.py](../tools/report_final_geometry_comparison.py) | Read-only reporting of the final geometry study; no simulator configuration edits. |
| [report_fixed_start_comparison.py](../tools/report_fixed_start_comparison.py) | Chapter-aligned comparison figures and an optional LaTeX subsection. |
| [report_hardware_optimisation.py](../tools/report_hardware_optimisation.py) | Reproduce the report and scientific figures for the completed local study. |
| [report_measured_optimisation.py](../tools/report_measured_optimisation.py) | Render measured-design results; install only a validated improving candidate. |
| [report_original_D_comparison.py](../tools/report_original_D_comparison.py) | Chapter-aligned comparison figures and an optional LaTeX subsection. |
| [report_task_prioritised_tradeoff.py](../tools/report_task_prioritised_tradeoff.py) | Writes the t027 comparison report; optional reserve_validation argument summarises t019. |
| [report_tradeoff_paths.py](../tools/report_tradeoff_paths.py) | Checks alternative initial configurations and plots post-hoc path diagnostics. |
| [run_broad_hardware_study.py](../tools/run_broad_hardware_study.py) | Four-variable, three-objective exploratory hardware-constrained CTR study. |
| [run_dense_ik_residual_study.py](../tools/run_dense_ik_residual_study.py) | Generate dense point-cloud and voxel plots of local numerical IK residual. |
| [run_design_study.py](../tools/run_design_study.py) | Run the legacy exploratory CTR workspace, dexterity and design workflow. |
| [run_fixed_start_comparison.py](../tools/run_fixed_start_comparison.py) | Separate hardware-constrained fixed-start study preserving original DLS settings. |
| [run_hardware_optimisation.py](../tools/run_hardware_optimisation.py) | Local, approximate-hardware CTR study; preserves production model and protocols. |
| [run_hybrid_ik_study.py](../tools/run_hybrid_ik_study.py) | Run the legacy fixed-start, multi-start and symmetry diagnostic study. |
| [run_original_D_comparison.py](../tools/run_original_D_comparison.py) | Separate hardware-constrained fixed-start study preserving original DLS settings. |
| [run_stage2_method_validation.py](../tools/run_stage2_method_validation.py) | Reproduce the lightweight numerical checks used to freeze Stage-2 settings. |
| [run_stage3_baseline_evaluation.py](../tools/run_stage3_baseline_evaluation.py) | Run the protocol-v1 Stage-3 baseline convergence and figure evaluation. |
| [run_stage4_1_ik_support_extension.py](../tools/run_stage4_1_ik_support_extension.py) | Complete Stage-4 inner-tip 10 mm IK-map support without changing thresholds. |
| [run_stage4_convergence_extension.py](../tools/run_stage4_convergence_extension.py) | Run the frozen Stage-4 inner-tip convergence and IK-support extension. |
| [run_stage5a_protocol_preflight.py](../tools/run_stage5a_protocol_preflight.py) | Freeze and audit Stage 5A without evaluating or optimising candidate designs. |
| [run_symmetry_expanded_study.py](../tools/run_symmetry_expanded_study.py) | Run the legacy quadrant-reduced CTR rotational diagnostic. |
| [strict_measured_search.py](../tools/strict_measured_search.py) | Non-regression search. Writes a separate study, never installs a candidate. |
| [task_prioritised_tradeoff_search.py](../tools/task_prioritised_tradeoff_search.py) | Separate benchmark-prioritised trade-off study; never modifies live configuration. |
| [test_regional_candidates.py](../tools/test_regional_candidates.py) | Baseline-defined regional screening with fresh, frozen-finalist validation. |
| [test_regional_candidates_round2.py](../tools/test_regional_candidates_round2.py) | Baseline-defined regional screening with fresh, frozen-finalist validation. |
| [validate_broad_hardware_study.py](../tools/validate_broad_hardware_study.py) | Independent validation of the frozen broad-search shortlist. No retuning. |
| [validate_hardware_candidates.py](../tools/validate_hardware_candidates.py) | Additional frozen-candidate numerical audits; does not tune designs. |
| [validate_tradeoff_reserve.py](../tools/validate_tradeoff_reserve.py) | Explicit second validation round of the remaining training-qualified candidate. |
| [workspace_generator.py](../tools/workspace_generator.py) | Earlier workspace generation program retained for historical analysis. |

## Tests, protocols and data

| Location | What the files contain / how to use them |
| --- | --- |
| [`tests/`](../tests/README.md) | Individual model, IK, controller mapping, workspace, protocol and study checks; run the named Python files or the documented suites. |
| [`config/`](../config/README.md) | Historical Stage 2–5A TOML settings. Final measured studies record their own `protocol.json` files in `output/`. |
| [`legacy/`](../legacy/README.md) | Earlier model and viewer versions; not the primary entry point. |
| `assets/workspace/*.npz` | Historical display sample caches; the current measured viewer generates separate caches. |
| [`research/verification/`](../research/verification/README.md) | Appendix D analytical checks, evidence and evaluated-design CSV. |
| `output/*/protocol.json` | Frozen sampling, limits, seeds and evaluation settings for that study. |
| `output/*/*records.json` | Candidate evaluations, training selection or held-out metrics; filenames identify the stage. |
| `output/*/outcome.json` | Acceptance decisions, including failed gates. |
| `output/*/source_snapshot/` and `source_hashes.json` | Original calculation sources and recorded SHA-256 fingerprints. |
| `output/chapter5_measured_revision/data/` | Dense comparison summaries; `.npz` arrays are restored from the release. |
| [`docs/dissertation/manifest.json`](dissertation/manifest.json) | Final report hash, page/figure mapping and image/data checksums. |
| [`research/downloads.json`](../research/downloads.json) | Release archive names, sizes, member counts and checksums. |
| [`docs/media/`](media/README.md) | Unmodified simulation and actuator-position animations from presentation slide 5. |
| [`hardware/`](../hardware/README.md) | Pending STL, CAD, exploded views, animations and verified bill of materials. |
