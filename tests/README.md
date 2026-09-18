# Model checks

`static_configuration_test_sectioned.py` exercises the active section-aware
kinematic model with aligned, rotated, and straight-section configurations.

`test_inverse_kinematics.py` checks inner/middle/outer endpoint solving, soft
endpoint locks, unreachable targets, deployment limits, and an 80-step path.

`test_tip_control_mapping.py` checks analogue, trigger, and D-pad target mapping,
endpoint interpolation, conditional FREE/SOFT/HARD workspace masks, keyboard
coordinate steps, negative-Z parked-tube geometry, and the front-plate mesh.

`test_motion_planner.py` checks that target planning leaves the live robot
unchanged, falls back to sampled workspace configurations when local IK stalls,
retains diverse alternative solutions, and produces valid direct and
retract/reorient/advance routes, including the fully retracted origin state.

`test_workspace_map.py` checks workspace sampling constraints, endpoint-specific
maps, target projection into the smooth envelope, continuous surface generation,
and the optional cached blue/red/grey diagnostic meshes.

`test_analysis_protocol.py` checks the versioned Stage-2 methodology configuration,
baseline and source provenance, prefix-stable five-DOF canonical sampling,
disjoint training/validation IDs, separate boundary diagnostics, render-only
continuous sweeps, and arbitrary-angle rotational equivariance for every tube
endpoint.

`test_design_analysis.py` checks the compatibility analysis workflow, positional
Jacobians, dexterity, local IK residuals from one fixed fully retracted state,
workspace size metrics, bounded tube-design perturbations, sensitivity records,
and exploratory optimisation.

`test_stage2_analysis.py` checks the fixed physical Jacobian scale and central
differences, boundary masking, exact swept-annular volumes, internal voids,
common-region coverage, spatial weighting, canonical-copy deduplication,
stratified target selection, and the separation of numerical convergence from
the 0.5 mm evaluation threshold.

Run it without opening plot windows:

```bash
MPLBACKEND=Agg ./.venv/bin/python tests/static_configuration_test_sectioned.py
./.venv/bin/python tests/test_inverse_kinematics.py
./.venv/bin/python tests/test_tip_control_mapping.py
./.venv/bin/python tests/test_motion_planner.py
./.venv/bin/python tests/test_workspace_map.py
./.venv/bin/python tests/test_analysis_protocol.py
./.venv/bin/python tests/test_design_analysis.py
./.venv/bin/python tests/test_stage2_analysis.py
```

The later protocol checks are also runnable from a fresh checkout:

```bash
./.venv/bin/python tests/test_stage4_extension.py
./.venv/bin/python tests/test_stage4_1_support.py
./.venv/bin/python tests/test_stage5a_protocol.py
```

`test_stage4_extension.py` checks the frozen convergence gates;
`test_stage4_1_support.py` checks physical-volume IK support requirements.
`test_stage5a_protocol.py` validates the frozen optimisation protocol and all 128
bound corners. To additionally audit the archived local Stage-4/4.1 datasets, run:

```bash
./.venv/bin/python tests/test_stage5a_protocol.py --with-evidence
```

The evidence audit requires `results/evaluation/` files that are deliberately not
included in Git. It is explicitly opt-in; ordinary protocol checks do not require
those local research outputs.

## Historical live profile check

Run `python tests/test_viewer_profiles_live.py` on a graphical desktop to check the older root viewer. It uses the current earlier measured candidate from its JSON configuration. These files are standalone scripts, not a unittest discovery suite; invoke the commands above individually. The current final-report viewers have separate unittest suites documented in the research guide.
