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

Run it without opening plot windows:

```bash
MPLBACKEND=Agg ./.venv/bin/python tests/static_configuration_test_sectioned.py
./.venv/bin/python tests/test_inverse_kinematics.py
./.venv/bin/python tests/test_tip_control_mapping.py
./.venv/bin/python tests/test_motion_planner.py
./.venv/bin/python tests/test_workspace_map.py
```
