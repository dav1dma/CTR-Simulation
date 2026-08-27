# CTR Simulation

An interactive simulator for exploring the movement and workspace of a concentric-tube robot. The tubes can be extended and rotated using a PS5 controller, keyboard, or mouse, while the robot’s shape, tip position, and orientation update in real time.

The workspace plot below shows the tip positions reached across 10,000 simulated tube configurations.

![Section-aware CTR workspace](docs/images/ctr_workspace_sectioned.png)

## Run the simulator

Create the project environment and install its dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Start the current viewer:

```bash
./.venv/bin/python interactive_ctr_vispy.py
```

The full controller and keyboard key is displayed in the viewer sidebar.
The status and parameter interface stays consistent between Joint and Tip mode,
including all three endpoint XYZ positions. Only the Controls section switches
to show the bindings relevant to the active mode.

To try constrained Cartesian tip-position control while keeping the original
joint-control viewer unchanged, run:

```bash
./.venv/bin/python interactive_ctr_tip_control.py
```

In this version, L3 or Tab switches between joint and tip-control modes. In tip
mode, L1/R1 cycle the controlled inner, middle, or outer endpoint. The left
stick moves its red target in global X/Y, L2/R2 move it backwards/forwards along
the endpoint's current direction, and the D-pad provides fine X/Y movement.
Holding Square changes D-pad Up/Down to fine Z movement and also enables slower
analogue movement. Press Cross once to calculate and preview a solution, then
press Cross again to run the smooth simulated movement. Circle cancels a target
or stops an executing movement. Keyboard users can choose an endpoint with
1/2/3, select X/Y/Z and adjust with Left/Right, use Enter to confirm, and Escape
to cancel. Camera orbit uses the right stick or mouse drag; zoom uses the mouse
wheel. Normal target motion is 40 mm/s; Square reduces it to 5 mm/s for
precision placement.

While a solution preview is displayed, D-pad Up/Down chooses the direct or
three-stage retract/reorient/advance route, and D-pad Left/Right cycles through
meaningfully different IK configurations for the same target. Alternative tube
shapes appear as translucent ghosts. Keyboard users can press M for the route
and Left/Right for the solution. The retract route brings all exposed lengths to
zero, changes the rotations at the plate origin, and advances to the confirmed
IK configuration.

Endpoints remain FREE after reaching a target. In tip mode, tap Options to cycle
the selected endpoint through FREE, SOFT LOCK, and HARD LOCK; hold Options to
export. Soft locks permit a 5 mm error and hard locks use a 1.5 mm tolerance.
The sidebar reports each constraint and error. Three XYZ endpoint targets create
nine constraints for only six actuator inputs, so an arbitrary three-point
shape is not always exactly achievable. Reset releases all locks.

The viewer uses the plate reference at Z=0. Exposed robot geometry and the
sampled workspace lie mainly at positive Z, while the portions of each tube
parked behind the plate are drawn as straight nested shafts at negative Z. At
full retraction the curved backbone collapses to the origin, but the complete
colour-coded tube lengths remain visible behind the translucent plate.

The preview includes translucent final tube positions. A cyan path represents
direct simultaneous actuator interpolation, while a purple path represents the
retract/reorient/advance sequence. These are simulation aids, not
collision-checked surgical trajectories; the current model does not yet
represent a straight introducer sheath. Anatomy collision checking, sheath
constraints, and scan registration are later integration stages.

The live viewer displays the earlier smooth, translucent 360-degree workspace
boundary. It changes with the selected endpoint because the outer and middle
tubes have smaller workspaces than the inner tube. Every controller and keyboard
target step is constrained to this surface, so the red target cannot leave the
displayed reachable envelope. The boundary and restart maps use 12,000 shared
valid actuator configurations. Press Triangle to hide or show the workspace
together with the orientation guides. When other endpoints are locked, green
points show sampled configurations satisfying their strict tolerances and amber
points show the additional positions available within the relaxed soft-lock
tolerance. A diamond marker identifies the nearest sampled feasible projection
when the red requested target lies away from that conditional region. Local
Jacobian null-space samples supplement the global cache so narrow hard-lock
regions do not disappear solely because the cached map is coarse.

## Project layout

- `interactive_ctr_vispy.py` — current interactive application
- `interactive_ctr_tip_control.py` — separate inverse-kinematics tip controller
- `ctr_inverse_kinematics.py` — constrained numerical tip-position solver
- `ctr_motion_planner.py` — confirmed target planning and smooth execution path
- `ctr_workspace_map.py` — sampled workspace generation and loading
- `CTR_superPosKin_fun_sectioned.py` — section-aware forward kinematics
- `tube_parameters.py` — tube geometry and material parameters
- `tests/` — current model checks
- `tools/` — controller diagnostics and workspace generation
- `assets/cad/` — CAD assets prepared for the digital-twin stage
- `config/` — future CAD joint and actuator calibration data
- `docs/images/` — selected documentation images
- `legacy/` — superseded implementation retained for reference

Generated screenshots are written to `exports/`. Workspace datasets and plots
created by the generator are written to `results/`. Both directories are
excluded from Git.

## Validate the current model

```bash
MPLBACKEND=Agg ./.venv/bin/python tests/static_configuration_test_sectioned.py
```

## Generate a workspace

```bash
./.venv/bin/python tools/workspace_generator.py
```

The generator uses 10,000 configurations by default and may take some time.
The compact sampled map can be regenerated with:

```bash
./.venv/bin/python tools/generate_workspace_map.py
```

Regenerate the optional legacy three-zone diagnostic map with:

```bash
./.venv/bin/python tools/generate_reachability_zones.py
```

Regenerate the endpoint-specific maps used by tip control with:

```bash
./.venv/bin/python tools/generate_endpoint_workspace_maps.py
```

## CAD integration

See `assets/cad/README.md` for the required assembly layout. Moving components
must remain separate so their linear and rotary transforms can be driven by the
simulator's deployment and rotation values.
