# Run the interactive simulator

The main launcher compares the **original tubes** with the final report's **proposed candidate t027** using the same measured hardware constraints. It opens a desktop window; GitHub displays only the animation and source files.

## Setup

Follow the environment and dependency commands in the [main README](../README.md#install-and-launch). Run commands from the repository folder with that environment active:

```bash
python launch_simulation.py --configuration original
python launch_simulation.py --configuration proposed
```

The first launch samples 12,000 feasible states and caches the endpoint workspaces. The cache is saved under `proposed_tradeoff_simulator/exports/viewer_workspace_cache/`. Cached files and pose exports are local outputs and are ignored by Git.

Optional settings:

```bash
python launch_simulation.py --configuration proposed --mode hardware-108.5
python launch_simulation.py --configuration original --joint-only
python launch_simulation.py --configuration proposed --waypoints path/to/export.csv
```

`hardware-108.5` uses the report's alternative maximum adjacent gap. The normal mode uses 108 mm. A compatible CSV loads its last pose; it does not play an entire trajectory. Every row must match the active profile and satisfy the limits before the viewer changes pose.

## Controls

| Task | Keyboard / mouse | PS5 controller |
| --- | --- | --- |
| Joint / Cartesian mode | Tab | L3 |
| Select a tube / endpoint | 1 / 2 / 3 | L1 / R1 cycle |
| Joint translation / rotation | W/S / A/D | D-pad Left/Right / Up/Down |
| Move Cartesian target | X/Y/Z selects axis; Left/Right adjusts | Left stick X/Y; L2/R2 along endpoint direction |
| Fine Cartesian movement | Small arrow-key steps | D-pad; hold Square for fine Z and slower analogue movement |
| Preview target, then execute | Enter twice | Cross twice |
| Cancel target / stop movement | Escape | Circle |
| Change route during preview | M | D-pad Up/Down |
| Cycle IK solutions during preview | Left/Right | D-pad Left/Right |
| Orbit / zoom | Drag / scroll | Right stick or mouse |
| Independent / coordinated carriages | C | Use C on keyboard |
| Original / proposed tubes | F7 | Use F7 on keyboard |
| 108 / 108.5 mm gap | F6 | Use F6 on keyboard |
| Reset / export / close | R / E / Q | See the live sidebar |

The on-screen sidebar gives the active bindings. Endpoint locks, waypoint saving and undo are also available there. In Cartesian mode a first confirmation calculates a preview; a second starts the simulated motion. Alternative IK solutions and direct/retract routes can give different intermediate shapes for the same endpoint target.

F6 and F7 reset the pose, plans and undo history. Export any poses you want to retain before switching. In coordinated joint mode, neighbouring carriages can follow the selected one by the minimum displacement needed to keep the coupled limits satisfied. No commands are sent to physical motors.

## Understanding the display

- Blue, green and orange represent the inner, middle and outer elements. The inner element is a straight wire.
- The outside/front plate surface is the reference origin. Exposed material length is different from endpoint XYZ position.
- The actuator diagram shows carriage travel and gaps. The three nominal strokes are 100 mm; actuator bodies are 71.5 mm long and adjacent gaps are constrained to 8.5–108 mm.
- Reset is 75/0/0 mm exterior exposure for both report configurations. It is not all carriages at their rear stops.
- The workspace is a smooth approximation from sampled states. A target inside it may still fail the available IK search. A failed solve does not prove geometric unreachability.
- Behind-plate tube geometry is schematic. The model excludes negative exposures and enforces inner ≥ middle ≥ outer exterior exposure.

## Configuration files

`proposed_tradeoff_simulator/tube_parameters.py` defines the original tubes. `proposed_tradeoff_simulator/optimised_configuration.json` contains t027. The internal `optimised` key is retained for compatibility; the design did not pass every validation criterion.

`measured_hardware_simulator/` contains the original measured-hardware snapshot and a different earlier candidate. It supports the study's provenance and imports. Use the root launcher for the final-report comparison.

## If it does not open

- Confirm the environment is active and dependencies installed with `python -m pip install -r requirements.txt`.
- Use a local graphical desktop with OpenGL support. A text-only server cannot display the interactive window. [Saved results](../docs/dissertation/README.md) and the numerical checks remain usable without a viewer.
- Wait for the first workspace calculation to finish. Controller connection is optional; keyboard and mouse work without it.
- If a CSV is rejected, check that it came from the same geometry and gap profile. Older unlabelled exports are deliberately rejected.
- For a fresh cache, close the application and remove only its `exports/viewer_workspace_cache/` directory. It is regenerated on launch.

The release was checked on macOS with Python 3.14.3. Commands for Windows and Linux are provided, but those platforms were not tested during this release.
