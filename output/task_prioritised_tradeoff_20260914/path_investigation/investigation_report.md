# Investigation of the two large path deviations

## Finding
Both reference paths can be followed within the existing ideal model and measured hardware limits. The large errors are associated with the numerical controller becoming trapped on an unfavourable configuration branch as deployment reaches a boundary. They are not evidence that the reference paths lie outside the candidate geometry’s reachable workspace.

This is a post-hoc diagnostic, not independent validation of an upgraded controller. Neither tube parameters nor the live viewer were changed.

| Case | Original worst error | Independent pointwise maximum | Alternative start, same 20 steps: densely checked maximum |
|---|---:|---:|---:|
| bank0_path10 | 12.050094 mm | 0.002407 mm | 0.026828 mm |
| bank1_path14 | 37.841387 mm | 0.006424 mm | 0.115255 mm |

## What happened
On bank 0, path 10, the outer exposure reaches 72.5 mm (the front actuator at its full-extension limit). The controller then makes very little progress despite the reference moving on. On bank 1, path 14, the outer exposure reaches zero; that is the exterior-only model boundary, not the front carriage’s physical retraction stop. Much of the controller state then remains nearly unchanged while error accumulates.

The existing solver computes a damped update, clips deployment fractions to bounds and accepts only error-reducing steps. On these branches, extending the iteration budget does not solve the problem. Reducing reference step size alone also fails. This is consistent with branch selection and boundary handling, rather than simply insufficient iterations. It does not establish a unique mathematical cause for every stalled step.

## Diagnostic controls
- Reproduced both saved 81-check traces to within 1e-7 mm.
- Increased iterations from 40 to 200 with unchanged starts and waypoints: both still failed.
- Increased waypoints from 20 steps to 200: both still failed from their original starts.
- Independently solved all 81 reference positions: all residuals below 0.007 mm. This alone would not prove a continuous path.
- Followed each path backwards from a separately solved endpoint, then used the resulting configuration at the original start for forward tracking. Both follow successfully.
- Repeated the forward path with the original 20-step and 40-iteration controller budget, changing only the starting configuration. Checked each resulting segment at 100 subdivisions (2,001 positions). Fine 200-step trajectories were also checked at 20,001 positions. All sampled states satisfy the same coupled constraints.
- A different bounded trust-region solver from the original start fixed the second path, but did not fully resolve the first. Simply swapping the local solver is therefore not a complete strategy.

## What this means for the optimisation decision
The candidate’s observed 43/48 score remains the correct score for the original controller and initialization protocol. Do not rewrite it as an improved score using only these repaired cases. The two large failures are now explained as avoidable numerical path-following failures in this model: feasible low-error alternatives have been constructed. This makes the tube candidate more promising than a diagnosis of geometric unreachability would. It does not yet establish robust path following from arbitrary configurations.

A different start configuration places the tubes differently even though the tip starts at the same position. Reaching that alternative posture from an already fixed physical posture was not tested. If the start posture is prescribed, a posture transition or another feasible branch must be planned, rather than instantly changing actuator positions.

## Recommended next implementation
Add multiple initial IK solutions, path look-ahead to choose a feasible continuous branch, and boundary-aware recovery. Stop or flag execution when predicted residual exceeds the task tolerance instead of continuing with accumulating error. Any recovery must preserve continuous actuator motion and validate intervening positions. Then compare the original and proposed tubes using the same improved controller on fresh, untouched paths. Keep point-to-point and path-following claims separate.

This investigation proves feasibility only at the checked resolution in the ideal exterior model. It does not test physical loads, torsion, friction, rate/acceleration limits, anatomical collision or experimental accuracy.

## Reproduction
```sh
MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/investigate_tradeoff_paths.py
MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/report_tradeoff_paths.py
```
Raw traces, independent pointwise states, reference endpoints, boundary coordinates and dense checks are saved alongside this report.
