# Benchmark-prioritised tube configuration study

## Recommendation and scope
Prioritise reliable reaching in the declared target regions and continuous path completion, supported by weak-region positional dexterity. Treat global workspace volume and peak reach as secondary within explicit loss limits. The forward corridor and weak region are frozen mathematical benchmarks, not a surgical task or anatomy. Different task targets can change this recommendation.

Candidate passes all predeclared held-out trade-off limits in both repeats: **False**. No claim of a global optimum or physical validation. The live simulator and previous configuration were preserved.

## Parameters
Order is inner/rear, middle/middle, outer/front. Lengths are provisional chuck-front-to-tip material lengths.

| Parameter | Inner | Middle | Outer |
|---|---:|---:|---:|
| Total (mm) | 350.0000 | 177.5000 | 87.5000 |
| Straight (mm) | 350.0000 | 87.5000 | 22.5000 |
| Curved (mm) | 0 | 90.0000 | 65.0000 |
| Precurvature (1/m) | 0 | 20.6200 | 13.6650 |

Digits reproduce the simulation; they are not manufacturing accuracy. Curved material longer than maximum exposure is not identifiable by this exterior model.

## Independent results
Rates below are fractions; lengths mm and volume cm³. Values average two independent banks. Raw repeats remain in validation_records.json.

| Metric | Original | Previous candidate | Proposed | Proposed − original |
|---|---:|---:|---:|---:|
| global_ik_success | 0.632761 | 0.708067 | 0.686269 | +0.053508 |
| weak_ik_success | 0.542261 | 0.619531 | 0.577524 | +0.035263 |
| corridor_ik_success | 0.690137 | 0.797465 | 0.750565 | +0.060428 |
| trajectory_completion | 0.958333 | 0.958333 | 0.875000 | -0.083333 |
| trajectory_p95_mm | 0.098574 | 0.117974 | 0.704251 | +0.605677 |
| global_diagnostic_success | 0.989680 | 0.970958 | 0.991457 | +0.001777 |
| weak_diagnostic_success | 1.000000 | 1.000000 | 0.998246 | -0.001754 |
| corridor_diagnostic_success | 1.000000 | 0.998058 | 1.000000 | +0.000000 |
| global_isotropy_mean | 0.359180 | 0.358248 | 0.370247 | +0.011067 |
| global_isotropy_p10 | 0.193818 | 0.163300 | 0.202644 | +0.008825 |
| weak_isotropy_mean | 0.190366 | 0.220323 | 0.206882 | +0.016516 |
| weak_isotropy_p10 | 0.126802 | 0.143270 | 0.140552 | +0.013750 |
| corridor_isotropy_mean | 0.177537 | 0.227386 | 0.202223 | +0.024685 |
| corridor_isotropy_p10 | 0.124168 | 0.131831 | 0.126706 | +0.002538 |
| volume_cm3 | 2375.633095 | 2456.529106 | 2398.802341 | +23.169246 |
| global_cell_retention | 1.000000 | 0.996360 | 1.000000 | +0.000000 |
| weak_cell_retention | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| corridor_cell_retention | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| tip_distance_max_mm | 172.451437 | 170.518145 | 172.414905 | -0.036532 |
| radial_extent_mm | 113.234136 | 114.202235 | 113.152919 | -0.081217 |
| z_max_mm | 169.801523 | 169.549515 | 171.016459 | +1.214936 |
| z_min_mm | 63.962247 | 61.840233 | 63.746721 | -0.215526 |
| global_ik_p95_mm | 85.358788 | 81.616183 | 82.487149 | -2.871638 |
| weak_ik_p95_mm | 54.868453 | 54.125361 | 54.867216 | -0.001237 |
| corridor_ik_p95_mm | 9.860264 | 8.841502 | 10.018308 | +0.158043 |

For success rates, multiply differences by 100 to obtain percentage points. Isotropy, volume and reach percentage changes use the baseline denominator.

## Paired uncertainty
Paired bootstrap intervals quantify finite target-sample uncertainty only; they do not quantify model error or manufacturing uncertainty.
- global: +5.35 percentage points; paired 95% interval [+4.08, +6.60].
- weak: +3.53 percentage points; paired 95% interval [+2.19, +5.03].
- corridor: +6.04 percentage points; paired 95% interval [+3.91, +8.21].

## Method and limitations
The search screens prior candidate interpolations and new Sobol designs, uses equal target banks and the existing solver, then freezes one design before two new validation banks. Each bank has 2,048 global targets and 512 targets in each priority region, 131,072 feasible configurations per design, and 24 reference joint-space paths. Spatial metrics use a fixed 5 mm radial/axial annular occupancy grid and fixed original reference regions, weighted by annular volume. This rotational representation assumes unrestricted common rotation. Sampled occupancy and a smoothed envelope do not guarantee every enclosed point is reachable.

Fixed-start IK uses 40 iterations and 0.5 mm success. Two nearest-start attempts are a separate diagnostic and do not prove geometric reachability or real-time control performance. Paths use warm starts, 21 waypoints and intermediate checks against continuous original reference paths. Only 48 paths are evaluated; identical completion is evidence for these paths, not all possible motions.

Isotropy uses a fixed normalization and s_min/s_max of the positional Jacobian. It measures local directional balance, not orientation dexterity, force capability or feasible directional motion at a stop. Means, lower tails, coverage, target success and paths must be interpreted together.

All 100 mm end stops, 71.5 mm actuator bodies and coupled 8.5–108 mm gaps are applied. The 108.5 mm sensitivity case is saved separately. Negative exposures and tips inside another tube are excluded, rather than silently clipped. This excludes some possibly physical configurations. Installed tube retention, internal guidance, rotation limits, friction, elastic torsion and stability, manufacturing constraints and loaded accuracy remain unresolved. Bounds are exploratory; the candidate is not fabrication-ready.

The ranking and allowed losses are explicit engineering choices saved before selection, not externally established clinical requirements. Gains and losses describe the combined parameter change; this study does not attribute each gain to a particular length or curvature without a separate controlled ablation.

## Reproduction
From the project folder:
```sh
MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/task_prioritised_tradeoff_search.py search
MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/task_prioritised_tradeoff_search.py validate
MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/report_task_prioritised_tradeoff.py
```
Screen records are resumable. Keep previous study inputs at their recorded paths. Protocol, candidate selection, raw states/points/IK/path errors, sensitivity, and parameter JSON are saved alongside this report. Existing simulator configuration was not changed.

## Sequential validation disclosure
This is the reserve candidate selected from the original training-qualified list after the first candidate failed path protection. New seeds and the unchanged acceptance thresholds are recorded in protocol_amendment.json. Reproduce this round with `MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/validate_tradeoff_reserve.py`, then `MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/report_task_prioritised_tradeoff.py reserve_validation`. Both candidates failed the combined rule. See ../recommendation.md for the conditional point-to-point interpretation.
