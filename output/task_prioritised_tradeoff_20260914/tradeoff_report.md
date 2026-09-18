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
| Precurvature (1/m) | 0 | 21.3700 | 14.0400 |

Digits reproduce the simulation; they are not manufacturing accuracy. Curved material longer than maximum exposure is not identifiable by this exterior model.

## Independent results
Rates below are fractions; lengths mm and volume cm³. Values average two independent banks. Raw repeats remain in validation_records.json.

| Metric | Original | Previous candidate | Proposed | Proposed − original |
|---|---:|---:|---:|---:|
| global_ik_success | 0.620624 | 0.704497 | 0.689601 | +0.068977 |
| weak_ik_success | 0.536645 | 0.626479 | 0.580338 | +0.043694 |
| corridor_ik_success | 0.684752 | 0.788897 | 0.744327 | +0.059576 |
| trajectory_completion | 0.916667 | 0.895833 | 0.895833 | -0.020833 |
| trajectory_p95_mm | 0.086630 | 0.406068 | 0.153062 | +0.066433 |
| global_diagnostic_success | 0.990306 | 0.971826 | 0.991617 | +0.001311 |
| weak_diagnostic_success | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| corridor_diagnostic_success | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| global_isotropy_mean | 0.362652 | 0.361566 | 0.382324 | +0.019672 |
| global_isotropy_p10 | 0.198698 | 0.167750 | 0.212519 | +0.013821 |
| weak_isotropy_mean | 0.190306 | 0.220340 | 0.214340 | +0.024034 |
| weak_isotropy_p10 | 0.127961 | 0.142855 | 0.142819 | +0.014859 |
| corridor_isotropy_mean | 0.177322 | 0.228584 | 0.208146 | +0.030824 |
| corridor_isotropy_p10 | 0.121355 | 0.132429 | 0.128216 | +0.006861 |
| volume_cm3 | 2338.915731 | 2445.926230 | 2513.666822 | +174.751091 |
| global_cell_retention | 1.000000 | 0.981800 | 1.000000 | +0.000000 |
| weak_cell_retention | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| corridor_cell_retention | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| tip_distance_max_mm | 172.577116 | 170.747587 | 172.586442 | +0.009326 |
| radial_extent_mm | 111.375122 | 112.673824 | 114.112085 | +2.736963 |
| z_max_mm | 170.163858 | 170.190101 | 171.653082 | +1.489224 |
| z_min_mm | 64.026323 | 61.680969 | 63.057156 | -0.969166 |
| global_ik_p95_mm | 88.471266 | 83.235140 | 84.695572 | -3.775693 |
| weak_ik_p95_mm | 57.735932 | 55.334184 | 57.734541 | -0.001391 |
| corridor_ik_p95_mm | 9.516304 | 9.230546 | 9.119563 | -0.396741 |

For success rates, multiply differences by 100 to obtain percentage points. Isotropy, volume and reach percentage changes use the baseline denominator.

## Paired uncertainty
Paired bootstrap intervals quantify finite target-sample uncertainty only; they do not quantify model error or manufacturing uncertainty.
- global: +6.90 percentage points; paired 95% interval [+5.61, +8.15].
- weak: +4.37 percentage points; paired 95% interval [+2.95, +5.94].
- corridor: +5.96 percentage points; paired 95% interval [+3.89, +8.05].

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
