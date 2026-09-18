# Final CTR geometry-method comparison

Neither frozen search finalist passed all declared selection and validation gates. No replacement is recommended from this study.

The original and current proposed simulator configurations were preserved. All results are predictions of the same ideal exterior curvature-superposition model and unchanged numerical IK/controller. No clinical or experimental performance is established.

## Final assessment on fresh targets and paths

| Configuration | Global IK | Weak-region IK | Difficult-region IK | Completed paths | Pooled tracking P95 (mm) | Worst checked error (mm) |
|---|---|---|---|---|---|---|
| Original | 62.61% | 54.25% | 46.75% | 231/256 | 0.1468 | 66.115 |
| Current proposed | 67.24% | 58.08% | 58.91% | 226/256 | 0.1712 | 84.429 |
| NSGA-II finalist | 66.38% | 57.47% | 56.35% | 230/256 | 0.1533 | 80.096 |
| Staged-search finalist | 70.01% | 61.91% | 62.62% | 237/256 | 0.1301 | 76.274 |

Each design was evaluated on two new banks, each containing 8,192 global, 2,048 weak-region, 2,048 difficult-region and 1,024 corridor targets, 262,144 workspace states and 128 paths. Targets are shared between designs. The global score uses only global targets; overlapping regional groups are not counted twice. Paths comprise representative, weak-endpoint and actuator/model-boundary stress strata. Aggregate path performance refers to this declared mixture, not a clinical distribution. Each route uses the unchanged 20 control steps and 40-iteration warm-start solver and is checked at 2,001 positions. The dense check does not certify continuous-time feasibility between samples.

The pooled P95 above uses all final path samples. It is not the mean of bank-level P95 values. The old 44/48 versus 43/48 counts belong to the earlier development study and are not mixed with these new tests. All failed trajectories remain included.

## Frozen designs

| Configuration | Total inner/middle/outer (mm) | Curved middle/outer (mm) | Precurvature middle/outer (1/m) |
|---|---|---|---|
| Original | 350.0000 / 170.0000 / 80.0000 | 90.0000 / 65.0000 | 19.1200 / 14.0400 |
| Current proposed | 350.0000 / 177.5000 / 87.5000 | 90.0000 / 65.0000 | 21.3700 / 14.0400 |
| NSGA-II finalist | 350.0000 / 176.6507 / 80.4605 | 85.0424 / 70.7106 | 19.1200 / 14.0400 |
| Staged-search finalist | 350.0170 / 180.8477 / 89.6528 | 82.8119 / 72.0831 | 21.2253 / 14.2636 |

The inner element remains straight. Straight material lengths equal total minus curved length. Digits permit numerical reproduction and do not imply fabrication accuracy. Curved material exceeding maximum feasible exposure is exterior-equivalent; it is not a physically validated interchangeable design.

## Acceptance outcomes

- **proposed:** path_completion, tracking_p95, gap_108_5_sensitivity
- **nsga2:** path_completion, tracking_p95, repeat0_global_isotropy_p10, gap_108_5_sensitivity
- **staged:** path_completion, repeat0_global_isotropy_p10, selection_passed, gap_108_5_sensitivity

Every gate is required. A diagnostic finalist that failed dense selection cannot be rescued by a favourable held-out result. Statistical acceptance uses conservative paired intervals, with 97.5% intervals for the two method finalists; insufficient precision is not evidence of equivalence. Region and path margins are study-specific engineering choices, not literature-derived clinical requirements. The intervals describe variation under these sampled numerical test banks, not uncertainty in the physical robot. The P95 interval uses 400 paired whole-path bootstrap resamples; its tail estimates are approximate, and a result close to the margin should be treated cautiously. Detailed point estimates, intervals, individual gates and the 108.5 mm gap sensitivity are in `outcome.json` and `gap_sensitivity.json`.

Original-only equal-budget final supported-cell repeat loss: 0.000%. Coverage is also checked at 2.5, 5 and 10 mm grids. Full original-cell losses, including sparse boundary cells, remain reported in the outcome; the acceptance mask was frozen from adequately sampled original development cells. No candidate can improve a regional mean by deleting its failed original targets.

| Configuration | Completion change (pp) | Completion interval (pp) | P95 change (mm) | P95 bootstrap interval (mm) |
|---|---|---|---|---|
| Current proposed | -1.95 | [-9.75, +5.93] | +0.0244 | [-0.0591, +0.1338] |
| NSGA-II finalist | -0.39 | [-8.87, +8.11] | +0.0065 | [-0.0713, +0.1416] |
| Staged-search finalist | +2.34 | [-5.19, +9.78] | -0.0167 | [-0.0794, +0.0235] |

Failure of a confidence-interval gate does not establish that the candidate is worse. It means this study did not establish the declared margin. In particular, the staged finalist improved observed completion, but its interval extends below the permitted two-percentage-point loss. The NSGA-II finalist lost one completed path, and its P95 interval did not rule out an increase greater than 0.05 mm. Both finalists also failed the global lower-tail isotropy safeguard in one repeat. The staged finalist had already failed selection and therefore remains diagnostic, irrespective of favourable held-out averages. Worst checked errors remain substantial for all four configurations and must be reported alongside P95.

## Workspace and positional dexterity

| Configuration | Maximum radius (mm) | Maximum Z (mm) | Maximum tip distance (mm) | Supported reference cells retained | Mean global isotropy | Global isotropy P10 |
|---|---|---|---|---|---|---|
| Original | 113.15 | 171.23 | 173.38 | 100.00% | 0.3485 | 0.1532 |
| Current proposed | 115.09 | 172.70 | 173.39 | 100.00% | 0.3626 | 0.1591 |
| NSGA-II finalist | 117.23 | 171.09 | 172.71 | 100.00% | 0.3526 | 0.1417 |
| Staged-search finalist | 121.62 | 171.47 | 173.34 | 100.00% | 0.3673 | 0.1526 |

Values in this table are the mean of the two bank-specific estimates. Acceptance used each repeat separately; averaging must not conceal a failed repeat. Supported occupancy retention does not mean every original point, orientation or continuous route remains reachable. All reach values describe the sampled ideal model.

## Method comparison

| Method | Seed | Evaluator slots | Unique completed | Unresolved | Geometry rejections | Completed evaluator minutes | Wall minutes | Feasible hypervolume |
|---|---|---|---|---|---|---|---|---|
| nsga2 | 71011 | 256 | 255 | 1 | 22 | 30.5 | 30.8 | 0.72484 |
| nsga2 | 71012 | 256 | 253 | 0 | 35 | 32.0 | 32.3 | 0.74750 |
| nsga2 | 71013 | 256 | 253 | 0 | 31 | 31.5 | 31.8 | 0.70834 |
| staged | 71011 | 256 | 254 | 2 | 87 | 34.8 | 35.1 | 0.71854 |
| staged | 71012 | 256 | 255 | 1 | 85 | 35.1 | 35.4 | 0.74682 |
| staged | 71013 | 256 | 256 | 0 | 88 | 35.6 | 35.9 | 0.74728 |

Each method used three seeds and 256 evaluator slots per seed. Analytically infeasible geometry proposals were replaced using the same bounded Sobol reserve rule and logged separately. Cached exterior-equivalent evaluations and numerical sampling failures are retained in the records. Completed-evaluator minutes exclude unsuccessful sampling attempts; wall minutes include them and are measured from launch-log creation to completion-marker time. Runs execute concurrently, so these are not isolated CPU benchmarks. Counts refer to the full evaluator, not a trace of every internal FK call. Feasible hypervolume is a summary of the two screen objectives against a fixed failure-fraction reference point (1.01, 1.01); it is not workspace volume. It is measured on development samples and does not establish generalisation or global optimality.

Both methods used the same original/proposed seed designs, objectives, geometry bounds, evaluator settings, acceptance margins and dense reassessment allowance. This is a controlled comparison of candidate-generation strategies, not an exact rerun of the historical 1:2:2 regional scoring algorithm. Previous inspected targets, paths and maps are development evidence. The finalists were frozen before opening the final validation banks. Three search seeds provide only a limited repeatability check.

## Interpretation and limitations

The study distinguishes a better candidate from a better optimisation method. A single successful run would not establish general superiority of NSGA-II. Small changes or plateaus under the allocated budget cannot establish convergence to the global optimum. If neither finalist passes, retain the existing references and report the failed criteria and trade-offs without changing thresholds after validation.

The hardware remains 100 mm nominal stroke per actuator, 71.5 mm actuator bodies and 8.5–108 mm coupled gaps. The108.5mm sensitivity case is disqualifying only: it cannot rescue a primary failure. Lengths remain provisional chuck-to-tip material lengths with no 35 mm offset. Internal guidance, retention, rotation limits and manufacture-certified curvature bounds remain unresolved. The model omits torsion, friction, elastic stability, contact, anatomy, loading and physical validation. Positional isotropy uses the existing normalisation and does not guarantee feasible motion in every direction at an actuator stop.

The reporting script computes annular occupancy volume from explicit radial/axial cell pairs so negative axial coordinates, if present in broad candidates, cannot corrupt a packed cell-ID volume estimate. Volume is descriptive and was not a search objective or acceptance gate. Fixed-reference coverage and isotropy use the unchanged reference-cell identities.

## Reproducibility and files

- `protocol.json`, `preflight_amendment.json`, `environment.json`: frozen settings and pre-search corrections.
- `source_snapshot/`, `source_hashes.json`, `protected_hashes.json`: implementation and preservation evidence.
- `checks.json`, `pilot.json`: kernel checks, runtime and original-only occupancy calibration.
- `runs/`: complete proposals, rejections, objective histories and shortlists.
- `selection/`, `selection_records.json`, `frozen_champions.json`: selection evidence and frozen candidates.
- `banks/`, `validation/`, `sensitivity/`: target coordinates/source states, paths, resulting states/errors and workspace samples.
- `outcome.json`, `reported_metrics.json`: decision gates and reported numbers.
- `figures/`: standalone comparison figures, PDFs and PNGs.
- `candidate_nsga2.json`, `candidate_staged.json`: standalone study exports; neither is installed in a simulator.

Run the checks and stages from the project root using `.venv/bin/python tools/final_geometry_comparison.py COMMAND`. Search commands additionally take `nsga2` or `staged` and one of 71011, 71012, 71013. Use the existing study only to resume unchanged settings; use a new study directory and new reserved seeds for a different protocol. Report generation uses `.venv/bin/python tools/report_final_geometry_comparison.py`. The NSGA-II library is isolated under the study's dependencies folder; existing model libraries retain precedence.

Method references: Deb et al. (2002), https://doi.org/10.1109/4235.996017 ; Bergeles et al. (2015), https://doi.org/10.1109/TRO.2014.2378431 ; Baykal et al. (2015), https://doi.org/10.1109/IROS.2015.7353999 . These support the method discussion, not the numerical acceptance margins.
