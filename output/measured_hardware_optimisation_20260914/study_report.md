# Measured-hardware tube configuration study

Selected candidate: [348.16, 181.98, 92.14, 95.02, 65.91, 22.03, 13.98] in protocol variable order. Independent validation acceptance: True.
The original physical tubes remain the baseline. This is a bounded numerical search, not a global optimum or a fabrication certification.

## Parameters

| Tube | Chuck-to-tip total (mm) | Straight (mm) | Distal curved (mm) | Precurvature (1/m) |
|---|---:|---:|---:|---:|
| Inner | 348.16 | 348.16 | 0 | 0 |
| Middle | 181.98 | 86.96 | 95.02 | 22.03 |
| Outer | 92.14 | 26.23 | 65.91 | 13.98 |

Diameters, 75 GPa model stiffness and straight inner wire retained. Three total lengths, two curved lengths and two precurvatures were explored. No stock-length assumption or 35 mm gripping offset was used. Positive behind-plate straight length is not proof of sufficient retention. Curved material within motors/guides is idealised as constrained straight internally; actual passage, support, friction, torsion, elastic stability and manufacturable curvatures are unresolved. The selected lengths describe distal material from the chuck; any gripping engagement or rear overhang must be specified separately.

The middle curved section (95.02 mm) exceeds its maximum exposure (86.98 mm). Any middle curved length between that exposure and the total length is exterior-equivalent in this model. The reported value is the searched representative, not a uniquely identified optimum. Two-decimal model parameters are not manufacturing tolerances.

| Actuator | Exterior material exposure range (mm) | Carriage position from rear stop (mm) |
|---|---:|---:|
| Inner/rear | 73.16–173.16 | 0.00–100.00 |
| Middle | 0.00–86.98 | 13.02–100.00 |
| Outer/front | 0.00–77.14 | 22.86–100.00 |

These are coordinate extrema over feasible coupled states, not independently available simultaneous strokes. The maximum middle-to-front gap remains only 18.34 mm under exterior tip ordering (original:18.5 mm). This candidate does not resolve that independent-motion restriction. The rear-to-middle gap is limited by ordering to 94.68 mm, so increasing the mechanical maximum to108.5 mm does not expand this candidate's exterior domain.

## Independent fixed-start IK

| Region | Original success | Candidate success | Original P95 (mm) | Candidate P95 (mm) |
|---|---:|---:|---:|---:|
| global | 66.20% | 73.48% | 86.948 | 81.475 |
| weak | 57.49% | 66.52% | 51.063 | 51.049 |
| corridor | 67.84% | 78.53% | 9.888 | 9.345 |

Targets are identical, actual baseline-reachable samples in frozen baseline-defined cells; global, weak and corridor sets overlap spatially. This is a benchmark, not an anatomical task. Each set uses cell-stratified samples and annular-volume weights. There are 2,048 global, 512 weak and 512 corridor target draws. Success means <=0.5 mm residual, not physical accuracy. All failures remain in residual statistics. The 40-iteration solver starts at the most-retracted feasible exterior state with zero rotations for each design. This can be singular or close to constraints. These results must not be compared directly with previous chapters using different targets and obsolete hardware bounds. Batch solver equations match the viewer with the documented study settings; interactive solver defaults and its additional control features differ.

## Positional isotropy on fixed baseline cells

| Region | Original mean | Candidate mean | Original spatial P10 | Candidate spatial P10 |
|---|---:|---:|---:|---:|
| global | 0.3489 | 0.3437 | 0.1727 | 0.1309 |
| weak | 0.1905 | 0.2201 | 0.1265 | 0.1435 |
| corridor | 0.1779 | 0.2180 | 0.1226 | 0.1236 |

Cell size 5 mm in radius and height, minimum five observations. Missing/unsupported cells score zero in the fixed reference. Scales [350,170,80,pi,pi,pi] are fixed across designs. Isotropy describes the ideal local Jacobian, not guaranteed available motion at a physical boundary. P10 of spatial cell medians is distinct from the success-weighted IK-solution metric used in selection. The latter includes zero for failed targets and can have zero P10 when failures exceed 10%.

## Attribution, same hardware throughout

| Configuration | Global success | Weak-region success | Corridor success |
|---|---:|---:|---:|
| original | 66.20% | 57.49% | 67.84% |
| optimised | 73.48% | 66.52% | 78.53% |
| lengths_only | 70.12% | 66.36% | 65.08% |
| curves_only | 57.88% | 53.51% | 62.69% |

Lengths-only applies selected totals to original curved sections and curvatures. Curves-only applies selected curved sections and curvatures to original totals. Interactions mean these effects need not add. No increased-stroke benefit is claimed: all cases retain 100 mm carriage strokes, 71.5 mm bodies and 8.5–108 mm coupled gaps.

The two-nearest-start diagnostic is saved separately in validation_results.json. It uses a candidate-specific independent feasible sample bank and 60 iterations per start. It diagnoses local-solver sensitivity and does not prove geometric unreachability for failed targets. It is not mixed with primary fixed-start scores.

With that diagnostic, global success is 99.35% for original and 98.80% for candidate. Thus improved fixed-start success should not be interpreted as an equally large geometric-coverage gain. Both designs solve every sampled weak-region and corridor target in this diagnostic.

## Workspace and sensitivities

Original 65,536-sample annular occupancy estimate: 2306.71 cm³; candidate: 2417.06 cm³. Candidate retention of the frozen baseline's occupied cells: 91.29%. Occupancy depends on sample size and grid and does not fill holes or prove all enclosed points reachable. Full revolutions assume unrestricted axial rotations.

Raw physical states, tips, occupancy, target positions, returned IK states and errors are retained. sampling_sensitivity.json reports an independent 131,072-sample run at 2.5 and 5 mm grids. gap_sensitivity.json repeats fixed targets with 108.5 mm maximum gap; it is an alternate assumed stop setting, not a measured tolerance distribution. Material properties and internal guidance were not validated through this sensitivity test.

The trade-off is explicit: fixed-reference global mean isotropy and its lower spatial decile decrease, while weak-region and corridor averages improve. The changed workspace is not a superset of the original. The selection utility gives equal weight to the three regional success scores plus a smaller normalised weak/corridor solution-isotropy reward; these are exploratory engineering preferences, not clinically established weights. paired_sampling_uncertainty.json contains 2,000 paired bootstrap resamples; these intervals reflect target-sampling variation only, not physical or model uncertainty.

## Reproduce

From the project folder, run `.venv/bin/python tools/optimise_measured_ctr.py screen`, then `select`, then `refine`, then `fine`, then `validate`, then `verify_selected`; finally `.venv/bin/python tools/report_measured_optimisation.py`. Seeds, selection criteria and exploratory bounds are in protocol.json. The pre-alignment pilot is archived separately and is not final evidence. Selected parameters were frozen before independent validation. Source hashes identify the numerical code used. `demo_optimised.py` exercises the native viewer separately.

## Viewer

Launch `measured_hardware_simulator/Launch Optimised CTR.command`. F7 switches original/optimised, F6 switches the maximum-gap case, C switches independent/coordinated joint control. Switching resets the state and selects separate caches. The original tube parameter file and previous research results are retained. The selected JSON is the live optimised configuration; the obsolete hard-coded profile is backed up in legacy_before_replacement.
