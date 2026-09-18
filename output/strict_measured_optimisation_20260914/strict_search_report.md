# Strict measured-hardware non-regression search

**Outcome: No non-equivalent meaningful candidate passed every gate in this bounded search.** The active optimised configuration was not changed by this study. This is a bounded numerical result, not proof that a dominating design cannot exist.

## Search and gates

1,354 unique rounded, feasible-sampled designs were evaluated in the spatial screen. 5.0 are baseline/exterior-equivalent controls and cannot count as improvements. 4 exceptionally thin operating domains exhausted the uniform rejection-sampling budget and are listed separately; they were not declared mechanically infeasible. Broad, local and curvature-grid candidates include previous designs and new samples, with all variable bounds unchanged from the previous study. The inner element remains straight and diameters/stiffness are fixed.

The denser selection stage evaluated 25 non-control designs and the original baseline; 0 passed every selection metric with a meaningful improvement. 3 finalists were frozen for independent validation. If none passed selection, the closest failures were still validated diagnostically; validation does not erase an earlier selection failure.

Each independent validation uses 131,072 physical configurations per design and 3,072 identical baseline-derived target draws (2,048 global, 512 weak, 512 corridor). There are two independent configuration/target seeds. The original baseline is rerun with each seed, not compared against an old sample with a different size. The regions are the previously frozen original-tube benchmark regions, not anatomical regions.

All 25 metrics must be non-regressing. No overall score can offset a failed metric. The only tolerances are 1e-10 for fractions, 1e-8 for isotropy and 1e-6 for distances/volumes, solely for floating-point equality. These are stricter than statistical non-inferiority margins; no sampling uncertainty allowance was introduced after observing results. A passing design must also improve at least one success fraction by 0.5 percentage points, an isotropy metric by 1%, or volume by 1%. Exterior-equivalent changes are excluded.

## Independent failures

| Candidate ID | Failed metrics: replicate1 | Failed metrics: replicate2 |
|---|---:|---:|
| s0999 | 4 | 5 |
| s0998 | 5 | 6 |
| s1032 | 3 | 6 |
| current_tradeoff | 7 | 7 |

The original compared with itself passes as a control. A candidate failing even one metric is rejected under the requested rule. Gap 108.5 mm sensitivity is an additional gate only for otherwise fully passing candidates; an empty gap_sensitivity.json means none qualified for that gate.

## Closest selection-stage candidate: s0999

This candidate is shown to explain the result, not presented as a replacement.

| Tube | Chuck-to-tip length (mm) | Curved length (mm) | Precurvature (1/m) |
|---|---:|---:|---:|
| Inner | 350.0000 | 0.0000 | 0.0000 |
| Middle | 170.0000 | 90.0000 | 19.7450 |
| Outer | 80.0000 | 65.0000 | 14.2275 |

First independent replicate:

| Metric | Original | Candidate | Non-regression |
|---|---:|---:|---|
| global isotropy mean | 0.360337  | 0.370102  | Pass |
| global isotropy p10 | 0.196771  | 0.204663  | Pass |
| global cell retention | 100 % | 100 % | Pass |
| weak isotropy mean | 0.19038  | 0.196683  | Pass |
| weak isotropy p10 | 0.127434  | 0.131318  | Pass |
| weak cell retention | 100 % | 100 % | Pass |
| corridor isotropy mean | 0.177901  | 0.181365  | Pass |
| corridor isotropy p10 | 0.122583  | 0.125182  | Pass |
| corridor cell retention | 100 % | 100 % | Pass |
| volume cm3 | 2371.9 cm³ | 2419.42 cm³ | Pass |
| radial extent mm | 112.301 mm | 114.004 mm | Pass |
| z max mm | 169.384 mm | 169.483 mm | Pass |
| z min mm | 63.8757 mm | 63.442 mm | Pass |
| tip distance max mm | 172.198 mm | 172.2 mm | Pass |
| global ik success | 62.4815 % | 61.6423 % | FAIL |
| global ik p95 mm | 90.4515 mm | 91.6605 mm | FAIL |
| global diagnostic success | 99.062 % | 99.9314 % | Pass |
| weak ik success | 57.3093 % | 57.4408 % | Pass |
| weak ik p95 mm | 54.4739 mm | 54.4739 mm | Pass |
| weak diagnostic success | 99.4927 % | 100 % | Pass |
| corridor ik success | 67.5299 % | 67.8287 % | Pass |
| corridor ik p95 mm | 14.3777 mm | 14.8164 mm | FAIL |
| corridor diagnostic success | 100 % | 100 % | Pass |
| trajectory completion | 91.6667 % | 91.6667 % | Pass |
| trajectory p95 mm | 0.191246 mm | 0.192176 mm | FAIL |

## What the measurements mean

Mean and lower-decile isotropy are computed on fixed global, weak-region and corridor cells, with fixed [350,170,80,pi,pi,pi] Jacobian scaling. Missing or insufficiently sampled cells receive zero; therefore support/coverage losses can reduce isotropy scores. This is not solely dexterity at shared points. Each cell is 5 mm in radius and height, with at least five observations for an isotropy median. Cell retention is weighted by swept annular volume. Occupancy volume and coordinate extrema are sample-based estimates; they do not certify all interior points as reachable.

Fixed-start IK uses the same 40-iteration, 0.5 mm-success test for every design, starting at its most-retracted feasible exterior pose with zero rotations. Regional P95 errors include failed attempts. The two-nearest-start diagnostic separately tests baseline target coverage with 60 iterations per start. Numerical failure does not prove geometric unreachability. The numerical model, starts and targets differ from older dissertation results using obsolete hardware limits.

Trajectory testing uses 24 smooth baseline-generated paths, 21 solved waypoints and five joint-interpolation positions per segment. The reference is the continuous baseline path; candidate exposure/rotation interpolation is checked against it. Completion requires every evaluated tip error to stay within 0.5 mm, and every intermediate state must obey the coupled hardware constraints. P95 includes all evaluated path errors. It does not evaluate anatomy collisions, loads, actuator timing or experimental control accuracy.

Track utilisation is a diagnostic, not a substitute for these gates. All designs retain the measured 100 mm nominal strokes, 71.5 mm actuator bodies and 8.5–108 mm clearances. Installed material lengths, retention, internal passage and straightening, rotation limits and manufacturable curvatures remain provisional. The model omits torsion, friction, elastic stability and physical validation.

## Reproduce and inspect

From the project folder run `.venv/bin/python tools/strict_measured_search.py checks`, then `search`, `assess`, `validate`, and `report`. The search can resume its own checkpoint. Start with a fresh output directory to rerun from scratch. Saved NPZ files contain source target states, resulting IK states/errors, workspace points/occupancy/isotropy and trajectory states/errors. Source snapshots and hashes are retained.

- `protocol.json`
- `screen_records.json`
- `selection_records.json`
- `frozen_finalists.json`
- `validation_records.json`
- `outcome.json`

Do not replace the existing candidate on the basis of being close to passing. A failure-free empirical comparison would still be conditional on the listed model, metrics and samples, rather than a universal improvement guarantee.
