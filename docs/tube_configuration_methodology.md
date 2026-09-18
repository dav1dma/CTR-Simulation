# Tube configuration development and reproduction

Reviewed against the saved code and results on 16 September 2026. This document explains the numerical design studies; it does not certify a physical robot. No search was rerun for this documentation review.

## 1. Which configuration is which?

All lengths below are provisional material lengths from the brass chuck tip to the distal tip. Tube order is inner/rear, middle/middle, outer/front. The inner element is a straight wire.

| Configuration | Total lengths (mm) | Curved lengths (mm) | Precurvatures (m⁻¹) | Status |
| --- | --- | --- | --- | --- |
| Original | 350 / 170 / 80 | 0 / 90 / 65 | 0 / 19.12 / 14.04 | Reference design |
| Earlier measured-search candidate | 348.16 / 181.98 / 92.14 | 0 / 95.02 / 65.91 | 0 / 22.03 / 13.98 | Separate earlier candidate; not the Chapter 5 proposed design |
| **Dissertation proposed, t027** | **350 / 177.5 / 87.5** | **0 / 90 / 65** | **0 / 21.37 / 14.04** | Selected point-to-point trade-off; failed the combined held-out acceptance rule |
| Final NSGA-II finalist | 350 / 176.650734 / 80.460520 | 0 / 85.042355 / 70.710571 | 0 / 19.12 / 14.04 | Not accepted; not an installed replacement |
| Final staged-search finalist | 350.016962 / 180.847709 / 89.652801 | 0 / 82.811943 / 72.083106 | 0 / 21.225344 / 14.263641 | Diagnostic finalist; not accepted or installed |

Use the JSON files for full numerical precision. Printed digits allow reproduction; they do not state manufacturing tolerances. Straight length equals total minus curved length.

The dissertation design is stored in [the proposed viewer configuration](../proposed_tradeoff_simulator/optimised_configuration.json) and [the t027 frozen record](../output/task_prioritised_tradeoff_20260914/frozen_candidate.json). The older measured viewer has a different [configuration file](../measured_hardware_simulator/optimised_configuration.json). The word `optimised` in these filenames is a historical software label, not an acceptance decision. Final-study exports are [NSGA-II](../output/final_geometry_method_comparison_20260916/candidate_nsga2.json) and [staged](../output/final_geometry_method_comparison_20260916/candidate_staged.json).

## 2. Traceable origin of the dissertation design

The implemented lineage is:

1. **Measured-hardware exploratory search.** [optimise_measured_ctr.py](../tools/optimise_measured_ctr.py) combined structured length changes and 256 Sobol proposals, then screened feasible designs using sampled positional isotropy and reference coverage. Saved files contain 190 screen records and 13 shortlisted designs including the original. Selection used a shared IK target bank. Local refinement proposed 128 Sobol perturbations and 28 single-variable perturbations; 138 cumulative records survive in `refinement_records.json`. A subsequent 9 × 9 curvature grid and 64 local Sobol proposals produced 283 cumulative `fine_records.json` records. These are records, not necessarily distinct designs. This stage selected the **earlier measured-search candidate**, not t027.
2. **Strict non-regression attempt.** [strict_measured_search.py](../tools/strict_measured_search.py) reused earlier proposals, added broad and local samples and a curvature grid, then tested whether every gated metric could be maintained. It did not find a meaningful, non-equivalent candidate passing every gate. Its frozen diagnostic finalists became additional development inputs to the later search. This does not prove that such a design does not exist.
3. **Benchmark-prioritised trade-off search.** [task_prioritised_tradeoff_search.py](../tools/task_prioritised_tradeoff_search.py) used the original and earlier selected candidate; interpolations at 25%, 50%, 75% and 100% toward the twelve earlier fine-search records ranked highest by weak-region plus corridor IK success; 64 new local Sobol proposals; and the strict-search finalists. Designs were rounded to four decimals and deduplicated. The saved run evaluated **115 valid designs for IK**, **26 at denser spatial resolution including two references**, and **eight candidates plus the original on training paths**.
4. **Training selection of t027.** Candidate `t027` was the highest-scoring candidate passing all of that search's training gates. Only `t027` and `t019` passed those gates. The original completed 44/48 training paths; t027 completed 45/48. Selection preceded the two validation banks.
5. **Held-out failure and qualified retention.** The two original validation banks showed better point-to-point IK but one fewer completed path overall and higher tracking P95. `outcome.json` explicitly records `accepted: false`. The saved [recommendation](../output/task_prioritised_tradeoff_20260914/recommendation.md) retained t027 as a separate point-to-point trade-off, not a successful all-purpose replacement. A disclosed sequential validation of reserve candidate t019 also failed. Neither failure is discarded.

The exact t027 interpolation can be checked without rerunning optimisation. The parent is record **187, zero-based**, in [fine_records.json](../output/measured_hardware_optimisation_20260914/fine_records.json):

```text
Vector order: [inner total, middle total, outer total,
               middle curved, outer curved, middle curvature, outer curvature]
Original = [350, 170, 80, 90, 65, 19.12, 14.04]
Parent   = [350, 180, 90, 90, 65, 22.12, 14.04]
t027     = Original + 0.75 × (Parent − Original)
         = [350, 177.5, 87.5, 90, 65, 21.37, 14.04]
```

The parent ranks seventh in the twelve-record source list. This correspondence is supported by the saved arrays and candidate-generation code; the old record does not itself contain an explicit parent ID. The [documentation audit](tube_configuration_evidence_audit.json) records the matching evidence and hashes.

Relative to the original, t027 changes middle and outer total lengths by **+7.5 mm**, and middle precurvature by **+2.25 m⁻¹**. Its straight lengths become **350 / 87.5 / 22.5 mm**. Inner length, both curved lengths and outer precurvature remain unchanged. These are combined changes; there is no controlled ablation establishing how much of each performance change was caused by each parameter.

## 3. Common physical domain and model

Let `d_j` be chuck-tip distance behind the outside/front face of the front plate, `L_j` material length and `e_j = L_j − d_j` exposed centreline material length. Exposure is not the Cartesian tip-to-plate distance of a curved tube. No 35 mm grip offset is used in these studies.

| Actuator / element | Retracted distance | Extended distance |
| --- | ---: | ---: |
| Rear / inner | 275 mm | 175 mm |
| Middle / middle | 195 mm | 95 mm |
| Front / outer | 115 mm | 15 mm |

Every carriage has a nominal 100 mm stroke. For each adjacent rear/ahead pair, the code enforces `8.5 ≤ d_rear − d_ahead − 71.5 ≤ 108` mm simultaneously with the carriage bounds. Actuator length 71.5 mm includes the rear shaft. The alternative maximum gap is 108.5 mm; this sensitivity retains the measured 0.5 mm discrepancy explicitly. The 8.5 mm inter-actuator minimum is not a clearance requirement between the front chuck and front plate.

[MeasuredDesignLimits](../measured_hardware_simulator/design_constraints.py) also enforces the exterior model domain `e_inner ≥ e_middle ≥ e_outer ≥ 0`. Negative exposures are invalid; they are not clipped into a valid physical state. This exterior-only restriction excludes tips behind the plate or inside another tube even if some could be physically possible. Geometry proposals must have a nonempty coupled domain and nonnegative straight material lengths.

The original maximum exterior exposures are 175 / 75 / 65 mm; t027 gives 175 / 82.5 / 72.5 mm. For both designs, middle–outer material length difference is 90 mm, so exterior nesting limits their gap to 90 − 71.5 = **18.5 mm**, despite the mechanically available 108 mm. Equal length increases do not remove this relative-motion restriction.

The forward model uses section-aware superposition of intrinsic curvatures weighted by bending stiffness. It is not a torsion/contact/stability model. Original diameters, material parameters and the straight inner element are fixed across geometry comparisons. Attribution for the supplied model is separate from attribution for these search scripts and algorithms; do not describe NSGA-II or the search score as part of the supplied forward model.

The evaluator uses the same constrained damped least-squares IK: most-retracted feasible exterior start, zero rotations, 40 iterations, 0.01 mm stopping tolerance, and 0.5 mm success threshold. The normalised update is capped at 0.1, finite-difference step is 10⁻⁴ and the normal-equation regularisation is `+I`. The separate diagnostic uses a 16,384-state bank, seed 35107, two nearest starts and 60 iterations each. It is used to initialise paths consistently, not to replace fixed-start IK success. Trajectories use warm starts, 20 control steps and no automatic recovery. Geometry optimisation did not alter these rules.

Positional isotropy is the smallest/largest singular-value ratio of the existing range-scaled positional Jacobian. The fixed scaling is `[350,170,80,π,π,π]`, with translation in mm. Straight-inner rotation contributes no shape change. This measure describes local directional balance, not orientation control, force capability or achievable directions at a mechanical stop.

## 4. Variables, bounds and historical evaluation

| Design variable | Broad measured/final-study bounds | New local proposals in t027 search |
| --- | ---: | ---: |
| Inner total length (mm) | 320–380 | 345–355 |
| Middle total length (mm) | 170–230 | 170–182 |
| Outer total length (mm) | 80–140 | 80–92 |
| Middle curved length (mm) | 30–110 | 55–90 |
| Outer curved length (mm) | 20–90 | 45–70 |
| Middle precurvature (m⁻¹) | 12–27 | 18–25 |
| Outer precurvature (m⁻¹) | 9–21 | 11–17 |

The local column applies to the 64 newly sampled proposals, not to every inherited/interpolated candidate. These are exploratory engineering bounds, not manufacturer-certified ranges.

The inherited original reference uses 65,536 feasible states, 5 mm radial/axial cells and at least five states per supported cell. The weak region is the lowest 20% annular-volume-weighted original cell-median isotropy. The forward corridor is the supported subset with radius below 10 mm and axial coordinate 40–130 mm. These are mathematical benchmarks; no clinical task was established.

The historical target-bank generator samples actual original-model reachable positions. It chooses supported cells uniformly and assigns annular-volume weights `2 × radial_bin + 1`. It intersects the reference with cells found in each source pool; rare absent cells can therefore be omitted. The final comparison later tightened this rule. Sampling is not uniform over Cartesian workspace merely because carriage proposals are uniform.

The t027 search ranks the IK gains using:

```text
score = global success gain + 2 × weak-region gain + 2 × corridor gain
```

IK screening uses 512 global and 128 targets in each priority region. The top 24 additional designs plus both references receive 65,536-state spatial checks. Up to eight eligible candidates receive 48 training paths. Training gates allow volume/reach −5%, reference retention −3 percentage points, global mean isotropy −5%, global P10 −10%, global IK −2 points, regional IK −1 point, diagnostic success −1 point, and weak-region P10 −5%. Path completion must not fall; mean bank P95 may increase by at most 0.05 mm. A meaningful gain is either mean priority-region IK +3 points, or weak P10 +10% without priority IK loss. The exact conjunction is `guards()` in the search script. These weights and thresholds are project choices, not published clinical limits.

Validation uses two banks, each with 2,048 global plus 512 targets per priority region, 131,072 workspace states and 24 paths. The historical path routine checks four new interpolation positions per control segment plus the initial state: **81 checked positions per path**, not 2,001. Some protocol prose calls this “five intermediate positions”; the actual loop includes the already-known segment start among five fractions, then evaluates the remaining four. Later dense failure diagnosis uses 2,001 positions and remains a separate assessment.

## 5. Seeds and data lineage

| Stage | Seeds |
| --- | --- |
| Measured original regions / screening / designs | 33111 / 33112 / 33113 |
| Measured IK selection | 34101 |
| Measured local proposals | 34201 and 34279; fine proposals 34301 |
| Strict proposals | Broad 41102; local 41103, 41104; refinement 41201–41206 |
| Strict screen / selection workspace / selection targets | 41101 / 42101 / 42102 |
| t027 local proposals / training targets / spatial checks | 51104 / 51102 / 51101 |
| t027 training paths | 51103, 51105 |
| t027 validation workspace | 52101, 53101 |
| t027 validation targets | 52102, 53102 |
| t027 validation paths | 52103, 53103 |
| Reserve t019 workspace / targets / paths | 56101, 57101 / 56102, 57102 / 56103, 57103 |
| Chapter 5 primary / repeat spatial samples | 61101 / 62101 |
| Chapter 5 targets / original-only calibration | Source pools 61202–61205 / 61302–61305 |
| Chapter 5 support top-up | 61402–61407 used; maximum 16 batches allowed |

The historical target helper also seeds cell/point selection with target seed +1. The strict validation and remaining helper seeds are specified in their saved protocols and source. Sobol samples use SciPy's `scramble=True`; rejection, clipping, truncation and inherited designs mean the resulting design pool is not an untouched Sobol net.

Do not mix the following results:

| Assessment of original versus t027 | Global IK success | Paths | Meaning |
| --- | --- | --- | --- |
| Original trade-off validation | 62.06% → 68.96% | 44/48 → 43/48 | Two original held-out banks; mean bank tracking P95 0.0866 → 0.1531 mm |
| Dense Chapter 5 characterisation | 62.28% → 66.99% | Reuses separately reported trajectory evidence | Fixed design, denser targets and different aggregation/support; not another optimisation |
| Final method-comparison reference tests | 62.61% → 67.24% | 231/256 → 226/256 | New global target construction and declared path stress mixture |

Chapter 5 has 59,341 targets after a support-based top-up. The top-up added targets by cell counts, not by candidate errors; the saved support log reaches approximately 90.48% supported reference occupancy. Its separate calibration/region definition, target weighting and common-support isotropy must not be substituted for the historical search metrics. [chapter5_measured_study.py](../tools/chapter5_measured_study.py), [chapter5_topup_support.py](../tools/chapter5_topup_support.py) and `output/chapter5_measured_revision/data/` record this stage.

The two post-hoc alternative-start recovery examples did not change tube geometry and did not replace failed paths in completion counts. Their improvement is evidence of start/solver sensitivity, not evidence that the geometry passed the path gate.

## 6. Final method-comparison study

[final_geometry_comparison.py](../tools/final_geometry_comparison.py) compares two candidate-generation strategies under the same evaluator. It is **not** an exact repeat of the historical 1:2:2 scoring search.

Both methods minimise global fixed-start IK failure and trajectory failure, subject to screening safeguards. They share the original/proposed initial designs, bounds, samples, hardware, model, solver and starting rules. NSGA-II uses pymoo 0.6.2, population 32 and seven offspring batches of 32, SBX crossover probability 0.9 and distribution index 15, and polynomial mutation probability 1 per vector, 1/7 per variable, index 20. Feasible solutions are preferred; among infeasible solutions, the evaluator supplies summed positive safeguard deficits. Nondominated sorting and crowding guide the evolutionary search.

The staged method uses the same initial 32 designs, another 128 Sobol proposals, 32 curvature-grid proposals (4 × 4 around each reference), and 64 local proposals around four leaders. Local spans are `[3,4,4,8,8,1.5,1.5]` in design-vector order. Both methods stop at **256 evaluator slots for each of three seeds: 71011, 71012, 71013**. Analytically invalid designs are replaced using the same 8,192-point Sobol reserve seeded by run seed +900. Rejections are logged. Sampler exhaustion consumes a slot and is unresolved, not proof of physical infeasibility. Equivalent exterior geometries share a cache. There were 1,536 slots, 1,526 completed unique-within-run evaluations, six cache hits and four unresolved attempts; “unique” does not mean distinct across all six runs.

| Evaluation stage | Workspace states/design | Global / weak / difficult / corridor targets | Paths | Checked positions/path |
| --- | ---: | --- | ---: | ---: |
| Screen, seed 71100 | 8,192 | 1,024 / 256 / 256 / 128 | 24 | 81 |
| Dense selection, seed 72100 | 65,536 | 4,096 / 1,024 / 1,024 / 512 | 96 | 401 |
| Each final bank, seeds 73100 and 74100 | 262,144 | 8,192 / 2,048 / 2,048 / 1,024 | 128 | 2,001 |

Workspace evaluation uses bank seed +1. Original source pools use bank seed +10 onward; representative path endpoints use bank seed +30. Global target cells are drawn proportional to annular volume, then actual full-azimuth original FK points are selected. Every frozen original reference cell must be represented in the source pool or bank generation fails. Regional oversamples do not count twice in the global objective.

The weak region is inherited from original data. The difficult IK region is the worst 20% annular-volume mass of original calibration cells with at least 100 targets, with deterministic cell-ID tie breaking. The corridor remains a secondary readout. Final paths are half representative, one quarter weak-endpoint and one quarter boundary stress paths. This is an engineering test mixture, not a clinical trajectory distribution.

Four distinct candidates per run receive dense reassessment. One per method is frozen. A candidate failing selection is marked diagnostic and cannot be rescued by favourable held-out results. Selection prefers passing gates, then completion, then global IK and tracking measures. Only after the freeze are final banks generated.

Final acceptance requires global IK gain at least 3 percentage points with a positive paired interval lower bound; weak/difficult IK interval lower bounds at least −2 points; no observed completion loss with completion interval lower bound at least −2 points; and a path-block bootstrap upper bound on P95 increase at most 0.05 mm. Point safeguards protect supported coverage within 2 points, sampled reach within 2%, and global/weak/difficult mean and P10 isotropy within 5% in both repeats. Coverage is checked at 2.5, 5 and 10 mm and against baseline repeatability. The 108.5 mm gap sensitivity can disqualify, not rescue.

Paired success intervals use conservative simultaneous exact binomial bounds on discordant wins and losses. The implemented level is **97.5%**, allowing for two frozen method finalists; the saved protocol also contains shorthand “95%” wording, superseded by its classification field and the actual implementation. P95 uses 400 paired whole-path bootstrap resamples, seed 75100, with 1.25% and 98.75% quantiles. These are conditional numerical sampling intervals, not physical-model uncertainty. They can be conservative or imprecise, particularly with only 256 paths. A failed uncertainty gate need not establish deterioration.

No finalist passed every gate. The staged finalist had the best fresh-test point estimates for global IK (70.01%), completion (237/256) and pooled P95 (0.1301 mm), compared with original 62.61%, 231/256 and 0.1468 mm. However:

- It failed selection completion: 85/96 against original 88/96.
- Its fresh completion gain was +2.34 points, with conservative interval **−5.19 to +9.78 points**, failing the −2-point lower-bound requirement.
- Global isotropy P10 fell **7.67%** in the first repeat, beyond the permitted 5%; it rose in the second repeat. The average must not conceal the first failure.
- The 108.5 mm sensitivity also failed the global P10 safeguard.
- Worst checked deviation increased **66.115 → 76.274 mm**, although P95 improved. Worst error was reported, not silently treated as a newly added acceptance gate.

The NSGA-II finalist passed selection but failed final completion, P95 uncertainty, first-repeat global P10 and gap sensitivity. A modest evolutionary run therefore did not establish a superior replacement or general algorithm superiority. See [the final report](../output/final_geometry_method_comparison_20260916/study_report.md) and [decision record](../output/final_geometry_method_comparison_20260916/outcome.json).

## 7. Reproduction without overwriting the archive

The scripts use fixed output paths; some resume checkpoints and others overwrite summaries. **Run the commands below only in a disposable copy of the project.** Keep this reviewed archive unchanged. Existing saved banks are now inspected evidence, not fresh tests for a new adaptive study.

### Environment

Historical verification records Python 3.14.3, NumPy 2.5.2 and SciPy 1.18.0. The final study additionally records pymoo 0.6.2 on macOS arm64. `requirements.txt` is a compatibility list, not an exact historical lock. The final study's `isolated_requirements_lock.txt` describes the added dependency directory; its NumPy/SciPy entries are not the libraries actually used because the existing environment took precedence. Use `environment.json` for the numerical versions. A complete historical GUI/transitive-package lock is not recorded.

Example setup in the disposable copy, using Python 3.14.3 for the closest recorded match:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install numpy==2.5.2 scipy==1.18.0 pymoo==0.6.2
```

Platform and library differences may alter floating-point ties and search paths. Saved arrays are the authoritative historical evidence; matching a seed alone is not a bitwise-reproduction guarantee.

### Reproduce the t027 search

Provide these exact prerequisites at their relative paths in the disposable copy:

- `output/measured_hardware_optimisation_20260914/{protocol.json,baseline_regions.npz,selected_frozen.json,fine_records.json}`;
- `output/strict_measured_optimisation_20260914/frozen_finalists.json`;
- the source files and measured simulator modules listed in the study's `source_hashes.json`.

Do not copy `output/task_prioritised_tradeoff_20260914/` into this clean replay, so the new search cannot resume the historical checkpoints. Then:

```sh
.venv/bin/python tools/task_prioritised_tradeoff_search.py search
.venv/bin/python tools/task_prioritised_tradeoff_search.py validate
.venv/bin/python tools/report_task_prioritised_tradeoff.py
.venv/bin/python tools/validate_tradeoff_reserve.py
.venv/bin/python tools/report_task_prioritised_tradeoff.py reserve_validation
```

To regenerate the prerequisite searches instead of using archived inputs, start with their output directories absent in the disposable copy and run, in order:

```sh
.venv/bin/python tools/optimise_measured_ctr.py screen
.venv/bin/python tools/optimise_measured_ctr.py select
.venv/bin/python tools/optimise_measured_ctr.py refine
.venv/bin/python tools/optimise_measured_ctr.py fine
.venv/bin/python tools/strict_measured_search.py search
.venv/bin/python tools/strict_measured_search.py assess
```

This produces the candidate-generation prerequisites; it does not regenerate every historical diagnostic/validation figure. Historical measured validation is `optimise_measured_ctr.py validate`; strict validation is `strict_measured_search.py validate`. Their reports and scripts remain separately archived. The current source hashes matched the saved manifests at this documentation review, but no replay was executed to establish end-to-end identity.

### Reproduce Chapter 5 characterisation

With the historical t027 prerequisites present and `output/chapter5_measured_revision/` initially absent in the disposable copy:

```sh
.venv/bin/python tools/chapter5_measured_study.py
.venv/bin/python tools/chapter5_topup_support.py
```

The top-up deliberately replaces target/IK summaries in that disposable study. Use the archived figures and their generation scripts for the final presentation; this documentation task has not regenerated them. Post-hoc path diagnosis is in `tools/investigate_tradeoff_paths.py` and its archived `path_investigation/` results.

### Reproduce the final method comparison

In another disposable copy, keep the historical reference files `output/chapter5_measured_revision/data/{original_spatial.npz,regions.json,calibration_targets.npz,calibration_original.npz,summary.json}` and the t027 frozen record at its original path. Keep the model source and both simulator configuration files. Start with `output/final_geometry_method_comparison_20260916/` absent and install the recorded numerical dependencies above.

```sh
.venv/bin/python tools/final_geometry_comparison.py prepare
.venv/bin/python tools/final_geometry_comparison.py checks
.venv/bin/python tools/final_geometry_comparison.py pilot
for method in nsga2 staged; do
  for seed in 71011 71012 71013; do
    .venv/bin/python tools/final_geometry_comparison.py search "$method" "$seed"
  done
done
.venv/bin/python tools/final_geometry_comparison.py select
.venv/bin/python tools/final_geometry_comparison.py validate
.venv/bin/python tools/report_final_geometry_comparison.py
```

Reporting is a separate script; `final_geometry_comparison.py report` is not implemented despite its introductory docstring. The saved `preflight_amendment.json` records three corrections made before search: equal-budget reference occupancy calibration, disqualifying gap sensitivity, and common replacement of analytically invalid proposals. Preserve this file with the historical run. Sequential execution above avoids contention; original concurrent wall times are not isolated CPU benchmarks.

For plots from existing results alone, run `report_final_geometry_comparison.py` in a disposable archive copy with `validation/`, `sensitivity/`, `runs/`, final JSON records and protected files present. Do not rerun search merely to redraw plots.

### What must accompany a GitHub release

Documentation and source alone do not reproduce saved figures if their data inputs are absent. Include the parameter/protocol/decision JSONs, manifests, source snapshots, target banks, per-design workspace/IK/path NPZ files, run records and the report scripts. Keep the historical studies separate. Large numerical archives can be versioned release assets with checksums and a stable download link. That release has **not** been made by this documentation task. Do not upload `.venv/` or the final study's installed `dependencies/` tree as research data.

## 8. Evidence limits and unresolved assumptions

- Saved manifests match current relevant source hashes, and the t027 values match both the frozen record and proposed viewer JSON. The historical parent relationship is reconstructed from exact saved vectors and the generation rule; it was not explicitly logged as a lineage field.
- Saved records support the ranking decision. They do not establish a clinical justification for the corridor weights, exploratory bounds or numerical tolerances. The later conditional retention of t027 is documented in `recommendation.md`; it must not be rewritten as passing the original combined rule.
- Older search records do not retain every rejected proposal or a complete timestamped decision history. Counts of records should not be presented as exhaustive coverage of the continuous design space. The final study logs rejections/cache use more explicitly.
- This audit did not inspect a final submitted dissertation PDF. “Dissertation configuration” is verified against the proposed viewer and Chapter 5 study sources, not every sentence or table of the manuscript.
- Installed chuck-to-tip lengths, retention, actual guide passage and rotation stops are unresolved. The original middle curved material is 90 mm long but has at most 75 mm exposed; t027 exposes at most 82.5 mm. The exterior model cannot distinguish excess distal curved material that never emerges. It cannot justify treating those physical tubes as interchangeable inside guides.
- Unrestricted rotation, straight coaxial guidance and the absence of friction, torsion, elastic instability, loads and anatomy limit the claims. Coupled-stop feasibility is necessary but not sufficient for manufacture or physical operation.
- All performance changes are numerical. Physical positioning, repeatability, fabrication accuracy and clinical usefulness remain unverified. Failure of a finite search to meet all gates does not prove no better geometry exists; using a simple algorithm is only one possible limitation, not an established cause of failure.

## 9. Published methods versus project choices

Verified references are provided in [BibTeX](tube_configuration_references.bib). Verification used original papers, author/institution copies or publisher records; checked 16 September 2026.

| Reference | What it supports | What it does not establish here |
| --- | --- | --- |
| Sobol’ (1967), [original journal record](https://www.mathnet.ru/eng/zvmmf7334), DOI [10.1016/0041-5553(67)90144-9](https://doi.org/10.1016/0041-5553(67)90144-9) | Low-discrepancy sequence foundation | Optimality of this staged heuristic or uniform workspace coverage |
| Joe and Kuo (2008), [institutional record](https://researchcommons.waikato.ac.nz/items/6ded8a27-0cb2-4564-b221-35e814f5b93b), DOI [10.1137/070709359](https://doi.org/10.1137/070709359) | Sobol direction-number construction; [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html) identifies its implementation | Balance guarantees after this project's rejection/clipping |
| Deb et al. (2002), [original paper](https://dynamics.org/~altenber/UH_ICS/EC_REFS/MULTI_OBJ/DebPratapAgarwalMeyarivan.pdf), DOI [10.1109/4235.996017](https://doi.org/10.1109/4235.996017) | NSGA-II nondominated ranking, diversity and constrained optimisation | That NSGA-II produced t027, or must outperform the staged method |
| Deb and Agrawal (1995), [original SBX paper](https://content.wolfram.com/uploads/sites/13/2018/02/09-2-2.pdf) | Simulated binary crossover | This project's population, mutation settings or budget |
| Blank and Deb (2020), [author copy](https://www.julianblank.com/_static/research/ieee20-pymoo.pdf), DOI [10.1109/ACCESS.2020.2990567](https://doi.org/10.1109/ACCESS.2020.2990567) | pymoo optimisation software | Clinical validation or the acceptance margins |
| Bergeles et al. (2015), [original paper](https://rvim.online/publication/bergeles2015concentric/bergeles2015concentric.pdf), DOI [10.1109/TRO.2014.2378431](https://doi.org/10.1109/TRO.2014.2378431) | CTR design with task/anatomical constraints and elastic stability; Section IV-C uses Nelder–Mead after smoothing the cost and discusses pattern search in earlier implementations | That this project implements its anatomy/stability model or algorithm |
| Baykal, Torres and Alterovitz (2015), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4778735/), DOI [10.1109/IROS.2015.7353999](https://doi.org/10.1109/IROS.2015.7353999) | CTR design-set optimisation combining adaptive simulated annealing with RRT motion planning | That this project's prescribed-path IK tests are RRT or anatomy-aware planning |

The staged Sobol/grid/interpolation/refinement workflow is a **project-specific heuristic**. NSGA-II is a published general optimisation method adapted here. The two CTR papers provide relevant design precedents, not evidence for the particular regions, 0.5 mm threshold, 3-point gain, 2-point losses, 5% dexterity margin or 0.05 mm tracking allowance used here.

See the [research guide](../research/README.md) for the release downloads and main commands.
