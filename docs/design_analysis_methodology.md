# CTR positional design-analysis methodology

## Status

The calculation definitions below were introduced in Stage 2. The repository
now also includes the Stage-3 baseline runner, Stage-4 convergence extension,
Stage-4.1 IK-support extension, and Stage-5A optimisation protocol/preflight.
The base methodology is in `config/analysis_protocol_v1.toml`; separately named
extension protocols freeze later sample counts and gates.

Local archived Stage-4 runs completed the primary inner-tip workspace/isotropy
convergence checks, and Stage 4.1 completed the IK spatial-support requirement.
Their generated datasets remain outside Git under `results/evaluation/`.
Stage 5A defines methodology only; the Stage-5B pilot is incomplete, and full
optimisation and final independent design validation have not been completed.

Earlier exploratory directories under `results/` must not be mixed with formal
protocol datasets. Run manifests and validation gates determine evidence status;
a figure or the presence of a runner alone does not certify a result.

## Agreed scope

The primary endpoint is the distal inner-tube tip. The middle- and outer-tube
endpoints will be reported separately as secondary characteristics, without an
arbitrary equal-weighted combined score.

The study includes:

- positional workspace and common-region coverage;
- positional Jacobian isotropy;
- numerical inverse-kinematics residual and convergence.

Tip orientation is not an optimisation objective. Force capability and loaded
stiffness capability are outside the available project scope. Young's modulus
remains in the forward shape model, but is fixed at 75 GPa for all three tubes
during optimisation and will later be treated only as uncertainty.

Because no anatomical target region has been supplied, the occupied baseline
inner-tip workspace is the provisional fixed task region for candidate design
comparisons. This is a methodological comparison region, not a clinically
validated surgical workspace.

The model interprets the reported pre-curvatures as 19.12 and 14.04 per metre.
That physically plausible interpretation remains explicitly provisional because
the supervisor-provided unit label has not been confirmed.

## Independent canonical configurations

The ideal model is continuously equivariant under a common rotation of all
tubes about Z. A common rotation therefore changes the displayed azimuth but
does not add an independent design configuration for axisymmetric positional
statistics.

Protocol-v1 samples five independent variables:

1. three nested deployment fractions;
2. inner-to-outer relative rotation;
3. middle-to-outer relative rotation.

The outer rotation is fixed to zero as a common-angle gauge. Three independent
absolute angles are not sampled because that would repeatedly sample the same
relative tube state at different display azimuths.

Every random configuration has a stable ID derived from the sampler version,
split, seed and canonical index. Deployment and rotation use separate fixed
PCG64 streams so an N-row dataset is an exact prefix of a larger dataset made
with the same split and seed. The same fractions, relative rotations and IDs are
reused when comparing tube designs; only the decoded physical deployments change
with tube length.

The eight deployment-cube corners are deterministic boundary diagnostics. They
are not inserted into the random rows and are excluded from independent sample
counts, spatial statistics, training and validation.

Training and validation use different seeds and namespaced IDs. Final candidate
selection must use training data only; validation remains untouched until the
shortlisted designs and reporting rules have been frozen.

The full six-input sampler in `ctr_workspace_map.py` remains available for the
interactive viewer. Its XYZ cache is a visualisation and restart aid, not a
protocol-v1 analysis dataset.

## Continuous rotational symmetry

For a sampled tip at azimuth `atan2(y, x)`, the formal analysis removes the full
azimuth and stores the state on the positive-X/Z canonical half-plane as
`(radius, 0, z)`. The matching common actuator-angle change preserves both
relative tube rotations.

Scalar statistics and minimum-sample checks are calculated from unique
canonical IDs before any angular sweep. A 12-, 90-, 360- or finer-angle sweep
may change graphical smoothness, but it must not change:

- the number of independent configurations or targets;
- spatial statistical support;
- medians, quantiles or failure rates;
- inclusion or masking of a cell.

Rendered copies retain their source canonical ID and display angle. They are
never relabelled as additional independent configurations or targets.

The older quadrant-folding helpers and four-quadrant studies are retained for
legacy diagnostic plots. Their rotated rows cannot be used to claim a larger
statistical sample size under protocol v1.

## Stage-2 Jacobian definition

The positional Jacobian is differentiated in physical actuator coordinates:

`[inner mm, middle mm, outer mm, inner rad, middle rad, outer rad]`.

Every design then uses the same baseline-derived input scale of
`[350 mm, 170 mm, 80 mm, pi rad, pi rad, pi rad]`. This produces a
baseline-range-normalised positional Jacobian for fair cross-design comparison.
It must not be described as an actuator-velocity isotropy because motor speed
limits have not been supplied.

All three endpoint Jacobians share the same forward-model evaluations. A
three-point central difference is used only where a two-sided neighbourhood
exists inside the nested deployment constraints. Exact or numerically
indistinguishable constraint-boundary states are masked rather than assigned a
misleading one-sided six-input isotropy.

The Stage-2 convergence check used 64 prefix-stable configurations, all three
endpoints, the baseline and two admissible perturbed designs. The selected
steps are 0.1 mm and 0.001 rad. Against the next refinements (0.03 mm and
0.0003 rad), the largest observed relative Jacobian change was below
`3.1e-7`, and the largest absolute isotropy change was below `1.9e-7`. Both are
well within the predeclared 0.5% and 0.002 numerical QA limits. These results
show numerical step stability, not physical-model accuracy.

## Stage-2 spatial definitions

Independent configurations are aggregated on a globally aligned radial–Z grid
before any angular display sweep. A cell spanning radii `r0` to `r1` and height
`dz` represents the exact swept volume

`pi * (r1^2 - r0^2) * dz`.

Workspace size is the sum of empirically occupied cells. Internal empty cells
are not filled, smoothed or added to the workspace. Empty cells disconnected
from the outer-radius and upper/lower Z boundaries are reported separately as
enclosed unsampled cells; they are not claimed to be proven inaccessible until
they persist under sample-count and cell-size convergence.

Local isotropy statistics use unique canonical IDs. Maximum is secondary;
mean, median, quartiles and IQR require at least 30 valid independent values,
and the configuration-level lower tenth percentile requires at least 50.
Whole-workspace typical isotropy is calculated from cell medians and exact
swept-volume weights. Unreachable and under-supported cells are reported by
coverage/support fractions rather than silently assigned zero.

The provisional fixed baseline task region is discovered from one canonical
configuration bank. IK target candidates come from a separate bank. Targets
are actual known-reachable forward-model points selected without replacement
approximately equally across occupied cells. Their source actuator states prove
reachability but are never used to initialise IK. Each canonical target is
solved once; a continuous 360-degree sweep remains display-only. Each target is
weighted by its cell's swept physical volume divided by the number of selected
targets in that cell, so aggregate residuals and success fractions do not
overweight the small-volume cells near the axis.

## Remaining provisional settings

- Spatial cell size begins at 10 mm and will be compared with 5 and 15 mm.
- At least 30 independent samples are required for a reported cell median, 50
  for a lower-tail statistic, and 100 for a failure fraction or 95th percentile.
- The application-level evaluation threshold remains 0.5 mm. It is a task
  classification threshold, not the numerical solver stopping condition.
- A 96-target Stage-2 pilot compared solver tolerances of 0.1, 0.01 and
  0.001 mm. A provisional 0.01 mm stopping tolerance and 40-iteration limit
  were selected. Tightening to 0.001 mm did not alter the 0.5 mm success rate;
  increasing the iteration limit from 40 to 60 did not alter that rate or the
  residual p95. This selection must be rechecked using the full frozen baseline
  target set before optimisation.
- The fixed Jacobian rotation scale will also be reported with 0.5x and 2x
  sensitivity checks because no unique mixed prismatic/revolute scaling can be
  claimed without actuator speed or task-specific weighting data.

Until these items are resolved, optimisation remains disabled in the protocol.

## Reproducibility and manifests

A formal run manifest contains:

- the resolved protocol and semantic SHA-256 hash;
- the numerical tube-parameter snapshot and hash;
- Git commit, branch, dirty status and analysis-source file hashes;
- Python, NumPy, SciPy and Matplotlib versions;
- command, effective settings and explicit protocol deviations;
- run status.

Source hashes are required because the current analysis tree contains untracked
and modified files; a Git commit alone cannot reproduce it. Semantic hashes are
calculated from canonical JSON, so TOML comments or key order do not alter the
scientific configuration hash.

The current `tools/run_design_study.py` can still exercise the exploratory
workflow. It writes the versioned manifest before calculation and labels its
legacy spatial, target and objective definitions as protocol deviations. Its output is not a
protocol-v1 baseline.

Future formal results will use a separate versioned directory rooted at
`results/evaluation/ctr-eval-v1/`. Old result folders will not be overwritten or
retroactively given protocol-v1 manifests.

## Stage-2 validation

Run the focused protocol and sampling checks with:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ctr-mpl-cache \
  ./.venv/bin/python tests/test_analysis_protocol.py
```

These checks verify the baseline hash, protocol consistency, prefix-stable random
streams, disjoint splits, separate boundary cases, fixed common-angle gauge,
continuous arbitrary-angle equivariance for all endpoints, canonicalisation,
and unique-ID counts that are invariant to rendering resolution.

Run the Stage-2 calculation checks with:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ctr-mpl-cache \
  ./.venv/bin/python tests/test_stage2_analysis.py
```

They verify fixed physical Jacobian scaling, finite-difference convergence,
rotational equivariance, boundary masking, exact annular volumes, internal
voids, coverage, physical-volume weighting, independent stratified targets,
solver termination records and the separate 0.5 mm evaluation threshold.

The next stage is the full baseline convergence and figure run. Optimisation
must remain disabled until the 5/10/15 mm cell comparison, increasing-N
workspace stability, target support, solver pilot revalidation and independent
validation checks pass.
