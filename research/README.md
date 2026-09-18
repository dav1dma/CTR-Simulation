# Sampling, optimisation and reproduction

The sampling routines draw feasible carriage positions and relative tube rotations, apply the section-aware forward model, and summarise occupied radial-height cells and positional isotropy. The search pipeline varies total tube lengths, curved lengths and precurvatures while retaining measured travel and clearance limits; it prioritises reaching gains in weak regions and the forward corridor, then checks frozen candidates on independent targets and paths. The report's proposed candidate is t027. Its dense point-to-point gains coexist with poorer held-out path completion, so the code and evidence retain its failed acceptance checks as well as its improvements.

## Choose what you want to do

| Task | Where to start |
| --- | --- |
| View the results without running code | [Final-report gallery](../docs/dissertation/README.md) |
| Read the small saved statistics | [Summary](../docs/dissertation/data/summary.json), [path outcome](../docs/dissertation/data/path_outcome.json), [candidate](../docs/dissertation/data/frozen_candidate.json) |
| Generate a small workspace example | `python research/sample_workspace.py --samples 4096` |
| Check the packaged summary/figure provenance | `python research/check_saved_results.py` |
| Download the original arrays and source snapshots | [Research data release](https://github.com/dav1dma/CTR-Simulation/releases/tag/final-project-2026) |
| Reproduce the geometry search and dense study | Commands below and [detailed methodology](../docs/tube_configuration_methodology.md) |

Activate the environment from the [main README](../README.md#install-and-launch) before running commands. Run them from the repository root. The small example saves a scatter plot, CSV positions and JSON metadata in `results/sample_workspace/`; it is an illustration, not a replacement for the dense report result.

## Research data downloads

The Git checkout contains code, figures, small saved result records, and the 4.2 MB original reference-region map needed by research imports. Larger arrays are supplied as separate release assets so a normal clone stays manageable. Download the required `.tar.gz` files and extract them **from the repository root**; each archive restores its recorded `output/...` paths. The [download manifest](downloads.json) records sizes, member counts and SHA-256 checksums; `SHA256SUMS.txt` accompanies the downloads.

| Release asset | Contents and purpose |
| --- | --- |
| `chapter5-dense-data.tar.gz` | Original/proposed spatial states, identical targets and IK residuals, support top-up evidence and frozen source for the final dense comparison |
| `tradeoff-search-data.tar.gz` | The 115-design search, training and independent validation banks, candidate t027, reserve-candidate failure and path diagnosis |
| `measured-search-data.tar.gz` | Earlier measured search, reference regions and candidates that supplied later search inputs |
| `strict-search-data.tar.gz` | Strict non-regression attempt and its frozen diagnostic finalists |
| `supplementary-method-comparison.tar.gz` | Later NSGA-II/staged study; separate from the report's t027 comparison; neither finalist accepted |

For the complete numerical archive, download all five. For the dense raw-data check, only `chapter5-dense-data.tar.gz` is required:

```bash
tar -xzf chapter5-dense-data.tar.gz
python research/check_saved_results.py --raw
```

Each archive includes a `research/manifests/` member list with file-level hashes. Source snapshots are frozen evidence, not separate applications to double-click. The supplementary archive excludes installed third-party dependency binaries; its environment and dependency version records are retained.

## Main programs

| Program | What it does / how to use it |
| --- | --- |
| [sample_workspace.py](sample_workspace.py) | Small original/proposed feasible-sampling example; run with `--samples 4096` or see `--help` |
| [optimise_measured_ctr.py](../tools/optimise_measured_ctr.py) | Shared measured sampling, forward calculation, isotropy and batch IK kernels; also runs the initial measured search |
| [strict_measured_search.py](../tools/strict_measured_search.py) | Tests a strict requirement that all selected metrics avoid regression |
| [task_prioritised_tradeoff_search.py](../tools/task_prioritised_tradeoff_search.py) | Runs the staged search that selected t027; commands `search`, then `validate` |
| [validate_tradeoff_reserve.py](../tools/validate_tradeoff_reserve.py) | Independent subsequent check of the only remaining training-pass candidate, t019 |
| [chapter5_measured_study.py](../tools/chapter5_measured_study.py) | Dense characterisation of the frozen original/t027 pair |
| [chapter5_topup_support.py](../tools/chapter5_topup_support.py) | Adds targets according to cell support counts, preserving the initial bank and fixed design |
| [chapter5_measured_figures.py](../tools/chapter5_measured_figures.py) | Regenerates data maps; published gallery images are extracted from the final report to preserve its exact appearance |
| [investigate_tradeoff_paths.py](../tools/investigate_tradeoff_paths.py) | Diagnoses two failed proposed paths without replacing their validation scores |
| [report_tradeoff_paths.py](../tools/report_tradeoff_paths.py) | Additional alternative-start checks and path-diagnosis figures |

`ctr_sampling.py` is the earlier general canonical-sampling library. The final measured study uses `samples()` in `optimise_measured_ctr.py`. The shared vector kernel imported from older hardware scripts is reused with the measured limits; their old 35 mm grip offset is not applied to the final study.

## Reproduce the report studies

Study programs use fixed output paths and can resume or replace records. Work in a **separate copy** and keep the downloaded evidence unchanged. For a clean t027 replay, keep the prerequisite `output/measured_hardware_optimisation_20260914/` records and `output/strict_measured_optimisation_20260914/frozen_finalists.json`, but move the existing `output/task_prioritised_tradeoff_20260914/` outside that replay copy before starting. Then run:

```bash
python tools/task_prioritised_tradeoff_search.py search
python tools/task_prioritised_tradeoff_search.py validate
python tools/report_task_prioritised_tradeoff.py
python tools/validate_tradeoff_reserve.py
python tools/report_task_prioritised_tradeoff.py reserve_validation
```

For the dense comparison, retain the t027 frozen record and prerequisites, but start without `output/chapter5_measured_revision/` in the replay copy:

```bash
python tools/chapter5_measured_study.py
python tools/chapter5_topup_support.py
python tools/chapter5_measured_figures.py
```

These are full research workloads and may take a long time. The small workspace example is the quick starting point. See the [detailed methodology](../docs/tube_configuration_methodology.md#7-reproduction-without-overwriting-the-archive) for regenerating the earlier search inputs, all random seeds, numerical settings and the separate supplementary comparison. That supplementary study additionally uses `pymoo==0.6.2`.

Numerical reproduction depends on software versions as well as seeds. Historical records identify Python 3.14.3, NumPy 2.5.2 and SciPy 1.18.0. [requirements-tested.txt](../requirements-tested.txt) captures the main dependencies used for this release's checks; [requirements.txt](../requirements.txt) is the broader compatibility list. A full historical transitive environment was not recorded.

[Checks completed for this release](../docs/RELEASE_CHECKS.md).

## Verification and interpretation

```bash
python -m unittest discover -s proposed_tradeoff_simulator -p test_measured.py -v
python -m unittest discover -s measured_hardware_simulator -p test_measured.py -v
python research/check_saved_results.py
python research/verification/verification_check.py
```

Run the two simulator suites in separate processes because their snapshots have the same module names. [Appendix D evidence](verification/additional_verification.json) contains the analytical single-arc, straight-element and direct carriage-gap checks. [evaluated_candidates.csv](verification/evaluated_candidates.csv) lists the 115 designs actually evaluated.

Workspace is sampled annular occupancy, not exact reachable volume. Isotropy is a range-normalised positional Jacobian measure, not orientation or stiffness. IK success means the specific fixed-start solver achieved ≤0.5 mm model residual, not measured physical accuracy. Path completion requires all 81 checked positions to meet that threshold; reported means of bank-level P95 values are not pooled percentiles. Post-hoc recovery does not replace the 43/48 proposed score.
