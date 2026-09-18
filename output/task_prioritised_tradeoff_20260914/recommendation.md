# Decision: a point-to-point trade-off, not an all-purpose replacement

For the dissertation's numerical Cartesian tip-control study, prioritise fixed-start target-reaching success and dexterity in declared difficult regions. These address whether the controller can reach targets and avoid weak local directional performance. Global occupancy volume and maximum reach are secondary: a larger envelope alone says little about usable control. For continuous navigation, path completion and tracking residual must instead remain primary. No anatomical task or experimental performance has been established.

The search evaluated 115 valid designs for IK, 26 designs including references at denser spatial resolution, and eight candidate designs on training paths. Two passed the training trade-off gates. Both were subsequently assessed on fresh independent banks. The second assessment was an explicitly documented sequential extension; neither candidate passed the predeclared requirement to preserve path completion and the tracking-error allowance. The acceptance thresholds were not changed after seeing validation results.

## Candidate worth retaining for a point-to-point study

Inner/middle/outer chuck-to-tip lengths: **350 / 177.5 / 87.5 mm**.
Straight lengths: **350 / 87.5 / 22.5 mm**.
Distal curved lengths: **0 / 90 / 65 mm**.
Precurvatures: **0 / 21.37 / 14.04 m⁻¹**.

Independent comparisons against the original tubes, with the same measured hardware, solver settings, target banks and spatial resolution:

| Metric | Original | Candidate | Change |
|---|---:|---:|---:|
| Global fixed-start IK success | 62.06% | 68.96% | +6.90 percentage points |
| Weak-region success | 53.66% | 58.03% | +4.37 points |
| Forward-corridor success | 68.48% | 74.43% | +5.96 points |
| Weak-region lower-decile isotropy | 0.12796 | 0.14282 | +11.61% |
| Global mean isotropy | 0.36265 | 0.38232 | +5.42% |
| Global lower-decile isotropy | 0.19870 | 0.21252 | +6.96% |
| Sampled annular occupancy volume | 2338.92 cm³ | 2513.67 cm³ | +7.47% |
| Completed paths | 44/48 | 43/48 | One fewer; −2.08 points |
| Mean of the two banks' P95 tracking residuals | 0.0866 mm | 0.1531 mm | +0.0664 mm; +76.7% |

These are averages of two independent banks except the summed path counts. The path P95 figure is an average of two percentiles, not a pooled percentile. The fixed-start success gains have positive paired target-bootstrap intervals in all three regions; this does not establish real-world accuracy. The path count is too small to establish a general probability of failure.

The candidate retains 100% of the frozen *well-supported* original reference cells in these samples. This is not proof of retaining every original reachable position. Maximum sampled tip distance is effectively unchanged. Thus the observed trade-off is in path behaviour, not a smaller workspace. It improves point-to-point results less than the previously installed candidate, while recovering its lost global isotropy; all three are compared in the saved data.

**Recommendation:** retain this as a separate, defensible candidate for a dissertation claim of improved ideal-model point-to-point numerical control, explicitly accepting the observed path-performance trade-off. Do not present it as passing the broader optimisation acceptance rule or as improved continuous navigation. A path-focused study should keep searching or separately investigate the controller. The active configuration has not been replaced.

## Hardware and interpretation

Coupled carriage limits remain enforced. Maximum exterior exposures become 175 / 82.5 / 72.5 mm, with minimum exposures 75 / 0 / 0 mm. Middle/front maximum gap within the exterior nesting model remains **18.5 mm**: increasing both tube lengths equally does not remove that relative-motion restriction. Improvement therefore does not require using the full 108 mm gap.

The middle curve is longer than its maximum exterior exposure. This exterior-only model cannot uniquely identify the excess curved material or validate its passage through internal guides. The provisional chuck-to-tip interpretation, retention, guidance, unrestricted rotation and unvalidated material/manufacturing assumptions remain. The 108.5 mm sensitivity case produces the same spatial and IK metrics because this candidate's exterior nesting domain already makes that upper stop inactive.

The reserve candidate (same lengths, curvatures 0 / 20.62 / 13.665 m⁻¹) is fully retained in `reserve_validation`; it also failed path protections. No numerical results are experimental verification.

See `tradeoff_report.md`, `protocol.json`, `validation_records.json`, `paired_success_intervals.json`, `verification.json`, and `proposed_configuration.json`. Raw sampled workspaces, target banks, solver outcomes and trajectories are retained in NPZ files. The original simulator, currently installed candidate and earlier study results remain unchanged.
