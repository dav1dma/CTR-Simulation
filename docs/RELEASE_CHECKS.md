# September 2026 release checks

The prepared source checkout was checked on macOS with Python 3.14.3 and the main dependency versions in `requirements-tested.txt`. Full optimisation searches were not rerun for publication.

- Both measured-hardware simulator suites: **11 tests each passed**, covering original/alternative geometry, coupled stops, coordinated movement, IK, motion routes, cache identity and CSV validation.
- **14 standalone root regression scripts passed**, including sampling, model/Jacobian consistency, IK, motion planning, historical protocols and live profile switching. The historical live-profile check was updated to read the current earlier-candidate geometry instead of a superseded hard-coded curvature and length expectation; it now runs only when explicitly invoked. Model and solver sources were not changed by this packaging work.
- Both native viewers opened and rendered successfully. The original viewer completed its movement/export/import check with a 0.002752 mm numerical residual; the proposed viewer completed its corresponding check with a 0.000476 mm residual. These are software checks, not physical accuracy measurements. Rendered screenshots were inspected.
- All **12 final-report gallery images** and the eight saved summary records matched their recorded SHA-256 checksums.
- Recomputing occupied-cell volume and volume-weighted fixed-start success from the packaged dense archive reproduced **2396.25 / 2582.78 cm³** and **62.280 / 66.993%**, respectively, with 262,144 states per design and 59,341 common targets.
- Appendix D analytical and direct-constraint fixtures passed: two straight references, 15 single-arc cases, and 27 decoded states for each original/proposed design.
- The small workspace sampling example produced valid original/proposed states, CSV outputs and a plot.
- Research archive checksums and member paths were checked. The complete report PDF, presentation, local environments and installed dependency binaries are excluded.

For repeatable commands, see the [research guide](../research/README.md). Windows/Linux desktop rendering and a physical PS5 controller were not tested during this release. Archived datasets, seeds and recorded environments support inspection and replay, but do not guarantee bitwise-identical results on another system.
