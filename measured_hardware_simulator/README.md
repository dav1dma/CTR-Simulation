# Measured-hardware model and earlier candidate

This folder provides the measured coupled carriage limits and model modules used by the numerical studies. Its original configuration is 350/170/80 mm. Its `optimised_configuration.json` contains the **earlier measured-search candidate**, 348.16/181.98/92.14 mm, rather than the final report's t027.

For the final-report original/proposed comparison, use the [root launcher](../docs/SIMULATION.md). This folder is retained unchanged as a numerical dependency and historical reference; replacing its candidate would break the design lineage.

To inspect its original configuration explicitly:

```bash
python measured_hardware_simulator/interactive_ctr_tip_control.py --tubes original
```

To inspect the earlier candidate, change `--tubes original` to `--tubes optimised`. See the [configuration lineage](../docs/tube_configuration_methodology.md) before comparing results. The full movement and interface controls are described in the [simulation guide](../docs/SIMULATION.md).

Key files: `design_constraints.py` derives geometry-dependent exposure limits; `ctr_operating_profile.py` selects the gap and geometry; `coordinated_control.py` handles coupled carriage movement; `test_measured.py` checks limits, IK, routes, cache identity and imports. `original_source_hashes.json` records the pre-adaptation source and is not a checksum manifest of the current folder.
