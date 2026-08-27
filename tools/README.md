# Tools

- `ps5_button_identifier.py` reports controller button numbers.
- `ps5_trigger_identifier.py` reports the L2 and R2 axis values.
- `ps5_controller_test.py` provides a broader controller-input check.
- `workspace_generator.py` calculates a 10,000-configuration workspace and
  writes its CSV and plot to the ignored `results/` directory.
- `generate_workspace_map.py` regenerates the compact 12,000-point map used by
  the workspace analysis tools.
- `generate_endpoint_workspace_maps.py` generates the cached inner, middle, and
  outer endpoint samples used for smooth workspace limits and IK restarts.
- `generate_reachability_zones.py` generates the optional legacy blue/red/grey
  diagnostic map from 80,000 valid configurations.

Run tools from the repository root with the project environment, for example:

```bash
./.venv/bin/python tools/ps5_button_identifier.py
```
