# Tools

- `ps5_button_identifier.py` reports controller button numbers.
- `ps5_trigger_identifier.py` reports the L2 and R2 axis values.
- `ps5_controller_test.py` provides a broader controller-input check.
- `workspace_generator.py` calculates a 10,000-configuration workspace and
  writes its CSV and plot to the ignored `results/` directory.

Run tools from the repository root with the project environment, for example:

```bash
./.venv/bin/python tools/ps5_button_identifier.py
```
