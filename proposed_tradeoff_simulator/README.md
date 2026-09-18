# Final-report original and proposed simulator

This folder contains the interactive viewer for the original 350/170/80 mm tubes and proposed candidate **t027**, 350/177.5/87.5 mm. Both use the same measured carriage constraints.

From the project root, use:

```bash
python launch_simulation.py --configuration original
python launch_simulation.py --configuration proposed
```

F7 switches between them. The inherited internal name `optimised` means **t027 only in this folder**. It does not mean the design passed every acceptance criterion. Full controls, assumptions, cache behaviour and CSV instructions are in the [simulation guide](../docs/SIMULATION.md).

The numerical code is retained from the final project. Automatic path-start selection and post-hoc recovery are not silently added to this controller. Tests are in `test_measured.py`; `verify_and_capture.py` performs a native proposed-viewer check.
