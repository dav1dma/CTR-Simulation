# Legacy implementation

These files are retained for historical comparison and are not used by the
current VisPy application:

- `interactive_ctr_viewer.py` — superseded PyVista viewer
- `CTR_superPosKin_fun_compatible.py` — inherited pre-sectioned model
- `static_configuration_test.py` — checks for the inherited model
- `compare_workspaces.py` — comparison helper for historical datasets

The core requirements intentionally omit PyVista and pandas, which are only
needed by legacy files. Historical generated datasets remain available through
the Git history before the repository-cleanup commit.
