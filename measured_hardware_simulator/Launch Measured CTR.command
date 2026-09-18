#!/bin/zsh
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ctr-measured-mpl"
exec .venv/bin/python measured_hardware_simulator/interactive_ctr_tip_control.py
