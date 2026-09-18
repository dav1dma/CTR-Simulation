#!/bin/zsh
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ctr-proposed-mpl"
exec .venv/bin/python proposed_tradeoff_simulator/interactive_ctr_tip_control.py --tubes optimised
