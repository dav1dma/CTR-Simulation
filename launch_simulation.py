"""Start the final-report original or proposed CTR in an isolated process."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', choices=('original', 'proposed'), default='original')
    parser.add_argument('--mode', choices=('hardware', 'hardware-108.5'), default='hardware')
    parser.add_argument('--joint-only', action='store_true', help='Open joint controls without Cartesian target planning')
    parser.add_argument('--waypoints', type=Path, help='Load the final pose from a compatible exported CSV')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    entry = 'interactive_ctr_vispy.py' if args.joint_only else 'interactive_ctr_tip_control.py'
    command = [sys.executable, str(root / 'proposed_tradeoff_simulator' / entry),
               '--tubes', 'optimised' if args.configuration == 'proposed' else 'original',
               '--mode', args.mode]
    if args.waypoints:
        command += ['--waypoints', str(args.waypoints.resolve())]
    return subprocess.call(command, cwd=root)

if __name__ == '__main__':
    raise SystemExit(main())
