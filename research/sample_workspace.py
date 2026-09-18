"""Generate a small measured-hardware workspace sample for both report designs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import optimise_measured_ctr as model
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/sample_workspace')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be a positive integer')
    args.output.mkdir(parents=True, exist_ok=True)
    record = json.loads((ROOT / 'output/task_prioritised_tradeoff_20260914/frozen_candidate.json').read_text())
    designs = {'original': model.BASE, 'proposed': np.array(record['design'])}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, layout='constrained')
    summary = {'samples_per_design': args.samples, 'seed': args.seed,
               'scope': 'Illustrative sample; not the dense report workspace estimate.',
               'rotation': 'Straight-inner rotation fixed at zero; two relative rotations sampled.',
               'designs': {}}
    for ax, (name, design) in zip(axes, designs.items()):
        states = model.samples(design, args.samples, args.seed)
        model.limits(design).validate(states[:, :3] / 1000)
        points = model.fk(states, design)
        np.savetxt(args.output / f'{name}.csv', np.column_stack([states, points]), delimiter=',',
                   header='inner_exposure_mm,middle_exposure_mm,outer_exposure_mm,inner_rotation_rad,middle_rotation_rad,outer_rotation_rad,tip_x_mm,tip_y_mm,tip_z_mm', comments='')
        radius = np.linalg.norm(points[:, :2], axis=1)
        ax.scatter(radius, points[:, 2], s=2, alpha=.45, color='#286f9e' if name == 'original' else '#14846f')
        ax.set(title=name.title(), xlabel='Radial distance (mm)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.2)
        summary['designs'][name] = {'design_vector': design.tolist(), 'sample_count': len(states)}
    axes[0].set_ylabel('Z from outside front plate (mm)')
    fig.suptitle(f'Illustrative feasible workspace samples: {args.samples:,} per design')
    fig.savefig(args.output / 'workspace-sample.png', dpi=180)
    plt.close(fig)
    (args.output / 'metadata.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Saved workspace-sample.png, original.csv, proposed.csv and metadata.json in {args.output}')


if __name__ == '__main__':
    main()
