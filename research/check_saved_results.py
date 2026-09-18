"""Check packaged report evidence, optionally recomputing metrics from raw arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def check():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', action='store_true', help='Also check arrays from chapter5-dense-data.tar.gz')
    args = parser.parse_args()
    gallery = ROOT / 'docs/dissertation'
    manifest = read(gallery / 'manifest.json')
    assert manifest['report_pdf_pages'] == 88
    assert len(manifest['figures']) == 12
    for entry in manifest['figures'] + manifest['data']:
        path = gallery / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
    summary = read(gallery / 'data/summary.json')
    overlay = read(gallery / 'data/overlay.json')
    np.testing.assert_allclose(overlay['shared'] + overlay['lost'], summary['original']['workspace_cm3'])
    np.testing.assert_allclose(overlay['shared'] + overlay['gained'], summary['proposed']['workspace_cm3'])
    candidate = read(gallery / 'data/frozen_candidate.json')
    assert candidate['id'] == 't027'
    geometry = read(ROOT / 'proposed_tradeoff_simulator/optimised_configuration.json')
    np.testing.assert_allclose(candidate['design'], geometry['total_length_mm'] + geometry['curved_length_mm'][1:] + geometry['precurvature_per_m'][1:])
    assert read(gallery / 'data/path_outcome.json')['accepted'] is False
    print('PASS: final-report figure/data hashes, workspace totals, t027 geometry and failed acceptance record')
    if args.raw:
        data = ROOT / 'output/chapter5_measured_revision/data'
        needed = ['targets.npz', 'original_spatial.npz', 'proposed_spatial.npz', 'original_ik.npz', 'proposed_ik.npz']
        missing = [name for name in needed if not (data / name).exists()]
        if missing:
            parser.error('Extract chapter5-dense-data.tar.gz from the repository root first. Missing: ' + ', '.join(missing))
        with np.load(data / 'targets.npz') as targets:
            ids = targets['cell_ids']
            assert len(targets['targets']) == 59341
            cells, counts = np.unique(ids, return_counts=True)
            weights = (2 * (ids // 1000) + 1) / counts[np.searchsorted(cells, ids)]
        for name in ('original', 'proposed'):
            with np.load(data / f'{name}_spatial.npz') as spatial:
                assert len(spatial['states']) == 262144
                computed = np.floor(np.linalg.norm(spatial['points'][:, :2], axis=1) / 5).astype(int) * 1000 + np.floor(spatial['points'][:, 2] / 5).astype(int)
                np.testing.assert_array_equal(np.unique(computed), spatial['cells'])
                volume = np.sum(2 * (spatial['cells'] // 1000) + 1) * np.pi * 5**3 / 1000
                np.testing.assert_allclose(volume, summary[name]['workspace_cm3'], rtol=1e-12)
            with np.load(data / f'{name}_ik.npz') as ik:
                errors = ik['errors']
                assert len(errors) == len(ids)
                success = np.average(errors <= .5, weights=weights) * 100
                mean = np.average(errors, weights=weights)
                np.testing.assert_allclose(success, summary[name]['global_ik']['success_pct'], rtol=1e-12)
                np.testing.assert_allclose(mean, summary[name]['global_ik']['mean_mm'], rtol=1e-12)
                print(f'PASS: {name}: {volume:.2f} cm³ occupied, {success:.3f}% fixed-start success, {len(errors):,} targets')


if __name__ == '__main__':
    check()
