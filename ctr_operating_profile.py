"""Explicit live-viewer operating limits; historical analysis defaults stay unchanged."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import numpy as np
from tube_parameters import build_supervisor_ctr_parameters

MODES = ('hardware', 'tube-length')
DESIGNS = ('original', 'optimised')

@dataclass(frozen=True)
class DeploymentLimits:
    lower_m: tuple[float, float, float]
    upper_m: tuple[float, float, float]

    def __post_init__(self):
        lo, hi = np.asarray(self.lower_m), np.asarray(self.upper_m)
        if lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite([lo, hi])):
            raise ValueError('Limits need three finite lower and upper bounds')
        if np.any(lo < 0) or np.any(hi <= lo) or np.any(np.diff(lo) > 0) or np.any(np.diff(hi) > 0):
            raise ValueError('Limits must be positive-span and ordered inner >= middle >= outer')

    def decode(self, fractions):
        f = np.asarray(fractions, dtype=float)
        if f.shape[-1:] != (3,) or not np.all(np.isfinite(f)):
            raise ValueError('Expected finite deployment fractions ending in three values')
        f = np.clip(f, 0, 1);lo=np.asarray(self.lower_m);hi=np.asarray(self.upper_m)
        o=lo[2]+f[...,2]*(hi[2]-lo[2])
        ml=np.maximum(lo[1],o);m=ml+f[...,1]*(hi[1]-ml)
        il=np.maximum(lo[0],m);i=il+f[...,0]*(hi[0]-il)
        return np.stack([i,m,o],axis=-1)

    def validate(self, deployment):
        d=np.asarray(deployment,dtype=float)
        if d.shape[-1:] != (3,) or not np.all(np.isfinite(d)):
            raise ValueError('Expected finite exposures ending in three values')
        tol=1e-7
        if np.any(d < np.asarray(self.lower_m)-tol) or np.any(d > np.asarray(self.upper_m)+tol) or np.any(np.diff(d,axis=-1)>tol):
            raise ValueError('Exposure is outside the active limits or violates outer <= middle <= inner')
        d=np.clip(d,self.lower_m,self.upper_m).copy()
        d[...,1]=np.maximum(d[...,1],d[...,2]);d[...,0]=np.maximum(d[...,0],d[...,1])
        return d

    def encode(self, deployment):
        i,m,o=self.validate(deployment);lo=np.asarray(self.lower_m);hi=np.asarray(self.upper_m)
        ml=max(lo[1],o);il=max(lo[0],m)
        return np.clip([(i-il)/(hi[0]-il) if hi[0]>il else 0,
                        (m-ml)/(hi[1]-ml) if hi[1]>ml else 0,
                        (o-lo[2])/(hi[2]-lo[2])],0,1)

    def interval(self, deployment, tube):
        self.validate(deployment)
        low=self.lower_m[tube];high=self.upper_m[tube]
        if tube<2:low=max(low,deployment[tube+1])
        if tube>0:high=min(high,deployment[tube-1])
        return low,high

@dataclass(frozen=True)
class OperatingProfile:
    mode: str = 'hardware'
    design: str = 'original'

    def __post_init__(self):
        if self.mode not in MODES or self.design not in DESIGNS:
            raise ValueError('Unknown operating mode or tube design')

    @property
    def parameters(self):
        p=build_supervisor_ctr_parameters()
        if self.design=='optimised':
            from measured_hardware_simulator.ctr_operating_profile import OperatingProfile as MeasuredProfile
            return MeasuredProfile(design='optimised').parameters
        return p

    @property
    def limits(self):
        if self.mode=='hardware' and self.design=='optimised':
            from measured_hardware_simulator.ctr_operating_profile import OperatingProfile as MeasuredProfile
            return MeasuredProfile(design='optimised').limits
        if self.mode=='hardware':return DeploymentLimits((.038,0.,0.),(.135,.047,.0275))
        return DeploymentLimits((0.,0.,0.),tuple(sum(v) for v in self.parameters['l_t']))

    @property
    def reset_m(self):
        # Preserve the earlier tube-length viewer's starting pose for comparison.
        return np.array(self.limits.lower_m) if self.mode=='hardware' else np.array([.120,.070,.040])

    @property
    def label(self):
        if self.mode=='hardware' and self.design=='optimised':return 'Measured coupled hardware limits (108 mm gap)'
        return 'Estimated hardware limits' if self.mode=='hardware' else 'Tube-length limits (comparison)'

    def carriage_displacement_mm(self, exposure_m):
        self.limits.validate(exposure_m)
        if self.mode!='hardware':return None
        if self.design=='optimised':
            totals=np.array([sum(v)*1000 for v in self.parameters['l_t']])
            return np.asarray(exposure_m)*1000-(totals-[275.,195.,115.])
        # Signed exposures at the measured fully-back carriage positions.
        return np.asarray(exposure_m)*1000-np.array([38.,-62.,-72.])

    def description(self, exposure_m):
        if self.mode=='hardware' and self.design=='optimised':
            from measured_hardware_simulator.ctr_operating_profile import OperatingProfile as MeasuredProfile
            return MeasuredProfile(design='optimised').description(exposure_m)+'\nUse measured viewer for equal-hardware baseline comparison.'
        lo=np.array(self.limits.lower_m)*1000;hi=np.array(self.limits.upper_m)*1000
        p=self.parameters;k=p['kappa_0'];lc=[a[1]*1000 for a in p['l_t']]
        carriage=self.carriage_displacement_mm(exposure_m)
        rows=[self.label.upper(),f'Tubes: {self.design.title()} (inner / middle / outer)',
              'Exposure limits (mm): '+ ' / '.join(f'{a:g}-{b:g}' for a,b in zip(lo,hi)),
              f'Curvature (1/m): {k[0]:g} / {k[1]:g} / {k[2]:g}',
              'Curved lengths (mm): '+' / '.join(f'{x:g}' for x in lc),
              'Assumed grip-to-tip lengths: '+' / '.join(f'{sum(v)*1000:g}' for v in p['l_t'])+' mm']
        if carriage is not None:
            rows += ['Estimated carriage travel from back (mm):', ' / '.join(f'{x:.1f}' for x in carriage)+'   of 97 / 109 / 99.5',
                     'Grip offset assumed 35 mm; fit not verified.', 'Exterior model: tips at/beyond front plate only.']
        else:rows += ['Model comparison range; not installed travel.']
        return '\n'.join(rows)

def viewer_arguments():
    parser=argparse.ArgumentParser(description='Interactive CTR with explicit operating profiles')
    parser.add_argument('--mode',choices=MODES,default='hardware')
    parser.add_argument('--tubes',choices=DESIGNS,default='original')
    parser.add_argument('--waypoints',help='Load an exported CSV; incompatible states are rejected')
    return parser.parse_args()
