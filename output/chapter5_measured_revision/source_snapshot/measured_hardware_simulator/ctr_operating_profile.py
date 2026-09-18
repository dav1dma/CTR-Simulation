"""Measured coupled carriage constraints. All lengths in metres internally.

Original 350/170/80 mm lengths provisionally mean chuck-front to distal tip.
Only exterior states with nested, nonnegative material exposures are modelled.
"""
from dataclasses import dataclass
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from tube_parameters import build_supervisor_ctr_parameters
try:
    from .design_constraints import MeasuredDesignLimits
except ImportError:
    from design_constraints import MeasuredDesignLimits

MODES = ('hardware', 'hardware-108.5')
DESIGNS = ('original','optimised')
MODEL_ID = 'measured-ctr-20260914-v1'

@dataclass(frozen=True)
class DeploymentLimits:
    lower_m: tuple = (.075, 0., 0.)
    upper_m: tuple = (.175, .075, .065)
    max_gap_mm: float = 108.

    def __post_init__(self):
        if self.lower_m != (.075,0.,0.) or self.upper_m != (.175,.075,.065):
            raise ValueError('This simulator supports measured original-tube bounds only')
        if self.max_gap_mm not in (108.,108.5):
            raise ValueError('Maximum gap must be 108 or 108.5 mm')

    @property
    def inner_difference_min(self):
        return (108.5-self.max_gap_mm)/1000

    def validate(self, exposure):
        e=np.asarray(exposure,dtype=float)
        if e.shape[-1:]!=(3,) or not np.all(np.isfinite(e)):
            raise ValueError('Expected three finite tube exposures')
        # Roundoff allowance is 0.00001 mm; never repair or clip imported exposures.
        tol=1e-8
        if np.any(e < 0):
            raise ValueError('Exterior model limit: negative exposure places a tip behind the plate')
        if np.any(e < np.asarray(self.lower_m)-tol) or np.any(e > np.asarray(self.upper_m)+tol):
            raise ValueError('Actuator end-stop limit: exposure outside measured carriage range')
        i,m,o=np.moveaxis(e,-1,0)
        if np.any(i<m-tol) or np.any(m<o-tol):
            raise ValueError('Tube ordering limit: outer <= middle <= inner is required')
        if np.any(i-m < self.inner_difference_min-tol) or np.any(i-m > .100+tol) or np.any(m-o > .010+tol):
            raise ValueError('Coupled stop limit: adjacent chuck-to-rear gap outside 8.5–'+str(self.max_gap_mm)+' mm')
        return e.copy()

    def decode(self, fractions):
        f=np.asarray(fractions,dtype=float)
        if f.shape[-1:]!=(3,) or not np.all(np.isfinite(f)):
            raise ValueError('Expected finite fractions ending in three values')
        f=np.clip(f,0,1)
        o=f[...,2]*.065
        m=o+f[...,1]*(np.minimum(.075,o+.010)-o)
        low=np.maximum(.075,m+self.inner_difference_min)
        high=np.minimum(.175,m+.100)
        i=low+f[...,0]*(high-low)
        return np.stack((i,m,o),axis=-1)

    def encode(self, exposure):
        i,m,o=self.validate(exposure)
        lo=max(.075,m+self.inner_difference_min); hi=min(.175,m+.100)
        return np.clip([(i-lo)/(hi-lo) if hi>lo else 0.,
                        (m-o)/(min(.075,o+.010)-o),o/.065],0,1)

    def interval(self, exposure, tube):
        i,m,o=self.validate(exposure)
        if tube==0: return max(.075,m+self.inner_difference_min),min(.175,m+.100)
        if tube==1: return max(0.,o,i-.100),min(.075,o+.010,i-self.inner_difference_min)
        if tube==2: return max(0.,m-.010),min(.065,m)
        raise ValueError('Unknown tube')

    def movement_reason(self, exposure, tube, requested):
        trial=np.array(exposure,copy=True); trial[tube]=requested
        try: self.validate(trial)
        except ValueError as exc: return str(exc)
        return ''

@dataclass(frozen=True)
class OperatingProfile:
    mode: str='hardware'
    design: str='original'

    def __post_init__(self):
        if self.mode not in MODES or self.design not in DESIGNS:
            raise ValueError('Unknown measured hardware profile or tube design')

    @property
    def parameters(self):
        if self.design=='original':return build_supervisor_ctr_parameters()
        path=Path(__file__).resolve().parent/'optimised_configuration.json'
        record=json.loads(path.read_text())
        if record['hardware_model']!='measured-ctr-20260914-v1':raise ValueError('Optimised configuration has incompatible hardware provenance')
        return record['model_parameters']
    @property
    def total_length_mm(self):return np.round([sum(v)*1000 for v in self.parameters['l_t']],9)
    @property
    def limits(self):
        gap=108.5 if self.mode=='hardware-108.5' else 108.
        return DeploymentLimits(max_gap_mm=gap) if self.design=='original' else MeasuredDesignLimits(tuple(self.total_length_mm),gap)
    @property
    def reset_m(self): return self.limits.decode(np.zeros(3))
    @property
    def label(self): return f'Measured coupled limits • gap {self.limits.max_gap_mm:g} mm'
    @property
    def identity(self):
        identity=f'{MODEL_ID}-gap-{self.limits.max_gap_mm:g}'
        if self.design=='optimised':identity+='-optimised-'+hashlib.sha256(json.dumps(self.parameters,sort_keys=True).encode()).hexdigest()[:12]
        return identity

    def carriage_displacement_mm(self, exposure):
        return self.limits.validate(exposure)*1000-(self.total_length_mm-np.array([275.,195.,115.]))

    def description(self, exposure):
        e=self.limits.validate(exposure)*1000
        d=self.total_length_mm-e
        g=d[:-1]-d[1:]-71.5
        travel=self.carriage_displacement_mm(exposure)
        return '\n'.join([self.label.upper(),
            f'{self.design.title()} tubes | order: inner / middle / outer',
            'Carriage travel from rear stop (mm), of 100:',
            ' / '.join(f'{v:.1f}' for v in travel),
            'Chuck to front plate (mm): '+' / '.join(f'{v:.1f}' for v in d),
            'Adjacent gaps rear / front (mm): '+' / '.join(f'{v:.1f}' for v in g),
            'Assumed chuck-to-tip: '+' / '.join(f'{v:g}' for v in self.total_length_mm)+' mm',
            'Retention unknown; no 35 mm offset.',
            'Guidance unmodelled; rotations unrestricted.',
            'Exterior model: tips at / beyond plate only.'])

def viewer_arguments():
    p=argparse.ArgumentParser(description='Separate original CTR simulator with measured coupled stops')
    p.add_argument('--mode',choices=MODES,default='hardware')
    p.add_argument('--tubes',choices=DESIGNS,default='original')
    p.add_argument('--waypoints',help='Import only CSV states matching this measured profile')
    return p.parse_args()
