"""Design-dependent measured carriage polytope; metres in public interface."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MeasuredDesignLimits:
    total_length_mm: tuple = (350.,170.,80.)
    max_gap_mm: float = 108.

    def __post_init__(self):
        t=np.asarray(self.total_length_mm,float)
        if t.shape!=(3,) or not np.all(np.isfinite(t)) or np.any(t<=0):
            raise ValueError('Three positive finite chuck-to-tip lengths required')
        if self.max_gap_mm not in (108.,108.5): raise ValueError('Unsupported gap')
        lo=np.maximum(0,t-[275,195,115]); hi=t-[175,95,15]
        a=np.maximum(0,t[:-1]-t[1:]-71.5-self.max_gap_mm)
        b=t[:-1]-t[1:]-80
        # Bound propagation is exact for this three-variable chain.
        for _ in range(6):
            for j in (0,1):
                lo[j]=max(lo[j],lo[j+1]+a[j]); hi[j]=min(hi[j],hi[j+1]+b[j])
                lo[j+1]=max(lo[j+1],lo[j]-b[j]); hi[j+1]=min(hi[j+1],hi[j]-a[j])
        if np.any(b<a) or np.any(hi<lo): raise ValueError('Empty exterior operating domain')
        object.__setattr__(self,'lower_m',tuple(lo/1000))
        object.__setattr__(self,'upper_m',tuple(hi/1000))
        object.__setattr__(self,'difference_lower_m',a/1000)
        object.__setattr__(self,'difference_upper_m',b/1000)

    def validate(self, exposure):
        e=np.asarray(exposure,float); tol=1e-8
        if e.shape[-1:]!=(3,) or not np.all(np.isfinite(e)): raise ValueError('Three finite exposures required')
        if np.any(e<0): raise ValueError('Exterior model limit: negative tube exposure')
        if np.any(e<np.asarray(self.lower_m)-tol) or np.any(e>np.asarray(self.upper_m)+tol):
            raise ValueError('Actuator end-stop limit')
        diff=e[...,:-1]-e[...,1:]
        if np.any(diff<self.difference_lower_m-tol) or np.any(diff>self.difference_upper_m+tol):
            raise ValueError('Coupled clearance or tube ordering limit')
        return e.copy()

    def decode(self,fractions):
        f=np.asarray(fractions,float)
        if f.shape[-1:]!=(3,) or not np.all(np.isfinite(f)): raise ValueError('Three finite fractions required')
        f=np.clip(f,0,1);lo=self.lower_m;hi=self.upper_m;a=self.difference_lower_m;b=self.difference_upper_m
        o=lo[2]+f[...,2]*(hi[2]-lo[2])
        ml=np.maximum(lo[1],o+a[1]); mh=np.minimum(hi[1],o+b[1]); m=ml+f[...,1]*(mh-ml)
        il=np.maximum(lo[0],m+a[0]); ih=np.minimum(hi[0],m+b[0]); i=il+f[...,0]*(ih-il)
        return np.stack([i,m,o],axis=-1)

    def encode(self,exposure):
        i,m,o=self.validate(exposure);lo=self.lower_m;hi=self.upper_m;a=self.difference_lower_m;b=self.difference_upper_m
        ml=max(lo[1],o+a[1]);mh=min(hi[1],o+b[1]);il=max(lo[0],m+a[0]);ih=min(hi[0],m+b[0])
        return np.clip([(i-il)/(ih-il) if ih>il else 0,(m-ml)/(mh-ml) if mh>ml else 0,(o-lo[2])/(hi[2]-lo[2]) if hi[2]>lo[2] else 0],0,1)

    def interval(self,exposure,tube):
        e=self.validate(exposure);lo=self.lower_m[tube];hi=self.upper_m[tube];a=self.difference_lower_m;b=self.difference_upper_m
        if tube<2: lo=max(lo,e[tube+1]+a[tube]);hi=min(hi,e[tube+1]+b[tube])
        if tube>0: lo=max(lo,e[tube-1]-b[tube-1]);hi=min(hi,e[tube-1]-a[tube-1])
        return lo,hi

    def movement_reason(self,exposure,tube,requested):
        e=np.array(exposure,copy=True);e[tube]=requested
        try:self.validate(e)
        except ValueError as exc:return str(exc)
        return ''

    @property
    def linear_constraints_mm(self):
        A=np.vstack([np.eye(3),-np.eye(3),[1,-1,0],[-1,1,0],[0,1,-1],[0,-1,1]])
        a=self.difference_lower_m*1000;b=self.difference_upper_m*1000
        return A,np.r_[np.array(self.upper_m)*1000,-np.array(self.lower_m)*1000,b[0],-a[0],b[1],-a[1]]
