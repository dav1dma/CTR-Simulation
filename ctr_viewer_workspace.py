"""Profile-keyed viewer caches, separate from frozen research datasets."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import zipfile
import numpy as np
from ctr_operating_profile import OperatingProfile
from ctr_inverse_kinematics import ConstrainedTipIK
from ctr_workspace_map import EndpointWorkspaceMaps

CACHE_ROOT=Path(__file__).resolve().parent/'exports'/'viewer_workspace_cache'

def workspace_identity(profile, sample_count=12000, seed=42):
    root=Path(__file__).resolve().parent
    sources=['CTR_superPosKin_fun_sectioned.py','ctr_inverse_kinematics.py',
             'ctr_operating_profile.py','ctr_viewer_workspace.py']
    return dict(version=1,mode=profile.mode,design=profile.design,parameters=profile.parameters,
                lower_m=profile.limits.lower_m,upper_m=profile.limits.upper_m,
                sample_count=sample_count,seed=seed,points_per_section=3,
                source_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources})

def profile_workspace(profile: OperatingProfile, *, sample_count=12000, seed=42, cache_root=CACHE_ROOT):
    if sample_count<8:raise ValueError('At least eight samples are required')
    metadata=json.dumps(workspace_identity(profile,sample_count,seed),sort_keys=True)
    key=hashlib.sha256(metadata.encode()).hexdigest()[:24]
    folder=Path(cache_root);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{profile.mode}-{profile.design}-{key}.npz'
    if path.exists():
        try:
            with np.load(path,allow_pickle=False) as f:
                if f['metadata'].item()!=metadata:raise ValueError('Workspace metadata mismatch')
                tips=f['tips_mm'];d=f['deployment_m'];a=f['rotation_rad']
            if tips.shape!=(3,sample_count,3) or d.shape!=(sample_count,3) or a.shape!=d.shape or not np.all(np.isfinite(tips)) or not np.all(np.isfinite(a)):
                raise ValueError('Invalid workspace arrays')
            profile.limits.validate(d)
            return EndpointWorkspaceMaps(tips,d,a)
        except (ValueError,KeyError,OSError,EOFError,zipfile.BadZipFile):
            print('Rebuilding invalid viewer workspace cache:',path.name)
    print(f'Generating {sample_count:,} workspace states: {profile.label}, {profile.design}',flush=True)
    rng=np.random.default_rng(seed);fractions=rng.random((sample_count,3))
    fractions[:8]=[[i,m,o] for i in (0,1) for m in (0,1) for o in (0,1)]
    deployment=profile.limits.decode(fractions);angles=rng.uniform(-np.pi,np.pi,(sample_count,3))
    solver=ConstrainedTipIK(profile.parameters,deployment_limits=profile.limits,model_points_per_section=3)
    tips=np.array([solver.forward_endpoints_mm(d,a) for d,a in zip(deployment,angles)]).transpose(1,0,2)
    with tempfile.NamedTemporaryFile(dir=folder,suffix='.npz',delete=False) as f:
        temporary=Path(f.name)
        np.savez_compressed(f,metadata=metadata,tips_mm=tips.astype(np.float32),deployment_m=deployment.astype(np.float32),rotation_rad=angles.astype(np.float32))
    os.replace(temporary,path)
    return EndpointWorkspaceMaps(tips,deployment,angles)
