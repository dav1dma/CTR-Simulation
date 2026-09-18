"""Regression checks for shared viewer bounds and legacy-analysis compatibility."""
import sys,tempfile,csv
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from ctr_operating_profile import OperatingProfile
from ctr_inverse_kinematics import ConstrainedTipIK
from ctr_viewer_workspace import profile_workspace, workspace_identity
from ctr_motion_planner import MotionPlan,build_motion_route,route_state_at_time,RETRACT_STRATEGY
from interactive_ctr_vispy import VisPyCTRViewer


def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected rejected state')


def run_checks():
    rng=np.random.default_rng(113)
    for mode in ['hardware','tube-length']:
        for design in ['original','optimised']:
            p=OperatingProfile(mode,design);lim=p.limits
            states=lim.decode(rng.random((1000,3)))
            lim.validate(states)
            assert np.allclose([lim.decode(lim.encode(d)) for d in states],states)
            lim.validate(p.reset_m)
            solver=ConstrainedTipIK(p.parameters,deployment_limits=lim,max_iterations=40)
            for d in states[:6]:
                for tube in range(3):
                    q=d.copy();a,b=lim.interval(d,tube)
                    for v in [a,b]:q[tube]=v;lim.validate(q)
                target=solver.forward_tip_mm(d,np.zeros(3))
                result=solver.solve(target,p.reset_m,np.zeros(3))
                lim.validate(result.deployment_m)
            start=states[0];goal=states[1]
            plan=MotionPlan(np.zeros(3),start,np.zeros(3),goal,np.ones(3),np.zeros(3),0,0,'test',0,0,{},np.zeros(3))
            route=build_motion_route(plan,RETRACT_STRATEGY,deployment_limits=lim)
            assert np.allclose(route.phases[0].goal_deployment_m,lim.lower_m)
            for t in np.linspace(0,route.total_duration_s,101):lim.validate(route_state_at_time(route,t)[0])
    hardware=OperatingProfile();lim=hardware.limits
    for bad in [[0,0,0],[.12,.07,.04],[.14,.04,.02],[.05,.02,.03]]:
        rejected(lambda:lim.validate(bad))
    assert np.allclose(hardware.carriage_displacement_mm([.135,.047,.0275]),[97,109,99.5])
    # Historical callers still receive tube-length bounds and the original decode.
    legacy=ConstrainedTipIK(hardware.parameters)
    assert np.allclose(legacy.decode_deployment(np.zeros(3)),0)
    assert np.allclose(legacy.decode_deployment(np.ones(3)),[.35,.17,.08])
    with tempfile.TemporaryDirectory() as tmp:
        a=profile_workspace(hardware,sample_count=80,cache_root=tmp)
        b=profile_workspace(hardware,sample_count=80,cache_root=tmp)
        assert np.allclose(a.tips_mm,b.tips_mm)
        for p in [OperatingProfile('hardware','optimised'),OperatingProfile('tube-length','original')]:
            c=profile_workspace(p,sample_count=80,cache_root=tmp);p.limits.validate(c.deployment_m)
        assert len(list(Path(tmp).glob('*.npz')))==3
        # A corrupted matching file must be rebuilt, not displayed.
        for f in Path(tmp).glob('hardware-original*'):f.write_bytes(b'invalid cache')
        rebuilt=profile_workspace(hardware,sample_count=80,cache_root=tmp)
        assert np.allclose(a.tips_mm,rebuilt.tips_mm)
        # Validate whole import before touching the viewer.
        path=Path(tmp)/'invalid.csv'
        path.write_text('inner_extension_mm,middle_extension_mm,outer_extension_mm,inner_rotation_deg,middle_rotation_deg,outer_rotation_deg\n120,70,40,0,0,0\n')
        stub=SimpleNamespace(profile=hardware,limits=lim)
        rejected(lambda:VisPyCTRViewer.load_waypoints(stub,path))
    print('Operating profile checks passed: bounds, IK, routes, cache isolation, invalid imports and legacy defaults.')

if __name__=='__main__':run_checks()
