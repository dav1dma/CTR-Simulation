"""Independent source-model and constraint checks for the local hardware study."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_hardware_optimisation import *
from ctr_design_analysis import position_jacobians_all_endpoints


def run():
    for offset in [30,35,40]:
        x,s=bank(1024,121,offset);lo,hi=limits(offset)
        assert np.all(s[:,:3]>=lo) and np.all(s[:,:3]<=hi)
        assert np.all(s[:,0]>=s[:,1]) and np.all(s[:,1]>=s[:,2])
    _,s=bank(32,122)
    for k in [BASE,BOUNDS[:,0],BOUNDS[:,1]]:
        d=baseline_design();d.precurvature_per_m[1:]=k;solver=ConstrainedTipIK(d.to_parameters())
        for state in s:
            assert np.allclose(fk(state,k)[0],solver.forward_tip_mm(state[:3]/1000,state[3:]),atol=1e-8,rtol=0)
            expected=position_jacobians_all_endpoints(solver,state[:3]/1000,state[3:],translation_step_mm=.01,rotation_step_rad=.0001).scaled_jacobians[0]
            assert np.allclose(jac(state,k)[0],expected,atol=1e-5,rtol=1e-6)
    center=np.column_stack([np.linspace(40,130,91),np.zeros((91,5))])
    assert np.allclose(fk(center,BASE),np.column_stack([np.zeros((91,2)),center[:,0]]),atol=1e-10)
    # Recover independently generated known feasible positions.
    targets=fk(s,BASE)
    errors,states,_=solve_batch(targets,BASE,seed=123,nstarts=8,iterations=120)
    assert np.all(errors<=.5), errors.max()
    lo,hi=limits();assert np.all(states[:,:3]>=lo) and np.all(states[:,:3]<=hi)
    print('PASS: hardware bounds, nesting, FK and Jacobian agreement, exact centreline and independent IK recovery.')

if __name__=='__main__':run()
