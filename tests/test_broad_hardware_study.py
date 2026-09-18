"""Tests for the broad study's model, constraints and Pareto selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_broad_hardware_study import *


def run():
    verify()
    def r(v,vi=0):return dict(zip(OBJECTIVES,v),violation=vi)
    rows=[r([1,1,1]),r([2,2,2]),r([3,1,2]),r([99,99,99],.1)]
    assert set(fronts(rows)[0])=={1,2}
    assert len(select(rows,2))==2
    for offset in [30,35,40]:
        _,states=h.bank(1024,13801,offset);lo,hi=h.limits(offset)
        assert np.all(states[:,:3]>=lo) and np.all(states[:,:3]<=hi)
        assert np.all(states[:,0]>=states[:,1]) and np.all(states[:,1]>=states[:,2])
    _,s=h.bank(16,13802);d=np.array([24,18,20,10.])
    targets=forward(s,d);er,states,_=h.solve_batch(targets,d,seed=13803,nstarts=8,iterations=120)
    assert np.all(er<=.5),er
    # The fully curved exterior-equivalent design reproduces the long-curve model.
    assert np.max(np.abs(forward(s,[28.68,21.06,47,27.5])-forward(s,[28.68,21.06,90,65])))<1e-8
    print('PASS: general source-model verification, Pareto constraints, hardware bounds and short-curve IK.')

if __name__=='__main__':run()
