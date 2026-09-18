"""Additional frozen-candidate numerical audits; does not tune designs."""
from run_hardware_optimisation import *
from scipy.stats import spearmanr
from ctr_design_analysis import position_jacobians_all_endpoints


def run():
    chosen=json.loads((OUT/'shortlist_frozen.json').read_text())
    results=[]
    _,s=bank(262144,9501)
    bp=fk(s,BASE);ids=cells(bp);unique=np.unique(ids)
    # One independently generated, known-reachable baseline target per spatial cell.
    first=np.array([np.flatnonzero(ids==i)[0] for i in unique])
    bt=bp[first];weights=(2*(unique//1000)+1).astype(float)
    ref=reference(s)
    for c in chosen:
        k=np.array(c['k']);record=dict(label=c['label'],fine=evaluate(k,s,reference(s,h=1.25),h=1.25),dense=evaluate(k,s,ref))
        er,st,_=solve_batch(bt,k,seed=9502,iterations=100,nstarts=8)
        record['baseline_targets']=dict(n=len(bt),weighted_success=float(np.average(er<=.5,weights=weights)),max_error_mm=float(er.max()),p95_error_mm=float(np.percentile(er,95)))
        # Independent source-model check at found target configurations.
        d=baseline_design();d.precurvature_per_m[1:]=k;sol=ConstrainedTipIK(d.to_parameters())
        ix=np.linspace(0,len(st)-1,min(32,len(st))).astype(int)
        source=np.array([sol.forward_tip_mm(a[:3]/1000,a[3:]) for a in st[ix]])
        record['solved_source_model_discrepancy_mm']=float(np.max(np.abs(source-fk(st[ix],k))))
        # Paired spatial cell resampling quantifies finite-cell variation, not physical uncertainty.
        pp=fk(s,k);ms=(rz(pp)[:,0]<10)&(pp[:,2]>=40)&(pp[:,2]<130)
        v,_=values(s[ms],k);cc=cells(pp[ms]);near,w=region_cells()
        scores=np.array([np.median(v[cc==a]) if np.sum(cc==a)>=5 else 0 for a in near])
        record['cell_scores']=scores.tolist()
        # Shifted grid sensitivity inside the exact same cylinder.
        h=2.5;shift=1.25;ci=cells(pp[ms],h,shift);sv=[];ww=[]
        for ri in range(-1,4):
            rl=max(0,ri*h+shift);rh=min(10,(ri+1)*h+shift)
            for zi in range(15,52):
                zl=max(40,zi*h+shift);zh=min(130,(zi+1)*h+shift)
                if rh<=rl or zh<=zl:continue
                m=ci==ri*1000+zi
                sv.append(np.median(v[m]) if np.sum(m)>=5 else 0.);ww.append((rh*rh-rl*rl)*(zh-zl))
        record['shifted_grid_isotropy']=float(np.average(sv,weights=ww))
        # Track five parallel lines, no restarting or retracting between waypoints.
        xy=np.array([[0,0],[5,0],[-5,0],[0,5],[0,-5.]])
        targets=np.column_stack([xy,np.full(5,40.)]);ee,ss,x=solve_batch(targets,k,seed=9503)
        history=[ss.copy()];errors=[ee.copy()]
        for z in np.arange(41.,131.):
            ee,ss,x=solve_batch(np.column_stack([xy,np.full(5,z)]),k,iterations=35,initial=x)
            history.append(ss.copy());errors.append(ee.copy())
        history=np.array(history);error=np.array(errors)
        # Linear physical-actuator interpolation preserves linear bounds and nesting.
        between=[];deviation=[]
        for t in np.linspace(0,1,11):
            inter=(1-t)*history[:-1]+t*history[1:];pos=fk(inter.reshape(-1,6),k).reshape(90,5,3)
            desired=np.broadcast_to(xy,(90,5,2));dev=np.linalg.norm(pos[:,:,:2]-desired,axis=2)
            zgoal=np.arange(40.,130.)[:,None]+t
            deviation.append(np.max(np.sqrt(dev**2+(pos[:,:,2]-zgoal)**2)))
            between.append(pos)
        record['paths']=dict(lines=5,z_range_mm=[40,130],waypoint_spacing_mm=1,intermediate_subdivisions=10,max_waypoint_error_mm=float(error.max()),max_intermediate_error_mm=float(max(deviation)),max_deployment_step_mm=float(np.max(np.abs(np.diff(history[:,:,:3],axis=0)))),max_rotation_step_rad=float(np.max(np.abs((np.diff(history[:,:,3:],axis=0)+np.pi)%(2*np.pi)-np.pi))))
        np.savez_compressed(OUT/(c['label']+'_paths.npz'),states=history,errors=error,xy=xy)
        # Curvature sensitivity +/-2% independently and common-mode, assumed not measured.
        record['curvature_sensitivity']=[]
        for mult in [[.98,.98],[.98,1.02],[1.02,.98],[1.02,1.02]]:
            v1=evaluate(k*np.array(mult),s[:32768],reference(s[:32768]))
            record['curvature_sensitivity'].append(dict(multipliers=mult,**v1))
        # Stronger IK restarts and per-target finite movements on frozen validation targets.
        data=np.load(OUT/(c['label']+'_validation.npz'));targets=data['targets'][:64]
        er,_,_=solve_batch(targets,k,seed=9504,nstarts=8,iterations=120)
        record['stronger_ik']=dict(n=64,success=float(np.mean(er<=.5)),max_error_mm=float(er.max()))
        initial_states=data['states'][:64];lo,hi=limits()
        initial=np.column_stack([(initial_states[:,0]-lo[0])/(hi[0]-lo[0]), initial_states[:,1]/np.minimum(initial_states[:,0],hi[1]),initial_states[:,2]/np.maximum(np.minimum(initial_states[:,1],hi[2]),1e-12),initial_states[:,4:]])
        correction_errors=[]
        for axis in range(3):
            for sign in [-1,1]:
                dx=np.eye(3)[axis]*sign*.5
                ce,_,_=solve_batch(targets+dx,k,initial=initial,iterations=100)
                correction_errors.extend(ce.tolist())
        record['local_corrections']=dict(n=len(correction_errors),success=float(np.mean(np.array(correction_errors)<=.5)),strict_success_0_1mm=float(np.mean(np.array(correction_errors)<=.1)),max_error_mm=float(max(correction_errors)))
        results.append(record);jsonout('additional_validation.json',results)
        print(c['label'],record['dense'],record['baseline_targets'],record['paths'],flush=True)
    # Paired bootstrap: uniform resampling of cell indices, retain volume weights.
    rng=np.random.default_rng(9510);n=len(near);sample=rng.integers(0,n,(2000,n))
    base=np.array(results[0]['cell_scores'])
    for r in results:
        diff=np.array(r['cell_scores'])-base
        draws=np.sum(diff[sample]*w[sample],axis=1)/np.sum(w[sample],axis=1)
        r['paired_cell_bootstrap_95_interval']=np.percentile(draws,[2.5,97.5]).tolist()
    jsonout('additional_validation.json',results)
    # Post-search ranking diagnostic, not a pre-search gate or opportunity to retune.
    grid=[r for r in json.loads((OUT/'search_records.json').read_text()) if r['method']=='grid']
    selection=grid[::max(1,len(grid)//12)]
    high=[evaluate(r['k'],s[:65536],reference(s[:65536]))['isotropy'] for r in selection]
    rho=float(spearmanr([r['isotropy'] for r in selection],high).statistic)
    jsonout('rank_diagnostic.json',dict(n=len(selection),spearman=rho,role='post-search validation only; no retuning'))

if __name__=='__main__':run()
