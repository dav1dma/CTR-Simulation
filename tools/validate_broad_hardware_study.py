"""Independent validation of the frozen broad-search shortlist. No retuning."""
from run_broad_hardware_study import *


def freeze_reference():
    _,states=h.bank(65536,11201)
    # Matches the final pre-validation refinement task region exactly.
    ref=spatial_data(BASE,states)
    write('reference_frozen.json',dict(ids=ref['ids'],cell_size_mm=2.5,seed=11201,n=65536,role='final refinement baseline cells; frozen before independent validation'))
    return ref


def common_reference(ref,baseline):
    return dict(ids=ref['ids'],near=baseline['near'])


def paths(design):
    xy=np.array([[0.,0.],[5,0],[-5,0],[0,5],[0,-5]])
    targets=np.column_stack([xy,np.full(5,40.)]);er,s,x=h.solve_batch(targets,design,seed=13401)
    history=[s];errors=[er]
    for z in np.arange(41.,131.):
        er,s,x=h.solve_batch(np.column_stack([xy,np.full(5,z)]),design,iterations=40,initial=x)
        history.append(s);errors.append(er)
    history=np.array(history);errors=np.array(errors);deviation=[];minprogress=np.inf
    delta=np.diff(history,axis=0);delta[:,:,3:]=(delta[:,:,3:]+np.pi)%(2*np.pi)-np.pi
    previous=None
    for t in np.linspace(0,1,11):
        inter=history[:-1]+t*delta;p=forward(inter.reshape(-1,6),design).reshape(90,5,3)
        zgoal=np.arange(40.,130.)[:,None]+t
        deviation.append(float(np.max(np.sqrt(np.sum((p[:,:,:2]-xy)**2,axis=2)+(p[:,:,2]-zgoal)**2))))
        if previous is not None:minprogress=min(minprogress,float(np.min(p[:,:,2]-previous[:,:,2])))
        previous=p
    return dict(max_waypoint_error_mm=float(errors.max()),max_interpolated_error_mm=max(deviation),pass_0_5mm=max(deviation)<=.5,max_translation_step_mm=float(np.max(np.abs(delta[:,:,:3]))),max_rotation_step_rad=float(np.max(np.abs(delta[:,:,3:]))),minimum_substep_z_progress_mm=minprogress),history,errors


def run():
    chosen=json.loads((OUT/'shortlist_frozen.json').read_text());ref=freeze_reference();start=time.time()
    # Baseline cell target set remains physically identical for every design.
    _,bs=h.bank(131072,13101);bp=forward(bs,BASE);ids=h.cells(bp)
    refids=ref['ids'];first=[];tw=[]
    for cell in refids:
        inds=np.flatnonzero(ids==cell)
        if len(inds):first.append(inds[0]);tw.append(2*(cell//1000)+1)
    bt=bp[first];tw=np.array(tw)
    u=qmc.Sobol(3,scramble=True,seed=13201).random_base2(10)
    rr=10*np.sqrt(u[:,0]);ang=2*np.pi*u[:,1]
    near=np.column_stack([rr*np.cos(ang),rr*np.sin(ang),40+90*u[:,2]])
    u2=qmc.Sobol(3,scramble=True,seed=13202).random_base2(9)
    rr=100*np.sqrt(u2[:,0]);ang=2*np.pi*u2[:,1]
    broad=np.column_stack([rr*np.cos(ang),rr*np.sin(ang),135*u2[:,2]])
    axis=np.column_stack([np.zeros((91,2)),np.arange(40.,131.)])
    np.savez_compressed(OUT/'validation_targets.npz',near=near,broad=broad,axis=axis,baseline=bt,baseline_weights=tw)
    # Compute reference quality for each validation budget independently; membership fixed.
    refs={};banks={}
    for n in [32768,131072]:
        _,states=h.bank(n,13301);banks[n]=states;refs[n]=common_reference(ref,spatial_data(BASE,states))
    results=[]
    for c in chosen:
        d=np.array(c['design']);record=dict(label=c['label'],design=d.tolist(),spatial=[])
        for n in banks:record['spatial'].append(dict(n=n,**evaluate(d,banks[n],refs[n])))
        print('Spatial done',c['label'],round(time.time()-start,1),flush=True)
        nearerr,ns,_=h.solve_batch(near,d,seed=13402,nstarts=4,iterations=80)
        be,_,_=h.solve_batch(bt,d,seed=13402,nstarts=6,iterations=100)
        ae,_,_=h.solve_batch(axis,d,seed=13402,nstarts=4,iterations=80)
        # Uniform broad-region targets are not generated from any candidate workspace.
        we,_,_=h.solve_batch(broad,d,seed=13402,nstarts=4,iterations=60)
        record['targets']=dict(near_n=len(near),near_success=float(np.mean(nearerr<=.5)),near_strict_success=float(np.mean(nearerr<=.1)),near_max_mm=float(nearerr.max()),baseline_n=len(bt),baseline_weighted_success=float(np.average(be<=.5,weights=tw)),baseline_max_mm=float(be.max()),axis_success=float(np.mean(ae<=.5)),axis_max_mm=float(ae.max()),broad_n=len(broad),broad_success=float(np.mean(we<=.5)),broad_ik_volume_estimate_cm3=float(np.mean(we<=.5)*np.pi*100**2*135/1000))
        print('Targets done',c['label'],record['targets'],flush=True)
        record['paths'],history,errors=paths(d)
        lo,hi=h.limits();initial=np.column_stack([(ns[:64,0]-lo[0])/(hi[0]-lo[0]),ns[:64,1]/np.minimum(ns[:64,0],hi[1]),ns[:64,2]/np.maximum(np.minimum(ns[:64,1],hi[2]),1e-12),ns[:64,4:]])
        corrections=[]
        for ax in np.eye(3):
            for sign in [-1,1]:
                err,_,_=h.solve_batch(near[:64]+sign*.5*ax,d,initial=initial,iterations=80);corrections.extend(err.tolist())
        record['corrections']=dict(n=len(corrections),success_0_1mm=float(np.mean(np.array(corrections)<=.1)),max_mm=max(corrections))
        record['offset_sensitivity']=[]
        for offset in [30.,40.]:
            er,_,_=h.solve_batch(near[:128],d,offset=offset,seed=13402,iterations=80)
            record['offset_sensitivity'].append(dict(offset_mm=offset,n=128,success=float(np.mean(er<=.5)),max_mm=float(er.max())))
        # Standard source model, including curved-section transitions, at solved states.
        sol=ConstrainedTipIK(design_object(d).to_parameters());ix=np.arange(0,len(ns),64)
        expected=np.array([sol.forward_tip_mm(v[:3]/1000,v[3:]) for v in ns[ix]])
        record['source_model_discrepancy_mm']=float(np.max(np.abs(expected-forward(ns[ix],d))))
        record['parameter_sensitivity']=[]
        for mult in [[.98,.98],[.98,1.02],[1.02,.98],[1.02,1.02]]:
            dd=d.copy();dd[:2]*=mult
            record['parameter_sensitivity'].append(dict(curvature_multipliers=mult,**evaluate(dd,banks[32768],refs[32768])))
        # Curved lengths have independent +/-1 mm uncertainty for this sensitivity only.
        for change in [-1.,1.]:
            dd=d.copy();dd[2:]+=change
            record['parameter_sensitivity'].append(dict(curved_length_change_mm=change,**evaluate(dd,banks[32768],refs[32768])))
        np.savez_compressed(OUT/(c['label']+'_validation.npz'),near_errors=nearerr,baseline_errors=be,axis_errors=ae,broad_errors=we,near_states=ns,path_states=history,path_errors=errors)
        record['passes_control_checks']=bool(record['targets']['near_success']>=.99 and record['targets']['baseline_weighted_success']>=.95 and record['paths']['pass_0_5mm'] and record['corrections']['success_0_1mm']>=.99)
        results.append(record);write('validation.json',results)
        print('Completed',c['label'],'control_pass',record['passes_control_checks'],round(time.time()-start,1),flush=True)
    write('validation_summary.json',dict(seconds=time.time()-start,seeds=[13101,13201,13202,13301,13401,13402],final_candidates=len(chosen),refinement_region_frozen_before_validation=True,source_model_is_not_physical_validation=True))

if __name__=='__main__':run()
