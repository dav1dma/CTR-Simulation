"""Reproducible bounded search using measured coupled stops, never old offsets.

Commands: screen, select, validate. Selection precedes held-out validation.
All geometry and success metrics are predictions of the ideal exterior model.
"""
from pathlib import Path
import sys,os,json,time,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctr-measured-opt-mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'measured_hardware_simulator'))
import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree
from design_constraints import MeasuredDesignLimits
from tube_parameters import build_supervisor_ctr_parameters
# Reuse the section-aware vector kernel only; none of its old limit functions.
from run_broad_hardware_study import forward
OUT=ROOT/'output/measured_hardware_optimisation_20260914'
BASE=np.array([350.,170.,80.,90.,65.,19.12,14.04])
BOUNDS=np.array([[320,380],[170,230],[80,140],[30,110],[20,90],[12,27],[9,21]],float)
SCALE=np.array([350,170,80,np.pi,np.pi,np.pi])

def write(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v)))

def limits(d,gap=108.):return MeasuredDesignLimits(tuple(d[:3]),gap)
def fk(s,d):return forward(s,np.r_[d[5:7],d[3:5]])
def pars(d):
    p=build_supervisor_ctr_parameters();lc=np.r_[0,d[3:5]]
    p['l_t']=np.column_stack([d[:3]-lc,lc]).tolist()
    p['l_t']=(np.array(p['l_t'])/1000).tolist();p['kappa_0']=[0.,*d[5:7]]
    return p
def sobol(n,dim,seed):return qmc.Sobol(dim,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
def decode(x,d,gap=108.):
    x=np.atleast_2d(x)
    return np.column_stack([limits(d,gap).decode(x[:,:3])*1000,np.zeros(len(x)),x[:,3:]*np.pi])
def samples(d,n,seed,gap=108.):
    # Rejection from physical carriage box gives uniform feasible configurations.
    chunks=[];source=qmc.Sobol(5,scramble=True,seed=seed)
    total=0
    while total<n:
        u=source.random(8192);travel=u[:,:3]*100
        e=d[:3]-[275,195,115]+travel
        dist=d[:3]-e;g=dist[:,:-1]-dist[:,1:]-71.5
        mask=(e>=0).all(1)&(np.diff(e,axis=1)<=0).all(1)&(g>=8.5).all(1)&(g<=gap).all(1)
        s=np.column_stack([e[mask],np.zeros(mask.sum()),(2*u[mask,3:]-1)*np.pi])
        chunks.append(s);total+=len(s)
        if len(chunks)>1000:raise ValueError('Insufficient feasible physical sampling volume')
    return np.concatenate(chunks)[:n]
def iso(s,d,step=.01):
    s=np.atleast_2d(s);J=np.zeros((len(s),3,6))
    # Derivative of physical exposures; constraints reported separately.
    # One-sided at zero/order boundaries, symmetric wherever possible.
    for j in (0,1,2,4,5):
        plus=s.copy();minus=s.copy();h=step if j<3 else step/100
        plus[:,j]+=h;minus[:,j]-=h
        if j<3:
            low=s[:,j+1] if j<2 else np.zeros(len(s))
            high=s[:,j-1] if j>0 else np.full(len(s),d[0])
            plus[:,j]=np.minimum(plus[:,j],high);minus[:,j]=np.maximum(minus[:,j],low)
        den=plus[:,j]-minus[:,j]
        J[:,:,j]=(fk(plus,d)-fk(minus,d))/np.maximum(den,1e-12)[:,None]*SCALE[j]
    sv=np.linalg.svd(J,compute_uv=False)
    return sv[:,-1]/np.maximum(sv[:,0],1e-12)
def cells(p,h=5.):return np.floor(np.linalg.norm(p[:,:2],axis=1)/h).astype(int)*1000+np.floor(p[:,2]/h).astype(int)
def wquant(v,w,q):
    i=np.argsort(v);return float(np.asarray(v)[i[np.searchsorted(np.cumsum(np.asarray(w)[i])/sum(w),q)]])
def spatial(d,n,seed,h=5.,gap=108.):
    s=samples(d,n,seed,gap);p=fk(s,d);v=iso(s,d);ids=cells(p,h)
    order=np.argsort(ids);u,start,count=np.unique(ids[order],return_index=True,return_counts=True)
    med=np.array([np.median(v[order[b:b+c]]) if c>=5 else 0 for b,c in zip(start,count)])
    return dict(states=s,points=p,iso=v,ids=ids,cells=u,medians=med,counts=count)
def map_values(sp,ids):
    idx=np.searchsorted(sp['cells'],ids);idx=np.minimum(idx,len(sp['cells'])-1)
    return np.where(sp['cells'][idx]==ids,sp['medians'][idx],0)
def solve(targets,d,initial=None,iterations=40,gap=108.):
    n=len(targets);x=np.zeros((n,5)) if initial is None else np.array(initial,copy=True)
    err=np.linalg.norm(fk(decode(x,d,gap),d)-targets,axis=1)
    stalled=np.zeros(n,bool)
    for _ in range(iterations):
        ids=np.flatnonzero((err>.01)&~stalled)
        if not len(ids):break
        a=x[ids];p=fk(decode(a,d,gap),d);res=targets[ids]-p;J=np.empty((len(ids),3,5));h=1e-4
        for j in range(5):
            plus=a.copy();step=np.where(a[:,j]+h>1,-h,h) if j<3 else np.full(len(a),h)
            plus[:,j]+=step
            J[:,:,j]=(fk(decode(plus,d,gap),d)-p)/step[:,None]
        delta=np.einsum('nji,nj->ni',J,np.linalg.solve(J@J.transpose(0,2,1)+np.eye(3),res[...,None])[...,0])
        delta=np.clip(delta,-.1,.1);accepted=np.zeros(len(ids),bool)
        for alpha in (1.,.5,.25,.1):
            trial=a+alpha*delta;trial[:,:3]=np.clip(trial[:,:3],0,1)
            e=np.linalg.norm(fk(decode(trial,d,gap),d)-targets[ids],axis=1);better=(e<err[ids]-1e-9)&~accepted
            x[ids[better]]=trial[better];err[ids[better]]=e[better]
            accepted|=better
        stalled[ids[~accepted]]=True
    s=decode(x,d,gap);limits(d,gap).validate(s[:,:3]/1000)
    return err,s,x
def diagnostic(targets,d,gap=108.):
    bank=samples(d,16384,35107,gap);_,near=cKDTree(fk(bank,d)).query(targets,k=2)
    best=np.full(len(targets),np.inf);states=np.zeros((len(targets),6))
    for j in range(2):
        s=bank[near[:,j]];x=np.column_stack([np.array([limits(d,gap).encode(e/1000) for e in s[:,:3]]),s[:,4:]/np.pi])
        er,st,_=solve(targets,d,x,60,gap);m=er<best;best[m]=er[m];states[m]=st[m]
    return best,states
def targetbank(n,seed,ref):
    s=samples(BASE,65536,seed);p=fk(s,BASE);ids=cells(p)
    rng=np.random.default_rng(seed+1);allids=ref['cells'][ref['counts']>=5]
    pools=[allids,ref['weak_cells'],ref['corridor_cells']]
    selected=[];groups=[];weights=[]
    for group,pool in enumerate(pools):
        present=np.intersect1d(pool,np.unique(ids));k=n if group==0 else max(64,n//4)
        # Uniform cell choice with annular-volume weights, real reachable points.
        chosen=rng.choice(present,k,replace=True)
        for c in chosen:
            selected.append(rng.choice(np.flatnonzero(ids==c)));weights.append(2*(c//1000)+1);groups.append(group)
    ix=np.array(selected)
    return dict(targets=p[ix],source_states=s[ix],groups=np.array(groups),weights=np.array(weights,float))
def metrics(t,err,st,d):
    v=iso(st,d);r={}
    for j,name in enumerate(('global','weak','corridor')):
        m=t['groups']==j;w=t['weights'][m];success=err[m]<=.5
        r[name]=dict(success=float(np.average(success,weights=w)),p95_mm=wquant(err[m],w,.95),
                     effective_isotropy_mean=float(np.average(np.where(success,v[m],0),weights=w)),
                     effective_isotropy_p10=wquant(np.where(success,v[m],0),w,.1))
    return r
def verify():
    from CTR_superPosKin_fun_sectioned import superPosKin
    ds=[BASE,np.array([355,205,110,55,35,23,16])];maximum=0
    for d in ds:
        s=samples(d,24,33101)
        for row,p in zip(s,fk(s,d)):
            expected=np.array(superPosKin(pars(d),{'ul':row[:3]/1000,'uphi':row[3:]},{'n_p':1,'isPlot':False})[0][:3])*1000
            maximum=max(maximum,float(np.max(np.abs(expected-p))))
        lim=limits(d);a=lim.decode(sobol(4096,3,33102));lim.validate(a)
        dist=d[:3]-a*1000;g=dist[:,:-1]-dist[:,1:]-71.5
        assert np.min(g)>=8.5-1e-7 and np.max(g)<=108+1e-7
        for e in a[::100]:np.testing.assert_allclose(lim.decode(lim.encode(e)),e,atol=1e-12)
    assert maximum<1e-7
    write('verification.json',dict(forward_max_absolute_error_mm=maximum,checked_against='inherited section-aware forward model',states=48))
def screen():
    verify();start=time.time()
    protocol=dict(id='measured-optimisation-20260914-v1',baseline=BASE,bounds=BOUNDS,
        variables=['inner_total_mm','middle_total_mm','outer_total_mm','middle_curve_mm','outer_curve_mm','middle_curvature_per_m','outer_curvature_per_m'],
        measured=dict(retracted_mm=[275,195,115],extended_mm=[175,95,15],body_mm=71.5,gap_mm=[8.5,108],sensitivity_gap_mm=108.5),
        assumptions=['Lengths mean chuck-tip to distal-tip material length; no grip offset','Straight coaxial internal path assumed, guidance/contact not simulated','Distal precurves, straight inner wire, original diameters and 75 GPa stiffness fixed','Unrestricted rotations; no friction, torsion, elastic stability or loaded accuracy','Bounds exploratory, not supplier or manufacturing limits'],
        methods=dict(cell_mm=5,minimum_support=5,physical_jacobian_scale=SCALE,weak_region='lowest 20% annular-volume-weighted baseline supported cell median isotropy',corridor='baseline supported subset r<10, z40..130mm',targets='actual baseline reachable positions, cell-stratified with annular weights',ik='single start at most-retracted feasible state, zero rotations, 40 iterations, damping1mm, step0.1, stop0.01mm, success0.5mm; same algorithm/budget per design',
        screen='256 Sobol designs plus structured controls; 8192 feasible uniform physical states; shortlist 12 on frozen-region isotropy and retention',selection='fresh training targets: require global/weak/corridor success no worse by >0.02 absolute; rank equal global/weak/corridor success plus 0.25 normalized weak and corridor effective mean isotropy',validation='freeze selected design before independent targets; no validation retuning'),
        seeds=dict(region=33111,screen=33112,designs=33113,selection=34101,validation=36101))
    write('protocol.json',protocol)
    ref=spatial(BASE,65536,33111);supported=ref['counts']>=5;ids=ref['cells'][supported];v=ref['medians'][supported];w=2*(ids//1000)+1
    ref['weak_cells']=ids[v<=wquant(v,w,.2)]
    ref['corridor_cells']=ids[(ids//1000<2)&(ids%1000>=8)&(ids%1000<26)]
    np.savez_compressed(OUT/'baseline_regions.npz',**ref)
    designs=[BASE]
    for dm in (0,10,20,30,40):
        for do in (0,10,20,30):
            if do<=dm:
                designs.extend([np.array([350,170+dm,80+do,90,65,19.12,14.04]),np.array([350,170+dm,80+do,60,40,22,16])])
    designs.extend(BOUNDS[:,0]+sobol(256,7,33113)*(BOUNDS[:,1]-BOUNDS[:,0]))
    rows=[]
    for index,d in enumerate(designs):
        if d[3]>d[1] or d[4]>d[2]:continue
        try:limits(d)
        except ValueError:continue
        sp=spatial(d,8192,33112);med=map_values(sp,ids);ret=float(np.average(np.isin(ids,sp['cells']),weights=w))
        weak=np.isin(ids,ref['weak_cells']);corr=np.isin(ids,ref['corridor_cells'])
        vals=dict(retention=ret,global_iso=float(np.average(med,weights=w)),weak_iso=float(np.average(med[weak],weights=w[weak])),corridor_iso=float(np.average(med[corr],weights=w[corr])))
        if not rows:base=vals.copy()
        score=sum(vals[k]/max(base[k],1e-5) for k in ('weak_iso','corridor_iso'))+vals['global_iso']/base['global_iso']-20*max(0,.95-ret)
        rows.append(dict(design=d,score=score,**vals))
        if index%20==0:print('screen',index,len(designs),'seconds',round(time.time()-start),flush=True);write('screen_records.json',rows)
    write('screen_records.json',rows)
    ranked=sorted(rows[1:],key=lambda r:r['score'],reverse=True)
    shortlist=[rows[0]]
    for r in ranked:
        if min(np.linalg.norm((r['design']-v['design'])/(BOUNDS[:,1]-BOUNDS[:,0])) for v in shortlist)>.08:
            shortlist.append(r)
        if len(shortlist)==13:break
    write('shortlist.json',shortlist);print('shortlist',[(r['design'].tolist(),round(r['score'],3)) for r in shortlist],flush=True)
def select():
    ref=dict(np.load(OUT/'baseline_regions.npz'));t=targetbank(512,34101,ref);np.savez_compressed(OUT/'selection_targets.npz',**t)
    rows=json.loads((OUT/'shortlist.json').read_text());out=[]
    for r in rows:
        d=np.array(r['design']);er,st,_=solve(t['targets'],d);m=metrics(t,er,st,d)
        if not out:base=m
        eligible=all(m[g]['success']>=base[g]['success']-.02 for g in base)
        score=sum(m[g]['success'] for g in base)+.25*sum(m[g]['effective_isotropy_mean']/max(base[g]['effective_isotropy_mean'],1e-6) for g in ('weak','corridor'))
        result=dict(design=d,metrics=m,eligible=eligible,score=score);out.append(result)
        print('selection',len(out),result,flush=True);write('selection_records.json',out)
    selected=max((r for r in out if r['eligible']),key=lambda r:r['score'])
    # Round before independent validation, retaining a reproducible selected design.
    selected['design']=np.round(selected['design'],2)
    write('selected_frozen.json',selected)
def refine():
    """Training-only local exploration; freeze before opening held-out results."""
    out=json.loads((OUT/'selection_records.json').read_text());base=out[0]['metrics']
    t=dict(np.load(OUT/'selection_targets.npz'));best=np.array(max((r for r in out if r['eligible']),key=lambda r:r['score'])['design'])
    designs=[]
    for centre in (BASE,best):
        span=np.array([15,20,15,30,25,5,5])
        designs.extend(np.clip(centre+(sobol(64,7,34201+len(designs))*2-1)*span,BOUNDS[:,0],BOUNDS[:,1]))
        for j in range(7):
            for sign in (-1,1):
                d=centre.copy();d[j]+=sign*span[j]/2;designs.append(np.clip(d,BOUNDS[:,0],BOUNDS[:,1]))
    protocol=json.loads((OUT/'protocol.json').read_text())
    protocol['methods']['refinement']='Training-only: 128 Sobol local samples plus 28 axis perturbations around original and best shortlist design; per-region p95 gate baseline +1mm; no held-out feedback'
    write('protocol.json',protocol)
    for r in out:r['eligible']=r['eligible'] and all(r['metrics'][g]['p95_mm']<=base[g]['p95_mm']+1 for g in base)
    for i,d in enumerate(designs):
        if d[3]>d[1] or d[4]>d[2]:continue
        try:limits(d)
        except ValueError:continue
        er,st,_=solve(t['targets'],d);m=metrics(t,er,st,d)
        eligible=all(m[g]['success']>=base[g]['success']-.02 and m[g]['p95_mm']<=base[g]['p95_mm']+1 for g in base)
        score=sum(m[g]['success'] for g in base)+.25*sum(m[g]['effective_isotropy_mean']/max(base[g]['effective_isotropy_mean'],1e-6) for g in ('weak','corridor'))
        out.append(dict(design=d,metrics=m,eligible=eligible,score=score))
        if i%10==0:
            print('refine',i,'best',max((r for r in out if r['eligible']),key=lambda r:r['score']),flush=True)
            write('refinement_records.json',out)
    selected=max((r for r in out if r['eligible']),key=lambda r:r['score']);selected['design']=np.round(selected['design'],2)
    # Check the rounded design against the same training gates before freezing.
    er,st,_=solve(t['targets'],selected['design']);selected['rounded_training_metrics']=metrics(t,er,st,selected['design'])
    write('refinement_records.json',out);write('selected_frozen.json',selected)
def fine():
    out=json.loads((OUT/'refinement_records.json').read_text());base=out[0]['metrics'];t=dict(np.load(OUT/'selection_targets.npz'))
    centre=np.array(max(out,key=lambda r:r['score'])['design']);designs=[]
    for km in np.linspace(centre[5]-2,centre[5]+2,9):
        for ko in np.linspace(centre[6]-2,centre[6]+2,9):
            d=centre.copy();d[5:]=[km,ko];designs.append(d)
    span=np.array([4,5,5,10,10,2,2])
    designs.extend(np.clip(centre+(sobol(64,7,34301)*2-1)*span,BOUNDS[:,0],BOUNDS[:,1]))
    protocol=json.loads((OUT/'protocol.json').read_text());protocol['methods']['fine']='Training only: 9x9 curvature grid +/-2/m and 64 local Sobol samples around best score, unchanged selection gates; no held-out feedback';write('protocol.json',protocol)
    for i,d in enumerate(designs):
        if d[3]>d[1] or d[4]>d[2]:continue
        try:limits(d)
        except ValueError:continue
        er,st,_=solve(t['targets'],d);m=metrics(t,er,st,d)
        eligible=all(m[g]['success']>=base[g]['success']-.02 and m[g]['p95_mm']<=base[g]['p95_mm']+1 for g in base)
        score=sum(m[g]['success'] for g in base)+.25*sum(m[g]['effective_isotropy_mean']/max(base[g]['effective_isotropy_mean'],1e-6) for g in ('weak','corridor'))
        out.append(dict(design=d,metrics=m,eligible=eligible,score=score))
        if i%20==0:print('fine',i,'best',max((r for r in out if r['eligible']),key=lambda r:r['score']),flush=True)
    selected=max((r for r in out if r['eligible']),key=lambda r:r['score']);selected['design']=np.round(selected['design'],2)
    er,st,_=solve(t['targets'],selected['design']);selected['rounded_training_metrics']=metrics(t,er,st,selected['design'])
    write('fine_records.json',out);write('selected_frozen.json',selected)
def validate():
    chosen=np.array(json.loads((OUT/'selected_frozen.json').read_text())['design']);ref=dict(np.load(OUT/'baseline_regions.npz'))
    t=targetbank(2048,36101,ref);np.savez_compressed(OUT/'validation_targets.npz',**t);results={}
    for name,d in [('original',BASE),('optimised',chosen),('lengths_only',np.r_[chosen[:3],BASE[3:]]),('curves_only',np.r_[BASE[:3],chosen[3:]])]:
        er,st,_=solve(t['targets'],d);m=metrics(t,er,st,d);de,ds=diagnostic(t['targets'],d)
        sp=spatial(d,65536,36102);w=2*(ref['cells']//1000)+1
        m['diagnostic']=metrics(t,de,ds,d)
        m['workspace']=dict(sample_count=len(sp['points']),occupied_annular_volume_cm3=float(sum(2*(sp['cells']//1000)+1)*np.pi*5**3/1000),baseline_cell_retention=float(np.average(np.isin(ref['cells'],sp['cells']),weights=w)),max_radius_mm=float(np.linalg.norm(sp['points'][:,:2],axis=1).max()),z_min_mm=float(sp['points'][:,2].min()),z_max_mm=float(sp['points'][:,2].max()))
        np.savez_compressed(OUT/(name+'_validation.npz'),errors=er,states=st,diagnostic_errors=de,diagnostic_states=ds,**{'workspace_'+k:v for k,v in sp.items()})
        results[name]=m;write('validation_results.json',results);print('validated',name,m,flush=True)
    sensitivity={}
    for name,d in [('original',BASE),('optimised',chosen)]:
        er,st,_=solve(t['targets'],d,gap=108.5);sensitivity[name]=metrics(t,er,st,d)
    write('gap_sensitivity.json',sensitivity)
    # Independent pair of sampling resolutions; no candidate reselection.
    convergence={}
    for name,d in [('original',BASE),('optimised',chosen)]:
        sp=spatial(d,131072,36103);convergence[name]={}
        for h in (2.5,5.):
            ids=np.unique(cells(sp['points'],h));convergence[name][str(h)]=dict(n=131072,volume_cm3=float(sum(2*(ids//1000)+1)*np.pi*h**3/1000))
        np.savez_compressed(OUT/(name+'_dense_workspace.npz'),**sp)
    write('sampling_sensitivity.json',convergence)
    write('source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'measured_hardware_simulator/design_constraints.py',ROOT/'tools/run_broad_hardware_study.py',ROOT/'tools/run_hardware_optimisation.py',ROOT/'measured_hardware_simulator/CTR_superPosKin_fun_sectioned.py']})
def verify_selected():
    from ctr_inverse_kinematics import ConstrainedTipIK
    t=dict(np.load(OUT/'validation_targets.npz'));d=np.array(json.loads((OUT/'selected_frozen.json').read_text())['design']);rng=np.random.default_rng(36601);result={}
    for group,name in enumerate(('global','weak','corridor')):
        m=t['groups']==group;w=t['weights'][m]
        a=np.load(OUT/'original_validation.npz')['errors'][m];b=np.load(OUT/'optimised_validation.npz')['errors'][m]
        delta=(b<=.5).astype(float)-(a<=.5);ix=rng.integers(0,len(w),(2000,len(w)))
        boot=np.sum(delta[ix]*w[ix],axis=1)/np.sum(w[ix],axis=1)
        result[name]=dict(paired_success_gain_pp=100*np.average(delta,weights=w),bootstrap_95_percent_interval_pp=100*np.quantile(boot,[.025,.975]))
    write('paired_sampling_uncertainty.json',result);checks={}
    for name,design in [('original',BASE),('optimised',d)]:
        er,_,_=solve(t['targets'][:12],design)
        sol=ConstrainedTipIK(pars(design),deployment_limits=limits(design),max_iterations=40,damping_mm=1,solver_tolerance_mm=.01,max_normalised_step=.1)
        differences=[]
        for target,e in zip(t['targets'][:12],er):
            answer=sol.solve(target,limits(design).decode(np.zeros(3)),np.zeros(3))
            exact=np.linalg.norm(sol.forward_tip_mm(answer.deployment_m,answer.rotation_rad)-target);differences.append(abs(exact-e))
        s=samples(design,100,36701);jd=np.max(abs(iso(s,design)-iso(s,design,.005)))
        checks[name]=dict(batch_vs_viewer_max_residual_difference_mm=max(differences),jacobian_step_halving_max_isotropy_change=jd)
        assert max(differences)<1e-5 and jd<1e-4
    write('solver_and_jacobian_verification.json',checks)
    # Retain the actual numerical sources and library versions alongside results.
    import shutil,scipy
    paths=[Path(__file__),ROOT/'tools/run_broad_hardware_study.py',ROOT/'tools/run_hardware_optimisation.py',ROOT/'ctr_design_analysis.py']
    paths += [ROOT/'measured_hardware_simulator'/n for n in ['design_constraints.py','CTR_superPosKin_fun_sectioned.py','tube_parameters.py','ctr_inverse_kinematics.py','ctr_operating_profile.py']]
    for p in paths:
        target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write('source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    write('runtime.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__))
    print('Selected-design solver, Jacobian and paired uncertainty checks saved',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
