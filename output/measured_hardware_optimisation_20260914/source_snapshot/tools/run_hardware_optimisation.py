"""Local, approximate-hardware CTR study; preserves production model and protocols.
Search is restricted to the fully exposed-curvature regime and verified against FK.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctr-hardware-mpl')
import sys, json, time, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
from ctr_design_analysis import baseline_design
from ctr_inverse_kinematics import ConstrainedTipIK

OUT=Path('results/hardware_optimisation_20260908')
BASE=np.array([19.12,14.04])
BOUNDS=np.array([[17.208,21.032],[12.636,15.444]])
SCALE=np.array([350.,170.,80.,np.pi,np.pi,np.pi])
D=baseline_design()
EI=D.youngs_modulus_gpa*(D.od_mm**4-D.id_mm**4)


def limits(offset=35.):
    return np.array([3.+offset,0.,0.]), np.array([100.+offset,12.+offset,-7.5+offset])


def decode(x,offset=35.):
    x=np.atleast_2d(x); lo,hi=limits(offset)
    a=lo[0]+(hi[0]-lo[0])*x[:,0]
    b=np.minimum(a,hi[1])*x[:,1]
    c=np.minimum(b,hi[2])*x[:,2]
    return np.column_stack([a,b,c,np.zeros(len(x)),x[:,3],x[:,4]])


def bank(n,seed,offset=35.):
    x=qmc.Sobol(5,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
    x[:,3:]=(x[:,3:]*2-1)*np.pi
    return x,decode(x,offset)


def fk(states,k):
    s=np.atleast_2d(states); n=len(s)
    # Millimetre units; tube curvature vector convention matches source model.
    vec=np.stack([np.cos(s[:,3:]),np.sin(s[:,3:])],axis=2)*np.array([0.,*k])[None,:,None]/1000
    p=np.zeros((n,3)); R=np.broadcast_to(np.eye(3),(n,3,3)).copy()
    for ids,L in [([0,1,2],s[:,2]),([0,1],s[:,1]-s[:,2]),([0],s[:,0]-s[:,1])]:
        v=(vec[:,ids]*EI[None,ids,None]).sum(1)/EI[ids].sum()
        kap=np.linalg.norm(v,axis=1); phi=np.arctan2(v[:,0],v[:,1]); t=kap*L
        cp,sp=np.cos(phi),np.sin(phi); ct,st=np.cos(t),np.sin(t)
        h=.5*L*t*np.sinc(t/(2*np.pi))**2
        local=np.column_stack([cp*h,sp*h,L*np.sinc(t/np.pi)])
        p+=np.einsum('nij,nj->ni',R,local)
        Q=np.empty_like(R)
        Q[:,0,0]=1+cp*cp*(ct-1);Q[:,0,1]=cp*sp*(ct-1);Q[:,0,2]=cp*st
        Q[:,1,0]=Q[:,0,1];Q[:,1,1]=1+sp*sp*(ct-1);Q[:,1,2]=sp*st
        Q[:,2,0]=-cp*st;Q[:,2,1]=-sp*st;Q[:,2,2]=ct
        R=R@Q
    return p


def jac(states,k,step=.01):
    s=np.atleast_2d(states); J=np.empty((len(s),3,6))
    # Interior differences shrink near tube-order boundaries.
    margins=np.column_stack([s[:,0]-s[:,1],np.minimum(s[:,1]-s[:,2],s[:,0]-s[:,1]),np.minimum(s[:,2],s[:,1]-s[:,2])])
    for j in range(6):
        h=np.maximum(1e-7,np.minimum(step,.25*margins[:,j])) if j<3 else np.full(len(s),step/100)
        a=s.copy();b=s.copy();a[:,j]+=h;b[:,j]-=h
        J[:,:,j]=(fk(a,k)-fk(b,k))/(2*h[:,None])*SCALE[j]
    return J


def values(states,k,step=.01):
    sv=np.linalg.svd(jac(states,k,step),compute_uv=False)
    return sv[:,-1]/np.maximum(sv[:,0],1e-12),sv[:,-1]


def rz(p):return np.column_stack([np.linalg.norm(p[:,:2],axis=1),p[:,2]])

def cells(p,h=2.5,shift=0.):
    a=np.floor((rz(p)-shift)/h).astype(int)
    return a[:,0]*1000+a[:,1]


def region_cells(h=2.5):
    rr,zz=np.meshgrid(np.arange(int(10/h)),np.arange(int(40/h),int(130/h)),indexing='ij')
    ids=(rr*1000+zz).ravel(); w=((rr+1)**2-rr**2).ravel().astype(float)
    return ids,w


def evaluate(k,states,reference,h=2.5):
    p=fk(states,k); ids=cells(p,h); near,w=region_cells(h)
    mask=(rz(p)[:,0]<10)&(p[:,2]>=40)&(p[:,2]<130)
    si,weak=values(states[mask],k); ni=ids[mask]
    means=[];ws=[];covered=[]
    for cell in near:
        m=ni==cell; enough=np.sum(m)>=5
        covered.append(np.any(m));means.append(np.median(si[m]) if enough else 0.);ws.append(np.median(weak[m]) if enough else 0.)
    shared=np.isin(reference[0],np.unique(ids))
    return dict(coverage=float(np.average(covered,weights=w)),isotropy=float(np.average(means,weights=w)),weak=float(np.average(ws,weights=w)),retention=float(np.average(shared,weights=reference[1])),supported=float(np.average([np.sum(ni==c)>=5 for c in near],weights=w)))


def reference(states,k=BASE,h=2.5):
    ids=np.unique(cells(fk(states,k),h)); radius=ids//1000
    return ids,(2*radius+1).astype(float)


def jsonout(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)))


def verify():
    x,s=bank(64,901)
    err=[]
    for k in [BASE,BOUNDS[:,0],BOUNDS[:,1]]:
        d=baseline_design();d.precurvature_per_m[1:]=k
        solver=ConstrainedTipIK(d.to_parameters())
        expected=np.array([solver.forward_tip_mm(a[:3]/1000,a[3:]) for a in s])
        err.append(float(np.max(np.abs(expected-fk(s,k)))))
    straight=fk(s,[0,0]); assert np.max(np.abs(straight-np.column_stack([np.zeros((len(s),2)),s[:,0]])))<1e-10
    r=s.copy();r[:,3:]+=.7
    p=fk(s,BASE);co,ss=np.cos(.7),np.sin(.7)
    # Source phi convention reverses common input rotation in Cartesian space.
    expected=p@np.array([[co,-ss,0],[ss,co,0],[0,0,1]])
    symmetry=float(np.max(np.abs(fk(r,BASE)-expected)))
    assert max(err)<1e-8 and symmetry<1e-8
    delta=float(np.max(np.abs(jac(s,BASE,.01)-jac(s,BASE,.005))))
    d=baseline_design();d.curved_length_mm[1:]*=.9;sol=ConstrainedTipIK(d.to_parameters())
    curve_err=max(np.max(np.abs(sol.forward_tip_mm(a[:3]/1000,a[3:])-fk(a,BASE))) for a in s[:16])
    result=dict(fk_max_abs_error_mm=err,symmetry_error_mm=symmetry,jacobian_step_max_abs_change=delta,curved_length_minus10percent_max_difference_mm=float(curve_err))
    jsonout('verification.json',result);print(result,flush=True)


def search():
    start=time.time();verify()
    _,s=bank(16384,8101);ref=reference(s)
    b=evaluate(BASE,s,ref);print('baseline',b,flush=True)
    records=[];cache={}
    def assess(k,method):
        key=tuple(np.round(k,8))
        if key not in cache:
            v=evaluate(k,s,ref);cache[key]=v;records.append(dict(k=list(k),method=method,**v))
        return cache[key]
    assess(BASE,'baseline')
    # Full 11 by 11 grid makes the bounded two-variable landscape inspectable.
    for a in np.linspace(*BOUNDS[0],11):
        for b1 in np.linspace(*BOUNDS[1],11):assess([a,b1],'grid')
    print('grid done',len(records),round(time.time()-start,1),flush=True)
    # Equal family of criteria explores competing improvements without a clinical weighting claim.
    for mix in [0.,.5,1.]:
        def obj(k):
            v=assess(k,'differential_evolution')
            if v['retention']<.95:return 100+100*(.95-v['retention'])
            return -(mix*v['isotropy']/max(b['isotropy'],1e-9)+(1-mix)*v['weak']/max(b['weak'],1e-9)+v['coverage'])
        for seed in [8201,8202]:
            result=differential_evolution(obj,BOUNDS,popsize=6,maxiter=8,seed=seed,polish=False,tol=.005)
            print('search',mix,seed,result.x,flush=True)
    jsonout('search_records.json',records)
    # Frozen shortlist based ONLY on training data. Include geometrically distinct score extremes.
    viable=[r for r in records if r['retention']>=.95]
    chosen=[dict(k=BASE.tolist(),label='baseline')]
    criteria=[('isotropy',lambda r:r['isotropy']),('weak_direction',lambda r:r['weak']),('balanced',lambda r:r['isotropy']/b['isotropy']+r['weak']/b['weak'])]
    for label,score in criteria:
        for r in sorted(viable,key=score,reverse=True):
            if min(np.linalg.norm((np.array(r['k'])-c['k'])/BASE) for c in chosen)>.02:
                chosen.append(dict(k=r['k'],label=label,training=r));break
    jsonout('shortlist_frozen.json',chosen)
    jsonout('search_summary.json',dict(seconds=time.time()-start,evaluations=len(records),baseline=b,region=dict(radius_mm=10,z_mm=[40,130]),bounds=BOUNDS,seeds=[8101,8201,8202],minimum_retention=.95,minimum_samples_per_cell=5,grip_offset_mm=35,curved_lengths_fixed_mm=[0,90,65],total_grip_to_tip_lengths_mm=[350,170,80],status='local ideal-model search; no manufacturing certification'))
    print('SHORTLIST',chosen,flush=True)


def solve_batch(targets,k,seed=9101,offset=35.,iterations=70,nstarts=4,initial=None):
    # Numerically bounded multistart IK; all sampled and solved states share physical limits.
    if initial is None:
        x,s=bank(32768,seed,offset);p=fk(s,k);tree=cKDTree(p)
        _,inds=tree.query(targets,k=nstarts)
        v=x[np.asarray(inds).ravel()].copy()
    else:
        nstarts=1;v=np.asarray(initial).copy()
    t=np.repeat(targets,nstarts,axis=0)
    best=v.copy();be=np.linalg.norm(fk(decode(v,offset),k)-t,axis=1)
    for it in range(iterations):
        active=be>.002
        if not np.any(active):break
        ids=np.where(active)[0];w=v[ids];tar=t[ids];pos=fk(decode(w,offset),k);res=tar-pos
        J=np.empty((len(w),3,5))
        for j in range(5):
            plus=w.copy();minus=w.copy();h=1e-5
            plus[:,j]+=h;minus[:,j]-=h
            if j<3:plus[:,j]=np.clip(plus[:,j],0,1);minus[:,j]=np.clip(minus[:,j],0,1)
            J[:,:,j]=(fk(decode(plus,offset),k)-fk(decode(minus,offset),k))/(plus[:,j]-minus[:,j])[:,None]
        step=np.einsum('nji,nj->ni',J,np.linalg.solve(J@J.transpose(0,2,1)+.04*np.eye(3),res[...,None])[...,0])
        scale=np.maximum(1,np.max(np.abs(step)/np.array([.15,.15,.15,.4,.4]),axis=1));step/=scale[:,None]
        for alpha in [1.,.3,.1]:
            candidate=w+alpha*step;candidate[:,:3]=np.clip(candidate[:,:3],0,1);candidate[:,3:]=(candidate[:,3:]+np.pi)%(2*np.pi)-np.pi
            er=np.linalg.norm(fk(decode(candidate,offset),k)-tar,axis=1);better=er<be[ids]
            v[ids[better]]=candidate[better];best[ids[better]]=candidate[better];be[ids[better]]=er[better]
    errors=be.reshape(-1,nstarts);j=np.argmin(errors,axis=1);selected=best.reshape(-1,nstarts,5)[np.arange(len(targets)),j]
    return errors[np.arange(len(targets)),j],decode(selected,offset),selected


def validate():
    chosen=json.loads((OUT/'shortlist_frozen.json').read_text());start=time.time()
    # New spatially uniform cylinder targets, and a separate exact centreline.
    u=qmc.Sobol(3,scramble=True,seed=9301).random_base2(9)
    r=10*np.sqrt(u[:,0]);a=2*np.pi*u[:,1]
    targets=np.column_stack([r*np.cos(a),r*np.sin(a),40+90*u[:,2]])
    axial=np.column_stack([np.zeros((91,2)),np.linspace(40,130,91)])
    allresults=[]
    for c in chosen:
        k=np.array(c['k']);record=dict(label=c['label'],k=k.tolist(),spatial=[])
        for n in [32768,131072]:
            _,s=bank(n,9201);ref=reference(s)
            for h in [2.5,5.]:record['spatial'].append(dict(n=n,cell_mm=h,**evaluate(k,s,reference(s,h=h),h)))
        err,states,_=solve_batch(targets,k)
        ae,astates,ax=solve_batch(axial,k)
        # Lower capability at physical bounds: mark separately, not a symmetric-Jacobian guarantee.
        lo,hi=limits();m=np.min(np.column_stack([states[:,:3]-lo,hi-states[:,:3],states[:,0]-states[:,1],states[:,1]-states[:,2]]),axis=1)
        iso,weak=values(states,k);success=err<=.5
        record['ik']=dict(target_count=len(err),success_fraction=float(success.mean()),success_0_1mm=float(np.mean(err<=.1)),success_1mm=float(np.mean(err<=1)),p95_mm=float(np.percentile(err,95)),max_mm=float(err.max()),median_isotropy_interior=float(np.median(iso[success&(m>.01)])),median_weak_interior=float(np.median(weak[success&(m>.01)])),interior_fraction=float(np.mean(success&(m>.01))),axial_success=float(np.mean(ae<=.5)),axial_max_error_mm=float(ae.max()))
        record['offset_sensitivity']=[]
        for offset in [30.,40.]:
            er,_,_=solve_batch(targets[:128],k,offset=offset)
            record['offset_sensitivity'].append(dict(offset_mm=offset,success_fraction=float(np.mean(er<=.5)),max_mm=float(er.max())))
        np.savez_compressed(OUT/(c['label']+'_validation.npz'),targets=targets,errors=err,states=states,axis_targets=axial,axis_errors=ae,axis_states=astates,axis_latent=ax,isotropy=iso,weak=weak,interior_margin=m)
        allresults.append(record);jsonout('validation.json',allresults)
        print('VALIDATED',c['label'],record['ik'],round(time.time()-start,1),flush=True)
    jsonout('validation_summary.json',dict(seconds=time.time()-start,validation_seed=9301,configuration_seed=9201,ik_bank_seed=9101,region='r <=10 mm, z 40..130 mm',offset_range_role='assumed sensitivity, not measured tolerance'))

def benchmark():
    records=json.loads((OUT/'search_records.json').read_text())
    _,s=bank(16384,8101);ref=reference(s)
    rng=np.random.default_rng(8401);items=[]
    for k in rng.uniform(BOUNDS[:,0],BOUNDS[:,1],size=(len(records),2)):
        items.append(dict(k=k.tolist(),**evaluate(k,s,ref)))
    best=max((v for v in items if v['retention']>=.95),key=lambda v:v['isotropy'])
    jsonout('random_benchmark.json',dict(evaluations=len(items),seed=8401,best=best,role='equal unique-evaluation budget; validation benchmark only, no shortlist changes',records=items))


if __name__=='__main__':
    {'verify':verify,'search':search,'validate':validate,'benchmark':benchmark}[sys.argv[1]]()
