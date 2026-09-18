"""Four-variable, three-objective exploratory hardware-constrained CTR study.
All conclusions are conditional on the ideal exterior model and midpoint mounting.
Existing local study and production parameters remain untouched.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctr-hardware-mpl')
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import qmc,spearmanr
import run_hardware_optimisation as h
from ctr_design_analysis import baseline_design,position_jacobians_all_endpoints
from ctr_inverse_kinematics import ConstrainedTipIK

OUT=Path('results/broad_hardware_optimisation_20260908')
REPORT=Path('output/broad_hardware_optimisation')
BASE=np.array([19.12,14.04,90.,65.])
A=np.array([21.032,15.444,90.,65.])
# Curved lengths include shorter active sections; upper endpoints represent the
# exterior-equivalent families Lc_middle>=47, Lc_outer>=27.5 at nominal mounting.
BOUNDS=np.array([[9.56,28.68],[7.02,21.06],[10.,47.],[5.,27.5]])
CONFIG=dict(study='broad-hardware-v1',grip_offset_mm=35,grip_to_tip_lengths_mm=[350,170,80],exposure_limits_mm=[[38,135],[0,47],[0,27.5]],curvature_unit='1/m',bounds=BOUNDS.tolist(),bounds_basis='exploratory +/-50% curvature and shorter curved sections; not supplier-certified',near_region=dict(radius_mm=10,z_mm=[40,130]),volume_region=dict(radius_mm=100,z_mm=[0,135]),objectives=['sampled_volume_in_fixed_region','mean_cell_median_isotropy_on_frozen_baseline_region','near_axis_mean_cell_median_isotropy'],constraints=dict(baseline_region_retention=.95,near_axis_sampled_coverage=.99,near_axis_isotropy_relative_to_baseline=.95),sampling='shared scrambled Sobol physical-actuator samples; independent search/validation seeds',normalisation=[350,170,80,float(np.pi),float(np.pi),float(np.pi)],minimum_cell_support=5,validation_must_not_retune=True,orientation_and_physical_stability='not evaluated',curve_family_caveat='lengths above nominal exposure caps are exterior-equivalent, but not mechanically interchangeable')


def write(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v)))


def forward(states,design):
    s=np.atleast_2d(states);d=np.asarray(design)
    k=d[:2];lc=d[2:] if len(d)>2 else np.array([90.,65.]);n=len(s)
    starts=np.maximum(0,s[:,1:3]-lc)
    edges=np.sort(np.column_stack([np.zeros(n),s[:,:3],starts]),axis=1)
    vec=np.stack([np.cos(s[:,3:]),np.sin(s[:,3:])],axis=2)*np.array([0.,*k])[None,:,None]/1000
    p=np.zeros((n,3));R=np.broadcast_to(np.eye(3),(n,3,3)).copy()
    for j in range(5):
        left,right=edges[:,j],edges[:,j+1];L=right-left;mid=(left+right)/2
        active=mid[:,None]<s[:,:3]
        curved=np.column_stack([np.zeros(n,dtype=bool),mid[:,None]>=starts])
        denom=np.maximum(np.sum(active*h.EI,axis=1),1e-30)
        v=np.sum(vec*(active*curved*h.EI)[...,None],axis=1)/denom[:,None]
        kap=np.linalg.norm(v,axis=1);phi=np.arctan2(v[:,0],v[:,1]);t=kap*L
        cp,sp=np.cos(phi),np.sin(phi);ct,st=np.cos(t),np.sin(t)
        bend=.5*L*t*np.sinc(t/(2*np.pi))**2
        local=np.column_stack([cp*bend,sp*bend,L*np.sinc(t/np.pi)])
        p+=np.einsum('nij,nj->ni',R,local)
        Q=np.empty_like(R)
        Q[:,0,0]=1+cp*cp*(ct-1);Q[:,0,1]=cp*sp*(ct-1);Q[:,0,2]=cp*st
        Q[:,1,0]=Q[:,0,1];Q[:,1,1]=1+sp*sp*(ct-1);Q[:,1,2]=sp*st
        Q[:,2,0]=-cp*st;Q[:,2,1]=-sp*st;Q[:,2,2]=ct
        R=R@Q
    return p

# Explicit model adapter for the already-tested bounded solver and physical
# Jacobian utilities. This is confined to this process, not a source-file edit.
h.fk=forward


def design_object(d):
    b=baseline_design();b.precurvature_per_m[1:]=d[:2];b.curved_length_mm[1:]=d[2:]
    return b


def verify():
    _,s=h.bank(40,11001);rng=np.random.default_rng(11002)
    ds=[BASE,A,*rng.uniform(BOUNDS[:,0],BOUNDS[:,1],(6,4))]
    maximum=0.;jd=0.
    for d in ds:
        sol=ConstrainedTipIK(design_object(d).to_parameters())
        expected=np.array([sol.forward_tip_mm(v[:3]/1000,v[3:]) for v in s])
        maximum=max(maximum,float(np.max(np.abs(expected-forward(s,d)))))
        for v in s[:4]:
            J=position_jacobians_all_endpoints(sol,v[:3]/1000,v[3:],translation_step_mm=.01,rotation_step_rad=.0001).scaled_jacobians[0]
            jd=max(jd,float(np.max(np.abs(h.jac(v,d)[0]-J))))
    # Curved-to-straight transition continuity and all-straight analytic case.
    state=np.array([100.,30.,15.,0.,.4,-.8]);d=np.array([24.,18.,30.,15.])
    t=state.copy();t[1]+=1e-7
    continuity=float(np.linalg.norm(forward(t,d)-forward(state,d)))
    zero=forward(s,[0,0,20,10]);assert np.max(np.abs(zero-np.column_stack([np.zeros((len(s),2)),s[:,0]])))<1e-9
    assert maximum<1e-8 and jd<2e-5 and continuity<1e-5
    write('verification.json',dict(states=320,max_fk_discrepancy_mm=maximum,max_jacobian_discrepancy=jd,transition_perturbation_mm=1e-7,transition_endpoint_change_mm=continuity))
    print('Verified general model',maximum,jd,flush=True)


def spatial_data(design,states,step=2.5):
    p=forward(states,design);rz=h.rz(p);ids=h.cells(p,step)
    iso,weak=h.values(states,design)
    order=np.argsort(ids);sorted_ids=ids[order];unique,begin,count=np.unique(sorted_ids,return_index=True,return_counts=True)
    median=np.array([np.median(iso[order[b:b+c]]) if c>=5 else 0 for b,c in zip(begin,count)])
    weakest=np.array([np.median(weak[order[b:b+c]]) if c>=5 else 0 for b,c in zip(begin,count)])
    volmask=(rz[:,0]<100)&(p[:,2]>=0)&(p[:,2]<135)
    volids=np.unique(ids[volmask]);volume=float(np.sum(2*(volids//1000)+1)*np.pi*step**3/1000)
    # Explicitly restrict observations to the fixed central task region.
    mask=(rz[:,0]<10)&(p[:,2]>=40)&(p[:,2]<130)
    near,w=h.region_cells(step);ni=ids[mask];iv=iso[mask]
    nm=np.array([np.median(iv[ni==c]) if np.sum(ni==c)>=5 else 0 for c in near])
    return dict(ids=unique,medians=median,weak=weakest,counts=count,volume=volume,near=float(np.average(nm,weights=w)),near_coverage=float(np.average(np.isin(near,ni),weights=w)))


def evaluate(design,states,ref,step=2.5):
    z=spatial_data(design,states,step);ids=ref['ids'];weights=(2*(ids//1000)+1).astype(float)
    ind=np.searchsorted(z['ids'],ids);exists=(ind<len(z['ids']))
    ind=np.minimum(ind,len(z['ids'])-1);exists&=z['ids'][ind]==ids
    med=np.where(exists,z['medians'][ind],0.);weak=np.where(exists,z['weak'][ind],0.)
    mean=float(np.average(med,weights=weights));ret=float(np.average(exists,weights=weights))
    # Weighted lower spatial decile over the common region.
    order=np.argsort(med);cum=np.cumsum(weights[order])/np.sum(weights);lower=float(med[order[np.searchsorted(cum,.1)]])
    violation=max(0,.95-ret)+max(0,.99-z['near_coverage'])+max(0,.95-z['near']/max(ref['near'],1e-9))
    return dict(volume_cm3=z['volume'],overall_isotropy=mean,near_isotropy=z['near'],overall_p10_isotropy=lower,overall_weak=float(np.average(weak,weights=weights)),retention=ret,near_coverage=z['near_coverage'],violation=violation,valid=violation<1e-12)


OBJECTIVES=['volume_cm3','overall_isotropy','near_isotropy']

def fronts(rows):
    remaining=list(range(len(rows)));out=[]
    while remaining:
        front=[]
        for i in remaining:
            ri=rows[i];vi=ri['violation'];a=np.array([ri[k] for k in OBJECTIVES])
            dominated=False
            for j in remaining:
                if j==i:continue
                rj=rows[j];vj=rj['violation'];b=np.array([rj[k] for k in OBJECTIVES])
                if (vj<vi-1e-12) or (abs(vj-vi)<=1e-12 and np.all(b>=a) and np.any(b>a)):
                    dominated=True;break
            if not dominated:front.append(i)
        out.append(front);remaining=[i for i in remaining if i not in front]
    return out


def select(rows,size):
    selected=[]
    for f in fronts(rows):
        if len(selected)+len(f)<=size:selected+=f;continue
        dist=np.zeros(len(f))
        for key in OBJECTIVES:
            values=np.array([rows[i][key] for i in f]);order=np.argsort(values);dist[order[[0,-1]]]=np.inf
            if values.max()>values.min():dist[order[1:-1]]+=(values[order[2:]]-values[order[:-2]])/(values.max()-values.min())
        selected+=[f[i] for i in np.argsort(-dist)[:size-len(selected)]];break
    return [rows[i] for i in selected]


def search():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'shortlist_frozen.json').exists():raise RuntimeError('Refusing to overwrite frozen search')
    write('protocol.json',CONFIG);verify();start=time.time()
    pilot=qmc.LatinHypercube(4,seed=11101).random(16)*(BOUNDS[:,1]-BOUNDS[:,0])+BOUNDS[:,0]
    pilot=np.vstack([BASE,A,pilot]);pilot_results={}
    for n in [8192,32768]:
        _,s=h.bank(n,11201);ref=spatial_data(BASE,s)
        pilot_results[n]=[evaluate(d,s,ref) for d in pilot]
    correlations={k:float(spearmanr([r[k] for r in pilot_results[8192]],[r[k] for r in pilot_results[32768]]).statistic) for k in OBJECTIVES}
    n=8192 if min(correlations.values())>=.85 else 32768
    write('pilot.json',dict(designs=pilot,results=pilot_results,rank_correlations=correlations,chosen_n=n,threshold=.85))
    print('Pilot',correlations,'selected N',n,flush=True)
    _,s=h.bank(n,11201);ref=spatial_data(BASE,s);records=[];cache={}
    def assess(d,method):
        key=tuple(np.round(d,7))
        if key not in cache:
            r=dict(design=list(d),method=method,**evaluate(d,s,ref));cache[key]=r;records.append(r)
        return cache[key]
    assess(BASE,'baseline');assess(A,'candidate_A')
    lhs=qmc.LatinHypercube(4,seed=11301).random(128)*(BOUNDS[:,1]-BOUNDS[:,0])+BOUNDS[:,0]
    for i,d in enumerate(lhs):
        assess(d,'space_filling')
        if i%32==31:print('Initial designs',i+1,round(time.time()-start,1),flush=True);write('search_checkpoint.json',records)
    # Boundary controls: preserve long-curve exterior-equivalent family explicitly.
    for m in [1.,1.1,1.25,1.5]:assess([19.12*m,14.04*m,47,27.5],'long_curve_control')
    for seed in [11401,11402]:
        rng=np.random.default_rng(seed);population=select(records,32)
        for generation in range(6):
            children=[]
            for i in range(32):
                p1,p2=rng.choice(len(population),2,replace=False)
                a=np.array(population[p1]['design']);b=np.array(population[p2]['design'])
                a=np.clip(a,BOUNDS[:,0],BOUNDS[:,1]);b=np.clip(b,BOUNDS[:,0],BOUNDS[:,1])
                blend=rng.uniform(-.2,1.2,4);d=blend*a+(1-blend)*b
                mutate=rng.random(4)<.4;d+=mutate*rng.normal(0,.08,4)*(BOUNDS[:,1]-BOUNDS[:,0])
                d=np.clip(d,BOUNDS[:,0],BOUNDS[:,1]);children.append(assess(d,f'pareto_search_{seed}'))
            population=select(population+children,32)
            write('search_checkpoint.json',records);print('Generation',seed,generation+1,'unique',len(records),round(time.time()-start,1),flush=True)
    write('search_records.json',records)
    feasible=[r for r in records if r['valid']];pf=[feasible[i] for i in fronts(feasible)[0]]
    # Medium-fidelity training reevaluation before any final validation is opened.
    candidates=select(pf,min(16,len(pf)))
    _,ss=h.bank(65536,11201);rr=spatial_data(BASE,ss)
    medium=[dict(design=r['design'],**evaluate(r['design'],ss,rr)) for r in candidates]
    medium=[r for r in medium if r['valid']]
    if not medium:raise RuntimeError('No candidate passes training refinement')
    pf2=[medium[i] for i in fronts(medium)[0]]
    write('pareto_refinement.json',dict(low_front=pf,medium_candidates=medium,medium_front=pf2))
    chosen=[dict(label='baseline',design=BASE.tolist()),dict(label='candidate_A',design=A.tolist())]
    for key,label in zip(OBJECTIVES,['maximum_workspace','overall_dexterity','near_axis_dexterity']):
        r=max(pf2,key=lambda x:x[key])
        if not any(np.allclose(r['design'],v['design']) for v in chosen):chosen.append(dict(label=label,design=r['design'],training=r))
    # Balanced compromise maximises the weakest normalised objective across this front.
    mat=np.array([[r[k] for k in OBJECTIVES] for r in pf2]);norm=(mat-mat.min(0))/np.maximum(mat.max(0)-mat.min(0),1e-12)
    idx=int(np.argmax(np.min(norm,axis=1)));r=pf2[idx]
    if not any(np.allclose(r['design'],v['design']) for v in chosen):chosen.append(dict(label='balanced',design=r['design'],training=r))
    # A shorter-curve alternative is retained if not already represented, to test the new degree of freedom.
    short=[r for r in medium if r['design'][2]<40 or r['design'][3]<23]
    if short:
        r=max(short,key=lambda x:x['overall_isotropy'])
        if not any(np.allclose(r['design'],v['design']) for v in chosen):chosen.append(dict(label='short_curve_alternative',design=r['design'],training=r))
    write('shortlist_frozen.json',chosen);write('search_summary.json',dict(unique_designs=len(records),seconds=time.time()-start,training_configurations=n,refinement_configurations=65536,selected_count=len(chosen),search_method='space-filling initialisation, non-dominated sorting and crowding selection with bounded recombination/mutation',benchmark='initial 128-design space-filling population'))
    print('FROZEN',chosen,flush=True)


if __name__=='__main__':
    {'verify':verify,'search':search}[sys.argv[1]]()
