"""Isolated, budget-matched geometry-only study. Never installs a design.

Commands: prepare, checks, pilot, search METHOD SEED, select, validate, report.
All inspected historical data are development evidence. Final seeds are reserved.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctr-final-mpl')
import sys, json, time, hashlib, shutil, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/final_geometry_method_comparison_20260916'
# Append: retain the existing numpy/scipy/model environment, isolate new library.
sys.path.append(str(OUT/'dependencies'))
import optimise_measured_ctr as h
BASE=h.BASE.copy()
PROPOSED=np.array([350,177.5,87.5,90,65,21.37,14.04])
BOUNDS=h.BOUNDS.copy()
SEEDS=[71011,71012,71013]
REGION_NAMES=['global','weak','difficult','corridor']

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item() if isinstance(a,np.generic) else str(a)))
def read(path):return json.loads(Path(path).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path):return dict(np.load(path))
def volume(ids,step=5):return float(np.sum(2*(np.asarray(ids)//1000)+1)*np.pi*step**3/1000)
def key(d):
    d=np.array(d,float).copy();lim=h.limits(d)
    d[3:5]=np.minimum(d[3:5],np.array(lim.upper_m)[1:]*1000)
    return tuple(np.round(d,8))
def canonical(d):
    d=np.asarray(d,float).copy()
    if np.any(d<BOUNDS[:,0]-1e-9) or np.any(d>BOUNDS[:,1]+1e-9):raise ValueError('Design outside declared bounds')
    if d[3]>d[1] or d[4]>d[2]:raise ValueError('Negative straight material length')
    h.limits(d)
    return d
def regions():return {k:np.array(v,int) for k,v in read(OUT/'regions.json').items()}

def source_hashes():
    files=[ROOT/'tools'/n for n in ['final_geometry_comparison.py','optimise_measured_ctr.py','run_broad_hardware_study.py','run_hardware_optimisation.py']]
    files += [ROOT/'measured_hardware_simulator'/n for n in ['design_constraints.py','CTR_superPosKin_fun_sectioned.py','tube_parameters.py','ctr_inverse_kinematics.py']]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}

def prepare():
    if (OUT/'protocol.json').exists():
        assert read(OUT/'source_hashes.json')==source_hashes(),'Code changed after protocol freeze'
        return
    OUT.mkdir(parents=True,exist_ok=True)
    hist=ROOT/'output/chapter5_measured_revision/data'
    sp=load(hist/'original_spatial.npz');ref=sp['cells']
    old=read(hist/'regions.json');cal=load(hist/'calibration_targets.npz');e=np.load(hist/'calibration_original.npz')['errors']
    ids,counts=np.unique(cal['cell_ids'],return_counts=True)
    failure=np.array([np.mean(e[cal['cell_ids']==c]>.5) for c in ids])
    valid=counts>=100;ids=ids[valid];failure=failure[valid]
    order=np.lexsort((ids,-failure));weights=2*(ids//1000)+1
    n=np.searchsorted(np.cumsum(weights[order]),.2*np.sum(weights))+1
    reg=dict(global_=ref,weak=np.intersect1d(ref,old['weak']),difficult=ids[order[:n]],corridor=np.array(old['corridor']),supported=sp['cells'][sp['counts']>=30])
    reg['global']=reg.pop('global_');write(OUT/'regions.json',reg)
    write(OUT/'region_basis.json',dict(difficult='Worst 20% annular-volume mass of original calibration cells with >=100 targets; ties by cell ID',calibration_sha=digest(hist/'calibration_original.npz'),reference_sha=digest(hist/'original_spatial.npz'),region_overlap='Separate regional readouts; no duplicated weights in global objective'))
    protocol=dict(id='final-geometry-method-comparison-v1',status='Frozen before candidate search',original=BASE,proposed=PROPOSED,bounds=BOUNDS,
        variables=['inner_total_mm','middle_total_mm','outer_total_mm','middle_curved_mm','outer_curved_mm','middle_kappa_per_m','outer_kappa_per_m'],
        objectives=['minimise global fixed-start IK failure fraction','minimise trajectory failure fraction'],
        constraints=dict(chuck_distances_retracted_mm=[275,195,115],chuck_distances_extended_mm=[175,95,15],actuator_length_mm=71.5,gap_mm=[8.5,108],sensitivity_gap_mm=108.5,exterior_order='inner>=middle>=outer>=0'),
        sampling=dict(screen_states=8192,screen_targets=[1024,256,256,128],screen_paths=24,selection_states=65536,selection_targets=[4096,1024,1024,512],selection_paths=96,validation_states=262144,validation_targets=[8192,2048,2048,1024],validation_paths_per_bank=128,validation_banks=2),
        search=dict(methods=['nsga2','staged'],seeds=SEEDS,budget_per_run=256,nsga='32 population +7 offspring batches32; SBX p=.9 eta15; polynomial mutation per-variable1/7 eta20; feasibility-first rank/crowding; common Sobol initial32 including references',staged='Common initial32 +128 Sobol +32 curvature grids (4x4 around each reference) +64 local proposals around4 development leaders',dense_allowance='4 distinct candidates/run; references extra equal shared overhead',comparison='Same objectives/guards/fidelity, not an exact rerun of historical hand-weighted scoring',stop='256 evaluator slots after analytic feasibility filtering; rejected geometries replaced from a predeclared Sobol stream, logged separately; cache hits/model calls/time reported. Sampler exhaustion consumes its attempted evaluation. No outcome-driven budget extension'),
        evaluation=dict(fixed_start='Existing most-retracted feasible exterior decode([0,0,0]), zero rotations; unchanged40-iteration DLS',path_initialisation='Existing two-nearest-start diagnostic,60 iterations each; same for all geometries; no recovery',paths='Original feasible joint-linear shortest-angle paths; 21 waypoints,40 iterations/warm-start; 4 checks/segment screen,100 checks/segment final; dense maximum must<=0.5mm',global_targets='Annular-volume-proportional cell draws, random full-azimuth original-FK points; no hidden canonical-plane targets',rotation='Unrestricted middle/outer rotation; inner straight rotation has no effect',isotropy='Existing fixed range-scaled positional Jacobian; missing/low-support cell values zero; fixed original region denominators'),
        margins=dict(global_ik_gain=.03,regional_ik_loss=.02,completion_observed_loss=0,completion_noninferiority=.02,p95_increase_mm=.05,coverage_loss=.02,reach_loss_fraction=.02,isotropy_loss_fraction=.05),
        statistical_rule='IK gain>=3pp and paired95% interval lower>0; regional interval lower>=-2pp. Path observed completion>=baseline AND conservative paired-binomial difference lower>=-2pp. P95 path-block bootstrap upper increase<=.05mm. Coverage robust supported-reference loss<=2% in each equal-budget repeat and resolution sensitivity; if sampling calibration cannot resolve2%, coverage is inconclusive, not relaxed. Reach/isotropy point margins required in both repeats; sampling sensitivity reported. All gates conjunctions; no validation tuning.',
        classification='accepted only if selection and all final gates pass; failed or inconclusive otherwise. Two frozen method champions evaluated; Bonferroni intervals97.5% for their statistical gates. Three seeds descriptive, not universal algorithm superiority.',
        assumptions=['Lengths provisional chuck-tip-to-distal-tip material lengths; no35mm offset','Straight internal guidance/retention unverified','Curves beyond maximum exposure exterior-equivalent; material choices not interchangeable physically','No manufacture-certified bounds, torsion, friction, elastic stability, contact, physical measurement or clinical validation'],
        seeds=dict(screen=71100,selection=72100,validation=[73100,74100]),historical='All previously inspected results are development evidence; reserved seeds not opened until champions frozen')
    write(OUT/'protocol.json',protocol)
    write(OUT/'source_hashes.json',source_hashes())
    protected={str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('optimised_configuration.json') if OUT not in p.parents}
    for p in [ROOT/'output/task_prioritised_tradeoff_20260914/frozen_candidate.json',ROOT/'output/chapter5_measured_revision/data/summary.json']:protected[str(p.relative_to(ROOT))]=digest(p)
    write(OUT/'protected_hashes.json',protected)
    for name in source_hashes():
        target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    import pymoo
    write(OUT/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pymoo=pymoo.__version__,platform=platform.platform()))
    print('Protocol frozen',flush=True)

def bank(seed,counts,npaths):
    path=OUT/'banks'/f'bank_{seed}.npz'
    if path.exists():return load(path)
    reg=regions();ref=reg['global'];ss=[];pp=[];ii=[]
    for j in range(8):
        st=h.samples(BASE,262144,seed+10+j);p=h.fk(st,BASE);ids=h.cells(p);ss.append(st);pp.append(p);ii.append(ids)
        if not len(np.setdiff1d(ref,np.unique(np.concatenate(ii)))):break
    st=np.concatenate(ss);p=np.concatenate(pp);ids=np.concatenate(ii)
    missing=np.setdiff1d(ref,np.unique(ids))
    if len(missing):raise RuntimeError(f'Cannot silently omit original reference cells: {missing}')
    rng=np.random.default_rng(seed);ix=[];groups=[]
    order=np.argsort(ids);u,a,c=np.unique(ids[order],return_index=True,return_counts=True)
    for j,name in enumerate(REGION_NAMES):
        pool=reg[name];w=2*(pool//1000)+1;chosen=rng.choice(pool,counts[j],p=w/sum(w))
        for cell in chosen:
            k=np.searchsorted(u,cell);ix.append(order[a[k]+rng.integers(c[k])]);groups.append(j)
    ix=np.array(ix);nr=npaths//2;nw=npaths//4;nb=npaths-nr-nw
    ends=h.samples(BASE,2*nr,seed+30).reshape(nr,2,6)
    wi=np.flatnonzero(np.isin(ids,reg['weak']));weak=st[rng.choice(wi,2*nw)].reshape(nw,2,6)
    # Boundary stress paths: end lies exactly on a face of the unchanged feasible polytope.
    f=rng.random((nb,2,3))
    for k in range(nb):f[k,1,k%3]=k%2
    be=h.limits(BASE).decode(f.reshape(-1,3))*1000
    bs=np.column_stack([be,np.zeros(2*nb),rng.uniform(-np.pi,np.pi,(2*nb,2))]).reshape(nb,2,6)
    endpoints=np.concatenate([ends,weak,bs]);endpoints[:,1,3:]=endpoints[:,0,3:]+(endpoints[:,1,3:]-endpoints[:,0,3:]+np.pi)%(2*np.pi)-np.pi
    t=dict(targets=p[ix],source_states=st[ix],cell_ids=ids[ix],groups=np.array(groups),endpoints=endpoints,path_categories=np.r_[np.zeros(nr,int),np.ones(nw,int),np.full(nb,2,int)])
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**t)
    return t

def paths(d,t,subdivisions=4,gap=108):
    ep=t['endpoints'];a,b=ep[:,0],ep[:,1];ref=lambda u:h.fk(a+(b-a)*u,BASE)
    er,st=h.diagnostic(ref(0),d,gap);lim=h.limits(d,gap)
    x=np.column_stack([[lim.encode(e/1000) for e in st[:,:3]],st[:,4:]/np.pi])
    errors=[er];states=[st.copy()]
    for j in range(1,21):
        old=st.copy();er,st,x=h.solve(ref(j/20),d,x,40,gap);states.append(st.copy())
        # Batch FK at all intervening samples; controller itself is unchanged.
        f=np.arange(1,subdivisions+1)/subdivisions
        mid=old[:,None,:]+(st-old)[:,None,:]*f[None,:,None]
        lim.validate(mid[...,:3]/1000)
        r=a[:,None,:]+(b-a)[:,None,:]*((j-1+f)/20)[None,:,None]
        ee=np.linalg.norm(h.fk(mid.reshape(-1,6),d)-h.fk(r.reshape(-1,6),BASE),axis=1).reshape(len(a),-1)
        errors.extend(ee[:,k] for k in range(subdivisions))
    er=np.stack(errors,axis=1);mx=er.max(1);ok=mx<=.5
    metrics=dict(path_completion=float(ok.mean()),path_p95_mm=float(np.percentile(er,95)),path_max_mm=float(mx.max()),path_mean_max_mm=float(mx.mean()))
    for k,name in enumerate(['representative','weak','boundary']):metrics['path_'+name+'_completion']=float(ok[t['path_categories']==k].mean())
    return metrics,dict(path_errors=er,path_states=np.array(states),path_max_errors=mx)

def spatial(d,n,seed,gap=108):
    sp=h.spatial(d,n,seed,gap=gap);reg=regions();p=sp['points'];m={}
    for name in REGION_NAMES:
        ids=reg[name];w=2*(ids//1000)+1;v=h.map_values(sp,ids)
        m[name+'_isotropy_mean']=float(np.average(v,weights=w));m[name+'_isotropy_p10']=h.wquant(v,w,.1)
        m[name+'_retention']=float(np.average(np.isin(ids,sp['cells']),weights=w))
    ids=reg['supported'];m['supported_retention']=float(np.average(np.isin(ids,sp['cells']),weights=2*(ids//1000)+1))
    m.update(volume_cm3=volume(sp['cells']),radial_mm=float(np.linalg.norm(p[:,:2],axis=1).max()),zmax_mm=float(p[:,2].max()),zmin_mm=float(p[:,2].min()),reach_mm=float(np.linalg.norm(p,axis=1).max()))
    return m,sp

def evaluate(d,t,n,seed,subdivisions=4,gap=108):
    start=time.monotonic();d=canonical(d)
    er,st,_=h.solve(t['targets'],d,gap=gap);m={}
    for j,name in enumerate(REGION_NAMES):
        e=er[t['groups']==j];m[name+'_ik_success']=float(np.mean(e<=.5));m[name+'_ik_p95_mm']=float(np.percentile(e,95))
    tr,raw=paths(d,t,subdivisions,gap);m.update(tr)
    sm,sp=spatial(d,n,seed+1,gap);m.update(sm)
    raw.update(ik_errors=er,ik_states=st,workspace_points=sp['points'],workspace_cells=sp['cells'],workspace_counts=sp['counts'],workspace_medians=sp['medians'])
    return dict(design=d,metrics=m,seconds=time.monotonic()-start),raw

def deficits(m,b,selection=False):
    g={}
    for region in ['weak','difficult']:
        g[region+'_ik']=b[region+'_ik_success']-.02-m[region+'_ik_success']
    g['coverage']=b['supported_retention']-.02-m['supported_retention']
    for k in ['radial_mm','zmax_mm','reach_mm']:g[k]=(.98*b[k]-m[k])/max(b[k],1e-8)
    for reg in ['global','weak','difficult']:
        for suffix in ['isotropy_mean','isotropy_p10']:
            k=reg+'_'+suffix;g[k]=(.95*b[k]-m[k])/max(b[k],1e-8)
    if selection:
        g['meaningful_global_gain']=b['global_ik_success']+.03-m['global_ik_success']
        g['completion']=b['path_completion']-m['path_completion']
        g['tracking_p95']=(m['path_p95_mm']-b['path_p95_mm']-.05)/.5
    return {k:float(v) for k,v in g.items()}
def cv(m,b):return sum(max(0,v) for v in deficits(m,b).values())
def rank(r,b):
    g=deficits(r['metrics'],b,True);v=sum(max(0,x) for x in g.values());m=r['metrics']
    return (v>1e-10,v,-m['path_completion'],-m['global_ik_success'],m['path_mean_max_mm'],m['path_p95_mm'])

def checks():
    from CTR_superPosKin_fun_sectioned import superPosKin
    worst=0
    for d in [BASE,PROPOSED,np.array([355,205,110,55,35,23,16])]:
        st=h.samples(d,12,71002)
        for row,p in zip(st,h.fk(st,d)):
            expected=np.array(superPosKin(h.pars(d),{'ul':row[:3]/1000,'uphi':row[3:]},{'n_p':1,'isPlot':False})[0][:3])*1000
            worst=max(worst,float(abs(expected-p).max()))
        lim=h.limits(d);e=lim.decode(h.sobol(128,3,71003));lim.validate(e)
        dist=d[:3]-1000*e;g=dist[:,:-1]-dist[:,1:]-71.5
        assert np.all((g>=8.5-1e-8)&(g<=108+1e-8))
    assert worst<1e-7
    equiv=BASE.copy();equiv[3:5]=[100,70]
    assert key(BASE)==key(equiv)
    np.testing.assert_allclose(h.fk(h.samples(BASE,128,71004),BASE),h.fk(h.samples(BASE,128,71004),equiv),atol=1e-9)
    t=bank(71100,[1024,256,256,128],24)
    # Same old trajectory routine, same endpoint input: compare vectorised checks with explicit loop.
    m,raw=paths(BASE,t);ep=t['endpoints'];manual=[]
    for j in range(20):
        for f in np.arange(1,5)/4:
            st=raw['path_states'][j]+f*(raw['path_states'][j+1]-raw['path_states'][j])
            tar=h.fk(ep[:,0]+(ep[:,1]-ep[:,0])*(j+f)/20,BASE)
            manual.append(np.linalg.norm(h.fk(st,BASE)-tar,axis=1))
    np.testing.assert_allclose(np.array(manual).T,raw['path_errors'][:,1:],atol=1e-9)
    assert len(t['targets'])==1664
    h.limits(BASE).validate(t['source_states'][:,:3]/1000)
    np.testing.assert_allclose(h.fk(t['source_states'],BASE),t['targets'])
    write(OUT/'checks.json',dict(forward_max_discrepancy_mm=worst,coupled_limits=True,equivalence_cache=True,full_azimuth_targets=True,path_interpolation_matches_explicit_loop=True))
    print('Kernel/constraint/target/path checks passed',flush=True)

def pilot():
    t=bank(71100,[1024,256,256,128],24);rows=[]
    rng=np.random.default_rng(71001);ds=[BASE,PROPOSED]
    while len(ds)<8:
        d=rng.uniform(BOUNDS[:,0],BOUNDS[:,1])
        try:canonical(d)
        except ValueError:continue
        ds.append(d)
    for i,d in enumerate(ds):
        r,raw=evaluate(d,t,8192,71100);rows.append(r);print('Pilot',i,round(r['seconds'],2),'seconds',flush=True)
        if i<2:np.savez_compressed(OUT/f'pilot_{i}.npz',**raw)
    reg=regions();rep=[]
    for seed in [71250,71251]:
        _,sp=spatial(BASE,262144,seed);rep.append(sp)
    ids=reg['supported'];w=2*(ids//1000)+1
    loss=float(np.average(~np.isin(ids,np.intersect1d(rep[0]['cells'],rep[1]['cells'])),weights=w))
    a=np.load(OUT/'pilot_1.npz')['path_errors'].max(1)<=.5
    b=np.load(OUT/'pilot_0.npz')['path_errors'].max(1)<=.5
    write(OUT/'pilot.json',dict(rows=rows,serial_search_seconds_estimate=1536*np.median([r['seconds'] for r in rows]),baseline_equal_budget_supported_loss=loss,coverage_margin_resolvable_in_development=loss<=.02,path_disagreements=int(np.sum(a!=b)),path_count=len(a),path_precision_note='24 pilot paths cannot establish power; retain256 final paths and permit inconclusive inference, no relaxation',note='Estimate excludes selection/validation; concurrency changes timings'))

def search(method,seed):
    prepare()
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.repair import Repair
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.indicators.hv import HV
    run=OUT/'runs'/f'{method}_{seed}';run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():return
    old=read(run/'records.json') if (run/'records.json').exists() else []
    cache={tuple(r['key']):r for r in old if r['status']=='evaluated'}
    t=bank(71100,[1024,256,256,128],24);records=[];baseline=None
    replacements=BOUNDS[:,0]+h.sobol(8192,7,seed+900)*(BOUNDS[:,1]-BOUNDS[:,0]);replacement_index=0;geometry_rejections=[]
    def feasible_proposal(d):
        nonlocal replacement_index
        d=np.array(d,float)
        while True:
            try:return canonical(d)
            except ValueError as exc:
                geometry_rejections.append(dict(design=d,reason=str(exc)))
                if replacement_index>=len(replacements):raise RuntimeError('Predeclared feasible-proposal replacement budget exhausted')
                d=replacements[replacement_index];replacement_index+=1
    def assess(d):
        nonlocal baseline
        i=len(records);r=dict(slot=i,design=np.array(d),status='infeasible')
        try:
            d=canonical(d);k=key(d)
            if k in cache:r={**cache[k], 'slot':i,'cached':True,'design':d}
            else:
                r,raw=evaluate(d,t,8192,71100);r.update(slot=i,status='evaluated',key=k,cached=False)
                cache[k]=r
            if baseline is None:baseline=r['metrics']
            r['cv']=cv(r['metrics'],baseline)
            r['objectives']=[1-r['metrics']['global_ik_success'],1-r['metrics']['path_completion']]
        except ValueError as exc:
            r.update(reason=str(exc),cv=100,objectives=[1,1],status='sampling_unresolved' if 'sampling' in str(exc).lower() else 'infeasible')
        records.append(r);write(run/'records.json',records)
        if len(records)%16==0:print(method,seed,len(records),'of256',flush=True)
        return r
    initial=np.array([feasible_proposal(d) for d in np.vstack([BASE,PROPOSED,BOUNDS[:,0]+h.sobol(32,7,seed)[:30]*(BOUNDS[:,1]-BOUNDS[:,0])])])
    if method=='nsga2':
        class Problem(ElementwiseProblem):
            def __init__(self):super().__init__(n_var=7,n_obj=2,n_ieq_constr=1,xl=BOUNDS[:,0],xu=BOUNDS[:,1])
            def _evaluate(self,x,out,*args,**kwargs):
                r=assess(x);out['F']=r['objectives'];out['G']=[r['cv']]
        class FeasibleRepair(Repair):
            def _do(self,problem,X,**kwargs):return np.array([feasible_proposal(d) for d in X])
        alg=NSGA2(pop_size=32,n_offsprings=32,sampling=initial,crossover=SBX(prob=.9,eta=15),mutation=PM(prob=1,prob_var=1/7,eta=20),repair=FeasibleRepair(),eliminate_duplicates=True)
        minimize(Problem(),alg,('n_gen',8),seed=seed,verbose=False)
    elif method=='staged':
        for d in initial:assess(d)
        for u in h.sobol(128,7,seed+100):assess(feasible_proposal(BOUNDS[:,0]+u*(BOUNDS[:,1]-BOUNDS[:,0])))
        for centre in [BASE,PROPOSED]:
            for km in np.linspace(centre[5]-2,centre[5]+2,4):
                for ko in np.linspace(centre[6]-2,centre[6]+2,4):
                    d=centre.copy();d[5:]=km,ko;assess(feasible_proposal(d))
        leaders=[]
        for r in sorted([r for r in records if r['status']=='evaluated'],key=lambda r:rank(r,baseline)):
            if all(np.linalg.norm((np.array(r['design'])-d)/(BOUNDS[:,1]-BOUNDS[:,0]))>.01 for d in leaders):leaders.append(np.array(r['design']))
            if len(leaders)==4:break
        while len(leaders)<4:leaders.append(BASE)
        for j,d in enumerate(leaders):
            for u in h.sobol(16,7,seed+200+j):assess(feasible_proposal(np.clip(d+(2*u-1)*[3,4,4,8,8,1.5,1.5],BOUNDS[:,0],BOUNDS[:,1])))
    else:raise ValueError(method)
    assert len(records)==256,len(records)
    hv=HV(ref_point=np.array([1.01,1.01]));progress=[]
    for n in range(32,257,32):
        rs=[r for r in records[:n] if r['status']=='evaluated' and r['cv']<=1e-10]
        progress.append(dict(evaluations=n,hypervolume=float(hv(np.array([r['objectives'] for r in rs]))) if rs else 0,feasible=len(rs)))
    write(run/'progress.json',progress)
    short=[];seen=set()
    for r in sorted([r for r in records if r['status']=='evaluated'],key=lambda r:rank(r,baseline)):
        k=tuple(r['key'])
        if k in seen or k in [key(BASE),key(PROPOSED)]:continue
        short.append(r);seen.add(k)
        if len(short)==4:break
    write(run/'shortlist.json',short)
    write(run/'geometry_rejections.json',geometry_rejections)
    write(run/'complete.json',dict(method=method,seed=seed,slots=len(records),unique_evaluated=len({tuple(r['key']) for r in records if r['status']=='evaluated'}),seconds=sum(r.get('seconds',0) for r in records if not r.get('cached',False))))

def select():
    prepare();t=bank(72100,[4096,1024,1024,512],96);rows={};base=None
    requests=[('original',BASE),('proposed',PROPOSED)]
    for method in ['nsga2','staged']:
        for seed in SEEDS:
            for j,r in enumerate(read(OUT/'runs'/f'{method}_{seed}'/'shortlist.json')):requests.append((f'{method}_{seed}_{j}',np.array(r['design'])))
    for name,d in requests:
        p=OUT/'selection'/f'{name}.json'
        if p.exists():r=read(p)
        else:
            r,raw=evaluate(d,t,65536,72100,subdivisions=20);write(p,r);np.savez_compressed(p.with_suffix('.npz'),**raw)
        if base is None:base=r['metrics']
        r['failures']=[k for k,v in deficits(r['metrics'],base,True).items() if v>1e-10];rows[name]=r
        print('Selection',name,'failures',r['failures'],flush=True)
    champions={}
    for method in ['nsga2','staged']:
        names=[n for n in rows if n.startswith(method)]
        name=min(names,key=lambda n:rank(rows[n],base))
        champions[method]=dict(id=name,**rows[name],diagnostic_only=bool(rows[name]['failures']))
    write(OUT/'selection_records.json',rows);write(OUT/'frozen_champions.json',champions)
    print('CHAMPIONS FROZEN', {m:c['id'] for m,c in champions.items()},flush=True)

def paired_success_ci(a,b,alpha=.025):
    # Conservative paired difference: simultaneous exact binomial bounds on wins/losses.
    a=np.asarray(a,bool);b=np.asarray(b,bool);n=len(a);wins=int(np.sum(a&~b));loss=int(np.sum(~a&b))
    def ci(k):return (0 if k==0 else beta.ppf(alpha/4,k,n-k+1),1 if k==n else beta.ppf(1-alpha/4,k+1,n-k))
    w,l=ci(wins),ci(loss)
    return dict(change=float(np.mean(a)-np.mean(b)),lower=float(w[0]-l[1]),upper=float(w[1]-l[0]),wins=wins,losses=loss,n=n,confidence=1-alpha)

def validate():
    prepare();champ=read(OUT/'frozen_champions.json');write(OUT/'validation_freeze_hash.json',dict(champions_sha=digest(OUT/'frozen_champions.json'),protocol_sha=digest(OUT/'protocol.json')))
    designs={'original':BASE,'proposed':PROPOSED,**{k:np.array(v['design']) for k,v in champ.items()}}
    records={k:[] for k in designs}
    for rep,seed in enumerate([73100,74100]):
        t=bank(seed,[8192,2048,2048,1024],128)
        for name,d in designs.items():
            p=OUT/'validation'/f'{name}_{rep}.json'
            if p.exists():r=read(p)
            else:
                r,raw=evaluate(d,t,262144,seed,subdivisions=100);write(p,r);np.savez_compressed(p.with_suffix('.npz'),**raw)
            records[name].append(r);print('Validated',rep,name,flush=True)
    # Equal-budget baseline repeat calibrates occupancy detectability before interpreting 2% margin.
    reg=regions();braw=[load(OUT/'validation'/f'original_{r}.npz') for r in range(2)]
    stable=reg['supported'];w=2*(stable//1000)+1
    repeat_loss=float(np.average(~np.isin(stable,np.intersect1d(braw[0]['workspace_cells'],braw[1]['workspace_cells'])),weights=w))
    outcome={}
    for name in ['proposed','nsga2','staged']:
        raw=[load(OUT/'validation'/f'{name}_{r}.npz') for r in range(2)];gates={};intervals={}
        for j,region in enumerate(REGION_NAMES):
            aa=[];bb=[]
            for r,seed in enumerate([73100,74100]):
                t=load(OUT/'banks'/f'bank_{seed}.npz');mask=t['groups']==j;aa.extend(raw[r]['ik_errors'][mask]<=.5);bb.extend(braw[r]['ik_errors'][mask]<=.5)
            ci=paired_success_ci(aa,bb);intervals[region+'_ik']=ci
            if region=='global':gates['global_ik_gain']=ci['change']>=.03 and ci['lower']>0
            if region in ['weak','difficult']:gates[region+'_ik_noninferiority']=ci['lower']>=-.02
        a=np.concatenate([x['path_errors'] for x in raw]);b=np.concatenate([x['path_errors'] for x in braw])
        ci=paired_success_ci(a.max(1)<=.5,b.max(1)<=.5);intervals['path_completion']=ci
        gates['path_completion']=ci['change']>=-1e-10 and ci['lower']>=-.02
        rng=np.random.default_rng(75100);diff=[]
        for _ in range(400):
            ix=rng.integers(0,len(a),len(a));diff.append(np.percentile(a[ix],95)-np.percentile(b[ix],95))
        intervals['path_p95_increase_mm']=dict(change=float(np.percentile(a,95)-np.percentile(b,95)),lower=float(np.quantile(diff,.0125)),upper=float(np.quantile(diff,.9875)),resamples=400)
        gates['tracking_p95']=intervals['path_p95_increase_mm']['upper']<=.05
        for rep in range(2):
            m=records[name][rep]['metrics'];bm=records['original'][rep]['metrics']
            for k,v in deficits(m,bm,False).items():gates[f'repeat{rep}_{k}']=v<=1e-10
        gates['coverage_resolvable']=repeat_loss<=.02
        # Grid sensitivity on same matched physical samples, preserving full original occupancy.
        grid=[]
        for rep in range(2):
            for step in [2.5,5,10]:
                bi=h.cells(braw[rep]['workspace_points'],step);ai=h.cells(raw[rep]['workspace_points'],step)
                ids,cnt=np.unique(bi,return_counts=True);supported=ids[cnt>=30]
                loss=volume(np.setdiff1d(supported,np.unique(ai)),step)/volume(supported,step)
                grid.append(dict(repeat=rep,cell_mm=step,supported_original_loss=loss,full_original_loss=volume(np.setdiff1d(ids,np.unique(ai)),step)/volume(ids,step)))
        gates['coverage_resolution_sensitivity']=all(v['supported_original_loss']<=.02 for v in grid)
        if name in champ:gates['selection_passed']=not champ[name]['diagnostic_only']
        outcome[name]=dict(accepted=all(gates.values()),gates=gates,intervals=intervals,grid_sensitivity=grid,failed=[k for k,v in gates.items() if not v])
    write(OUT/'validation_records.json',records);write(OUT/'outcome.json',dict(candidates=outcome,baseline_repeat_supported_loss=repeat_loss,live_configuration_changed=False,not_global_optimality=True))
    sensitivity={}
    t=load(OUT/'banks/bank_73100.npz')
    for name,d in designs.items():
        p=OUT/'sensitivity'/f'{name}.json'
        if p.exists():r=read(p)
        else:
            r,raw=evaluate(d,t,262144,73100,100,108.5);write(p,r);np.savez_compressed(p.with_suffix('.npz'),**raw)
        sensitivity[name]=r
    write(OUT/'gap_sensitivity.json',sensitivity)
    # Sensitivity must not be an afterthought: it can disqualify, never rescue.
    for name in outcome:
        failed=[k for k,v in deficits(sensitivity[name]['metrics'],sensitivity['original']['metrics'],True).items() if v>1e-10]
        outcome[name]['gates']['gap_108_5_sensitivity']=not failed
        outcome[name]['gap_sensitivity_failures']=failed
        outcome[name]['failed']=[k for k,v in outcome[name]['gates'].items() if not v]
        outcome[name]['accepted']=all(outcome[name]['gates'].values())
    write(OUT/'outcome.json',dict(candidates=outcome,baseline_repeat_supported_loss=repeat_loss,live_configuration_changed=False,not_global_optimality=True))
    for p,sha in read(OUT/'protected_hashes.json').items():assert digest(ROOT/p)==sha,p
    print('VALIDATION COMPLETE', {n:r['accepted'] for n,r in outcome.items()},flush=True)

if __name__=='__main__':
    command=sys.argv[1]
    if command=='search':search(sys.argv[2],int(sys.argv[3]))
    else:globals()[command]()
