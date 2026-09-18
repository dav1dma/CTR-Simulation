"""Non-regression search. Writes a separate study, never installs a candidate.

Run: search, assess, validate, report. All acceptance gates are conjunctions.
The former trade-off study supplies only tested model kernels and frozen regions.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctr-mpl')
from pathlib import Path
import sys,json,time,hashlib,shutil
import numpy as np
import optimise_measured_ctr as h
OUT=h.ROOT/'output/strict_measured_optimisation_20260914'
OLD=h.OUT
BASE=h.BASE
CURRENT=np.array(json.loads((OLD/'selected_frozen.json').read_text())['design'])
REF=dict(np.load(OLD/'baseline_regions.npz'))
REGIONS={'global':REF['cells'][REF['counts']>=5],'weak':REF['weak_cells'],'corridor':REF['corridor_cells']}

def write(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a)))

def spatial(d,n,seed,gap=108.):
    sp=h.spatial(d,n,seed,gap=gap);p=sp['points'];result={}
    for name,ids in REGIONS.items():
        w=2*(ids//1000)+1;v=h.map_values(sp,ids)
        result[name+'_isotropy_mean']=float(np.average(v,weights=w))
        result[name+'_isotropy_p10']=h.wquant(v,w,.1)
        result[name+'_cell_retention']=float(np.average(np.isin(ids,sp['cells']),weights=w))
    result.update(volume_cm3=float(sum(2*(sp['cells']//1000)+1)*np.pi*5**3/1000),
        radial_extent_mm=float(np.linalg.norm(p[:,:2],axis=1).max()),z_max_mm=float(p[:,2].max()),
        z_min_mm=float(p[:,2].min()),tip_distance_max_mm=float(np.linalg.norm(p,axis=1).max()))
    return result,sp

def tolerance(k):
    if 'success' in k or 'retention' in k or 'completion' in k:return 1e-10
    if 'isotropy' in k:return 1e-8
    return 1e-6
def lower_better(k):return 'p95_mm' in k or k=='z_min_mm'
def compare(a,b):
    failures=[];deficits=[]
    for k in b:
        delta=(b[k]-a[k]) if lower_better(k) else (a[k]-b[k])
        if delta < -tolerance(k):
            failures.append(dict(metric=k,original=b[k],candidate=a[k],change=a[k]-b[k]))
            deficits.append(-delta/max(abs(b[k]),.01))
    return dict(pass_all=not failures,failures=failures,failed_count=len(failures),
        maximum_relative_deficit=max(deficits,default=0),sum_relative_deficit=sum(deficits))
def rank(r):
    c=r['comparison'];return c['failed_count'],c['maximum_relative_deficit'],c['sum_relative_deficit']
def equivalent(d):
    # Distal curves longer than every possible exposure are unidentifiable here.
    return np.allclose(d[:3],BASE[:3],atol=1e-9,rtol=0) and np.allclose(d[5:],BASE[5:],atol=1e-9,rtol=0) and d[3]>=75 and d[4]>=65
def meaningful(a,b):
    return any(a[k]>=b[k]+.005 for k in b if 'success' in k) or any(a[k]>=b[k]*1.01 and a[k]-b[k]>1e-5 for k in b if 'isotropy' in k) or a['volume_cm3']>=b['volume_cm3']*1.01

def search():
    protocol=dict(id='strict-measured-nonregression-v1',baseline=BASE,current_tradeoff=CURRENT,bounds=h.BOUNDS,
        physical_constraints=json.loads((OLD/'protocol.json').read_text())['measured'],
        assumptions=json.loads((OLD/'protocol.json').read_text())['assumptions'],
        acceptance='Every gated estimate must be no worse; no averaging gains against losses. Only floating-point tolerances: rates1e-10, isotropy1e-8, mm/cm3 1e-6. At least one success improvement0.5 percentage point, isotropy1%, or volume1% required. No physical uncertainty allowance.',
        metrics=['global/weak/corridor mean and P10 cell isotropy','global/weak/corridor reference cell retention','annular occupancy volume','radial maximum, z maximum, z minimum, maximum tip distance','global/weak/corridor fixed-start success and P95 error','global/weak/corridor two-start diagnostic success','trajectory completion and P95 tracking residual'],
        screen=dict(n=8192,seed=41101,shortlist=24,refinement='top6 distinct designs, 32 local Sobol each',sampling_budget='Up to1000 batches of8192 physical proposals; exceptionally thin domains exhausting the sampler are logged separately, not declared infeasible'),
        selection=dict(n=65536,seed=42101,targets=512,target_seed=42102),
        validation=dict(n=131072,seeds=[43101,44101],targets=2048,target_seeds=[43102,44102],gap_cases=[108,108.5],trajectory_seed=43103),
        trajectories='24 smooth baseline joint paths, 21 waypoints; candidate starts by two-nearest-start IK then 40-iteration warm-start IK. Five intermediate joint positions per segment checked against the continuous baseline path, <=0.5mm completion. No anatomy or contact.',
        regions='Unchanged from previous original-tube region bank; benchmark regions, not anatomy',
        directional_extent='Preserve sampled radial/z bounds and maximal distance; symmetry assumes unrestricted common rotation',
        selection_rule='Freeze all meaningful stage2 passes (up to8) plus up to3 nearest failing candidates for diagnostic validation. Failing candidates cannot be promoted unless they also pass stage2. No validation-driven redesign.',
        scope='Straight inner wire and original diameters/stiffness fixed. Bounded search is not proof of nonexistence or a global optimum.')
    write('protocol.json',protocol)
    shutil.copy2(OLD/'baseline_regions.npz',OUT/'baseline_regions.npz')
    live=h.ROOT/'measured_hardware_simulator/optimised_configuration.json'
    write('original_live_hash.json',dict(path=str(live),sha256=hashlib.sha256(live.read_bytes()).hexdigest()))
    candidates=[BASE,CURRENT,np.array([350,170,80,100,70,19.12,14.04])]
    for file in ('screen_records.json','fine_records.json'):
        candidates.extend(np.array(r['design']) for r in json.loads((OLD/file).read_text()))
    candidates.extend(h.BOUNDS[:,0]+h.sobol(256,7,41102)*(h.BOUNDS[:,1]-h.BOUNDS[:,0]))
    for span,seed in [(np.array([10,10,10,20,15,3,3]),41103),(np.array([2,2,2,3,3,.5,.5]),41104)]:
        candidates.extend(np.clip(BASE+(h.sobol(128,7,seed)*2-1)*span,h.BOUNDS[:,0],h.BOUNDS[:,1]))
    for km in np.linspace(17.12,23.12,17):
        for ko in np.linspace(12.04,17.04,17):
            d=BASE.copy();d[5:]=[km,ko];candidates.append(d)
    checkpoint=OUT/'screen_records.json'
    records=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    for r in records:r['design']=np.array(r['design'])
    seen={tuple(r['design']) for r in records};start=time.time()
    skipped=json.loads((OUT/'sampling_skips.json').read_text()) if (OUT/'sampling_skips.json').exists() else []
    seen.update(tuple(r['design']) for r in skipped)
    def assess(d,phase):
        key=tuple(np.round(d,4))
        if key in seen:return
        seen.add(key)
        if d[3]>d[1] or d[4]>d[2]:return
        try:h.limits(d)
        except ValueError:return
        # Round once before search evaluation; do not change parameters after selection.
        d=np.round(d,4)
        try:metrics,sp=spatial(d,8192,41101)
        except ValueError as exc:
            if 'sampling volume' not in str(exc):raise
            skipped.append(dict(design=d,reason='Uniform rejection sampling budget exhausted; not a proof of infeasibility'))
            write('sampling_skips.json',skipped);return
        if not records:write('screen_baseline.json',metrics)
        original=metrics if not records else records[0]['metrics']
        row=dict(id=f's{len(records):04d}',design=d,phase=phase,equivalent=equivalent(d),metrics=metrics,comparison=compare(metrics,original))
        records.append(row)
        if len(records)%50==0:
            write('screen_records.json',records)
            print('Screened',len(records),'elapsed seconds',round(time.time()-start),'non-equivalent passes',sum(r['comparison']['pass_all'] and not r['equivalent'] for r in records),flush=True)
    for d in candidates:assess(d,'initial')
    leaders=[]
    for r in sorted(records[1:],key=rank):
        if r['equivalent']:continue
        if all(np.linalg.norm((r['design']-v)/(h.BOUNDS[:,1]-h.BOUNDS[:,0]))>.005 for v in leaders):leaders.append(r['design'])
        if len(leaders)==6:break
    for i,d in enumerate(leaders):
        span=np.array([1,1,1,2,2,.3,.3])
        for v in np.clip(d+(h.sobol(32,7,41201+i)*2-1)*span,h.BOUNDS[:,0],h.BOUNDS[:,1]):assess(v,'local refinement')
    write('screen_records.json',records)
    candidates=sorted([r for r in records if not r['equivalent']],key=rank)
    # Current comparison is always retained regardless of its rank.
    selected=candidates[:24]
    if not any(np.allclose(r['design'],CURRENT) for r in selected):selected.append(next(r for r in records if np.allclose(r['design'],CURRENT)))
    write('shortlist.json',selected)
    print('SCREEN COMPLETE',len(records),'shortlist',len(selected),flush=True)

def ikmetrics(d,t,gap=108.):
    er,st,x=h.solve(t['targets'],d,gap=gap);di,ds=h.diagnostic(t['targets'],d,gap);r={}
    for j,name in enumerate(REGIONS):
        m=t['groups']==j;w=t['weights'][m]
        r[name+'_ik_success']=float(np.average(er[m]<=.5,weights=w))
        r[name+'_ik_p95_mm']=h.wquant(er[m],w,.95)
        r[name+'_diagnostic_success']=float(np.average(di[m]<=.5,weights=w))
    return r,dict(errors=er,states=st,latent=x,diagnostic_errors=di,diagnostic_states=ds)

def assess():
    t=h.targetbank(512,42102,REF);np.savez_compressed(OUT/'selection_targets.npz',**t)
    rows=[dict(id='original',design=BASE)]+json.loads((OUT/'shortlist.json').read_text());results=[]
    for r in rows:
        d=np.array(r['design']);m,sp=spatial(d,65536,42101);ik,raw=ikmetrics(d,t);m.update(ik)
        if not results:baseline=m
        comp=compare(m,baseline);r=dict(id=r['id'],design=d,metrics=m,comparison=comp,meaningful=meaningful(m,baseline),equivalent=equivalent(d))
        results.append(r);np.savez_compressed(OUT/(r['id']+'_selection.npz'),**raw,**{'workspace_'+k:v for k,v in sp.items()})
        write('selection_records.json',results);print('Assessed',r['id'],'failed',comp['failed_count'],[a['metric'] for a in comp['failures']],flush=True)
    passed=[r for r in results[1:] if r['comparison']['pass_all'] and r['meaningful'] and not r['equivalent']]
    passed.sort(key=lambda r: -sum(r['metrics'][g+'_ik_success'] for g in REGIONS))
    chosen=passed[:8]
    for r in sorted(results[1:],key=rank):
        if r['id'] not in [x['id'] for x in chosen] and not r['equivalent']:chosen.append(r)
        if len(chosen)>=max(3,len(passed[:8])):break
    write('frozen_finalists.json',chosen)
    print('SELECTION COMPLETE: strict meaningful passes',len(passed),'frozen validation IDs',[r['id'] for r in chosen],flush=True)

def trajectory(d,seed,gap=108.):
    endpoints=h.samples(BASE,48,seed).reshape(24,2,6);a=endpoints[:,0];b=endpoints[:,1]
    b[:,3:]=a[:,3:]+(b[:,3:]-a[:,3:]+np.pi)%(2*np.pi)-np.pi
    reference=lambda u:h.fk(a+(b-a)*u,BASE)
    first=reference(0);er,st=h.diagnostic(first,d,gap)
    lim=h.limits(d,gap)
    x=np.column_stack([np.array([lim.encode(s[:3]/1000) for s in st]),st[:,4:]/np.pi])
    failures=er>.5;errors=[er];allstates=[st];alltargets=[first]
    for j,u in enumerate(np.linspace(0,1,21)[1:],1):
        old=st.copy();er,st,x=h.solve(reference(u),d,x,40,gap)
        for f in np.linspace(0,1,5)[1:]:
            mid=old+(st-old)*f;lim.validate(mid[:,:3]/1000)
            e=np.linalg.norm(h.fk(mid,d)-reference((j-1+f)/20),axis=1)
            failures|=e>.5;errors.append(e)
        allstates.append(st);alltargets.append(reference(u))
    e=np.stack(errors,axis=1)
    return dict(trajectory_completion=float(np.mean(~failures)),trajectory_p95_mm=float(np.percentile(e,95))),dict(trajectory_errors=e,trajectory_states=np.array(allstates),trajectory_targets=np.array(alltargets),trajectory_source_endpoints=endpoints)

def validate():
    frozen=json.loads((OUT/'frozen_finalists.json').read_text())
    records=[dict(id='original',design=BASE)]+frozen
    if not any(np.allclose(r['design'],CURRENT) for r in records):records.append(dict(id='current_tradeoff',design=CURRENT))
    results={}
    for replicate,(seed,tseed) in enumerate([(43101,43102),(44101,44102)]):
        t=h.targetbank(2048,tseed,REF);np.savez_compressed(OUT/f'validation_targets_{replicate}.npz',**t)
        for row in records:
            d=np.array(row['design']);name=row['id'];m,sp=spatial(d,131072,seed);ik,raw=ikmetrics(d,t);m.update(ik)
            tr,trraw=trajectory(d,43103+replicate);m.update(tr)
            if name=='original':base=m
            comparison=compare(m,base);results.setdefault(name,[]).append(dict(metrics=m,comparison=comparison,meaningful=meaningful(m,base)))
            np.savez_compressed(OUT/f'{name}_validation_{replicate}.npz',**raw,**trraw,**{'workspace_'+k:v for k,v in sp.items()})
            write('validation_records.json',results)
            print('Validated',replicate,name,'failed',comparison['failed_count'],[a['metric'] for a in comparison['failures']],flush=True)
    # Sensitivity is another gate for otherwise fully passing candidates only.
    passed=[r for r in frozen if r['comparison']['pass_all'] and r['meaningful'] and all(v['comparison']['pass_all'] and v['meaningful'] for v in results[r['id']])]
    sensitivity={}
    if passed:
        t=dict(np.load(OUT/'validation_targets_0.npz'))
        for r in [dict(id='original',design=BASE)]+passed:
            d=np.array(r['design']);m,sp=spatial(d,131072,45101,108.5);ik,_=ikmetrics(d,t,108.5);m.update(ik);tr,_=trajectory(d,43103,108.5);m.update(tr)
            if r['id']=='original':base=m
            sensitivity[r['id']]=dict(metrics=m,comparison=compare(m,base))
        passed=[r for r in passed if sensitivity[r['id']]['comparison']['pass_all']]
    write('gap_sensitivity.json',sensitivity)
    write('outcome.json',dict(passing_ids=[r['id'] for r in passed],selected_id=passed[0]['id'] if passed else None,
        status='validated non-regressing candidate found' if passed else 'No non-equivalent meaningful candidate passed every gate in this bounded search',
        live_configuration_changed=False,validation_replicates=2,metrics_per_replicate=len(base),not_proof_of_nonexistence=True))
    live=json.loads((OUT/'original_live_hash.json').read_text());assert hashlib.sha256(Path(live['path']).read_bytes()).hexdigest()==live['sha256']
    sources=[Path(__file__),Path(h.__file__),h.ROOT/'measured_hardware_simulator/design_constraints.py',h.ROOT/'tools/run_broad_hardware_study.py',h.ROOT/'tools/run_hardware_optimisation.py']
    for p in sources:
        target=OUT/'source_snapshot'/p.relative_to(h.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write('source_hashes.json',{str(p.relative_to(h.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

def checks():
    b={'global_isotropy_mean':.3,'global_isotropy_p10':.15,'global_ik_success':.9,'global_ik_p95_mm':2.}
    assert compare(b,b)['pass_all']
    a={**b,'global_isotropy_mean':.5,'global_isotropy_p10':.14};assert not compare(a,b)['pass_all']
    assert not compare({**b,'global_ik_success':.899},b)['pass_all']
    assert not compare({**b,'global_ik_p95_mm':2.1},b)['pass_all']
    s=h.samples(BASE,128,41001);d=np.array([350,170,80,100,70,19.12,14.04])
    np.testing.assert_allclose(h.fk(s,BASE),h.fk(s,d),atol=1e-10)
    assert equivalent(d) and not equivalent(CURRENT)
    write('checks.json',dict(non_compensatory_gates=True,direction_of_error_metrics=True,equivalent_curve_family_control=True))
    print('Strict gate checks passed',flush=True)

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outcome=json.loads((OUT/'outcome.json').read_text())
    screen=json.loads((OUT/'screen_records.json').read_text());selection=json.loads((OUT/'selection_records.json').read_text())
    validation=json.loads((OUT/'validation_records.json').read_text());frozen=json.loads((OUT/'frozen_finalists.json').read_text())
    skipped=json.loads((OUT/'sampling_skips.json').read_text()) if (OUT/'sampling_skips.json').exists() else []
    closest=min(frozen,key=rank);cid=closest['id'];a=validation['original'][0]['metrics'];b=validation[cid][0]['metrics']
    failures={v['metric'] for v in validation[cid][0]['comparison']['failures']}
    metric_rows=[]
    for k in a:
        rate='success' in k or 'retention' in k or 'completion' in k
        factor=100 if rate else 1
        unit='%' if rate else ('mm' if k.endswith('_mm') else ('cm³' if k.endswith('_cm3') else ''))
        metric_rows.append(f"| {k.replace('_',' ')} | {a[k]*factor:.6g} {unit} | {b[k]*factor:.6g} {unit} | {'FAIL' if k in failures else 'Pass'} |")
    checks='\n'.join(f"| {name} | {vals[0]['comparison']['failed_count']} | {vals[1]['comparison']['failed_count']} |" for name,vals in validation.items() if name!='original')
    d=np.array(closest['design']);parameter_rows='\n'.join(f'| {n} | {total:.4f} | {curve:.4f} | {kap:.4f} |' for n,total,curve,kap in zip(('Inner','Middle','Outer'),d[:3],[0,*d[3:5]],[0,*d[5:]]))
    meaningful_passes=sum(r['comparison']['pass_all'] and r['meaningful'] and not r['equivalent'] for r in selection[1:])
    source='\n'.join(f'- `{name}`' for name in ('protocol.json','screen_records.json','selection_records.json','frozen_finalists.json','validation_records.json','outcome.json'))
    text=f'''# Strict measured-hardware non-regression search

**Outcome: {outcome['status']}.** The active optimised configuration was not changed by this study. This is a bounded numerical result, not proof that a dominating design cannot exist.

## Search and gates

{len(screen):,} unique rounded, feasible-sampled designs were evaluated in the spatial screen. {sum(r['equivalent'] for r in screen)} are baseline/exterior-equivalent controls and cannot count as improvements. {len(skipped)} exceptionally thin operating domains exhausted the uniform rejection-sampling budget and are listed separately; they were not declared mechanically infeasible. Broad, local and curvature-grid candidates include previous designs and new samples, with all variable bounds unchanged from the previous study. The inner element remains straight and diameters/stiffness are fixed.

The denser selection stage evaluated {len(selection)-1} non-control designs and the original baseline; {meaningful_passes} passed every selection metric with a meaningful improvement. {len(frozen)} finalists were frozen for independent validation. If none passed selection, the closest failures were still validated diagnostically; validation does not erase an earlier selection failure.

Each independent validation uses131,072 physical configurations per design and3,072 identical baseline-derived target draws (2,048 global,512 weak,512 corridor). There are two independent configuration/target seeds. The original baseline is rerun with each seed, not compared against an old sample with a different size. The regions are the previously frozen original-tube benchmark regions, not anatomical regions.

All25 metrics must be non-regressing. No overall score can offset a failed metric. The only tolerances are1e-10 for fractions,1e-8 for isotropy and1e-6 for distances/volumes, solely for floating-point equality. These are stricter than statistical non-inferiority margins; no sampling uncertainty allowance was introduced after observing results. A passing design must also improve at least one success fraction by0.5 percentage points, an isotropy metric by1%, or volume by1%. Exterior-equivalent changes are excluded.

## Independent failures

| Candidate ID | Failed metrics: replicate1 | Failed metrics: replicate2 |
|---|---:|---:|
{checks}

The original compared with itself passes as a control. A candidate failing even one metric is rejected under the requested rule. Gap108.5mm sensitivity is an additional gate only for otherwise fully passing candidates; an empty gap_sensitivity.json means none qualified for that gate.

## Closest selection-stage candidate: {cid}

This candidate is shown to explain the result, not presented as a replacement.

| Tube | Chuck-to-tip length (mm) | Curved length (mm) | Precurvature (1/m) |
|---|---:|---:|---:|
{parameter_rows}

First independent replicate:

| Metric | Original | Candidate | Non-regression |
|---|---:|---:|---|
{chr(10).join(metric_rows)}

## What the measurements mean

Mean and lower-decile isotropy are computed on fixed global, weak-region and corridor cells, with fixed[350,170,80,pi,pi,pi] Jacobian scaling. Missing or insufficiently sampled cells receive zero; therefore support/coverage losses can reduce isotropy scores. This is not solely dexterity at shared points. Each cell is5mm in radius and height, with at least five observations for an isotropy median. Cell retention is weighted by swept annular volume. Occupancy volume and coordinate extrema are sample-based estimates; they do not certify all interior points as reachable.

Fixed-start IK uses the same40-iteration,0.5mm-success test for every design, starting at its most-retracted feasible exterior pose with zero rotations. Regional P95 errors include failed attempts. The two-nearest-start diagnostic separately tests baseline target coverage with60 iterations per start. Numerical failure does not prove geometric unreachability. The numerical model, starts and targets differ from older dissertation results using obsolete hardware limits.

Trajectory testing uses24 smooth baseline-generated paths,21 solved waypoints and five joint-interpolation positions per segment. The reference is the continuous baseline path; candidate exposure/rotation interpolation is checked against it. Completion requires every evaluated tip error to stay within0.5mm, and every intermediate state must obey the coupled hardware constraints. P95 includes all evaluated path errors. It does not evaluate anatomy collisions, loads, actuator timing or experimental control accuracy.

Track utilisation is a diagnostic, not a substitute for these gates. All designs retain the measured100mm nominal strokes,71.5mm actuator bodies and8.5–108mm clearances. Installed material lengths, retention, internal passage and straightening, rotation limits and manufacturable curvatures remain provisional. The model omits torsion, friction, elastic stability and physical validation.

## Reproduce and inspect

From the project folder run `.venv/bin/python tools/strict_measured_search.py checks`, then `search`, `assess`, `validate`, and `report`. The search can resume its own checkpoint. Start with a fresh output directory to rerun from scratch. Saved NPZ files contain source target states, resulting IK states/errors, workspace points/occupancy/isotropy and trajectory states/errors. Source snapshots and hashes are retained.

{source}

Do not replace the existing candidate on the basis of being close to passing. A failure-free empirical comparison would still be conditional on the listed model, metrics and samples, rather than a universal improvement guarantee.
'''
    (OUT/'strict_search_report.md').write_text(text)
    names=[k for k in validation if k!='original'];x=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(8,4))
    for j in range(2):
        y=[validation[k][j]['comparison']['failed_count'] for k in names]
        bars=ax.bar(x+(j-.5)*.34,y,.34,label=f'Independent replicate {j+1}')
        ax.bar_label(bars,fmt='%d')
    ax.set_xticks(x,names);ax.set_ylabel('Failed non-regression metrics (of25)');ax.set_title('A qualifying design must pass every metric')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,25)
    fig.tight_layout();fig.savefig(OUT/'non_regression_failures.png',dpi=180);plt.close(fig)
    print(outcome['status'],flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
