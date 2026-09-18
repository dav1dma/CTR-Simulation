"""Read-only reporting of the final geometry study; no simulator configuration edits."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctr-final-mpl')
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/final_geometry_method_comparison_20260916'
def read(p):return json.loads(Path(p).read_text())
def write(p,o):Path(p).write_text(json.dumps(o,indent=2))
def mean(r,k):return float(np.mean([x['metrics'][k] for x in r]))
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def true_volume(points,step=5):
    # Explicit (r,z) pairs correctly handle negative z, unlike a signed packed ID.
    ij=np.floor(np.column_stack([np.linalg.norm(points[:,:2],axis=1),points[:,2]])/step).astype(int)
    ij=np.unique(ij,axis=0)
    return float(np.sum(2*ij[:,0]+1)*np.pi*step**3/1000)

def main():
    records=read(OUT/'validation_records.json');out=read(OUT/'outcome.json');frozen=read(OUT/'frozen_champions.json')
    names=['original','proposed','nsga2','staged'];labels=['Original','Current proposed','NSGA-II finalist','Staged-search finalist']
    designs={k:records[k][0]['design'] for k in names};metrics={}
    for name in names:
        raw=[dict(np.load(OUT/'validation'/f'{name}_{r}.npz')) for r in range(2)]
        err=np.concatenate([x['path_errors'] for x in raw]);mx=err.max(1)
        metrics[name]={k:mean(records[name],k) for k in records[name][0]['metrics']}
        metrics[name].update(path_completed=int(np.sum(mx<=.5)),path_total=len(mx),pooled_path_p95_mm=float(np.percentile(err,95)),worst_path_mm=float(mx.max()),paths_over_5mm=int(np.sum(mx>5)),paths_over_10mm=int(np.sum(mx>10)),volume_cm3=float(np.mean([true_volume(x['workspace_points']) for x in raw])),negative_z_observed=bool(any(np.any(x['workspace_points'][:,2]<0) for x in raw)))
    write(OUT/'reported_metrics.json',metrics)
    winners=[n for n in ['nsga2','staged'] if out['candidates'][n]['accepted']]
    verdict=('One or more frozen finalists passed the declared gates: '+', '.join(winners)+'.') if winners else 'Neither frozen search finalist passed all declared selection and validation gates. No replacement is recommended from this study.'
    rows=[]
    for name,label in zip(names,labels):
        m=metrics[name];rows.append([label,f"{100*m['global_ik_success']:.2f}%",f"{100*m['weak_ik_success']:.2f}%",f"{100*m['difficult_ik_success']:.2f}%",f"{m['path_completed']}/{m['path_total']}",f"{m['pooled_path_p95_mm']:.4f}",f"{m['worst_path_mm']:.3f}"])
    geom=[]
    for name,label in zip(names,labels):
        d=np.array(designs[name]);geom.append([label,' / '.join(f'{x:.4f}' for x in d[:3]),' / '.join(f'{x:.4f}' for x in d[3:5]),' / '.join(f'{x:.4f}' for x in d[5:])])
    runrows=[]
    for method in ['nsga2','staged']:
        for seed in [71011,71012,71013]:
            p=OUT/'runs'/f'{method}_{seed}';r=read(p/'complete.json');progress=read(p/'progress.json');rr=read(p/'records.json')
            logfile=OUT/f'{method}_{seed}.log';stat=logfile.stat();started=getattr(stat,'st_birthtime',stat.st_ctime)
            wall=(p/'complete.json').stat().st_mtime-started
            unresolved=sum(x['status']!='evaluated' for x in rr)
            runrows.append([method,seed,r['slots'],r['unique_evaluated'],unresolved,len(read(p/'geometry_rejections.json')),f"{r['seconds']/60:.1f}",f"{wall/60:.1f}",f"{progress[-1]['hypervolume']:.5f}"])
    failed=[]
    for name in ['proposed','nsga2','staged']:
        oc=out['candidates'][name];failed.append(f"- **{name}:** "+(', '.join(oc['failed']) if oc['failed'] else 'all declared gates passed'))
    secondary=[]
    uncertainty=[]
    for name,label in zip(names,labels):
        m=metrics[name]
        secondary.append([label,f"{m['radial_mm']:.2f}",f"{m['zmax_mm']:.2f}",f"{m['reach_mm']:.2f}",f"{100*m['supported_retention']:.2f}%",f"{m['global_isotropy_mean']:.4f}",f"{m['global_isotropy_p10']:.4f}"])
        if name!='original':
            ci=out['candidates'][name]['intervals']['path_completion'];pc=out['candidates'][name]['intervals']['path_p95_increase_mm']
            uncertainty.append([label,f"{100*ci['change']:+.2f}",f"[{100*ci['lower']:+.2f}, {100*ci['upper']:+.2f}]",f"{pc['change']:+.4f}",f"[{pc['lower']:+.4f}, {pc['upper']:+.4f}]"])
    text=f'''# Final CTR geometry-method comparison

{verdict}

The original and current proposed simulator configurations were preserved. All results are predictions of the same ideal exterior curvature-superposition model and unchanged numerical IK/controller. No clinical or experimental performance is established.

## Final assessment on fresh targets and paths

{table(['Configuration','Global IK','Weak-region IK','Difficult-region IK','Completed paths','Pooled tracking P95 (mm)','Worst checked error (mm)'],rows)}

Each design was evaluated on two new banks, each containing 8,192 global, 2,048 weak-region, 2,048 difficult-region and 1,024 corridor targets, 262,144 workspace states and 128 paths. Targets are shared between designs. The global score uses only global targets; overlapping regional groups are not counted twice. Paths comprise representative, weak-endpoint and actuator/model-boundary stress strata. Aggregate path performance refers to this declared mixture, not a clinical distribution. Each route uses the unchanged 20 control steps and 40-iteration warm-start solver and is checked at 2,001 positions. The dense check does not certify continuous-time feasibility between samples.

The pooled P95 above uses all final path samples. It is not the mean of bank-level P95 values. The old 44/48 versus 43/48 counts belong to the earlier development study and are not mixed with these new tests. All failed trajectories remain included.

## Frozen designs

{table(['Configuration','Total inner/middle/outer (mm)','Curved middle/outer (mm)','Precurvature middle/outer (1/m)'],geom)}

The inner element remains straight. Straight material lengths equal total minus curved length. Digits permit numerical reproduction and do not imply fabrication accuracy. Curved material exceeding maximum feasible exposure is exterior-equivalent; it is not a physically validated interchangeable design.

## Acceptance outcomes

{chr(10).join(failed)}

Every gate is required. A diagnostic finalist that failed dense selection cannot be rescued by a favourable held-out result. Statistical acceptance uses conservative paired intervals, with 97.5% intervals for the two method finalists; insufficient precision is not evidence of equivalence. Region and path margins are study-specific engineering choices, not literature-derived clinical requirements. The intervals describe variation under these sampled numerical test banks, not uncertainty in the physical robot. The P95 interval uses 400 paired whole-path bootstrap resamples; its tail estimates are approximate, and a result close to the margin should be treated cautiously. Detailed point estimates, intervals, individual gates and the 108.5 mm gap sensitivity are in `outcome.json` and `gap_sensitivity.json`.

Original-only equal-budget final supported-cell repeat loss: {100*out['baseline_repeat_supported_loss']:.3f}%. Coverage is also checked at 2.5, 5 and 10 mm grids. Full original-cell losses, including sparse boundary cells, remain reported in the outcome; the acceptance mask was frozen from adequately sampled original development cells. No candidate can improve a regional mean by deleting its failed original targets.

{table(['Configuration','Completion change (pp)','Completion interval (pp)','P95 change (mm)','P95 bootstrap interval (mm)'],uncertainty)}

Failure of a confidence-interval gate does not establish that the candidate is worse. It means this study did not establish the declared margin. In particular, the staged finalist improved observed completion, but its interval extends below the permitted two-percentage-point loss. The NSGA-II finalist lost one completed path, and its P95 interval did not rule out an increase greater than 0.05 mm. Both finalists also failed the global lower-tail isotropy safeguard in one repeat. The staged finalist had already failed selection and therefore remains diagnostic, irrespective of favourable held-out averages. Worst checked errors remain substantial for all four configurations and must be reported alongside P95.

## Workspace and positional dexterity

{table(['Configuration','Maximum radius (mm)','Maximum Z (mm)','Maximum tip distance (mm)','Supported reference cells retained','Mean global isotropy','Global isotropy P10'],secondary)}

Values in this table are the mean of the two bank-specific estimates. Acceptance used each repeat separately; averaging must not conceal a failed repeat. Supported occupancy retention does not mean every original point, orientation or continuous route remains reachable. All reach values describe the sampled ideal model.

## Method comparison

{table(['Method','Seed','Evaluator slots','Unique completed','Unresolved','Geometry rejections','Completed evaluator minutes','Wall minutes','Feasible hypervolume'],runrows)}

Each method used three seeds and 256 evaluator slots per seed. Analytically infeasible geometry proposals were replaced using the same bounded Sobol reserve rule and logged separately. Cached exterior-equivalent evaluations and numerical sampling failures are retained in the records. Completed-evaluator minutes exclude unsuccessful sampling attempts; wall minutes include them and are measured from launch-log creation to completion-marker time. Runs execute concurrently, so these are not isolated CPU benchmarks. Counts refer to the full evaluator, not a trace of every internal FK call. Feasible hypervolume is a summary of the two screen objectives against a fixed failure-fraction reference point (1.01, 1.01); it is not workspace volume. It is measured on development samples and does not establish generalisation or global optimality.

Both methods used the same original/proposed seed designs, objectives, geometry bounds, evaluator settings, acceptance margins and dense reassessment allowance. This is a controlled comparison of candidate-generation strategies, not an exact rerun of the historical 1:2:2 regional scoring algorithm. Previous inspected targets, paths and maps are development evidence. The finalists were frozen before opening the final validation banks. Three search seeds provide only a limited repeatability check.

## Interpretation and limitations

The study distinguishes a better candidate from a better optimisation method. A single successful run would not establish general superiority of NSGA-II. Small changes or plateaus under the allocated budget cannot establish convergence to the global optimum. If neither finalist passes, retain the existing references and report the failed criteria and trade-offs without changing thresholds after validation.

The hardware remains 100 mm nominal stroke per actuator, 71.5 mm actuator bodies and 8.5–108 mm coupled gaps. The108.5mm sensitivity case is disqualifying only: it cannot rescue a primary failure. Lengths remain provisional chuck-to-tip material lengths with no 35 mm offset. Internal guidance, retention, rotation limits and manufacture-certified curvature bounds remain unresolved. The model omits torsion, friction, elastic stability, contact, anatomy, loading and physical validation. Positional isotropy uses the existing normalisation and does not guarantee feasible motion in every direction at an actuator stop.

The reporting script computes annular occupancy volume from explicit radial/axial cell pairs so negative axial coordinates, if present in broad candidates, cannot corrupt a packed cell-ID volume estimate. Volume is descriptive and was not a search objective or acceptance gate. Fixed-reference coverage and isotropy use the unchanged reference-cell identities.

## Reproducibility and files

- `protocol.json`, `preflight_amendment.json`, `environment.json`: frozen settings and pre-search corrections.
- `source_snapshot/`, `source_hashes.json`, `protected_hashes.json`: implementation and preservation evidence.
- `checks.json`, `pilot.json`: kernel checks, runtime and original-only occupancy calibration.
- `runs/`: complete proposals, rejections, objective histories and shortlists.
- `selection/`, `selection_records.json`, `frozen_champions.json`: selection evidence and frozen candidates.
- `banks/`, `validation/`, `sensitivity/`: target coordinates/source states, paths, resulting states/errors and workspace samples.
- `outcome.json`, `reported_metrics.json`: decision gates and reported numbers.
- `figures/`: standalone comparison figures, PDFs and PNGs.
- `candidate_nsga2.json`, `candidate_staged.json`: standalone study exports; neither is installed in a simulator.

Run the checks and stages from the project root using `.venv/bin/python tools/final_geometry_comparison.py COMMAND`. Search commands additionally take `nsga2` or `staged` and one of 71011, 71012, 71013. Use the existing study only to resume unchanged settings; use a new study directory and new reserved seeds for a different protocol. Report generation uses `.venv/bin/python tools/report_final_geometry_comparison.py`. The NSGA-II library is isolated under the study's dependencies folder; existing model libraries retain precedence.

Method references: Deb et al. (2002), https://doi.org/10.1109/4235.996017 ; Bergeles et al. (2015), https://doi.org/10.1109/TRO.2014.2378431 ; Baykal et al. (2015), https://doi.org/10.1109/IROS.2015.7353999 . These support the method discussion, not the numerical acceptance margins.
'''
    (OUT/'study_report.md').write_text(text)
    for name in ['nsga2','staged']:
        d=np.array(designs[name]);curved=np.r_[0,d[3:5]]
        write(OUT/f'candidate_{name}.json',dict(study='final_geometry_method_comparison_20260916',method=name,installed=False,accepted=out['candidates'][name]['accepted'],failed_gates=out['candidates'][name]['failed'],total_lengths_mm=d[:3].tolist(),curved_lengths_mm=curved.tolist(),straight_lengths_mm=(d[:3]-curved).tolist(),precurvature_per_m=[0,*d[5:].tolist()],grip_offset_mm=0,length_interpretation='Provisional chuck tip to distal tip; not fabrication-ready',hardware='Measured100mm coupled strokes; gap8.5–108mm; sensitivity108.5mm'))
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#4c78a8','#59a89c','#9467bd','#dd9d31']
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for method,color in [('nsga2',colors[2]),('staged',colors[3])]:
        curves=[]
        for seed in [71011,71012,71013]:
            p=read(OUT/'runs'/f'{method}_{seed}'/'progress.json');x=[r['evaluations'] for r in p];y=[r['hypervolume'] for r in p];curves.append(y);ax[0].plot(x,y,color=color,alpha=.35)
            r=read(OUT/'runs'/f'{method}_{seed}'/'records.json');r=[z for z in r if z['status']=='evaluated'];ax[1].scatter([100*z['metrics']['global_ik_success'] for z in r],[100*z['metrics']['path_completion'] for z in r],s=7,alpha=.12,color=color)
        ax[0].plot(x,np.median(curves,axis=0),color=color,lw=2,label=method)
        ax[1].scatter([],[],color=color,label=method)
    ax[0].set(xlabel='Evaluator slots per run',ylabel='Feasible objective hypervolume',title='Development progress: median and three runs');ax[0].legend()
    ax[1].set(xlabel='Development global IK success (%)',ylabel='Development path completion (%)',title='Evaluated designs, including failed safeguards');ax[1].legend()
    fig.tight_layout();fig.savefig(figures/'method_comparison.pdf');fig.savefig(figures/'method_comparison.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.2));x=np.arange(4)
    for ax,k,title,factor in zip(axes,['global_ik_success','path_completion','pooled_path_p95_mm'],['Fresh global IK success','Fresh path completion','Fresh tracking P95'],[100,100,1]):
        vals=[factor*metrics[n][k] for n in names];ax.bar(x,vals,color=colors);ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_title(title);ax.set_ylabel('%' if factor==100 else 'mm')
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.2f}' if factor==100 else f'{v:.3f}',ha='center',va='bottom',fontsize=8)
        ax.set_ylim(0,100 if factor==100 else max(vals)*1.15)
    fig.suptitle('Frozen designs on identical fresh tests; bars do not imply acceptance',fontsize=11)
    fig.tight_layout();fig.savefig(figures/'final_validation.pdf');fig.savefig(figures/'final_validation.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,['global_ik','path_completion'],['Global IK success difference','Path completion difference']):
        for j,name in enumerate(names[1:]):
            ci=out['candidates'][name]['intervals'][metric];v=100*ci['change'];lo=100*ci['lower'];hi=100*ci['upper']
            ax.errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='o',color=colors[j+1],capsize=4)
        ax.axvline(0,color='grey',lw=1);ax.set_yticks(range(3),labels[1:]);ax.set_xlabel('Change from original (percentage points)');ax.set_title(title)
    fig.suptitle('Paired conservative 97.5% intervals; held-out evaluation',fontsize=11);fig.tight_layout();fig.savefig(figures/'validation_differences.pdf');fig.savefig(figures/'validation_differences.png',dpi=200);plt.close(fig)
    fig=plt.figure(figsize=(12,9));clouds=[]
    for name in names:
        p=np.load(OUT/'validation'/f'{name}_0.npz')['workspace_points'];clouds.append(p[::max(1,len(p)//16000)])
    allp=np.concatenate(clouds);span=np.max(np.ptp(allp,axis=0));centre=(allp.max(0)+allp.min(0))/2
    for k,(name,label,p) in enumerate(zip(names,labels,clouds)):
        ax=fig.add_subplot(2,2,k+1,projection='3d');ax.scatter(*p.T,s=2,alpha=.18,color=colors[k],linewidths=0,depthshade=False,rasterized=True);ax.scatter([0],[0],[0],color='black',s=30,marker='+')
        ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=label,xlim=(centre[0]-span/2,centre[0]+span/2),ylim=(centre[1]-span/2,centre[1]+span/2),zlim=(min(0,centre[2]-span/2),max(0,centre[2]+span/2)))
        ax.set_box_aspect([1,1,1]);ax.view_init(elev=23,azim=-60)
    fig.suptitle('Sampled ideal-model tip positions — same axes; no guarantee of interior reachability',fontsize=11);fig.tight_layout();fig.savefig(figures/'workspace_samples.pdf');fig.savefig(figures/'workspace_samples.png',dpi=200);plt.close(fig)
    protected=read(OUT/'protected_hashes.json');changed=[p for p,h in protected.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    write(OUT/'preservation_verification.json',dict(checked_files=len(protected),changed_files=changed))
    print(verdict,flush=True)
    print(table(['Configuration','Global IK','Weak IK','Difficult IK','Paths','P95mm','Maxmm'],rows),flush=True)

if __name__=='__main__':main()
