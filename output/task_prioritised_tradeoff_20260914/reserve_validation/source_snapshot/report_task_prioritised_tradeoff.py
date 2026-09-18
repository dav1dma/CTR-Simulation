import json,hashlib,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import task_prioritised_tradeoff_search as q
h=q.h;s=q.s
if len(sys.argv)>1:q.OUT=q.OUT/sys.argv[1]
out=q.OUT
r=json.loads((out/'frozen_candidate.json').read_text());v=json.loads((out/'validation_records.json').read_text());d=np.array(r['design'])
keys=['global_ik_success','weak_ik_success','corridor_ik_success','trajectory_completion','trajectory_p95_mm','global_diagnostic_success','weak_diagnostic_success','corridor_diagnostic_success','global_isotropy_mean','global_isotropy_p10','weak_isotropy_mean','weak_isotropy_p10','corridor_isotropy_mean','corridor_isotropy_p10','volume_cm3','global_cell_retention','weak_cell_retention','corridor_cell_retention','tip_distance_max_mm','radial_extent_mm','z_max_mm','z_min_mm','global_ik_p95_mm','weak_ik_p95_mm','corridor_ik_p95_mm']
means={n:{k:float(np.mean([a['metrics'][k] for a in z])) for k in keys} for n,z in v.items()}
q.write('mean_metrics.json',means)
# Paired target bootstrap per fixed region and independent bank. Numerical sampling uncertainty only.
ci={};rng=np.random.default_rng(54109)
for g,name in enumerate(s.REGIONS):
 ds=[];ws=[]
 for rep in range(2):
  t=dict(np.load(out/f'validation_targets_{rep}.npz'));mask=t['groups']==g
  a=np.load(out/f'original_validation_{rep}.npz')['errors'][mask]<=.5
  b=np.load(out/f'proposed_validation_{rep}.npz')['errors'][mask]<=.5
  ds.append(b.astype(float)-a);ws.append(t['weights'][mask])
 boot=[]
 for _ in range(2000):
  vals=[]
  for delta,w in zip(ds,ws):
   ix=rng.integers(0,len(w),len(w));vals.append(np.average(delta[ix],weights=w[ix]))
  boot.append(np.mean(vals)*100)
 ci[name]=dict(mean_gain_pp=100*(means['proposed'][name+'_ik_success']-means['original'][name+'_ik_success']),paired_bootstrap_95pct_pp=np.percentile(boot,[2.5,97.5]))
q.write('paired_success_intervals.json',ci)
active=h.ROOT/'measured_hardware_simulator/optimised_configuration.json'
config=json.loads(active.read_text());config.update(name='Proposed benchmark-prioritised trade-off',status='Separate numerical candidate; see held-out validation; not installed or experimentally verified',total_length_mm=d[:3].tolist(),curved_length_mm=[0,*d[3:5]],straight_length_mm=(d[:3]-[0,*d[3:5]]).tolist(),precurvature_per_m=[0,*d[5:]],model_parameters=h.pars(d),study=str(out.relative_to(h.ROOT)),validation_accepted=json.loads((out/'outcome.json').read_text())['accepted'],validation_gate=json.loads((out/'protocol.json').read_text())['thresholds'])
q.write('proposed_configuration.json',config)
fig,ax=plt.subplots(1,2,figsize=(11,4.7))
labels=['Global','Weak region','Corridor','Paths'];ks=keys[:4];x=np.arange(4)
for i,(n,label,c) in enumerate([('original','Original','#4477aa'),('previous','Previous candidate','#aaaaaa'),('proposed','Proposed','#228877')]):
 ax[0].bar(x+(i-1)*.25,[100*means[n][k] for k in ks],.24,label=label,color=c)
ax[0].set(xticks=x,xticklabels=labels,ylabel='Success / completion (%)',ylim=(0,105));ax[0].legend(fontsize=8)
ks2=['weak_isotropy_p10','global_isotropy_mean','global_isotropy_p10','volume_cm3','tip_distance_max_mm']
change=[100*(means['proposed'][k]/means['original'][k]-1) for k in ks2]
ax[1].barh(['Weak-region P10','Global mean','Global P10','Occupancy volume','Maximum tip distance'],change,color=['#228877' if x>=0 else '#cc7744' for x in change]);ax[1].axvline(0,color='black',lw=.8);ax[1].set_xlabel('Proposed change from original (%)')
for j,x in enumerate(change):ax[1].text(x,j,f' {x:+.1f}%',va='center',ha='left' if x>=0 else 'right',fontsize=9)
ax[1].margins(x=.35);fig.suptitle('Hardware-constrained original tubes vs proposed trade-off\nIndependent ideal-model benchmarks; averages of two sample banks',fontsize=12);fig.tight_layout();fig.savefig(out/'tradeoff_summary.png',dpi=180);plt.close(fig)
lines=['# Benchmark-prioritised tube configuration study','','## Recommendation and scope','Prioritise reliable reaching in the declared target regions and continuous path completion, supported by weak-region positional dexterity. Treat global workspace volume and peak reach as secondary within explicit loss limits. The forward corridor and weak region are frozen mathematical benchmarks, not a surgical task or anatomy. Different task targets can change this recommendation.','',f"Candidate passes all predeclared held-out trade-off limits in both repeats: **{config['validation_accepted']}**. No claim of a global optimum or physical validation. The live simulator and previous configuration were preserved.",'','## Parameters','Order is inner/rear, middle/middle, outer/front. Lengths are provisional chuck-front-to-tip material lengths.','', '| Parameter | Inner | Middle | Outer |','|---|---:|---:|---:|',f"| Total (mm) | {d[0]:.4f} | {d[1]:.4f} | {d[2]:.4f} |",f"| Straight (mm) | {d[0]:.4f} | {d[1]-d[3]:.4f} | {d[2]-d[4]:.4f} |",f"| Curved (mm) | 0 | {d[3]:.4f} | {d[4]:.4f} |",f"| Precurvature (1/m) | 0 | {d[5]:.4f} | {d[6]:.4f} |",'','Digits reproduce the simulation; they are not manufacturing accuracy. Curved material longer than maximum exposure is not identifiable by this exterior model.','', '## Independent results','Rates below are fractions; lengths mm and volume cm³. Values average two independent banks. Raw repeats remain in validation_records.json.','', '| Metric | Original | Previous candidate | Proposed | Proposed − original |','|---|---:|---:|---:|---:|']
for k in keys:lines.append(f"| {k} | {means['original'][k]:.6f} | {means['previous'][k]:.6f} | {means['proposed'][k]:.6f} | {means['proposed'][k]-means['original'][k]:+.6f} |")
lines+=['','For success rates, multiply differences by 100 to obtain percentage points. Isotropy, volume and reach percentage changes use the baseline denominator.','', '## Paired uncertainty','Paired bootstrap intervals quantify finite target-sample uncertainty only; they do not quantify model error or manufacturing uncertainty.']
for g,a in ci.items():lines.append(f"- {g}: {a['mean_gain_pp']:+.2f} percentage points; paired 95% interval [{a['paired_bootstrap_95pct_pp'][0]:+.2f}, {a['paired_bootstrap_95pct_pp'][1]:+.2f}].")
lines+=['','## Method and limitations','The search screens prior candidate interpolations and new Sobol designs, uses equal target banks and the existing solver, then freezes one design before two new validation banks. Each bank has 2,048 global targets and 512 targets in each priority region, 131,072 feasible configurations per design, and 24 reference joint-space paths. Spatial metrics use a fixed 5 mm radial/axial annular occupancy grid and fixed original reference regions, weighted by annular volume. This rotational representation assumes unrestricted common rotation. Sampled occupancy and a smoothed envelope do not guarantee every enclosed point is reachable.','', 'Fixed-start IK uses 40 iterations and 0.5 mm success. Two nearest-start attempts are a separate diagnostic and do not prove geometric reachability or real-time control performance. Paths use warm starts, 21 waypoints and intermediate checks against continuous original reference paths. Only 48 paths are evaluated; identical completion is evidence for these paths, not all possible motions.','', 'Isotropy uses a fixed normalization and s_min/s_max of the positional Jacobian. It measures local directional balance, not orientation dexterity, force capability or feasible directional motion at a stop. Means, lower tails, coverage, target success and paths must be interpreted together.','', 'All 100 mm end stops, 71.5 mm actuator bodies and coupled 8.5–108 mm gaps are applied. The 108.5 mm sensitivity case is saved separately. Negative exposures and tips inside another tube are excluded, rather than silently clipped. This excludes some possibly physical configurations. Installed tube retention, internal guidance, rotation limits, friction, elastic torsion and stability, manufacturing constraints and loaded accuracy remain unresolved. Bounds are exploratory; the candidate is not fabrication-ready.','', 'The ranking and allowed losses are explicit engineering choices saved before selection, not externally established clinical requirements. Gains and losses describe the combined parameter change; this study does not attribute each gain to a particular length or curvature without a separate controlled ablation.','', '## Reproduction','From the project folder:','```sh','MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/task_prioritised_tradeoff_search.py search','MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/task_prioritised_tradeoff_search.py validate','MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/report_task_prioritised_tradeoff.py','```','Screen records are resumable. Keep previous study inputs at their recorded paths. Protocol, candidate selection, raw states/points/IK/path errors, sensitivity, and parameter JSON are saved alongside this report. Existing simulator configuration was not changed.']
(out/'tradeoff_report.md').write_text('\n'.join(lines)+'\n')
files=['tools/task_prioritised_tradeoff_search.py','tools/validate_tradeoff_reserve.py','tools/report_task_prioritised_tradeoff.py','tools/strict_measured_search.py','tools/optimise_measured_ctr.py','tools/run_broad_hardware_study.py','measured_hardware_simulator/design_constraints.py']
hashes={};snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
for f in files:
 p=h.ROOT/f
 if p.exists():hashes[f]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,snap/p.name)
hashes[str(active.relative_to(h.ROOT))]=hashlib.sha256(active.read_bytes()).hexdigest();q.write('source_hashes.json',hashes)
print(json.dumps(dict(design=d.tolist(),accepted=config['validation_accepted'],ci=ci),indent=2,default=lambda a:a.tolist()))
