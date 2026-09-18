"""Summarise frozen independent validation; no design retuning or production edits."""
from run_broad_hardware_study import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEST=Path('output/comparative_study');DEST.mkdir(parents=True,exist_ok=True)
chosen=json.loads((OUT/'shortlist_frozen.json').read_text())[:3]
metrics=json.loads((OUT/'comparison.json').read_text())[:3]
names=['Original','Candidate A','B1'];colors=['#687681','#3588a1','#bd642e']
_,states=h.bank(262144,13301)
fields=[];res=[]
for c in chosen:
 fields.append(spatial_data(c['design'],states))
 res.append(np.load(OUT/(c['label']+'_validation.npz')))
fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharex=True,sharey=True,layout='constrained')
for ax,c,name in zip(axes,chosen,names):
 p=forward(states,c['design']);rz=h.rz(p);counts,rr,zz=np.histogram2d(rz[:,0],p[:,2],bins=[np.arange(0,102.5,2.5),np.arange(0,140,2.5)])
 ax.pcolormesh(rr,zz,np.ma.masked_where(counts.T==0,np.ones_like(counts.T)),cmap=matplotlib.colors.ListedColormap(['#3588a1']),vmin=0,vmax=1)
 ax.plot([0,10,10,0],[40,40,130,130],color='#ee9435');ax.set(title=name,xlabel='Radius from Z axis (mm)',xlim=(0,100),ylim=(0,135))
axes[0].set_ylabel('Z from front plate (mm)');fig.suptitle('Sampled reachable tip workspace • orange outline: near-axis test region');fig.savefig(DEST/'workspace_comparison.png',dpi=180);plt.close(fig)
# Common spatial bins, each with >=5 configurations per design.
common=fields[0]['ids'][fields[0]['counts']>=5]
for f in fields[1:]:common=np.intersect1d(common,f['ids'][f['counts']>=5])
common=common[(common//1000<40)&(common%1000<54)&(common%1000>=0)]
med=np.array([f['medians'][np.searchsorted(f['ids'],common)] for f in fields])
fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True,layout='constrained');lim=max(abs(med[2]-med[0]).max(),abs(med[2]-med[1]).max());spatial={}
for ax,j in zip(axes,[0,1]):
 delta=med[2]-med[j];grid=np.full((40,54),np.nan);grid[common//1000,common%1000]=delta
 im=ax.imshow(grid.T,origin='lower',extent=[0,100,0,135],aspect='auto',cmap='RdBu',vmin=-lim,vmax=lim)
 ax.set(title='B1 minus '+names[j],xlabel='Radius from Z axis (mm)');w=2*(common//1000)+1
 spatial[names[j]]={'supported_common_cells':len(common),'volume_weighted_fraction_improved':float(np.average(delta>1e-8,weights=w)),'volume_weighted_fraction_worse':float(np.average(delta< -1e-8,weights=w))}
axes[0].set_ylabel('Z from front plate (mm)');fig.colorbar(im,ax=axes,label='Difference in cell-median positional isotropy');fig.suptitle('Where dexterity improves or worsens • blank = insufficient common support');fig.savefig(DEST/'dexterity_difference.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained');summaries=[]
for ax,key,title in zip(axes,['near_errors','baseline_errors','axis_errors'],['Near-axis targets (1,024)','Baseline targets (593)','Centreline targets (91)']):
 for name,color,r in zip(names,colors,res):
  err=r[key].ravel();ax.step(np.maximum(np.sort(err),1e-12),np.arange(1,len(err)+1)/len(err),where='post',label=name,color=color)
  summaries.append(dict(design=name,target_set=key,n=len(err),median_mm=float(np.median(err)),p95_mm=float(np.percentile(err,95)),maximum_mm=float(err.max()),success_at_0_1mm=float(np.mean(err<=.1))))
 ax.set(xscale='log',xlabel='Numerical residual (mm)',title=title,ylim=(0,1.01));ax.grid(alpha=.2)
axes[0].set_ylabel('Fraction of targets at or below residual');axes[-1].legend();fig.suptitle('Identical targets and solver budgets • smaller residual is to the left');fig.savefig(DEST/'ik_residual_distributions.png',dpi=180);plt.close(fig)
ia,_=h.values(res[1]['near_states'],chosen[1]['design']);ib,_=h.values(res[2]['near_states'],chosen[2]['design'])
paired=dict(n=len(ia),B1_higher=int(np.sum(ib>ia+1e-8)),B1_lower=int(np.sum(ib<ia-1e-8)),interpretation='At the saved IK solutions; not optimal achievable isotropy at each target.')
summary=dict(primary_metrics=metrics,spatial_comparison=spatial,paired_near_target_dexterity=paired,residual_summaries=summaries)
(DEST/'comparison_data.json').write_text(json.dumps(summary,indent=2))
rows='\n'.join(f"| {n} | {m['volume_cm3']:.1f} | {m['overall_isotropy']:.5f} | {m['near_isotropy']:.5f} | {m['path_max_error_mm']:.6f} |" for n,m in zip(names,metrics))
errs='\n'.join(f"| {s['design']} | {s['target_set'].replace('_errors','')} | {s['median_mm']:.3g} | {s['p95_mm']:.3g} | {s['maximum_mm']:.3g} |" for s in summaries)
text=f'''# Comparative study: original tubes, Candidate A and B1

Prepared 9 September 2026. This is a separate hardware-constrained numerical comparison, extending the completed exploratory optimisation. It does not replace the historical Stage 4/4.1 results or claim completion of the old Stage 5 protocol. No production parameters were changed.

## Research question and comparison method

Does the selected broader-search design increase reachable workspace and positional dexterity while preserving target reaching and continuous axial motion under the same estimated hardware constraints?

All three designs share the 35 mm assumed grip offset, actuator limits, nested deployment ordering, total lengths, diameters, material model, Jacobian scaling, target positions and solver budgets. Curvature and curved lengths differ as recorded in the accompanying data. Primary workspace/dexterity evidence uses 262,144 independent shared configurations, seed 13301, and 2.5 mm radial–axial cells. Candidates were frozen before validation; this package performs post-hoc descriptive analysis of those saved validation results, not a new untouched confirmatory experiment. No candidate is retuned here.

Workspace is occupied swept-cell volume within radius 100 mm and Z=0–135 mm. Overall isotropy is evaluated on the frozen original-design reference region; near-axis isotropy uses radius 10 mm and Z=40–130 mm. These positional metrics do not measure orientation dexterity, stiffness or physical accuracy.

## Main results

| Design | Workspace (cm³) | Overall isotropy | Near-axis isotropy | Maximum path residual (mm) |
|---|---:|---:|---:|---:|
{rows}

![Workspace](workspace_comparison.png)

Figure 1. Occupied radial–axial cells from the same sampling budget. Rotating the section about Z gives the model's axisymmetric volume under unrestricted rotation. Filled cells are a discretised estimate, not proof that every contained point is reachable. The orange outline is the prescribed near-axis task region.

![Dexterity changes](dexterity_difference.png)

Figure 2. B1 minus comparator cell-median positional isotropy. Blue indicates improvement; red indicates deterioration. Each displayed cell has at least five configuration samples for all three designs; blank regions are excluded from comparison. This common-support map is a supplementary analysis, distinct from the primary frozen-region score. It compares distributions of configurations in spatial cells, not identical configurations or best possible IK solutions.

At identical near-axis targets, B1 has higher isotropy in {paired['B1_higher']}/{paired['n']} saved solutions and lower isotropy in {paired['B1_lower']}/{paired['n']}. This shows that average improvement is not uniform improvement. Alternative IK solutions can change the comparison.

## IK and continuous control

![Residual distributions](ik_residual_distributions.png)

Figure 3. Empirical residual distributions for identical targets. Each target counts equally in this supplementary figure. The historical dissertation's fixed-start IK protocol differs from this bounded multistart solver; their residual distributions must not be pooled or directly ranked.

| Design | Target set | Median residual (mm) | 95th percentile (mm) | Maximum (mm) |
|---|---|---:|---:|---:|
{errs}

All three designs passed every near-axis, centreline and sampled baseline target within 0.5 mm. All passed five 90 mm axial paths, interpolation checks and local ±0.5 mm correction checks. Residual improvements are mixed and numerically tiny; they are not the principal optimisation claim. There is no evidence here of better measured robot accuracy.

## Validation already completed

The broader study checked model/Jacobian agreement, independent sample seeds, sample sizes, finer cells, stronger IK restarts on a shared subset, intermediate path feasibility and assumed mounting/curvature/length perturbations. These support numerical interpretation within the specified model. A 40 mm grip-offset sensitivity case lost two of 128 targets for every design; fit remains conditional on the approximate mounting.

## How to use this in the dissertation

Add a separate tube-configuration comparison subsection after the existing baseline numerical results. Present Figure 1 and the main table first, Figure 2 second, and use Figure 3 or its table to establish retained numerical target-reaching performance. Keep the other Pareto candidates and detailed sensitivity results in the appendix. Describe B1 as a selected candidate found within the stated bounds, not globally optimal or build-ready.

A defensible conclusion is that B1 expands predicted workspace and improves average positional isotropy relative to Candidate A while retaining the tested target-reaching and continuous position-control performance; some local dexterity values deteriorate and IK residuals do not improve uniformly.

## Remaining work, in priority order

1. Decide whether position-only forward access matches the intended task. If the tip must point along +Z or maintain an instrument orientation, add an explicitly defined orientation tolerance and constrained pose/path tests before claiming that capability.
2. Confirm installed grip-to-tip lengths, clamps, guides, end margins, rotational limits and supplier-approved curvature. The numerical search bounds are exploratory, not manufacturer-certified.
3. If comparison with the old fixed-start IK method is required, run a separate hardware-feasible fixed-start experiment for all three designs with a frozen common target set and identical budgets. The current multistart results answer a different question.
4. For physical performance claims, evaluate torsion/stability and loaded behaviour, then obtain experimental tip tracking and repeated trials. Numerical residual cannot substitute for this.

The numerical comparison can be reported now with these boundaries. Further parameter optimisation is not necessary to begin writing it.

## Provenance

Sources: results/broad_hardware_optimisation_20260908/shortlist_frozen.json, comparison.json, *_validation.npz, validation_targets.npz, robustness.json and the broader study report. Reproduce this package with tools/build_comparative_study.py using the project Python environment. comparison_data.json contains the tabulated numerical summaries. Existing raw results are preserved.
'''
(DEST/'comparative_study.md').write_text(text)
print(json.dumps({'output':str(DEST),'spatial':spatial,'paired':paired},indent=2))
