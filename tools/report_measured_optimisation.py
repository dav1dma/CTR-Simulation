"""Render measured-design results; install only a validated improving candidate."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctr-mpl')
import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optimise_measured_ctr import OUT,ROOT,BASE,pars,limits,iso,map_values,wquant,cells

def main():
    selected=json.loads((OUT/'selected_frozen.json').read_text());d=np.array(selected['design'])
    data=json.loads((OUT/'validation_results.json').read_text());base=data['original'];opt=data['optimised']
    banks={name:dict(np.load(OUT/(name+'_validation.npz'))) for name in ('original','optimised')}
    ref=dict(np.load(OUT/'baseline_regions.npz'));regions={}
    for region,ids in [('global',ref['cells'][ref['counts']>=5]),('weak',ref['weak_cells']),('corridor',ref['corridor_cells'])]:
        w=2*(ids//1000)+1;regions[region]={}
        for name,bank in banks.items():
            sp={k:bank['workspace_'+k] for k in ('cells','medians','counts')};v=map_values(sp,ids)
            regions[region][name]=dict(mean_cell_median=float(np.average(v,weights=w)),p10_cell_median=wquant(v,w,.1),supported_fraction=float(np.average(v>0,weights=w)))
    (OUT/'regional_spatial_metrics.json').write_text(json.dumps(regions,indent=2))
    fig=plt.figure(figsize=(13,6));colors=['#3274b8','#168c80']
    for j,(name,bank) in enumerate(banks.items()):
        ax=fig.add_subplot(1,2,j+1,projection='3d');p=bank['workspace_points']
        ax.scatter(*p[::8].T,s=1,alpha=.10,color=colors[j],rasterized=True)
        ax.scatter([0],[0],[0],marker='+',s=90,color='black')
        ax.set(xlim=(-150,150),ylim=(-150,150),zlim=(0,210),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=name.title()+' tubes — measured coupled limits')
        ax.set_box_aspect((300,300,210));ax.view_init(24,-58)
    fig.suptitle('Sampled inner-tip workspace • original hardware • same sampling and scale')
    fig.text(.05,.02,'Approximate ideal exterior model; dots are retained sampled positions. No continuous interior-reach guarantee.',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'workspace_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    names=['Global','Weak region','Forward corridor'];x=np.arange(3)
    for j,name in enumerate(('original','optimised')):
        y=[100*data[name][g]['success'] for g in ('global','weak','corridor')]
        axes[0].bar(x+(j-.5)*.34,y,.34,label=name.title(),color=colors[j])
        y=[regions[g][name]['mean_cell_median'] for g in ('global','weak','corridor')]
        axes[1].bar(x+(j-.5)*.34,y,.34,color=colors[j])
        y=[data[name][g]['p95_mm'] for g in ('global','weak','corridor')]
        axes[2].bar(x+(j-.5)*.34,y,.34,color=colors[j])
    for ax in axes:ax.set_xticks(x,names,rotation=12);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Weighted success within 0.5 mm (%)');axes[0].legend(frameon=False)
    axes[1].set_ylabel('Mean cell-median positional isotropy');axes[2].set_ylabel('95th-percentile numerical residual (mm)')
    fig.suptitle('Independent validation • identical baseline targets • fixed-start IK')
    fig.tight_layout();fig.savefig(OUT/'performance_comparison.png',dpi=180);plt.close(fig)
    gain=max(opt[g]['success']-base[g]['success'] for g in ('global','weak','corridor'))
    success_gate=all(opt[g]['success']>=base[g]['success']-.02 for g in ('global','weak','corridor'))
    eligible=success_gate and gain>.005 and not np.allclose(d,BASE)
    configuration=dict(name='Optimised configuration — measured hardware',hardware_model='measured-ctr-20260914-v1',status='independently evaluated ideal-model candidate; not experimentally verified',
        tube_order=['inner/rear','middle/middle','outer/front'],total_length_mm=d[:3].tolist(),curved_length_mm=[0.,*d[3:5]],straight_length_mm=np.round(d[:3]-np.r_[0,d[3:5]],9).tolist(),precurvature_per_m=[0.,*d[5:7]],
        model_parameters=pars(d),hardware=json.loads((OUT/'protocol.json').read_text())['measured'],assumptions=json.loads((OUT/'protocol.json').read_text())['assumptions'],study=str(OUT.relative_to(ROOT)),
        validation_accepted=eligible,validation_gate='All regional fixed-start success losses <=2 percentage points and at least one gain >0.5 point; geometry differs. Additional coverage/dexterity/tail tradeoffs reported explicitly.')
    (OUT/'candidate_parameters.json').write_text(json.dumps(configuration,indent=2))
    rows='\n'.join(f'| {n} | {a:g} | {b:g} | {c:g} | {k:g} |' for n,a,b,c,k in zip(('Inner','Middle','Outer'),d[:3],d[:3]-np.r_[0,d[3:5]],np.r_[0,d[3:5]],np.r_[0,d[5:7]]))
    table='\n'.join(f"| {g} | {100*base[g]['success']:.2f}% | {100*opt[g]['success']:.2f}% | {base[g]['p95_mm']:.3f} | {opt[g]['p95_mm']:.3f} |" for g in ('global','weak','corridor'))
    spatial='\n'.join(f"| {g} | {regions[g]['original']['mean_cell_median']:.4f} | {regions[g]['optimised']['mean_cell_median']:.4f} | {regions[g]['original']['p10_cell_median']:.4f} | {regions[g]['optimised']['p10_cell_median']:.4f} |" for g in regions)
    ablation='\n'.join(f"| {name} | "+' | '.join(f"{100*data[name][g]['success']:.2f}%" for g in ('global','weak','corridor'))+' |' for name in data)
    lim=limits(d);elo=np.array(lim.lower_m)*1000;ehi=np.array(lim.upper_m)*1000
    travel_lo=elo-(d[:3]-[275,195,115]);travel_hi=ehi-(d[:3]-[275,195,115])
    travel='\n'.join(f'| {n} | {a:.2f}–{b:.2f} | {c:.2f}–{e:.2f} |' for n,a,b,c,e in zip(('Inner/rear','Middle','Outer/front'),elo,ehi,travel_lo,travel_hi))
    report=f'''# Measured-hardware tube configuration study

Selected candidate: {d.tolist()} in protocol variable order. Independent validation acceptance: {eligible}.
The original physical tubes remain the baseline. This is a bounded numerical search, not a global optimum or a fabrication certification.

## Parameters

| Tube | Chuck-to-tip total (mm) | Straight (mm) | Distal curved (mm) | Precurvature (1/m) |
|---|---:|---:|---:|---:|
{rows}

Diameters, 75 GPa model stiffness and straight inner wire retained. Three total lengths, two curved lengths and two precurvatures were explored. No stock-length assumption or 35 mm gripping offset was used. Positive behind-plate straight length is not proof of sufficient retention. Curved material within motors/guides is idealised as constrained straight internally; actual passage, support, friction, torsion, elastic stability and manufacturable curvatures are unresolved. The selected lengths describe distal material from the chuck; any gripping engagement or rear overhang must be specified separately.

The middle curved section ({d[3]:.2f} mm) exceeds its maximum exposure ({ehi[1]:.2f} mm). Any middle curved length between that exposure and the total length is exterior-equivalent in this model. The reported value is the searched representative, not a uniquely identified optimum. Two-decimal model parameters are not manufacturing tolerances.

| Actuator | Exterior material exposure range (mm) | Carriage position from rear stop (mm) |
|---|---:|---:|
{travel}

These are coordinate extrema over feasible coupled states, not independently available simultaneous strokes. The maximum middle-to-front gap remains only {d[1]-d[2]-71.5:.2f} mm under exterior tip ordering (original:18.5 mm). This candidate does not resolve that independent-motion restriction. The rear-to-middle gap is limited by ordering to {d[0]-d[1]-71.5:.2f} mm, so increasing the mechanical maximum to108.5 mm does not expand this candidate's exterior domain.

## Independent fixed-start IK

| Region | Original success | Candidate success | Original P95 (mm) | Candidate P95 (mm) |
|---|---:|---:|---:|---:|
{table}

Targets are identical, actual baseline-reachable samples in frozen baseline-defined cells; global, weak and corridor sets overlap spatially. This is a benchmark, not an anatomical task. Each set uses cell-stratified samples and annular-volume weights. There are 2,048 global, 512 weak and 512 corridor target draws. Success means <=0.5 mm residual, not physical accuracy. All failures remain in residual statistics. The 40-iteration solver starts at the most-retracted feasible exterior state with zero rotations for each design. This can be singular or close to constraints. These results must not be compared directly with previous chapters using different targets and obsolete hardware bounds. Batch solver equations match the viewer with the documented study settings; interactive solver defaults and its additional control features differ.

## Positional isotropy on fixed baseline cells

| Region | Original mean | Candidate mean | Original spatial P10 | Candidate spatial P10 |
|---|---:|---:|---:|---:|
{spatial}

Cell size 5 mm in radius and height, minimum five observations. Missing/unsupported cells score zero in the fixed reference. Scales [350,170,80,pi,pi,pi] are fixed across designs. Isotropy describes the ideal local Jacobian, not guaranteed available motion at a physical boundary. P10 of spatial cell medians is distinct from the success-weighted IK-solution metric used in selection. The latter includes zero for failed targets and can have zero P10 when failures exceed 10%.

## Attribution, same hardware throughout

| Configuration | Global success | Weak-region success | Corridor success |
|---|---:|---:|---:|
{ablation}

Lengths-only applies selected totals to original curved sections and curvatures. Curves-only applies selected curved sections and curvatures to original totals. Interactions mean these effects need not add. No increased-stroke benefit is claimed: all cases retain 100 mm carriage strokes, 71.5 mm bodies and 8.5–108 mm coupled gaps.

The two-nearest-start diagnostic is saved separately in validation_results.json. It uses a candidate-specific independent feasible sample bank and 60 iterations per start. It diagnoses local-solver sensitivity and does not prove geometric unreachability for failed targets. It is not mixed with primary fixed-start scores.

With that diagnostic, global success is {100*base['diagnostic']['global']['success']:.2f}% for original and {100*opt['diagnostic']['global']['success']:.2f}% for candidate. Thus improved fixed-start success should not be interpreted as an equally large geometric-coverage gain. Both designs solve every sampled weak-region and corridor target in this diagnostic.

## Workspace and sensitivities

Original 65,536-sample annular occupancy estimate: {base['workspace']['occupied_annular_volume_cm3']:.2f} cm³; candidate: {opt['workspace']['occupied_annular_volume_cm3']:.2f} cm³. Candidate retention of the frozen baseline's occupied cells: {100*opt['workspace']['baseline_cell_retention']:.2f}%. Occupancy depends on sample size and grid and does not fill holes or prove all enclosed points reachable. Full revolutions assume unrestricted axial rotations.

Raw physical states, tips, occupancy, target positions, returned IK states and errors are retained. sampling_sensitivity.json reports an independent 131,072-sample run at 2.5 and 5 mm grids. gap_sensitivity.json repeats fixed targets with 108.5 mm maximum gap; it is an alternate assumed stop setting, not a measured tolerance distribution. Material properties and internal guidance were not validated through this sensitivity test.

The trade-off is explicit: fixed-reference global mean isotropy and its lower spatial decile decrease, while weak-region and corridor averages improve. The changed workspace is not a superset of the original. The selection utility gives equal weight to the three regional success scores plus a smaller normalised weak/corridor solution-isotropy reward; these are exploratory engineering preferences, not clinically established weights. paired_sampling_uncertainty.json contains 2,000 paired bootstrap resamples; these intervals reflect target-sampling variation only, not physical or model uncertainty.

## Reproduce

From the project folder, run `.venv/bin/python tools/optimise_measured_ctr.py screen`, then `select`, then `refine`, then `fine`, then `validate`, then `verify_selected`; finally `.venv/bin/python tools/report_measured_optimisation.py`. Seeds, selection criteria and exploratory bounds are in protocol.json. The pre-alignment pilot is archived separately and is not final evidence. Selected parameters were frozen before independent validation. Source hashes identify the numerical code used. `demo_optimised.py` exercises the native viewer separately.

## Viewer

Launch `measured_hardware_simulator/Launch Optimised CTR.command`. F7 switches original/optimised, F6 switches the maximum-gap case, C switches independent/coordinated joint control. Switching resets the state and selects separate caches. The original tube parameter file and previous research results are retained. The selected JSON is the live optimised configuration; the obsolete hard-coded profile is backed up in legacy_before_replacement.
'''
    (OUT/'study_report.md').write_text(report)
    if eligible:
        shutil.copy2(OUT/'candidate_parameters.json',ROOT/'measured_hardware_simulator/optimised_configuration.json')
    print(json.dumps(dict(accepted=eligible,design=d.tolist(),success_gain_max=gain,regions=regions),indent=2))

if __name__=='__main__':main()
