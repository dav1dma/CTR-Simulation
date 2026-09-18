"""Reproduce the report and scientific figures for the completed local study."""
from run_hardware_optimisation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv


def run():
    shortlist=json.loads((OUT/'shortlist_frozen.json').read_text())
    rows=json.loads((OUT/'additional_validation.json').read_text())
    validation=json.loads((OUT/'validation.json').read_text())
    search=json.loads((OUT/'search_records.json').read_text())
    report=Path('output/hardware_optimisation');report.mkdir(parents=True,exist_ok=True)
    labels=['Baseline','Candidate A','Candidate B','Candidate C']
    _,volume_states=bank(262144,9501);volumes=[]
    for candidate in shortlist:
        p=fk(volume_states,candidate['k']); estimates={}
        for h in [1.25,2.5,5.]:
            ids=np.unique(cells(p,h));estimates[str(h)]=float(np.sum(2*(ids//1000)+1)*np.pi*h**3/1000)
        volumes.append(dict(label=candidate['label'],volume_cm3_by_cell_mm=estimates))
    jsonout('workspace_volumes.json',volumes)
    benchmark=json.loads((OUT/'random_benchmark.json').read_text())
    base=rows[0]['dense'];comparison=[]
    for name,c,r,v in zip(labels,shortlist,rows,validation):
        comparison.append(dict(design=name,middle_curvature_per_m=c['k'][0],outer_curvature_per_m=c['k'][1],middle_bend_radius_mm=1000/c['k'][0],outer_bend_radius_mm=1000/c['k'][1],near_axis_isotropy=r['dense']['isotropy'],isotropy_change_percent=100*(r['dense']['isotropy']/base['isotropy']-1),weakest_scaled_singular_value=r['dense']['weak'],sampled_workspace_retained_percent=100*r['dense']['retention'],near_axis_ik_success_percent=100*v['ik']['success_fraction'],baseline_target_success_percent=100*r['baseline_targets']['weighted_success'],max_path_error_mm=r['paths']['max_intermediate_error_mm']))
    for row,v in zip(comparison,volumes):
        row['estimated_workspace_volume_cm3']=v['volume_cm3_by_cell_mm']['2.5']
    with (report/'candidate_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=comparison[0]);writer.writeheader();writer.writerows(comparison)
    # Configuration landscape: show evaluated grid only, no misleading interpolation.
    grid=[r for r in search if r['method'] in ['grid','baseline']]
    fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained')
    sc=ax.scatter([r['k'][0] for r in grid],[r['k'][1] for r in grid],c=[r['isotropy'] for r in grid],s=75,cmap='viridis')
    for name,c in zip(labels,shortlist):
        ax.scatter(*c['k'],marker='*',s=180,facecolor='white',edgecolor='black',zorder=3)
        ax.annotate(name.replace('Candidate ',''),c['k'],xytext=(0,-18),textcoords='offset points',ha='center',fontsize=9)
    ax.set(xlabel='Middle-tube precurvature (m⁻¹)',ylabel='Outer-tube precurvature (m⁻¹)',title='Local tube-design search: near-axis positional isotropy')
    fig.colorbar(sc,ax=ax,label='Volume-weighted cell-median isotropy');fig.savefig(report/'search_landscape.png',dpi=180);plt.close(fig)
    # Paired spatial map uses exact same definition as quantitative validation.
    _,states=bank(262144,9501);near,w=region_cells();maps=[]
    for c in [shortlist[0],shortlist[1]]:
        p=fk(states,c['k']);m=(rz(p)[:,0]<10)&(p[:,2]>=40)&(p[:,2]<130)
        iso,_=values(states[m],c['k']);ids=cells(p[m]);maps.append(np.array([np.median(iso[ids==i]) for i in near]).reshape(4,36))
    fig,axes=plt.subplots(1,3,figsize=(12,5),layout='constrained')
    for ax,data,title in zip(axes,[*maps,maps[1]-maps[0]],['Baseline','Candidate A','A minus baseline']):
        diff=title.startswith('A minus');im=ax.imshow(data.T,origin='lower',extent=[0,10,40,130],aspect='auto',cmap='coolwarm' if diff else 'viridis',vmin=-.03 if diff else 0,vmax=.03 if diff else .3)
        ax.set(xlabel='Radial distance (mm)',ylabel='Z (mm)',title=title);fig.colorbar(im,ax=ax,label='Isotropy difference' if diff else 'Positional isotropy')
    fig.suptitle('Hardware-constrained near-axis capability • 262,144 configurations per design')
    fig.savefig(report/'near_axis_comparison.png',dpi=180);plt.close(fig)
    # Whole sampled r-z envelope, not a filled guaranteed reachable volume.
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for c,color,name in zip([shortlist[0],shortlist[1]],['#555555','#008e95'],['Baseline','Candidate A']):
        q=rz(fk(states[:16384],c['k']));ax.scatter(q[:,0],q[:,1],s=1,alpha=.2,c=color,label=name)
    ax.plot([0,10,10,0],[40,40,130,130],color='#b14a36',label='Test region boundary');ax.set(xlabel='Radial distance (mm)',ylabel='Z (mm)',title='Sampled hardware-constrained inner-tip workspace');ax.legend(markerscale=4)
    fig.savefig(report/'workspace_comparison.png',dpi=180);plt.close(fig)
    # Unassembled intrinsic shapes using the current model's bending convention.
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for j,ax in enumerate(axes):
        total=[170,80][j];curved=[90,65][j];straight=total-curved
        for c,name,col in zip([shortlist[0],shortlist[1]],['Baseline','Candidate A'],['#555555','#008e95']):
            k=c['k'][j]/1000;t=np.linspace(0,curved,150)
            ax.plot([0,0],[0,straight],color=col);ax.plot((1-np.cos(k*t))/k,straight+np.sin(k*t)/k,color=col,label=name)
        ax.set_aspect('equal');ax.set(title=['Middle tube','Outer tube'][j],xlabel='Lateral coordinate (mm)',ylabel='Axial coordinate (mm)');ax.legend()
    fig.suptitle('Intrinsic tube shapes before assembly • ideal constant precurvature')
    fig.savefig(report/'tube_shapes.png',dpi=180);plt.close(fig)
    table='\n'.join(f"| {r['design']} | {r['middle_curvature_per_m']:.4f} | {r['outer_curvature_per_m']:.4f} | {r['near_axis_isotropy']:.5f} | {r['isotropy_change_percent']:+.2f}% | {r['sampled_workspace_retained_percent']:.2f}% |" for r in comparison)
    more='\n'.join(f"- {name}: 1.25 mm cell isotropy {r['fine']['isotropy']:.5f}; shifted-grid isotropy {r['shifted_grid_isotropy']:.5f}; maximum interpolated trajectory error {r['paths']['max_intermediate_error_mm']:.5f} mm; local correction success {100*r['local_corrections']['strict_success_0_1mm']:.1f}% at 0.1 mm tolerance." for name,r in zip(labels,rows))
    text=f'''# Hardware-constrained tube optimisation: local search results

## Result

Candidate A (middle precurvature **21.032 m⁻¹**, outer **15.444 m⁻¹**) was the highest-scoring evaluated design on the training criteria and retains its lead on the dense independent spatial evaluation. Its volume-weighted mean of near-axis cell-median positional isotropy rises from **{base['isotropy']:.5f} to {rows[1]['dense']['isotropy']:.5f}**, an improvement of **{comparison[1]['isotropy_change_percent']:.2f}%**. This is a local ideal-model candidate, not a proven global optimum or a manufacturing-certified design.

The baseline already reached all tested central targets. The demonstrated gain concerns local positional kinematic capability; this study does not establish that the baseline had an inaccessible central hole. Both precurvatures of A lie at the upper search bounds. A broader optimum has not been investigated.

## Parameters and alternatives

| Design | Middle curvature (m⁻¹) | Outer curvature (m⁻¹) | Near-axis isotropy | Change | Sampled baseline workspace retained |
|---|---:|---:|---:|---:|---:|
{table}

All candidates retain assumed grip-to-tip lengths **350/170/80 mm**, curved lengths **0/90/65 mm**, outside diameters **0.50/0.70/0.90 mm**, inside diameters **0/0.62/0.80 mm**, and nominal modulus **75 GPa**. Tube order is inner/middle/outer. Candidate A corresponds to nominal curvature radii **{comparison[1]['middle_bend_radius_mm']:.2f} mm** and **{comparison[1]['outer_bend_radius_mm']:.2f} mm**.

B changes middle/outer curvature by +8%/+10%; C by +6%/+10%; A changes both by +10%. B and C are nearby alternatives with slightly less middle-tube curvature, not independently established mechanical improvements. C retained all sampled baseline cells in the dense evaluation, whereas A retained about 99.73%. This small cell-based distinction should not be overstated: all candidates reached all 605 independently generated baseline targets within tolerance. There is no compelling evidence here for a large practical trade-off requiring multiple designs.

Curved-section lengths were removed from the search because the entire exposed portions remain intrinsically curved throughout their ±10% ranges: middle maximum exposure 47 mm is below 81 mm, and outer maximum exposure 27.5 mm is below 58.5 mm. Their fixed baseline values are representative of a family indistinguishable in this exterior-only model, not uniquely optimal lengths. Changing them could still matter to the unmodelled internal mechanism.

## Hardware approximation and study scope

- User-supplied rear-motor-face to front-plate distances: back 347→250 mm, middle 267→158 mm, front 187→87.5 mm.
- User-authorised approximate grip: 35 mm forward of rear motor face (midpoint of a 70 mm distance).
- Assumed back/middle/front actuator mapping to inner/middle/outer elements.
- Calculated travels: **97/109/99.5 mm**. Assumed installed lengths give inner exposure **38–135 mm**, middle signed exposure **−62–47 mm**, outer signed exposure **−72–27.5 mm**.
- This study restricts middle and outer tips to the plate or beyond; it does not simulate negative exterior deployment. Nested endpoint ordering is enforced for every sampled and solved configuration.
- Task region fixed before search: **r ≤ 10 mm; 40 ≤ Z ≤ 130 mm**, inner tip, no orientation requirement. The user did not provide a different region in response to the optional clarification, so this is a provisional engineering region, not anatomy-derived.
- Search bounds: **±10% of baseline precurvatures**, fixed other properties. These are computational bounds, not verified mould or material limits. The original reported curvature-unit ambiguity remains; this run uses the existing model interpretation in m⁻¹.
- Assumed straight internal path, grip-to-tip length interpretation, no verified clamping engagement, end-stop margins, guide interference, or rotational stops. External stroke compatibility is approximate. Precurved portions behind the plate and their guidance are not modelled.

## Search method and reproducibility

The search used 16,384 shared scrambled Sobol actuator configurations per candidate, an 11×11 curvature grid, and six short differential-evolution runs (three diagnostic criterion mixes, two seeds). It evaluated **{len(search)} unique designs**. A full grid is particularly useful after the design space reduces to two effective variables. Coverage and local kinematic capability were evaluated in fixed 2.5 mm radial–axial cells, with exact annular volume weighting and at least five configurations for a cell median. Unsupported cells receive zero capability. The retention constraint was 95% of the frozen sampled baseline workspace.

The objectives used positional isotropy and weakest scaled singular value with the existing immutable baseline scale [350,170,80,π,π,π]. This preserves a consistent convention across designs; it is not measured actuator velocity or force capability. In this restricted straight-inner-element model, the largest singular value stays approximately 350, making isotropy and weakest singular value nearly proportional. They therefore do not provide independent evidence of two different gains.

An equal-budget random benchmark evaluated {benchmark['evaluations']} additional designs using the same training evaluator; its best isotropy was {benchmark['best']['isotropy']:.5f}, below A's training score 0.14257. This supports the search outcome but does not prove global optimality. A post-search rank diagnostic across 12 spread-out grid designs gave Spearman correlation 1.0 between training and denser evaluation.

The shortlist was written before independent validation and was not retuned afterwards. The historical five-objective Stage-5A protocol was preserved and was not executed as if this were its completed Stage-5C/5D study. This is a separate local study. The rank-stability check was performed after the search as a diagnostic, not as a pre-search approval gate. Existing legacy baseline results remain unchanged and should not be combined numerically with this hardware-constrained baseline.

## Validation completed

- Accelerated batched FK agrees with the original sectioned FK on 192 sampled states across baseline and curvature-bound designs to approximately 3×10⁻¹⁴ mm; source FK was also checked on solved finalist configurations.
- Straight-element and zero-curvature analytic cases, rotational equivariance, hardware bounds, nested ordering, Jacobian agreement with the original implementation, finite-difference step refinement, and independent IK recovery checks passed.
- New shared configuration samples at 32,768, 131,072 and 262,144 states per design; grid resolutions 1.25, 2.5 and 5 mm; a half-cell shifted grid.
- **512 independent volume-uniform cylinder targets per design:** all reached within 0.1 mm (and hence within 0.5 mm).
- **91 exact centreline targets per design:** all reached within 0.5 mm. An explicit straight-inner-only path also confirms exact centreline access from Z=40 to 130 mm in the ideal model.
- **605 independent known-reachable baseline targets**, one per occupied radial–axial cell with annular weighting: all reached by all four designs within 0.5 mm. This is sampled retention evidence, not a proof for every possible point.
- Five continuous 90 mm lines (centreline and four lines offset by 5 mm), 1 mm waypoints, ten interpolation subdivisions, feasible previous-state initialisation without retracting or restarting at each waypoint.
- Stronger IK restart budgets on a subset; local ±0.5 mm corrections in six Cartesian directions; independent ±2% curvature sensitivity scenarios.

{more}

The mean isotropy gain is consistent across these spatial sampling/resolution checks. Per-cell bootstrap intervals in the raw results describe conditional finite-cell variability, not clinical confidence, physical accuracy or uncertainty over all possible robot models.

## Estimated total workspace volume

At 2.5 mm radial–axial cells and 262,144 configurations, baseline occupied-cell volume was **{volumes[0]['volume_cm3_by_cell_mm']['2.5']:.2f} cm³**, and A's was **{volumes[1]['volume_cm3_by_cell_mm']['2.5']:.2f} cm³**. This is a sampling- and resolution-dependent estimate. The companion raw volume table reports 1.25 and 5 mm grids as well; none is an exact continuous-workspace volume.

## Mounting uncertainty and limitations

At assumed gripping offsets 30 and 35 mm, all 128 offset-sensitivity targets were reached by every design. At 40 mm, each achieved **98.4375%**; the two failed targets were near the low-Z edge and their residuals were around 2.2 mm. This reflects an important sensitivity to the provisional installed geometry. The 30–40 mm range is an assumed sensitivity range, not a measured tolerance.

The current stiffness-superposition model does not include torsional stability, friction, loaded deflection, buckling or internal guidance. Increasing precurvature may matter to those behaviours. Neither physical fit nor mechanical safety has been verified. More accurate grip/guide geometry, a torsionally compliant model and eventually physical experiments are required before treating A as a build-ready recommendation.

## Files

- `candidate_comparison.csv`: machine-readable parameter and result table.
- `search_landscape.png`: evaluated local design landscape.
- `near_axis_comparison.png`: baseline, A and difference maps.
- `workspace_comparison.png`: sampled exterior workspace comparison.
- `tube_shapes.png`: intrinsic tube shapes before assembly.
- Raw records and arrays: `results/hardware_optimisation_20260908/`.
- Reproduction: `tools/run_hardware_optimisation.py`, `tools/validate_hardware_candidates.py`, `tools/report_hardware_optimisation.py`.

Production tube parameters and the interactive simulator have not been changed to the selected candidate.
'''
    (report/'optimisation_results.md').write_text(text)
    jsonout('candidate_comparison.json',comparison)
    files=[Path('tools/run_hardware_optimisation.py'),Path('tools/validate_hardware_candidates.py'),Path('tools/report_hardware_optimisation.py'),Path('tube_parameters.py'),Path('CTR_superPosKin_fun_sectioned.py'),OUT/'shortlist_frozen.json',OUT/'search_records.json',OUT/'validation.json',OUT/'additional_validation.json']
    jsonout('manifest.json',dict(study='midpoint-hardware-local-curvature-20260908',scope='conditional ideal-model study',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    print(json.dumps(comparison,indent=2))

if __name__=='__main__':run()
