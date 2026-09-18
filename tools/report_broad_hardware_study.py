"""Generate a concise report and charts from the frozen broad study."""
from run_broad_hardware_study import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'baseline':'Original','candidate_A':'Candidate A','maximum_workspace':'B1: workspace','overall_dexterity':'B2: overall','near_axis_dexterity':'B3: near-axis','balanced':'B4: compromise'}


def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    chosen=json.loads((OUT/'shortlist_frozen.json').read_text())
    validation=json.loads((OUT/'validation.json').read_text())
    robustness=json.loads((OUT/'robustness.json').read_text())
    records=json.loads((OUT/'search_records.json').read_text())
    refinement=json.loads((OUT/'pareto_refinement.json').read_text())
    comparisons=[];base=robustness[0]['results'][0];a=robustness[1]['results'][0]
    for c,v,r in zip(chosen,validation,robustness):
        f=r['results'][0];d=c['design']
        comparisons.append(dict(design=LABELS[c['label']],raw_label=c['label'],middle_curvature_per_m=d[0],outer_curvature_per_m=d[1],middle_curved_length_mm=d[2],outer_curved_length_mm=d[3],volume_cm3=f['volume_cm3'],volume_change_vs_original_percent=100*(f['volume_cm3']/base['volume_cm3']-1),overall_isotropy=f['overall_isotropy'],overall_change_vs_original_percent=100*(f['overall_isotropy']/base['overall_isotropy']-1),near_isotropy=f['near_isotropy'],near_change_vs_original_percent=100*(f['near_isotropy']/base['near_isotropy']-1),overall_p10=f['overall_p10_isotropy'],retention_percent=100*f['retention'],near_target_success_percent=100*v['targets']['near_success'],baseline_target_success_percent=100*v['targets']['baseline_weighted_success'],path_max_error_mm=v['paths']['max_interpolated_error_mm'],control_pass=v['passes_control_checks']))
    with (REPORT/'candidate_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=comparisons[0]);writer.writeheader();writer.writerows(comparisons)
    params=[]
    for c in chosen:params.append(dict(label=LABELS[c['label']],parameters=design_object(c['design']).as_dict(),hardware_approximation=CONFIG,status='exploratory; not manufacturing certified'))
    (REPORT/'candidate_parameters.json').write_text(json.dumps(params,indent=2))
    write('comparison.json',comparisons)
    # Search landscape is training evidence; validation chart below is independent.
    fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained')
    feasible=[r for r in records if r['valid']];bad=[r for r in records if not r['valid']]
    if bad:ax.scatter([r['volume_cm3'] for r in bad],[r['overall_isotropy'] for r in bad],color='#cccccc',s=12,label='Fails a numerical task constraint')
    sc=ax.scatter([r['volume_cm3'] for r in feasible],[r['overall_isotropy'] for r in feasible],c=[r['near_isotropy'] for r in feasible],cmap='viridis',s=22)
    ax.set(xlabel='Estimated workspace volume (cm³)',ylabel='Overall positional isotropy',title='Broad search • 516 distinct evaluated designs')
    fig.colorbar(sc,ax=ax,label='Near-axis isotropy');ax.text(.02,.97,'Training evaluation: 8,192 configurations/design',transform=ax.transAxes,va='top',fontsize=9)
    if bad:ax.legend(loc='lower right',fontsize=8)
    fig.savefig(REPORT/'search_tradeoffs.png',dpi=180);plt.close(fig)
    # Gains in separate panels to avoid treating percentages of different metrics as equivalent.
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
    names=[r['design'] for r in comparisons];colors=['#65717b','#62a6b1','#076c77','#478d95','#bd8750','#8b6bb1']
    for ax,key,title in zip(axes,['volume_cm3','overall_isotropy','near_isotropy'],['Workspace volume (cm³)','Overall positional isotropy','Near-axis positional isotropy']):
        vals=[r[key] for r in comparisons];ax.barh(names,vals,color=colors);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.set_xlim(0,max(vals)*1.22)
        for i,val in enumerate(vals):ax.text(val+max(vals)*.02,i,f'{val:.0f}' if key=='volume_cm3' else f'{val:.4f}',va='center',fontsize=8)
    fig.suptitle('Independent comparison • 262,144 configurations/design')
    fig.savefig(REPORT/'validated_comparison.png',dpi=180);plt.close(fig)
    # Exact same near-axis region; no interpolation into empty cells.
    _,states=h.bank(131072,13601);near,w=h.region_cells();maps=[]
    for c in [chosen[0],chosen[1],chosen[2]]:
        p=forward(states,c['design']);m=(h.rz(p)[:,0]<10)&(p[:,2]>=40)&(p[:,2]<130)
        iso,_=h.values(states[m],c['design']);ids=h.cells(p[m]);maps.append(np.array([np.median(iso[ids==i]) if np.sum(ids==i)>=5 else np.nan for i in near]).reshape(4,36))
    fig,axes=plt.subplots(1,3,figsize=(11,5),layout='constrained')
    for ax,data,title in zip(axes,maps,['Original','Candidate A','B1: workspace candidate']):
        im=ax.imshow(data.T,origin='lower',extent=[0,10,40,130],aspect='auto',vmin=0,vmax=.4,cmap='viridis');ax.set(xlabel='Radial distance (mm)',ylabel='Z (mm)',title=title)
    fig.colorbar(im,ax=axes,label='Cell-median positional isotropy');fig.suptitle('Near-axis capability • common colour scale')
    fig.savefig(REPORT/'near_axis_maps.png',dpi=180);plt.close(fig)
    ptable='\n'.join(f"| {r['design']} | {r['middle_curvature_per_m']:.4f} | {r['outer_curvature_per_m']:.4f} | {r['middle_curved_length_mm']:.3f} | {r['outer_curved_length_mm']:.3f} |" for r in comparisons)
    rtable='\n'.join(f"| {r['design']} | {r['volume_cm3']:.1f} | {r['overall_isotropy']:.5f} | {r['near_isotropy']:.5f} | {r['retention_percent']:.2f}% |" for r in comparisons)
    detail='\n'.join(f"- {LABELS[r['label']]}: independent-seed near-axis score {r['results'][1]['near_isotropy']:.5f}; finer-grid score {r['fine_grid']['near_isotropy']:.5f}; stronger-solver classification changes {r['stronger_solver']['classification_changes']}/128." for r in robustness)
    b1=comparisons[2];f1=robustness[2]['results'][0]
    report=f'''# Broader tube-design optimisation: results and trade-offs

## Main finding

The broader search found designs that outperform Candidate A in the tested ideal-model metrics. **B1 is a strong practical comparison candidate within this study**, with middle/outer precurvatures **28.68/21.06 m⁻¹** and curved lengths **47/27.5 mm**. The wording "practical comparison candidate" refers to simplicity of numerical specification, not verified physical manufacture or fit.

Compared with the original tubes in the same hardware-constrained evaluation, B1 increases estimated workspace volume by **{b1['volume_change_vs_original_percent']:.1f}%**, overall positional isotropy by **{b1['overall_change_vs_original_percent']:.1f}%**, and near-axis isotropy by **{b1['near_change_vs_original_percent']:.1f}%**. Compared with Candidate A, the changes are **{100*(f1['volume_cm3']/a['volume_cm3']-1):.1f}%**, **{100*(f1['overall_isotropy']/a['overall_isotropy']-1):.1f}%**, and **{100*(f1['near_isotropy']/a['near_isotropy']-1):.1f}%**, respectively.

All frozen finalists passed the specified numerical target and control checks. Small differences among B1, B2 and B4 should not be read as decisive superiority: the fine ranking changes with sampling. The candidate selected for highest near-axis training score did not consistently retain that lead in independent evaluation. There is a strong gain over the original and A, but much weaker evidence for a uniquely best member of the new group.

**This is still a bounded exploratory search, not a proven global optimum.** All new finalists put outer curvature at the +50% upper bound. Further numerical improvements may exist, but supplier-approved shape-setting bounds, internal guidance and a torsional stability model remain missing. The experiment therefore does not justify simply manufacturing the largest-curvature design.

## Parameters

| Design | Middle curvature (m⁻¹) | Outer curvature (m⁻¹) | Middle curved length (mm) | Outer curved length (mm) |
|---|---:|---:|---:|---:|
{ptable}

Names B1–B4 identify the frozen shortlist; their descriptions record their **training selection role**, not a guarantee they win that metric in validation. Displayed values are rounded; the machine-readable parameter file preserves the evaluated values.

All designs retain assumed grip-to-tip lengths **350/170/80 mm**, OD **0.50/0.70/0.90 mm**, ID **0/0.62/0.80 mm**, straight inner element, and nominal modulus **75 GPa**. No production simulator parameters were changed.

At the nominal mounting, curved lengths at or above **47 mm middle and 27.5 mm outer** are exterior-equivalent in this model. B1's 47/27.5 mm values therefore do not prove those lengths uniquely optimal. Retaining 90/65 mm at the same B1 curvatures gives the same nominal exterior kinematics, verified numerically; internal guidance and mounting-offset sensitivity can differ, so it is not an unconditional mechanical equivalence.

## Independent numerical comparison

Primary table: 262,144 new shared actuator configurations, 2.5 mm radial–axial cells. Overall isotropy is the volume-weighted mean of cell medians over a **common frozen baseline region**, with uncovered/under-supported cells set to zero. Near-axis isotropy uses the fixed cylinder r≤10 mm, Z=40–130 mm. Workspace is estimated occupied volume within r≤100 mm, Z=0–135 mm. These are engineering regions, not anatomy-derived requirements.

| Design | Workspace (cm³) | Overall isotropy | Near-axis isotropy | Frozen baseline cells retained |
|---|---:|---:|---:|---:|
{rtable}

Independent samples can miss a few baseline boundary cells; even the baseline's observed occupancy may be below 100% of the frozen discovery region. This is sampling support, not proof of physical loss of baseline reach. Volume estimates depend on sampling and cell size. They must not be compared directly with the older unconstrained report workspace or a differently sampled earlier run.

## How the balance was handled

Hard numerical requirements were at least 95% baseline-region retention, at least 99% sampled near-axis coverage, and at least 95% of baseline near-axis mean isotropy. Candidates competed on three separate objectives: useful-region volume, common-region overall isotropy, and near-axis isotropy. Non-dominated sorting retained designs for which improving one objective required sacrificing another; crowding kept the search spread out. Tip-control tests were final acceptance checks rather than a weighted residual-error objective.

The balanced candidate was selected by maximising its weakest min–max normalised objective over the refined front. This is an explicitly chosen numerical compromise rule, not an objective clinical preference. The final recommendation should consider modelling uncertainty and manufacturing effort as well as small score differences.

The inherited Jacobian input scale [350,170,80,π,π,π] is held fixed across designs. These isotropy scores describe local positional direction balance under that convention, not force, tip orientation or calibrated physical actuator speed. Weakest-direction singular values and lower spatial deciles are also saved in the raw data.

## Search and checks

- Generalised batch FK includes exposed straight-to-curved section transitions. It was checked against the original sectioned FK on 320 states across eight designs; maximum discrepancy was below 10⁻⁸ mm. Jacobians, transition continuity, zero curvature, ordering, hardware limits and known-target IK checks passed.
- Bounds: middle curvature 9.56–28.68 m⁻¹; outer 7.02–21.06 m⁻¹; middle curved length 10–47 mm; outer 5–27.5 mm. Curvature ranges are ±50% exploratory bounds, not supplier-approved limits.
- Pilot: 18 designs at 8,192 and 32,768 configurations. Objective rank correlations exceeded 0.99, passing the predefined 0.85 pilot criterion.
- Search: **516 unique designs**, including 128 space-filling designs, boundary controls and two seeded evolutionary phases with non-dominated/crowding selection. The second phase continued from accumulated results; these are not claimed as two independent complete searches.
- Up to 16 Pareto candidates were refined with 65,536 training configurations. The final baseline region and shortlist were fixed before independent validation. The baseline discovery region was refined at this stage; that change preceded validation.
- Validation: 32,768, 131,072 and 262,144 configuration samples; an additional independent sample seed; and a 1.25 mm grid sensitivity comparison. No designs were retuned after validation.
- Every design reached all **1,024 near-axis targets**, **91 centreline targets**, and **593 independent baseline targets** within 0.5 mm. All near-axis targets also met 0.1 mm.
- Every design passed five 90 mm straight-line paths with 1 mm waypoints and ten interpolation subdivisions, positive sampled axial progress and feasible actuator ordering. These are position paths; pointing direction was not constrained.
- Every design passed the local ±0.5 mm correction checks at 0.1 mm tolerance. Path tracking errors are ideal-model residuals and are not physical accuracy claims.
- 512 volume-uniform targets in the fixed broad region provide a second, independent coverage comparison. That region intentionally includes unreachable locations; its success fraction is not the IK success rate over known-reachable targets. Stronger restarts were checked on the same 128-target subset for every design.
- Sensitivity cases included grip offsets 30/40 mm, independent ±2% curvature perturbations and ±1 mm curved-length changes. These are assumed perturbations, not measured production tolerances.

{detail}

The gain over A survives these checks, while the exact ranking of closely spaced finalists should be treated cautiously. Bootstrap intervals saved in robustness data describe conditional spatial-cell variability, not uncertainty in the physical model.

## Hardware and physical limits

The agreed midpoint mounting uses a grip 35 mm forward of each rear motor face. It gives carriage travels **97/109/99.5 mm** and nominal exterior deployment limits **38–135/0–47/0–27.5 mm**, enforcing inner≥middle≥outer. Middle/outer tips behind the plate are excluded from this exterior-only study.

At a 40 mm assumed grip offset, each design reaches 126 of 128 sensitivity targets; two near the low-Z boundary fail. The approximate mounting affects feasible access. Clamp engagement, end margins, rotary stops, shaft/guide clearance and actual installed tube lengths remain unverified.

The model does not account for torsional instability, friction, internal straightening/guidance, loaded deflection or experimental calibration. Larger precurvatures increase the importance of checking those omissions. Original curvature units remain interpreted as m⁻¹ pending confirmation. Supplier capability ranges do not certify a specific shape-set tube.

## Recommended use

Use **Original, Candidate A and B1** for the main dissertation comparison. B2 and B4 can be reported as close alternatives, with B3 illustrating that a training near-axis optimum may not retain its apparent advantage. Preserve all results and the bounds so the claim remains "best candidates found in the stated model and search domain".

Before treating B1 as build-ready, confirm feasible shape-setting radii and lengths, the internal guide/clamp geometry, and torsional/physical behaviour. The numerical search establishes potential, not a verified supplier order specification.

## Reproduce and inspect

Source scripts:
- `tools/run_broad_hardware_study.py search` (refuses to overwrite an existing frozen shortlist)
- `tools/validate_broad_hardware_study.py`
- `tools/check_broad_study_robustness.py`
- `tools/report_broad_hardware_study.py`
- `tests/test_broad_hardware_study.py`

Use the project Python environment with `MPLBACKEND=Agg`. Raw settings, metrics, frozen reference/shortlist, target arrays and trajectory states are in `results/broad_hardware_optimisation_20260908/`. The directory name preserves the study setup date; completion occurred on 9 September 2026.
'''
    (REPORT/'broader_optimisation_results.md').write_text(report)
    files=[Path('tools/run_broad_hardware_study.py'),Path('tools/validate_broad_hardware_study.py'),Path('tools/check_broad_study_robustness.py'),Path('tools/report_broad_hardware_study.py'),Path('tools/run_hardware_optimisation.py'),Path('CTR_superPosKin_fun_sectioned.py'),OUT/'protocol.json',OUT/'reference_frozen.json',OUT/'shortlist_frozen.json',OUT/'validation.json',OUT/'robustness.json']
    write('manifest.json',dict(completion_date='2026-09-09',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},physical_validation=False,production_parameters_unchanged=True))
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':run()
