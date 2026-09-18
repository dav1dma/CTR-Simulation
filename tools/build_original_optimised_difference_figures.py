"""Descriptive difference/overlay figures; no search, retuning or production changes."""
from run_original_D_comparison import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,ListedColormap,BoundaryNorm
from matplotlib.patches import Patch,Rectangle
import zipfile,hashlib
DEST=Path('output/original_optimised_differences');DEST.mkdir(parents=True,exist_ok=True)

def wq(v,w,q):
 ix=np.argsort(v);return float(v[ix[np.searchsorted(np.cumsum(w[ix])/sum(w),q)]])
def arr(ids,vals):
 g=np.full((40,54),np.nan);valid=(ids//1000>=0)&(ids//1000<40)&(ids%1000>=0)&(ids%1000<54);g[ids[valid]//1000,ids[valid]%1000]=vals[valid];return g.T

def run():
 _,s=h.bank(262144,13301);fields=[spatial_data(d,s) for d in DESIGNS]
 cells=[]
 for f in fields:
  ids=f['ids'];cells.append(ids[(ids//1000>=0)&(ids//1000<40)&(ids%1000<54)])
 common=np.intersect1d(cells[0],cells[1]);support=np.ones(len(common),bool);med=[]
 for f in fields:
  ix=np.searchsorted(f['ids'],common);support&=f['counts'][ix]>=30;med.append(f['medians'][ix])
 cid=common[support];delta=(med[1]-med[0])[support];limit=float(np.max(abs(delta)));norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
 fig,axes=plt.subplots(1,2,figsize=(10.5,5),layout='constrained')
 for ax in axes:
  ax.imshow(arr(common,np.ones(len(common))),origin='lower',extent=[0,100,0,135],aspect='auto',cmap=ListedColormap(['#dadada']),vmin=0,vmax=1)
  im=ax.imshow(arr(cid,delta),origin='lower',extent=[0,100,0,135],aspect='auto',cmap='RdBu',norm=norm,interpolation='nearest');ax.set(xlabel='Radial distance (mm)',ylabel='Z from front plate (mm)')
 axes[0].set(title='Common workspace',xlim=(0,100),ylim=(0,135));axes[0].add_patch(Rectangle((0,40),10,90,fill=False,edgecolor='#222222',lw=1))
 axes[1].set(title='Forward-corridor detail',xlim=(0,10),ylim=(40,130))
 fig.colorbar(im,ax=axes,label='Optimised − original cell-median isotropy');fig.suptitle('Where positional dexterity changes • identical colour scale in both panels')
 for ext in ['png','pdf']:fig.savefig(DEST/f'dexterity_difference.{ext}',dpi=180)
 plt.close(fig)
 gained=np.setdiff1d(cells[1],cells[0]);lost=np.setdiff1d(cells[0],cells[1]);union=np.union1d(cells[0],cells[1]);category=np.zeros(len(union));category[np.isin(union,gained)]=1;category[np.isin(union,lost)]=2
 cmap=ListedColormap(['#aebbc2','#25896d','#d97732']);fig,ax=plt.subplots(figsize=(7,5.5),layout='constrained');ax.imshow(arr(union,category),origin='lower',extent=[0,100,0,135],aspect='auto',cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5,2.5],3),interpolation='nearest');ax.set(xlabel='Radial distance (mm)',ylabel='Z from front plate (mm)',title='Sampled workspace overlay • 2.5 mm cells')
 ax.add_patch(Rectangle((0,40),10,90,fill=False,edgecolor='#222222',lw=1));ax.legend(handles=[Patch(color=cmap.colors[i],label=n) for i,n in enumerate(['Shared reach','Optimised only','Original only'])],loc='lower right')
 for ext in ['png','pdf']:fig.savefig(DEST/f'workspace_overlay.{ext}',dpi=180)
 plt.close(fig)
 volume=lambda ids:float(np.sum(2*(ids//1000)+1)*np.pi*2.5**3/1000)
 regions={k:np.array(v) for k,v in json.loads(Path('results/regional_comparison_20260910_round2/regions.json').read_text()).items()}
 targets=np.load(R/'targets.npz');tid=targets['cell_ids'];errors=[np.load(R/(n+'.npz'))['errors'] for n in ['Original','D']]
 names={'forward_corridor':'Forward corridor','low_dexterity':'Low-dexterity region','IK_hotspots':'IK problem region','remaining':'Remaining region'};rows=[]
 for name,ids in regions.items():
  mask=np.isin(cid,ids);used=cid[mask];w=2*(used//1000)+1
  a=fields[0]['medians'][np.searchsorted(fields[0]['ids'],used)];b=fields[1]['medians'][np.searchsorted(fields[1]['ids'],used)]
  tm=np.isin(tid,ids);u,n=np.unique(tid[tm],return_counts=True);tw=np.array([(2*(c//1000)+1)/n[np.searchsorted(u,c)] for c in tid[tm]])
  r=dict(region=names[name],cell_count=len(used),common_supported_region_volume_fraction=float(sum(w)/sum(2*(ids//1000)+1)),isotropy_original=float(np.average(a,weights=w)),isotropy_optimised=float(np.average(b,weights=w)),isotropy_p10_original=wq(a,w,.1),isotropy_p10_optimised=wq(b,w,.1),targets=int(sum(tm)),target_region_volume_fraction=float(sum(2*(u//1000)+1)/sum(2*(ids//1000)+1)))
  r['isotropy_change_percent']=100*(r['isotropy_optimised']/r['isotropy_original']-1)
  for label,e in zip(['original','optimised'],errors):r['success_'+label]=float(np.average(e[tm]<=.5,weights=tw));r['p95_'+label]=wq(e[tm],tw,.95)
  rows.append(r)
 data=dict(regions=rows,workspace=dict(original_cm3=volume(cells[0]),optimised_cm3=volume(cells[1]),shared_cm3=volume(common),gained_cm3=volume(gained),lost_cm3=volume(lost)),settings=dict(configuration_seed=13301,n=262144,cell_size_mm=2.5,dexterity_cell_support=30,delta='optimised minus original; absolute isotropy units',region_source='round2 frozen baseline-defined memberships',target_source=str(R/'targets.npz'),interpretation='post-hoc descriptive comparison; not a retest of the original five-sample regional selection gates'))
 (DEST/'summary_data.json').write_text(json.dumps(data,indent=2))
 t1='\n'.join(f"| {r['region']} | {r['isotropy_original']:.5f} | {r['isotropy_optimised']:.5f} | {r['isotropy_change_percent']:+.2f}% | {r['isotropy_p10_original']:.5f} | {r['isotropy_p10_optimised']:.5f} | {100*r['common_supported_region_volume_fraction']:.1f}% |" for r in rows)
 t2='\n'.join(f"| {r['region']} | {100*r['success_original']:.3f}% | {100*r['success_optimised']:.3f}% | {r['p95_original']:.5f} | {r['p95_optimised']:.5f} | {r['targets']:,} |" for r in rows)
 captions=dict(dexterity_difference='Optimised minus original cell-median positional isotropy on common cells with at least 30 configurations for each design. Blue indicates higher isotropy, red lower isotropy; grey denotes common occupied cells with insufficient support and white excludes cells outside the comparison. The right panel enlarges the fixed forward corridor using the same colour scale. Differences are absolute isotropy units, not percentages or physical accuracy.',workspace_overlay='Sampled radial-height workspace overlay. Grey denotes cells occupied by both designs, green optimised-only cells and orange original-only cells. Each occupied cell represents a full swept annulus under the rotation-symmetry assumption. Boundary gains and losses depend on sampling and cell size; occupancy does not prove every enclosed point reachable. The outlined box is the forward corridor.')
 report=f'''# Original versus optimised: differences and regional summary

## Dexterity difference map

![Dexterity difference](dexterity_difference.png)

{captions['dexterity_difference']}

## Workspace overlay

![Workspace overlay](workspace_overlay.png)

{captions['workspace_overlay']}

Shared sampled volume: {volume(common):.2f} cm³; optimised-only volume: {volume(gained):.2f} cm³; original-only volume: {volume(lost):.2f} cm³. Original total: {volume(cells[0]):.2f} cm³; optimised total: {volume(cells[1]):.2f} cm³.

## Regional summary — positional dexterity

| Region | Mean cell median: original | Optimised | Change | Spatial P10: original | Optimised | Common supported volume |
|---|---:|---:|---:|---:|---:|---:|
{t1}

Means and P10 are volume-weighted across the SAME supported cells for both designs. P10 is the lower spatial decile of cell medians, not the tenth percentile across all actuator states. Cells need at least 30 configuration samples per design. The supported-volume column shows the fraction of the fixed region represented; excluded cells are not treated as zero. Consequently these supplementary scores differ from the earlier optimisation gate metrics, which used five-sample support and zeros for unsupported reference cells. Do not substitute these figures into the old gate decisions.

## Regional summary — fixed-start IK

| Region | Success: original | Optimised | P95 residual: original (mm) | Optimised (mm) | Shared targets |
|---|---:|---:|---:|---:|---:|
{t2}

These are the saved fresh 19901 target-set results from the final original/optimised comparison. Both designs use the same targets and solver/start settings. Statistics are weighted by swept-cell volume divided by targets in the cell. Regions overlap, so row counts must not be added. These are region-level tails, not per-cell p95 estimates. The low-dexterity and IK-problem regions are derived from original data; the remaining region excludes the union of hotspots and the forward corridor. A target's numerical fixed-start failure does not establish that it is geometrically unreachable.

## Use in the dissertation

Retain the original side-by-side plots. Add the difference map to show where changes occur, and the overlay to distinguish extra reach from shared reach. Use the regional tables to quantify the changes instead of adjusting separate colour scales to exaggerate differences. The gains remain modest; the figures do not claim uniform improvement, manufacturing readiness or physical accuracy.

PNG/PDF versions and copyable LaTeX tables/figure blocks are included. New scores are descriptive, not a new optimisation or a fresh selection validation. No parameters or prior results were changed.
'''
 (DEST/'differences_and_regional_summary.md').write_text(report)
 tex='% Requires graphicx and booktabs, already present in the report.\n'
 for k,caption in captions.items():tex+='\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.95\\linewidth]{figures/optimised_differences/'+k+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:optimised-'+k.replace('_','-')+'}\n\\end{figure}\n\n'
 tex+='\\begin{table}[htbp]\n\\centering\\small\n\\caption{Regional positional isotropy over common supported cells. O: original; Opt: optimised. P10 is the weighted lower spatial decile of cell medians.}\n\\begin{tabular}{lrrrrr}\n\\toprule\nRegion & Mean O & Mean Opt & Change (\\%) & P10 O & P10 Opt \\\\\n\\midrule\n'
 for r in rows:tex+=f"{r['region']} & {r['isotropy_original']:.5f} & {r['isotropy_optimised']:.5f} & {r['isotropy_change_percent']:+.2f} & {r['isotropy_p10_original']:.5f} & {r['isotropy_p10_optimised']:.5f} "+r'\\'+'\n'
 tex+='\\bottomrule\n\\end{tabular}\n\\label{tab:regional-dexterity-difference}\n\\end{table}\n\n'
 tex+='\\begin{table}[htbp]\n\\centering\\small\n\\caption{Regional fixed-start IK comparison. Success is residual at most 0.5 mm. O: original; Opt: optimised.}\n\\begin{tabular}{lrrrr}\n\\toprule\nRegion & Success O (\\%) & Success Opt (\\%) & P95 O (mm) & P95 Opt (mm) \\\\\n\\midrule\n'
 for r in rows:tex+=f"{r['region']} & {100*r['success_original']:.3f} & {100*r['success_optimised']:.3f} & {r['p95_original']:.5f} & {r['p95_optimised']:.5f} "+r'\\'+'\n'
 tex+='\\bottomrule\n\\end{tabular}\n\\label{tab:regional-ik-difference}\n\\end{table}\n'
 tex+='\n% Regional support fractions and interpretation must accompany the tables.\n'
 tex+='Isotropy statistics use common cells with at least 30 configurations per design. Supported region volumes are '+', '.join(f"{r['region']}: {100*r['common_supported_region_volume_fraction']:.1f}\\%" for r in rows)+'. The regions overlap and their counts must not be summed. These descriptive scores use a different support convention from the optimisation acceptance checks.\n'
 (DEST/'figure_and_table_blocks.tex').write_text(tex)
 (DEST/'README.txt').write_text('Upload figures/optimised_differences from ZIP to Overleaf. Copy the selected blocks from figure_and_table_blocks.tex. Read differences_and_regional_summary.md for interpretation. Existing side-by-side plots remain unchanged. Figures visually inspected; LaTeX fragments not locally compiled.\n')
 with zipfile.ZipFile(DEST/'difference_maps_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in DEST.glob('*.pdf'):z.write(f,'figures/optimised_differences/'+f.name)
  for n in ['figure_and_table_blocks.tex','README.txt','differences_and_regional_summary.md']:z.write(DEST/n,n)
 (DEST/'manifest.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),settings=data['settings']),indent=2))
 print(json.dumps(data,indent=2),flush=True)
if __name__=='__main__':run()
