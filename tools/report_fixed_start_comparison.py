"""Chapter-aligned comparison figures and an optional LaTeX subsection."""
from run_fixed_start_comparison import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm,ListedColormap
import hashlib,zipfile

def run():
 summary=json.loads((R/'summary.json').read_text());target=np.load(R/'targets.npz');ids=target['cell_ids'];ref=np.array(json.loads((OUT/'reference_frozen.json').read_text())['ids']);raw=[np.load(R/(n+'.npz')) for n in ['Original','B1']]
 _,states=h.bank(262144,13301);fields=[spatial_data(d,states) for d in DESIGNS]
 def export(fig,name):
  for ext in ['png','pdf']:fig.savefig(P/(name+'.'+ext),dpi=180,bbox_inches='tight')
  plt.close(fig)
 def array(cells,vals):
  g=np.full((40,54),np.nan);valid=(cells//1000>=0)&(cells//1000<40)&(cells%1000<54);g[cells[valid]//1000,cells[valid]%1000]=vals[valid];return g.T
 def pair(grids,title,bar,name,*,vmin=0,vmax=1,cmap='viridis',norm=None,reference=False):
  fig,axes=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True,layout='constrained')
  for ax,grid,label in zip(axes,grids,['Original','B1']):
   if reference:ax.imshow(array(ref,np.ones(len(ref))),origin='lower',extent=[0,100,0,135],aspect='auto',cmap=ListedColormap(['#cccccc']),vmin=0,vmax=1)
   kw=dict(norm=norm) if norm else dict(vmin=vmin,vmax=vmax)
   im=ax.imshow(grid,origin='lower',extent=[0,100,0,135],aspect='auto',cmap=cmap,**kw);ax.set(title=label,xlabel='Radial distance (mm)')
  axes[0].set_ylabel('Z from front plate (mm)');fig.colorbar(im,ax=axes,label=bar);fig.suptitle(title);export(fig,name)
 pair([array(f['ids'],np.ones(len(f['ids']))) for f in fields],'Inner-tip workspace • identical estimated hardware limits','Occupied cell','workspace',cmap=ListedColormap(['#3588a1']))
 pair([array(f['ids'][f['counts']>=30],f['medians'][f['counts']>=30]) for f in fields],'Median positional isotropy • at least 30 configurations per cell','Range-normalised positional isotropy','isotropy_median',vmax=max(np.max(f['medians']) for f in fields))
 cell,count=np.unique(ids,return_counts=True)
 maps={}
 for key,q,support in [('median',50,30),('p95',95,100),('failure',None,100)]:
  maps[key]=[]
  for data in raw:
   vals=np.array([np.percentile(data['errors'][ids==c],q) if q else 100*np.mean(data['errors'][ids==c]>.5) for c in cell]);vals[count<support]=np.nan;maps[key].append(array(cell,vals))
 maxerr=max(float(np.nanmax(g)) for k in ['median','p95'] for g in maps[k]);norm=SymLogNorm(linthresh=.01,linscale=1,vmin=0,vmax=maxerr)
 for key in ['median','p95']:pair(maps[key],('Median' if key=='median' else '95th-percentile')+' fixed-start IK residual','Numerical residual (mm)','ik_'+key,cmap='RdYlGn_r',norm=norm,reference=True)
 pair(maps['failure'],'Fixed-start failure fraction • residual > 0.5 mm','Targets exceeding threshold (%)','ik_failure',vmax=100,cmap='YlOrRd',reference=True)
 pair([array(cell,count),array(cell,count)],'Shared target support • identical targets for both designs','Independent targets per cell','target_support',vmax=170,reference=True)
 # Intrinsic free shapes, separated to avoid implying assembled geometry.
 fig,axes=plt.subplots(1,3,figsize=(11,5),layout='constrained');length=[350,170,80]
 for j,ax in enumerate(axes):
  for d,label,color,style in zip(DESIGNS,['Original','B1'],['#3588a1','#bc642d'],['-','--']):
   lc=0 if j==0 else d[j+1];k=0 if j==0 else d[j-1]/1000;straight=length[j]-lc;t=np.linspace(0,lc,160);x=np.zeros_like(t) if k==0 else (1-np.cos(k*t))/k;z=straight+t if k==0 else straight+np.sin(k*t)/k
   ax.plot(np.r_[0,0,x],np.r_[0,straight,z],color=color,ls=style,label=label)
  ax.set(title=['Inner element (overlapping shapes)','Middle tube','Outer tube'][j],xlabel='Lateral offset (mm)',ylabel='Axial position from grip (mm)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
 axes[-1].legend();fig.suptitle('Intrinsic tube shapes • straight section followed by constant-curvature arc');export(fig,'tube_shapes')
 old=json.loads((OUT/'comparison.json').read_text());wm=[old[0],old[2]]
 table='\n'.join(f"| {r['name']} | {w['volume_cm3']:.1f} | {w['overall_isotropy']:.5f} | {w['near_isotropy']:.5f} | {100*r['success_fraction']:.3f}% | {r['median_mm']:.6g} | {r['p95_mm']:.6g} |" for r,w in zip(summary,wm))
 parameters='''| Parameter (inner / middle / outer) | Original | B1 |
|---|---|---|
| Assumed grip-to-tip length (mm) | 350 / 170 / 80 | 350 / 170 / 80 |
| Curved length (mm) | 0 / 90 / 65 | 0 / 47 / 27.5 |
| Precurvature (m⁻¹) | 0 / 19.12 / 14.04 | 0 / 28.68 / 21.06 |
| OD (mm) | 0.50 / 0.70 / 0.90 | 0.50 / 0.70 / 0.90 |
| ID (mm) | 0 / 0.62 / 0.80 | 0 / 0.62 / 0.80 |
| Young's modulus (GPa) | 75 / 75 / 75 | 75 / 75 / 75 |'''
 captions={
 'tube_shapes':'Intrinsic, unloaded tube shapes calculated from the stated straight and curved sections. Tubes are shown separately, not as the assembled CTR; tube thickness is not represented. The inner element is unchanged. B1’s shorter curved sections are not uniquely optimal: retaining original curved lengths is exterior-equivalent at the nominal deployment caps, but not necessarily mechanically equivalent.',
 'workspace':'Predicted inner-tip occupied workspace with identical hardware bounds, 262,144 configurations per design and 2.5 mm swept cells. B1 expands lateral reach. Cell occupancy does not prove every enclosed point reachable; unrestricted common rotation provides the axisymmetric interpretation.',
 'isotropy_median':'Median positional isotropy in radial–axial cells with at least 30 configuration samples. Both designs use the same colour scale and fixed baseline actuator normalisation. Blank cells are unoccupied or insufficiently supported; they are not assigned zero dexterity. Scores are positional and do not measure stiffness or physical accuracy.',
 'ik_median':'Median fixed-start numerical residual on identical original-design targets. Cells require at least 30 targets. Grey is insufficient support inside the frozen reference region; white is outside it. Both residual figures share a nonlinear scale, linear below 0.01 mm and logarithmic above, with no clipping.',
 'ik_p95':'95th-percentile fixed-start numerical residual, requiring at least 100 targets per cell. It describes the upper tail of the sampled residual distribution, not a maximum, confidence bound or safety guarantee. Colour scale and masks follow the companion median figure.',
 'ik_failure':'Fraction of identical targets whose fixed-start residual exceeds 0.5 mm, requiring at least 100 targets per cell. Failure from this start does not prove geometric unreachability. The original design’s targets are independently known reachable from FK; B1 failures are not, by themselves, evidence of workspace loss.',
 'target_support':'Independent targets per spatial cell, identical for both designs. At least 100 targets support p95/failure estimates across 92.28% of the frozen reference volume, passing the retained 90% support criterion. Keep this figure in the appendix.'}
 figures='\n\n'.join(f"![{key}]({key}.png)\n\n{cap}" for key,cap in captions.items())
 report=f'''# Original versus B1: chapter-aligned comparison

This package implements the supervisor-requested parameter, tube-shape, workspace, dexterity and IK comparisons. It supplements the existing baseline results; it does not replace or rerun the historical Stage 4/4.1 protocol.

## Parameters

{parameters}

## Main results

| Design | Workspace (cm³) | Overall isotropy | Near-axis isotropy | Fixed-start success ≤0.5 mm | Median residual (mm) | p95 residual (mm) |
|---|---:|---:|---:|---:|---:|---:|
{table}

B1 increases estimated workspace by 113.9%, overall positional isotropy by 29.2% and near-axis isotropy by 34.7% relative to the original, using the same hardware-constrained evaluation. Fixed-start success is effectively unchanged, about 95.9% for both. B1 lowers the global median residual but slightly raises the p95 and mean residuals; this is not a universal IK improvement. The supported design benefit is greater workspace and higher average positional isotropy.

## Method continuity and necessary changes

The single-start six-variable damped least-squares algorithm retains the original settings: 40 iterations, damping 1 mm, normalised step cap 0.1, forward-difference step 0.0001, stopping tolerance 0.01 mm, evaluation threshold 0.5 mm and line-search scales 1, 0.5, 0.25, 0.1. There are no restarts, nearest-neighbour initialisations or warm starts. Every target starts at exposure 38/0/0 mm and zero rotations. Hardware-adapted outer-first nested fractions retain the original parameterisation structure while respecting the new bounds; the inner minimum is 38 mm. This is the closest feasible analogue of the original zero-exposure start, not an identical actuator state.

The vectorised implementation was checked against the original scalar solver with only the deployment encoding/decoding adapted, on 12 targets per design. Maximum residual disagreement was below 3e-9 mm. Returned deployments are checked against hardware limits and nesting.

A new independent 1,048,576-configuration original-design bank (seed 14901) supplied 97,869 known-reachable targets, up to 170 per frozen reference cell. Targets are expressed in the positive x–z canonical plane, as in the old protocol; results are conditional on that convention and fixed start, not an evaluation of arbitrary target azimuths. Both designs receive exactly the same targets. Global statistics use swept-cell volume divided by the number of targets in each cell; percentiles are weighted empirical quantiles. Spatial median maps require 30 targets and p95/failure maps require 100. The p95/failure support covers 92.28% of reference volume and passes the 90% gate. Global summaries include all target-bearing cells; map support masks affect displayed local statistics.

Workspace and isotropy reuse independent broad-study configurations (262,144, seed 13301). The 2.5 mm cells suit the smaller hardware envelope and differ from the old 10 mm grid. Overall isotropy retains the frozen baseline region and near-axis isotropy uses radius 10 mm, Z=40–130 mm. The isotropy map's 30-sample mask follows the earlier reporting convention; the pre-existing scalar optimisation scores use their documented five-sample support rule and must not be interpreted as averages of the displayed map.

Do not compare these volumes or fixed-start success rates directly with the historical 48,656.99 cm³ / 91.96% results: bounds, starting state, grid and targets differ. The present original-versus-B1 comparison controls those settings across designs. The new targets were generated after candidate selection; candidates were not retuned after seeing these results.

## Figures and takeaway captions

{figures}

## Placement in the dissertation

Keep the existing evaluation narrative. Add one subsection, “Effect of tube-configuration optimisation,” with the parameter table, tube shapes, workspace and median isotropy comparisons. Present the fixed-start median, p95 and failure maps in the same sequence as the original results. Put the target-support figure, Candidate A and detailed search/sensitivity evidence in the appendix. Each map is available as PNG and vector PDF. figure_blocks.tex supplies optional figure blocks; comparison_subsection.tex supplies a compact editable subsection without replacing any existing chapter.

## Limits

The 35 mm grip offset and grip-to-tip lengths remain assumed. No manufacturer has certified B1, and guides, clamping, rotation stops, torsion, loads and physical accuracy are not validated. Intrinsic tube drawings are geometric illustrations. Orientation optimisation remains outside scope. No production simulator parameters or original formal results were changed.
'''
 (P/'comparison_results.md').write_text(report)
 def esc(s):return s.replace('%',r'\%').replace('≤',r'$\leq$').replace('³',r'$^3$').replace('–','--').replace('’',"'")
 blocks='\n\n'.join('\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.95\\linewidth]{figures/optimisation_comparison/'+k+'.pdf}\n\\caption{'+esc(v)+'}\n\\label{fig:opt-'+k.replace('_','-')+'}\n\\end{figure}' for k,v in captions.items())
 (P/'figure_blocks.tex').write_text(blocks)
 latex=r'''% Separate hardware-constrained comparison; adapt prose to your dissertation.
\subsection{Effect of tube-configuration optimisation}
The original tube configuration was compared with candidate B1 under identical estimated actuator limits. The existing tube lengths, diameters and nominal elastic moduli were retained. Middle and outer precurvatures increased from 19.12 and 14.04~m$^{-1}$ to 28.68 and 21.06~m$^{-1}$, respectively; B1 used curved lengths of 47 and 27.5~mm. These remain exploratory parameters rather than a verified manufacturing specification.

\begin{table}[htbp]
\centering
\caption{Original and B1 tube parameters, listed inner/middle/outer. Lengths are assumed grip-to-tip values.}
\begin{tabular}{lcc}
\hline
Parameter & Original & B1 \\
\hline
Total length (mm) & 350/170/80 & 350/170/80 \\
Curved length (mm) & 0/90/65 & 0/47/27.5 \\
Precurvature (m$^{-1}$) & 0/19.12/14.04 & 0/28.68/21.06 \\
Outer diameter (mm) & 0.50/0.70/0.90 & 0.50/0.70/0.90 \\
Inner diameter (mm) & 0/0.62/0.80 & 0/0.62/0.80 \\
Young's modulus (GPa) & 75/75/75 & 75/75/75 \\
\hline
\end{tabular}
\end{table}

Estimated occupied workspace increased from 519.0 to 1110.0~cm$^3$, while overall positional isotropy increased from 0.26671 to 0.34465. Within the specified near-axis region it increased from 0.13218 to 0.17802. These correspond to increases of 113.9\%, 29.2\% and 34.7\%, respectively. Local improvement was not uniform.

For the supplementary fixed-start comparison, both designs were evaluated on 97,869 identical original-design targets. The original damped least-squares settings were retained, with a feasible initial exposure of 38/0/0~mm and zero rotations. Volume-weighted success within 0.5~mm was 95.895\% for the original and 95.891\% for B1. Median residual decreased from 0.000172 to 0.0000240~mm, whereas the 95th percentile increased slightly from 0.00947 to 0.00968~mm. Thus, B1's advantage was workspace and average positional isotropy rather than improved fixed-start success. These are numerical residuals, not measured positioning accuracy.

The hardware bounds, feasible start, spatial resolution and target set differ from the earlier baseline evaluation; the two studies must not be directly pooled. The common-target comparison controls these choices across the two designs. At least 100 targets per cell supported the p95 and failure maps across 92.28\% of the frozen reference volume.
'''
 (P/'comparison_subsection.tex').write_text(latex)
 (P/'README.txt').write_text('Read comparison_results.md first. For Overleaf, upload PDF figures to figures/optimisation_comparison/. Insert comparison_subsection.tex where appropriate and select figure_blocks.tex blocks. Put target_support in the appendix. Requires graphicx. Files are supplementary; no existing chapter was edited. Draft prose and captions should be reviewed in your own voice.\n')
 manifest=dict(files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('tools/run_fixed_start_comparison.py'),R/'protocol.json',R/'summary.json',R/'targets.npz',R/'Original.npz',R/'B1.npz']},production_unchanged=True)
 (P/'manifest.json').write_text(json.dumps(manifest,indent=2))
 with zipfile.ZipFile(P/'overleaf_comparison.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in P.glob('*.pdf'):z.write(f,'figures/optimisation_comparison/'+f.name)
  for n in ['comparison_subsection.tex','figure_blocks.tex','README.txt']:z.write(P/n,n)
 print('Wrote',P,flush=True)
if __name__=='__main__':run()
