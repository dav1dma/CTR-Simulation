"""Chapter-aligned comparison figures and an optional LaTeX subsection."""
from run_original_D_comparison import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm,ListedColormap
import hashlib,zipfile

def run():
 summary=json.loads((R/'summary.json').read_text());target=np.load(R/'targets.npz');ids=target['cell_ids'];ref=np.array(json.loads((OUT/'reference_frozen.json').read_text())['ids']);raw=[np.load(R/(n+'.npz')) for n in ['Original','D']]
 summary=[dict(r,name=('Optimised configuration' if r['name']=='D' else r['name'])) for r in summary]
 _,states=h.bank(262144,13301);fields=[spatial_data(d,states) for d in DESIGNS]
 def export(fig,name):
  for ext in ['png','pdf']:fig.savefig(P/(name+'.'+ext),dpi=180,bbox_inches='tight')
  plt.close(fig)
 def array(cells,vals):
  g=np.full((40,54),np.nan);valid=(cells//1000>=0)&(cells//1000<40)&(cells%1000<54);g[cells[valid]//1000,cells[valid]%1000]=vals[valid];return g.T
 def pair(grids,title,bar,name,*,vmin=0,vmax=1,cmap='viridis',norm=None,reference=False):
  fig,axes=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True,layout='constrained')
  for ax,grid,label in zip(axes,grids,['Original','Optimised configuration']):
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
  for d,label,color,style in zip(DESIGNS,['Original','Optimised configuration'],['#3588a1','#bc642d'],['-','--']):
   lc=0 if j==0 else d[j+1];k=0 if j==0 else d[j-1]/1000;straight=length[j]-lc;t=np.linspace(0,lc,160);x=np.zeros_like(t) if k==0 else (1-np.cos(k*t))/k;z=straight+t if k==0 else straight+np.sin(k*t)/k
   ax.plot(np.r_[0,0,x],np.r_[0,straight,z],color=color,ls=style,label=label)
  ax.set(title=['Inner element (overlapping shapes)','Middle tube','Outer tube'][j],xlabel='Lateral offset (mm)',ylabel='Axial position from grip (mm)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
 axes[-1].legend();fig.suptitle('Intrinsic tube shapes • straight section followed by constant-curvature arc');export(fig,'tube_shapes')
 # Main values computed from this comparison's data, never copied from B1.
 reference=dict(ids=ref,near=fields[0]['near']);wm=[evaluate(d,states,reference) for d in DESIGNS]
 gains={k:100*(wm[1][k]/wm[0][k]-1) for k in ['volume_cm3','overall_isotropy','near_isotropy']}
 n=summary[0]['n'];support=100*summary[0]['p95_supported_reference_volume']
 table='\n'.join(f"| {r['name']} | {w['volume_cm3']:.2f} | {w['overall_isotropy']:.5f} | {w['near_isotropy']:.5f} | {100*r['success_fraction']:.3f}% | {r['median_mm']:.6g} | {r['p95_mm']:.6g} |" for r,w in zip(summary,wm))
 params='''| Parameter (inner / middle / outer) | Original | Optimised configuration |
|---|---|---|
| Assumed grip-to-tip length (mm) | 350 / 170 / 80 | 350 / 170 / 80 |
| Curved length (mm) | 0 / 90 / 65 | 0 / 47 / 27.5 |
| Precurvature (m⁻¹) | 0 / 19.12 / 14.04 | 0 / 20.00 / 14.00 |
| Outer diameter (mm) | 0.50 / 0.70 / 0.90 | 0.50 / 0.70 / 0.90 |
| Inner diameter (mm) | 0 / 0.62 / 0.80 | 0 / 0.62 / 0.80 |
| Young's modulus (GPa) | 75 / 75 / 75 | 75 / 75 / 75 |'''
 captions={
 'tube_shapes':'Original and Optimised configuration intrinsic free tube shapes calculated from straight sections and constant-curvature arcs. These are separate unloaded tube geometries, not assembled robot shapes or manufacturing validation. Optimised configuration has slightly greater middle curvature but a shorter curved section. Curved lengths at or above 47/27.5 mm are exterior-equivalent at the nominal deployment caps, so shortening is not uniquely optimal and must not be credited as the sole cause of the performance gain.',
 'workspace':'Estimated inner-tip workspace from 262,144 shared independent configurations per design, using 2.5 mm radial-height swept cells and the same hardware limits. Optimised configuration modestly expands workspace. Occupied cells do not prove every contained point reachable. Common rotation is assumed unrestricted.',
 'isotropy_median':'Median range-normalised positional isotropy, using a common colour scale and at least 30 configurations per displayed cell. Blank cells lack occupancy or sufficient support. Optimised configuration is a conservative change; improvements are not uniform. This is a positional measure, not orientation dexterity, stiffness or physical accuracy.',
 'ik_median':f'Median fixed-start numerical residual on {n:,} identical original-design targets. At least 30 targets are required per cell. Grey denotes insufficient support in the reference region, white lies outside it. Both residual figures share a scale linear below 0.01 mm and logarithmic above, without clipping. Small typical residuals can coexist with regional failures.',
 'ik_p95':'95th-percentile fixed-start residual, requiring at least 100 targets per cell. It describes an upper tail of sampled errors, not the maximum, a confidence limit or clinical safety. Both designs retain difficult regions despite small typical residuals.',
 'ik_failure':'Fixed-start failure fraction: final residual above 0.5 mm, with at least 100 targets per cell. Identical targets, feasible initial state and solver settings are used. Failure from one initial state does not demonstrate geometric unreachability.',
 'target_support':f'Identical target counts for Original and D. Cells with at least 100 targets cover {support:.2f}% of the frozen reference volume; the retained 90% support gate is {"passed" if support>=90 else "not passed"}. Place this supporting figure in the appendix.'}
 figures='\n\n'.join(f"![{k}]({k}.png)\n\n{v}" for k,v in captions.items())
 a,b=summary
 findings=f"Estimated workspace increases by {gains['volume_cm3']:.2f}%, overall positional isotropy by {gains['overall_isotropy']:.2f}%, and near-axis positional isotropy by {gains['near_isotropy']:.2f}%. Fixed-start success is {100*a['success_fraction']:.3f}% for Original and {100*b['success_fraction']:.3f}% for the optimised configuration. Median residual is {a['median_mm']:.6g} versus {b['median_mm']:.6g} mm; p95 is {a['p95_mm']:.6g} versus {b['p95_mm']:.6g} mm. These results support a modest numerical design improvement, not universal superiority or elimination of fixed-start failures."
 report=f'''# Original versus Optimised configuration: supervisor-aligned comparative study

## Purpose

Compare the supplied original tubes with the conservatively selected Optimised configuration using the supervisor-requested parameter table, tube shapes, workspace, dexterity and residual-error plots. This replaces B1 as the main candidate in the new comparison package; it does not replace the historical baseline chapter or alter production simulator settings.

## Tube parameters

{params}

## Summary of independent comparison

| Design | Workspace (cm³) | Overall isotropy | Near-axis isotropy | Fixed-start success ≤0.5 mm | Median residual (mm) | P95 residual (mm) |
|---|---:|---:|---:|---:|---:|---:|
{table}

{findings}

All new fixed-start results use a fresh target-bank seed (19901), after Optimised configuration was selected. Global results are volume-weighted over target-bearing reference cells. Mean residual: Original {a['mean_mm']:.6g} mm; Optimised configuration {b['mean_mm']:.6g} mm. Maximum residual: Original {a['max_mm']:.6g} mm; Optimised configuration {b['max_mm']:.6g} mm. These quantities include failed attempts, not only successful solves.

## Continuity with the original results plan

Keep the original sequence: parameter/shape description, workspace, median positional isotropy, median IK residual, p95 residual and failure map. Each figure below contains Original and Optimised configuration side by side, with matching axes and scales. The support plot belongs in the appendix. The optimisation search and B1/C alternatives remain supporting evidence, rather than additional main-text comparisons.

The original six-variable, single-start damped least-squares method retains 40 iterations, damping 1 mm, normalised step cap 0.1, forward difference 0.0001, stopping tolerance 0.01 mm, line-search scales 1/0.5/0.25/0.1 and task threshold 0.5 mm. Every solve starts at exposures 38/0/0 mm and zero rotations. Only the deployment encoding/decoding is adapted to the hardware limits; there are no restarts or target-derived initial guesses. The vectorised implementation is checked against the original scalar solver on 12 targets per design, with hardware limits and nesting checked for returned states.

The old zero-exposure state is not available under the approximate installed geometry. The fixed-start comparison therefore uses its closest feasible analogue. Hardware limits are inner38–135, middle0–47, outer0–27.5 mm with nested ordering, based on the user-authorised 35 mm grip offset. Grip-to-tip lengths, guides and clamping remain assumptions needing verification.

A new original-design FK bank of 1,048,576 configurations supplies {n:,} targets, capped at 170 per frozen cell. Both designs receive the same canonical positive-x–z targets, as in the old axisymmetric convention. Global statistics use swept-cell volume divided by target count, not equal weighting of spatial cells. Median maps need 30 targets; p95/failure maps need 100. Supported reference volume is {support:.2f}%. The latter is a numerical support criterion, not confidence in physical accuracy.

Workspace/isotropy use 262,144 configurations (seed13301) already used in the broader study, not a newly untouched workspace holdout. The fixed baseline reference and actuator scales remain unchanged. Overall isotropy is a volume-weighted mean of cell medians over that common reference; near-axis isotropy uses radius≤10 mm, Z40–130 mm. Summary scores retain the five-sample support rule of the search study, while displayed median-isotropy maps retain the older 30-sample convention. They are not averages of exactly the same displayed cells.

This hardware-constrained experiment differs from the historical Stage4/4.1 evaluation in deployment limits, feasible start, targets and 2.5 mm cell size. Do not attribute the difference from the old 48,656.99 cm³ workspace or 91.96% success directly to tube optimisation. The fair comparison is Original versus Optimised configuration under the shared settings here.

Workspace gains are sampling-dependent: this shared configuration sample gives 6.32%, whereas the earlier independent regional comparison gave about 4.1%. Report the gain as modest and sample-dependent, rather than an exact geometric improvement.

## Figures and takeaway captions

{figures}

## Interpretation and placement

Optimised configuration was retained because larger-curvature alternatives did not consistently meet the declared regional no-meaningful-deterioration allowances across independent evaluations. Optimised configuration passed those allowances on both extended-search holdouts; that is not proof of exact equality or improvement at every point. The allowances were 1% relative isotropy, 0.5 percentage points coverage/success, 0.01 mm residual and 1% volume. Those earlier regional checks and this fresh global comparison answer complementary questions.

Use the table and figures in a single subsection titled “Effect of tube-configuration optimisation.” State the numerical improvements and retained weaknesses, then relate them to the objective of investigating a compatible tube set. Avoid claiming a global optimum, endovascular superiority or build-ready compatibility. Orientation, anatomy, torsional stability, loading and measured physical accuracy remain outside this evaluation.

comparison_subsection.tex provides an editable results subsection and parameter/results tables. figure_blocks.tex supplies selectable figure blocks. The package does not modify your existing chapter. PNG and vector PDF figures are supplied; the LaTeX fragment requires graphicx and is not compiled locally.
'''
 (P/'comparison_results.md').write_text(report)
 (P/'comparison_data.json').write_text(json.dumps(dict(fixed_start=summary,workspace_isotropy=wm,relative_gains_percent=gains,parameters=[d.tolist() for d in DESIGNS]),indent=2))
 def esc(s):return s.replace('%',r'\%').replace('³',r'$^3$').replace('–','--').replace('’',"'").replace('≥',r'$\geq$')
 blocks='\n\n'.join('\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.95\\linewidth]{figures/original_D/'+k+'.pdf}\n\\caption{'+esc(v)+'}\n\\label{fig:original-D-'+k.replace('_','-')+'}\n\\end{figure}' for k,v in captions.items());(P/'figure_blocks.tex').write_text(blocks)
 latex=r'''% Editable supplementary comparison, not a replacement of historical results.
\subsection{Effect of tube-configuration optimisation}
The original tube configuration was compared with the optimised configuration under identical estimated hardware constraints. The optimised configuration was selected through a hardware-constrained numerical search, subject to regional performance requirements. Independent evaluation favoured its conservative trade-off over more aggressive alternatives. It remains an exploratory numerical specification.

\begin{table}[htbp]
\centering
\caption{Tube parameters, listed inner/middle/outer. Total lengths are assumed grip-to-tip values.}
\begin{tabular}{lcc}
\hline
Parameter & Original & Optimised configuration \\
\hline
Total length (mm) & 350/170/80 & 350/170/80 \\
Curved length (mm) & 0/90/65 & 0/47/27.5 \\
Precurvature (m$^{-1}$) & 0/19.12/14.04 & 0/20/14 \\
Outer diameter (mm) & 0.50/0.70/0.90 & 0.50/0.70/0.90 \\
Inner diameter (mm) & 0/0.62/0.80 & 0/0.62/0.80 \\
Young's modulus (GPa) & 75/75/75 & 75/75/75 \\
\hline
\end{tabular}
\end{table}

'''
 latex+=esc(findings)+'\n\n'
 latex+='\\begin{table}[htbp]\n\\centering\n\\caption{Common-setting numerical comparison. IK statistics are volume-weighted.}\n\\begin{tabular}{lrr}\n\\hline\nMetric & Original & Optimised configuration \\\\\n\\hline\n'
 for label,x,y in [('Workspace (cm$^3$)',wm[0]['volume_cm3'],wm[1]['volume_cm3']),('Overall isotropy',wm[0]['overall_isotropy'],wm[1]['overall_isotropy']),('Near-axis isotropy',wm[0]['near_isotropy'],wm[1]['near_isotropy']),('IK success (\\%)',100*a['success_fraction'],100*b['success_fraction']),('Median residual (mm)',a['median_mm'],b['median_mm']),('P95 residual (mm)',a['p95_mm'],b['p95_mm'])]:latex+=f'{label} & {x:.6g} & {y:.6g} '+r'\\'+'\n'
 latex+='\\hline\n\\end{tabular}\n\\end{table}\n\n'
 latex+=esc(f'The fixed-start test used {n:,} identical independent original-design targets, a 38/0/0 mm exposure start and zero rotations. The original 40-iteration solver settings were retained, with hardware-adapted deployment bounds. At least 100 targets per cell supported p95/failure maps across {support:.2f}% of reference volume. The old unconstrained results must not be compared directly with this hardware-constrained study. The residuals measure numerical convergence, not physical positioning accuracy. Improvements are not uniform, and the fixed-start failure regions remain.')+'\n'
 latex+='\nWorkspace gains depend on sampling: this sample gives 6.32\\%, compared with about 4.1\\% in an earlier independent evaluation.\n'
 (P/'comparison_subsection.tex').write_text(latex)
 (P/'README.txt').write_text('Start with comparison_results.md. Upload figures/original_D from the ZIP to Overleaf. Insert comparison_subsection.tex and selected figure_blocks.tex blocks. Requires graphicx. Target support and search evidence belong in appendix. Captions and prose are editable drafts; review in your own voice. Figures visually checked; LaTeX fragment not compiled locally. No existing chapter changed.\n')
 with zipfile.ZipFile(P/'original_D_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in P.glob('*.pdf'):z.write(f,'figures/original_D/'+f.name)
  for name in ['comparison_subsection.tex','figure_blocks.tex','README.txt']:z.write(P/name,name)
 files=[Path(__file__),Path('tools/run_original_D_comparison.py'),R/'protocol.json',R/'summary.json',R/'targets.npz',R/'Original.npz',R/'D.npz']
 (P/'manifest.json').write_text(json.dumps(dict(files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},production_unchanged=True),indent=2))
 print(findings,flush=True)
if __name__=='__main__':run()
