"""Figures from frozen results only: no solver or selection changes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctr-mpl')
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import task_prioritised_tradeoff_search as q
from CTR_superPosKin_fun_sectioned import superPosKin
h=q.h;SRC=q.OUT;OUT=SRC/'dissertation_figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'svg.fonttype':'none','savefig.facecolor':'white'})
C={'original':'#3073AD','proposed':'#159582'};names={'original':'Original tubes','proposed':'Proposed tubes'}
d=np.array(json.loads((SRC/'frozen_candidate.json').read_text())['design']);metrics=json.loads((SRC/'mean_metrics.json').read_text());vr=json.loads((SRC/'validation_records.json').read_text())
banks={n:[dict(np.load(SRC/f'{n}_validation_{rep}.npz')) for rep in range(2)] for n in names}
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=250);fig.savefig(OUT/(name+'.svg'));plt.close(fig)
def style3(ax):
 ax.set(xlim=(-135,135),ylim=(-135,135),zlim=(0,185),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)');ax.set_box_aspect((270,270,185));ax.view_init(24,-58)
 for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.fill=False;axis._axinfo['grid']['color']=(.82,.86,.90,.6)
 ax.scatter([0],[0],[0],marker='+',c='black',s=55)
def shell(ids):
 occupied={(int(k//1000),int(k%1000)) for k in ids};faces=[];th=np.linspace(0,2*np.pi,49)
 for r,z in occupied:
  for dr,dz in [(1,0),(-1,0),(0,1),(0,-1)]:
   if (r+dr,z+dz) in occupied or (dr==-1 and r==0):continue
   for t0,t1 in zip(th[:-1],th[1:]):
    if dr:
     rr=5*(r+(dr==1));face=[[rr*np.cos(t0),rr*np.sin(t0),5*z],[rr*np.cos(t1),rr*np.sin(t1),5*z],[rr*np.cos(t1),rr*np.sin(t1),5*(z+1)],[rr*np.cos(t0),rr*np.sin(t0),5*(z+1)]]
    else:
     zz=5*(z+(dz==1));face=[[rr*np.cos(t),rr*np.sin(t),zz] for rr,t in [(5*r,t0),(5*(r+1),t0),(5*(r+1),t1),(5*r,t1)]]
    faces.append(face)
 return faces
fig=plt.figure(figsize=(15,5.8))
for j,pair in enumerate([['original'],['proposed'],['original','proposed']]):
 ax=fig.add_subplot(1,3,j+1,projection='3d');style3(ax)
 for name in pair:
  b=banks[name][0];ax.add_collection3d(Poly3DCollection(shell(b['workspace_cells']),facecolor=C[name],edgecolor='none',alpha=.075,rasterized=True))
  points=b['workspace_points'][::16];ax.scatter(*points.T,s=.3,alpha=.08,color=C[name],rasterized=True)
 if len(pair)==1:
  design=h.BASE if pair[0]=='original' else d;e=np.array([120.,60.,55.]);h.limits(design).validate(e/1000)
  model=superPosKin(h.pars(design),{'ul':e/1000,'uphi':np.array([0,.7,-.3])},{'n_p':30,'isPlot':False})
  for section in model[3]:ax.plot(*(np.asarray(section)[:3]*1000),color='#263849',lw=2)
  ax.set_title(f"({'ab'[j]}) {names[pair[0]]}\n"+('350 / 170 / 80 mm' if j==0 else '350 / 177.5 / 87.5 mm'))
 else:ax.set_title('(c) Overlay');ax.legend(handles=[Line2D([0],[0],color=C[n],lw=5,label=names[n]) for n in names],loc='upper right',fontsize=9)
fig.suptitle('Hardware-constrained inner-tip workspace',fontsize=16,y=.97)
fig.text(.5,.09,'Same axes, camera and 131,072 feasible states per design · + marks the outside front-plate origin',ha='center',fontsize=10)
fig.text(.5,.045,'Approximate ideal-model field: 5 mm radial/axial occupied cells revolved about Z; enclosed positions are not guaranteed reachable.',ha='center',fontsize=9)
fig.subplots_adjust(left=.02,right=.97,bottom=.16,top=.85,wspace=.08);save(fig,'01_workspace_comparison')
# Performance: retain original controller protocol and both repeats.
fig,axs=plt.subplots(2,2,figsize=(12,8.5));x=np.arange(3)
for i,n in enumerate(names):
 y=[100*metrics[n][g+'_ik_success'] for g in ['global','weak','corridor']];axs[0,0].bar(x+(i-.5)*.34,y,.32,color=C[n],label=names[n])
 for rep in range(2):axs[0,0].scatter(x+(i-.5)*.34,[100*vr[n][rep]['metrics'][g+'_ik_success'] for g in ['global','weak','corridor']],s=15,c='black',marker=['o','x'][rep],zorder=3)
axs[0,0].set(xticks=x,xticklabels=['Global','Weak region','Corridor'],ylim=(0,100),ylabel='Weighted success (%)',title='(a) Fixed-start IK, within 0.5 mm');axs[0,0].legend(fontsize=9)
ks=['global_isotropy_mean','global_isotropy_p10','weak_isotropy_p10','corridor_isotropy_p10'];xx=np.arange(4)
for i,n in enumerate(names):axs[0,1].bar(xx+(i-.5)*.34,[metrics[n][k] for k in ks],.32,color=C[n])
axs[0,1].set(xticks=xx,xticklabels=['Global\nmean','Global\nP10','Weak\nP10','Corridor\nP10'],ylim=(0,.45),ylabel='Positional isotropy',title='(b) Global and lower-tail dexterity')
for i,n in enumerate(names):
 count=sum(np.all(b['trajectory_errors']<=.5,axis=1).sum() for b in banks[n]);axs[1,0].bar(i,count,color=C[n],width=.55);axs[1,0].text(i,count+.5,f'{count}/48',ha='center')
axs[1,0].set(xticks=[0,1],xticklabels=list(names.values()),ylim=(0,51),ylabel='Completed paths',title='(c) Original initialization; no recovery substitutions')
pooled={}
for n in names:
 er=np.concatenate([b['trajectory_errors'].ravel() for b in banks[n]]);order=np.sort(er);pct=100*np.arange(1,len(er)+1)/len(er);axs[1,1].plot(np.maximum(order,1e-6),pct,color=C[n],label=names[n]);pooled[n]={'pooled_p95_mm':float(np.percentile(er,95)),'max_mm':float(max(er)),'checks':len(er)}
axs[1,1].axvline(.5,color='gray',ls='--',lw=1);axs[1,1].axhline(95,color='gray',ls=':',lw=1);axs[1,1].set(xscale='log',xlim=(1e-5,60),ylim=(0,101),xlabel='Tracking deviation (mm; logarithmic scale)',ylabel='Checked positions at or below error (%)',title='(d) Full error distribution, including failures')
fig.suptitle('Independent numerical performance — same controller and hardware',fontsize=15)
fig.text(.5,.035,'Two independent banks: 3,072 targets and 24 paths each. Bars average banks; dots in (a) show individual banks.',ha='center',fontsize=9)
fig.text(.5,.012,'Path completion requires every checked deviation ≤ 0.5 mm. Curve in (d) pools 3,888 checks per design; checks are correlated within paths.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.065,1,.95),h_pad=3,w_pad=3);save(fig,'02_performance_comparison')
# Recovery shown separately; no replacement of benchmark data.
fig,axes=plt.subplots(2,2,figsize=(12,9.5));diag=SRC/'path_investigation';rr=json.loads((diag/'results.json').read_text())
for col,(name,r) in enumerate(rr.items()):
 old=dict(np.load(diag/f'{name}_reproduced.npz'));new=dict(np.load(diag/f'{name}_dense_check.npz'));a,b=np.array(r['source_endpoints']);ref=h.fk(a[None]+(b-a)[None]*new['u'][:,None],h.BASE)
 ax=axes[0,col];ax.remove();ax=fig.add_subplot(2,2,col+1,projection='3d');ax.plot(*ref.T,color='black',ls='--',lw=2,label='Reference');ax.plot(*h.fk(old['states'],d).T,color='#be5935',lw=1.7,label='Original start');ax.plot(*h.fk(new['states'],d).T,color=C['proposed'],lw=1.6,label='Alternative start');ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=f"({'ab'[col]}) Bank {col+1}, path {10 if col==0 else 14}");allpoints=np.vstack([ref,h.fk(old['states'],d),h.fk(new['states'],d)]);lo=allpoints.min(0);hi=allpoints.max(0);span=np.maximum(hi-lo,1);ax.set(xlim=(lo[0]-.08*span[0],hi[0]+.08*span[0]),ylim=(lo[1]-.08*span[1],hi[1]+.08*span[1]),zlim=(lo[2]-.08*span[2],hi[2]+.08*span[2]));ax.set_box_aspect(span);ax.view_init(24,-58);ax.legend(fontsize=8)
 ax=axes[1,col];ax.plot(old['u']*100,np.maximum(old['errors'],1e-6),color='#be5935',label='Original start');ax.plot(new['u']*100,np.maximum(new['errors'],1e-6),color=C['proposed'],label='Alternative start');ax.axhline(.5,color='gray',ls='--',label='Completion threshold');ax.set(yscale='log',ylim=(1e-5,60),xlabel='Path progress (%)',ylabel='Deviation (mm; logarithmic scale)',title=f"({'cd'[col]}) Maximum: {max(old['errors']):.2f} → {max(new['errors']):.3f} mm");ax.legend(fontsize=8,loc='lower right')
fig.suptitle('Diagnostic recovery by changing the initial tube configuration',fontsize=15)
fig.text(.5,.035,'Same proposed tubes, hardware limits, 20 steps and 40-iteration budget; alternative trajectories checked at 2,001 positions.',ha='center',fontsize=9)
fig.text(.5,.012,'Post-hoc investigation of two selected failures—not an independently evaluated initialization method or an amended completion score.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.065,1,.95),h_pad=4,w_pad=3);save(fig,'03_path_failure_and_recovery')
(OUT/'plotted_statistics.json').write_text(json.dumps({'means':metrics,'pooled_tracking':pooled},indent=2))
(OUT/'captions.md').write_text('''# Figure captions and use

**Figure 1. Hardware-constrained inner-tip workspace for the original and proposed tube configurations.** Panels show the original 350/170/80 mm tubes, proposed 350/177.5/87.5 mm tubes, and their overlay with identical axes, scale and camera. Each workspace uses 131,072 feasible states from the first independent validation bank. Translucent surfaces depict the boundaries of occupied 5 mm radial/axial cells revolved about the Z axis under the unrestricted-rotation assumption; faint dots show a matched subset of actual samples. The cross marks the outside front-plate origin. Representative backbones use identical feasible exposures (120/60/55 mm) and rotations (0/0.7/−0.3 rad). Occupancy is an approximate ideal-model representation, not a guarantee of reachability throughout the enclosed volume.

**Figure 2. Independent numerical comparison under the original controller procedure.** (a) Weighted fixed-start IK success within 0.5 mm; bars average two banks and dots show each bank. Each bank contains 2,048 global targets and 512 targets in each priority region. (b) Global mean and regional lower-decile positional isotropy on frozen original reference cells, with the same normalization. (c) Completed reference trajectories: 44/48 for the original and 43/48 for the proposed tubes, with no diagnostic recovery substituted. (d) Empirical cumulative tracking-error distributions pooling 3,888 checked positions per design; correlated positions are not independent experimental replicates. Dashed vertical and dotted horizontal lines mark 0.5 mm and the 95th percentile. All results use the same measured hardware and ideal model and are not physical accuracy measurements.

**Figure 3. Follow-up diagnosis of two failed proposed-tube trajectories.** The original initialization produces maximum deviations of 12.05 and 37.84 mm. Alternative tube configurations at essentially the same starting tip positions permit the paths to be followed with maxima of 0.027 and 0.115 mm when checked at 2,001 positions. Tube geometry, hardware limits, controller iteration budget and number of motion steps are unchanged. These are post-hoc recovery examples: the alternative starts were identified after observing failure. They do not replace the original 43/48 benchmark or establish an automatic initialization method. A transition from a prescribed existing posture to the alternative start has not been evaluated.

## Dissertation placement

Use Figures 1 and 2 for the original-versus-proposed geometry comparison. Place Figure 3 in a separate failure analysis subsection immediately afterwards. The fair original/improved-controller comparison remains outstanding; do not label these figures as results of that future method.

Files are supplied as 250 dpi PNG and SVG with editable text. All units are mm unless stated. Existing research results and viewers are preserved. The frozen first candidate uses curves 0/90/65 mm and precurvatures 0/21.37/14.04 per metre, with provisional chuck-to-tip material lengths and unresolved internal-guidance/manufacturing assumptions.

Reproduce from the project directory with `MPLCONFIGDIR=/tmp/ctr-mpl .venv/bin/python tools/build_tradeoff_dissertation_figures.py`.
''')
inputs=[SRC/'mean_metrics.json',SRC/'validation_records.json',SRC/'frozen_candidate.json']+[SRC/f'{n}_validation_{rep}.npz' for n in names for rep in range(2)]+list(diag.glob('*reproduced.npz'))+list(diag.glob('*dense_check.npz'))+[diag/'results.json']
(OUT/'manifest.json').write_text(json.dumps({'source_hashes':{str(p.relative_to(h.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'note':'Frozen first candidate, no new study or selection; existing-controller evaluation and post-hoc recovery distinguished.'},indent=2));shutil.copy2(__file__,OUT/Path(__file__).name)
print('Saved three figures in PNG and SVG, captions, plotted statistics and provenance.')
