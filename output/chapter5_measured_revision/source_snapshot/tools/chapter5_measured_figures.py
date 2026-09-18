"""Draft-style maps from the new dense measured-hardware study."""
import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,TwoSlopeNorm,SymLogNorm
from matplotlib.patches import Rectangle,Patch
import chapter5_measured_study as s
DATA=s.DATA;OUT=s.OUT;FIG=OUT/'overleaf/ch5';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
F={n:dict(np.load(DATA/f'{n}_spatial.npz')) for n in s.DESIGNS};I={n:dict(np.load(DATA/f'{n}_ik_maps.npz')) for n in s.DESIGNS};ref=np.array(json.loads((DATA/'reference.json').read_text()));names=['original','proposed'];labels=['Original','Proposed'];colors=['#3073ad','#149383'];SHAPE=(38,25);EXT=[0,125,0,190]
def arr(ids,v):
 a=np.full(SHAPE,np.nan);ids=np.asarray(ids,int);a[ids%1000,ids//1000]=v;return a
def axes(ax):
 ax.set(xlim=(0,125),ylim=(0,190),xlabel='Radial distance (mm)',ylabel='Z from outside front plate (mm)');ax.set_facecolor('white')
def save(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def pair(name,fields,title,cmap,norm=None,vmin=None,vmax=None,label='',support=30,reference=False):
 fig,axs=plt.subplots(1,2,figsize=(8.4,4.9),sharex=True,sharey=True,layout='constrained')
 for ax,n,t in zip(axs,names,labels):
  ids,values,count=fields[n];base=ref if reference else ids
  ax.imshow(arr(base,np.ones(len(base))),origin='lower',extent=EXT,aspect='auto',cmap=ListedColormap(['#b6bcc2']),vmin=0,vmax=1,interpolation='nearest')
  m=count>=support;im=ax.imshow(arr(ids[m],values[m]),origin='lower',extent=EXT,aspect='auto',cmap=cmap,norm=norm,vmin=vmin,vmax=vmax,interpolation='nearest');axes(ax);ax.set_title(t)
 fig.suptitle(title);fig.colorbar(im,ax=axs,label=label,shrink=.85);save(fig,name)
fig,axs=plt.subplots(1,3,figsize=(10,4.5),layout='constrained')
for j,ax in enumerate(axs):
 for n,d in s.DESIGNS.items():
  total=d[j];curve=0 if j==0 else d[j+2];k=0 if j==0 else d[j+4]/1000;straight=total-curve;z=np.linspace(0,total,300);u=np.maximum(z-straight,0)
  xx=np.zeros(len(z)) if k==0 else (1-np.cos(k*u))/k;zz=z if k==0 else np.minimum(z,straight)+np.sin(k*u)/k
  ax.plot(xx,zz,label=n.title(),color=colors[names.index(n)],ls='-' if n=='original' else '--')
 ax.set_title(['Inner element','Middle tube','Outer tube'][j]);ax.set(xlabel='Lateral offset (mm)',ylabel='Axial position from chuck (mm)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
axs[2].legend();fig.suptitle('Intrinsic tube shapes: straight section followed by distal curved section');save(fig,'tube_shapes')
pair('workspace',{n:(F[n]['cells'],np.ones(len(F[n]['cells'])),F[n]['counts']) for n in names},'Sampled inner-tip workspace | same measured hardware',ListedColormap(['#31869b']),vmin=0,vmax=1,label='Occupied cell',support=1)
a=F['original']['cells'];b=F['proposed']['cells'];shared=np.intersect1d(a,b);gain=np.setdiff1d(b,a);loss=np.setdiff1d(a,b)
fig,ax=plt.subplots(figsize=(6.5,5.4),layout='constrained')
for ids,c in [(shared,'#b5bec4'),(gain,'#239879'),(loss,'#df8438')]:ax.imshow(arr(ids,np.ones(len(ids))),origin='lower',extent=EXT,aspect='auto',cmap=ListedColormap([c]),vmin=0,vmax=1,interpolation='nearest')
axes(ax);ax.add_patch(Rectangle((0,40),10,90,fill=False,color='black',ls='--',lw=1));ax.legend(handles=[Patch(color=c,label=l) for c,l in [('#b5bec4','Shared occupied cells'),('#239879','Proposed only'),('#df8438','Original only')]],fontsize=8,loc='upper right');ax.set_title('Workspace overlay | 5 mm cells');save(fig,'workspace_overlay')
pair('isotropy_median',{n:(F[n]['cells'],F[n]['medians'],F[n]['counts']) for n in names},'Median positional isotropy | at least 30 states per cell','viridis',vmin=0,vmax=max(float(F[n]['medians'][F[n]['counts']>=30].max()) for n in names),label='Range-normalised positional isotropy')
common=np.intersect1d(a,b);support=np.ones(len(common),bool);med=[]
for n in names:
 ix=np.searchsorted(F[n]['cells'],common);support&=F[n]['counts'][ix]>=30;med.append(F[n]['medians'][ix])
ids=common[support];delta=(med[1]-med[0])[support];lim=float(np.max(abs(delta)))
fig,axs=plt.subplots(1,2,figsize=(8.4,5),layout='constrained')
for ax in axs:
 ax.imshow(arr(common,np.ones(len(common))),origin='lower',extent=EXT,aspect='auto',cmap=ListedColormap(['#b6bcc2']),vmin=0,vmax=1,interpolation='nearest');im=ax.imshow(arr(ids,delta),origin='lower',extent=EXT,aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),interpolation='nearest');axes(ax)
axs[0].set_title('Common workspace');axs[0].add_patch(Rectangle((0,40),10,90,fill=False,color='black',ls='--',lw=1));axs[1].set(xlim=(0,10),ylim=(40,130),title='Forward-corridor detail');fig.colorbar(im,ax=axs,label='Proposed minus original isotropy');fig.suptitle('Where positional dexterity changes | identical colour scale');save(fig,'dexterity_difference')
maxerr=max(float(I[n]['p95'][I[n]['counts']>=100].max()) for n in names);norm=SymLogNorm(linthresh=.01,linscale=1,vmin=0,vmax=maxerr)
for key,threshold,title in [('median',30,'Median fixed-start IK residual'),('p95',100,'95th-percentile fixed-start IK residual')]:pair('ik_'+key,{n:(I[n]['cells'],I[n][key],I[n]['counts']) for n in names},title,'RdYlGn_r',norm=norm,label='Numerical residual (mm)',support=threshold,reference=True)
pair('ik_failure',{n:(I[n]['cells'],I[n]['failure']*100,I[n]['counts']) for n in names},'Fixed-start failure fraction | residual > 0.5 mm','YlOrRd',vmin=0,vmax=100,label='Targets exceeding threshold (%)',support=100,reference=True)
fig,ax=plt.subplots(figsize=(6.6,5.2),layout='constrained');f=I['original'];ax.imshow(arr(ref,np.ones(len(ref))),origin='lower',extent=EXT,aspect='auto',cmap=ListedColormap(['#b6bcc2']),vmin=0,vmax=1);im=ax.imshow(arr(f['cells'],f['counts']),origin='lower',extent=EXT,aspect='auto',cmap='viridis',vmin=0,vmax=192);axes(ax);fig.colorbar(im,ax=ax,label='Identical targets per cell');ax.set_title('IK map support | 100 targets required for P95/failure');save(fig,'target_support')
checks=json.loads((DATA/'sampling_checks.json').read_text());fig,axs=plt.subplots(1,2,figsize=(8.8,4),layout='constrained')
for n,c in zip(names,colors):
 for cell,style in [('2.5',':'),('5','-'),('10','--')]:
  nums=[65536,131072,262144];axs[0].plot(np.array(nums)/1000,[checks[n]['convergence_volume_cm3'][str(k)][cell] for k in nums],ls=style,c=c,label=f'{n.title()}, {cell} mm')
 axs[1].bar(names.index(n),checks[n]['repeat_primary_retention']*100,color=c)
axs[0].set(xlabel='Feasible states (thousands)',ylabel='Occupied annular volume (cm³)',title='Grid and sample-count sensitivity');axs[0].legend(fontsize=7);axs[1].set(xticks=[0,1],xticklabels=labels,ylim=(0,105),ylabel='Primary occupied volume retained (%)',title='Independent 131,072-state repeat');save(fig,'sampling_support')
# Preserve historical reference figure; new interactive screenshot is clearly assigned to candidate.
shutil.copy2(s.h.ROOT/'output/chapter5_numerical_studies/figures/ch5/ch7_workspace_3d.png',FIG/'historical_workspace.png')
shutil.copy2(qpath:=s.q.OUT/'workspace_viewer/optimised_viewer.png',FIG/'proposed_viewer.png')
shutil.copy2(s.q.OUT/'dissertation_figures/03_path_failure_and_recovery.png',FIG/'path_recovery.png')
print('Chapter figures saved',FIG,flush=True)
