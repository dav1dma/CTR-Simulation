"""Render saved verified reach example; sampled envelopes are illustrative."""
from pathlib import Path
import json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path('output/extended_reach_example')
r=json.loads((P/'verification.json').read_text());data=np.load(P/'example_data.npz')
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig=plt.figure(figsize=(13,7.6));ax=fig.add_axes([.025,.23,.52,.64],projection='3d');detail=fig.add_axes([.64,.29,.31,.5])
colors=['#297eaf','#df7b35'];tube_colors=['#2478bc','#309454','#e99024']
for name,color in zip(['original','optimised'],colors):
 rz=data[name+'_rz'];bins=np.arange(0,137.5,2.5);zz=[];rr=[]
 for l,u in zip(bins[:-1],bins[1:]):
  vals=rz[(rz[:,1]>=l)&(rz[:,1]<u),0]
  if len(vals):zz.append((l+u)/2);rr.append(vals.max())
 theta=np.linspace(0,2*np.pi,80);rr=np.array(rr);zz=np.array(zz)
 X=rr[:,None]*np.cos(theta);Y=rr[:,None]*np.sin(theta);Z=np.tile(zz[:,None],(1,len(theta)))
 ax.plot_surface(X,Y,Z,color=color,alpha=.055,linewidth=0,shade=False)
 for angle in np.linspace(0,2*np.pi,8,endpoint=False):ax.plot(rr*np.cos(angle),rr*np.sin(angle),zz,color=color,alpha=.3,lw=.6)
 detail.plot(rr,zz,color=color,lw=1,alpha=.7)
for name,state,style in [('original',r['original_state'],'--'),('optimised',r['optimised_state'],'-')]:
 pp=data[name+'_backbone'][:,:3];s=data[name+'_s']
 # Visible outer, middle and inner backbone portions.
 for t,(a,b) in enumerate([(state[1],state[0]),(state[2],state[1]),(0,state[2])]):
  mask=(s>=a-1e-7)&(s<=b+1e-7)
  if mask.any():ax.plot(*pp[mask].T,color=tube_colors[t],ls=style,lw=2.3)
 detail.plot(pp[:,0],pp[:,2],color=colors[0 if name=='original' else 1],ls=style,lw=2.4)
 ax.scatter(*pp[-1],s=30,color=colors[0 if name=='original' else 1])
t=np.array(r['target_mm']);orig=data['original_backbone'][-1,:3]
ax.scatter(*t,marker='*',s=140,color='#a02067',depthshade=False,zorder=10)
ax.scatter(0,0,0,color='#222222',s=25);ax.text(0,0,-8,'Insertion origin',fontsize=8)
ax.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='z from front plate (mm)',xlim=(-70,70),ylim=(-70,70),zlim=(0,140));ax.set_box_aspect([1,1,1]);ax.view_init(elev=22,azim=-64)
ax.set_title('3D workspace and exposed robot shapes',pad=10)
detail.scatter(t[0],t[2],marker='*',s=150,color='#a02067',zorder=8)
detail.scatter(orig[0],orig[2],s=45,color=colors[0],zorder=8)
detail.plot([orig[0],t[0]],[orig[2],t[2]],color='#a02067',ls=':',lw=1.4)
detail.annotate(f"Best original residual\n{r['original_best_residual_mm']:.3f} mm",xy=((orig[0]+t[0])/2,(orig[2]+t[2])/2),xytext=(t[0]-5,t[2]+3.5),fontsize=10,arrowprops={'arrowstyle':'-','color':'#555555'})
detail.set(xlim=(t[0]-6,t[0]+3),ylim=(t[2]-5,t[2]+5),xlabel='x (mm)',ylabel='z from front plate (mm)',title='Tip close-up in the target plane')
detail.set_aspect('equal',adjustable='box');detail.grid(alpha=.2)
fig.suptitle('A verified reach in the additional sampled workspace',fontsize=18,weight='bold',y=.975)
fig.text(.5,.919,'Same tube-length and actuator constraints · model predictions, not measured hardware',ha='center',fontsize=11,color='#444444')
handles=[Line2D([],[],color=colors[0],ls='--',label='Original: best solution found'),Line2D([],[],color=colors[1],label='Optimised: verified target state'),Line2D([],[],marker='*',color='#a02067',ls='',markersize=10,label='Target')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.155),ncol=3,frameon=False)
fig.text(.5,.118,'3D tube colours: outer orange · middle green · inner blue. Dashed: original; solid: optimised.',ha='center',fontsize=9.5)
fig.text(.5,.085,f"Original: {r['original_starts']} starts, best residual {r['original_best_residual_mm']:.3f} mm.  Optimised: known generating state; independent IK also reaches target.",ha='center',fontsize=10)
fig.text(.5,.030,'Thin inset lines and translucent surfaces trace sampled outer envelopes. These do not establish reachability of every enclosed point.\nFailure of the original multistart search does not prove geometric unreachability.',ha='center',fontsize=9,color='#555555')
# Old caption retained only in source comment:
# fig.text(.5,.042,'Translucent surfaces trace sampled outer envelopes and do not establish reachability of every enclosed point.\nFailure of the original multistart search does not prove geometric unreachability.',ha='center',fontsize=9,color='#555555')
for ext in ['png','pdf']:fig.savefig(P/f'extended_reach_3d.{ext}',dpi=240)
plt.close(fig)
rows=[]
for name in ['original','optimised']:
 s=r[name+'_state'];rows.append(f"| {name.title()} | "+' / '.join(f'{x:.3f}' for x in s[:3])+' | '+' / '.join(f'{x:.2f}' for x in np.degrees((np.array(s[3:])+np.pi)%(2*np.pi)-np.pi))+' |')
report=f'''# Extended reach example

Target (x, y, z): {r['target_mm']} mm. Selected from optimised-only sampled cells in the same 262,144-state bank (seed 13301). Five spatially separated targets were screened for a visible difference; this is an illustrative selected case, not an unbiased performance estimate.

Original: {r['original_starts']} bounded least-squares starts including the optimised state, nearby original sampled states and 128 random starts. Best residual {r['original_best_residual_mm']:.9f} mm. Optimised generating state residual {r['optimised_generating_residual_mm']:.3g} mm; 32 independent random-start solves also tested, best {r['optimised_independent_best_residual_mm']:.9g} mm. These diagnostic searches use up to 300 function evaluations and do not replace the formal fixed-start study. No global unreachability proof is claimed.

| Design | Exposure inner / middle / outer (mm) | Rotation inner / middle / outer (degrees) |
|---|---|---|
'''+ '\n'.join(rows)+f'''

## Why the apparently straighter free tubes can reach farther

Under the nominal caps, exposed middle and outer lengths do not exceed 47 and 27.5 mm. Both the original 90/65 mm and optimised 47/27.5 mm distal curved sections therefore cover the complete exposed portions. Their proximal straight lengths in the free-tube drawing are not exposed in this study.

Restoring curved lengths to 90/65 mm while retaining optimised curvatures 20/14 m^-1 gave a maximum tip difference of {r['curved_length_ablation_max_mm']:.6g} mm over {r['ablation_configurations']:,} shared configurations. Thus the shorter curved sections are not the cause of this nominal model's workspace gain. The changed intrinsic curvatures alter stiffness-weighted bending in the overlap sections and the direction of the distal straight inner element. The middle curvature increases from 19.12 to 20 m^-1; outer changes only from 14.04 to 14 m^-1.

Original tubes at exactly the optimised actuator settings put the tip at {r['original_same_inputs_tip_mm']} mm, a displacement of {r['same_inputs_tip_difference_mm']:.6f} mm from the target. Reoptimising the original settings reduces its best residual to the figure's value.

Backbones were generated using the separate section-aware model and checked against the study forward model to below 1e-8 mm. Envelope surfaces are illustrative radial outer boundaries and may bridge unsampled gaps; they are not exact continuous reachable volumes. The target is verified by FK independently of the drawn surfaces. Original files and results were not changed.
'''
(P/'explanation.md').write_text(report)
caption='Illustrative target reached by the optimised configuration in an optimised-only sampled workspace cell. The 3D view shows the individual exposed backbone portions and sampled outer workspace envelopes; the inset resolves the small tip-position difference. The original configuration is shown at the best solution found from '+str(r['original_starts'])+' starting states, with residual '+f"{r['original_best_residual_mm']:.3f}"+' mm. This local-search result does not prove original-design unreachability. The target was selected to illustrate a difference and is not representative of average performance. All geometry is numerical.'
(P/'figure.tex').write_text('\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{extended_reach_3d.pdf}\n\\caption{'+caption+'}\n\\label{fig:extended-reach-example}\n\\end{figure}\n')
with zipfile.ZipFile(P/'extended_reach_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in ['extended_reach_3d.pdf','extended_reach_3d.png','figure.tex','explanation.md','verification.json']:z.write(P/n,n)
print(report)
