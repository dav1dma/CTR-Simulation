"""Illustrate the saved comparison's intrinsic tube geometry, without retuning."""
from pathlib import Path
import json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
out=Path('output/revised_tube_shapes');out.mkdir(parents=True,exist_ok=True)
ds=json.loads(Path('results/original_D_comparison_20260910/protocol.json').read_text())['designs']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(11,11))
fig.subplots_adjust(left=.08,right=.97,bottom=.21,top=.86,hspace=.65,wspace=.23)
colors=['#247c9c','#c86629'];styles=['-','--'];names=['Original','Optimised']
for j,(tube,total) in enumerate([('Middle',170),('Outer',80)]):
 for row in range(2):
  ax=axs[row,j]
  for idx,d in enumerate(ds):
   k=d[j]/1000;lc=d[j+2];straight=total-lc
   t=np.linspace(0,lc,500);x=(1-np.cos(k*t))/k;z=np.sin(k*t)/k
   offset=straight if row==0 else 0
   if row==0:ax.plot([0,0],[0,straight],c=colors[idx],ls=styles[idx],lw=2.1)
   ax.plot(x,z+offset,c=colors[idx],ls=styles[idx],lw=2.1)
   ax.scatter([0],[offset],s=32,c=colors[idx],marker=['o','D'][idx],zorder=5)
   theta=k*lc
   ax.annotate('',xy=(x[-1]+7*np.sin(theta),z[-1]+offset+7*np.cos(theta)),xytext=(x[-1],z[-1]+offset),arrowprops={'arrowstyle':'->','color':colors[idx],'lw':1.6})
   if row==0:
    text=f'{names[idx]}: straight {straight:g} mm; curved {lc:g} mm'
   else:
    text=f'{names[idx]}: κ = {d[j]:g} m⁻¹; R = {1/k:.2f} mm; θ = {np.degrees(theta):.2f}°'
   ax.text(0,-.24-idx*.09,text,transform=ax.transAxes,color=colors[idx],fontsize=9.2)
  ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
  ax.set_xlabel('Lateral offset (mm)')
  ax.set_ylabel('Axial distance from grip (mm)' if row==0 else 'Axial distance from curve start (mm)')
  if row==0:ax.set_xlim(-8,85);ax.set_ylim(-5,178)
  else:ax.set_xlim(-8,85);ax.set_ylim(-5,75)
  ax.set_title(f'{"(a)" if j==0 and row==0 else "(b)" if row==0 else "(c)" if j==0 else "(d)"} {tube}: '+('complete shape' if row==0 else 'aligned curved sections'),fontsize=12,pad=12)
fig.suptitle('Tube curvature and curved length',fontsize=20,weight='bold',y=.978)
fig.text(.5,.94,'Individual unloaded shapes · equal axis scaling · shared limits within each row',ha='center',fontsize=11,color='#444444')
fig.legend(handles=[Line2D([0],[0],c=colors[0],lw=2,label='Original'),Line2D([0],[0],c=colors[1],lw=2,ls='--',label='Optimised'),Line2D([0],[0],c='#555555',marker='o',ls='',label='Curve start (diamond: optimised)')],loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,frameon=False,fontsize=10)
fig.text(.5,.025,'Inner element unchanged: straight, 350 mm. Arrows show tip tangent direction.\nκ: intrinsic curvature    R: bend radius    θ = κL_c: total bend angle',ha='center',fontsize=10,color='#444444')
for ext in ['png','pdf']:fig.savefig(out/f'tube_shapes_revised.{ext}',dpi=250)
plt.close(fig)
caption='Intrinsic shapes of the original and optimised middle and outer tubes. The top row shows complete shapes from the assumed gripping point; the bottom row aligns the starts of the curved sections. Markers identify curvature transitions and arrows indicate tip tangents. Equal axis scaling and matching limits are used within each row. The optimised middle tube has slightly tighter curvature but a smaller total bend angle because its curved section is shorter. The outer curvatures are almost identical. The straight inner element is unchanged and omitted. These are individual unloaded geometries, not assembled robot shapes or physical measurements.'
(out/'figure.tex').write_text('\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=\\linewidth]{tube_shapes_revised.pdf}\n    \\caption{'+caption+'}\n    \\label{fig:optimised_tube_shapes}\n\\end{figure}\n')
with zipfile.ZipFile(out/'tube_shapes_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['tube_shapes_revised.png','tube_shapes_revised.pdf','figure.tex']:z.write(out/name,name)
print(out)
