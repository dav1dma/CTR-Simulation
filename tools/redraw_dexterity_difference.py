"""Redraw the existing comparison with one map; preserve original study outputs."""
from build_original_optimised_difference_figures import *

out=Path('output/revised_dexterity_figure');out.mkdir(parents=True,exist_ok=True)
_,states=h.bank(262144,13301)
fields=[spatial_data(d,states) for d in DESIGNS]
common=np.intersect1d(fields[0]['ids'],fields[1]['ids'])
common=common[(common//1000>=0)&(common//1000<40)&(common%1000>=0)&(common%1000<54)]
support=np.ones(len(common),bool);med=[]
for f in fields:
 ix=np.searchsorted(f['ids'],common)
 support &= f['counts'][ix]>=30
 med.append(f['medians'][ix])
cid=common[support];delta=(med[1]-med[0])[support]
limit=float(np.max(abs(delta)))
summary=json.loads(Path('output/original_optimised_differences/summary_data.json').read_text())
region=next(r for r in summary['regions'] if r['region']=='Forward corridor')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(7.2,7.7))
ax=fig.add_axes([.12,.25,.63,.63]);cax=fig.add_axes([.80,.25,.035,.63])
ax.set_facecolor('#f0f0f0')
ax.imshow(arr(common,np.ones(len(common))),origin='lower',extent=[0,100,0,135],aspect='auto',cmap=ListedColormap(['#aeb5bd']),vmin=0,vmax=1,interpolation='nearest')
im=ax.imshow(arr(cid,delta),origin='lower',extent=[0,100,0,135],aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),interpolation='nearest')
ax.set(xlim=(0,65),ylim=(0,135),xlabel='Radial distance from insertion axis (mm)',ylabel='Axial distance from front plate, z (mm)')
ax.set_xticks(np.arange(0,61,10))
ax.add_patch(Rectangle((0,40),10,90,fill=False,edgecolor='#20262d',lw=1.2,linestyle='--'))
ax.annotate('Forward corridor',xy=(10,62),xytext=(26,34),fontsize=10,arrowprops={'arrowstyle':'-','color':'#343a40','lw':1},color='#343a40')
cb=fig.colorbar(im,cax=cax,ticks=[-.06,-.04,-.02,0,.02,.04,.06]);cb.set_label('Change in cell-median positional isotropy',labelpad=10)
fig.text(.12,.952,'Change in positional dexterity',fontsize=17,weight='bold')
fig.text(.12,.916,'Optimised minus original · blue: higher; red: lower',fontsize=10.5,color='#444444')
fig.text(.12,.166,'Forward corridor: mean isotropy',fontsize=11,weight='bold')
fig.text(.12,.133,f"{region['isotropy_original']:.5f} → {region['isotropy_optimised']:.5f}   ({region['isotropy_change_percent']:+.2f}%)",fontsize=13)
fig.text(.12,.099,'Mean central dexterity is largely unchanged.',fontsize=10.5,color='#444444')
fig.legend(handles=[Patch(facecolor='#aeb5bd',label='Insufficient common support'),Patch(facecolor='#f0f0f0',edgecolor='#d0d0d0',label='Outside common occupied cells')],loc='lower left',bbox_to_anchor=(.105,.018),frameon=False,fontsize=8.5,ncol=2)
for ext in ['pdf','png']:fig.savefig(out/f'dexterity_difference_revised.{ext}',dpi=300)
plt.close(fig)
caption='Change in cell-median positional isotropy between the optimised and original configurations. Blue indicates higher isotropy and red lower isotropy; white within supported cells indicates little change. Grey cells have insufficient common support, while the pale-grey background lies outside common occupied cells. Each compared cell contains at least 30 sampled configurations per design. The dashed outline marks the forward corridor ($r\\leq10$ mm, $40\\leq z\\leq130$ mm), where volume-weighted mean cell-median isotropy increased by only 0.31\\%. Differences are absolute isotropy units, not percentage changes. The original colour limits are retained.'
tex='\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=0.85\\linewidth]{dexterity_difference_revised.pdf}\n    \\caption{'+caption+'}\n    \\label{fig:isotropy_difference}\n\\end{figure}\n'
(out/'figure.tex').write_text(tex)
np.savez_compressed(out/'plotted_data.npz',cell_ids=cid,delta=delta)
(out/'README.txt').write_text('Upload dexterity_difference_revised.pdf to the Overleaf project root. Replace the previous difference figure block with figure.tex; do not add a second copy of the label. Data, 30-sample support and colour limits are unchanged. Only presentation is revised: one panel, radial axis ends at 65 mm (all common cells retained), corridor summary below map. Original analysis files are preserved.\n')
with zipfile.ZipFile(out/'revised_figure_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in ['dexterity_difference_revised.pdf','dexterity_difference_revised.png','figure.tex','README.txt']:z.write(out/n,n)
assert np.max(common//1000)*2.5+2.5<=65
print('Created revised figure; cells:',len(cid),'colour limits:',limit)
