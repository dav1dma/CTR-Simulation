"""Baseline-defined regional screening with fresh, frozen-finalist validation."""
from run_fixed_start_comparison import solve
from run_broad_hardware_study import *
DOUT=Path('results/regional_comparison_20260910_supported');DEST=Path('output/regional_comparison_supported');DOUT.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)

def dump(name,x): (DOUT/name).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v)))
def wq(v,w,q):
 ix=np.argsort(v);return float(v[ix[np.searchsorted(np.cumsum(w[ix])/sum(w),q)]])
def metrics(d,s,targets,tid,regions):
 field=spatial_data(d,s);e,_,_=solve(targets,d);result={}
 for name,ids in regions.items():
  ix=np.searchsorted(field['ids'],ids);ix=np.minimum(ix,len(field['ids'])-1);ok=(field['ids'][ix]==ids)&(field['counts'][ix]>=5);v=np.where(ok,field['medians'][ix],0);w=2*(ids//1000)+1
  mask=np.isin(tid,ids);ci,ct=np.unique(tid[mask],return_counts=True);tw=np.array([(2*(c//1000)+1)/ct[np.searchsorted(ci,c)] for c in tid[mask]]);err=e[mask]
  result[name]=dict(isotropy_mean=float(np.average(v,weights=w)),isotropy_p10=wq(v,w,.1),supported_coverage=float(np.average(ok,weights=w)),success=float(np.average(err<=.5,weights=tw)),median_mm=wq(err,tw,.5),p95_mm=wq(err,tw,.95),targets=len(err))
 return dict(regions=result,volume_cm3=field['volume']),e

def compare(r,b):
 failures=[];exact=[]
 for name,a in r['regions'].items():
  base=b['regions'][name]
  for key in ['isotropy_mean','isotropy_p10','supported_coverage','success','median_mm','p95_mm']:
   if key.startswith('isotropy'):loss=base[key]-a[key];tol=.01*base[key]
   elif key in ['supported_coverage','success']:loss=base[key]-a[key];tol=.005
   else:loss=a[key]-base[key];tol=.01
   if loss>1e-10:exact.append(name+': '+key)
   if loss>tol+1e-10:failures.append(name+': '+key)
 if r['volume_cm3']<.99*b['volume_cm3']:failures.append('workspace')
 return dict(passes=len(failures)==0,failures=failures,strict_deteriorations=exact)

def run():
 if (DOUT/'final.json').exists():raise RuntimeError('Preserving completed run')
 _,discovery=h.bank(131072,13301);f=spatial_data(BASE,discovery);ref=np.array(json.loads((OUT/'reference_frozen.json').read_text())['ids']);ix=np.searchsorted(f['ids'],ref);ix=np.minimum(ix,len(f['ids'])-1);v=np.where(f['ids'][ix]==ref,f['medians'][ix],0);eligible=(f['ids'][ix]==ref)&(f['counts'][ix]>=30);cut=wq(v[eligible],2*(ref[eligible]//1000)+1,.2)
 old=np.load('results/fixed_start_comparison_20260909/targets.npz');oe=np.load('results/fixed_start_comparison_20260909/Original.npz')['errors'];oi=old['cell_ids'];ic=[]
 for c in ref:
  m=oi==c
  if sum(m)>=100 and np.mean(oe[m]>.5)>=.1:ic.append(c)
 near,_=h.region_cells();regions={'forward_corridor':near,'low_dexterity':ref[eligible&(v<=cut)],'IK_hotspots':np.array(ic,dtype=int)};regions['remaining']=np.setdiff1d(ref,np.unique(np.concatenate(list(regions.values()))))
 dump('regions.json',regions)
 protocol=dict(region_basis='original only; forward r<10 Z40..130, bottom20% volume-weighted original cell-median isotropy among cells with >=30 discovery configurations, original cells>=100targets and failure>=10%, remaining excludes union',tolerances=dict(isotropy_relative=.01,coverage_and_success_absolute=.005,residual_mm=.01,workspace_relative=.01),interpretation='numerical non-inferiority allowances, not clinical limits or statistical confidence; exact deteriorations also reported',screen_states=16384,screen_targets_per_cell=8,validation_states=262144,validation_targets_per_cell=120,validation_seed=15902,no_post_validation_retuning=True)
 dump('protocol.json',protocol)
 # Fixed compact list includes stronger/weaker and unequal curvatures, shorter sections.
 designs=[BASE,np.array([28.68,21.06,47,27.5]),A]
 designs += [np.array([km,ko,lm,lo]) for km,ko,lm,lo in [(23,18,47,27.5),(25,18,47,27.5),(27,18,47,27.5),(23,21.06,47,27.5),(25,21.06,47,27.5),(28.68,16,47,27.5),(28.68,19,47,27.5),(26,20,40,25),(28.68,21.06,35,22),(25,21.06,30,20)]]
 dump('screen_designs.json',designs)
 chosen=[]
 for c in np.unique(oi):chosen.extend(np.flatnonzero(oi==c)[:8])
 # Ensure corridor cells missing in old reference have known positions by using original old near targets.
 t=old['targets'][chosen];tid=oi[chosen];nt=np.load(OUT/'validation_targets.npz')['near'];t=np.vstack([t,nt]);tid=np.r_[tid,h.cells(nt)]
 _,s=h.bank(16384,15901);rows=[]
 for i,d in enumerate(designs):
  m,_=metrics(d,s,t,tid,regions);row=dict(index=i,design=d,metrics=m)
  if i==0:base=m
  row['comparison']=compare(m,base);rows.append(row);dump('screen.json',rows);print('Screen',i,'failures',len(row['comparison']['failures']),flush=True)
 # Choose by fewest failed constraints then weakest regional mean-isotropy ratio.
 pool=rows[1:];pool.sort(key=lambda r:(len(r['comparison']['failures']),-min(r['metrics']['regions'][k]['isotropy_mean']/base['regions'][k]['isotropy_mean'] for k in regions)))
 finalists=[rows[0],rows[1]]
 for r in pool:
  if r['index']!=1:finalists.append(r);break
 dump('frozen_finalists.json',finalists)
 # Fresh FK target stream, no reuse of screen targets/source states as initial guesses.
 _,bank=h.bank(1048576,15902);p=forward(bank,BASE);ids=h.cells(p);allcells=np.unique(np.concatenate(list(regions.values())));order=np.argsort(ids,kind='stable');u,b,c=np.unique(ids[order],return_index=True,return_counts=True);sel=[]
 for cell in allcells:
  j=np.searchsorted(u,cell)
  if j<len(u) and u[j]==cell:sel.extend(order[b[j]:b[j]+min(120,c[j])])
 sel=np.array(sel);t=np.column_stack([np.linalg.norm(p[sel,:2],axis=1),np.zeros(len(sel)),p[sel,2]]);tid=ids[sel];np.savez_compressed(DOUT/'fresh_targets.npz',targets=t,cell_ids=tid)
 _,s=h.bank(262144,15903);out=[]
 for row in finalists:
  m,e=metrics(row['design'],s,t,tid,regions)
  if row['index']==0:base=m
  r=dict(index=row['index'],design=row['design'],metrics=m,comparison=compare(m,base));out.append(r);np.savez_compressed(DOUT/f"final_{row['index']}.npz",errors=e);print('Validated',row['index'],r['comparison'],flush=True)
 dump('final.json',out)
 lines=['# Regional candidate test','', 'Regions and numerical non-inferiority allowances were fixed from original-design data before screening. Thirteen designs were screened. Original, B1 and one additional candidate were frozen and evaluated using fresh configurations and targets. No physical or global-optimum claim is made.','', 'Allowances: up to 1% relative loss in regional isotropy; 0.5 percentage points in sampled supported coverage or success; 0.01 mm increase in median/p95 numerical residual; 1% volume loss. These are pragmatic numerical allowances, not clinical tolerances. Exact deteriorations are retained in the raw results.','']
 for row in out:
  lines += [f"## {'Original' if row['index']==0 else 'B1' if row['index']==1 else 'Regional candidate C'}",f"Middle/outer curvature and curved length: {np.asarray(row['design']).tolist()} (m⁻¹, m⁻¹, mm, mm).",f"Passes every regional allowance: {row['comparison']['passes']}. Failed checks: {', '.join(row['comparison']['failures']) or 'none'}.",'','| Region | Mean isotropy | P10 isotropy | Supported cells (volume %) | IK success (%) | Median residual (mm) | P95 residual (mm) |','|---|---:|---:|---:|---:|---:|---:|']
  for name,m in row['metrics']['regions'].items():lines.append(f"| {name} | {m['isotropy_mean']:.5f} | {m['isotropy_p10']:.5f} | {100*m['supported_coverage']:.2f} | {100*m['success']:.3f} | {m['median_mm']:.6g} | {m['p95_mm']:.6g} |")
  lines.append('')
 lines += ['## Interpretation','This bounded test establishes whether these candidates meet the declared regional conditions, not whether a better design exists elsewhere. Regional metrics do not guarantee every point or every IK solution improves. Supported-cell coverage is a sampling/support proxy, distinct from geometric reachability. Regions overlap, so their scores must not be summed. The remaining-region definition excludes all hotspot/corridor cells.','', 'The fresh targets are original-design FK points in the canonical positive x–z plane. Both designs use the same 38/0/0 mm, zero-rotation start and original 40-iteration solver. Tail metrics here are global within each region; no per-cell p95 claims are made. Same provisional hardware/model limitations as the previous studies apply.']
 (DEST/'regional_test_results.md').write_text('\n'.join(lines));print('DONE',flush=True)
if __name__=='__main__':run()
