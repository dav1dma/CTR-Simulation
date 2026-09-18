"""Further bounded search; frozen regions, unchanged tolerances, two holdout seeds."""
from test_regional_candidates_round2 import metrics,compare
from run_broad_hardware_study import *
import hashlib
R3=Path('results/regional_comparison_20260910_extended');P3=Path('output/regional_comparison_extended')
D=np.array([20.,14.,47.,27.5])
def save(n,x): (R3/n).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v)))
def targets(seed,cap,n,regions):
 _,s=h.bank(n,seed);p=forward(s,BASE);ids=h.cells(p);order=np.argsort(ids,kind='stable');u,b,c=np.unique(ids[order],return_index=True,return_counts=True);sel=[]
 for cell in np.unique(np.concatenate(list(regions.values()))):
  j=np.searchsorted(u,cell)
  if j<len(u) and u[j]==cell:sel.extend(order[b[j]:b[j]+min(cap,c[j])])
 sel=np.array(sel);return np.column_stack([np.linalg.norm(p[sel,:2],axis=1),np.zeros(len(sel)),p[sel,2]]),ids[sel]
def merit(m,ref):
 return min(m['regions'][k]['isotropy_mean']/ref['regions'][k]['isotropy_mean'] for k in ref['regions'])
def run():
 if (R3/'final.json').exists():raise RuntimeError('preserve finished results')
 R3.mkdir(parents=True,exist_ok=True);P3.mkdir(parents=True,exist_ok=True)
 regions={k:np.array(v,dtype=int) for k,v in json.loads(Path('results/regional_comparison_20260910_round2/regions.json').read_text()).items()};save('regions.json',regions)
 designs=[BASE,D,np.array([28.68,21.06,47,27.5])]
 for km in [19.5,20.,20.5,21.,21.5,22.]:
  for ko in [13.8,14.,14.2]:designs.append(np.array([km,ko,47,27.5]))
 for km,ko in [(19.12,14.5),(19.12,15),(18.8,14.6),(19.6,14.6),(20.4,13.4),(21.5,13.2),(23,13),(24,14),(22,15)]:designs.append(np.array([km,ko,47,27.5]))
 for lc,lo in [(45,27.5),(40,27.5),(35,27.5),(47,25),(47,20),(40,22)]:
  for km,ko in [(20,14),(22,15)]:designs.append(np.array([km,ko,lc,lo]))
 unique=[]
 for d in designs:
  if not any(np.array_equal(d,x) for x in unique):unique.append(d)
 designs=unique
 save('protocol.json',dict(designs=designs,regions='exact round2 frozen memberships',tolerances='unchanged: iso 1%; coverage/success 0.005; residual0.01mm; volume1%',screen=[16384,8],refine=[65536,16],validation=[dict(seed=17902,states=262144,targets_per_cell=120),dict(seed=18902,states=131072,targets_per_cell=60)],selection='feasible against original and D, then max weakest regional mean-isotropy ratio vs D; secondary volume preference',replacement='pass original and D allowances on BOTH holdouts; >=1% forward mean isotropy OR >=5% volume gain over D on both; otherwise retain D',max_refined_new=8,max_validated_new=3,no_post_validation_retuning=True))
 t,ids=targets(17900,16,262144,regions);np.savez_compressed(R3/'training_targets.npz',targets=t,cell_ids=ids)
 take=np.concatenate([np.flatnonzero(ids==c)[:8] for c in np.unique(ids)])
 _,s=h.bank(16384,17901);rows=[]
 for i,d in enumerate(designs):
  m,_=metrics(d,s,t[take],ids[take],regions)
  if i==0:base=m
  if i==1:dm=m
  r=dict(index=i,design=d,metrics=m,original=compare(m,base))
  if i>=1:r['D']=compare(m,dm)
  rows.append(r);save('screen.json',rows);print('Screen',i+1,'/',len(designs),'baseline failures',len(r['original']['failures']),flush=True)
 pool=rows[2:];pool.sort(key=lambda r:(len(r['original']['failures'])+len(r['D']['failures']),-merit(r['metrics'],dm),-r['metrics']['volume_cm3']))
 selected=[rows[0],rows[1],*pool[:8]];_,s=h.bank(65536,17901);refined=[]
 for r in selected:
  m,_=metrics(r['design'],s,t,ids,regions)
  if r['index']==0:base=m
  if r['index']==1:dm=m
  q=dict(index=r['index'],design=r['design'],metrics=m,original=compare(m,base))
  if r['index']>=1:q['D']=compare(m,dm)
  refined.append(q);save('refinement.json',refined);print('Refined',r['index'],q['original']['passes'],flush=True)
 pool=refined[2:];pool.sort(key=lambda r:(len(r['original']['failures'])+len(r['D']['failures']),-merit(r['metrics'],dm),-r['metrics']['volume_cm3']))
 finalists=[refined[0],refined[1],*pool[:3]];save('frozen_finalists.json',finalists)
 final=[]
 for seed,n,cap in [(17902,262144,120),(18902,131072,60)]:
  t,ids=targets(seed,cap,1048576 if cap==120 else 524288,regions);np.savez_compressed(R3/f'targets_{seed}.npz',targets=t,cell_ids=ids);_,s=h.bank(n,seed+1);results=[]
  for r in finalists:
   m,e=metrics(r['design'],s,t,ids,regions)
   if r['index']==0:base=m
   if r['index']==1:dm=m
   q=dict(index=r['index'],design=r['design'],metrics=m,original=compare(m,base))
   if r['index']>=1:
    q['D']=compare(m,dm);q['near_gain']=m['regions']['forward_corridor']['isotropy_mean']/dm['regions']['forward_corridor']['isotropy_mean']-1;q['volume_gain']=m['volume_cm3']/dm['volume_cm3']-1;q['qualifies']=q['original']['passes'] and q['D']['passes'] and (q['near_gain']>=.01 or q['volume_gain']>=.05)
   results.append(q);np.savez_compressed(R3/f"result_{seed}_{r['index']}.npz",errors=e);save(f'validation_{seed}.json',results);print('Validated',seed,r['index'],q['original']['passes'],q.get('qualifies'),flush=True)
  final.append(dict(seed=seed,results=results))
 save('final.json',final)
 valid=[r['index'] for r in final[0]['results'][2:] if r['qualifies'] and next(x for x in final[1]['results'] if x['index']==r['index'])['qualifies']]
 verdict=dict(qualifying_replacements=valid,retain_D=not valid,designs_screened=len(designs),new_finalists=3)
 save('decision.json',verdict)
 lines=['# Extended regional comparison','',f"Screened {len(designs)} total configurations, refined eight alternatives plus Original/D, and validated three alternatives plus Original/D on two separate target/configuration seeds.",'','Regions and numerical allowances are unchanged. The replacement rule was fixed before evaluation: pass the original and D allowances on both holdouts and improve forward-corridor mean isotropy by at least 1% or workspace by at least 5% over D on both. This is a practical comparison rule, not a statistical-confidence or clinical criterion.','',f"Qualifying replacement indices: {valid}. Retain D: {not valid}."]
 for block in final:
  lines += ['',f"## Evaluation seed {block['seed']}",'','| Design | Middle/outer curvature | Curved lengths | Volume cm³ | Forward mean isotropy | Low-region p95 mm | Pass original | Pass D | Replacement rule |','|---|---|---|---:|---:|---:|---|---|---|']
  for r in block['results']:
   d=r['design'];m=r['metrics'];label='Original' if r['index']==0 else 'D' if r['index']==1 else 'E'+str(r['index']);lines.append(f"| {label} | {d[0]:.3f}/{d[1]:.3f} | {d[2]:.1f}/{d[3]:.1f} | {m['volume_cm3']:.2f} | {m['regions']['forward_corridor']['isotropy_mean']:.5f} | {m['regions']['low_dexterity']['p95_mm']:.5f} | {r['original']['passes']} | {r.get('D',{}).get('passes','—')} | {r.get('qualifies','—')} |")
   if r['index']>=2:lines += [f"\n{label} failed checks versus original: {r['original']['failures']}; versus D: {r['D']['failures']}.\n"]
 lines+=['','## Limits','This is a bounded computational search, not a global optimum or a physical qualification. Missing/under-supported spatial cells receive zero isotropy in regional scoring; coverage is a sample-support proxy. Hardware geometry, materials, model omissions and fixed-start conventions remain those of the preceding studies. Target-region statistics overlap and must not be added. No production configuration or original results were modified.']
 (P3/'extended_search_results.md').write_text('\n'.join(lines));save('manifest.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),production_unchanged=True));print('DONE',verdict,flush=True)
if __name__=='__main__':run()
