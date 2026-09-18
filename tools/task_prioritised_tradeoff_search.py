"""Separate benchmark-prioritised trade-off study; never modifies live configuration."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
import strict_measured_search as s
h=s.h
OUT=h.ROOT/'output/task_prioritised_tradeoff_20260914'
OUT.mkdir(exist_ok=True,parents=True)
def write(n,x):
 (OUT/n).write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a)))
def score(m,b):
 return sum((m[g+'_ik_success']-b[g+'_ik_success'])*w for g,w in [('global',1),('weak',2),('corridor',2)])
def guards(m,b,paths=True):
 tests={
 'volume_loss_at_most_5pct':m['volume_cm3']>=.95*b['volume_cm3'],
 'reach_loss_at_most_5pct':m['tip_distance_max_mm']>=.95*b['tip_distance_max_mm'],
 'reference_retention_loss_at_most_3pp':m['global_cell_retention']>=b['global_cell_retention']-.03,
 'global_mean_isotropy_loss_at_most_5pct':m['global_isotropy_mean']>=.95*b['global_isotropy_mean'],
 'global_p10_isotropy_loss_at_most_10pct':m['global_isotropy_p10']>=.9*b['global_isotropy_p10'],
 'global_success_loss_at_most_2pp':m['global_ik_success']>=b['global_ik_success']-.02,
 'diagnostic_coverage_loss_at_most_1pp':all(m[g+'_diagnostic_success']>=b[g+'_diagnostic_success']-.01 for g in s.REGIONS),
 'priority_success_loss_at_most_1pp':all(m[g+'_ik_success']>=b[g+'_ik_success']-.01 for g in ['weak','corridor']),
 'weak_p10_loss_at_most_5pct':m['weak_isotropy_p10']>=.95*b['weak_isotropy_p10'],
 'meaningful_gain': ((m['weak_ik_success']+m['corridor_ik_success']-b['weak_ik_success']-b['corridor_ik_success'])/2>=.03 or (m['weak_isotropy_p10']>=1.1*b['weak_isotropy_p10'] and min(m[g+'_ik_success']-b[g+'_ik_success'] for g in ['weak','corridor'])>=0))}
 if paths:tests['path_completion_no_loss']=m['trajectory_completion']>=b['trajectory_completion']-1e-10;tests['path_p95_allowance_005mm']=m['trajectory_p95_mm']<=b['trajectory_p95_mm']+.05
 return [k for k,v in tests.items() if not v]
def search():
 write('protocol.json',dict(baseline=h.BASE,priorities='Weak-region and forward-corridor fixed-start IK success, weak-region P10 positional isotropy; protect path completion. Global volume/peak reach are secondary.',thresholds='Volume/reach -5%; global supported reference retention -3pp; global mean isotropy -5%, P10 -10%; global fixed-start success -2pp; priority success -1pp; diagnostic success -1pp; weak P10 -5%; path completion no loss, tracking P95 +0.05mm. Meaningful: priority mean success +3pp OR weak P10 +10% with no priority success loss. Engineering benchmark choices, not clinical requirements.',ranking='2 weak +2 corridor +1 global success gains. Dense eligibility before ranking; paths assessed before freeze.',screen='Prior candidates plus 64 local Sobol candidates, original/current interpolations; 512 global +128 each priority targets; top24 dense at65536 states. Up to8 finalists path-tested on48 paths.',validation='Two fresh target banks each2048 global +512 each priority; 131072 states each;24 paths each. Freeze before validation. Report both repeats; no validation-driven tuning.',assumptions=json.loads((h.OUT/'protocol.json').read_text())['assumptions'],hardware=json.loads((h.OUT/'protocol.json').read_text())['measured']))
 candidates=[h.BASE,s.CURRENT]
 old=json.loads((h.OUT/'fine_records.json').read_text())
 old.sort(key=lambda r:-(r['metrics']['weak']['success']+r['metrics']['corridor']['success']))
 for r in old[:12]:
  for f in [.25,.5,.75,1]:candidates.append(h.BASE+f*(np.array(r['design'])-h.BASE))
 bounds=np.array([[345,355],[170,182],[80,92],[55,90],[45,70],[18,25],[11,17]])
 candidates.extend(bounds[:,0]+h.sobol(64,7,51104)*(bounds[:,1]-bounds[:,0]))
 candidates.extend(np.array(r['design']) for r in json.loads((s.OUT/'frozen_finalists.json').read_text()))
 candidates=list({tuple(np.round(d,4)):np.round(d,4) for d in candidates}.values())
 t=h.targetbank(512,51102,s.REF);np.savez_compressed(OUT/'training_targets.npz',**t)
 records=json.loads((OUT/'screen_records.json').read_text()) if (OUT/'screen_records.json').exists() else []
 b=records[0]['metrics'] if records else None
 for j,d in enumerate(candidates):
  if any(r['id']==f't{j:03d}' for r in records):continue
  try:h.limits(d)
  except ValueError:continue
  m,raw=s.ikmetrics(d,t)
  if j==0:b=m
  records.append(dict(id=f't{j:03d}',design=d,metrics=m,score=score(m,b)))
  write('screen_records.json',records)
  if j%10==0:print('IK screen',j,'of',len(candidates),flush=True)
 selected=[records[0],records[1]]+sorted(records[2:],key=lambda r:-r['score'])[:24]
 dense=[]
 for r in selected:
  m,sp=s.spatial(np.array(r['design']),65536,51101);m.update(r['metrics']);r['metrics']=m
  if not dense:b=m
  r['failures']=guards(m,b,False);dense.append(r);write('dense_records.json',dense)
  print('Dense',r['id'],r['failures'],flush=True)
 eligible=[r for r in dense[1:] if not r['failures']]
 pathrows=[dense[0]]+sorted(eligible,key=lambda r:-r['score'])[:8]
 for r in pathrows:
  trs=[]
  for seed in [51103,51105]:
   m,raw=s.trajectory(np.array(r['design']),seed);trs.append(m);np.savez_compressed(OUT/f"{r['id']}_training_paths_{seed}.npz",**raw)
  r['metrics'].update({k:np.mean([x[k] for x in trs]) for k in trs[0]})
  if r['id']=='t000':b=r['metrics']
  r['failures']=guards(r['metrics'],b);print('Paths',r['id'],r['failures'],flush=True)
 write('path_selection.json',pathrows)
 good=[r for r in pathrows[1:] if not r['failures']]
 write('frozen_candidate.json',max(good,key=lambda r:r['score']) if good else None)
 print('FROZEN',max(good,key=lambda r:r['score'])['id'] if good else 'NONE',flush=True)
def validate():
 r=json.loads((OUT/'frozen_candidate.json').read_text())
 if not r:return
 results={}
 for rep in range(2):
  t=h.targetbank(2048,52102+1000*rep,s.REF);np.savez_compressed(OUT/f'validation_targets_{rep}.npz',**t)
  for name,d in [('original',h.BASE),('proposed',np.array(r['design'])),('previous',s.CURRENT)]:
   m,sp=s.spatial(d,131072,52101+1000*rep);ik,raw=s.ikmetrics(d,t);m.update(ik);tr,trraw=s.trajectory(d,52103+1000*rep);m.update(tr)
   if name=='original':b=m
   results.setdefault(name,[]).append(dict(metrics=m,failures=guards(m,b)))
   np.savez_compressed(OUT/f'{name}_validation_{rep}.npz',**raw,**trraw,**{'workspace_'+k:v for k,v in sp.items()})
   write('validation_records.json',results);print('Validate',rep,name,results[name][-1]['failures'],flush=True)
 d=np.array(r['design']);m,sp=s.spatial(d,131072,52101,108.5);ik,raw=s.ikmetrics(d,dict(np.load(OUT/'validation_targets_0.npz')),108.5);m.update(ik)
 write('gap_sensitivity.json',m)
 write('outcome.json',dict(candidate=r['design'],accepted=all(not v['failures'] for v in results['proposed']),note='No claim of global optimum; live simulator unchanged.'))
if __name__=='__main__':globals()[sys.argv[1]]()
