"""Explicit second validation round of the remaining training-qualified candidate."""
import json
import numpy as np
import task_prioritised_tradeoff_search as q
h=q.h;s=q.s
old=q.OUT;q.OUT=old/'reserve_validation';q.OUT.mkdir(exist_ok=True)
r=next(a for a in json.loads((old/'path_selection.json').read_text()) if a['id']=='t019')
q.write('protocol_amendment.json',dict(reason='The highest training-ranked candidate failed original held-out path protections. Evaluate the only remaining candidate already passing all training gates. No redesign or relaxed limits.',candidate=r['design'],spatial_seeds=[56101,57101],target_seeds=[56102,57102],path_seeds=[56103,57103],selection='Selected from original training results only; wholly new validation banks. Sequential exploration is disclosed; no general claim that a single test proves optimality.'))
q.write('frozen_candidate.json',r);q.write('protocol.json',json.loads((old/'protocol.json').read_text()))
results={}
for rep in range(2):
 t=h.targetbank(2048,56102+1000*rep,s.REF);np.savez_compressed(q.OUT/f'validation_targets_{rep}.npz',**t)
 for name,d in [('original',h.BASE),('proposed',np.array(r['design'])),('previous',s.CURRENT)]:
  m,sp=s.spatial(d,131072,56101+1000*rep);ik,raw=s.ikmetrics(d,t);m.update(ik);tr,trraw=s.trajectory(d,56103+1000*rep);m.update(tr)
  if name=='original':b=m
  results.setdefault(name,[]).append(dict(metrics=m,failures=q.guards(m,b)))
  np.savez_compressed(q.OUT/f'{name}_validation_{rep}.npz',**raw,**trraw,**{'workspace_'+k:v for k,v in sp.items()})
  q.write('validation_records.json',results);print('Reserve',rep,name,results[name][-1]['failures'],flush=True)
d=np.array(r['design']);m,sp=s.spatial(d,131072,56101,108.5);ik,raw=s.ikmetrics(d,dict(np.load(q.OUT/'validation_targets_0.npz')),108.5);m.update(ik);q.write('gap_sensitivity.json',m)
q.write('outcome.json',dict(candidate=r['design'],accepted=all(not v['failures'] for v in results['proposed']),note='Second sequential validation; first failure retained. Live simulator unchanged.'))
