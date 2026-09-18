"""Count-only support top-up; original results archived, no design/solver changes."""
import json,shutil
import numpy as np
import chapter5_measured_study as s
D=s.DATA;archive=D/'initial_support_83pct';archive.mkdir(exist_ok=True)
for name in ['targets.npz','original_ik.npz','proposed_ik.npz','original_ik_maps.npz','proposed_ik_maps.npz','summary.json']:
 if not (archive/name).exists():shutil.copy2(D/name,archive/name)
s.write('support_addendum.json',dict(reason='Initial100-target support83.64% below predeclared90% goal. Add independent original-FK targets to deficient cells; stop once goal met or16 batches exhausted.',seed_start=61402,batch_states=262144,max_batches=16,cell_cap=192,selection='Cell counts only, no candidate outcomes or error-dependent selection. Existing calibration regions remain frozen.'))
t=dict(np.load(archive/'targets.npz'));ref=np.array(json.loads((D/'reference.json').read_text()));base_n=len(t['targets']);log=[]
for j in range(16):
 cells,counts=np.unique(t['cell_ids'],return_counts=True);support=s.volume(cells[counts>=100])/s.volume(ref)
 if support>=.90:break
 st=s.h.samples(s.h.BASE,262144,61402+j);pts=s.h.fk(st,s.h.BASE);ids=s.h.cells(pts);order=np.argsort(ids,kind='stable');u,start,count=np.unique(ids[order],return_index=True,return_counts=True);selected=[]
 for c in ref:
  k=np.searchsorted(cells,c);have=counts[k] if k<len(cells) and cells[k]==c else 0
  if have>=192:continue
  k=np.searchsorted(u,c)
  if k<len(u) and u[k]==c:selected.extend(order[start[k]:start[k]+min(192-have,count[k])])
 ix=np.array(selected);t={k:np.concatenate([t[k],v]) for k,v in dict(targets=pts[ix],source_states=st[ix],cell_ids=ids[ix]).items()};cells,counts=np.unique(t['cell_ids'],return_counts=True);support=s.volume(cells[counts>=100])/s.volume(ref);log.append(dict(batch=j+1,added=len(ix),support=support));print('Topup',log[-1],flush=True)
np.savez_compressed(D/'targets.npz',**t);s.write('support_topup_log.json',log)
for name,d in s.DESIGNS.items():
 old=dict(np.load(archive/f'{name}_ik.npz'));new=s.solve_bank(name+' additional',d,{k:v[base_n:] for k,v in t.items()});np.savez_compressed(D/f'{name}_ik.npz',**{k:np.concatenate([old[k],new[k]]) for k in old})
s.run()
