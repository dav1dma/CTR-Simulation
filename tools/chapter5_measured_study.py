"""Frozen-candidate dense comparison for Chapter 5; separate from optimisation."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctr-mpl')
import json,sys,time,hashlib,shutil
from pathlib import Path
import numpy as np
import task_prioritised_tradeoff_search as q
h=q.h;OUT=h.ROOT/'output/chapter5_measured_revision';DATA=OUT/'data';DATA.mkdir(parents=True,exist_ok=True)
D=np.array(json.loads((q.OUT/'frozen_candidate.json').read_text())['design']);DESIGNS={'original':h.BASE,'proposed':D}
def write(n,o):
 (DATA/n).write_text(json.dumps(o,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a)))
def volume(ids,step=5):return float(np.sum(2*(np.asarray(ids)//1000)+1)*np.pi*step**3/1000)
def fields(states,d,step=5):
 p=h.fk(states,d);eta=h.iso(states,d);ids=h.cells(p,step);order=np.argsort(ids,kind='stable');u,start,count=np.unique(ids[order],return_index=True,return_counts=True)
 med=np.array([np.median(eta[order[a:a+n]]) for a,n in zip(start,count)])
 return dict(states=states,points=p,eta=eta,ids=ids,cells=u,counts=count,medians=med)
def choose_targets(seed,reference,cap):
 # Separate feasible source pools, no target-derived IK starting states.
 arrays=[]
 for j in range(4):arrays.append(h.samples(h.BASE,262144,seed+j))
 states=np.concatenate(arrays);p=h.fk(states,h.BASE);ids=h.cells(p);order=np.argsort(ids,kind='stable');cells,start,count=np.unique(ids[order],return_index=True,return_counts=True);selected=[]
 for c in reference:
  k=np.searchsorted(cells,c)
  if k<len(cells) and cells[k]==c:selected.extend(order[start[k]:start[k]+min(cap,count[k])])
 ix=np.array(selected);return dict(targets=p[ix],source_states=states[ix],cell_ids=ids[ix])
def solve_bank(name,design,t):
 chunks=[];ss=[]
 for j in range(0,len(t['targets']),4096):
  e,st,_=h.solve(t['targets'][j:j+4096],design);chunks.append(e);ss.append(st)
  print(name,'solved',min(j+4096,len(t['targets'])),'/',len(t['targets']),flush=True)
 return dict(errors=np.concatenate(chunks),states=np.concatenate(ss))
def local(t,errors):
 ids=t['cell_ids'];u,c=np.unique(ids,return_counts=True)
 return dict(cells=u,counts=c,median=np.array([np.median(errors[ids==a]) for a in u]),p95=np.array([np.percentile(errors[ids==a],95) for a in u]),failure=np.array([np.mean(errors[ids==a]>.5) for a in u]))
def aggregate(t,errors,cells=None):
 mask=np.ones(len(errors),bool) if cells is None else np.isin(t['cell_ids'],cells)
 ids=t['cell_ids'][mask];e=errors[mask];u,c=np.unique(ids,return_counts=True)
 if not len(e):return None
 w=(2*(ids//1000)+1)/c[np.searchsorted(u,ids)]
 return dict(n=len(e),success_pct=float(np.average(e<=.5,weights=w)*100),median_mm=h.wquant(e,w,.5),p95_mm=h.wquant(e,w,.95),mean_mm=float(np.average(e,weights=w)),max_mm=float(max(e)))
def run():
 protocol=dict(designs=DESIGNS,primary_cell_mm=5,spatial_n=262144,spatial_seed=61101,spatial_repeat_n=131072,repeat_seed=62101,spatial_support=30,reference='Original primary spatial cells; regions frozen before main comparison',targets='Full azimuth actual original-FK positions, NOT canonical positive-x-z targets; same targets for both designs. Equal cap of192 per reference cell from four independent262144 feasible pools.',target_seed=61202,target_cap=192,target_support=100,support_goal=.90,calibration_seed=61302,calibration_cap=128,regions='Corridor r<10,z40..130; weak region frozen in earlier measured study mapped to5mm cells; IK-problem cells defined using independent original-only calibration, >=100 targets and >=10% failures; remaining excludes union. Regions overlap.',solver='Existing 40-iteration DLS, damping1, step0.1, derivative1e-4, stop0.01mm, success0.5mm, no restarts, zero rotations, same75/0/0mm start for these designs.',weighting='Each target weight proportional to swept cell volume / target count in cell; failed attempts retained.',checks='Sample-count prefixes65536/131072/262144, grid2.5/5/10mm, independent spatial repeat, candidate gap108.5 sensitivity, scalar/viewer solver check.',scope='Post-selection dense characterisation; design fixed. No improved initialization implemented or retuning. Historical and new results not directly comparable.')
 if not (DATA/'protocol.json').exists():write('protocol.json',protocol)
 spatial={}
 for name,d in DESIGNS.items():
  path=DATA/f'{name}_spatial.npz'
  if not path.exists():
   f=fields(h.samples(d,262144,61101),d);np.savez_compressed(path,**f)
  spatial[name]=dict(np.load(path));print('Spatial',name,len(spatial[name]['cells']),flush=True)
 ref=spatial['original']['cells'];write('reference.json',ref)
 regions={'corridor':ref[(ref//1000<2)&(ref%1000>=8)&(ref%1000<26)],'weak':np.intersect1d(ref,q.s.REGIONS['weak'])}
 cp=DATA/'calibration_targets.npz'
 if not cp.exists():np.savez_compressed(cp,**choose_targets(61302,ref,128))
 cal=dict(np.load(cp));cp=DATA/'calibration_original.npz'
 if not cp.exists():np.savez_compressed(cp,**solve_bank('calibration original',h.BASE,cal))
 loc=local(cal,np.load(cp)['errors']);regions['ik_problem']=loc['cells'][(loc['counts']>=100)&(loc['failure']>=.1)]
 regions['remaining']=np.setdiff1d(ref,np.unique(np.concatenate(list(regions.values()))));write('regions.json',regions)
 tp=DATA/'targets.npz'
 if not tp.exists():np.savez_compressed(tp,**choose_targets(61202,ref,192))
 t=dict(np.load(tp));summary={}
 for name,d in DESIGNS.items():
  path=DATA/f'{name}_ik.npz'
  if not path.exists():np.savez_compressed(path,**solve_bank(name,d,t))
  e=np.load(path)['errors'];lf=local(t,e);np.savez_compressed(DATA/f'{name}_ik_maps.npz',**lf)
  support=volume(lf['cells'][lf['counts']>=100])/volume(ref)
  summary[name]=dict(global_ik=aggregate(t,e),regions={r:aggregate(t,e,c) for r,c in regions.items()},p95_supported_reference_fraction=support,workspace_cm3=volume(spatial[name]['cells']))
  print(name,summary[name],flush=True)
 write('summary.json',summary)
 checks={}
 for name,d in DESIGNS.items():
  f=spatial[name];convergence={}
  for n in [65536,131072,262144]:convergence[str(n)]={str(cell):volume(np.unique(h.cells(f['points'][:n],cell)),cell) for cell in [2.5,5,10]}
  rp=DATA/f'{name}_repeat.npz'
  if not rp.exists():np.savez_compressed(rp,**fields(h.samples(d,131072,62101),d))
  repeat=dict(np.load(rp));checks[name]=dict(convergence_volume_cm3=convergence,repeat_volume_cm3=volume(repeat['cells']),repeat_primary_retention=volume(np.intersect1d(f['cells'],repeat['cells']))/volume(f['cells']))
 common=np.intersect1d(spatial['original']['cells'][spatial['original']['counts']>=30],spatial['proposed']['cells'][spatial['proposed']['counts']>=30]);desc={}
 for region,ids in {'global':ref,**regions}.items():
  c=np.intersect1d(common,ids);w=2*(c//1000)+1;desc[region]={'support_fraction':volume(c)/volume(ids) if len(ids) else None,'cells':len(c)}
  for name,f in spatial.items():
   v=f['medians'][np.searchsorted(f['cells'],c)];desc[region][name]=dict(mean=float(np.average(v,weights=w)),p10=h.wquant(v,w,.1)) if len(c) else None
 write('spatial_regions.json',desc);write('sampling_checks.json',checks)
 overlay={k:volume(ids) for k,ids in [('shared',np.intersect1d(spatial['original']['cells'],spatial['proposed']['cells'])),('gained',np.setdiff1d(spatial['proposed']['cells'],spatial['original']['cells'])),('lost',np.setdiff1d(spatial['original']['cells'],spatial['proposed']['cells']))]};write('overlay.json',overlay)
 from ctr_inverse_kinematics import ConstrainedTipIK
 validation={}
 for name,d in DESIGNS.items():
  sol=ConstrainedTipIK(h.pars(d),deployment_limits=h.limits(d),max_iterations=40,damping_mm=1,solver_tolerance_mm=.01,max_normalised_step=.1);sel=np.linspace(0,len(t['targets'])-1,12,dtype=int);er=np.load(DATA/f'{name}_ik.npz')['errors'][sel];diff=[]
  for target,e in zip(t['targets'][sel],er):
   a=sol.solve(target,h.limits(d).decode(np.zeros(3)),np.zeros(3));diff.append(abs(a.position_error_mm-e))
  validation[name]=max(diff);assert max(diff)<1e-5
 write('solver_verification.json',validation)
 # Sensitivity of candidate workspace, no parameter changes.
 st=h.samples(D,262144,61101,108.5);write('gap_sensitivity.json',dict(states_identical=bool(np.array_equal(st,spatial['proposed']['states'])),reason='Both length differences below179.5mm, so nesting already implies gap<108mm'))
 write('source_hashes.json',{str(p.relative_to(h.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),h.ROOT/'tools/optimise_measured_ctr.py',h.ROOT/'measured_hardware_simulator/design_constraints.py',h.ROOT/'measured_hardware_simulator/ctr_inverse_kinematics.py']})
 print('DENSE STUDY COMPLETE',flush=True)
if __name__=='__main__':run()
