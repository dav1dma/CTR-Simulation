"""Separate hardware-constrained fixed-start study preserving original DLS settings."""
from run_broad_hardware_study import *
R=Path('results/original_D_comparison_20260910');P=Path('output/original_D_comparison')
DESIGNS=[BASE,np.array([20.,14.,47.,27.5])]

def decode6(v):
 v=np.atleast_2d(v);f=np.clip(v[:,:3],0,1);o=f[:,2]*27.5;m=o+f[:,1]*(47-o);low=np.maximum(38,m);i=low+f[:,0]*(135-low)
 return np.column_stack([i,m,o,v[:,3:]*np.pi])

class HardwareIK(ConstrainedTipIK):
 def decode_deployment(self,f):return decode6(np.r_[f,[0,0,0]])[0,:3]/1000
 def encode_deployment(self,deployment):
  i,m,o=np.asarray(deployment)*1000;low=max(38,m)
  return np.array([(i-low)/(135-low),(m-o)/(47-o),o/27.5])

def solve(target,d):
 n=len(target);v=np.zeros((n,6));p=forward(decode6(v),d);er=np.linalg.norm(target-p,axis=1);live=np.ones(n,bool);its=np.zeros(n,int)
 for it in range(40):
  live&=er>.01;idx=np.flatnonzero(live)
  if not len(idx):break
  w=v[idx];pos=p[idx];e=target[idx]-pos;J=np.empty((len(idx),3,6))
  for j in range(6):
   step=np.full(len(idx),1e-4)
   if j<3:step[w[:,j]+1e-4>1]=-1e-4
   q=w.copy();q[:,j]+=step;q[:,:3]=np.clip(q[:,:3],0,1)
   J[:,:,j]=(forward(decode6(q),d)-pos)/step[:,None]
  change=np.linalg.solve(J.transpose(0,2,1)@J+np.eye(6),np.einsum('nji,nj->ni',J,e)[...,None])[...,0]
  change=np.clip(change,-.1,.1);accepted=np.zeros(len(idx),bool)
  for scale in [1.,.5,.25,.1]:
   todo=np.flatnonzero(~accepted)
   if not len(todo):break
   q=w[todo]+scale*change[todo];q[:,:3]=np.clip(q[:,:3],0,1);qp=forward(decode6(q),d);qe=np.linalg.norm(target[idx[todo]]-qp,axis=1);ok=qe<er[idx[todo]]-1e-9
   local=todo[ok];glob=idx[local];v[glob]=q[ok];p[glob]=qp[ok];er[glob]=qe[ok];accepted[local]=True
  its[idx]=it+1;live[idx[~accepted]]=False
 s=decode6(v);assert np.all(s[:,0]>=38-1e-8) and np.all(s[:,:3]<=[135+1e-8,47+1e-8,27.5+1e-8]) and np.all(s[:,0]>=s[:,1]-1e-8) and np.all(s[:,1]>=s[:,2]-1e-8)
 return er,s,its

def run():
 if (R/'summary.json').exists():raise RuntimeError('Preserving completed study')
 R.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True)
 protocol=dict(designs=[d.tolist() for d in DESIGNS],start_exposure_mm=[38,0,0],start_rotation_rad=[0,0,0],solver='original six-variable forward-difference DLS, hardware-adapted outer-first fractions',iterations=40,damping_mm=1,normalised_step=.1,finite_difference=1e-4,stopping_mm=.01,success_mm=.5,line_search=[1,.5,.25,.1],target_seed=19901,target_candidates=1048576,targets_per_cell_cap=170,cell_mm=2.5,median_support=30,p95_failure_support=100,primary_support_gate=.90,canonical_targets='positive x-z plane; same convention as old protocol',reference='frozen broad-study original cells',no_restarts=True,no_retuning=True,changes_from_old='hardware deployment bounds and fractions, feasible start, 2.5 mm resolution, new targets; not identical old protocol')
 (R/'protocol.json').write_text(json.dumps(protocol,indent=2))
 _,s=h.bank(1048576,19901);p=forward(s,BASE);ids=h.cells(p);ref=np.array(json.loads((OUT/'reference_frozen.json').read_text())['ids']);order=np.argsort(ids,kind='stable');unique,begin,count=np.unique(ids[order],return_index=True,return_counts=True);selected=[]
 for cell in ref:
  loc=np.searchsorted(unique,cell)
  if loc<len(unique) and unique[loc]==cell:selected.extend(order[begin[loc]:begin[loc]+min(count[loc],170)])
 selected=np.array(selected);p=p[selected];targets=np.column_stack([np.linalg.norm(p[:,:2],axis=1),np.zeros(len(p)),p[:,2]]);tid=ids[selected]
 np.savez_compressed(R/'targets.npz',targets=targets,cell_ids=tid,source_sample_indices=selected)
 # Independent implementation check against original scalar solver with hardware encoding.
 checks=[]
 for d in DESIGNS:
  test=targets[np.linspace(0,len(targets)-1,12,dtype=int)];err,st,it=solve(test,d);sol=HardwareIK(design_object(d).to_parameters(),damping_mm=1,solver_tolerance_mm=.01,max_iterations=40,finite_difference_step=1e-4,max_normalised_step=.1)
  scalar=[sol.solve(t,np.array([.038,0,0]),np.zeros(3)) for t in test];dis=float(np.max(np.abs(err-[r.position_error_mm for r in scalar])));assert dis<1e-4,dis;checks.append(dis)
 (R/'verification.json').write_text(json.dumps(dict(max_residual_difference_mm=checks,scalar_checks_per_design=12),indent=2))
 print('Verified scalar agreement',checks,'targets',len(targets),flush=True)
 summaries=[]
 for name,d in zip(['Original','D'],DESIGNS):
  errors=[];states=[];iterations=[]
  for start in range(0,len(targets),4096):
   e,st,it=solve(targets[start:start+4096],d);errors.extend(e);states.extend(st);iterations.extend(it)
  errors=np.array(errors);np.savez_compressed(R/(name+'.npz'),errors=errors,states=states,iterations=iterations)
  cell,counts=np.unique(tid,return_counts=True);weights=np.array([(2*(c//1000)+1)/counts[np.searchsorted(cell,c)] for c in tid]);weights/=weights.sum()
  ix=np.argsort(errors);cum=np.cumsum(weights[ix]);quant=lambda q:float(errors[ix[np.searchsorted(cum,q)]])
  support=float(np.sum((2*(cell[counts>=100]//1000)+1))/np.sum(2*(ref//1000)+1))
  row=dict(name=name,n=len(errors),success_fraction=float(np.sum(weights*(errors<=.5))),convergence_fraction=float(np.sum(weights*(errors<=.01))),median_mm=quant(.5),p95_mm=quant(.95),mean_mm=float(weights@errors),max_mm=float(errors.max()),p95_supported_reference_volume=support,support_gate_pass=support>=.90)
  summaries.append(row);print(row,flush=True)
 (R/'summary.json').write_text(json.dumps(summaries,indent=2))

if __name__=='__main__':run()
