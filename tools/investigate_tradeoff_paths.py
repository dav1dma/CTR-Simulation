"""Post-selection diagnostics of two failed paths; does not change validation or viewer."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import task_prioritised_tradeoff_search as q
h=q.h;ROOT=h.ROOT;SRC=q.OUT;OUT=SRC/'path_investigation';OUT.mkdir(exist_ok=True)
d=np.array(json.loads((SRC/'frozen_candidate.json').read_text())['design']);lim=h.limits(d)
def write(n,x):
 (OUT/n).write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a)))
def encode(st):return np.r_[lim.encode(st[:3]/1000),st[4:]/np.pi]
def trace(a,b,initial,steps=20,iterations=40,method='existing'):
 x=encode(initial);st=initial.copy();states=[st];errors=[np.linalg.norm(h.fk(st[None],d)[0]-h.fk(a[None],h.BASE)[0])];us=[0.]
 for j in range(1,steps+1):
  u=j/steps;target=h.fk((a+(b-a)*u)[None],h.BASE)[0];old=st.copy()
  if method=='existing':er,ss,xx=h.solve(target[None],d,x[None],iterations);st=ss[0];x=xx[0]
  else:
   sol=least_squares(lambda v:h.fk(h.decode(v,d),d)[0]-target,x,bounds=([0,0,0,-np.inf,-np.inf],[1,1,1,np.inf,np.inf]),max_nfev=200,ftol=1e-11,xtol=1e-11,gtol=1e-11)
   x=sol.x;st=h.decode(x,d)[0]
  for f in [.25,.5,.75,1]:
   u=(j-1+f)/steps;mid=old+(st-old)*f;lim.validate(mid[:3]/1000)
   errors.append(np.linalg.norm(h.fk(mid[None],d)[0]-h.fk((a+(b-a)*u)[None],h.BASE)[0]));us.append(u)
  states.append(st)
 return dict(states=np.array(states),errors=np.array(errors),u=np.array(us))
def stats(r):return dict(max_mm=float(max(r['errors'])),p95_mm=float(np.percentile(r['errors'],95)),above_05=int(sum(r['errors']>.5)),checks=len(r['errors']))
def investigate():
 write('protocol.json',dict(purpose='Diagnose two large candidate deviations without modifying old results or active simulator',design=d,cases=['bank0 path10','bank1 path14'],experiments=['reproduce existing warm-start path','200 iterations same waypoints','200 smaller steps same controller','independent two-nearest-start solves at81 reference positions','bounded trust-region least squares warm-start path','reverse continuation from independently solved endpoint'],limits='Same hardware and exterior model. Pointwise reachability does not imply a continuous feasible path. Post-hoc diagnostics are not new held-out performance claims.'))
 results={}
 for rep,idx in [(0,9),(1,13)]:
  name=f'bank{rep}_path{idx+1}';saved=dict(np.load(SRC/f'proposed_validation_{rep}.npz'));a,b=saved['trajectory_source_endpoints'][idx];initial=saved['trajectory_states'][0,idx];u=np.linspace(0,1,81);targets=h.fk(a[None]+(b-a)[None]*u[:,None],h.BASE)
  point_errors,point_states=h.diagnostic(targets,d)
  experiments={}
  for tag,steps,it,method in [('reproduced',20,40,'existing'),('more_iterations',20,200,'existing'),('smaller_steps',200,40,'existing'),('trust_region',20,200,'trust')]:
   r=trace(a,b,initial,steps,it,method);experiments[tag]=stats(r);np.savez_compressed(OUT/f'{name}_{tag}.npz',**r)
   if tag=='reproduced':assert np.max(abs(r['errors']-saved['trajectory_errors'][idx]))<1e-7
   print(name,tag,experiments[tag],flush=True)
  r=trace(b,a,point_states[-1],200,40);experiments['reverse_small_steps']=stats(r);np.savez_compressed(OUT/f'{name}_reverse.npz',**r)
  # Use a continuous reverse-traced start as an alternative start for forward traversal.
  r2=trace(a,b,r['states'][-1],200,40);experiments['alternative_start_small_steps']=stats(r2);np.savez_compressed(OUT/f'{name}_alternative_start.npz',**r2)
  st=saved['trajectory_states'][:,idx];latent=np.array([encode(x) for x in st]);dist=d[:3]-st[:,:3];gaps=dist[:,:-1]-dist[:,1:]-71.5
  np.savez_compressed(OUT/f'{name}_pointwise.npz',targets=targets,errors=point_errors,states=point_states,original_latent=latent,original_gaps=gaps,reference_u=u)
  results[name]=dict(experiments=experiments,pointwise=dict(max_mm=float(max(point_errors)),failures=int(sum(point_errors>.5))),original_first_failed_check=int(np.flatnonzero(saved['trajectory_errors'][idx]>.5)[0]),latent=latent,gaps=gaps,source_endpoints=[a,b])
  write('results.json',results);print(name,'pointwise',results[name]['pointwise'],'reverse',experiments['reverse_small_steps'],'alt',experiments['alternative_start_small_steps'],flush=True)

if __name__=='__main__':investigate()
