"""Independent illustrative multistart check; does not change study results."""
from run_original_D_comparison import *
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from CTR_superPosKin_fun_sectioned import superPosKin
DEST=Path('output/extended_reach_example');DEST.mkdir(parents=True,exist_ok=True)
_,states=h.bank(262144,13301)
points=[forward(states,d) for d in DESIGNS]
rz=[np.column_stack([np.linalg.norm(p[:,:2],axis=1),p[:,2]]) for p in points]
ids=[h.cells(p) for p in points]
gained=np.setdiff1d(np.unique(ids[1]),np.unique(ids[0]))
mask=np.isin(ids[1],gained)&(rz[1][:,0]<100)&(rz[1][:,1]>=0)&(rz[1][:,1]<135)
indices=np.flatnonzero(mask);tree=cKDTree(rz[0]);dist,_=tree.query(rz[1][indices])
chosen=[]
for ii in indices[np.argsort(dist)[::-1]]:
 if all(np.linalg.norm(rz[1][ii]-rz[1][jj])>4 for jj in chosen):chosen.append(ii)
 if len(chosen)==5:break
# Use the same feasible six-input parameterisation as the comparison.
def encode(s):
 i,m,o=s[:3];lo=max(38,m)
 return np.r_[(i-lo)/(135-lo) if lo<135 else 0,(m-o)/(47-o),o/27.5,s[3:]/np.pi]
def orient(s):
 s=s.copy();p=forward(s,DESIGNS[1])[0];s[3:]+=np.arctan2(p[1],p[0])
 s[3:]=(s[3:]+np.pi)%(2*np.pi)-np.pi
 return s
rng=np.random.default_rng(230911)
starts=np.column_stack([rng.random((128,3)),rng.uniform(-1,1,(128,3))])
lo=np.r_[np.zeros(3),[-3,-3,-3]];hi=np.r_[np.ones(3),[3,3,3]]
def search(target,d,initial):
 results=[]
 for v in initial:
  fit=least_squares(lambda q:forward(decode6(q),d)[0]-target,np.clip(v,lo,hi),bounds=(lo,hi),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  results.append((float(np.linalg.norm(fit.fun)),decode6(fit.x)[0],fit.nfev))
 return sorted(results,key=lambda x:x[0])
trials=[]
for ii in chosen:
 s=orient(states[ii]);target=forward(s,DESIGNS[1])[0]
 nearest=tree.query(rz[1][ii],k=4)[1]
 # Canonicalise original nearest states using their own predicted azimuth.
 near=[]
 for jj in nearest:
  q=states[jj].copy();p=forward(q,DESIGNS[0])[0];q[3:]+=np.arctan2(p[1],p[0]);q[3:]=(q[3:]+np.pi)%(2*np.pi)-np.pi;near.append(encode(q))
 res=search(target,DESIGNS[0],[encode(s),*near,*starts[:12]])
 trials.append((res[0][0],ii,s,target,res,near))
 print('screen',int(ii),'original best residual',res[0][0],flush=True)
best=max(trials,key=lambda x:x[0]);_,ii,s,target,screen,near=best
original=search(target,DESIGNS[0],[encode(s),*near,*starts])
optimised=search(target,DESIGNS[1],starts[:32])
original_state=original[0][1]
# Verify vectorised and separately implemented model for both displayed backbones.
def backbone(q,d):
 r=superPosKin(design_object(d).to_parameters(),{'ul':(q[:3]/1000).tolist(),'uphi':q[3:].tolist()},{'n_p':100,'isPlot':False})
 pp=[];ss=[];offset=0
 for loc,sec in zip(r[2],r[3]):
  pp.extend(np.column_stack(sec[:3])*1000);ss.extend((np.array(loc)+offset)*1000);offset+=loc[-1]
 assert np.linalg.norm(np.array(r[0])[:3]*1000-forward(q,d)[0])<1e-8
 return np.array(ss),np.array(pp)
bodies=[backbone(q,d) for q,d in zip([original_state,s],DESIGNS)]
# Ablation: restore original curved lengths, retaining optimised curvatures.
equiv=forward(states,[20,14,90,65]);ablation=float(np.max(np.linalg.norm(equiv-points[1],axis=1)))
same=forward(s,DESIGNS[0])[0]
summary={'target_mm':target.tolist(),'sample_index':int(ii),'target_cell':int(ids[1][ii]),'original_best_residual_mm':original[0][0],'original_starts':len(original),'optimised_independent_best_residual_mm':optimised[0][0],'optimised_independent_starts':len(optimised),'optimised_generating_residual_mm':float(np.linalg.norm(forward(s,DESIGNS[1])[0]-target)),'original_state':original_state.tolist(),'optimised_state':s.tolist(),'original_same_inputs_tip_mm':same.tolist(),'same_inputs_tip_difference_mm':float(np.linalg.norm(same-target)),'curved_length_ablation_max_mm':ablation,'ablation_configurations':len(states),'screened_targets':len(trials),'original_all_residuals_mm':[r[0] for r in original],'note':'Illustrative target selected for visible difference from optimised-only sampled cells. Bounded multistart local search is not proof of original unreachability. Same hardware limits; diagnostic solver differs from formal fixed-start study.'}
(DEST/'verification.json').write_text(json.dumps(summary,indent=2))
np.savez_compressed(DEST/'example_data.npz',original_rz=rz[0],optimised_rz=rz[1],original_s=bodies[0][0],original_backbone=bodies[0][1],optimised_s=bodies[1][0],optimised_backbone=bodies[1][1])
print(json.dumps({k:v for k,v in summary.items() if k!='original_all_residuals_mm'},indent=2),flush=True)
