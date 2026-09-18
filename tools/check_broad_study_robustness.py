"""Post-selection sampling and solver robustness; no parameter retuning."""
from run_broad_hardware_study import *
from scipy.stats import spearmanr


def run():
    chosen=json.loads((OUT/'shortlist_frozen.json').read_text())
    frozen=json.loads((OUT/'reference_frozen.json').read_text());ids=np.array(frozen['ids'])
    rows=[];start=time.time();datasets={};references={}
    for seed,n in [(13301,262144),(13601,131072)]:
        _,s=h.bank(n,seed);datasets[seed]=s
        references[seed]=dict(ids=ids,near=spatial_data(BASE,s)['near'])
    for c in chosen:
        d=np.array(c['design']);row=dict(label=c['label'],results=[])
        for seed,s in datasets.items():
            row['results'].append(dict(seed=seed,n=len(s),**evaluate(d,s,references[seed])))
        # Finer grid: use a common independently sampled baseline at that resolution.
        s=datasets[13601];fine_ref=spatial_data(BASE,s,1.25)
        row['fine_grid']=evaluate(d,s,fine_ref,1.25)
        p=forward(s,d);m=(h.rz(p)[:,0]<10)&(p[:,2]>=40)&(p[:,2]<130)
        si,sw=h.values(s[m],d);cc=h.cells(p[m]);near,w=h.region_cells()
        row['near_cells']=[float(np.median(si[cc==cell])) if np.sum(cc==cell)>=5 else 0. for cell in near]
        # Compare source Jacobian resolution on new interior configurations.
        jac1=h.jac(s[:64],d,.01);jac2=h.jac(s[:64],d,.005)
        row['jacobian_step_max_abs_change']=float(np.max(np.abs(jac1-jac2)))
        # Harder broad-target solves: retest the same subset for all designs.
        targets=np.load(OUT/'validation_targets.npz')['broad'][:128]
        er,_,_=h.solve_batch(targets,d,seed=13701,nstarts=8,iterations=120)
        original=np.load(OUT/(c['label']+'_validation.npz'))['broad_errors'][:128]
        row['stronger_solver']=dict(n=128,original_success=float(np.mean(original<=.5)),stronger_success=float(np.mean(er<=.5)),classification_changes=int(np.sum((er<=.5)!=(original<=.5))))
        rows.append(row);write('robustness.json',rows)
        print(c['label'],row['results'][-1],row['stronger_solver'],round(time.time()-start,1),flush=True)
    rng=np.random.default_rng(13702);sample=rng.integers(0,len(w),(2000,len(w)))
    for row in rows:
        for refrow in [rows[0],rows[1]]:
            delta=np.array(row['near_cells'])-np.array(refrow['near_cells'])
            boot=np.sum(delta[sample]*w[sample],axis=1)/np.sum(w[sample],axis=1)
            row['near_paired_cell_interval_vs_'+refrow['label']]=np.percentile(boot,[2.5,97.5]).tolist()
    write('robustness.json',rows)
    write('robustness_summary.json',dict(seconds=time.time()-start,cell_bootstrap_role='conditional spatial-cell variation; not physical model uncertainty',design_parameters_unchanged=True))

if __name__=='__main__':run()
