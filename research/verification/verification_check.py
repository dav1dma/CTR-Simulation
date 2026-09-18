from pathlib import Path
import importlib.util, sys, hashlib, json, itertools
import numpy as np
root=Path.cwd(); src=root/'output/chapter5_measured_revision/source_snapshot/measured_hardware_simulator'
def module(name):
 spec=importlib.util.spec_from_file_location(name,src/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
fk=module('CTR_superPosKin_fun_sectioned'); limits=module('design_constraints'); data=module('tube_parameters')
def tip(p,e,a):return np.array(fk.superPosKin(p,{'ul':e,'uphi':a},{'n_p':5,'isPlot':False})[0][:3])*1000
base=data.build_supervisor_ctr_parameters()
rows=[]
def add(name,errors,n):
 v=float(max(errors));assert v<1e-8,(name,v);rows.append(dict(check=name,cases=n,max_tip_error_mm=v,tolerance_mm=1e-8))
add('75 mm straight inner element',[np.linalg.norm(tip(base,[.075,0,0],[0,0,0])-[0,0,75])],1)
p=dict(base,kappa_0=[0,0,0]); add('all straight, distinct deployed endpoints',[np.linalg.norm(tip(p,[.12,.07,.04],[.5,1,2])-[0,0,120])],1)
p={'n_t':1,'l_t':[[.08,.09]],'E':[75e9],'r':[[0,.00025]],'kappa_0':[19.12]};err=[]
for e,a in itertools.product([.04,.09-1e-7,.09,.09+1e-7,.14],[0,np.pi/2,.7]):
 arc=min(e,.09);straight=max(0,e-.09);k=19.12
 expected=np.array([(1-np.cos(k*arc))*np.sin(a)/k,(1-np.cos(k*arc))*np.cos(a)/k,straight+np.sin(k*arc)/k])*1000
 err.append(np.linalg.norm(tip(p,[e],[a])-expected))
add('single distal arc, including transition boundary',err,len(err))
checks=[]
for L in [(350,170,80),(350,177.5,87.5)]:
 lm=limits.MeasuredDesignLimits(L); f=np.array(list(itertools.product([0,.5,1],repeat=3)));e=lm.decode(f);d=np.array(L)-e*1000;g=d[:,:-1]-d[:,1:]-71.5
 assert np.all(d>=[175-1e-9,95-1e-9,15-1e-9]) and np.all(d<=[275+1e-9,195+1e-9,115+1e-9]);assert np.all(g>=8.5-1e-9) and np.all(g<=108+1e-9);assert np.all(e[:,:-1]>=e[:,1:]-1e-12)
 for q in e:lm.validate(q)
 assert np.allclose(lm.decode([0,0,0])*1000,[75,0,0])
 checks.append(dict(lengths_mm=L,cases=27,all_direct_distance_gap_order_checks_passed=True))
out=dict(date='2026-09-16',scope='Additional analytical and constraint verification of archived dense-study implementation; no study results or geometry retuned',files={x:hashlib.sha256((src/(x+'.py')).read_bytes()).hexdigest() for x in ['CTR_superPosKin_fun_sectioned','design_constraints','tube_parameters']},analytic=rows,constraints=checks)
(root/'results/verification').mkdir(parents=True,exist_ok=True)
(root/'results/verification/additional_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
