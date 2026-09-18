"""Run with ../.venv/bin/python -m unittest discover -s measured_hardware_simulator."""
import unittest
import tempfile
import csv
from pathlib import Path
import numpy as np
from ctr_operating_profile import OperatingProfile, DeploymentLimits
from ctr_inverse_kinematics import ConstrainedTipIK
from ctr_motion_planner import MotionPlan, build_motion_route, route_state_at_time, DIRECT_STRATEGY, RETRACT_STRATEGY
from ctr_viewer_workspace import profile_workspace, workspace_identity
from interactive_ctr_vispy import VisPyCTRViewer
from coordinated_control import coordinated_move
from design_constraints import MeasuredDesignLimits

class MeasuredTests(unittest.TestCase):
    def setUp(self):
        self.profile=OperatingProfile(); self.limits=self.profile.limits

    def test_design_dependent_limits_match_direct_measured_geometry(self):
        rng=np.random.default_rng(509)
        for totals in ([350,170,80],[350,180,90],[365,210,110],[335,185,85]):
            for gap in (108.,108.5):
                lim=MeasuredDesignLimits(tuple(totals),gap)
                e=lim.decode(rng.random((300,3)));lim.validate(e)
                distances=np.asarray(totals)-e*1000
                self.assertTrue(np.all(distances>=[175-1e-7,95-1e-7,15-1e-7]))
                self.assertTrue(np.all(distances<=[275+1e-7,195+1e-7,115+1e-7]))
                gaps=distances[:,:-1]-distances[:,1:]-71.5
                self.assertTrue(np.all((gaps>=8.5-1e-7)&(gaps<=gap+1e-7)))
                for row in e[::30]:
                    np.testing.assert_allclose(lim.decode(lim.encode(row)),row,atol=1e-12)
                    for tube in range(3):
                        moved=coordinated_move(lim,row,tube,150)
                        lim.validate(np.linspace(row,moved,30))

    def test_selected_design_ik_routes_and_cache_provenance(self):
        p=OperatingProfile(design='optimised');p.limits.validate(p.reset_m)
        self.assertNotEqual(p.identity,self.profile.identity)
        rng=np.random.default_rng(903);a,b=p.limits.decode(rng.random((2,3)))
        sol=ConstrainedTipIK(p.parameters,deployment_limits=p.limits,max_iterations=60)
        rot=np.array([0.,.3,-.4]);target=sol.forward_tip_mm(a,rot)
        answer=sol.solve(target,a,rot);self.assertTrue(answer.reached)
        plan=MotionPlan(np.zeros(3),a,rot,b,np.zeros(3),np.zeros(3),0,0,'test',0,0,{},np.zeros(3))
        for strategy in (DIRECT_STRATEGY,RETRACT_STRATEGY):
            route=build_motion_route(plan,strategy,deployment_limits=p.limits)
            for t in np.linspace(0,route.total_duration_s,101):p.limits.validate(route_state_at_time(route,t)[0])
        with tempfile.TemporaryDirectory() as d:
            bank=profile_workspace(p,sample_count=80,cache_root=d)
            p.limits.validate(bank.deployment_m)

    def test_coordinated_followers_and_selected_priority(self):
        # Rear advancing forces both following stages; minimum necessary changes.
        out=coordinated_move(self.limits,np.array([.100,0.,0.]),0,20)
        np.testing.assert_allclose(out,[.120,.020,.010],atol=1e-12)
        # Middle retracting needs rear retraction and outer retraction for ordering.
        out=coordinated_move(self.limits,np.array([.140,.055,.050]),1,-20)
        np.testing.assert_allclose(out,[.135,.035,.035],atol=1e-12)
        # Free movement leaves both followers exactly unchanged.
        out=coordinated_move(self.limits,np.array([.130,.050,.045]),0,1)
        np.testing.assert_allclose(out,[.131,.050,.045],atol=1e-12)

    def test_coordinated_random_paths_and_end_stops(self):
        rng=np.random.default_rng(123)
        for gap in (108.,108.5):
            limits=DeploymentLimits(max_gap_mm=gap)
            for _ in range(150):
                e=limits.decode(rng.random(3)); tube=int(rng.integers(3)); amount=rng.uniform(-200,200)
                result=coordinated_move(limits,e,tube,amount)
                self.assertAlmostEqual(result[tube],np.clip(e[tube]+amount/1000,limits.lower_m[tube],limits.upper_m[tube]))
                for fraction in np.linspace(0,1,11):limits.validate(e+fraction*(result-e))

    def test_sampling_encode_and_physical_equivalence(self):
        rng=np.random.default_rng(19)
        for gap in (108.,108.5):
            limits=DeploymentLimits(max_gap_mm=gap)
            e=limits.decode(rng.random((20000,3))); limits.validate(e)
            d=np.array([350.,170.,80.])-e*1000
            g=d[:,:2]-d[:,1:]-71.5
            self.assertTrue(np.all((g>=8.5-1e-9)&(g<=gap+1e-9)))
            for row in e[::200]: np.testing.assert_allclose(limits.decode(limits.encode(row)),row,atol=1e-12)

    def test_boundaries_and_no_negative_clipping(self):
        for bad in ([.175,0,0],[.120,.060,.020],[.074,0,0],[.075,-1e-12,0],[.075,.075,.065]):
            with self.assertRaises(ValueError): self.limits.validate(bad)
        self.limits.validate([.0755,.075,.065])
        DeploymentLimits(max_gap_mm=108.5).validate([.075,.075,.065])
        np.testing.assert_allclose(self.profile.carriage_displacement_mm(self.profile.reset_m),[0,25,35])

    def test_manual_motion_cannot_cross_coupled_stops(self):
        v=VisPyCTRViewer.__new__(VisPyCTRViewer)
        v.limits=self.limits; v.deployment=np.array([.120,.050,.045]); v.robot_dirty=False
        for tube,amount in [(0,1000),(1,-1000),(2,1000),(2,-1000),(1,1000)]:
            v.selected_tube=tube; v.move(amount); self.limits.validate(v.deployment)
            self.assertIn('limit',v.last_action.lower())

    def test_solver_local_target_and_far_target_stay_valid(self):
        solver=ConstrainedTipIK(self.profile.parameters,deployment_limits=self.limits,max_iterations=60)
        e=np.array([.130,.050,.045]); r=np.array([0.,.4,-.3])
        goal=e+np.array([.001,0,0])
        target=solver.forward_tip_mm(goal,r)
        answer=solver.solve(target,e,r)
        self.limits.validate(answer.deployment_m)
        self.assertTrue(answer.reached)
        answer=solver.solve(np.array([1000.,0.,0.]),e,r)
        self.limits.validate(answer.deployment_m); self.assertFalse(answer.reached)

    def test_both_routes_and_all_intermediate_positions(self):
        rng=np.random.default_rng(12)
        for _ in range(20):
            a,b=self.limits.decode(rng.random((2,3)))
            plan=MotionPlan(np.zeros(3),a,np.zeros(3),b,np.ones(3),np.zeros(3),0,0,'test',0,0,{},np.zeros(3))
            for strategy in (DIRECT_STRATEGY,RETRACT_STRATEGY):
                route=build_motion_route(plan,strategy,deployment_limits=self.limits)
                for t in np.linspace(0,route.total_duration_s,101):
                    self.limits.validate(route_state_at_time(route,t)[0])

    def test_cache_identity_and_reloaded_states(self):
        self.assertNotEqual(workspace_identity(self.profile),workspace_identity(OperatingProfile('hardware-108.5')))
        with tempfile.TemporaryDirectory() as d:
            a=profile_workspace(self.profile,sample_count=80,cache_root=d)
            b=profile_workspace(self.profile,sample_count=80,cache_root=d)
            self.limits.validate(b.deployment_m)
            np.testing.assert_allclose(a.tips_mm,b.tips_mm,atol=1e-4)

    def test_import_atomic_and_profile_specific(self):
        v=VisPyCTRViewer.__new__(VisPyCTRViewer)
        v.profile=self.profile; v.limits=self.limits
        v.deployment=np.array([.120,.050,.045]); before=v.deployment.copy()
        record={'hardware_profile_id':self.profile.identity,'max_gap_mm':108,'operating_mode':'hardware','tube_design':'original'}
        for n,e in zip(('inner','middle','outer'),[120,50,45]):
            record[n+'_extension_mm']=e; record[n+'_rotation_deg']=0
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.csv'
            for bad in ({**record,'middle_extension_mm':70},{**record,'hardware_profile_id':'legacy'},{**record,'outer_extension_mm':-1}):
                with path.open('w') as f:
                    w=csv.DictWriter(f,fieldnames=record.keys());w.writeheader();w.writerows([record,bad])
                with self.assertRaises(ValueError):v.load_waypoints(path)
                np.testing.assert_array_equal(v.deployment,before)

if __name__=='__main__': unittest.main()
