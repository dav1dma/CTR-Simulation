"""Manual integration check; requires an available desktop/OpenGL session."""
import sys,csv,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vispy import app
from interactive_ctr_tip_control import VisPyCTRTipControlViewer
from ctr_motion_planner import RETRACT_STRATEGY
def run_checks():
    v=VisPyCTRTipControlViewer(workspace_samples=256)
    v.canvas.show()
    assert np.allclose(v.deployment,[.038,0,0])
    # Load valid exports, reject incompatible ones transactionally, and sync model target.
    with tempfile.TemporaryDirectory() as tmp:
     p=Path(tmp)/'state.csv'
     row=v.pose_record('test');row.update(inner_extension_mm=100,middle_extension_mm=40,outer_extension_mm=20)
     with p.open('w') as f:
      w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerow(row)
     v.load_waypoints(p);assert np.allclose(v.deployment,[.1,.04,.02])
     assert np.allclose(v.target_tip_mm,v.selected_endpoint_mm)
     target=v.planning_solver.forward_tip_mm([.101,.04,.02],np.zeros(3))
     attempt=v.target_planner.plan(target,v.deployment,v.rotation)
     assert attempt.plan is not None
     v.motion_strategy=RETRACT_STRATEGY;v.show_motion_plan(attempt.plan)
     for phase in v.motion_route.phases:
      v.limits.validate(phase.start_deployment_m);v.limits.validate(phase.goal_deployment_m)
     row['middle_extension_mm']=70
     with p.open('w') as f:
      w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerow(row)
     before=v.deployment.copy()
     try:v.load_waypoints(p)
     except ValueError:pass
     else:raise AssertionError('Invalid import accepted')
     assert np.array_equal(before,v.deployment)
    # Actual live switch rebuilds all profile-dependent objects.
    v=v.switch_profile(tubes='optimised');app.process_events()
    assert v.profile.design=='optimised' and v.motion_plan is None
    expected = json.loads((Path(__file__).resolve().parents[1]/'measured_hardware_simulator/optimised_configuration.json').read_text())['model_parameters']
    assert v.planning_solver.parameters == expected
    v=v.switch_profile(mode='tube-length');app.process_events()
    assert np.allclose(v.deployment,[.12,.07,.04])
    assert np.allclose(v.planning_solver.deployment_limits.upper_m,[sum(s) for s in expected['l_t']])
    v=v.switch_profile(mode='hardware');app.process_events()
    assert np.allclose(v.deployment,v.profile.reset_m)
    v.limits.validate(v.deployment)
    v.toggle_guides();v.toggle_guides();assert not v.joint_workspace_visual.visible
    v.canvas.close()
    print('LIVE CHECKS PASSED: imported poses, target synchronisation, IK preview/retract route, profile switching and workspace selection.')

if __name__ == '__main__':
    run_checks()
