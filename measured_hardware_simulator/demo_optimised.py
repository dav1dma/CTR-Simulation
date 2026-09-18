"""Exercise the selected design in the native viewer and retain a screenshot."""
import json,csv
from pathlib import Path
import numpy as np
from vispy import app,io
from interactive_ctr_tip_control import VisPyCTRTipControlViewer,TARGET_PREVIEW,TARGET_REACHED
from ctr_motion_planner import DIRECT_STRATEGY,RETRACT_STRATEGY

def main():
    out=Path(__file__).resolve().parents[1]/'output/measured_hardware_optimisation_20260914'
    v=VisPyCTRTipControlViewer(tubes='optimised',workspace_samples=12000)
    v.controller.disconnect();v.canvas.show();app.process_events()
    initial=v.limits.decode(np.array([.5,.5,.6]));v.deployment[:]=initial;v.rotation[:]=[0,.7,-.3]
    v.robot_dirty=True;v.update_robot();v.sync_target_to_tip()
    v.selected_tube=0;v.profile_key('c');v.keyboard_move(200)
    v.limits.validate(v.deployment);v.undo();np.testing.assert_allclose(v.deployment,initial)
    v.toggle_control_mode();desired=initial.copy();desired[0]+=.0005;v.limits.validate(desired)
    target=v.ik_solver.forward_tip_mm(desired,v.rotation)
    v.set_target_tip(target,'Selected-design feasible target');v.calculate_target_plan()
    assert v.tip_stage==TARGET_PREVIEW,v.last_action
    for strategy in (RETRACT_STRATEGY,DIRECT_STRATEGY):
        v.select_motion_strategy(strategy)
        for phase in v.motion_route.phases:v.limits.validate(np.linspace(phase.start_deployment_m,phase.goal_deployment_m,30))
    v.begin_execution(now=100)
    for t in np.linspace(100,100+v.execution_duration+.01,100):v.update_execution(t);v.limits.validate(v.deployment)
    assert v.tip_stage==TARGET_REACHED
    v.update_robot();record=v.pose_record('optimised_native_check')
    p=out/'optimised_native_pose.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=record);w.writeheader();w.writerow(record)
    v.load_waypoints(p)
    residual=float(np.linalg.norm(v.selected_endpoint_mm-target))
    v.last_action='Measured-hardware candidate | numerical validation complete';v.update_status();app.process_events()
    io.write_png(str(out/'optimised_viewer.png'),v.canvas.render())
    (out/'native_viewer_verification.json').write_text(json.dumps(dict(profile=v.profile.identity,coordinated_movement_and_undo=True,both_route_previews=True,executed_valid_states=100,csv_roundtrip=True,tip_residual_mm=residual),indent=2))
    print('Optimised native checks passed; residual',residual,flush=True)
    v.canvas.close();v.on_close()
if __name__=='__main__':main()
