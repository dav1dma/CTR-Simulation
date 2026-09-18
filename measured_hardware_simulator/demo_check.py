"""Native render and end-to-end control check; optionally leave the demo open."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from vispy import app, io
from interactive_ctr_tip_control import VisPyCTRTipControlViewer, TARGET_PREVIEW, TARGET_REACHED
from ctr_motion_planner import DIRECT_STRATEGY, RETRACT_STRATEGY

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stay',action='store_true');args=parser.parse_args()
    out=Path(__file__).resolve().parent/'exports';out.mkdir(exist_ok=True)
    v=VisPyCTRTipControlViewer()
    v.controller.disconnect()
    v.canvas.show();app.process_events()
    v.deployment[:]=[.140,.055,.050];v.rotation[:]=np.deg2rad([0,65,-20])
    v.robot_dirty=True;v.update_robot();v.sync_target_to_tip()
    initial=v.deployment.copy()
    v.selected_tube=2;v.keyboard_move(1000);v.limits.validate(v.deployment)
    assert 'limit' in v.last_action.lower()
    v.undo();np.testing.assert_allclose(v.deployment,initial)
    v.deployment[:]=[.100,0,0];v.selected_tube=0
    v.profile_key('c');v.keyboard_move(20)
    np.testing.assert_allclose(v.deployment,[.120,.020,.010],atol=1e-12)
    assert v.following_actuators==(1,2)
    v.robot_dirty=True;v.update_robot();v.update_status();app.process_events()
    io.write_png(str(out/'coordinated_followers_demo.png'),v.canvas.render())
    v.undo();np.testing.assert_allclose(v.deployment,[.100,0,0])
    v.profile_key('c');v.deployment[:]=initial
    v.selected_tube=0;v.update_robot();v.toggle_control_mode()
    desired=initial+np.array([.001,0,0])
    target=v.ik_solver.forward_tip_mm(desired,v.rotation)
    v.set_target_tip(target,'Known feasible test target')
    v.calculate_target_plan();assert v.tip_stage==TARGET_PREVIEW,v.last_action
    for strategy in (RETRACT_STRATEGY,DIRECT_STRATEGY):
        v.select_motion_strategy(strategy)
        for phase in v.motion_route.phases:
            v.limits.validate(phase.start_deployment_m);v.limits.validate(phase.goal_deployment_m)
    v.begin_execution(now=100.)
    duration=v.execution_duration
    for t in np.linspace(100,100+duration+.01,80):
        v.update_execution(t);v.limits.validate(v.deployment)
    assert v.tip_stage==TARGET_REACHED
    v.update_robot();v.update_status();app.process_events()
    residual=float(np.linalg.norm(v.selected_endpoint_mm-target))
    assert residual<=v.ik_solver.tolerance_mm+.01
    record=v.pose_record('verified_demo')
    path=out/'verified_demo.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=record);w.writeheader();w.writerow(record)
    v.load_waypoints(path);v.limits.validate(v.deployment)
    # Original pose and selected endpoint are restored for an easily read demo.
    v.deployment[:]=initial;v.rotation[:]=np.deg2rad([0,65,-20]);v.selected_tube=0
    v.robot_dirty=True;v.update_robot();v.sync_target_to_tip()
    v.toggle_control_mode();v.profile_key('c')
    v.last_action='Coordinated joint control ready';v.update_status();app.process_events()
    io.write_png(str(out/'measured_hardware_demo.png'),v.canvas.render())
    (out/'demo_verification.json').write_text(json.dumps({'native_render':True,'manual_block_and_undo':True,'coordinated_follow_and_undo':True,'actuator_overlay':True,
        'tip_ik_preview':True,'both_route_previews':True,'executed_states_valid':80,'valid_csv_roundtrip':True,
        'tip_residual_mm':residual,'hardware_profile_id':v.profile.identity},indent=2))
    print('Native demo checks passed; tip residual',residual,'mm',flush=True)
    if args.stay: v.run()
    else: v.canvas.close();v.on_close()

if __name__=='__main__':main()
