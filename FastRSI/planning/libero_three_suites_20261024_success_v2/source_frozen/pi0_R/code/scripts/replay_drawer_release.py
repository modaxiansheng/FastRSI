"""Offline contact/containment/grasp audit, never a policy observation source."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import time
from pathlib import Path
from stage_trace import png_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    spec = importlib.util.spec_from_file_location('runtime', '/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.LiberoRealRuntime(dict(benchmark_name='libero_10', task_id=3,
        init_state_id=38, seed=7, max_steps=520, control_mode='relative'))
    start = time.time()
    report = dict(status='RUNNING', source_trace_sha256=hashlib.sha256(raw).hexdigest(), rows=[],
        scope='Offline simulator diagnostics. Not online release/clearance evidence, teacher labels, or a skill update.')
    try:
        for row in rows:
            if row['method'] == 'reset':
                runtime.reset({'session_id': row['session_id']})
            else:
                runtime.action_chunk({'actions': row['actions']})
            assert runtime.step_idx == row['step']
            for camera in ('agentview_image', 'robot0_eye_in_hand_image'):
                digest = hashlib.sha256(png_bytes(runtime._last_obs[camera])).hexdigest()
                assert digest == row['images'][f"{row['step']:04d}_{camera}.png"]
            env = runtime.env.env
            bowl = env.get_object('akita_black_bowl_1')
            bowl_state = env.object_states_dict['akita_black_bowl_1']
            region = env.object_states_dict['white_cabinet_1_bottom_region']
            gripper = env.robots[0].gripper
            contain = bool(region.check_contain(bowl_state))
            contact = bool(region.check_contact(bowl_state))
            inside = bool(env._eval_predicate(['in', 'akita_black_bowl_1', 'white_cabinet_1_bottom_region']))
            assert inside == (contain and contact)
            joints = {name: float(env.sim.data.get_joint_qpos(name)) for name in env.sim.model.joint_names
                      if 'white_cabinet' in name}
            report['rows'].append(dict(step=row['step'], images=row['images'],
                region_contains_bowl=contain, site_contact_method_result=contact, native_inside=inside,
                gripper_contacts_bowl=bool(env.check_contact(gripper, bowl)),
                native_grasp=bool(env._check_grasp(gripper, bowl.contact_geoms)),
                native_open=bool(region.is_open()), native_closed=bool(region.is_close()),
                cabinet_joints=joints, bowl_xyz=bowl_state.get_geom_state()['pos'].tolist(),
                gripper_qpos=runtime._last_obs['robot0_gripper_qpos'].tolist(),
                overall_success=bool(runtime.env.check_success())))
        from libero.libero.envs.predicates.base_predicates import In
        report['source_definitions'] = {name: inspect.getsource(fn) for name, fn in
            [('in', In), ('region_contact', type(region).check_contact),
             ('region_contain', type(region).check_contain), ('open', type(region).is_open),
             ('close', type(region).is_close), ('grasp', env._check_grasp)]}
        report['status'] = 'EXACT_IMAGE_REPLAY'
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        raise
    finally:
        report['seconds'] = time.time() - start
        with args.output.open('x') as f:
            json.dump(report, f, indent=2)
        if runtime.env is not None:
            runtime.env.close()


if __name__ == '__main__':
    main()
