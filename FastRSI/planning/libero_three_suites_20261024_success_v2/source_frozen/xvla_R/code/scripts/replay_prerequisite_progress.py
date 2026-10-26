"""Locate native subgoal transitions after execution, with exact image binding."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
from stage_trace import png_bytes

def main():
    p=argparse.ArgumentParser();p.add_argument('--stream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cabinet-joints',action='store_true');a=p.parse_args()
    case=json.loads((a.stream.parent/'ROSTER.json').read_text())[0]
    assert case['suite']=='libero_90' and case['task_id']==8 and case['max_steps']==400
    path=a.stream/'stage_trace'/case['id']/'TRACE.jsonl';raw=path.read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
    spec=importlib.util.spec_from_file_location('runtime','/core/PhyAgentOS/runtime/targets/remote/libero/server.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rt=mod.LiberoRealRuntime(dict(benchmark_name=case['suite'],task_id=case['task_id'],init_state_id=case['init_state_id'],seed=7,max_steps=400,control_mode='relative'))
    start=time.time();result=dict(status='RUNNING',source_sha256=hashlib.sha256(raw).hexdigest(),rows=[],scope='Root episode offline native subgoal diagnostic, never online model features or learning labels')
    try:
        for row in rows:
            if row['method']=='reset':rt.reset({'session_id':row['session_id']})
            else:rt.action_chunk({'actions':row['actions']})
            assert rt.step_idx==row['step']
            for camera in ('agentview_image','robot0_eye_in_hand_image'):
                assert hashlib.sha256(png_bytes(rt._last_obs[camera])).hexdigest()==row['images'][f"{row['step']:04d}_{camera}.png"]
            env=rt.env.env
            result['rows'].append(dict(step=row['step'],images=row['images'],predicates=[dict(predicate=g,value=bool(env._eval_predicate(g))) for g in env.parsed_problem['goal_state']],success=bool(rt.env.check_success())))
            if a.cabinet_joints:
                cabinet=env.get_object('wooden_cabinet_1')
                result['rows'][-1]['cabinet_joints']={j:env.sim.data.get_joint_qpos(j).tolist()
                    for j in cabinet.joints}
                result['rows'][-1]['goal_region_joints']=list(env.object_sites_dict['wooden_cabinet_1_top_region'].joints)
                result['rows'][-1]['open_ranges']=cabinet.object_properties['articulation']['default_open_ranges']
        result['status']='EXACT_IMAGE_REPLAY'
    except BaseException as e:result.update(status='FAILED',error=repr(e));raise
    finally:
        result['seconds']=time.time()-start
        with a.output.open('x') as f:json.dump(result,f,indent=2)
        if rt.env is not None:rt.env.close()

if __name__=='__main__':main()
