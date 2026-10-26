"""Bind one unmodified LIBERO-90 prerequisite task before an OS development run."""
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
from stage_trace import png_bytes

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from libero.libero import benchmark, get_libero_path
    suite=benchmark.get_benchmark_dict()['libero_90']()
    name='KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it'
    matches=[i for i in range(suite.n_tasks) if suite.get_task(i).name==name]
    assert len(matches)==1
    index=matches[0];task=suite.get_task(index)
    bddl=Path(suite.get_task_bddl_file_path(index))
    init=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
    spec=importlib.util.spec_from_file_location('runtime','/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rt=mod.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=index,init_state_id=0,seed=7,max_steps=400,control_mode='relative'))
    start=time.time();out=dict(status='RUNNING')
    try:
        rt.reset({'session_id':'prerequisite_asset_probe'})
        env=rt.env.env
        goals=env.parsed_problem['goal_state']
        out.update(status='PASSED',case=dict(mode='development',suite='libero_90',task_id=index,init_state_id=0,
            task_description=task.language,max_steps=400,development_suite_extension='libero90_prerequisite_v1'),
            task_name=name,initial_goal_predicates=[dict(predicate=g,value=bool(env._eval_predicate(g))) for g in goals],
            initial_success=bool(rt.env.check_success()),source_sha256={str(bddl):hashlib.sha256(bddl.read_bytes()).hexdigest(),str(init):hashlib.sha256(init.read_bytes()).hexdigest()},
            initial_images={camera:hashlib.sha256(png_bytes(rt._last_obs[camera])).hexdigest() for camera in ('agentview_image','robot0_eye_in_hand_image')},
            native_step_limit_reference='Frozen openpi/examples/libero/main.py: libero_90 max_steps=400',
            scope='Asset/reset validation only; privileged predicates never supplied to the Agent or VLA.')
    except BaseException as exc:out.update(status='FAILED',error=repr(exc));raise
    finally:
        out['seconds']=time.time()-start;a.output.write_text(json.dumps(out,indent=2))
        if rt.env is not None:rt.env.close()

if __name__=='__main__':main()
