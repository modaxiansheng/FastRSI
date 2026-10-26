"""Reset two registered native tasks; no action, policy, or learned claim."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from stage_trace import png_bytes


def main():
    from libero.libero import benchmark, get_libero_path
    from fastrsi.development_suite import COMPOSITION_GOALS, COMPOSITION_EXTENSION
    output = Path('/project/runs/cabinet_composition_baseline_v1_20261006/ASSET.json')
    assert not output.exists()
    suite = benchmark.get_benchmark_dict()['libero_90']()
    spec = importlib.util.spec_from_file_location('runtime','/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    start=time.time(); rows=[]; result=dict(status='RUNNING',cases=rows)
    try:
        for index, goal in COMPOSITION_GOALS.items():
            task=suite.get_task(index)
            assert task.language==goal
            assert task.name=='KITCHEN_SCENE10_'+goal.replace(' ','_')
            bddl=Path(suite.get_task_bddl_file_path(index))
            init=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
            rt=mod.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=index,
                init_state_id=0,seed=7,max_steps=400,control_mode='relative'))
            try:
                rt.reset({'session_id':f'cabinet_asset_t{index}'})
                env=rt.env.env
                rows.append(dict(case=dict(mode='development',suite='libero_90',task_id=index,
                    init_state_id=0,task_description=goal,max_steps=400,
                    native_experience_maintenance=True,development_suite_extension=COMPOSITION_EXTENSION),
                    task_name=task.name,initial_success=bool(rt.env.check_success()),
                    diagnostic_initial_predicates=[dict(predicate=g,value=bool(env._eval_predicate(g)))
                        for g in env.parsed_problem['goal_state']],
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [bddl,init]},
                    images={k:hashlib.sha256(png_bytes(rt._last_obs[k])).hexdigest()
                        for k in ['agentview_image','robot0_eye_in_hand_image']}))
            finally:
                if rt.env is not None: rt.env.close()
        result['status']='PASSED'
    except BaseException as e:
        result.update(status='FAILED',error=repr(e));raise
    finally:
        result.update(seconds=time.time()-start,robot_actions=0,policy_calls=0,
            scope='Asset/reset binding only; hidden predicates are diagnostic and never controller input. Separate tasks, not a composed same-episode success.')
        output.write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
