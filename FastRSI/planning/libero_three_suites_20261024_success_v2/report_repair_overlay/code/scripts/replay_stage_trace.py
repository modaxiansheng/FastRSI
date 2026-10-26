"""Offline evaluator-only replay of recorded actions; no model or controller calls."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
from stage_trace import png_bytes

def main():
    ap=argparse.ArgumentParser();ap.add_argument('trace');ap.add_argument('output');a=ap.parse_args()
    rows=[json.loads(s) for s in Path(a.trace).read_text().splitlines()]
    spec=importlib.util.spec_from_file_location('runtime','/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    runtime=mod.LiberoRealRuntime({'benchmark_name':'libero_10','task_id':4,'init_state_id':31,'seed':7,'max_steps':520,'control_mode':'relative'})
    output={'status':'RUNNING','scope':'Offline privileged evaluator diagnostics; no policy inputs or experience updates.','rows':[],'source_trace_sha256':hashlib.sha256(Path(a.trace).read_bytes()).hexdigest()};start=time.time()
    try:
        for row in rows:
            if row['method']=='reset':runtime.reset({'session_id':row['session_id']})
            else:runtime.action_chunk({'actions':row['actions']})
            assert runtime.step_idx==row['step']
            matches={}
            for camera in ('agentview_image','robot0_eye_in_hand_image'):
                name=f"{row['step']:04d}_{camera}.png";matches[camera]=hashlib.sha256(png_bytes(runtime._last_obs[camera])).hexdigest()==row['images'][name]
            if not all(matches.values()):raise RuntimeError('Replay image mismatch at '+str(row['step']))
            env=runtime.env.env
            predicates=[{'predicate':s,'value':bool(env._eval_predicate(s))} for s in env.parsed_problem['goal_state']]
            output['rows'].append({'step':row['step'],'images_match':matches,'predicates':predicates,'overall_success':bool(runtime.env.check_success())})
        output['status']='EXACT_IMAGE_REPLAY'
    except BaseException as exc:output.update(status='UNVERIFIED_REPLAY',error=repr(exc));raise
    finally:
        output['seconds']=time.time()-start
        with open(a.output,'x') as f:json.dump(output,f,indent=2)
        if runtime.env is not None:runtime.env.close()
    print(json.dumps({'status':output['status'],'steps':len(rows),'seconds':output['seconds']}))
if __name__=='__main__':main()
