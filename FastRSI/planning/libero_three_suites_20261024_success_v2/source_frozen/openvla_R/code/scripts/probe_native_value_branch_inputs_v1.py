"""Replay shared factual prefixes to validate same-state branch supervision.

CPU simulation only. No model training, actor calls, branch outcome fabrication,
or changing the completed native evaluation. Two development examples are used.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from replay_stack_progress import RUNTIME_SOURCE

P=Path('/project');O=Path('/check');KEYS=('observation/image','observation/wrist_image','observation/state')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    spec=importlib.util.spec_from_file_location('branch_native_runtime',RUNTIME_SOURCE)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    reports=[];tick=time.perf_counter()
    for version in (9,10):
        name=f'native_value_phase_v{version}_20261010';root=P/'runs'/name
        reg=json.loads((root/'REGISTRATION.json').read_text());initial=reg['initial']
        n=root/'N/stream';f=root/'F/stream'
        nc=list(map(json.loads,(n/'value_service/VALUE_POLICY_CALLS.jsonl').read_text().splitlines()))
        fc=list(map(json.loads,(f/'value_service/VALUE_POLICY_CALLS.jsonl').read_text().splitlines()))
        selected=next(r for r in fc if r['selected_index']==1);index=selected['index'];step=selected['step']
        source=f/'value_service'/(selected['evidence_name']+'_request.msgpack')
        response=f/'value_service'/(selected['evidence_name']+'_response.msgpack')
        expected=unpackb(source.read_bytes());candidates=unpackb(response.read_bytes())['candidates']
        other=unpackb((n/'value_service'/(nc[index]['evidence_name']+'_request.msgpack')).read_bytes())
        for k in KEYS:assert np.array_equal(expected[k],other[k])
        sid=selected['session_id'];trace=list(map(json.loads,(f/'stage_trace'/sid/'TRACE.jsonl').read_text().splitlines()))
        prefix=[r for r in trace if r['method']=='action_chunk' and r['step']<=step]
        out=O/f'init{initial}';out.mkdir(exist_ok=False)
        runtime=None;case_tick=time.perf_counter();result=dict(status='RUNNING',initial=initial,fork_step=step,
            source_campaign=name,source_request_sha256=sha(source),source_response_sha256=sha(response),
            replay_actions=0,actor_inferences=0,parameter_updates=0,branch_outcomes_generated=0)
        try:
            runtime=mod.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=64,init_state_id=initial,seed=7,
                camera_height=256,camera_width=256,max_steps=400,max_chunk_size=50,num_steps_wait=10,
                control_mode='relative',record_dir=None))
            runtime.reset(dict(session_id=f'branch_probe_i{initial}'))
            for i,r in enumerate(prefix):
                runtime.action_chunk(dict(actions=r['actions'],chunk_id=f'prefix{i}'))
                assert runtime.step_idx==r['step'] and not runtime.success
            result['replay_actions']=runtime.step_idx;assert runtime.step_idx==step
            current={'observation/image':np.asarray(runtime._last_obs['agentview_image']),
                'observation/wrist_image':np.asarray(runtime._last_obs['robot0_eye_in_hand_image']),
                'observation/state':mod._libero_state(runtime._last_obs)}
            result['input_exact']={k:bool(np.array_equal(current[k],expected[k])) for k in KEYS}
            assert all(result['input_exact'].values()),result['input_exact']
            assert runtime.language==expected['prompt']
            np.savez_compressed(out/'CANDIDATES.npz',candidates=candidates)
            (out/'PLAN.json').write_text(json.dumps(dict(version='same_state_candidate_branch_v1',initial=initial,
                noise_base=selected['noise_base'],fork_step=step,next_policy_index=index+1,
                prefix=[r['actions'] for r in prefix],candidate_file='CANDIDATES.npz',
                source_request=str(source.relative_to(P)),source_request_sha256=sha(source),
                source_response=str(response.relative_to(P)),source_response_sha256=sha(response),
                arms=[0,1],max_steps=400,continuation='same unchanged pi, one candidate, same explicit future noises',
                actual_branch_outcome_labels=None,role='DEVELOPMENT_NOT_HELD_OUT_SCORE'),indent=2)+'\n')
            result['status']='PASSED_EXACT_SAME_STATE_INPUT'
        except BaseException as e:
            result.update(status='FAILED_INPUT_REPRODUCTION',error=repr(e));raise
        finally:
            if runtime is not None:runtime.close()
            result['seconds_including_close']=time.perf_counter()-case_tick
            (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');reports.append(result)
    (O/'RESULT.json').write_text(json.dumps(dict(status='PASSED',cases=reports,seconds=time.perf_counter()-tick,
        scope='Development supervision feasibility, not learned efficiency or completed branch comparison.'),indent=2)+'\n')
    print(json.dumps(reports),flush=True)

if __name__=='__main__':main()
