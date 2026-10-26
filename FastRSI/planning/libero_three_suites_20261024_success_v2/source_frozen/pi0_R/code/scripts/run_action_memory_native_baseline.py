"""Observe each normal H10 boundary; only eligible memory hits bypass the VLA."""
import asyncio, hashlib, json, time
from pathlib import Path
import run_native_baseline as native
from cost_trace import span,emit
from fastrsi.action_memory import ActionMemory,load_bound,GOAL
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter

original_case=native.run_case

async def run_case(case,accepted):
    spec=case['action_memory_probe']
    if (case['suite']!='libero_90' or case['task_id']!=64 or case['init_state_id']!=1 or
            case['mode']!='development' or case['task_description']!=GOAL or
            case['feedback_window_mode']!='fixed10' or case['max_steps']!=400 or
            type(spec['enabled']) is not bool):raise ValueError('Unbound memory pilot')
    with span('action_memory_load'):
        value=load_bound(Path(spec['artifact']),spec['sha256']); memory=ActionMemory(value)
    original=native.OpenPIClientPolicyWrapper.infer
    original_input=OpenPIPi05Adapter.to_policy_input
    current={}
    def bind(adapter,observation,context):
        current.update(session=context['session_id'],observation=observation['observation_id'])
        return original_input(adapter,observation,context)
    rows=[]
    def infer(policy,observation):
        emit(dict(event='action_memory_input',case=case['id'],runtime_session=current['session'],
                  observation_id=current['observation'],prompt=observation.get('prompt'),
                  robot_state=observation['observation/state'].tolist(),
                  array_bindings={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest())
                                  for k,v in observation.items() if hasattr(v,'tobytes')}))
        # Same frozen source is available to both arms; N runs normal inference.
        if spec['enabled'] and current['session']==case['id']:
            with span('action_memory_lookup'):
                actions,decision=memory.choose(observation)
        else:actions,decision=None,dict(reason='native_control_or_retry')
        start=time.perf_counter()
        if actions is None:
            output=original(policy,observation);origin='vla'
        else:
            output={'actions':actions,'policy_meta':{'prediction_origin':'observed_action_memory','source_step':decision['source_step']}}
            origin='memory'
        row=dict(event='action_memory_choice',case=case['id'],origin=origin,decision=decision,
                 runtime_session=current['session'],observation_id=current['observation'],
                 service_seconds=time.perf_counter()-start,artifact_sha256=spec['sha256'])
        rows.append(row);emit(row)
        with (native.ROOT/'ACTION_MEMORY_CALLS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return output
    native.OpenPIClientPolicyWrapper.infer=infer
    OpenPIPi05Adapter.to_policy_input=bind
    try:return await original_case(case,accepted)
    finally:
        native.OpenPIClientPolicyWrapper.infer=original
        OpenPIPi05Adapter.to_policy_input=original_input
        (native.ROOT/'ACTION_MEMORY_RECEIPT.json').write_text(json.dumps(dict(enabled=spec['enabled'],
            actual_vla_calls=sum(r['origin']=='vla' for r in rows),memory_calls=sum(r['origin']=='memory' for r in rows),
            total_action_decisions=len(rows),memory_sha256=spec['sha256'],persistent_model_updates=0),indent=2))

native.run_case=run_case
if __name__=='__main__':asyncio.run(native.main())
