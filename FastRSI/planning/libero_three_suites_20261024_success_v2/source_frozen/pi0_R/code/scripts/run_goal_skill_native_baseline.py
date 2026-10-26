"""Full native Agent, explicit goal-prefix requests, then unchanged real policy."""
import asyncio,hashlib,json,re,time
import numpy as np
import run_native_baseline as native
from cost_trace import emit,span
from fastrsi.goal_skill import active,marker,load_skill
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
original_case=native.run_case

async def run_case(case,accepted):
    spec=case['goal_skill_probe'];skill=load_skill(spec['artifact'],spec['sha256']);steps=len(skill['waypoints'])
    if case['feedback_window_mode']!='fixed10' or case['max_steps']!=400:raise ValueError('Unbound goal skill protocol')
    infer0=native.OpenPIClientPolicyWrapper.infer;input0=OpenPIPi05Adapter.to_policy_input;output0=OpenPIPi05Adapter.from_policy_output
    current={};rows=[]
    def bind(adapter,observation,context):
        match=re.fullmatch(r'libero_obs_(\d+)',str(observation['observation_id']))
        if match is None:raise ValueError('Missing physical step binding')
        current.update(session=context['session_id'],step=int(match[1]),observation=observation['observation_id'])
        return input0(adapter,observation,context)
    def infer(policy,observation):
        tick=time.perf_counter();reuse=active(spec['enabled'],current['session'],case['id'],current['step'],steps,skill.get('start_step',0))
        if reuse:
            emit(dict(event='actual_policy_input',case=case['id'],prompt=observation.get('prompt'),
                prediction_origin='goal_skill_request',robot_state=observation['observation/state'].tolist(),
                array_bindings={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest())
                                for k,v in observation.items() if hasattr(v,'tobytes')}))
            tag=marker(spec['sha256'],current['session'],current['step'],steps=steps)
            output={'actions':np.zeros((10,7),np.float32),'policy_meta':{'prediction_origin':'explicit_goal_skill_request','goal_skill':tag}}
        else:
            with span('goal_skill_native_inference',case=case['id']):output=infer0(policy,observation)
        row=dict(event='goal_skill_choice',case=case['id'],runtime_session=current['session'],step=current['step'],origin='goal_skill' if reuse else 'vla',seconds=time.perf_counter()-tick,goal=observation.get('prompt'),skill_sha256=spec['sha256'])
        rows.append(row);emit(row)
        with (native.ROOT/'GOAL_SKILL_CALLS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return output
    def decode(adapter,output,context):
        chunk=output0(adapter,output,context);tag=output.get('policy_meta',{}).get('goal_skill')
        if tag is not None:chunk['provenance']={**chunk.get('provenance',{}),'goal_skill':tag}
        return chunk
    native.OpenPIClientPolicyWrapper.infer=infer;OpenPIPi05Adapter.to_policy_input=bind;OpenPIPi05Adapter.from_policy_output=decode
    try:return await original_case(case,accepted)
    finally:
        native.OpenPIClientPolicyWrapper.infer=infer0;OpenPIPi05Adapter.to_policy_input=input0;OpenPIPi05Adapter.from_policy_output=output0
        (native.ROOT/'GOAL_SKILL_RECEIPT.json').write_text(json.dumps(dict(enabled=spec['enabled'],skill_requests=sum(x['origin']=='goal_skill' for x in rows),actual_vla_calls=sum(x['origin']=='vla' for x in rows),boundary_requests=len(rows),training_updates=0,source_steps=steps,source_start_step=skill.get('start_step',0),scope='Fixed development intervention, not learned selection. Skipped VLA draws alter later RNG positions; no synthetic inference calls are charged as real.'),indent=2))
native.run_case=run_case
if __name__=='__main__':asyncio.run(native.main())
