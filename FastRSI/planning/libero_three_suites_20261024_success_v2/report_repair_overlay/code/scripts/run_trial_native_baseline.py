"""Full native Agent trial with a single declared suffix intervention."""
import asyncio,hashlib,json,os,re,signal,time
from pathlib import Path
from cost_trace import span,emit
from fastrsi.trial_prefix import signature
from fastrsi.feedback_window import FeedbackWindow
from fastrsi.resident_client import from_environment
from run_resident_native_baseline import run_owned_session
import run_native_baseline as native
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter

async def main():
    root=native.ROOT;roster=json.loads((root.parent/'ROSTER.json').read_text());case=roster[0]
    config=json.loads(Path(os.environ['FASTRSI_TRIAL_SPEC']).read_text())
    assert len(roster)==1 and case['mode']=='development' and case['feedback_window_mode']=='fixed10'
    client=from_environment();original=OpenPIPi05Adapter.to_policy_input;infer=native.OpenPIClientPolicyWrapper.infer
    current={};records=[];cached={}
    if config['mode']=='reuse' and config.get('cache_loop',False):
        import numpy as np
        source=Path(config['source_trace']);raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config['source_trace_sha256']:raise ValueError('Cached trace changed')
        trace=list(map(json.loads,raw.splitlines()))
        policy_path=Path(config['source_policy']);raw=policy_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config['source_policy_sha256']:raise ValueError('Cached policy log changed')
        policy=list(map(json.loads,raw.splitlines()))
        for i in range(config['boundary']//10):
            row=policy[i];actions=np.asarray(trace[i+1]['actions'],dtype=row['actions']['dtype'])
            if row['step']!=i*10 or signature(actions)!=row['actions']:raise ValueError('Stored action differs from policy prediction')
            cached[row['step']]=(row,actions)

    original_window=FeedbackWindow.apply
    def window(controller,chunk,context):
        step=int(re.fullmatch(r'libero_obs_(\d+)',context['source_observation_id'])[1])
        controller.mode='fixed5' if config['mode']!='source' and step>=config['boundary'] else 'fixed10'
        out,event=original_window(controller,chunk,context)
        event['trial_suffix_commit_steps']=config['suffix_commit_steps']
        return out,event
    def bind(adapter,observation,context):
        payload=original(adapter,observation,context)
        match=re.fullmatch(r'libero_obs_(\d+)',str(observation.get('observation_id','')))
        if not match or context['session_id']!=case['id']:raise ValueError('Unexpected root/observation in trial')
        step=int(match[1]);current['step']=step
        return payload
    def request(policy,observation):
        tick=time.perf_counter();origin='policy'
        step=current['step'];inputs={k:signature(v) for k,v in observation.items() if hasattr(v,'tobytes')}
        if step in cached:
            saved,actions=cached.pop(step)
            if saved['prompt']!=observation.get('prompt') or saved['inputs']!=inputs:raise ValueError('Cached prefix no longer applicable')
            origin='paid_prefix_cache'
            output=dict(actions=actions.copy(),policy_meta=dict(policy_latency_ms=0.0,prediction_origin=origin,
                        source_trace_sha256=config['source_trace_sha256']))
        else:
            if cached:raise ValueError('Skipped cached prefix boundary')
            output=infer(policy,observation)
        row=dict(origin=origin,step=step,prompt=observation.get('prompt'),
                 inputs={k:signature(v) for k,v in observation.items() if hasattr(v,'tobytes')},
                 actions=signature(output['actions']),policy_meta=output.get('policy_meta'),seconds=time.perf_counter()-tick)
        records.append(row)
        with (root/'TRIAL_POLICY.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
        return output
    async def execute():
        if config['mode']=='reuse':
            command=dict(op='restore_trial_prefix',request_id='restore-'+client.session_id,session_id=client.session_id,
                         source_session=config['source_session'],source_inferences=config['boundary']//10)
            with span('trial_policy_restore'):
                receipt=await asyncio.to_thread(client._receipt,command,time.monotonic()+client.timeout_seconds)
            if receipt['prefix_sha256']!=config['policy_prefix_sha256']:raise ValueError('Wrong source policy checkpoint')
            (root/'TRIAL_RESTORE.json').write_text(json.dumps(receipt,indent=2))
        await native.main()
    FeedbackWindow.apply=window
    OpenPIPi05Adapter.to_policy_input=bind;native.OpenPIClientPolicyWrapper.infer=request
    loop=asyncio.get_running_loop();task=asyncio.current_task();loop.add_signal_handler(signal.SIGTERM,task.cancel)
    try:
        await run_owned_session(execute,client,roster,root,int(os.environ['FASTRSI_RESIDENT_MAX_INFERENCES']),span_factory=span)
    finally:
        FeedbackWindow.apply=original_window
        OpenPIPi05Adapter.to_policy_input=original;native.OpenPIClientPolicyWrapper.infer=infer
        loop.remove_signal_handler(signal.SIGTERM)
        (root/'TRIAL_CALLS.json').write_text(json.dumps(dict(actual_policy_requests=sum(r['origin']=='policy' for r in records),
            cached_requests=sum(r['origin']=='paid_prefix_cache' for r in records),
            inherited_policy_requests=config['boundary']//10 if config['mode']=='reuse' else 0,
            task_goal=case['task_description'],trial=config),indent=2))

if __name__=='__main__':asyncio.run(main())
