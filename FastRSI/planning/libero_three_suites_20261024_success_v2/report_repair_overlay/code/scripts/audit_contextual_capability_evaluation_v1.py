"""Account complete fixed evaluation, no persistent memory or component updates."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def command_sha(actions):
    values=[v for action in actions for v in action]
    return hashlib.sha256(struct.pack('<%df'%len(values),*values)).hexdigest()


def main(a):
    run=a.run;manifest=read(run/'RAW_MANIFEST.json')['files']
    for item in manifest:
        p=run/item['path']
        assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
    cleanup=read(run/'FRESH_CLEANUP_CHECK.json')
    assert len(cleanup['containers'])==16
    assert all(not c['state']['Running'] and not c['state']['OOMKilled'] for c in cleanup['containers'])
    protocol=read(run/'PROTOCOL.json');status=read(run/'STATUS.json');case=read(run/'PLAN.json')['cases'][0];goal=case['task_description']
    assert status['status']=='COMPLETED'
    assert case['mode']=='evaluation' and case['persistent_updates'] is False
    session='net_t4_i'+str(case['init_state_id'])
    arms={};prefixes={};first_inputs={};accepted_sources={}
    for arm in ('N','F','R','S'):
        stream=run/arm/'stream';report=read(stream/'RESULT.json')['cases'][0]
        assert report['case']['mode']=='evaluation' and 'experience_admission' not in report
        assert report['experience_before']==report['experience_after']
        audit=rows(stream/session/'AGENT_EVENTS.jsonl')
        assert any(e['event']=='agent_bound' and e['evaluation'] for e in audit)
        assert any(e['event']=='agent_closed' and e['active_tasks_remaining']==0 for e in audit)
        assert any(e['event']=='task_context_retained' and e['messages']>0 for e in audit)
        assert (stream/session/'workspace/.FASTRSI_EVALUATION_CLOSED.json').is_file()
        events=rows(stream/'OS_COST_EVENTS.jsonl');inputs=[e for e in events if e['event']=='actual_policy_input']
        ends=[e for e in events if e['event']=='end'];traces={}
        for p in sorted((stream/'stage_trace').glob('*/TRACE.jsonl')):traces[p.parent.name]=rows(p)
        assert all(e['prompt']==goal for e in inputs)
        windows=[e for e in events if e['event']=='feedback_window_choice']
        assert all(e['requested_steps']==10 and e['original_steps']==10 for e in windows)
        physical=0;submissions=0
        for key,trace in traces.items():
            assert trace[0]['method']=='reset' and sum(t['method']=='reset' for t in trace)==1
            for left,right in zip(trace,trace[1:]):
                assert right['method']=='action_chunk' and 0<right['step']-left['step']<=len(right['actions'])
            physical+=trace[-1]['step'];submissions+=len(trace)-1
        serverfile=stream/'CONTEXTUAL_POLICY_EVENTS.jsonl'
        server=[e for e in rows(serverfile) if e['event']=='contextual_policy_cost'] if serverfile.exists() else []
        used=[e for e in events if e['event']=='contextual_capability_result' and e['used']]
        attempts=[e for e in events if e['event']=='contextual_capability_result']
        if arm=='N':
            assert not server and not used
            decoder=len(inputs);prefix=len(inputs);local=0
        else:
            assert len(server)==len(inputs)
            decoder=sum(e['teacher_decoder_calls'] for e in server)
            extra=sum(e['extra_encoder_calls'] for e in server)
            prefix=decoder+extra;local=extra
            assert sum(e.get('used',False) for e in server)==len(used)
            assert len(attempts)<=2
            component=protocol['contextual_components'][arm]['sha256']
            for e in attempts:
                assert e['artifact_sha256']==component
                trace=traces[e['session_id']]
                found=next(t for t in trace if t['step']>e['physical_steps'])
                assert command_sha(found['actions'])==e['actions_float32_sha256']
            receipt=read(stream/'CONTEXTUAL_CAPABILITY_RECEIPT.json')
            assert receipt['completed'] and receipt['mode']=='evaluation' and receipt['audit']['parameter_updates_during_task']==0
            assert receipt['evaluation_component_protection']['sha256']==component
            artifact=run/'components'/Path(protocol['contextual_components'][arm]['artifact']).name
            assert hashlib.sha256(artifact.read_bytes()).hexdigest()==component
        prefixes[arm]=[{k:v for k,v in t.items() if k!='session_id'} for t in traces[session]]
        first_inputs[arm]=[i['array_bindings'] for i in inputs]
        accepted_sources[arm]=report['accepted_source']
        qwen=[read(p) for p in sorted((stream/'service_calls').glob('call_*/RECEIPT.json'))]
        stops=read(stream/'STOP_RECEIPT.json')
        allocated=sum((datetime.datetime.fromisoformat(s['finished'].replace('Z','+00:00'))-
            datetime.datetime.fromisoformat(s['started'].replace('Z','+00:00'))).total_seconds()
            for s in stops if s['role'] in ('pi05','qwen'))
        worker=read(run/arm/'STATUS.json')
        arms[arm]=dict(evaluation_isolation=dict(persistent_hashes_equal=True,component_immutable=True,agent_closed=True,
            background_tasks=0,task_context_retained=True,blocked_operations=[e.get('operation') for e in audit if e['event'] in ('update_blocked','filesystem_update_blocked')]),environment_success=report['environment_success'],verification_status=report['verification_status'],
            chain_status=report['chain_status'],admission_status=report.get('experience_admission',{}).get('status'),
            episodes=len(traces),physical_steps=physical,action_submissions=submissions,
            policy_rpc=len(inputs),full_teacher_decoder_calls=decoder,prefix_encoder_calls=prefix,
            local_predictor_calls=local,local_outputs=len(used),candidate_attempts=attempts,
            qwen_service_calls=len(qwen),agent_model_calls=sum(e['name']=='agent_model_call' for e in ends),
            verification_model_calls=sum(e['name']=='verification_model' for e in ends),
            qwen_generation_seconds=sum(r['generation_seconds'] for r in qwen),
            qwen_prompt_tokens=sum(r['prompt_tokens'] for r in qwen),qwen_completion_tokens=sum(r['completion_tokens'] for r in qwen),
            prefix_feature_seconds=sum(e.get('prefix_feature_seconds',0) for e in server),
            local_decision_seconds=sum(e.get('local_decision_seconds',0) for e in server),
            local_trigger_seconds=sum(e['seconds'] for e in events if e['event']=='contextual_trigger_cost'),
            startup_seconds=worker['startup_seconds'],controller_seconds=worker['controller_seconds'],
            worker_seconds=worker['full_wall_seconds'],allocated_model_card_seconds=allocated)
    prefix_limit=min([e['physical_steps'] for a in arms.values() for e in a['candidate_attempts']],default=0)
    for arm in arms:
        prefixes[arm]=[t for t in prefixes[arm] if t['step']<=prefix_limit]
        # Include the current observation at the first intervention boundary,
        # not only observations from completed original-policy chunks.
        first_inputs[arm]=first_inputs[arm][:prefix_limit//10+1]
    for arm in ('F','R','S'):
        assert prefixes[arm]==prefixes['N'],arm
        assert first_inputs[arm]==first_inputs['N'],arm
        assert accepted_sources[arm]==accepted_sources['N'],arm
    out=dict(status='VERIFIED_FIXED_EVALUATION',case=case,
        arms=arms,verified_raw_files=len(manifest),owned_containers_stopped=16,
        original_prefix_equal=True,original_prefix_physical_steps=prefix_limit,same_initial_experience=True,original_goal_H10_W10=True,
        parent_seconds=status['seconds'],sum_worker_seconds=sum(v['worker_seconds'] for v in arms.values()),
        allocated_model_card_seconds=sum(v['allocated_model_card_seconds'] for v in arms.values()),
        timing_note='Worker includes startup and nested stages. Parallel parent is throughput, not sum of work. Allocated card time is not GPU busy time.',
        new_context_evaluation=True,cyclic_distillation=False,default_changed=False)
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'SUMMARY.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
