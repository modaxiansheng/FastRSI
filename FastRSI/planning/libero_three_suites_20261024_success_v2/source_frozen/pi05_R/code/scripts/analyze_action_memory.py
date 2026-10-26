"""Bind each memory hit to current legal input and the actions actually submitted."""
import argparse,hashlib,json,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.action_memory import ActionMemory,GOAL,LIMITS
from fastrsi.decision_corpus import checked_image,png_pixels
from analyze_stack_flow import summarize_attempts
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_bytes().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def analyze(run):
    batch=read(run/'STATUS.json');assert batch['status']=='COMPLETED'
    artifact=read(run/'MEMORY.json');protocol=read(run/'PROTOCOL.json')
    assert sha(run/'MEMORY.json')==protocol['action_memory_probe']['sha256']
    rows=[];sources={};root_traces={}
    for arm in ('N','F'):
        group=run/arm;stream=group/'stream';result=read(stream/'RESULT.json')
        assert result['status']=='COMPLETED' and len(result['cases'])==1
        row=result['cases'][0];case=row['case'];sid=case['id']
        expected=dict(suite='libero_90',task_id=64,init_state_id=1,mode='development',max_steps=400,
                      task_description=GOAL,feedback_window_mode='fixed10',id='net_t64_i1')
        assert all(type(case.get(k)) is type(v) and case[k]==v for k,v in expected.items())
        assert read(group/'ROSTER.json')==[case]
        assert all(read(run/'PLAN.json')['cases'][0][k]==v for k,v in expected.items() if k not in ('id','feedback_window_mode'))
        assert row['chain_status']=='PASSED' and case['action_memory_probe']['enabled']==(arm=='F')
        events=lines(stream/'OS_COST_EVENTS.jsonl');calls=lines(stream/'ACTION_MEMORY_CALLS.jsonl')
        inputs=[e for e in events if e.get('event')=='action_memory_input']
        actual=[e for e in events if e.get('event')=='actual_policy_input']
        windows=[e for e in events if e.get('event')=='feedback_window_choice']
        assert len(calls)==len(inputs)==len(windows)
        traces={s['session_id']:lines(stream/'stage_trace'/s['session_id']/'TRACE.jsonl') for s in row['sessions']}
        root_traces[arm]=traces[sid]
        attempts=summarize_attempts(row,traces,windows,lines(stream/'TARGET_COST_EVENTS.jsonl'))
        # The legacy helper calls every window a VLA call; replace that name here.
        for a in attempts:a['action_decisions']=a.pop('vla_calls')
        memory=ActionMemory(artifact);checked=0
        source_table={e['source_step']:e for e in artifact['entries']}
        for inp,call,w in zip(inputs,calls,windows):
            runtime=inp['runtime_session'];assert runtime==call['runtime_session']==w['runtime_session_id']
            assert inp['observation_id']==call['observation_id']==w['source_observation_id']
            assert w['mode']=='fixed10'
            before=next(i for i in traces[runtime] if i['observation_id']==inp['observation_id'])
            index=traces[runtime].index(before);after=traces[runtime][index+1]
            observation={'prompt':inp['prompt'],'observation/state':np.asarray(inp['robot_state'],dtype=np.float32)}
            sb=inp['array_bindings']['observation/state']
            assert hashlib.sha256(observation['observation/state'].tobytes()).hexdigest()==sb['sha256']
            for key,camera in [('observation/image','agentview_image'),('observation/wrist_image','robot0_eye_in_hand_image')]:
                name=f'{before["step"]:04d}_{camera}.png';blob=(stream/'stage_trace'/runtime/name).read_bytes()
                checked_image(blob,before['images'][name],inp['array_bindings'][key]);pixels,shape=png_pixels(blob)
                observation[key]=np.frombuffer(pixels,dtype=np.uint8).reshape(shape)
            if arm=='F' and runtime==sid:
                actions,decision=memory.choose(observation)
                assert decision['reason']==call['decision']['reason']
                assert decision.get('source_step')==call['decision'].get('source_step')
            else:actions=None
            assert (actions is not None)==(call['origin']=='memory')
            if actions is not None:
                np.testing.assert_array_equal(actions,np.asarray(after['actions'],dtype=np.float32))
                np.testing.assert_array_equal(actions,np.asarray(source_table[decision['source_step']]['actions'],dtype=np.float32))
                assert inp['prompt']==GOAL
                checked+=1
        vla=sum(c['origin']=='vla' for c in calls)
        assert len(actual)==vla
        receipt=read(stream/'ACTION_MEMORY_RECEIPT.json')
        assert receipt['actual_vla_calls']==vla and receipt['memory_calls']==checked
        assert receipt['total_action_decisions']==len(calls) and receipt['persistent_model_updates']==0
        spans=defaultdict(float)
        for e in events:
            if e.get('event')=='end':spans[e['name']]+=e['duration_seconds']
        worker=read(group/'STATUS.json');inheritance=read(stream/sid/'INHERITANCE.json')
        rejected=[c for c in calls if c['decision']['reason']=='similarity_rejected']
        dominant=Counter(max(c['decision']['metrics'],key=lambda k:c['decision']['metrics'][k]/LIMITS[k]) for c in rejected)
        rows.append(dict(arm=arm,environment_success=row['environment_success'],semantic=[o['verdict'] for o in row['outcomes']],
            chain=row['chain_status'],admission=row['experience_admission']['status'],attempts=attempts,
            all_steps=sum(a['steps'] for a in attempts),physical_resets=len(attempts),
            action_decisions=len(calls),actual_vla_calls=vla,memory_calls=checked,
            decisions=dict(Counter(c['decision']['reason'] for c in calls)),
            rejection_dominant_metric=dict(dominant),
            closest_rejected=min(rejected,key=lambda c:c['decision']['score']) if rejected else None,
            memory_hits=[dict(observation_id=c['observation_id'],source_step=c['decision']['source_step'],score=c['decision']['score']) for c in calls if c['origin']=='memory'],
            qwen_calls=len(list((stream/'service_calls').glob('*/REQUEST.json'))),
            agent_calls=sum(len(list((stream/a['session_id']).glob('agent_call_*/REQUEST.json'))) for a in attempts),
            case_seconds=row['seconds'],controller_seconds=worker['controller_seconds'],
            worker_seconds=worker['full_wall_seconds'],startup_seconds=worker['startup_seconds'],nested_stage_seconds=dict(spans),
            initial_experience_seal=inheritance['source_seal_sha256'],current_input_and_submitted_action_audit='PASSED'))
        for p in [group/'ROSTER.json',group/'STATUS.json',stream/'RESULT.json',stream/'OS_COST_EVENTS.jsonl',stream/'ACTION_MEMORY_CALLS.jsonl',stream/'ACTION_MEMORY_RECEIPT.json',*[stream/'stage_trace'/s/'TRACE.jsonl' for s in traces]]:
            sources[str(p.relative_to(run))]=sha(p)
    assert rows[0]['initial_experience_seal']==rows[1]['initial_experience_seal']
    sources['MEMORY.json']=sha(run/'MEMORY.json');sources['PROTOCOL.json']=sha(run/'PROTOCOL.json')
    return dict(status='ANALYZED',rows=rows,batch_wall_seconds=batch['seconds'],source_entries=len(artifact['entries']),
        same_actual_root_trace=root_traces['N']==root_traces['F'],
        compile_seconds=artifact['compile_seconds'],source_episode='task64/init0, already-paid discovery',
        parameter_training=False,learned_applicability=False,learning_net_gain_established=False,
        scope='One unseen-by-memory developer initial state. Source-bound instance action reuse, not fixed-rule versus learned-gate comparison or formal generalization.',
        accounting='Nested spans overlap. Source discovery remains charged. Action decisions and genuine VLA requests are different quantities. No speed inference from parallel campaign wall time.',source_sha256=sources)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    value=analyze(a.run)
    with a.output.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))
