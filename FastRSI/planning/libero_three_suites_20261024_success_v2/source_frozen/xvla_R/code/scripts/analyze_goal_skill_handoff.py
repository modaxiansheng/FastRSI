"""Audit explicit prefix execution, same-episode handoff and complete Agent costs."""
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.decision_corpus import checked_image
from analyze_stack_flow import summarize_attempts

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_bytes().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def analyze(run):
    batch=read(run/'STATUS.json');assert batch['status']=='COMPLETED'
    protocol=read(run/'PROTOCOL.json');bound=sha(run/'GOAL_SKILL.json')
    assert bound==protocol['goal_skill_probe']['sha256']
    rows=[];sources={};roots={}
    for arm in ('N','F'):
        group=run/arm;stream=group/'stream';result=read(stream/'RESULT.json')
        assert result['status']=='COMPLETED' and len(result['cases'])==1
        row=result['cases'][0];case=row['case'];sid=case['id']
        assert row['chain_status']=='PASSED'
        assert read(group/'ROSTER.json')==[case]
        assert case['goal_skill_probe']['enabled']==(arm=='F') and case['goal_skill_probe']['sha256']==bound
        expected=read(run/'PLAN.json')['cases'][0]
        assert all(case[k]==v for k,v in expected.items())
        assert case['feedback_window_mode']=='fixed10' and case['planning_entry']=='public_session_flow'
        events=lines(stream/'OS_COST_EVENTS.jsonl');calls=lines(stream/'GOAL_SKILL_CALLS.jsonl')
        actual=[e for e in events if e.get('event')=='actual_policy_input']
        windows=[e for e in events if e.get('event')=='feedback_window_choice']
        assert len(calls)==len(windows)==len(actual)
        traces={s['session_id']:lines(stream/'stage_trace'/s['session_id']/'TRACE.jsonl') for s in row['sessions']}
        roots[arm]=traces[sid]
        attempts=summarize_attempts(row,traces,windows,lines(stream/'TARGET_COST_EVENTS.jsonl'))
        for a in attempts:a['boundary_requests']=a.pop('vla_calls')
        execution=lines(stream/'GOAL_SKILL_EXECUTION.jsonl') if (stream/'GOAL_SKILL_EXECUTION.jsonl').exists() else []
        used=0;image_checks=0
        for inp,call,w in zip(actual,calls,windows):
            runtime=call['runtime_session'];step=call['step']
            assert runtime==w['runtime_session_id'] and step==w['step']
            assert call['goal']==case['task_description']==inp['prompt']
            before=next(i for i in traces[runtime] if i['step']==step)
            after=traces[runtime][traces[runtime].index(before)+1]
            assert after['step']-before['step']<=10
            for key,camera in [('observation/image','agentview_image'),('observation/wrist_image','robot0_eye_in_hand_image')]:
                name=f'{step:04d}_{camera}.png';blob=(stream/'stage_trace'/runtime/name).read_bytes()
                checked_image(blob,before['images'][name],inp['array_bindings'][key]);image_checks+=1
            is_skill=arm=='F' and runtime==sid and step<130
            assert (call['origin']=='goal_skill')==is_skill
            if is_skill:
                receipt=execution[used];used+=1
                assert receipt['session_id']==runtime and receipt['start_step']==step and receipt['end_step']==after['step']
                assert receipt['skill_sha256']==bound and receipt['object_pose_used'] is False
                np.testing.assert_array_equal(np.asarray(receipt['actual_actions'],np.float32),np.asarray(after['actions'],np.float32))
                assert not np.array_equal(np.asarray(receipt['actual_actions']),np.zeros((10,7)))
                assert inp['prediction_origin']=='goal_skill_request'
            else:
                np.testing.assert_array_equal(np.asarray(w['predicted_actions'][:after['step']-step],np.float32),np.asarray(after['actions'][:after['step']-step],np.float32))
        assert used==len(execution)
        if arm=='F':
            assert used==13 and [r['start_step'] for r in execution]==list(range(0,130,10))
            assert any(c['origin']=='vla' and c['runtime_session']==sid and c['step']==130 for c in calls)
        else:assert used==0
        target_events=lines(stream/'TARGET_COST_EVENTS.jsonl')
        resets=[e for e in target_events if e.get('event')=='native_observation' and e.get('method')=='reset']
        assert sum(e['session_id']==sid for e in resets)==1,'Episode reset during handoff'
        vla=sum(c['origin']=='vla' for c in calls);receipt=read(stream/'GOAL_SKILL_RECEIPT.json')
        assert receipt['actual_vla_calls']==vla and receipt['skill_requests']==used and receipt['boundary_requests']==len(calls)
        assert sum(e.get('event')=='end' and e.get('name')=='policy_client' for e in events)==vla
        spans={}
        for origin,ledger in [('os',events),('target',target_events)]:
            total=defaultdict(float)
            for e in ledger:
                if e.get('event')=='end':total[e['name']]+=e['duration_seconds']
            spans[origin]=dict(total)
        worker=read(group/'STATUS.json');inheritance=read(stream/sid/'INHERITANCE.json')
        rows.append(dict(arm=arm,environment_success=row['environment_success'],semantic=[o['verdict'] for o in row['outcomes']],chain=row['chain_status'],admission=row['experience_admission']['status'],attempts=attempts,
            all_steps=sum(a['steps'] for a in attempts),physical_resets=len(resets),boundary_requests=len(calls),actual_vla_calls=vla,skill_requests=used,goal_control_updates=sum(len(x['actual_actions']) for x in execution),
            qwen_calls=len(list((stream/'service_calls').glob('*/REQUEST.json'))),agent_calls=sum(len(list((stream/a['session_id']).glob('agent_call_*/REQUEST.json'))) for a in attempts),
            case_seconds=row['seconds'],controller_seconds=worker['controller_seconds'],worker_seconds=worker['full_wall_seconds'],startup_seconds=worker['startup_seconds'],nested_stage_seconds=spans,
            initial_experience_seal=inheritance['source_seal_sha256'],image_checks=image_checks,input_action_and_handoff_audit='PASSED'))
        for p in [group/'STATUS.json',stream/'RESULT.json',stream/'OS_COST_EVENTS.jsonl',stream/'TARGET_COST_EVENTS.jsonl',stream/'GOAL_SKILL_CALLS.jsonl',stream/'GOAL_SKILL_RECEIPT.json',*[stream/'stage_trace'/s/'TRACE.jsonl' for s in traces]]:
            sources[str(p.relative_to(run))]=sha(p)
    assert rows[0]['initial_experience_seal']==rows[1]['initial_experience_seal']
    assert roots['N'][0]==roots['F'][0]
    return dict(status='ANALYZED',rows=rows,batch_wall_seconds=batch['seconds'],same_initial_trace=True,training_runs=0,learned_applicability=False,learning_net_gain_established=False,
        scope='Previously seen developer init1; full public-optimized native Agent. Fixed source prefix plus policy handoff; not learned selection, independent evaluation, or original unoptimized RSI reproduction.',
        accounting='OS/target spans overlap. Method inherited acquisition/compilation/validation not zero. Parallel batch wall not method speed. 130 controller updates remain even when13 VLA calls are replaced.',source_sha256=sources)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    value=analyze(a.run)
    with a.output.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))
