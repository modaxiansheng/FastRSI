"""Account for real window choices and full lifecycle, including negative results."""
import collections, hashlib, json
from pathlib import Path
from analyze_representative_batch import analyze, lines, read, ROOT

NAME='feedback_window_v1_20261003'

def main():
    global NAME
    import argparse,re
    parser=argparse.ArgumentParser();parser.add_argument("--name",default=NAME);parser.add_argument("--completed-pairs-only",action="store_true");args=parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_]+",args.name):raise ValueError("Invalid run name")
    NAME=args.name
    run=ROOT/'runs'/NAME;out=ROOT/'planning'/NAME
    campaign_status=read(run/'STATUS.json')['status']
    assert campaign_status=='COMPLETED' or (args.completed_pairs_only and campaign_status=='FAILED')
    arms=read(run/'PROTOCOL.json').get('active_arms',['N','F','R'])
    pairs=sorted(run.glob('pair*'));excluded=[];selected=[]
    for pair in pairs:
        status=read(pair/'STATUS.json')['status']
        if status=='COMPLETED':selected.append(pair)
        else:
            assert args.completed_pairs_only
            spent={a:read(pair/a/'STATUS.json').get('full_wall_seconds',0) for a in arms}
            excluded.append({'pair':pair.name,'status':status,'reason':'No valid completed comparison; original infrastructure failure retained','paid_lifecycle_seconds_by_arm':spent,'paid_total_seconds':sum(spent.values())})
    assert selected
    details={};rows=[]
    for pair in selected:
        image_hashes=[];first_action_prefixes=[]
        for arm in arms:
            group=pair/arm;full=analyze(group);details[pair.name+'/'+arm]=full
            result=read(group/'stream/RESULT.json')['cases'][0];case=result['case']
            events=lines(group/'stream/OS_COST_EVENTS.jsonl');choices=[e for e in events if e.get('event')=='feedback_window_choice']
            inputs=[e for e in events if e.get('event')=='actual_policy_input']
            target=lines(group/'stream/TARGET_COST_EVENTS.jsonl')
            obs=[e for e in target if e.get('event')=='native_observation']
            own=[e for e in obs if str(e.get('session_id','')).startswith(case['id'])]
            assert all(e['prompt']==case['task_description'] for e in inputs)
            assert len(choices)==len(inputs) and choices
            assert result['chain_status']=='PASSED' and result['terminal_report']['read_coverage']['complete']
            assert len(result['native_terminal_by_session'])==1
            assert sum(e.get('method')=='reset' for e in own)==1
            assert full['cases'][0]['initial_image_hashes'];image_hashes.append(full['cases'][0]['initial_image_hashes'])
            assert result['terminal_report']['prefetch_enabled'] is False
            if read(run/'PROTOCOL.json').get('public_report_destination_binding'):
                assert result['terminal_report'].get('bound_report_destination') is True
            agent_events=lines(group/'stream'/case['id']/'AGENT_EVENTS.jsonl')
            mode=result.get('task_window_decision',result.get('window_learning_decision',{})).get('selected_mode',case['feedback_window_mode']);counter=collections.Counter(e['requested_steps'] for e in choices)
            if mode.startswith('fixed'):assert set(counter)=={int(mode[5:])}
            else:
                for e in choices:assert e['requested_steps']==(5 if e['within_chunk_transition'] or e['boundary_transition'] else 10)
            trace=lines(group/'stream/stage_trace'/case['id']/'TRACE.jsonl')
            submitted=[e for e in trace if e.get('method')=='action_chunk']
            assert len(submitted)==len(choices)
            first_action_prefixes.append(submitted[0]['actions'][:5])
            for choice,actual in zip(choices,submitted):
                assert len(actual['actions'])==choice['submitted_steps']
                assert [float(a[6])>0 for a in actual['actions']]==choice['predicted_gripper_positive'][:choice['submitted_steps']]
                assert actual['step']>choice['step'] and actual['step']<=choice['step']+choice['submitted_steps']
            steps=sum(v['steps'] for v in result['native_terminal_by_session'].values());assert steps<=520
            rows.append({'pair':pair.name,'arm':arm,'mode':mode,'task':case['task_id'],'init':case['init_state_id'],
                         'window_learning_decision':result.get('window_learning_decision'),
                         'environment_success':result['environment_success'],'verdicts':[o.get('verdict') for o in result['outcomes']],
                         'bound_report_destination':result['terminal_report'].get('bound_report_destination',False),
                         'report_destination_corrections':sum(e.get('event')=='terminal_report_destination_bound' for e in agent_events),
                         'report_guard_rejections':[e for e in agent_events if e.get('event') in ('terminal_phase_tool_blocked','report_artifact_rejected','report_required_reference_rejected')],
                         'chain':result['chain_status'],'admission':result.get('experience_admission',{}).get('status'),
                         'steps':steps,'vla_calls':len(inputs),'requested_window_counts':dict(counter),
                         'selector_seconds':sum(e['selector_seconds'] for e in choices),
                         'full_wall_seconds':read(group/'STATUS.json')['full_wall_seconds'],
                         'startup_seconds':read(group/'STATUS.json')['startup_seconds'],
                         'controller_seconds':read(group/'STATUS.json')['controller_seconds'],
                         'stage_seconds':full['stage_seconds'],'qwen_by_purpose':full['purpose_summary']})
        assert all(h==image_hashes[0] for h in image_hashes)
        assert all(len(prefix)==5 for prefix in first_action_prefixes)
        assert max(abs(float(v)-float(w)) for prefix in first_action_prefixes for a,b in zip(prefix,first_action_prefixes[0]) for v,w in zip(a,b))<1e-6
    totals={}
    for arm in arms:
        r=[x for x in rows if x['arm']==arm]
        totals[arm]={'environment_successes':sum(x['environment_success'] is True for x in r),
                     'semantic_successes':sum(x['verdicts']==['success'] for x in r),'episodes':len(r),
                     **{k:sum(x[k] for x in r) for k in ['vla_calls','steps','full_wall_seconds','controller_seconds','selector_seconds']}}
    summary={'scope':read(run/'PROTOCOL.json')['phase']+'; no learned gain or noninferiority claim',
             'campaign_status':campaign_status,'incomplete_pairs':excluded,
             'project_lifecycle_seconds_including_incomplete':sum(x['paid_total_seconds'] for x in excluded)+sum(r['full_wall_seconds'] for r in rows),
             'rows':rows,'totals':totals,'study_wall_seconds':read(run/'STATUS.json')['full_wall_seconds'],
             'accounting':'Lifecycle includes startup, native execution, validation, reporting, triggered native learning and teardown. Stage timings can overlap; never sum nested spans.',
             'protocol_sha256':hashlib.sha256((run/'PROTOCOL.json').read_bytes()).hexdigest()}
    (out/'ANALYSIS.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
