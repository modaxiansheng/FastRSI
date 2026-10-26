"""Inspect actual stream inheritance and calls; do not infer learned efficiency."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter,defaultdict

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verifier_session(body):
    if not str(body.get('messages',[{}])[0].get('content','')).startswith('Verify the Agent session'):
        return None
    for m in body['messages']:
        parts=m.get('content',[])
        if isinstance(parts,str):parts=[{'type':'text','text':parts}]
        for part in parts or []:
            text=part.get('text','')
            if 'VERIFICATION_BUNDLE:\n' in text:
                bundle,_=json.JSONDecoder().raw_decode(text.split('VERIFICATION_BUNDLE:\n',1)[1].lstrip())
                return bundle['session']['session_id']
    raise ValueError('Verification request lacks a bindable Session bundle')

def analyze(run):
    run=Path(run);status=read(run/'STATUS.json');report=read(run/'stream/RESULT.json')
    rows=report['cases'];output=[];last_source=None
    events=lines(run/'stream/OS_COST_EVENTS.jsonl')
    starts={e['span_id']:e for e in events if e.get('event')=='begin'}
    services=[(p,read(p)) for p in (run/'stream/service_calls').glob('*/REQUEST.json')]
    verifier_requests=[(p,verifier_session(body)) for p,body in services if verifier_session(body)]
    stage=defaultdict(float)
    for e in events:
        if e.get('event')=='end':stage[e['name']]+=e['duration_seconds']
    for i,row in enumerate(rows):
        case=row['case'];folder=run/'stream'/case['id'];inherit=read(folder/'INHERITANCE.json')
        requests=sorted(folder.glob('agent_call_*/REQUEST.json'),key=lambda p:int(p.parent.name.rsplit('_',1)[1]))
        payloads=[read(p) for p in requests];first=payloads[0].get('messages',[]) if payloads else []
        responses=[read(p.parent/'RESPONSE.json') for p in requests if (p.parent/'RESPONSE.json').exists()]
        prior_ids=[r['case']['id'] for r in rows[:i] if r.get('experience_admission',{}).get('status')=='ADMITTED']
        first_text=json.dumps(first,ensure_ascii=False)
        trace=lines(folder/'AGENT_EVENTS.jsonl')
        inputs=[e for e in events if e.get('event')=='actual_policy_input' and e.get('case')==case['id']]
        choices=[e for e in events if e.get('event')=='feedback_window_choice' and e.get('case')==case['id']]
        memory_calls=[str(p.relative_to(run)) for p,v in zip(requests,payloads)
                      if {t.get('function',{}).get('name') for t in v.get('tools',[])}=={'save_memory'}]
        terminal=row.get('terminal_report',{})
        phase_calls=Counter()
        for event in events:
            if event.get('event')!='begin' or event.get('name')!='agent_model_call':continue
            parent=starts.get(event.get('parent_id'),{})
            if parent.get('case')==case['id']:phase_calls[parent['name']]+=1
        session_ids={s['session_id'] for s in row.get('sessions',[])}
        actual_verifier_requests=[str(p.relative_to(run)) for p,sid in verifier_requests if sid in session_ids]
        history_file=folder/'workspace/sessions'/(case['conversation_key'].replace(':','_')+'.jsonl')
        history=lines(history_file)
        meta=next((e for e in history if e.get('_type')=='metadata'),{})
        source_matches=row.get('accepted_source')==last_source
        if case['mode']=='development':last_source=row.get('experience_admission',{}).get('next_source',last_source)
        item={'case':case,'chain_status':row.get('chain_status'),'environment_success':row.get('environment_success'),
            'semantic_verdicts':[o.get('verdict') for o in row.get('outcomes',[])],
            'admission':row.get('experience_admission'),'source_matches_previous_admission':source_matches,
            'inheritance':inherit,'first_request_has_prior_case_ids':[s for s in prior_ids if s in first_text],
            'first_request_history_role_counts':dict(Counter(m.get('role') for m in first[1:-1])),
            'first_request_sha256':sha(requests[0]) if requests else None,
            'agent_model_calls':len(requests),'agent_calls_by_phase':dict(phase_calls),
            'verifier_model_calls':len(actual_verifier_requests),'verifier_request_evidence':actual_verifier_requests,
            'agent_calls_with_saved_response':len(responses),
            'native_consolidation_requests':memory_calls,'last_consolidated':meta.get('last_consolidated'),
            'native_token_checks':[e.get('tokens') for e in trace if e.get('event')=='native_token_count'],
            'maintenance_enabled':terminal.get('native_experience_maintenance_enabled'),
            'maintenance_writes':terminal.get('maintenance_writes',[]),
            'vla_calls':len(inputs),'window_decisions':len(choices),
            'policy_prompt_unchanged':bool(inputs) and all(e.get('prompt')==case['task_description'] for e in inputs),
            'runtime_terminals':row.get('native_terminal_by_session'),
            'evaluation_learning_hashes_unchanged':row.get('experience_before')==row.get('experience_after') if case['mode']=='evaluation' else None,
            'case_seconds':row.get('seconds'),
            'queue_origins':[e.get('origin') for e in trace if e.get('event')=='queue_accepted'],
            'no_public_flow':not any(e.get('event')=='session_flow_outcome' for e in trace),
            'tool_blocks':[{'event':e['event'],'name':e.get('name'),'result':str(e.get('result',''))[:300]} for e in trace if e.get('event') in ('tool_blocked','terminal_phase_tool_blocked')]}
        output.append(item)
    complete=status.get('status')==report.get('status')=='COMPLETED'
    basic=complete and len(rows)==3 and all(x['chain_status']=='PASSED' and x['source_matches_previous_admission'] and x['inheritance']['preserved_before_agent'] for x in output)
    evaluation=[x for x in output if x['case']['mode']=='evaluation']
    basic=basic and len(evaluation)==1 and evaluation[0]['evaluation_learning_hashes_unchanged'] and evaluation[0]['maintenance_enabled'] is False
    continuity_exercised=bool(len(rows)>1 and rows[1].get('accepted_source'))
    evaluation_uses_admitted_snapshot=bool(evaluation and evaluation[0]['inheritance'].get('source'))
    # A clean run with every development result quarantined is not evidence of
    # actual accepted-experience continuity or evaluation using learned state.
    basic=basic and continuity_exercised and evaluation_uses_admitted_snapshot
    probe_path=run/'stream/planning_repair_probe/RESULT.json'
    probe=read(probe_path) if probe_path.exists() else None
    accounted=sum(c['agent_model_calls']+c['verifier_model_calls'] for c in output)+(len(probe['rows']) if probe else 0)
    receipts=[read(p.parent/'RECEIPT.json') for p,_ in services]
    calls_reconciled=accounted==len(services) and all(x.get('status')=='OK' for x in receipts)
    basic=basic and calls_reconciled
    return {'status':'PASSED' if basic else 'INCOMPLETE_OR_FAILED','scope':'Native continuity and fixed-evaluation isolation; not improvement effectiveness',
        'continuity_exercised':continuity_exercised,'evaluation_uses_admitted_snapshot':evaluation_uses_admitted_snapshot,
        'run_status':status,'cases':output,'stage_seconds_nested_or_overlapping':dict(stage),
        'qwen_service_requests_total':len(services),
        'qwen_calls_reconciled':calls_reconciled,'diagnostic_probe':probe,
        'qwen_service_totals_including_diagnostic':{k:sum(x.get(k,0) for x in receipts) for k in ('prompt_tokens','completion_tokens','generation_seconds','queue_seconds')},
        'full_wall_seconds':status.get('full_wall_seconds'),'controller_seconds':report.get('seconds'),
        'prior_failure_wall_seconds':read(run/'PROTOCOL.json').get('prior_cpu_failure',{}).get('full_wall_seconds',0),
        'prior_attempts':read(run/'PROTOCOL.json').get('prior_attempts',[]),
        'native_skill_generation_proven':False,'learning_efficiency_gain_established':False,'measured_saving_seconds':None,
        'decision_effect':'Prior information in a request proves availability/reading only; actual improvement needs a matched decision comparison.',
        'no_forced_consolidation':read(run/'PROTOCOL.json')['forced_consolidation'] is False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.run)
    with Path(a.output).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':result['status'],'cases':len(result['cases']),'full_wall_seconds':result['full_wall_seconds']}))
