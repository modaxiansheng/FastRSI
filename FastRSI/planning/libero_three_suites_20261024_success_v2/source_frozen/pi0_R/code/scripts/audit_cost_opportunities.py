"""Read-only audit of saved real requests and full-OS traces; no modeled savings."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_costs import summarize


def score_fields(value, path=''):
    """Locate evaluator-derived fields for input provenance, not decision features."""
    found=[]
    if isinstance(value,dict):
        for key,item in value.items():
            name=path+'.'+key if path else key
            if key in ('success','native_success','environment_success','reward','total_reward','return_value'):
                found.append(name)
            found.extend(score_fields(item,name))
    elif isinstance(value,list):
        for i,item in enumerate(value):found.extend(score_fields(item,path+f'[{i}]'))
    return found


def audit(folder):
    traces=sorted(folder.glob('*_COST_EVENTS.jsonl'))
    cost=summarize(traces)
    events=[json.loads(line) for p in traces for line in p.read_text().splitlines() if line.strip()]
    result=json.loads((folder/'RESULT.json').read_text())
    calls=[]
    for p in sorted((folder/'model_calls').glob('*.request.json')):
        req=json.loads(p.read_text())
        rp=p.with_name(p.name.replace('.request.','.response.'))
        resp=json.loads(rp.read_text()) if rp.exists() else {}
        texts=[part['text'] for m in req.get('messages',[]) if isinstance(m.get('content'),list)
               for part in m['content'] if part.get('type')=='text']
        bundle=json.loads(next(t for t in texts if 'VERIFICATION_BUNDLE:\n' in t).split('VERIFICATION_BUNDLE:\n',1)[1])
        chars={key:len(bundle.get(key,'')) for key in ['history_md','lessons_md']}
        calls.append({'request':p.name,'request_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'session_id':bundle['session']['session_id'],
            'response_present':rp.exists(),'usage':resp.get('usage',{}),
            'max_tokens':req.get('max_tokens'),'history_chars':chars['history_md'],
            'lesson_chars':chars['lessons_md'],
            'serialized_bundle_chars':len(json.dumps(bundle,ensure_ascii=False,indent=2)),
            'evaluator_fields_in_original_bundle':score_fields(bundle),
            'accepted_skills':len(bundle.get('accepted_skills',{}))})
    by_id=cost['native_terminal_by_session']
    cases=[]
    for row in result.get('cases',[]):
        outcomes=row.get('outcomes',[])
        attempts=[]
        for outcome in outcomes:
            sid=outcome['session_id'];native=by_id.get(sid,{})
            attempts.append({'session_id':sid,'native_success':native.get('native_success'),
                'steps':native.get('steps'),'verdict':outcome.get('verdict'),
                'chain_code':outcome.get('code'),
                'false_accept_observed':native.get('native_success') is False and outcome.get('verdict')=='success',
                'false_reject_observed':native.get('native_success') is True and outcome.get('verdict')!='success'})
        cases.append({'suite':row.get('suite','libero_spatial'),'task_id':row.get('task_id',0),'init_state_id':row['init_state_id'],
            'mode':row['mode'],'environment_success':row['environment_success'],
            'chain_status':row['chain_status'],'seconds':row['seconds'],'attempts':attempts})
    decisions=[e for e in events if e.get('event')=='decision_input']
    categories=cost['categories']
    attempts=[a for c in cases for a in c['attempts']]
    return {'source':str(folder),'status':result['status'],'cost':cost,'cases':cases,'calls':calls,
        'counts':{'cases':len(cases),'native_successes':sum(c['environment_success'] for c in cases),
            'attempts':len(attempts),'model_calls':len(calls),
            'actual_recovery_attempts':sum(max(0,len(c['attempts'])-1) for c in cases),
            'proposed_replans':sum(a['verdict']=='replan' for a in attempts),
            'false_accepts':sum(a['false_accept_observed'] for a in attempts),
            'false_rejects':sum(a['false_reject_observed'] for a in attempts),
            'laya_model_calls':sum(bool(d['shadow']['called']) for d in decisions),
            'legal_option_sets':sorted(set(tuple(d['shadow']['offered']) for d in decisions))},
        'opportunities':[
            {'position':'semantic_verification','observed_calls':len(calls),
             'measured_service_seconds':categories.get('semantic_verification',{}).get('service_seconds'),
             'candidate_choices':['full_strong_analysis','accepted_evidence_based_verification'],
             'readiness':'Only strong path implemented; cheap verification requires independently validated current evidence. Past success alone cannot justify skipping verification.'},
            {'position':'history_and_experience_selection','observed_calls_with_history':sum(c['history_chars']>0 for c in calls),
             'candidate_choices':['full_history','retrieve_relevant_accepted_experience'],
             'readiness':'Real prompt scope can change through _load_bundle; information adequacy and latency effects untested. Characters are not token or time savings.'},
            {'position':'recovery_allocation','observed_recovery_attempts':sum(max(0,len(c['attempts'])-1) for c in cases),
             'candidate_choices':['execute_proposed_replan','reuse_validated_recovery','stop_candidate'],
             'readiness':'Original replan and terminal failure paths exist; no reusable recovery has been validated. Native-failure frequency alone is not avoidable cost.'},
            {'position':'candidate_experiment_allocation','observed_calls':None,
             'readiness':'No separate candidate-validation loop in current runner; missing instrumentation/operation, not evidence of zero cost.'},
            {'position':'service_startup','readiness':'Common engineering optimization; charge actual residency and startup. Not by itself experience-driven improvement.'}],
        'limits':['No treatment applied; no causal saving estimate.',
                  'Original verifier sees nested success/reward; its answer is not an independent quality label for a new router.',
                  'Shadow context previously forwarded runtime_result; never expose this unrestricted object to a future active router.',
                  'Span categories overlap; use campaign wall and resource residency for totals.',
                  'Development screening is not a held-out method comparison.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folders',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    report={'audit_version':1,'groups':[audit(p) for p in args.folders],
            'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps([{'source':r['source'],**r['counts']} for r in report['groups']],ensure_ascii=False,indent=2))
