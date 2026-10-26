"""Audit saved full-OS rounds; report observed costs and unresolved evidence separately."""
import argparse
import base64
import collections
from datetime import datetime
import hashlib
import json
import math
import re
from pathlib import Path
from summarize_costs import summarize


def elapsed(start,end):
    def parse(value):
        value=re.sub(r'(\.\d{6})\d+',r'\1',value)
        return datetime.fromisoformat(value.replace('Z','+00:00'))
    return (parse(end)-parse(start)).total_seconds()


def analyze(root):
    status=json.loads((root/'STATUS.json').read_text())
    report={'campaign_status':status['status'],'groups':[],'arms':{},
        'scope':'Engineering pilot on one task; not noninferiority or general fast-path quality evidence',
        'cost_scope':'Per-group wall includes preflight, startup/cache preparation, all development and evaluation, updates and teardown. Shared earlier setup and interrupted campaigns are separate; human time and actual GPU utilization are unmeasured.'}
    for group in status.get('groups',[]):
        folder=root/group['arm']/('round_'+str(group['round']))
        files=list(folder.glob('*_COST_EVENTS.jsonl'))
        cost=summarize(files) if files else {}
        result=json.loads((folder/'RESULT.json').read_text()) if (folder/'RESULT.json').exists() else {}
        version_path=folder.parent/('VERSION_'+str(group['round'])+'.json')
        if result.get('version_sha256'):
            assert hashlib.sha256(version_path.read_bytes()).hexdigest()==result['version_sha256'], 'Published version changed after evaluation'
            version=json.loads(version_path.read_text())
            assert all(lesson['session_id'].startswith(group['arm']+'_') for lesson in version['lessons']), 'Cross-arm experience contamination'
        events=[json.loads(line) for file in files for line in file.read_text().splitlines()]
        decisions=[e for e in events if e.get('event')=='decision_input']
        native=[e for e in events if e.get('event')=='native_observation' and e.get('method')=='action_chunk']
        calls=[]
        image_checks=[]
        for path in sorted((folder/'model_calls').glob('*.request.json')):
            request=json.loads(path.read_text());response_path=path.with_name(path.name.replace('.request.','.response.'))
            response=json.loads(response_path.read_text()) if response_path.exists() else {}
            texts=[x.get('text','') for m in request.get('messages',[]) if isinstance(m.get('content'),list) for x in m['content'] if x.get('type')=='text']
            bundle=json.loads(next(t for t in texts if 'VERIFICATION_BUNDLE:\n' in t).split('VERIFICATION_BUNDLE:\n',1)[1])
            image_parts=[x for m in request['messages'] if isinstance(m.get('content'),list)
                         for x in m['content'] if x.get('type')=='image_url']
            expected_paths=bundle.get('initial_rgb_paths',[])+bundle.get('final_rgb_paths',[])
            assert len(image_parts)==len(expected_paths)==4, 'Incomplete paired visual evidence'
            for rel,part in zip(expected_paths,image_parts):
                matches=[workspace/rel for workspace in folder.glob('*/workspace') if (workspace/rel).is_file()]
                assert len(matches)==1, 'Missing or ambiguous image artifact'
                sent=hashlib.sha256(base64.b64decode(part['image_url']['url'].split(',',1)[1])).hexdigest()
                stored=hashlib.sha256(matches[0].read_bytes()).hexdigest()
                assert sent==stored, 'Model received different image bytes'
                image_checks.append({'path':rel,'sha256':sent,'matched':True})
            calls.append({'request':path.name,'session_id':bundle['session']['session_id'],
                'actual_max_tokens':request.get('max_tokens'),'version':bundle.get('fixed_experience_version'),
                'lessons_bytes':len(bundle.get('lessons_md','').encode()),
                'accepted_skill_count':len(bundle.get('accepted_skills',{})),
                'usage':response.get('usage',{}),'response_present':response_path.exists()})
        receipts=json.loads((folder/'STOP_RECEIPT.json').read_text())
        residency=sum(elapsed(r['started'],r['finished']) for r in receipts if r['role'] in ('pi05','qwen'))
        cases=[{k:c.get(k) for k in ['mode','init_state_id','environment_success','verification_status',
                    'chain_status','seconds','input_version','input_token_budget','evaluation_updates_blocked']}
               for c in result.get('cases',[])]
        row={'arm':group['arm'],'round':group['round'],'status':group['status'],
             'full_wall_seconds':group['complete_wall_seconds'],
             'startup_seconds':group.get('startup_seconds'),'controller_seconds':group.get('result_seconds'),
             'paired_image_checks':image_checks,
             'allocated_hcu_seconds':residency,'cases':cases,'cost':cost,'actual_calls':calls,
             'shadow_decisions':len(decisions),
             'shadow_model_calls':sum(bool(e['shadow']['called']) for e in decisions),
             'legal_option_sets':[e['shadow']['offered'] for e in decisions],
             'feedback_reads':sum(bool(e.get('lessons_present')) for e in decisions),
             'evaluation_write_blocks':sum(e.get('event')=='persistent_update_blocked' for e in events),
             'engine_failed_attempts':sum(e.get('event')=='engine_completion' and not e.get('ok') for e in events),
             'policy_action_steps':sum(e.get('steps',0) for e in cost.get('native_terminal_by_session',{}).values()),
             'settling_env_steps':10*cost.get('categories',{}).get('environment_reset',{}).get('calls',0)}
        report['groups'].append(row)
    for arm in 'ABC':
        rows=[r for r in report['groups'] if r['arm']==arm]
        if not rows:continue
        cases=[c for r in rows for c in r['cases']];evals=[c for c in cases if c['mode']=='evaluation']
        report['arms'][arm]={'episodes':len(cases),'native_successes':sum(c['environment_success'] for c in cases),
            'evaluation_episodes':len(evals),'evaluation_successes':sum(c['environment_success'] for c in evals),
            'chain_failures':sum(c['chain_status']!='passed' for c in cases),
            'full_wall_seconds':sum(r['full_wall_seconds'] for r in rows),
            'controller_seconds':sum(r['controller_seconds'] or 0 for r in rows),
            'allocated_hcu_seconds':sum(r['allocated_hcu_seconds'] for r in rows),
            'prompt_tokens':sum(c['usage'].get('prompt_tokens',0) for r in rows for c in r['actual_calls']),
            'completion_tokens':sum(c['usage'].get('completion_tokens',0) for r in rows for c in r['actual_calls']),
            'model_calls':sum(len(r['actual_calls']) for r in rows),
            'policy_action_steps':sum(r['policy_action_steps'] for r in rows),
            'settling_env_steps':sum(r['settling_env_steps'] for r in rows),
            'engine_failed_attempts':sum(r['engine_failed_attempts'] for r in rows)}
    round_sets={arm:sorted(r['round'] for r in report['groups'] if r['arm']==arm and r['status']=='PASSED') for arm in 'ABC'}
    balanced=set(report['arms'])==set('ABC') and round_sets['A']==round_sets['B']==round_sets['C']
    balanced=balanced and len({r['episodes'] for r in report['arms'].values()})==1
    if balanced:
        a,b,c=[report['arms'][arm] for arm in 'ABC']
        report['descriptive_cost_reductions']={
            'C_vs_A_full_wall':1-c['full_wall_seconds']/a['full_wall_seconds'],
            'C_vs_B_full_wall':1-c['full_wall_seconds']/b['full_wall_seconds'],
            'causal_claim':False,'reason':'One small task, no independent repeat; B degenerates to slow path. Token ceiling change need not change actual computation.'}
    initial_hashes={arm:hashlib.sha256((root/arm/'VERSION_0.json').read_bytes()).hexdigest()
                    for arm in 'ABC' if (root/arm/'VERSION_0.json').exists()}
    report['initial_version_hashes']=initial_hashes
    report['identical_initial_experience']=len(initial_hashes)==3 and len(set(initial_hashes.values()))==1
    report['campaign_wall_seconds']=status.get('campaign_wall_seconds')
    report['source_manifest_sha256']=hashlib.sha256((root/'CODE_MANIFEST.json').read_bytes()).hexdigest()
    report['auditor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    report=analyze(args.root)
    with (args.root/'AUDIT_REPORT.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':report['campaign_status'],'arms':report['arms']},ensure_ascii=False,indent=2))
