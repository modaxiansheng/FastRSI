"""Descriptive paired replay report; deliberately no robot or learning gain claim."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(root):
    report=json.loads((root/'RESULT.json').read_text())
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    assert report['status']=='COMPLETED'
    rows=report['calls'];assert len(rows)==protocol['max_calls']
    paired=[]
    for tid in protocol['task_ids']:
        for repeat in range(len(protocol['orders'])):
            pair={r['scope']:r for r in rows if r['task_id']==tid and r['repeat']==repeat}
            assert set(pair)=={'full','current_episode'}
            a,b=pair['full'],pair['current_episode']
            assert a['source_sha256']==b['source_sha256']
            paired.append({'task_id':tid,'repeat':repeat,
                'removed_history_sessions':b['projection'].get('removed_sessions',0),
                'full_prompt_tokens':a['usage']['prompt_tokens'],
                'current_prompt_tokens':b['usage']['prompt_tokens'],
                'full_completion_tokens':a['usage']['completion_tokens'],
                'current_completion_tokens':b['usage']['completion_tokens'],
                'full_seconds':a['seconds'],'current_seconds':b['seconds'],
                'full_verdict':a['verdict']['verdict'],'current_verdict':b['verdict']['verdict'],
                'verdict_matches':a['verdict']['verdict']==b['verdict']['verdict'],
                'projection_seconds':b['projection_seconds']})
    totals={}
    for scope in ['full','current_episode']:
        items=[r for r in rows if r['scope']==scope]
        totals[scope]={'calls':len(items),'http_seconds':sum(r['seconds'] for r in items),
                       'projection_seconds':sum(r['projection_seconds'] for r in items),
                       'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in items),
                       'completion_tokens':sum(r['usage']['completion_tokens'] for r in items)}
    a,b=totals['full'],totals['current_episode']
    return {'status':'COMPLETED','paired':paired,'totals':totals,
            'observed_prompt_token_reduction':1-b['prompt_tokens']/a['prompt_tokens'],
            'observed_http_reduction':1-b['http_seconds']/a['http_seconds'],
            'verdict_matches':sum(p['verdict_matches'] for p in paired),
            'pair_count':len(paired),'laya_calls':0,'learned_updates':0,
            'source_status':json.loads((root/'STATUS.json').read_text()),
            'limits':['First call includes cold generation effects; all calls retained and order counterbalanced.',
                      'Four selected development inputs, repeated twice; repeats are not independent task samples.',
                      'Current scoring fields present in both original inputs; no independent visual-quality claim.',
                      'Only history was filtered; all skills and lessons remain. No learned routing policy.',
                      'No downstream environment executed under this intervention; no robot or end-to-end cost-reduction claim.',
                      'Full experiment wall, loading and residency are development costs, not ignored overhead.'],
            'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    report=analyze(args.root)
    with (args.root/'PAIRED_AUDIT.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ['totals','observed_prompt_token_reduction','observed_http_reduction','verdict_matches','pair_count']},indent=2))
