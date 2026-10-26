"""Validate saved screen roster, runtime budgets, version binding and image bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def verify(root):
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    roster=json.loads((root/'ROSTER.json').read_text())
    result=json.loads((root/'stream/RESULT.json').read_text())
    assert result['status']=='COMPLETED'
    assert len(result['cases'])==len(roster)==protocol['max_cases']
    images=[]
    for p in (root/'stream/model_calls').glob('*.request.json'):
        request=json.loads(p.read_text())
        parts=[x for m in request['messages'] if isinstance(m.get('content'),list) for x in m['content']]
        texts=[x['text'] for x in parts if x.get('type')=='text' and 'VERIFICATION_BUNDLE:\n' in x['text']]
        assert len(texts)==1
        bundle=json.loads(texts[0].split('VERIFICATION_BUNDLE:\n',1)[1])
        sent=[x for x in parts if x.get('type')=='image_url']
        paths=bundle['initial_rgb_paths']+bundle['final_rgb_paths']
        assert len(sent)==len(paths)==4
        for rel,part in zip(paths,sent):
            matches=[w/rel for w in (root/'stream').glob('development_*/workspace') if (w/rel).is_file()]
            assert len(matches)==1
            a=hashlib.sha256(matches[0].read_bytes()).hexdigest()
            b=hashlib.sha256(base64.b64decode(part['image_url']['url'].split(',',1)[1])).hexdigest()
            assert a==b
            images.append({'path':str(matches[0].relative_to(root)),'sha256':a})
    attempts=0
    for index,(row,spec) in enumerate(zip(result['cases'],roster)):
        assert row['task_id']==spec['task_id'] and row['init_state_id']==spec['init_state_id']
        assert row['task_description']==spec['task_description']
        assert row.get('suite','libero_spatial')==protocol['suite']
        assert row['input_version']==index and row['input_token_budget']==protocol['token_budget']
        assert row['max_steps']==protocol['max_steps_per_attempt'] and row['chain_status']=='passed'
        assert 1<=len(row['outcomes'])<=1+protocol['max_replans']
        attempts+=len(row['outcomes'])
        native=row['native_terminal_by_session']
        assert set(native)=={o['session_id'] for o in row['outcomes']}
        assert all(0<=n['steps']<=protocol['max_steps_per_attempt'] for n in native.values())
        # First policy attempt must use the suite's original language; later rewrites are separately recorded.
        assert row['sessions'][0]['task_description']==spec['task_description']
        state=json.loads((root/'stream'/f'VERSION_{index+1}.json').read_text())
        assert state['version']==index+1 and state['skills']=={} and state['token_budget']==protocol['token_budget']
    assert attempts<=protocol['max_attempts']
    assert len(images)==4*attempts, 'One paired evidence call expected per actual attempt'
    return {'scope':'Completed development baseline; no learned-control claim',
            'image_checks':images,'case_version_roster_checks':len(roster),
            'actual_attempts':attempts,'fixed_token_budget':protocol['token_budget'],
            'accepted_skill_count':0,'protocol_sha256':hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    result=verify(args.root)
    with (args.root/'EVIDENCE_INTEGRITY.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'verified_images':len(result['image_checks']),
                     'verified_cases':result['case_version_roster_checks'],'attempts':result['actual_attempts']}))
