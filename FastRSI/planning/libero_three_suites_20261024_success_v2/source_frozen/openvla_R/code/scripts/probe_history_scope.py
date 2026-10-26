"""Counterbalanced actual-inference probe; changes only retained history entries."""
import copy
import hashlib
import json
import os
import re
import time
import traceback
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
import yaml
from fastrsi.contracts import json_object_text
from PhyAgentOS.agent.session_verifier import VerificationVerdict
from PhyAgentOS.verification.service import _normalize


def get_bundle(request):
    for mi,message in enumerate(request['messages']):
        if not isinstance(message.get('content'),list):continue
        for pi,part in enumerate(message['content']):
            if part.get('type')=='text' and 'VERIFICATION_BUNDLE:\n' in part['text']:
                prefix,body=part['text'].split('VERIFICATION_BUNDLE:\n',1)
                return mi,pi,prefix,json.loads(body)
    raise ValueError('Missing complete source bundle')


def project(request, scope):
    result=copy.deepcopy(request)
    mi,pi,prefix,bundle=get_bundle(result)
    before=copy.deepcopy(bundle)
    if scope=='full':return result,{'scope':scope,'changed':False}
    if scope!='current_episode':raise ValueError('Unknown history scope')
    history=bundle['history_md']
    match=re.search(r'```yaml\s*\n(.*?)\n```',history,re.DOTALL)
    if not match:raise ValueError('No structured source history')
    document=yaml.safe_load(match.group(1))
    assert document['version']=='runtime_session_history_v1'
    sid=bundle['session']['session_id'];base=sid.split('_replan_',1)[0]
    retained={key:value for key,value in document['sessions'].items()
              if key==base or key.startswith(base+'_replan_')}
    assert sid in retained, 'Current evidence would be dropped'
    assert document['last_session_id'] in retained, 'History points outside current episode'
    removed=len(document['sessions'])-len(retained)
    document['sessions']=retained
    # Return byte-identical input when there is no membership change.
    if removed==0:return result,{'scope':scope,'changed':False,'removed_sessions':0}
    body=yaml.safe_dump(document,sort_keys=False,allow_unicode=True)
    bundle['history_md']=history[:match.start(1)]+body.rstrip('\n')+history[match.end(1):]
    assert {k:v for k,v in bundle.items() if k!='history_md'}=={k:v for k,v in before.items() if k!='history_md'}
    result['messages'][mi]['content'][pi]['text']=prefix+'VERIFICATION_BUNDLE:\n'+json.dumps(bundle,ensure_ascii=False,indent=2)
    # Images, system prompt, output budget and all non-bundle parts must match.
    check=copy.deepcopy(result)
    check['messages'][mi]['content'][pi]=copy.deepcopy(request['messages'][mi]['content'][pi])
    assert check==request
    return result,{'scope':scope,'changed':True,'removed_sessions':removed,
                   'history_chars_before':len(history),'history_chars_after':len(bundle['history_md']),
                   'lessons_preserved':True,'skills_preserved':True,'paired_images_preserved':True}


def main():
    root=Path(os.environ['FASTRSI_PROBE_DIR'])
    config=json.loads((root/'PROTOCOL.json').read_text())
    source=Path('/project/runs')/config['source_campaign']/'stream'
    candidates={}
    for p in (source/'model_calls').glob('*.request.json'):
        req=json.loads(p.read_text());bundle=get_bundle(req)[3]
        candidates[bundle['session']['session_id']]=(p,req)
    report={'status':'RUNNING','calls':[],'scope':config['purpose']}
    start=time.perf_counter();opener=build_opener(ProxyHandler({}))
    try:
        for tid in config['task_ids']:
            sid=f'SCREEN_r1_development_it{tid}_i{config["init_state_id"]}'
            p,request=candidates[sid]
            assert request['max_tokens']==config['max_tokens']
            for repeat,order in enumerate(config['orders']):
                for scope in order:
                    ts=time.perf_counter();req,receipt=project(request,scope)
                    projection=time.perf_counter()-ts
                    name=f't{tid}_repeat{repeat}_{scope}'
                    data=json.dumps(req,ensure_ascii=False).encode()
                    (root/(name+'.request.json')).write_bytes(data)
                    ts=time.perf_counter()
                    with opener.open(Request('http://127.0.0.1:49511/v1/chat/completions',data=data,
                          headers={'Content-Type':'application/json'}),timeout=180) as response:
                        answer=json.load(response)
                    seconds=time.perf_counter()-ts
                    (root/(name+'.response.json')).write_text(json.dumps(answer,ensure_ascii=False,indent=2))
                    choice=answer['choices'][0]
                    assert choice.get('finish_reason')=='stop'
                    verdict=VerificationVerdict.model_validate(_normalize(json.loads(json_object_text(choice['message']['content']))))
                    assert verdict.verifier_status=='completed'
                    row={'task_id':tid,'repeat':repeat,'scope':scope,'name':name,
                         'source_request':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                         'seconds':seconds,'projection_seconds':projection,'usage':answer.get('usage',{}),
                         'verdict':verdict.model_dump(mode='json'),'projection':receipt,
                         'controls_execution':False}
                    report['calls'].append(row)
                    (root/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                    print(json.dumps({k:row[k] for k in ['task_id','repeat','scope','seconds','usage']}),flush=True)
        assert len(report['calls'])==config['max_calls']
        report['status']='COMPLETED'
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        (root/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
