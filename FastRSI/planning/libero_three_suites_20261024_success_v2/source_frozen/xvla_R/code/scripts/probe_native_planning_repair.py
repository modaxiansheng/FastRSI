"""Real-model saved-request diagnostic check; no robot, no experience updates."""
import hashlib,json,os,time
from pathlib import Path
import httpx

root=Path(os.environ['FASTRSI_NATIVE_RUN']);root.mkdir(exist_ok=True)
source=Path('/project/runs/native_stream_v3_20261003/stream/native_stream_0_t0_i3')
feedback=('Error: draft saved but NOT dispatched. Missing required read: '
          'call read_file with path="skills/benchmarking/SKILL.md". '
          'Patching Session fields cannot fix a missing read. Do not append again. '
          'The read itself will recheck the saved draft; then follow any remaining diagnostic.')
rows=[];started=time.time()
for index in (4,10):
    folder=root/('call_'+str(index));folder.mkdir()
    original=json.loads((source/('agent_call_'+str(index))/'REQUEST.json').read_text())
    candidates=[]
    for p in (source.parent/'service_calls').glob('*/REQUEST.json'):
        data=json.loads(p.read_text())
        if data['messages']==original['messages']:candidates.append((p,data))
    assert len(candidates)==1,'Ambiguous saved service request'
    p,body=candidates[0]
    assert body['messages'][-1]['role']=='tool' and 'Native benchmarking skill was not read' in body['messages'][-1]['content']
    before=body['messages'][-1]['content'];body['messages'][-1]['content']=feedback
    (folder/'SOURCE.json').write_text(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'old_tool_result':before,'change':'Only last tool diagnostic; complete preceding history, tool schemas and generation settings retained'},indent=2))
    (folder/'UPDATED_REQUEST.json').write_text(json.dumps(body,indent=2))
    tick=time.time()
    with httpx.Client(timeout=120) as client:
        response=client.post('http://127.0.0.1:49510/v1/chat/completions',json=body)
        (folder/'HTTP_RESPONSE.json').write_text(json.dumps({'status':response.status_code,'body':response.json()},indent=2))
        response.raise_for_status();value=response.json()
    calls=value['choices'][0]['message'].get('tool_calls',[]);reads=[]
    for call in calls:
        f=call['function'];args=json.loads(f['arguments'])
        if f['name']=='read_file':reads.append(args.get('path',''))
        if f['name']=='read_files':reads.extend(args.get('paths',[]))
    correct=any(str(path).endswith('skills/benchmarking/SKILL.md') for path in reads)
    rows.append({'source_call':index,'seconds':time.time()-tick,'required_read_selected':correct,'tool_names':[c['function']['name'] for c in calls],'usage':value.get('usage')})
result={'status':'PASSED' if all(x['required_read_selected'] for x in rows) else 'FAILED','rows':rows,'seconds':time.time()-started,'robot_episodes':0,'scope':'Saved-input diagnostic selection only; does not prove fresh full planning or learning'}
(root/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if result['status']!='PASSED':raise SystemExit(1)
