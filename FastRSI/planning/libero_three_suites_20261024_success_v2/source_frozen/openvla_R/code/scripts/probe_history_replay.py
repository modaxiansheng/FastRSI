"""Replay saved failure context with an explicitly recorded common-tool API delta."""
import asyncio
import copy
import hashlib
import json
import os
import time
import traceback
import urllib.request
from pathlib import Path
from fastrsi.file_append import AppendYamlItemTool
from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block
from PhyAgentOS.runtime.schemas import SessionSpec

SOURCE=Path('/project/runs/native_protocol_v12_20261002/stream/service_calls/call_f635428ca4574a5eb11b728e4d4c5156/REQUEST.json')
OLD=Path('/project/runs/native_protocol_v12_20261002/stream/planning_history_probe/workspace/SESSIONS.md')

async def main():
    root=Path(os.environ['FASTRSI_NATIVE_RUN'])/'failed_request_replay';root.mkdir()
    source=SOURCE.read_bytes();(root/'SOURCE_REQUEST.json').write_bytes(source)
    request=json.loads(source);original=copy.deepcopy(request)
    tool=AppendYamlItemTool(workspace=root,allowed_dir=root)
    schema={'type':'function','function':{'name':tool.name,'description':tool.description,'parameters':tool.parameters}}
    instruction=('A common file API has now been approved. Complete the SAME pending Session append using append_yaml_item. '
        'Use the original SESSIONS.md path, list_key="sessions", and item containing ONLY the new Session object. '
        'Retain the original requested task and all execution fields. Do not reproduce historical records. '
        'Serialize task_description as the final property in item, after all nested mappings, without changing any field values. '
        'This supersedes only the earlier instruction to append using edit_file. Return one append_yaml_item tool call.')
    request['tools'].append(schema);request['messages'].append({'role':'user','content':instruction})
    assert request['messages'][:-1]==original['messages'] and request['tools'][:-1]==original['tools']
    assert {k:v for k,v in request.items() if k not in ('messages','tools')}=={k:v for k,v in original.items() if k not in ('messages','tools')}
    body=json.dumps(request,ensure_ascii=False).encode();(root/'REQUEST.json').write_bytes(body)
    (root/'REQUEST_DELTA.json').write_text(json.dumps({'added_tool':schema,'added_user_instruction':instruction,'all_prior_messages_and_tools_preserved':True},indent=2))
    start=time.time();row={'status':'RUNNING','source':str(SOURCE),
        'source_request_sha256':hashlib.sha256(source).hexdigest(),'request_sha256':hashlib.sha256(body).hexdigest(),
        'scope':'saved failure context plus approved append API and one explicit instruction; not byte-identical request replay; no robot or learning'}
    try:
        req=urllib.request.Request('http://127.0.0.1:49510/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=300) as response:result=json.load(response)
        (root/'RESPONSE.json').write_text(json.dumps(result,indent=2))
        choice=result['choices'][0];assert choice['finish_reason']=='tool_calls'
        calls=choice['message']['tool_calls'];assert len(calls)==1
        call=calls[0]['function'];assert call['name']=='append_yaml_item'
        args=json.loads(call['arguments']);assert Path(args['path'])==OLD
        assert args['list_key']=='sessions'
        target=root/'SESSIONS.md';target.write_bytes(OLD.read_bytes())
        before=read_yaml_block(target)['sessions'];old_ids={s['session_id'] for s in before}
        cloned=copy.deepcopy(args);cloned['path']=str(target)
        response=await tool.execute(**cloned);assert not response.startswith('Error'),response
        after=read_yaml_block(target)['sessions']
        assert after[:-1]==before and len(after)==len(before)+1,'History was modified'
        assert after[-1]['session_id'] not in old_ids
        SessionSpec.model_validate(after[-1])
        row.update(status='PASSED',service_call_id=result['id'],usage=result['usage'],
            argument_characters=len(call['arguments']),preserved_sessions=len(before),proposed_session=after[-1],
            limitation='Complete append and preserved history only; full Agent probes additionally validate execution restrictions.')
    except BaseException:
        row.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        row['seconds']=time.time()-start;(root/'RESULT.json').write_text(json.dumps(row,indent=2))
if __name__=='__main__':asyncio.run(main())
