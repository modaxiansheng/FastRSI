"""Keep native Agent and full-goal verification; alter only paid candidate inputs."""
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.request
from pathlib import Path
import run_native_baseline as native
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
from fastrsi.procedure_probe import bind, Controller
from cost_trace import emit, span


def teacher(spec, payload, previous, step, folder):
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=False)
    state={'goal':spec['goal'], 'current_step':step,
           'current_robot_state':[float(x) for x in payload['observation/state']],
           'facts':spec['facts'], 'allowed_values':['true','false','unknown'],
           'note':'Past and current images only. Gripper width does not by itself prove object release. Occlusion does not prove absence.'}
    parts=[{'type':'text','text':json.dumps(state)}]
    views=[]
    for when, record in [('previous_check',previous),('current',payload)]:
        if record is None:continue
        for view,key in [('front','observation/image'),('wrist','observation/wrist_image')]:
            b=io.BytesIO();Image.fromarray(record[key]).save(b,format='PNG');raw=b.getvalue()
            name=when+'_'+view+'.png';(folder/name).write_bytes(raw)
            views.append({'file':name,'sha256':hashlib.sha256(raw).hexdigest()})
            parts += [{'type':'text','text':when+' '+view}, {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}}]
    schema={'type':'object','properties':{k:{'type':'string','enum':['true','false','unknown']} for k in spec['facts']},'required':spec['facts'],'additionalProperties':False}
    request={'model':os.environ.get('FASTRSI_AGENT_MODEL','qwen3vl8b'),'temperature':0,'max_tokens':256,
      'messages':[{'role':'system','content':'Inspect the supplied robot images. Report current observable physical facts using report_facts. Treat uncertain, occluded or ambiguous conditions as unknown. For mug_released, require evidence that the mug is no longer held; mug_inside alone does not establish this. For stove_on, use visible switch/indicator evidence. Do not infer completion from the goal or elapsed time. This is an unvalidated development proposal.'}, {'role':'user','content':parts}],
      'tools':[{'type':'function','function':{'name':'report_facts','description':'Report current observed facts, not advice or a success verdict.','parameters':schema}}]}
    (folder/'INPUT.json').write_text(json.dumps({'state':state,'images':views},indent=2))
    raw=json.dumps(request).encode();(folder/'REQUEST.json').write_bytes(raw)
    start=time.perf_counter();facts={k:'unknown' for k in spec['facts']};error=None;result=None
    try:
        req=urllib.request.Request('http://127.0.0.1:49510/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=45) as response:result=json.load(response)
        (folder/'RESPONSE.json').write_text(json.dumps(result,indent=2))
        choice=result['choices'][0];calls=choice['message'].get('tool_calls') or []
        if choice['finish_reason'] not in ('stop','tool_calls') or len(calls)!=1 or calls[0]['function']['name']!='report_facts':raise ValueError('Malformed or incomplete teacher response')
        value=json.loads(calls[0]['function']['arguments'])
        from fastrsi.procedure_probe import choose
        choose(spec['task_id'],value);facts=value
    except Exception as e:error=type(e).__name__+': '+str(e)
    receipt={'step':step,'facts':facts,'seconds':time.perf_counter()-start,'error':error,'label_status':'TEACHER_PROPOSAL_NOT_GROUND_TRUTH','usage':(result or {}).get('usage')}
    (folder/'RESULT.json').write_text(json.dumps(receipt,indent=2))
    return receipt


original_case=native.run_case
async def run_case(case,accepted):
    probe=case['procedure_probe'];spec=bind(case,Path(probe['artifact']).read_bytes(),probe['sha256'])
    controller=Controller(spec);original=OpenPIPi05Adapter.to_policy_input;previous=None
    def wrapped(adapter,observation,context):
        nonlocal previous
        payload=original(adapter,observation,context)
        if context.get('session_id')!=case['id'] or context.get('task_description')!=spec['goal']:return payload
        match=re.fullmatch(r'libero_obs_(\d+)',str(observation.get('observation_id','')))
        if not match:raise ValueError('Missing original observation identity')
        step=int(match.group(1));choice='full_goal';facts=None
        if probe['enabled']:
            if controller.needs_check(step):
                with span('procedure_teacher',case=case['id'],step=step):
                    record=teacher(spec,payload,previous,step,native.ROOT/case['id']/'procedure_calls'/str(step))
                facts=record['facts'];controller.update(step,facts)
                previous={k:v.copy() if hasattr(v,'copy') else v for k,v in payload.items()}
                emit({'event':'procedure_teacher_result','case':case['id'],**record})
            prompt,choice=controller.prompt(step)
        else:prompt=spec['goal']
        emit({'event':'procedure_choice','case':case['id'],'session_id':context['session_id'],'step':step,
              'procedure':probe['procedure_id'],'choice':choice,'prompt':prompt,'facts':facts,
              'original_goal':spec['goal'],'artifact_sha256':probe['sha256'],'teacher_calls_so_far':len(controller.done)})
        return {**payload,'prompt':prompt}
    OpenPIPi05Adapter.to_policy_input=wrapped
    try:return await original_case(case,accepted)
    finally:OpenPIPi05Adapter.to_policy_input=original

native.run_case=run_case
if __name__=='__main__':asyncio.run(native.main())
