"""Real-model protocol acceptance through the unchanged native AgentLoop."""
import asyncio
import copy
import shutil
import json
import os
import time
import traceback
import uuid
from pathlib import Path
from PhyAgentOS.providers.custom_provider import CustomProvider
from PhyAgentOS.providers.base import GenerationSettings
from fastrsi.native_binding import bind_agent, finish_agent

async def main():
    out=Path(os.environ['FASTRSI_NATIVE_RUN']);w=out/'protocol_workspace';w.mkdir()
    nonce='receipt_'+uuid.uuid4().hex[:12]
    (w/'index.txt').write_text('The requested receipt is in detail.txt. Read that file; do not guess its contents.')
    (w/'detail.txt').write_text('Verified receipt code: '+nonce)
    provider=CustomProvider(api_base='http://127.0.0.1:49510/v1',default_model=os.environ.get('FASTRSI_AGENT_MODEL','qwen25vl7b'))
    provider.generation=GenerationSettings(temperature=0,max_tokens=1024)
    log=out/'PROTOCOL_AGENT_EVENTS.jsonl';agent=bind_agent(w,provider,log,max_iterations=6)
    report={'status':'RUNNING','native_agent_class':type(agent).__module__+'.'+type(agent).__name__}
    started=time.time()
    try:
        index_reply=await agent.process_direct('Read index.txt with the file tool and tell me which file it references.',session_key='cli:protocol')
        first=await agent.process_direct('Now use the file tool to read the referenced file from your last reply. Report its exact receipt code; do not guess.',session_key='cli:protocol')
        second=await agent.process_direct('What exact receipt code did you just read? Answer from this conversation.',session_key='cli:protocol')
        events=[json.loads(x) for x in log.read_text().splitlines()]
        reads=[x for x in events if x.get('event')=='tool' and x.get('name')=='read_file']
        assert any(x['params'].get('path','').endswith('index.txt') for x in reads)
        assert any(x['params'].get('path','').endswith('detail.txt') for x in reads)
        assert nonce in first and nonce in second,(first,second)
        assert not (w/'memory/MEMORY.md').exists(),'Short probe should not trigger consolidation'
        report.update(status='PASSED',index_reply=index_reply,first=first,second=second,nonce=nonce,read_calls=len(reads),scope='three user turns: discover reference, read referenced file, recall nonce')
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        await finish_agent(agent,log)
        await provider._client.close()
        report.update(seconds=time.time()-started)
        (out/'PROTOCOL_AGENT_RESULT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    # Realistic original OS context, same Session planner as development; no robot execution.
    from run_native_baseline import setup_workspace, plan_session
    case={'id':'protocol_t4_i20','mode':'development','task_id':4,'init_state_id':20,
          'task_description':'pick up the black bowl from the top drawer of the wooden cabinet and place it on the plate'}
    prior=Path('/project/runs/native_baseline_v4_20261002/stream/dev1_t4_i20/accepted_experience')
    assert prior.is_dir(), 'Missing real accepted development workspace for history replay'
    inherited_goal=json.loads((prior/'REQUEST.json').read_text())['goal']
    probes=[('planning_probe',case,None),
            ('planning_history_probe',{**case,'id':'protocol_t4_i23','init_state_id':23,'task_description':inherited_goal},prior)]
    # Structural long-history fixture: copy three completed records, without
    # claiming these duplicates are new robot episodes or learned experience.
    from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block, write_yaml_block
    long_history=out/'long_history_fixture';shutil.copytree(prior,long_history)
    doc=read_yaml_block(long_history/'SESSIONS.md');original=copy.deepcopy(doc['sessions'][0])
    for i in [2,3]:
        item=copy.deepcopy(original);item['session_id']+='__structural_fixture_'+str(i);doc['sessions'].append(item)
    write_yaml_block(long_history/'SESSIONS.md','Runtime Sessions',doc)
    probes.append(('planning_long_history_probe',{**case,'id':'protocol_long_t4_i23','init_state_id':23,'task_description':inherited_goal},long_history))
    for label,probe_case,accepted in probes:
        folder=out/label;folder.mkdir()
        workspace,spec,previous=setup_workspace(folder,probe_case,accepted)
        log=folder/'AGENT_EVENTS.jsonl'
        provider=CustomProvider(api_base='http://127.0.0.1:49510/v1',default_model=os.environ.get('FASTRSI_AGENT_MODEL','qwen25vl7b'))
        provider.generation=GenerationSettings(temperature=0,max_tokens=2048)
        agent=bind_agent(workspace,provider,log,max_iterations=10)
        planning={'status':'RUNNING','scope':'real native context and Session creation, no environment execution', 'prior_workspace':str(accepted) if accepted else None}
        start=time.time()
        try:
            response,planned=await plan_session(agent,probe_case,workspace,spec,previous,log)
            planning.update(status='PASSED',response=response,planned_session=planned,prior_session_count=len(previous),history_characters=len((workspace/'SESSIONS.md').read_text()))
        except BaseException:
            planning.update(status='FAILED',error=traceback.format_exc());raise
        finally:
            await finish_agent(agent,log);await provider._client.close()
            planning['seconds']=time.time()-start
            (folder/'RESULT.json').write_text(json.dumps(planning,indent=2))

if __name__=='__main__':asyncio.run(main())
