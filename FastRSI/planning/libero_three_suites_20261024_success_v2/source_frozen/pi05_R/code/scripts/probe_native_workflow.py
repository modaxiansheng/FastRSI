"""Three fixed experience contexts, two native tool-read workflows; no robot."""
import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from run_native_baseline import setup_workspace,measured_provider,plan_session
from fastrsi.native_binding import bind_agent,finish_agent,hashes
from PhyAgentOS.runtime.schemas import SessionSpec

async def main():
    root=Path(os.environ['FASTRSI_NATIVE_RUN']);protocol=json.loads((root.parent/'PROTOCOL.json').read_text())
    rows=[];start=time.time()
    try:
        for item in protocol['cases']:
            folder=root/item['label'];folder.mkdir()
            accepted=Path(item['accepted_source']) if item['accepted_source'] else None
            case={**protocol['task'],'id':'workflow_t4_i30','mode':'evaluation','conversation_key':'cli:workflow',
                'workflow':item['workflow'],'require_discovery_reads':True,'planning_only':True}
            w,spec,old=setup_workspace(folder,case,accepted);before=hashes(w)
            provider=measured_provider(folder,'agent',2048);log=folder/'EVENTS.jsonl'
            agent=bind_agent(w,provider,log,evaluation=True);tick=time.time()
            row={'label':item['label'],'workflow':item['workflow'],'accepted_source':item['accepted_source']}
            try:
                reply,planned=await plan_session(agent,case,w,spec,old,log)
                actual=SessionSpec.model_validate(planned).model_dump(mode='json')
                expected=SessionSpec.model_validate(spec).model_dump(mode='json')
                assert actual==expected,'Validated plan changed the original task or execution contract'
                events=[json.loads(l) for l in log.read_text().splitlines()]
                calls=sorted(folder.glob('agent_call_*/RESPONSE.json'),key=lambda p:int(p.parent.name.rsplit('_',1)[1]))
                outputs=[json.loads(p.read_text()) for p in calls]
                row.update(status='PASSED',reply=reply,normalized_plan=actual,model_calls=len(calls),
                    tool_calls_per_model_call=[len(x.get('tool_calls',[])) for x in outputs],
                    input_tokens=sum(x.get('usage',{}).get('prompt_tokens',0) for x in outputs),
                    output_tokens=sum(x.get('usage',{}).get('completion_tokens',0) for x in outputs),
                    queue_rejections=sum(x.get('event')=='queue_not_accepted' for x in events))
            except Exception:
                row.update(status='FAILED',error=traceback.format_exc())
            finally:
                await finish_agent(agent,log);await provider._client.close()
                row.update(seconds=time.time()-tick,fixed_experience_unchanged=hashes(w)==before,robot_dispatched=False)
                calls=list(folder.glob('agent_call_*/RESPONSE.json'))
                row.setdefault('model_calls',len(calls))
                row.setdefault('tool_calls_per_model_call',[len(json.loads(p.read_text()).get('tool_calls',[])) for p in sorted(calls,key=lambda p:int(p.parent.name.rsplit('_',1)[1]))])
                assert row['fixed_experience_unchanged']
                (folder/'RESULT.json').write_text(json.dumps(row,indent=2))
            rows.append(row);(root/'PROGRESS.json').write_text(json.dumps(rows,indent=2))
        value={'status':'COMPLETED','all_cases_passed':all(x['status']=='PASSED' for x in rows),'rows':rows,'seconds':time.time()-start,
            'scope':'Planning-only fixed workflow comparison; identical effective Session and frozen experience, no robot-quality, full-task speedup or learned-routing claim.'}
        (root/'RESULT.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
    except BaseException:
        (root/'PARTIAL.json').write_text(json.dumps({'rows':rows,'seconds':time.time()-start},indent=2));raise

if __name__=='__main__':asyncio.run(main())
