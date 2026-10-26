"""Real-model paired fixed-procedure diagnostic, then bounded full-chain evaluation."""
import asyncio,json,os,time,traceback
from pathlib import Path
from run_native_baseline import setup_workspace,measured_provider,plan_session,run_case,effective_execution_view
from fastrsi.native_binding import bind_agent,finish_agent,hashes
from fastrsi.workflow_reuse import compile_workflow,install_workflow
from fastrsi.experience_admission import validate_snapshot

ROOT=Path(os.environ['FASTRSI_NATIVE_RUN']);CAMP=ROOT.parent

def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,default=str))
def counts(folder):
    replies=[json.loads(p.read_text()) for p in folder.glob('agent_call_*/RESPONSE.json')]
    return {'agent_calls':len(replies),'prompt_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in replies),
        'completion_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in replies)}

async def main():
    protocol=json.loads((CAMP/'PROTOCOL.json').read_text());source=Path(protocol['source']);before_source=validate_snapshot(source)
    artifact=ROOT/'workflow';receipt=compile_workflow(source,Path(protocol['trace']),artifact)
    report={'status':'RUNNING','compilation':receipt,'planning':[],'full_chain':[],'scope':protocol['scope']}
    start=time.perf_counter()
    try:
        for item in protocol['planning_cases']:
            row={'label':item['label'],'arm':item['arm']};folder=ROOT/item['label'];folder.mkdir()
            case={**item['task'],'id':item['label'],'mode':'evaluation','require_discovery_reads':True,'planning_only':True}
            w,spec,old=setup_workspace(folder,case,source)
            if item['arm']=='B':
                install_workflow(w,artifact);case.update(workflow_artifact=str(artifact),workflow_macro=True)
            fixed=hashes(w);log=folder/'AGENT_EVENTS.jsonl';p=measured_provider(folder,'agent',2048)
            a=bind_agent(w,p,log,evaluation=True,read_files=True,workflow_macro=item['arm']=='B');tick=time.perf_counter()
            try:
                reply,planned=await plan_session(a,case,w,spec,old,log)
                assert 'maximum number of tool call iterations' not in reply, 'Native Agent did not complete its reply'
                assert effective_execution_view(planned)==effective_execution_view(spec),'Changed effective Session'
                row.update(status='PASSED',reply=reply)
            except Exception:row.update(status='FAILED',error=traceback.format_exc())
            finally:
                await finish_agent(a,log);await p._client.close()
                row.update(seconds=time.perf_counter()-tick,experience_unchanged=hashes(w)==fixed,**counts(folder))
                events=[json.loads(x) for x in log.read_text().splitlines()]
                row['workflow_expansions']=sum(x.get('event')=='workflow_expanded' for x in events)
                row['batch_calls']=sum(x.get('event')=='tool' and x.get('name')=='read_files' for x in events)
                row['actual_reads']=[x['params']['path'] for x in events if x.get('event')=='tool' and x.get('name')=='read_file']
                if item['arm']=='B':
                    req=json.loads((folder/'agent_call_1/REQUEST.json').read_text())
                    row['procedure_in_first_model_input']='Accepted discovery procedure' in json.dumps(req)
                (folder/'RESULT.json').write_text(json.dumps(row,indent=2));report['planning'].append(row);save('PROGRESS.json',report)
            assert row['status']=='PASSED' and row['experience_unchanged'], 'Planning or isolation failed'
        b=[r for r in report['planning'] if r['arm']=='B'];aa=[r for r in report['planning'] if r['arm']=='A']
        if not all(r['procedure_in_first_model_input'] and r['batch_calls']>0 for r in b):
            report['full_chain_gate']='SKIPPED: procedure did not change actual reads';return
        if sum(r['agent_calls'] for r in b)>=sum(r['agent_calls'] for r in aa):
            report['full_chain_gate']='SKIPPED: no model-call reduction in this paired diagnostic';return
        report['full_chain_gate']='PASSED';save('PROGRESS.json',report)
        # Full native Agent -> Watchdog -> policy -> verifier -> reporting, fixed experience.
        for item in protocol['full_cases']:
            case={**item['task'],'id':item['label'],'mode':'evaluation','require_discovery_reads':True,
                'require_clean_model_calls':True,'read_files_tool':True}
            if item['arm']=='B':case.update(workflow_artifact=str(artifact),workflow_macro=True)
            row,_=await run_case(case,source);row['arm']=item['arm'];row.update(counts(ROOT/item['label']))
            report['full_chain'].append(row);save('PROGRESS.json',report)
        report['status']='COMPLETED'
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        assert validate_snapshot(source)==before_source
        if report['status']=='RUNNING':report['status']='DIAGNOSTIC_ONLY'
        report['seconds']=time.perf_counter()-start;save('RESULT.json',report)
        print(json.dumps({'status':report['status'],'planning':len(report['planning']),'full_chain':len(report['full_chain']),'seconds':report['seconds']}),flush=True)

if __name__=='__main__':asyncio.run(main())
