"""New native Session reads the paid method state; dry planning only, no dispatch."""
import asyncio,hashlib,json,os,time,traceback
from pathlib import Path
from run_native_baseline import setup_workspace,measured_provider,plan_session
from fastrsi.native_binding import bind_agent,finish_agent,hashes
from fastrsi.instruction_memory import select

async def main():
    root=Path(os.environ['FASTRSI_NATIVE_RUN']);config=json.loads((root.parent/'TRANSFER.json').read_text());case=config['case'];folder=root/case['id'];folder.mkdir(parents=True)
    w,spec,old=setup_workspace(folder,case,Path(config['source']));before=hashes(w);p=measured_provider(folder,'agent',2048);log=folder/'AGENT_EVENTS.jsonl'
    agent=bind_agent(w,p,log,evaluation=True,read_files=True,max_iterations=10);tick=time.perf_counter();row={'status':'RUNNING'}
    try:
        reply,planned=await plan_session(agent,case,w,spec,old,log)
        assert 'maximum number of tool call iterations' not in reply
        request=json.loads((w/'REQUEST.json').read_text());assert planned['task_description']==request['instruction_decision']['instruction']
        reads=[e for e in map(json.loads,log.read_text().splitlines()) if e.get('event')=='tool' and e.get('name')=='read_file' and e.get('params',{}).get('path','').endswith('instruction-reuse/EXPERIENCE.json') and not e.get('result','').startswith('Error')]
        assert reads,'No real experience read'
        expected=json.loads((w/'skills/instruction-reuse/EXPERIENCE.json').read_text())
        parsed=[]
        for e in reads:
            numbered='\n'.join(line.split('| ',1)[1] for line in e['result'].splitlines() if '| ' in line and line.split('| ',1)[0].isdigit())
            try:
                if json.loads(numbered)==expected:parsed.append(e)
            except ValueError:pass
        assert parsed,'Experience read was incomplete'
        # Check the final rendered service prompt, not merely a copied artifact.
        prompts=list(root.glob('service_calls/*/MODEL_PROMPT.txt'))
        proof=[str(f.relative_to(root)) for f in prompts if any(e['result'] in f.read_text() for e in parsed)]
        assert proof,'Full experience absent from actual model prompt'
        row={'status':'PASSED','actual_experience_prompt_proofs':proof,'decision':request['instruction_decision'],'planned_instruction':planned['task_description'],'reply':reply,
             'experience_sha256':hashlib.sha256((w/'skills/instruction-reuse/EXPERIENCE.json').read_bytes()).hexdigest(),
             'scope':'New native Session really reads and applies learned-state choice. No robot execution, no quality or net-benefit claim.'}
    except BaseException:
        row.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        await finish_agent(agent,log);await p._client.close()
        unchanged=before==hashes(w);row.update(seconds=time.perf_counter()-tick,experience_unchanged=unchanged)
        if not unchanged:row.update(status='FAILED',isolation_error='Persistent state changed')
        (root/'RESULT.json').write_text(json.dumps(row,indent=2));print(json.dumps(row))
    assert unchanged,'Persistent evaluation state changed'
if __name__=='__main__':asyncio.run(main())
