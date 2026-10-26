"""One native batch-end rollover after result consumption, then fresh recall."""
import asyncio
import json
import os
import shutil
from pathlib import Path
from run_native_baseline import measured_provider
from fastrsi.native_binding import bind_agent,finish_agent,hashes

async def main():
    root=Path(os.environ['FASTRSI_NATIVE_RUN']);case=json.loads((root.parent/'ROSTER.json').read_text())[0]
    source=Path(os.environ.get('FASTRSI_TERMINAL_SOURCE',str(root/case['id']/'accepted_experience')))
    before=hashes(source)
    out=root/'terminal_memory';out.mkdir()
    workspace=out/'workspace';shutil.copytree(source,workspace)
    provider=measured_provider(out,'memory',4096);provider._client.timeout=600.0
    agent=bind_agent(workspace,provider,out/'EVENTS.jsonl')
    try:
        key=case['conversation_key'];messages=agent.sessions.get_or_create(key).messages
        (out/'SOURCE_CONVERSATION.json').write_text(json.dumps(messages,indent=2))
        assert any(m.get('role')=='tool' and 'num_steps' in str(m.get('content')) for m in messages)
        result=await agent.process_direct('/new',session_key=key)
        assert result=='New session started.',result
        memory=(workspace/'memory/MEMORY.md').read_text().strip()
        assert memory and case['id'] in memory
        reply=await agent.process_direct('What is the latest recorded state and result of Session '+case['id']+'? Answer from the existing memory; do not start or modify any task.',session_key='cli:next')
        requests=[json.loads(p.read_text()) for p in out.glob('memory_call_*/REQUEST.json')]
        assert any(memory in str(m.get('content')) for r in requests for m in r.get('messages',[]))
        value={'status':'PASSED','trigger':'one batch-end native /new','reply':reply,'memory':memory,
            'actual_memory_read':True,'source_includes_terminal_artifact':True,
            'source':str(source),'source_hashes_before':before,'source_unchanged_after':hashes(source)==before,
            'scope':'Lifecycle and recall, not a learned policy or robot improvement'}
        (out/'RESULT.json').write_text(json.dumps(value,indent=2))
    finally:
        await finish_agent(agent,out/'EVENTS.jsonl');await provider._client.close()
        assert hashes(source)==before,'Native rollover changed its accepted source'

if __name__=='__main__':asyncio.run(main())
