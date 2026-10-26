"""Recheck a copied saved task with fixed learning and current-record read coverage."""
import asyncio,copy,hashlib,json,os,shutil,time
from pathlib import Path
from run_native_baseline import measured_provider
from fastrsi.native_binding import bind_agent,finish_agent,hashes
from fastrsi.terminal_results import consume_terminal
from fastrsi.experience_admission import tree_hashes

async def main():
    root=Path(os.environ['FASTRSI_NATIVE_RUN']);source=Path(os.environ['FASTRSI_REPORT_SOURCE'])
    old=json.loads((source/'RESULT.json').read_text());case=old['case'];original=source/'workspace'
    source_before=tree_hashes(original);w=root/'workspace';shutil.copytree(original,w)
    log=root/'AGENT_EVENTS.jsonl';provider=measured_provider(root,'agent',2048)
    agent=bind_agent(w,provider,log,evaluation=True,read_files=True,max_iterations=10)
    # Restore only this saved task's conversation into the evaluation's in-memory cache.
    history=original/'sessions'/('cli_'+case['id']+'.jsonl')
    saved=[json.loads(line) for line in history.read_text().splitlines()]
    session=agent.sessions.get_or_create('cli:'+case['id'])
    session.messages=copy.deepcopy([m for m in saved if 'role' in m])
    session.last_consolidated=next(m for m in saved if m.get('_type')=='metadata').get('last_consolidated',0)
    restored=len(session.messages);fixed_before=hashes(w);canonical_before=tree_hashes(w/'artifacts')
    session_hash=hashlib.sha256((w/'SESSIONS.md').read_bytes()).hexdigest()
    started=time.time();result={'status':'RUNNING','source_case':case['id'],'restored_task_messages':restored,
      'original_chain_receipt_unchanged':True,'robot_executions':0,'verification_calls':0,'learning_updates':False}
    try:
        result['report']=await consume_terminal(agent,case,w,log)
        result['status']='COMPLETED'
    finally:
        await finish_agent(agent,log);await provider._client.close()
        result.update(seconds=time.time()-started,fixed_experience_unchanged=hashes(w)==fixed_before,
          source_unchanged=tree_hashes(original)==source_before,
          canonical_unchanged=tree_hashes(w/'artifacts')==canonical_before and hashlib.sha256((w/'SESSIONS.md').read_bytes()).hexdigest()==session_hash,
          source_history_sha256=hashlib.sha256(history.read_bytes()).hexdigest())
        (root/'RESULT.json').write_text(json.dumps(result,indent=2))
    assert result['fixed_experience_unchanged'] and result['source_unchanged'] and result['canonical_unchanged']
    assert result['report']['read_coverage']['complete'] and result['report']['read_coverage']['fresh_report_write']

if __name__=='__main__':asyncio.run(main())
