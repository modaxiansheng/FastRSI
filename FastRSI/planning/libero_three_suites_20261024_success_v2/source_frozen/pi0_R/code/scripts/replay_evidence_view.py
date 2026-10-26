"""Native report-only comparison on immutable saved results. No robot/verifier replay."""
import asyncio,copy,hashlib,json,os,shutil,time
from pathlib import Path
from run_native_baseline import measured_provider
from fastrsi.native_binding import bind_agent,finish_agent,hashes
from fastrsi.terminal_results import consume_terminal
from fastrsi.experience_admission import tree_hashes

async def main():
 root=Path(os.environ['FASTRSI_NATIVE_RUN']);jobs=json.loads((root.parent/'PROTOCOL.json').read_text())['jobs'];rows=[]
 for job in jobs:
  folder=root/job['name'];folder.mkdir();source=Path(job['source']);saved=json.loads((source/'RESULT.json').read_text());case=dict(saved['case']);original=source/'workspace';source_before=tree_hashes(original)
  case.update(terminal_evidence_view=job['view'],terminal_prefetch=False,terminal_bind_report_destination=True)
  w=folder/'workspace';shutil.copytree(original,w)
  old_report=(w/saved['terminal_report']['report']).resolve()
  assert w.resolve() in old_report.parents and old_report.is_file()
  old_report.unlink()  # New isolated replay copy only; original is immutable.
  before=hashes(w);canonical=tree_hashes(w/'artifacts');sessions_sha=hashlib.sha256((w/'SESSIONS.md').read_bytes()).hexdigest()
  provider=measured_provider(folder,'agent',2048);log=folder/'AGENT_EVENTS.jsonl';agent=bind_agent(w,provider,log,evaluation=True,read_files=True,max_iterations=10)
  history=original/'sessions'/('cli_'+case['id']+'.jsonl');messages=[json.loads(l) for l in history.read_text().splitlines()]
  content=[m for m in messages if 'role' in m]
  # Remove the original report turn and its outputs from all arms; no answer leakage.
  stop=next(i for i,m in enumerate(content) if m['role']=='user' and str(m.get('content','')).startswith('Execution and its original verification have finished.'))
  prefix=copy.deepcopy(content[:stop]);session=agent.sessions.get_or_create('cli:'+case['id']);session.messages=prefix;session.last_consolidated=0
  result={'name':job['name'],'source':job['source'],'view':job['view'],'robot_executions':0,'verification_calls':0,'learning_updates':False,'old_report_removed_from_copy':True,'prefix_messages':len(prefix),'prefix_sha256':hashlib.sha256(json.dumps(prefix,sort_keys=True).encode()).hexdigest(),'status':'RUNNING'};start=time.perf_counter()
  try:
   result['report']=await consume_terminal(agent,case,w,log);result['status']='COMPLETED'
  finally:
   await finish_agent(agent,log);await provider._client.close()
   result.update(seconds=time.perf_counter()-start,fixed_experience_unchanged=hashes(w)==before,source_unchanged=tree_hashes(original)==source_before,canonical_unchanged=tree_hashes(w/'artifacts')==canonical and hashlib.sha256((w/'SESSIONS.md').read_bytes()).hexdigest()==sessions_sha)
   (folder/'RESULT.json').write_text(json.dumps(result,indent=2));rows.append(result);(root/'RESULT.json').write_text(json.dumps({'status':'RUNNING','rows':rows},indent=2))
  assert result['fixed_experience_unchanged'] and result['source_unchanged'] and result['canonical_unchanged']
 (root/'RESULT.json').write_text(json.dumps({'status':'COMPLETED','rows':rows},indent=2))
if __name__=='__main__':asyncio.run(main())
