"""Automatically convert successful native feedback into isolated skill candidates."""
import fcntl,hashlib,json,shlex,time,traceback
import orchestrate_full_os as life
NAME='source_bank_compile_v1_20261005'
def main():
 lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=life.PROJECT/'runs'/NAME;out.mkdir(exist_ok=False);code=out/'code';(code/'scripts').mkdir(parents=True);(code/'src/fastrsi').mkdir(parents=True)
 for name in ['extract_control_targets.py','compile_experience_skill.py','replay_stack_progress.py','stage_trace.py']:(code/'scripts'/name).write_bytes((life.PROJECT/'scripts'/name).read_bytes())
 for name in ['experience_compiler.py','retarget_geometry.py']:(code/'src/fastrsi'/name).write_bytes((life.PROJECT/'src/fastrsi'/name).read_bytes())
 life.save(out/'CODE_MANIFEST.json',{str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*') if p.is_file()})
 sources=[];skipped=[]
 for host in ['187','188']:
  run=life.PROJECT/f'runs/source_bank_dev_{host}_v1_20261005';assert json.loads((run/'STATUS.json').read_text())['status']=='COMPLETED'
  for j in json.loads((run/'REGISTRATION.json').read_text())['jobs']:
   if j['reuse']:continue
   stream=run/j['child']/'F/stream';result=json.loads((stream/'RESULT.json').read_text())['cases'][0]
   trace=stream/'stage_trace'/result['case']['id']/'TRACE.jsonl';terminal=json.loads(trace.read_text().splitlines()[-1])
   if result['chain_status']=='PASSED' and result['environment_success'] is True and terminal['native_success'] is True:
    sources.append(dict(init=j['init'],stream=str(stream),result_sha256=hashlib.sha256((stream/'RESULT.json').read_bytes()).hexdigest()))
   else:skipped.append(dict(init=j['init'],reason='No completed successful root source; retain failure evidence.'))
 life.save(out/'REGISTRATION.json',dict(sources=sources,skipped=skipped,rule='All successful native root episodes from the registered development cohort, not selected by speed. Compile only robot control targets after exact image replay. Candidate creation is not transfer approval.'))
 start=time.time();life.deadline=start+500;state=dict(status='RUNNING',started_at=start);workers=[];pp=lambda p:str(p).replace(str(life.PROJECT),'/project')
 try:
  life.gate(out);life.save(out/'STATUS.json',state)
  for row in sources:
   from pathlib import Path
   stream=Path(row['stream']);folder=out/f"source{row['init']}";folder.mkdir()
   old=json.loads((stream/'LAUNCH_target.json').read_text())['argv'];image=next(x for x in old if x.startswith('sha256:'));argv=old[:old.index(image)];argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-'+str(row['init']);clean=[];i=0
   while i<len(argv):
    if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
    clean.append(argv[i]);i+=1
   first=['python',pp(code/'scripts/extract_control_targets.py'),'--stream',pp(stream),'--output',pp(folder/'CONTROLS.json')]
   second=['python',pp(code/'scripts/compile_experience_skill.py'),'--stream',pp(stream),'--replay',pp(folder/'CONTROLS.json'),'--output',pp(folder/'GOAL_SKILL.json')]
   workers.append(life.launch('compile'+str(row['init']),clean+[image,'timeout','400','bash','-c',shlex.join(first)+' && '+shlex.join(second)],folder))
  for cid in workers:life.wait_exit(cid,410)
  state.update(status='COMPLETED',compiled=len(sources),failed_sources_skipped=len(skipped))
 except BaseException:state.update(status='FAILED',error=traceback.format_exc())
 finally:
  life.save(out/'STOP_RECEIPT.json',life.stop_all());state.update(seconds=time.time()-start,finished_at=time.time());life.save(out/'STATUS.json',state);print(json.dumps(state),flush=True)
 if state['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
