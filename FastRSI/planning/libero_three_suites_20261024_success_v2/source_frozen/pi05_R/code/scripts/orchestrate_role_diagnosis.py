"""Parallel replay of already paid trajectories; diagnostic state never goes online."""
import fcntl,hashlib,json,time,traceback
import orchestrate_full_os as life
NAME='role_diagnosis_v1_20261005'
JOBS={'role5':'role_procedure_v1_20261005/init5/F/stream','role1':'role_procedure_v1_20261005/init1/F/stream'}
def main():
 lock=open(life.PROJECT/'runs/role_diagnosis_188.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=life.PROJECT/'runs'/NAME;out.mkdir(exist_ok=False);code=out/'code';code.mkdir()
 for name in ('replay_stack_progress.py','stage_trace.py'):(code/name).write_bytes((life.PROJECT/'scripts'/name).read_bytes())
 life.save(out/'CODE_MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.iterdir()})
 life.save(out/'REGISTRATION.json',dict(jobs=JOBS,scope='Exact replay of old actions, no new candidate or model calls. Privileged diagnostics isolated from online inputs.'))
 start=time.time();life.deadline=start+400;state=dict(status='RUNNING',started_at=start);workers=[]
 try:
  life.gate(out);life.save(out/'STATUS.json',state)
  for name,rel in JOBS.items():
   source=life.PROJECT/'runs'/rel;folder=out/name;folder.mkdir();old=json.loads((source/'LAUNCH_target.json').read_text())['argv'];image=next(x for x in old if x.startswith('sha256:'));argv=old[:old.index(image)];argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-'+name
   clean=[];i=0
   while i<len(argv):
    if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
    clean.append(argv[i]);i+=1
   pp=lambda p:str(p).replace(str(life.PROJECT),'/project')
   workers.append(life.launch(name,clean+[image,'timeout','300','python',pp(code/'replay_stack_progress.py'),'--stream',pp(source),'--output',pp(folder/'REPLAY.json')],folder))
  for cid in workers:life.wait_exit(cid,310)
  state['status']='COMPLETED'
 except BaseException:state.update(status='FAILED',error=traceback.format_exc())
 finally:
  life.save(out/'STOP_RECEIPT.json',life.stop_all());state.update(seconds=time.time()-start,finished_at=time.time());life.save(out/'STATUS.json',state);print(json.dumps(state),flush=True)
 if state['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
