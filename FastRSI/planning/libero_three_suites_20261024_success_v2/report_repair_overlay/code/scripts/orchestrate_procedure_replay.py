"""CPU exact replays only after complete policy execution; no online oracle."""
import fcntl,json,time,traceback
from pathlib import Path
import orchestrate_full_os as life
NAME='procedure_probe_v1_20261004'

def main():
 run=life.PROJECT/'runs'/NAME
 lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((run/'STATUS.json').read_text())['status']=='COMPLETED'
 out=life.PROJECT/'planning'/NAME/'replay_v1';out.mkdir(parents=True,exist_ok=False)
 life.deadline=time.time()+360;start=time.time();workers=[];status={'status':'RUNNING'}
 try:
  life.gate(out)
  for task,init in [(2,9),(9,15)]:
   for arm in ('N','F'):
    source=run/f'task{task}'/arm/'stream';folder=out/f'task{task}'/arm;folder.mkdir(parents=True)
    old=json.loads((source/'LAUNCH_target.json').read_text())['argv'];image=next(x for x in old if x.startswith('sha256:'));argv=old[:old.index(image)]
    argv[argv.index('--name')+1]=f'yuchen-fastrsi-procedure-replay-t{task}-{arm.lower()}'
    cleaned=[];i=0
    while i<len(argv):
     if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
     cleaned.append(argv[i]);i+=1
    pp=lambda p:str(p).replace(str(life.PROJECT),'/project')
    command=cleaned+[image,'timeout','300','python','/project/scripts/replay_procedure_facts.py',
       '--trace',pp(source/f'stage_trace/net_t{task}_i{init}/TRACE.jsonl'),
       '--output',pp(folder/'REPLAY.json'),'--task',str(task),'--init',str(init)]
    workers.append(life.launch('replay',command,folder))
  for cid in workers:life.wait_exit(cid,310)
  status['status']='COMPLETED'
 except BaseException:status.update(status='FAILED',error=traceback.format_exc())
 finally:
  life.save(out/'STOP_RECEIPT.json',life.stop_all());status.update(seconds=time.time()-start)
  life.save(out/'STATUS.json',status);print(json.dumps(status))
 if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
