"""Read actual actions after the run; never expose diagnostic state online."""
import fcntl,hashlib,json,signal,time,traceback
import orchestrate_full_os as life
NAME='m2_transfer_v1_20261005'
def main():
 run=life.PROJECT/'runs'/NAME
 lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((run/'STATUS.json').read_text())['status']=='COMPLETED'
 out=life.PROJECT/'planning'/NAME/'replay_v1';out.mkdir(parents=True,exist_ok=False)
 start=time.time();life.deadline=start+360;status={'status':'RUNNING','started_at':start};workers=[]
 def interrupt(n,f):raise KeyboardInterrupt('Interrupted '+str(n))
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 try:
  life.gate(out)
  code=out/'code';code.mkdir()
  for name in ['replay_stack_progress.py','stage_trace.py']:
   (code/name).write_bytes((life.PROJECT/'scripts'/name).read_bytes())
  life.save(out/'CODE_MANIFEST.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in code.iterdir()})
  for tid in [2,3]:
   source=run/f'init{tid}/F/stream';folder=out/f'init{tid}';folder.mkdir()
   old=json.loads((source/'LAUNCH_target.json').read_text())['argv'];image=next(x for x in old if x.startswith('sha256:'));argv=old[:old.index(image)]
   argv[argv.index('--name')+1]=f'yuchen-fastrsi-m2-replay-i{tid}'
   clean=[];i=0
   while i<len(argv):
    if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
    clean.append(argv[i]);i+=1
   pp=lambda p:str(p).replace(str(life.PROJECT),'/project')
   workers.append(life.launch('replay',clean+[image,'timeout','300','python',pp(code/'replay_stack_progress.py'),
      '--stream',pp(source),'--output',pp(folder/'REPLAY.json')],folder))
  for cid in workers:life.wait_exit(cid,310)
  status['status']='COMPLETED'
 except BaseException:status.update(status='FAILED',error=traceback.format_exc())
 finally:
  life.save(out/'STOP_RECEIPT.json',life.stop_all());status.update(seconds=time.time()-start,finished_at=time.time())
  life.save(out/'STATUS.json',status);print(json.dumps(status))
 if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
