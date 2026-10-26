"""Four isolated complete Agent workers; exact resources stopped by owners."""
import fcntl,hashlib,json,signal,subprocess,time,traceback
import orchestrate_full_os as life
NAME='procedure_probe_v1_20261004'

def main():
 run=life.PROJECT/'runs'/NAME
 lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (run/'STATUS.json').exists()
 start=time.time();life.deadline=start+1600;life.CAMPAIGN=run/'task2';life.BASE='/project/runs/'+NAME+'/task2'
 workers=[];logs=[];status={'status':'RUNNING','started_at':start}
 def stop(n,f):raise KeyboardInterrupt(str(n))
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  for task in ('task2','task9'):
   for group in (run/task,run/task/'N',run/task/'F'):
    for rel,h in json.loads((group/'CODE_MANIFEST.json').read_text()).items():
     assert hashlib.sha256((group/'code'/rel).read_bytes()).hexdigest()==h,rel
  life.gate(run)
  cpu=run/'cpu';cpu.mkdir()
  cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_procedure_probe test_planning_entry_binding test_task_snapshot_worker test_resident_worker -v'
  cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-procedure-v1-cpu',cpu)+[life.IMAGE,'timeout','120','bash','-lc',cmd],cpu)
  life.wait_exit(cid,130);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu']='PASSED';life.gate(run)
  for task in ('task2','task9'):
   campaign=run/task;protocol=json.loads((campaign/'PROTOCOL.json').read_text());protocol['absolute_deadline']=life.deadline;life.save(campaign/'PROTOCOL.json',protocol)
   for arm in ('N','F'):
    log=(campaign/arm/'controller.log').open('x');logs.append(log)
    p=subprocess.Popen(['python3',str(campaign/'code/scripts/run_placement_worker.py'),NAME+'/'+task,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append((task,arm,p))
  life.save(run/'DISPATCH.json',{'pids':{t+'/'+a:p.pid for t,a,p in workers},'deadline':life.deadline});life.save(run/'STATUS.json',status)
  while any(p.poll() is None for _,_,p in workers):
   if time.time()>life.deadline:raise TimeoutError('Bounded experiment expired')
   time.sleep(2)
  status['workers']={t+'/'+a:json.loads((run/t/a/'STATUS.json').read_text()) for t,a,_ in workers}
  assert all(p.returncode==0 and status['workers'][t+'/'+a]['status']=='COMPLETED' for t,a,p in workers)
  status['status']='COMPLETED'
 except BaseException:status.update(status='FAILED',error=traceback.format_exc())
 finally:
  for t,a,p in workers:
   if p.poll() is None:
    p.send_signal(signal.SIGTERM)
    try:p.wait(timeout=45)
    except subprocess.TimeoutExpired:status.setdefault('cleanup_attention',[]).append(t+'/'+a)
  life.save(run/'STOP_RECEIPT.json',life.stop_all())
  for log in logs:log.close()
  status.update(seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
 if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
