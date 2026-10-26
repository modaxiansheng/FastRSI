"""Four independent two-card workers; preserve bounded lifecycle and exact cleanup."""
import fcntl,hashlib,json,signal,socket,subprocess,time,traceback
import orchestrate_full_os as life
NAME='role_procedure_v1_20261005'
def main():
 run=life.PROJECT/'runs'/NAME;lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (run/'STATUS.json').exists()
 reg=json.loads((run/'REGISTRATION.json').read_text());start=time.time();life.deadline=start+reg['max_seconds'];workers=[];logs=[]
 state=dict(status='RUNNING',started_at=start,registration_sha256=hashlib.sha256((run/'REGISTRATION.json').read_bytes()).hexdigest())
 def interrupt(n,f):raise KeyboardInterrupt(str(n))
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 try:
  life.gate(run)
  for job in reg['jobs']:
   folder=run/job['child'];group=folder/'F'
   for rel,d in json.loads((group/'CODE_MANIFEST.json').read_text()).items():assert hashlib.sha256((group/'code'/rel).read_bytes()).hexdigest()==d,rel
   p=json.loads((folder/'PROTOCOL.json').read_text());assert hashlib.sha256((folder/'PROCEDURE.json').read_bytes()).hexdigest()==p['procedure_probe']['sha256']
   p['absolute_deadline']=life.deadline;life.save(folder/'PROTOCOL.json',p)
   for port in job['ports']:
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
  for job in reg['jobs']:
   child=job['child'];log=(run/child/'F/controller.log').open('x');logs.append(log)
   worker=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),NAME+'/'+child,'F'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   workers.append((child,worker))
  life.save(run/'DISPATCH.json',dict(pids={c:p.pid for c,p in workers}));life.save(run/'STATUS.json',state)
  while any(p.poll() is None for _,p in workers):
   if time.time()>life.deadline:raise TimeoutError('Campaign deadline')
   time.sleep(2)
  state['workers']={c:json.loads((run/c/'F/STATUS.json').read_text()) for c,p in workers}
  assert all(p.returncode==0 and state['workers'][c]['status']=='COMPLETED' for c,p in workers)
  state['status']='COMPLETED'
 except BaseException:state.update(status='FAILED',error=traceback.format_exc())
 finally:
  for c,p in workers:
   if p.poll() is None:
    p.send_signal(signal.SIGTERM)
    try:p.wait(timeout=45)
    except subprocess.TimeoutExpired:state.setdefault('cleanup_attention',[]).append(c)
  for f in logs:f.close()
  state.update(seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',state);print(json.dumps(state),flush=True)
 if state['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
