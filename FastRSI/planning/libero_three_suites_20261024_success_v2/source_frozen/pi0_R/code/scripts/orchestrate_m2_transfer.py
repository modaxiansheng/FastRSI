"""Four isolated workers on eight preflight-idle cards, one owned lock."""
import fcntl,hashlib,json,signal,socket,subprocess,time,traceback
import orchestrate_full_os as life
NAME='m2_transfer_v1_20261005'
def main():
 run=life.PROJECT/'runs'/NAME;lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (run/'STATUS.json').exists()
 start=time.time();life.deadline=start+1800;status={'status':'RUNNING','started_at':start};workers=[];logs=[]
 def interrupt(n,f):raise KeyboardInterrupt('Interrupted '+str(n))
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 try:
  reg=json.loads((run/'REGISTRATION.json').read_text());plan=json.loads((run/'DISPATCH_PLAN.json').read_text())
  assert hashlib.sha256((run/'REGISTRATION.json').read_bytes()).hexdigest()==plan['registration_sha256']
  for child in ('init2','init3'):
   folder=run/child
   for group in (folder,folder/'N',folder/'F'):
    for path,digest in json.loads((group/'CODE_MANIFEST.json').read_text()).items():assert hashlib.sha256((group/'code'/path).read_bytes()).hexdigest()==digest,path
   assert hashlib.sha256((folder/'GOAL_SKILL.json').read_bytes()).hexdigest()==reg['candidate_sha256']
   assert json.loads((folder/'PLAN.json').read_text())['cases'][0]['init_state_id']==int(child[-1])
   protocol=json.loads((folder/'PROTOCOL.json').read_text());protocol['absolute_deadline']=life.deadline
   life.save(folder/'EXECUTION_PROTOCOL.json',protocol);life.save(folder/'PROTOCOL.json',protocol)
  life.gate(run)
  for base in (24500,24600,24700,24800):
   for add in (0,2,10,20):
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',base+add))!=0
  status['artifact_preflight']='PASSED';status['runtime']='Frozen M1 snapshot; no retraining'
  for child in ('init2','init3'):
   for arm in ('N','F'):
    log=(run/child/arm/'controller.log').open('x');logs.append(log)
    p=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),NAME+'/'+child,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append((child,arm,p))
  life.save(run/'DISPATCH.json',dict(pids={c+'/'+a:p.pid for c,a,p in workers}));life.save(run/'STATUS.json',status)
  while any(p.poll() is None for _,_,p in workers):
   if time.time()>life.deadline:raise TimeoutError('Registered campaign deadline')
   time.sleep(2)
  status['workers']={c+'/'+a:json.loads((run/c/a/'STATUS.json').read_text()) for c,a,p in workers}
  for child in ('init2','init3'):
   rows={a:status['workers'][child+'/'+a] for a in ('N','F')}
   ok=all(rows[a]['status']=='COMPLETED' and p.returncode==0 for c,a,p in workers if c==child)
   life.save(run/child/'STATUS.json',dict(status='COMPLETED' if ok else 'FAILED',workers=rows,seconds=max(r.get('finished_at',time.time()) for r in rows.values())-min(r['started_at'] for r in rows.values()),scope='Pair concurrent elapsed; parent batch includes dispatch and shared preflight.'))
  assert all(p.returncode==0 and status['workers'][c+'/'+a]['status']=='COMPLETED' for c,a,p in workers)
  status['status']='COMPLETED'
 except BaseException:status.update(status='FAILED',error=traceback.format_exc())
 finally:
  for c,a,p in workers:
   if p.poll() is None:
    p.send_signal(signal.SIGTERM)
    try:p.wait(timeout=45)
    except subprocess.TimeoutExpired:status.setdefault('cleanup_attention',[]).append(c+'/'+a)
  for log in logs:log.close()
  life.save(run/'STOP_RECEIPT.json',life.stop_all());status.update(seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
 if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
