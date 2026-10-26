"""Independent CPU simulator probes of a registered local skill repair."""
import fcntl,hashlib,json,time,traceback
import orchestrate_full_os as life
NAME='phase_binding_probe_v1_20261005'
def main():
 lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=life.PROJECT/'runs'/NAME;assert not (out/'STATUS.json').exists()
 for rel,sha in json.loads((out/'INPUT_MANIFEST.json').read_text()).items():assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==sha
 reg=json.loads((out/'REGISTRATION.json').read_text());start=time.time();life.deadline=start+reg['max_seconds'];state=dict(status='RUNNING',started_at=start)
 pp=lambda p:str(p).replace(str(life.PROJECT),'/project')
 try:
  life.gate(out);life.save(out/'STATUS.json',state);workers=[]
  source=life.PROJECT/'runs/phase_choice_v1_20261005/init5_h262/F/stream/LAUNCH_target.json'
  old=json.loads(source.read_text())['argv'];image=next(x for x in old if x.startswith('sha256:'))
  for job in reg['jobs']:
   folder=out/job['name'];argv=old[:old.index(image)];argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-'+job['name'];clean=[];i=0
   while i<len(argv):
    if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
    clean.append(argv[i]);i+=1
   workers.append(life.launch(job['name'],clean+[image,'timeout','220','python',pp(out/'code/scripts/probe.py'),'--plan',pp(folder/'PLAN.json'),'--arm',job['arm'],'--output',pp(folder/'RESULT.json')],folder))
  for cid in workers:life.wait_exit(cid,230)
  state['status']='COMPLETED'
 except BaseException:state.update(status='FAILED',error=traceback.format_exc())
 finally:
  life.save(out/'STOP_RECEIPT.json',life.stop_all());state.update(seconds=time.time()-start,finished_at=time.time());life.save(out/'STATUS.json',state);print(json.dumps(state),flush=True)
 if state['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
