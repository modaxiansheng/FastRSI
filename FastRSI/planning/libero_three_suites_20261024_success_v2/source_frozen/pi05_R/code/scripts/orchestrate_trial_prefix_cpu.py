"""CPU exact-prefix gate for subsequent real candidate validation."""
import fcntl,hashlib,json,time,traceback
import orchestrate_full_os as life
NAME='trial_prefix_cpu_v4_20261004'
def main():
 root=life.PROJECT;run=root/'runs'/NAME;lock=open(root/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.time();life.deadline=start+500;state=dict(status='RUNNING',jobs=[])
 assert not (run/'STATUS.json').exists()
 try:
  life.gate(run)
  for rel,h in json.loads((run/'MANIFEST.json').read_text()).items():assert hashlib.sha256((run/rel).read_bytes()).hexdigest()==h
  source=root/'runs/prefix_memory_v1_20261004/N/stream'
  for mode in ('full','reuse'):
   folder=run/mode;folder.mkdir()
   old=json.loads((source/'LAUNCH_target.json').read_text())['argv'];image=next(x for x in old if x.startswith('sha256:'));argv=old[:old.index(image)];argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-'+mode
   clean=[];i=0
   while i<len(argv):
    if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
    clean.append(argv[i]);i+=1
   base='/project/runs/'+NAME
   cid=life.launch('replay',clean+[image,'timeout','210','python',base+'/code/scripts/replay_trial_prefix.py','--stream',str(source).replace(str(root),'/project'),'--boundary','180','--mode',mode,'--output',base+'/'+mode+'/RESULT.json'],folder)
   life.wait_exit(cid,220);life.save(folder/'STOP_RECEIPT.json',life.stop_all())
   state['jobs'].append(dict(mode=mode,result=json.loads((folder/'RESULT.json').read_text())))
  a,b=[j['result'] for j in state['jobs']]
  assert a['status']==b['status']=='EXACT_BOUNDARY_REPLAY'
  assert [(r['step'],r['physics'],r['native_success']) for r in a['rows']]==[(r['step'],r['physics'],r['native_success']) for r in b['rows']]
  for x,y in zip(a['rows'],b['rows']):
   if x['step']>=180:assert x['policy_observation']==y['policy_observation']
  state['status']='COMPLETED'
 except BaseException:state.update(status='FAILED',error=traceback.format_exc());raise
 finally:
  life.save(run/'STOP_RECEIPT.json',life.stop_all());state.update(seconds=time.time()-start);life.save(run/'STATUS.json',state);print(json.dumps({k:v for k,v in state.items() if k!='jobs'}),flush=True)
if __name__=='__main__':main()
