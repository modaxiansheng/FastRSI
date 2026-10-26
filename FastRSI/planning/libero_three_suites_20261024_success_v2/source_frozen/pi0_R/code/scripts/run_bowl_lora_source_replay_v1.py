"""Two CPU-only exact-action replays of saved commands; no fresh intervention."""
from concurrent.futures import ThreadPoolExecutor
import fcntl,hashlib,json,shutil,subprocess,time
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI');A=P.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1';R=P/'runs/bowl_lora_source_replay_v1_20261009'
def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def main():
 lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip()
 R.mkdir(exist_ok=False);(R/'code').mkdir();before=hashlib.sha256((P/'CURRENT_NATIVE_ENTRY.json').read_bytes()).hexdigest()
 for f in ['replay_bowl_lora_source_v1.py','replay_stack_progress.py','stage_trace.py']:shutil.copy2(P/'scripts'/f,R/'code'/f)
 save(R/'PROTOCOL.json',dict(arms=['original','adapted'],case='demo_0',model_calls=0,training_steps=0,code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (R/'code').iterdir()},scope='Native privileged diagnostics only, exact saved commands; no new policy or labels'))
 started=time.perf_counter()
 def run(arm):
  name='yuchen-fastrsi-lora-source-replay-'+arm+'-v1-20261009'
  argv=['docker','run','--name',name,'--network','none','--cpus','4','--memory','12g','--pids-limit','1024','--shm-size','2g','--entrypoint','python',
   '-e','PYTHONPATH=/workspace/openpi/third_party/libero:/core:/check/code','-e','LIBERO_CONFIG_PATH=/libero-config','-e','MUJOCO_GL=egl','-e','LIBGL_ALWAYS_SOFTWARE=1',
   '-v',str(P)+':/project:ro','-v',str(R)+':/check','-v',str(A/'dependencies/openpi')+':/workspace/openpi:ro','-v',str(A/'dependencies/PhyAgentOS-core')+':/core:ro','-v',str(P/'configs/libero')+':/libero-config:ro',
   'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95','/check/code/replay_bowl_lora_source_v1.py','--arm',arm,'--output','/check/'+arm]
  save(R/(arm+'_LAUNCH.json'),argv);tick=time.perf_counter()
  with (R/(arm+'.log')).open('x') as f:
   try:result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=300);code=result.returncode
   except subprocess.TimeoutExpired:
    code=124;subprocess.run(['docker','stop','-t','10',name],capture_output=True,timeout=40)
  info=json.loads(subprocess.check_output(['docker','inspect',name]))[0];save(R/(arm+'_CONTAINER.json'),info)
  receipt=dict(exit_code=code,seconds=time.perf_counter()-tick,state=info['State']);save(R/(arm+'_HOST.json'),receipt)
  assert not info['State']['Running'] and info['State']['Pid']==0
  return dict(arm=arm,**receipt)
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['original','adapted']))
 save(R/'HOST_RESULT.json',dict(results=results,parallel_seconds=time.perf_counter()-started,default_unchanged=before==hashlib.sha256((P/'CURRENT_NATIVE_ENTRY.json').read_bytes()).hexdigest()))
 assert all(x['exit_code']==0 for x in results)
 print('Saved-command replay completed, no model calls',flush=True)
if __name__=='__main__':main()
