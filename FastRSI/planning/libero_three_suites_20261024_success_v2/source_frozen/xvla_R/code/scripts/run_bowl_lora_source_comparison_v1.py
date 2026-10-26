"""Two device-matched source qualification arms run concurrently under one lease."""
from concurrent.futures import ThreadPoolExecutor
import fcntl,hashlib,json,os,re,shutil,subprocess,time
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
B=P/'planning/bowl_lora_source_comparison_v1_20261009'
def capture(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=40);return dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def save(name,data):(B/name).write_text(json.dumps(data,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);B.mkdir(exist_ok=False)
 before=sha(P/'CURRENT_NATIVE_ENTRY.json')
 pre=dict(hcu=capture(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse']),containers=capture(['docker','ps','--format','{{.ID}} {{.Names}}']),holders=capture(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render*']))
 save('PREFLIGHT.json',pre);memory=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',pre['hcu']['stdout'])
 assert pre['hcu']['returncode']==0 and len(memory)==8 and all(float(v)==0 for _,v in memory)
 assert pre['containers']['returncode']==0 and not pre['containers']['stdout'].strip()
 identities=[]
 for pid in [int(x) for x in pre['holders']['stdout'].split()]:
  comm=(Path('/proc')/str(pid)/'comm').read_text().strip();exe=str((Path('/proc')/str(pid)/'exe').resolve(strict=True));identities.append(dict(pid=pid,comm=comm,exe=exe));assert comm=='hymgr' and exe=='/usr/local/hyhal/bin/hymgr'
 save('HOLDER_IDENTITIES.json',identities)
 training=P/'runs/bowl_expert_lora400_v1_20261009/RESULT.json';tr=json.loads(training.read_text())
 assert tr['status']=='TRAINING_COMPLETED_PENDING_SOURCE_ROLLOUT' and tr['steps']==400
 worker=P/'scripts/run_bowl_lora_source_worker_v1.py'
 (B/'code').mkdir()
 for filename in ['run_bowl_lora_source_comparison_v1.py','run_bowl_lora_source_worker_v1.py','serve_bowl_lora_source_policy_v1.py','serve_bowl_source_policy_v1.py','qualify_bowl_source_policy_v1.py','bowl_expert_lora_v1.py']:
  shutil.copy2(P/'scripts'/filename,B/'code'/filename)
 save('PARENT_GATE.json',dict(pid=os.getpid(),worker_sha256=sha(worker),training_result_sha256=sha(training),arms=['original','adapted'],gpus=[0,1],cases=[0,7,25],default_sha256=before))
 started=time.perf_counter()
 def run(arm):
  argv=['python3',str(worker),arm];tick=time.perf_counter()
  with (B/(arm+'.log')).open('x') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=1300,env=dict(os.environ,FASTRSI_SOURCE_PARENT_PID=str(os.getpid())))
  receipt=dict(arm=arm,exit_code=r.returncode,host_seconds=time.perf_counter()-tick);save(arm+'_HOST.json',receipt);print(json.dumps(receipt),flush=True);return receipt
 with ThreadPoolExecutor(max_workers=2) as executor:results=list(executor.map(run,['original','adapted']))
 save('EXECUTION.json',dict(results=results,parallel_host_seconds=time.perf_counter()-started))
 assert all(x['exit_code']==0 for x in results),results
 roots={arm:P/'runs'/('bowl_lora_source_'+arm+'_v1_20261009') for arm in ('original','adapted')}
 loads={arm:json.loads((path/'POLICY_LOAD.json').read_text()) for arm,path in roots.items()}
 for field in ('cpu_rng_sha256','gpu_rng_sha256'):assert loads['original'][field]==loads['adapted'][field]
 arms={arm:json.loads((path/'AGGREGATE.json').read_text()) for arm,path in roots.items()}
 pairs=[]
 for index in range(3):
  a=arms['original']['cases'][index];b=arms['adapted']['cases'][index];assert a['case']==b['case'] and a['initial_inputs_exact'] and b['initial_inputs_exact']
  for camera in ('agentview_image','robot0_eye_in_hand_image'):
   assert sha(roots['original']/a['case']/('initial_'+camera+'.png'))==sha(roots['adapted']/b['case']/('initial_'+camera+'.png'))
  pairs.append(dict(case=a['case'],original=a,adapted=b,new_success=b['policy_success'] and not a['policy_success'],new_failure=a['policy_success'] and not b['policy_success']))
 totals={arm:dict(successes=sum(p[arm]['policy_success'] for p in pairs),actions=sum(p[arm]['policy_actions'] for p in pairs),vla_calls=sum(p[arm]['policy_vla_calls'] for p in pairs),episode_seconds=sum(p[arm]['policy_seconds'] for p in pairs)) for arm in ('original','adapted')}
 after=sha(P/'CURRENT_NATIVE_ENTRY.json');assert before==after
 eligible=totals['adapted']['successes']>=totals['original']['successes'] and not any(p['new_failure'] for p in pairs) and (totals['adapted']['actions']<totals['original']['actions'] or totals['adapted']['successes']>totals['original']['successes'])
 save('AGGREGATE.json',dict(pairs=pairs,totals=totals,source_qualification_passed=eligible,base_rng_equal=True,default_unchanged=True,default_promoted=False,heldout_generalization=False,complete_os_comparison=False,fast_rsi_gain_proven=False,
  training_host_seconds=json.loads((training.parent/'HOST_RESULT.json').read_text())['full_wall_seconds'],scope='Two trained source starts and one previously exposed validation start; native behavior qualification, not FastRSI gain'))
 save('ADMISSION.json',dict(status='SOURCE_QUALIFIED_NOT_PROMOTED' if eligible else 'REJECTED',default_promoted=False,next='Fresh contextual comparison required' if eligible else 'Do not expand this fixed candidate'))
 print(json.dumps(dict(totals=totals,source_qualification_passed=eligible)),flush=True)
if __name__=='__main__':main()
