"""Six predeclared correction-source trials under one parent lease; no training."""
from concurrent.futures import ThreadPoolExecutor
import fcntl,hashlib,json,os,re,shutil,subprocess,time
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
B=P/'planning/bowl_correction_acquisition_v1_20261009'
def capture(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=40)
 return dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):(B/name).write_text(json.dumps(data,indent=2)+'\n')
def main():
 lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 B.mkdir(exist_ok=False);before=sha(P/'CURRENT_NATIVE_ENTRY.json')
 pre=dict(hcu=capture(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse']),
  containers=capture(['docker','ps','--format','{{.ID}} {{.Names}}']),holders=capture(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render*']))
 save('PREFLIGHT.json',pre)
 memory=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',pre['hcu']['stdout'])
 assert pre['hcu']['returncode']==0 and len(memory)==8 and all(float(v)==0 for _,v in memory)
 assert pre['containers']['returncode']==0 and not pre['containers']['stdout'].strip()
 identities=[]
 for pid in [int(x) for x in pre['holders']['stdout'].split()]:
  comm=(Path('/proc')/str(pid)/'comm').read_text().strip();exe=str((Path('/proc')/str(pid)/'exe').resolve(strict=True))
  identities.append(dict(pid=pid,comm=comm,exe=exe));assert comm=='hymgr' and exe=='/usr/local/hyhal/bin/hymgr'
 save('HOLDER_IDENTITIES.json',identities)
 worker=P/'scripts/run_bowl_correction_worker_v1.py'
 (B/'code').mkdir()
 for name in ['run_bowl_correction_acquisition_v1.py','run_bowl_correction_worker_v1.py','qualify_bowl_correction_v1.py','serve_bowl_correction_policy_v1.py','replay_stack_progress.py','stage_trace.py']:
  shutil.copy2(P/'scripts'/name,B/'code'/name)
 source=P/'runs/bowl_lora_source_adapted_v1_20261009/demo_0/RESULT.json'
 save('PARENT_GATE.json',dict(pid=os.getpid(),worker_sha256=sha(worker),source_result_sha256=sha(source),default_sha256=before,
  arms=['seed7','seed19'],gpus=[0,1],source='Adapted demo0 real executed trajectory',start_steps=[120,250,270],
  total_budget=400,suffix_budget_per_seed=[280,150,130],maximum_new_suffix_actions=1120,maximum_prefix_replay_actions=1280,
  training_steps=0,new_prompts=False,partial_state_restore=False,online_privileged_features=False,
  candidate_sampler_reset='Post-load seeded RNG restored once at each intervention; not the original trajectory continued RNG',
  successful_suffix_use='Candidate supervision after real original-goal success; not per-action optimality or transferable gain'))
 started=time.perf_counter()
 def run(arm):
  tick=time.perf_counter()
  with (B/(arm+'.log')).open('x') as f:
   try:r=subprocess.run(['python3',str(worker),arm],stdout=f,stderr=subprocess.STDOUT,timeout=1400,env=dict(os.environ,FASTRSI_CORRECTION_PARENT_PID=str(os.getpid())));exit_code=r.returncode
   except subprocess.TimeoutExpired:exit_code=124
  receipt=dict(arm=arm,exit_code=exit_code,host_seconds=time.perf_counter()-tick);save(arm+'_HOST.json',receipt);return receipt
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['seed7','seed19']))
 save('EXECUTION.json',dict(results=results,parallel_host_seconds=time.perf_counter()-started))
 assert all(x['exit_code']==0 for x in results),results
 cases=[];cleanup=[]
 for arm in ['seed7','seed19']:
  root=P/'runs'/('bowl_correction_'+arm+'_v1_20261009')
  final=json.loads((root/'FINAL_STATE.json').read_text());assert final['default_unchanged'];cleanup+=final['containers']
  cases+=json.loads((root/'AGGREGATE.json').read_text())['cases']
 assert all(x['intervention_inputs_exact'] for x in cases) and len(cases)==6
 assert all(not x['state']['Running'] and x['state']['Pid']==0 for x in cleanup)
 assert before==sha(P/'CURRENT_NATIVE_ENTRY.json')
 valid=[x for x in cases if x['policy_success']]
 save('AGGREGATE.json',dict(status='COMPLETED',cases=cases,validated_suffixes=len(valid),total_trials=6,
  available_state_count=3,states_with_validated_suffixes=sorted(set(x['case'] for x in valid)),
  new_suffix_actions=sum(x['suffix_actions'] for x in cases),prefix_replay_actions=sum(x['prefix_actions'] for x in cases),
  new_policy_calls=sum(x['policy_vla_calls'] for x in cases),training_steps=0,fast_rsi_gain_proven=False,
  default_unchanged=True,scope='Correction-source search only; all failures and acquisition cost retained'))
 save('FINAL_STATE.json',dict(containers=cleanup,default_sha256=before))
 print(json.dumps(dict(validated_suffixes=len(valid),total_trials=6,cases=cases)),flush=True)
if __name__=='__main__':main()
