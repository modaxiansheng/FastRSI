"""Finalize completed trials without relaunching policy, training, or robot jobs."""
from pathlib import Path
import hashlib,json,subprocess
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
ENTRY='a3b27c5762d82656a07c4a8d65a454a42f5a64295ba3ebd104cc7fbe04244837'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def save(p,v):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 assert sha(P/'CURRENT_NATIVE_ENTRY.json')==ENTRY
 parent=P/'planning/bowl_lora_source_comparison_v1_20261009'
 save(parent/'RECOVERY.json',dict(completed_arm_exit_codes=[json.loads((parent/(a+'_HOST.json')).read_text())['exit_code'] for a in ('original','adapted')],analysis_only=True,repeated_executions=0,reason='SSH stdout connection ended after both arms completed; aggregates reconstructed from durable completed receipts',recovery_script_sha256=sha(P/'scripts/analyze_bowl_lora_source_comparison_v1.py')))
 phase=P/'runs/bowl_phase_lora_source_v1_20261009'
 a=json.loads((phase/'AGGREGATE.json').read_text())
 original=json.loads((parent/'AGGREGATE.json').read_text())['totals']['original']
 totals=dict(successes=a['policy_successes'],actions=a['policy_actions'],vla_calls=sum(x['policy_vla_calls'] for x in a['cases']),episode_seconds=sum(x['policy_seconds'] for x in a['cases']))
 save(phase/'ADMISSION.json',dict(status='REJECTED',totals=totals,reference=original,default_promoted=False,training_steps=0,gate_is_learned=False,next='Seal this fixed phase rule; no threshold search'))
 roots=[P/'runs/bowl_expert_lora400_v1_20261009',P/'runs/bowl_lora_source_original_v1_20261009',P/'runs/bowl_lora_source_adapted_v1_20261009',P/'runs/bowl_lora_source_replay_v1_20261009',phase,parent]
 containers=[]
 for root in roots:
  for f in root.glob('*CONTAINER.json'):
   data=json.loads(f.read_text()); rows=data if isinstance(data,list) else [data]
   for row in rows:
    if 'Name' in row:containers.append(row['Name'].lstrip('/'))
 # Include records whose format stores just State, using names from persisted launch argv.
 for root in roots[:5]:
  for f in root.glob('*LAUNCH.json'):
   cmd=json.loads(f.read_text())
   if isinstance(cmd,dict):cmd=cmd.get('argv',[])
   if '--name' in cmd:containers.append(cmd[cmd.index('--name')+1])
 containers=sorted(set(containers));states=[]
 for name in containers:
  obj=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
  state=obj['State'];assert not state['Running'] and state['Pid']==0
  states.append(dict(name=name,status=state['Status'],pid=state['Pid'],exit_code=state['ExitCode']))
 report=P/'planning/bowl_lora_source_report_v1_20261009';report.mkdir(exist_ok=True)
 save(report/'FINAL_READBACK.json',dict(containers=states,owned_count=len(states),default_sha256=ENTRY,default_unchanged=True,new_training_steps=400,new_policy_episodes=9,offline_exact_replays=2,cyclic_decision_distillation=False,phase_totals=totals))
 for root in roots:
  records=[]
  for f in sorted(root.rglob('*')):
   rel=f.relative_to(root)
   if f.is_file() and not any(x in ('tokenizer_cache','__pycache__') for x in rel.parts) and f.name!='COPY_MANIFEST.json':
    records.append(dict(path=str(rel),bytes=f.stat().st_size,sha256=sha(f)))
  save(root/'COPY_MANIFEST.json',dict(root=str(root),files=records,count=len(records),bytes=sum(x['bytes'] for x in records),excluded=['tokenizer_cache','__pycache__'],shared_frozen_dependencies_retained=True))
 print(json.dumps(dict(containers=len(states),phase_totals=totals,manifest_counts=[json.loads((r/'COPY_MANIFEST.json').read_text())['count'] for r in roots])))
if __name__=='__main__':main()
