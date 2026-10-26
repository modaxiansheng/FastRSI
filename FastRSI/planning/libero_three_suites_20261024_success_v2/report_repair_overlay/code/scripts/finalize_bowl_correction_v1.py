"""Finalize failed correction acquisition and data/reference provenance, no new jobs."""
from pathlib import Path
import hashlib,json,subprocess
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def save(p,obj):
 assert not p.exists(),str(p);p.write_text(json.dumps(obj,indent=2)+'\n')
def main():
 parent=P/'planning/bowl_correction_acquisition_v1_20261009';data=P/'planning/bowl_value_records_v2_20261009'
 agg=json.loads((parent/'AGGREGATE.json').read_text());assert agg['status']=='COMPLETED' and agg['validated_suffixes']==0
 rows=[json.loads(x) for x in (data/'TRANSITIONS.jsonl').read_text().splitlines()]
 train=[i for i,x in enumerate(rows) if x['initial_state_group'] in ('demo_0','demo_7')]
 validation=[i for i,x in enumerate(rows) if x['initial_state_group']=='demo_25']
 assert len(train)+len(validation)==len(rows) and len(train)==296 and len(validation)==120
 save(data/'SPLIT_POLICY.json',dict(status='NO_TRAINING_PERFORMED',train_initial_state_groups=['demo_0','demo_7'],
  diagnostic_validation_initial_state_groups=['demo_25'],training_row_indices=train,validation_row_indices=validation,
  source_validation_not_reclassified_as_training=True,heldout_generalization=False,
  rows_sha256=sha(data/'TRANSITIONS.jsonl'),rule='Keep source demo25 outside fitting; all three groups were exposed development/diagnostics, not final test'))
 out=P/'planning/bowl_correction_report_v1_20261009';out.mkdir(exist_ok=False)
 parent_final=json.loads((parent/'FINAL_STATE.json').read_text());containers=[]
 for row in parent_final['containers']:
  obj=json.loads(subprocess.check_output(['docker','inspect',row['name']]))[0]
  assert not obj['State']['Running'] and obj['State']['Pid']==0
  containers.append(dict(name=row['name'],state=obj['State']))
 name='yuchen-fastrsi-bowl-value-data-check-v1-20261009';obj=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
 assert not obj['State']['Running'] and obj['State']['Pid']==0
 containers.append(dict(name=name,state=obj['State']))
 before=json.loads((parent/'PARENT_GATE.json').read_text())['default_sha256'];assert sha(P/'CURRENT_NATIVE_ENTRY.json')==before
 cost=dict(parallel_host_seconds=json.loads((parent/'EXECUTION.json').read_text())['parallel_host_seconds'],
  full_episode_seconds=0.,suffix_seconds=0.,prefix_replay_seconds=0.,suffix_rpc_seconds=0.,suffix_environment_seconds=0.,
  suffix_evidence_seconds=0.,new_policy_calls=agg['new_policy_calls'],new_suffix_actions=agg['new_suffix_actions'],
  replayed_prefix_actions=agg['prefix_replay_actions'],successful_supervision_suffixes=0,training_steps=0)
 for arm in ['seed7','seed19']:
  root=P/'runs'/('bowl_correction_'+arm+'_v1_20261009')
  cost[arm+'_host_seconds']=json.loads((root/'FINAL_STATE.json').read_text())['host_wall_seconds']
  for step in [120,250,270]:
   r=json.loads((root/('step_'+str(step))/'RESULT.json').read_text())
   assert all(r['intervention_inputs_exact'].values()) and r['intermediate_state_restores']==0
   cost['full_episode_seconds']+=r['seconds_including_close'];cost['suffix_seconds']+=r['suffix_seconds'];cost['prefix_replay_seconds']+=r['prefix_replay_seconds']
   for row in r['rows']:
    for key in ['rpc','environment','evidence']:cost['suffix_'+key+'_seconds']+=row['model_'+key+'_seconds'] if key=='rpc' else row[key+'_seconds']
 save(out/'COST_BREAKDOWN.json',cost)
 save(out/'FINAL_READBACK.json',dict(status='COMPLETED_NOT_PROMOTED',containers=containers,default_unchanged=True,default_sha256=before,
  new_robot_continuation_trials=6,successful_suffixes=0,training_steps=0,value_data_check=json.loads((data/'VALIDATION_HOST.json').read_text())['status']))
 save(parent/'ADMISSION.json',dict(status='NO_VALIDATED_CORRECTIVE_SUFFIX',default_promoted=False,imitation_admitted=0,
  next='No further random retries at these three states; only external value-learning data preparation, not a trained model'))
 roots=[parent,P/'runs/bowl_correction_seed7_v1_20261009',P/'runs/bowl_correction_seed19_v1_20261009',P/'planning/bowl_value_records_v1_20261009',data,out]
 for root in roots:
  records=[]
  for f in sorted(root.rglob('*')):
   rel=f.relative_to(root)
   if f.is_file() and f.name!='COPY_MANIFEST.json' and not any(s in ('tokenizer_cache','__pycache__') for s in rel.parts):records.append(dict(path=str(rel),bytes=f.stat().st_size,sha256=sha(f)))
  save(root/'COPY_MANIFEST.json',dict(root=str(root),files=records,count=len(records),bytes=sum(r['bytes'] for r in records),exclude=['tokenizer_cache','__pycache__']))
 print(json.dumps(dict(owned_containers=len(containers),cost=cost,data_training_rows=len(train),data_validation_rows=len(validation))))
if __name__=='__main__':main()
