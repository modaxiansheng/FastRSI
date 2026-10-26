"""Read-only analysis of completed source arms; never launch or resume execution."""
import hashlib,json
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
B=P/'planning/bowl_lora_source_comparison_v1_20261009'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,obj):
 path=B/name
 assert not path.exists(), 'Preserve existing analysis receipts'
 path.write_text(json.dumps(obj,indent=2)+'\n')
def main():
 gate=json.loads((B/'PARENT_GATE.json').read_text())
 training=P/'runs/bowl_expert_lora400_v1_20261009/RESULT.json'
 assert sha(training)==gate['training_result_sha256']
 for arm in ['original','adapted']:
  assert json.loads((B/(arm+'_HOST.json')).read_text())['exit_code']==0
  final=json.loads((P/'runs'/('bowl_lora_source_'+arm+'_v1_20261009')/'FINAL_STATE.json').read_text())
  assert final['default_unchanged'] and len(final['containers'])==4
  assert all(not c['state']['Running'] and c['state']['Pid']==0 for c in final['containers'])
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
 after=sha(P/'CURRENT_NATIVE_ENTRY.json');assert gate['default_sha256']==after
 eligible=totals['adapted']['successes']>=totals['original']['successes'] and not any(p['new_failure'] for p in pairs) and (totals['adapted']['actions']<totals['original']['actions'] or totals['adapted']['successes']>totals['original']['successes'])
 save('AGGREGATE.json',dict(pairs=pairs,totals=totals,source_qualification_passed=eligible,base_rng_equal=True,default_unchanged=True,default_promoted=False,heldout_generalization=False,complete_os_comparison=False,fast_rsi_gain_proven=False,
  training_host_seconds=json.loads((training.parent/'HOST_RESULT.json').read_text())['full_wall_seconds'],scope='Two trained source starts and one previously exposed validation start; native behavior qualification, not FastRSI gain'))
 save('ADMISSION.json',dict(status='SOURCE_QUALIFIED_NOT_PROMOTED' if eligible else 'REJECTED',default_promoted=False,next='Fresh contextual comparison required' if eligible else 'Do not expand this fixed candidate'))
 print(json.dumps(dict(totals=totals,source_qualification_passed=eligible)),flush=True)
if __name__=='__main__':main()
