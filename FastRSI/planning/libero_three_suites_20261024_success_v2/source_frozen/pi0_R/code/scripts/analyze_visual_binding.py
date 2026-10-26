"""Post-execution audit; privileged poses used here ONLY to score estimation error."""
from pathlib import Path
import hashlib,json
import numpy as np
root=Path(__file__).resolve().parents[1];p=root/'planning/visual_binding_20261005'
plan=json.loads((p/'EXECUTION_PLAN.json').read_text())
for rel,sha in plan['sha256'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha,rel
out={'scope':'Two already-seen development runs; not held-out generalization or full-task learning gains.','arms':{}}
results={}
for arm in ['zero','rgb']:
 r=json.loads((p/arm/'RESULT.json').read_text());results[arm]=r
 assert r['status']=='COMPLETED' and len(r['rows'])==130
 images=0
 for row in r['rows']:
  for name,digest in row.get('images',{}).items():
   assert hashlib.sha256((p/arm/name).read_bytes()).hexdigest()==digest;images+=1
 final=r['rows'][-1];facts=final['evaluator'];b=facts['bowls']['akita_black_bowl_1'];stop=json.loads((p/arm/'STOP.json').read_text())[0]
 checks=dict(bottom_in_tray=facts['native_bottom_in_tray'],no_grasp=not b['native_grasp'],no_contact=not b['gripper_contact'],open_fingers=all(abs(x)>.03 for x in final['gripper_qpos']),open_command=final['action'][-1]<0)
 assert all(checks.values())==r['local_release']
 out['arms'][arm]=dict(local_checks=checks,local_release=r['local_release'],full_task_success=r['full_task_success'],exit_code=stop['exit_code'],stopped=not stop['running'],binding_seconds=r['binding_seconds'],runtime_seconds=r['seconds'],model_calls=r['model_calls'],training_updates=r['training_updates'],images_verified=images,grasp_steps=[row['step'] for row in r['rows'] if row['evaluator']['bowls']['akita_black_bowl_1']['native_grasp']],applied_delta=r['applied_delta'],opencv_version=r['opencv_version'])
source=json.loads((root/'planning/retarget_probe_20261005/SOURCE_RECHECK.json').read_text())['boundaries'][0]['bowls']['akita_black_bowl_1']['world_position']
target=results['rgb']['initial_evaluator_only']['bowls']['akita_black_bowl_1']['world_position']
truth=np.asarray(target)-source;estimate=results['rgb']['estimated_delta'];out['posthoc_error_audit']=dict(estimated_delta=estimate,evaluator_true_delta=truth.tolist(),error_m=float(np.linalg.norm(truth-estimate)),used_by_controller=False)
out['same_initial_evaluator_state']=results['rgb']['initial_evaluator_only']==results['zero']['initial_evaluator_only']
out['differing_action_steps']=sum(a['action']!=b['action'] for a,b in zip(results['zero']['rows'],results['rgb']['rows']))
out['wall_seconds']=json.loads((p/'STATUS.json').read_text())['wall_seconds'];out['task_steps']=260;out['reset_wait_steps']=20
out['decision']='Do not promote RGB correction. Prioritize bounded same-episode continuation after unshifted source-goal skill. No default or learned-router update.'
with (p/'ANALYSIS.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
