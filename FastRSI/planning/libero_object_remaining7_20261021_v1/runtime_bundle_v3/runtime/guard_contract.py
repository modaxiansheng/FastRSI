from pathlib import Path
import json,numpy as np,hashlib,time
from visual_kernel import held_gate
H=Path('/data');case=H/'regression/node188/initialskill03';rows=[json.loads(x) for x in (case/'EVENTS.jsonl').read_text().splitlines()]
actions={x['step']:x for x in rows if x['event']=='executed_control_action'};old=next(x for x in rows if x['event']=='visual_holding_decision');poses={}
for name in ['guard03pre67','guard03cut69']:
 p=H/'guard_relocalization/output'/name/'bbq_sauce_1.json';v=json.loads(p.read_text());assert v['GT_read'] is False and v['selection']=='FROZEN_LEARNED_ARGMAX_NO_GT_RERANK';poses[name]=np.asarray(v['T_world_object'])
e67=np.asarray(actions[67]['public_eef_after']);e69=np.asarray(actions[69]['public_eef_after']);obs=dict(robot0_eef_pos=e69[:3,3],robot0_eef_mat=e69[:3,:3],robot0_gripper_qpos=np.asarray(old['public_gripper_qpos']));previous=dict(relative_eef_object=(np.linalg.inv(e67)@poses['guard03pre67']).tolist());skill=json.loads((H/'release_compile/release_skills/dev12.json').read_text());result=held_gate(obs,poses['guard03cut69'],skill,previous)
out=dict(time=time.time(),original_decision=old['decision'],after_bounded_relocalization=result,gate_threshold_changed=False,new_physics=0,target_GT_read=False,new_model_generations_in_this_judge=0,extra_neural_RGB_images=2,neural_cost_source='guard_relocalization/output/RESULT.json',claim='Public-image/proprioception evidence-contract only, not an additional physical rescue')
with Path('/out/GUARD_CONTRACT.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
