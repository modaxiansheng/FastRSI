"""Fixed source-completion candidate versus exact recorded handoff replay."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.retarget_geometry import rotation_vector
from replay_stack_progress import RUNTIME_SOURCE,facts,bind_goal_predicates,validate_trace
from stage_trace import png_bytes

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--arm',choices=['recorded','completion'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists();root=Path('/project');plan=json.loads(a.plan.read_text())
 for rel,sha in plan['sha256'].items():assert digest(root/rel)==sha,rel
 skill=json.loads((root/plan['skill']).read_text());trace=[json.loads(v) for v in (root/plan['trace']).read_text().splitlines()];validate_trace(trace)
 assert len(skill['waypoints'])==262 and plan['handoff']==130
 spec=importlib.util.spec_from_file_location('completion_runtime',RUNTIME_SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 runtime=m.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=64,init_state_id=1,seed=7,max_steps=400,control_mode='relative'))
 result=dict(status='RUNNING',arm=a.arm,plan_sha256=digest(a.plan),rows=[],image_checks=0,model_calls=0,training_updates=0,scope='Recorded branch is exact replay, candidate is a real new physical execution. No timing comparison to live VLA; no full Agent or learned selection claim.')
 start=time.perf_counter()
 try:
  runtime.reset({'session_id':'skill_completion_'+a.arm});env=runtime.env.env;goals=bind_goal_predicates(env,64);c=env.robots[0].controller
  def observe(expected=None,actions=None):
   row=dict(step=runtime.step_idx,actions=actions,evaluator=facts(env,runtime._last_obs,64,goals),gripper_qpos=runtime._last_obs['robot0_gripper_qpos'].tolist(),images={})
   for camera in ('agentview_image','robot0_eye_in_hand_image'):
    name=f'{runtime.step_idx:04d}_{camera}.png';blob=png_bytes(runtime._last_obs[camera]);sha=hashlib.sha256(blob).hexdigest()
    if expected is not None:
     assert sha==expected['images'][name],name;result['image_checks']+=1
    if runtime.step_idx in (0,130,180,200,220,240,260,262,400) or runtime.success:
     (a.output.parent/name).write_bytes(blob);row['images'][name]=sha
   result['rows'].append(row)
  observe(trace[0])
  for row in trace[1:]:
   if a.arm=='completion' and row['step']>130:break
   runtime.action_chunk({'actions':row['actions']});assert runtime.step_idx==row['step'];observe(row,row['actions'])
  if a.arm=='completion':
   assert runtime.step_idx==130 and not runtime.done
   for i in range(130,262):
    w=skill['waypoints'][i];c.update()
    desired=np.r_[np.asarray(w['target_pos'])-c.ee_pos,rotation_vector(np.asarray(w['target_rot'])@c.ee_ori_mat.T)]
    scale=np.abs(c.output_max-c.output_min)/np.abs(c.input_max-c.input_min)
    command=(desired-(c.output_max+c.output_min)/2)/scale+(c.input_max+c.input_min)/2
    action=np.r_[np.clip(command,c.input_min,c.input_max),w['gripper_command']].astype(np.float32)
    runtime.action_chunk({'actions':[action.tolist()]});observe(actions=[action.tolist()])
    if runtime.done:break
  final=result['rows'][-1]['evaluator'];success=bool(runtime.env.check_success())
  assert success==all(x['value'] for x in final['goal_predicates'])
  result.update(status='COMPLETED',steps=runtime.step_idx,full_task_success=success,bottom_in_tray=final['native_bottom_in_tray'],top_on_bottom=final['native_top_on_bottom'])
 except BaseException as e:result.update(status='FAILED',error=repr(e));raise
 finally:
  result['seconds']=time.perf_counter()-start
  with a.output.open('x') as f:json.dump(result,f,indent=2)
  if runtime.env is not None:runtime.env.close()
 print(json.dumps({k:result[k] for k in ('status','arm','steps','full_task_success','seconds','image_checks')}),flush=True)
if __name__=='__main__':main()
