"""Bounded visual grasp/place execution. Control only consumes public observations and visual RPC replies."""
from pathlib import Path
import json,hashlib,importlib.util,time,traceback,asyncio,os,collections
import numpy as np
from vision_client import VisionClient
from vision_skill import correction_plan,action_for_pose,public_eef
from visual_kernel import initial_pose_gate,held_gate,place_plan
from object_frame_skill import rotation_vector
from control_adapter import backend_action_for_pose
H=Path('/eval');R=Path('/pilot')
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x) for x in v]
 return v
def save(p,v):
 with p.open('x') as f:json.dump(plain(v),f,indent=2,allow_nan=False)
def digest(v):return hashlib.sha256(json.dumps(plain(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def clone(obs):return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in obs.items()}
def save_first_native_input(obs,request,step,index):
 if index!=1:return None
 from PIL import Image
 p=R/'native_inputs'/str(step);p.mkdir(parents=True,exist_ok=False)
 Image.fromarray(obs['agentview_image']).save(p/'front.png');Image.fromarray(obs['robot0_eye_in_hand_image']).save(p/'wrist.png')
 np.save(str(p/'state.npy'),request['observation/state'],allow_pickle=False);np.save(str(p/'eef_mat.npy'),request['observation/eef_mat'],allow_pickle=False)
 save(p/'INPUT.json',dict(prompt=request['prompt'],session_id=request['session_id'],input_sha256=digest(request),files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()}))
 return p

def main():
 p=Path('/core/PhyAgentOS/runtime/targets/remote/libero/server.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='c89b483a3d5bf830b98a6470a6fa6c41e70ab12930fee4b7eab99b0d99fbba41'
 spec=importlib.util.spec_from_file_location('visual_grasp_place_core',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 from boundary_render_v2 import install
 def emit(v):
  with (R/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(plain(dict(time=time.time(),**v)))+'\n')
 install(m,enabled=True,emit=emit);job=json.loads((R/'JOB.json').read_text());row=job['row'];source=row['source'];model=row['model'];supported=row['skill'] is not None;src=row['skill'] or 'dev12';skill=json.loads((H/'grasp_skills'/(src+'.json')).read_text());release=json.loads((H/'release_skills'/(src+'.json')).read_text());vision=VisionClient(R/'vision_rpc',emit);rt=m.LiberoRealRuntime();actions=[];reason=None;error=None;final=None;started=time.time();native_generations=0;resident=None;policy_loop=None;extra_evidence_used=False;policy_offset=(R/'POLICY_COST_EVENTS.jsonl').stat().st_size if (R/'POLICY_COST_EVENTS.jsonl').exists() else 0
 def metadata(step,phase):return dict(subject=skill['subject'],variant=source['variant'],task_id=source['task_id'],step=step,phase=phase)
 def act(action,role):
  before=rt.step_idx;assert not rt.done and before<source['max_steps'];obs=clone(rt.observe());command=np.asarray(action,np.float32).reshape(1,7);rt.action_chunk(dict(chunk_id='visual_skill_'+str(before),actions=command));assert rt.step_idx==before+1;actions.append(command[0].copy());emit(dict(event='executed_control_action',step=rt.step_idx,role=role,action=command[0],public_eef_before=public_eef(obs),public_eef_after=public_eef(rt.observe()),official_success=bool(rt.success)))
 def subject(obs,step,phase,prior):
  value,proof=vision.request('subject',obs,metadata(step,phase),prior);assert value is not None;return value
 def check_and_plan(previous_obs,previous_pose):
  nonlocal extra_evidence_used
  obs=clone(rt.observe());prior=public_eef(obs)@np.linalg.inv(public_eef(previous_obs))@previous_pose;obj=subject(obs,rt.step_idx,'holding_recheck',prior)
  prev=held_gate(previous_obs,previous_pose,release);gate=held_gate(obs,obj,release,prev)
  emit(dict(event='visual_holding_decision',step=rt.step_idx,decision=gate,public_gripper_qpos=obs['robot0_gripper_qpos']))
  if gate['status']!='HELD_TARGET_ESTIMATE':return None,obs,obj,gate['reason']
  basket,proof=vision.request('basket',obs,metadata(rt.step_idx,'destination_recheck'))
  if basket is None:return None,obs,obj,'BASKET_POSE_UNKNOWN'
  plan=place_plan(obs,release,obj,basket);emit(dict(event='visual_placement_plan',step=rt.step_idx,plan=plan));return plan,obs,obj,None
 try:
  cfg=dict(benchmark_name=source['suite'],task_id=source['task_id'],init_state_id=0,max_steps=source['max_steps'],seed=7,control_mode='absolute' if model=='xvla' else 'relative',camera_height=256,camera_width=256,num_steps_wait=10)
  if job['mode'] in ['paired_fourth','ordinary_fourth']:
   from fastrsi.resident_client import ResidentClient
   # Sessions identify the exact registered condition and paired mode.
   transport_mode='ordinary' if job['mode']=='ordinary_fourth' else 'visual';resident=ResidentClient(int(os.environ['PI05_POLICY_PORT']),row['identity_id']+'--'+transport_mode,expected_action_horizon=row['wire_horizon']);entry=resident.begin('paired-begin-'+row['identity_id']+'-'+transport_mode,1600);assert entry['receipt']['paired_start_rng_hashes']==row['after_baseline_rng_hashes'];save(R/'RESIDENT_ENTRY.json',entry)
  rt.reset(dict(session_id='visual_'+row['identity_id'],libero=cfg));assert np.array_equal(np.asarray(rt.init_states[0]),np.load(H/'generated'/source['state_path'],allow_pickle=False));initial_state=digest(np.asarray(rt.env.get_sim_state())[1:]);initial_obs=clone(rt.observe());
  if 'baseline_reset_state_sha256' in row:assert initial_state==row['baseline_reset_state_sha256']
  first_input=digest({k:v for k,v in m._policy_observation(initial_obs,source['language'],'transport-independent').items() if k not in ['session_id','timestamp_ns','observation_id']})
  if 'baseline_first_input_sha256' in row:assert first_input==row['baseline_first_input_sha256']
  save(R/'INITIAL_INPUT_ACCEPTANCE.json',dict(reset_state_sha256=initial_state,initial_policy_input_sha256=first_input,original_prompt=source['language']))
  if job['mode']!='ordinary_fourth' and not supported:reason='SOURCE_SKILL_UNAVAILABLE'
  if job['mode']!='ordinary_fourth' and supported:
   initial=subject(initial_obs,0,'initial',None);gate=initial_pose_gate(initial,skill);emit(dict(event='visual_initial_gate',decision=gate))
   if gate['status']!='ACCEPT':reason=gate['reason']
   if reason is None:
    sequence,_=correction_plan(initial_obs,skill,initial);save(R/'GRASP_PLAN.json',dict(sequence=[dict(pose=p,grip=g,role=role) for p,g,role in sequence],initial_rgb_pose=initial,GT_read=False));assert len(sequence)+len(release['source_release_suffix'])<source['max_steps'];pre_obs=None;pre_step=None
    for index,(target,grip,role) in enumerate(sequence):
     if rt.done:break
     act(backend_action_for_pose(rt.observe(),np.asarray(target),grip,skill,model),role)
     if index==len(sequence)-3:pre_obs=clone(rt.observe());pre_step=rt.step_idx
    if not rt.done:
     assert pre_obs is not None
     pre_prior=initial@np.linalg.inv(np.asarray(release['source_initial_object_pose']))@np.asarray(release['source_pregrasp_object_pose']);pre_pose=subject(pre_obs,pre_step,'last_grasp_motion',pre_prior)
     plan,last_obs,last_pose,reason=check_and_plan(pre_obs,pre_pose)
     if last_pose[2,3]-initial[2,3]<=release['position_tolerance_m']:reason='VISUAL_LIFT_NOT_CONFIRMED'
     for phase_index,phase in enumerate(['lift_held_subject','align_above_destination','lower_to_release']):
      if reason or rt.done:break
      target=np.asarray(plan['targets'][phase_index]);count=0
      while not rt.done:
       now=public_eef(rt.observe());dp=float(np.linalg.norm(target[:3,3]-now[:3,3]));dr=float(np.linalg.norm(rotation_vector(target[:3,:3]@now[:3,:3].T)))
       if dp<=plan['position_tolerance_m'] and dr<=plan['rotation_tolerance_rad']:break
       if count>=plan['max_steps_per_phase'] or rt.step_idx+len(plan['suffix'])>=source['max_steps']:reason='BOUNDED_PLACEMENT_PHASE_LIMIT';break
       act(backend_action_for_pose(rt.observe(),target,plan['close_gripper'],skill,model),phase);count+=1
      emit(dict(event='placement_phase_end',phase=phase,step=rt.step_idx,position_error_m=dp,rotation_error_rad=dr,reason=reason))
      if reason or rt.done:break
      plan,last_obs,last_pose,reason=check_and_plan(last_obs,last_pose)
     if reason is None and not rt.done:
      for s in plan['suffix']:
       if rt.done:break
       act(backend_action_for_pose(rt.observe(),np.asarray(s['pose']),s['grip'],skill,model),'source_release_suffix')
      reason='OFFICIAL_TASK_SUCCESS' if rt.success else 'SOURCE_RELEASE_FINISHED_NEEDS_NATIVE_CONTINUATION'
  if job['mode'] in ['paired_fourth','ordinary_fourth'] and not rt.done:
   from initial_acquisition import acquire_boundary
   assert resident is not None
   boundary=acquire_boundary(resident,0,'paired-first-inference-'+row['identity_id']);save(R/'HANDOFF.json',dict(step=rt.step_idx,boundary=boundary,original_prompt=source['language'],environment_reset=False))
   async def finish_native():
    nonlocal native_generations
    async with m._PolicyWsClient('ws://127.0.0.1:'+os.environ['PI05_POLICY_PORT'],timeout_s=120.) as policy:
     policy.reset_session(resident.session_id)
     while not rt.done:
      obs=clone(rt.observe());request=m._policy_observation(obs,source['language'],resident.session_id);before=rt.step_idx;physical_before=digest(np.asarray(rt.env.get_sim_state())[1:]);out=await policy.infer(request);assert before==rt.step_idx and physical_before==digest(np.asarray(rt.env.get_sim_state())[1:]);native_generations+=1;input_dir=save_first_native_input(obs,request,before,native_generations);command=m._policy_actions(out)[:row['execution_horizon']];assert command.shape==(row['execution_horizon'],7);rt.action_chunk(dict(chunk_id='paired_native_'+str(before),actions=command));actual=command[:rt.step_idx-before];actions.extend(actual.copy());emit(dict(event='fresh_native_continuation',step=before,input_sha256=digest(request),input_evidence=None if input_dir is None else str(input_dir),executed_actions=actual,official_success=bool(rt.success)))
   policy_loop=asyncio.new_event_loop();policy_loop.run_until_complete(finish_native());assert rt.done
  final=bool(rt.success)
  # Every paired fourth attempt ends only at original success or the official cap.
  if job['mode']!='skill_regression':assert rt.done and rt.step_idx<=source['max_steps']
  assert job['mode'] in ['skill_regression','paired_fourth','ordinary_fourth']
  from PIL import Image
  for k,name in [('agentview_image','front'),('robot0_eye_in_hand_image','wrist')]:Image.fromarray(rt.observe()[k]).save(R/('final_'+name+'.png'))
 except BaseException:error=traceback.format_exc()
 finally:
  final_state=None
  if rt.env is not None:
   final_state=digest(np.asarray(rt.env.get_sim_state())[1:])
   try:rt.env.close();rt.env=None
   except BaseException:error=(error or '')+'\n'+traceback.format_exc()
  if resident is not None:
   try:
    if resident.status().get('active') is not None:save(R/'RESIDENT_END.json',resident.end('paired-end-'+row['identity_id']+'-'+transport_mode))
   except BaseException:error=(error or '')+'\n'+traceback.format_exc()
  if policy_loop is not None:policy_loop.close()
  try:vision.close() if job['mode']!='ordinary_fourth' and supported else None
  except BaseException:error=(error or '')+'\n'+traceback.format_exc()
  ledger=R/'POLICY_COST_EVENTS.jsonl';content=b''
  if ledger.exists():
   with ledger.open('rb') as f:f.seek(policy_offset);content=f.read()
  (R/'PHYSICAL_POLICY_EVENTS.jsonl').write_bytes(content);policy_counts=dict(collections.Counter(json.loads(line)['event'] for line in content.splitlines() if line))
  native_accounting_complete=policy_counts.get('policy_rpc_end',0)==native_generations
  if error is None:assert native_accounting_complete
  raw=dict(time=time.time(),identity=row['identity_id'],mode=job['mode'],software_ok=error is None,error=error,physical_success=final,episode_completed=bool(final) or len(actions)>=source['max_steps'],stop_reason=reason,official_actions=len(actions),executed_actions=actions,reset_state_sha256=locals().get('initial_state'),final_state_sha256=final_state,seconds=time.time()-started,new_native_policy_requests=native_generations,new_native_policy_generations=policy_counts.get('action_generation_end',0),policy_counts=policy_counts,native_accounting_complete=native_accounting_complete,policy_ledger_sha256=hashlib.sha256(content).hexdigest(),extra_visual_evidence_used=extra_evidence_used,visual_requests=vision.index,GT_control_input=False,scope='INTEGRATED_VISUAL_SKILL_REGRESSION_NOT_UNTUNED_CONFIRMATION' if job['mode']=='skill_regression' else 'MATCHED_SINGLE_EXTRA_ATTEMPT_NEW_MULTIBACKEND_DEVELOPMENT',money=None)
  save(R/'RAW.json',raw);save(R/'RESULT.json',{k:v for k,v in raw.items() if k!='executed_actions'})
 if error:raise RuntimeError(error)
if __name__=='__main__':main()
