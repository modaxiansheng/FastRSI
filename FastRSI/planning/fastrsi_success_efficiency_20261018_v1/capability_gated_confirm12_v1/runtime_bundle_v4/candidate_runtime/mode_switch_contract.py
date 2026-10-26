"""Second and final control-interface candidate: unchanged source commands, explicit mode handoff."""
from pathlib import Path
import json,time,hashlib,importlib.util,traceback
import numpy as np
from vision_skill import public_eef
R=Path('/out')
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,list):return [plain(x) for x in v]
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 return v
def save(name,v):
 with (R/name).open('x') as f:json.dump(plain(v),f,indent=2)
def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def main():
 started=time.time();records=[];error=None
 entry=json.loads(Path('/source/REPLAY_PROTOCOL.json').read_text())['rows'][0];assert entry['identity_id']=='dev12';p=Path(entry['source_raw']['path'].replace('/mnt/data/task/yuchen/FastPhyagentos/FastRSI/planning/fastrsi_fixed_retry32_20261016_v1','/history'));assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['source_raw']['sha256'];source=json.loads(p.read_text());sequence=np.asarray(source['executed_actions'],np.float32);assert source['physical_success'] and len(sequence)==128
 path=Path('/core/PhyAgentOS/runtime/targets/remote/libero/server.py');spec=importlib.util.spec_from_file_location('mode_contract_core',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 from boundary_render_v2 import install
 install(m,enabled=True,emit=lambda e:None)
 try:
  for mode in ['relative','absolute_then_skill_relative']:
   rt=m.LiberoRealRuntime();record=dict(mode=mode,frames=[],actions=[],success=None,error=None);records.append(record)
   try:
    rt.reset(dict(session_id='mode_source_'+mode,libero=dict(benchmark_name='libero_object',task_id=3,init_state_id=0,max_steps=280,seed=7,control_mode='relative' if mode=='relative' else 'absolute',camera_height=256,camera_width=256,num_steps_wait=10)));record['initial_state_sha256']=digest(rt.env.get_sim_state())
    if mode!='relative':
     before=digest(rt.env.get_sim_state());assert all(r.controller.use_delta is False for r in rt.env.robots)
     for robot in rt.env.robots:robot.controller.use_delta=True
     assert digest(rt.env.get_sim_state())==before;record['skill_mode_switch_kept_physics']=True
    for i,action in enumerate(sequence):
     if rt.done:break
     rt.action_chunk(dict(chunk_id=mode+'_'+str(i),actions=action[None,:]));record['frames'].append(public_eef(rt.observe()));record['actions'].append(action.copy())
    record['success']=bool(rt.success);record['official_actions']=rt.step_idx
    if mode!='relative':
     before=digest(rt.env.get_sim_state())
     for robot in rt.env.robots:robot.controller.use_delta=False
     assert digest(rt.env.get_sim_state())==before;record['native_mode_restored_without_reset']=True
    assert record['success'] and record['official_actions']==128
   except BaseException:record['error']=traceback.format_exc();raise
   finally:
    if rt.env is not None:rt.env.close()
    save(mode+'_RAW.json',record)
  a,b=records;assert a['initial_state_sha256']==b['initial_state_sha256'];aa=np.asarray(a['frames']);bb=np.asarray(b['frames']);position=float(np.max(np.linalg.norm(aa[:,:3,3]-bb[:,:3,3],axis=1)));rotation=float(np.max(np.abs(aa[:,:3,:3]-bb[:,:3,:3])));assert position<1e-4 and rotation<1e-3,(position,rotation)
  save('RESULT.json',dict(accepted=True,candidate='EXPLICIT_MODE_SWITCH_UNCHANGED_SOURCE_ACTIONS',source_only_not_new_rescue=True,physical_attempts=2,official_actions=256,settling_actions=20,model_calls=0,seconds=time.time()-started,maximum_position_difference_m=position,maximum_rotation_matrix_difference=rotation,all_raw_saved=True))
 except BaseException:error=traceback.format_exc();raise
 finally:save('SOURCE_ATTEMPT_ACCOUNTING.json',dict(seconds=time.time()-started,physical_attempts_started=len(records),actions=sum(len(r['actions']) for r in records),physical_successes=sum(r['success'] is True for r in records),error=error,source_only=True,money=None))
if __name__=='__main__':main()
