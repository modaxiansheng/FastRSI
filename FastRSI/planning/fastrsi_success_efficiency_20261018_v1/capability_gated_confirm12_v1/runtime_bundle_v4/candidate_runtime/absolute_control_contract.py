"""Known source actions replayed in both control representations; no task outcome used to tune."""
from pathlib import Path
import json,hashlib,importlib.util,time
import numpy as np
from control_adapter import relative_to_absolute
from vision_skill import public_eef
R=Path('/out')
def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def main():
 start=time.time();records=[];entry=json.loads(Path('/source/REPLAY_PROTOCOL.json').read_text())['rows'][0];assert entry['identity_id']=='dev12';rawpath=Path(entry['source_raw']['path'].replace('/mnt/data/task/yuchen/FastPhyagentos/FastRSI/planning/fastrsi_fixed_retry32_20261016_v1','/history'));assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==entry['source_raw']['sha256'];raw=json.loads(rawpath.read_text());sequence=np.asarray(raw['executed_actions'],np.float32);assert raw['physical_success'] and len(sequence)==128
 p=Path('/core/PhyAgentOS/runtime/targets/remote/libero/server.py');spec=importlib.util.spec_from_file_location('control_contract_core',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 from boundary_render_v2 import install
 install(m,enabled=True,emit=lambda e:None);skill=json.loads(Path('/eval/grasp_skills/dev12.json').read_text());saved=np.load('/eval/generated/'+entry['source']['state_path'],allow_pickle=False)
 for mode in ['relative','absolute']:
  rt=m.LiberoRealRuntime();states=[];commands=[];error=None
  try:
   rt.reset(dict(session_id='control_contract_'+mode,libero=dict(benchmark_name='libero_object',task_id=3,init_state_id=0,max_steps=280,seed=7,control_mode=mode,camera_height=256,camera_width=256,num_steps_wait=10)));assert np.array_equal(rt.init_states[0],saved);initial=np.asarray(rt.env.get_sim_state()).copy()
   for i,a in enumerate(sequence):
    if rt.done:break
    obs=rt.observe();action=a if mode=='relative' else relative_to_absolute(obs,a,skill);rt.action_chunk(dict(chunk_id=mode+'_'+str(i),actions=action[None,:]));states.append(public_eef(rt.observe()));commands.append(action)
   record=dict(mode=mode,physical_success=bool(rt.success),actions=rt.step_idx,initial_state_sha256=digest(initial),eef_trajectory=states,commands=commands)
   assert record['physical_success'],'Source representation replay did not complete'
   records.append(record)
  finally:
   if rt.env is not None:rt.env.close()
 a,b=records;assert a['actions']==b['actions']==128 and a['initial_state_sha256']==b['initial_state_sha256'];aa=np.asarray(a['eef_trajectory']);bb=np.asarray(b['eef_trajectory']);pos=float(np.max(np.linalg.norm(aa[:,:3,3]-bb[:,:3,3],axis=1)));rot=float(np.max(np.abs(aa[:,:3,:3]-bb[:,:3,:3])));assert pos<1e-4 and rot<1e-3,(pos,rot)
 def plain(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,list):return [plain(x) for x in v]
  if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
  return v
 with (R/'RESULT.json').open('x') as f:json.dump(dict(accepted=True,source_only_not_new_rescue=True,policy_generations=0,model_calls=0,physical_attempts=2,official_actions=256,settling_actions=20,max_eef_position_difference_m=pos,max_eef_rotation_matrix_difference=rot,seconds=time.time()-start,records=plain(records)),f,indent=2)
if __name__=='__main__':main()
