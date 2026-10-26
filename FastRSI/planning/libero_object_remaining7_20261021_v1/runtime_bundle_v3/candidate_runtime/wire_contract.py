"""Six actual RPCs, no environment construction, exact matching A/B after baseline-end checkpoint."""
import os,json,asyncio,importlib.util,hashlib
from pathlib import Path
import numpy as np
from fastrsi.resident_client import ResidentClient
from initial_acquisition import acquire_boundary
p=Path('/core/PhyAgentOS/runtime/targets/remote/libero/server.py');s=importlib.util.spec_from_file_location('contract_core',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
async def run():
 model=os.environ['FASTRSI_VLA_MODEL'];horizon={'pi0':50,'pi05':10,'openvla':1,'xvla':1}[model];records=[];outputs={}
 public=dict(agentview_image=np.zeros((256,256,3),np.uint8),robot0_eye_in_hand_image=np.zeros((256,256,3),np.uint8),robot0_eef_pos=np.array([0.,0.,.6]),robot0_eef_mat=np.diag([1.,-1.,-1.]),robot0_gripper_qpos=np.array([.04,-.04]),robot0_eef_quat=np.array([1.,0.,0.,0.]))
 for mode in ['base','ordinary','visual']:
  sid='wire-'+model+'--'+mode;c=ResidentClient(int(os.environ['PI05_POLICY_PORT']),sid,expected_action_horizon=horizon);begin=c.begin('wire-begin-'+sid,20);acquire_boundary(c,0,'wire-first-'+sid);responses=[]
  async with m._PolicyWsClient('ws://127.0.0.1:'+os.environ['PI05_POLICY_PORT'],timeout_s=180.) as policy:
   policy.reset_session(sid)
   for _ in range(2):
    obs=m._policy_observation(public,'Protocol preflight only',sid);v=await policy.infer(obs);a=m._policy_actions(v);assert a.shape==(horizon,7) and np.isfinite(a).all();responses.append(a.copy())
  end=c.end('wire-end-'+sid);outputs[mode]=responses;records.append(dict(mode=mode,begin=begin,end=end,actions_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in responses]))
 assert records[1]['begin']['receipt']['paired_start_rng_hashes']==records[2]['begin']['receipt']['paired_start_rng_hashes']==records[0]['end']['receipt']['end_rng_hashes']
 assert all(np.array_equal(a,b) for a,b in zip(outputs['ordinary'],outputs['visual']))
 with Path('/out/RESULT.json').open('x') as f:json.dump(dict(accepted=True,model=model,environment_constructors=0,physical_actions=0,true_policy_requests=6,wire_horizon=horizon,matching_actions=True,matching_start_rng=True,records=records),f,indent=2)
if __name__=='__main__':asyncio.run(run())
