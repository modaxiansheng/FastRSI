"""One frozen VLA policy-loop episode, no Agent or verifier/model updates."""
import asyncio,hashlib,importlib.util,json,os,time
from pathlib import Path
from fastrsi.resident_client import ResidentClient
ROOT=Path(os.environ['FASTRSI_NATIVE_RUN']);case=json.loads((ROOT.parent/'ROSTER.json').read_text())[0]
OUT=ROOT/case['id'];OUT.mkdir(exist_ok=False)
port=int(os.environ['FASTRSI_RESIDENT_POLICY_PORT']);horizon=int(os.environ['FASTRSI_RESIDENT_INFERENCE_HORIZON'])
client=ResidentClient(port,case['id'],expected_action_horizon=horizon)
module_spec=importlib.util.spec_from_file_location('pure_target','/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
from cost_trace import emit,span
from boundary_render_v2 import install as install_boundary
install_boundary(module,enabled=True,emit=emit)
for method,category in [('reset','environment_reset'),('observe','observation'),('action_chunk','environment_execution')]:
    original=getattr(module.LiberoRealRuntime,method)
    def wrap(original,method,category):
        def measured(self,*a,**kw):
            before=self.step_idx
            with span(category,session_id=self.session_id,method=method):value=original(self,*a,**kw)
            if method=='action_chunk':emit({'event':'actual_executed_actions','session_id':self.session_id,'count':self.step_idx-before,'step':self.step_idx})
            if method in ('reset','action_chunk'):emit({'event':'native_observation','session_id':self.session_id,'steps':self.step_idx,'native_success':bool(self.env.check_success()),'runtime_success':bool(self.success)})
            return value
        return measured
    setattr(module.LiberoRealRuntime,method,wrap(original,method,category))
original_policy_infer=module._PolicyWsClient.infer
async def measured_policy_infer(self,*a,**kw):
    with span("policy_client"):return await original_policy_infer(self,*a,**kw)
module._PolicyWsClient.infer=measured_policy_infer
async def main():
    started=time.perf_counter();model=os.environ['FASTRSI_VLA_MODEL'];runtime=module.LiberoRealRuntime()
    receipt={};result=None
    try:
        receipt['begin']=client.begin('pure-begin-'+case['id'],1600)
        payload=dict(suite=case['suite'],episodes=[{'task_id':case['task_id'],'init_state_id':case['init_state_id'],'task_description':case['task_description']}],session_id=case['id'],max_steps=case['max_steps'],seed=7,control_mode='absolute' if model=='xvla' else 'relative',policy_endpoint='ws://127.0.0.1:'+str(port),replan_every_steps=10,verification={'profile':'strict'},camera_height=256,camera_width=256,num_steps_wait=10,evidence_mode='all')
        with span('pure_vla_episode'):result=await runtime.benchmark(payload)
        assert len(result['episodes'])==1
        episode=result['episodes'][0]
        assert episode.get('error_code') is None,episode
        row=dict(case=case,chain_status='PASSED',status='PASSED',environment_success=bool(episode['success']),physical_first_success=bool(episode['success']),physical_final_success=bool(episode['success']),semantic_success=None,seconds=time.perf_counter()-started,benchmark=result,accepted_source=None,agent_model_calls=0,planning_model_calls=0,report_model_calls=0,verifier_model_calls=0,qwen_tokens=0,scope='Direct native VLA policy loop, one attempt, no Agent/verifier and no experience updates')
        (OUT/'RESULT.json').write_text(json.dumps(row,indent=2,default=lambda x:x.tolist()))
    finally:
        if runtime.env is not None:runtime.env.close()
        state=client.status()
        if state.get('active') is not None:
            assert state['active']['session_id']==case['id'];receipt['end']=client.end('pure-end-'+case['id'])
        receipt['final']=client.status();receipt['seconds']=time.perf_counter()-started
        assert receipt['final'].get('active') is None
        (OUT/'RESIDENT_SESSION.json').write_text(json.dumps(receipt,indent=2))
asyncio.run(main())
