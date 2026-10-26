"""Opt-in native-draw Agent target; preserves original verifier child Sessions.

Without FASTRSI_NATIVE_SAMPLER_CONFIG reset remains the original native reset.
No official fixed50 scoring claim is made when a sampler input is enabled.
"""
import importlib.util,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.native_goal_component_target_v1 import RegisteredGoalTarget
from cost_trace import emit,span
from stage_trace import capture
spec=importlib.util.spec_from_file_location('frozen_target','/core/PhyAgentOS/runtime/targets/remote/libero/server.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from fastrsi.native_sampler_agent_binding_v1 import install
sampler_path=os.environ.get('FASTRSI_NATIVE_SAMPLER_CONFIG')
restorer=install(module,None if sampler_path is None else json.loads(Path(sampler_path).read_text()))
component_path=os.environ.get('FASTRSI_COMPONENT_TARGET_CONFIG')
component_config=None if component_path is None else json.loads(Path(component_path).read_text())
if restorer is not None and component_config is not None:
    if (Path(component_config['workspace']).resolve()!=restorer.workspace or component_config['root_session']!=restorer.root_session):
        raise ValueError('Goal consumer and sampler must bind the same native root/workspace')
consumer=None if component_config is None else RegisteredGoalTarget(component_config,runtime_source=spec.origin)
root=Path(os.environ['FASTRSI_STAGE_TRACE']).parent

def execute_request(runtime,chunk):
    return (action0(runtime,chunk),None) if consumer is None else consumer.execute(runtime,chunk,action0)
reset0=module.LiberoRealRuntime.reset;action0=module.LiberoRealRuntime.action_chunk
live_root=os.environ.get('FASTRSI_LIVE_CONTROLS')
if live_root:
    from fastrsi.live_controller_trace import LiveControllerTrace
    from fastrsi.experience_compiler import canonical_sha

def record(self,method,chunk=None):
    emit(dict(event='native_observation',session_id=self.session_id,method=method,steps=self.step_idx,native_success=bool(self.env.check_success()),runtime_success=bool(self.success)))
    with span('stage_diagnostic_capture',session_id=self.session_id,step=self.step_idx):capture(self,method,() if chunk is None else (chunk,),os.environ['FASTRSI_STAGE_TRACE'])
def reset(self,*a,**kw):
    prior=getattr(self,'_fastrsi_live_controls',None)
    if prior is not None:prior.detach()
    previous_receipt=None if restorer is None else restorer.last_receipt
    try:
        with span('environment_reset',session_id=self.session_id):out=reset0(self,*a,**kw)
    finally:
        if restorer is not None and restorer.last_receipt is not None and restorer.last_receipt is not previous_receipt:
            row=dict(event='native_sampler_restore',**restorer.last_receipt)
            emit(row)
            with (root/'NATIVE_SAMPLER_RESTORE.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if live_root:
        with span('controller_target_logging',session_id=self.session_id,phase='setup'):
            self._fastrsi_live_controls=LiveControllerTrace(self,live_root,runtime_source=spec.origin)
    record(self,'reset');return out

def action(self,chunk):
    logger=getattr(self,'_fastrsi_live_controls',None) if live_root else None
    with span('environment_execution',session_id=self.session_id,method='action_chunk'):
        if logger is None:
            out,receipt=execute_request(self,chunk)
        else:
            with logger.chunk(chunk):
                out,receipt=execute_request(self,chunk)
                candidate=None
                if receipt is not None:
                    candidate=consumer.bound['program']
                logger.bind_execution(receipt,source_skill_sha256=None if candidate is None else canonical_sha(candidate))
    if receipt:
        row=dict(event='registered_goal_component_execution',session_id=self.session_id,**receipt)
        emit(row)
        with (root/'REGISTERED_GOAL_COMPONENT_EXECUTION.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        chunk={**chunk,'actions':receipt['actual_actions']}
    record(self,'action_chunk',chunk)
    if logger is not None:
        with span('controller_target_logging',session_id=self.session_id,phase='finalize'):
            logger.boundary(Path(os.environ['FASTRSI_STAGE_TRACE'])/self.session_id/'TRACE.jsonl')
        emit(dict(event='controller_target_logging',session_id=self.session_id,controls_count=len(logger.controls),logging_seconds=logger.logging_seconds,finalized=logger.finalized))
    return out
observe0=module.LiberoRealRuntime.observe
def observe(self,*a,**kw):
    with span("observation",session_id=self.session_id,method="observe"):return observe0(self,*a,**kw)
module.LiberoRealRuntime.observe=observe
module.LiberoRealRuntime.reset=reset;module.LiberoRealRuntime.action_chunk=action
module.main()
