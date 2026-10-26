"""Native target with explicit goal-skill action requests and actual action trace."""
import importlib.util,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.goal_skill import execute,load_skill
from cost_trace import emit,span
from stage_trace import capture
spec=importlib.util.spec_from_file_location('frozen_target','/core/PhyAgentOS/runtime/targets/remote/libero/server.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
config=json.loads(Path(os.environ['FASTRSI_GOAL_CONFIG']).read_text());skill=load_skill(config['artifact'],config['sha256']);root=Path(os.environ['FASTRSI_STAGE_TRACE']).parent
reset0=module.LiberoRealRuntime.reset;action0=module.LiberoRealRuntime.action_chunk

def record(self,method,chunk=None):
    emit(dict(event='native_observation',session_id=self.session_id,method=method,steps=self.step_idx,native_success=bool(self.env.check_success()),runtime_success=bool(self.success)))
    with span('stage_diagnostic_capture',session_id=self.session_id,step=self.step_idx):capture(self,method,() if chunk is None else (chunk,),os.environ['FASTRSI_STAGE_TRACE'])
def reset(self,*a,**kw):
    with span('environment_reset',session_id=self.session_id):out=reset0(self,*a,**kw)
    record(self,'reset');return out

def action(self,chunk):
    with span('environment_execution',session_id=self.session_id,method='action_chunk'):
        out,receipt=execute(self,chunk,skill,config['sha256'],config['root_session'],config['enabled'],action0)
    if receipt:
        row=dict(event='goal_skill_execution',session_id=self.session_id,**receipt)
        emit(row)
        with (root/'GOAL_SKILL_EXECUTION.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        chunk={**chunk,'actions':receipt['actual_actions']}
    record(self,'action_chunk',chunk);return out
observe0=module.LiberoRealRuntime.observe
def observe(self,*a,**kw):
    with span("observation",session_id=self.session_id,method="observe"):return observe0(self,*a,**kw)
module.LiberoRealRuntime.observe=observe
module.LiberoRealRuntime.reset=reset;module.LiberoRealRuntime.action_chunk=action
module.main()
