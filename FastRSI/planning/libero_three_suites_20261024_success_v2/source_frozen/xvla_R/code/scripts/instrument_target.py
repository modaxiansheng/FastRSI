"""Wrap the frozen TargetWS implementation with measurement only."""
import importlib.util
import os
from cost_trace import emit, span

path = '/core/PhyAgentOS/runtime/targets/remote/libero/server.py'
spec = importlib.util.spec_from_file_location('frozen_target', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def wrap(method_name, category):
    original = getattr(module.LiberoRealRuntime, method_name)
    def measured(self, *args, **kwargs):
        with span(category, session_id=self.session_id, method=method_name):
            result = original(self, *args, **kwargs)
        if method_name in ('reset', 'action_chunk'):
            emit({'event':'native_observation','session_id':self.session_id,
                  'method':method_name,'steps':self.step_idx,
                  'native_success':bool(self.env.check_success()),
                  'runtime_success':bool(self.success)})
        if method_name in ('reset', 'action_chunk') and os.environ.get('FASTRSI_STAGE_TRACE'):
            from stage_trace import capture
            with span('stage_diagnostic_capture',session_id=self.session_id,step=self.step_idx):
                capture(self,method_name,args,os.environ['FASTRSI_STAGE_TRACE'])
        return result
    setattr(module.LiberoRealRuntime, method_name, measured)


wrap('reset','environment_reset')
wrap('action_chunk','environment_execution')
wrap('observe','observation')
module.main()
