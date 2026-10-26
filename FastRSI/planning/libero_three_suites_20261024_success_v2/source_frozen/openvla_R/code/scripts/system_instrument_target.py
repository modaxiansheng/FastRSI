"""Common native TargetWS measurement including actual control commands."""
import hashlib
import importlib.util
import json
import os
from cost_trace import emit, span

spec = importlib.util.spec_from_file_location('frozen_target', '/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def wrap(name, category):
    original = getattr(module.LiberoRealRuntime, name)
    def measured(self, *args, **kwargs):
        if name == 'action_chunk':
            step_before = self.step_idx
            chunk = args[0] if args else kwargs['chunk']
            actions = chunk['actions']
            serialized = json.dumps(actions, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False, default=lambda value: value.tolist())
            emit({'event': 'actual_control_chunk', 'session_id': self.session_id,
                  'step_before': self.step_idx, 'actions_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
                  'requested_actions': len(actions)})
        with span(category, session_id=self.session_id, method=name):
            value = original(self, *args, **kwargs)
        if name == 'action_chunk':
            executed = self.step_idx - step_before
            assert 0 <= executed <= len(actions)
            for offset, action in enumerate(actions[:executed]):
                row = json.dumps(action, separators=(',', ':'), allow_nan=False,
                                 default=lambda value: value.tolist())
                emit({'event': 'actual_executed_action', 'session_id': self.session_id,
                      'step': step_before + offset + 1,
                      'action_sha256': hashlib.sha256(row.encode()).hexdigest()})
        if name in ('reset', 'action_chunk'):
            emit({'event': 'native_observation', 'session_id': self.session_id, 'method': name,
                  'steps': self.step_idx, 'native_success': bool(self.env.check_success()),
                  'runtime_success': bool(self.success)})
        return value
    setattr(module.LiberoRealRuntime, name, measured)

from boundary_render_v2 import install as install_boundary_render
install_boundary_render(module, enabled=os.environ.get('SYSTEM_TARGET_RENDER_BOUNDARIES')=='1', emit=emit)

wrap('reset', 'environment_reset')
wrap('action_chunk', 'environment_execution')
wrap('observe', 'observation')
module.main()
