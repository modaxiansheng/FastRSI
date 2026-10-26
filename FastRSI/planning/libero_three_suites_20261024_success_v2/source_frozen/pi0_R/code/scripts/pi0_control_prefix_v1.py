"""Opt-in PI0 receding-horizon execution. Model generation remains 50 steps."""
import hashlib
import json
import os
from pathlib import Path


def wrap_adapter(adapter_class, execution_steps, emit):
    if type(execution_steps) is not int or execution_steps not in (5, 10, 50):
        raise ValueError('Registered execution prefix must be 5, 10, or control 50')
    if getattr(adapter_class, '_pi0_control_prefix_installed', False):
        raise RuntimeError('Refuse repeated adapter installation')
    original = adapter_class.from_policy_output
    def convert(self, policy_output, session_ctx):
        if (session_ctx.get('benchmark') or {}).get('policy_id') != 'pi0_libero':
            raise ValueError('PI0-only candidate cannot execute another policy')
        execution = session_ctx.get('execution') or {}
        if execution.get('replan_every_steps') != execution_steps:
            raise ValueError('Session execution cadence differs from registered prefix')
        chunk = original(self, policy_output, session_ctx)
        actions = chunk['actions']
        if tuple(actions.shape) != (50, 7):
            raise ValueError('PI0 model must still generate 50 by 7 actions')
        import numpy as np
        if not np.isfinite(actions).all():
            raise ValueError('Non-finite PI0 action rejected')
        prefix = actions[:execution_steps].copy()
        chunk = dict(chunk, actions=prefix,
            action_contract={**chunk['action_contract'], 'shape': [execution_steps, 7]},
            timing={**chunk['timing'], 'horizon_steps': execution_steps},
            policy_meta={**chunk.get('policy_meta', {}),
                'model_generation_horizon': 50, 'execution_prefix_steps': execution_steps,
                'unused_tail_discarded': True})
        emit({'event': 'pi0_execution_prefix_pre_dispatch',
              'session_id': session_ctx['session_id'],
              'source_observation_id': session_ctx.get('source_observation_id'),
              'model_generation_horizon': 50, 'commanded_steps': execution_steps,
              'generated_actions_sha256': hashlib.sha256(actions.tobytes()).hexdigest(),
              'commanded_actions_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
              'unused_tail_discarded': True})
        return chunk
    adapter_class.from_policy_output = convert
    adapter_class._pi0_control_prefix_installed = True


def install(native, config_path):
    config = json.loads(Path(config_path).read_text())
    if config['model'] != 'pi0' or config['generation_horizon'] != 50:
        raise ValueError('Invalid PI0 model configuration')
    if os.environ['FASTRSI_VLA_MODEL'] != 'pi0' or os.environ['SYSTEM_EFFICIENCY_ARM'] not in ('P','R'):
        raise ValueError('Only matched PI0 P/R arms admitted')
    steps = config['execution_steps']
    root = Path(os.environ['FASTRSI_NATIVE_RUN'])
    root.mkdir(parents=True, exist_ok=True)
    receipt = root/'PI0_CONTROL_BINDING.json'
    with receipt.open('x') as f:
        json.dump(dict(config, arm=os.environ['SYSTEM_EFFICIENCY_ARM'],
                       config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()),f,indent=2)
    event_file = (root/'PI0_CONTROL_EVENTS.jsonl').open('x', buffering=1)
    sequence = 0
    def emit(event):
        nonlocal sequence
        sequence += 1
        event_file.write(json.dumps(dict(event, sequence=sequence))+'\n')
        event_file.flush()
    from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
    wrap_adapter(OpenPIPi05Adapter, steps, emit)
    original = native._write_workspace
    def write_workspace(path, args):
        args.replan_every_steps = steps
        return original(path, args)
    native._write_workspace = write_workspace
