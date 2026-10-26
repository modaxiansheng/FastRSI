"""Precisely bound development branch inside the unchanged complete native Agent."""
import asyncio
import hashlib
import json
import re
from pathlib import Path

import run_native_baseline as native
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
from cost_trace import emit, span
from fastrsi.boundary_probe import PrefixGuard, bind, reference, guarded_commit
from stage_trace import png_bytes


def observation_step(value):
    match = re.fullmatch(r'libero_obs_(\d+)', str(value))
    if not match:
        raise ValueError('Missing original observation identity')
    return int(match.group(1))


def array_bindings(payload):
    return {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                'sha256': hashlib.sha256(v.tobytes()).hexdigest()}
            for k, v in payload.items() if hasattr(v, 'shape') and hasattr(v, 'tobytes')}


original_case = native.run_case


async def run_case(case, accepted):
    probe = case['boundary_probe']
    with span('boundary_reference_load', case=case['id']):
        spec = bind(case, Path(probe['artifact']).read_bytes(), probe['sha256'])
        guard = PrefixGuard(reference(spec, Path(spec['reference_trace']['path']).read_bytes(),
                                      Path(spec['reference_cost']['path']).read_bytes()),
                            boundary=spec['boundary_step'], goal=spec['goal'], remaining_goal=spec['remaining_goal'])
    boundary, goal = spec['boundary_step'], spec['goal']
    original_input = OpenPIPi05Adapter.to_policy_input
    original_output = OpenPIPi05Adapter.from_policy_output
    original_commit = native.commit_experience
    changed = 0
    root_last_step = None

    def wrapped_input(adapter, observation, context):
        nonlocal changed, root_last_step
        # Sticky rejection also blocks native retries following a bad prefix.
        guard.check_healthy()
        payload = original_input(adapter, observation, context)
        if context.get('session_id') != case['id']:
            if not guard.matched:
                guard.reject('Native retry cannot bypass an incomplete original prefix')
            return payload
        if context.get('task_description') != goal:
            guard.reject('Root goal changed before the bound intervention')
        try:
            step = observation_step(observation.get('observation_id'))
            if step <= boundary:
                with span('boundary_prefix_observation_check', case=case['id'], step=step):
                    image_hashes = {
                        f'{step:04d}_{camera}.png': hashlib.sha256(png_bytes(payload[key])).hexdigest()
                        for camera, key in [('agentview_image', 'observation/image'),
                                            ('robot0_eye_in_hand_image', 'observation/wrist_image')]
                    }
                    guard.observe(step, array_bindings(payload), image_hashes, payload['prompt'])
                emit({'event': 'boundary_prefix_observation_verified', 'case': case['id'],
                      'step': step, 'artifact_sha256': probe['sha256']})
            elif not guard.matched:
                guard.reject('Skipped the original intervention boundary')
            root_last_step = step
            prompt = guard.prompt(step, probe['enabled'])
            changed += int(prompt != goal)
            emit({'event': 'boundary_choice', 'case': case['id'], 'step': step,
                  'enabled': probe['enabled'], 'prompt': prompt, 'original_goal': goal,
                  'prefix_verified': guard.matched, 'teacher_calls': 0,
                  'scope': spec['scope'], 'artifact_sha256': probe['sha256']})
            return {**payload, 'prompt': prompt}
        except Exception as error:
            guard.reject(f'Boundary input rejected: {type(error).__name__}: {error}')

    def wrapped_output(adapter, payload, context):
        guard.check_healthy()
        chunk = original_output(adapter, payload, context)
        if context.get('session_id') != case['id']:
            if not guard.matched:
                guard.reject('Native retry output cannot bypass original prefix')
            return chunk
        try:
            step = observation_step(context.get('source_observation_id'))
            if step != root_last_step:
                guard.reject('Policy output observation differs from the checked input')
            if step < boundary:
                with span('boundary_prefix_action_check', case=case['id'], step=step):
                    actions = chunk['actions']
                    guard.actions(step, actions.tolist() if hasattr(actions, 'tolist') else actions)
                emit({'event': 'boundary_prefix_actions_verified', 'case': case['id'],
                      'step': step, 'actions': 10, 'before_submission': True})
            return chunk
        except Exception as error:
            guard.reject(f'Boundary output rejected: {type(error).__name__}: {error}')

    OpenPIPi05Adapter.to_policy_input = wrapped_input
    OpenPIPi05Adapter.from_policy_output = wrapped_output
    native.commit_experience = guarded_commit(guard, original_commit)
    try:
        result = await original_case(case, accepted)
        guard.finish()
        return result
    finally:
        OpenPIPi05Adapter.to_policy_input = original_input
        OpenPIPi05Adapter.from_policy_output = original_output
        native.commit_experience = original_commit
        receipt = {'case': case['id'], 'artifact_sha256': probe['sha256'],
                   'scope': spec['scope'], 'reference_trace': spec['reference_trace'],
                   'reference_cost': spec['reference_cost'], 'enabled': probe['enabled'],
                   'boundary_step': boundary, 'root_max_steps': spec['max_steps'],
                   'verified_observations': guard.observations_verified,
                   'verified_action_chunks': guard.action_chunks_verified,
                   'prefix_matched': guard.matched, 'failure': guard.failure,
                   'changed_policy_prompts': changed, 'teacher_calls': 0,
                   'status': 'PREFIX_MATCHED' if guard.matched and not guard.failure else 'REJECTED',
                   'start_binding': 'Original visible observations and regenerated actions, not a saved simulator state',
                   'learning_label': None}
        emit({'event': 'boundary_receipt', **receipt})
        with (native.ROOT / f'BOUNDARY_{case["id"]}.json').open('x') as file:
            json.dump(receipt, file, indent=2)


native.run_case = run_case
if __name__ == '__main__':
    asyncio.run(native.main())
