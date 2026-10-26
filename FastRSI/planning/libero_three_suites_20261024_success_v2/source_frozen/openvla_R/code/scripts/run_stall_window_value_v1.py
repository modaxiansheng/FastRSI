"""Original Agent/Skills, unchanged policy actions, opt-in feedback-only probe."""
import asyncio
import json
import re
from pathlib import Path
import run_native_baseline as native
from cost_trace import emit
from fastrsi.feedback_window import FeedbackWindow
from fastrsi.stall_window_value_v1 import StallWindowValue
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter

original_case = native.run_case


async def run_case(case, accepted):
    mode = case.get('stall_window_probe')
    if mode is None:
        return await original_case(case, accepted)
    if mode not in ('observe_only', 'switch_once') or case.get('feedback_window_mode') != 'fixed10':
        raise ValueError('Unregistered feedback probe')
    rule = StallWindowValue(case, mode == 'switch_once')
    old_input = OpenPIPi05Adapter.to_policy_input
    old_horizon = FeedbackWindow._choose_horizon
    events = []

    def bind(adapter, observation, context):
        payload = old_input(adapter, observation, context)
        if context.get('session_id') == case['id']:
            match = re.fullmatch(r'libero_obs_(\d+)', str(observation.get('observation_id', '')))
            if match is None:
                raise ValueError('Missing actual root observation boundary')
            event = rule.observe(payload['observation/state'], int(match.group(1)),
                                 context['session_id'], payload['prompt'])
            event.update(event='stall_window_observation', session_id=case['id'])
            events.append(event)
            emit(event)
        return payload

    def horizon(controller, signs, within, boundary, step):
        if controller.case['id'] != case['id']:
            return old_horizon(controller, signs, within, boundary, step)
        return rule.horizon(step)

    OpenPIPi05Adapter.to_policy_input = bind
    FeedbackWindow._choose_horizon = horizon
    completed = False
    try:
        result = await original_case(case, accepted)
        completed = True
        return result
    finally:
        OpenPIPi05Adapter.to_policy_input = old_input
        FeedbackWindow._choose_horizon = old_horizon
        with (native.ROOT/'STALL_WINDOW_RECEIPT.json').open('x') as f:
            json.dump(dict(completed=completed, mode=mode, trigger_step=rule.trigger_step,
                           observation_count=len(events), learned=False,
                           replaced_motion=False, original_actions_and_sampling=True), f, indent=2)


native.run_case = run_case
if __name__ == '__main__':
    asyncio.run(native.main())
